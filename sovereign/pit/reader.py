"""The only door onto point-in-time data.

The design constraint is that leakage must be structurally impossible, not
discouraged. Three properties get us there:

1. You cannot obtain a reader without an as-of instant. `AsOfReader` is only
   constructible through `view(as_of(...))`, and `as_of(None)` raises.

2. You cannot express a leaking query. The reader does not accept SQL. It takes
   a registered fact name and an entity, and BUILDS the predicate itself. The
   `published_ts < as_of` cut is not something a caller can forget, reorder or
   comment out — there is no parameter for it.

3. A fact with no publication instant cannot be read at all. It raises
   NotPointInTime instead of quietly returning rows whose knowability is unknown.

What this module deliberately does NOT do is trust the caller. Every row it
returns is re-checked against the as-of instant in Python before it is handed
back (`_verify`), so a mistake in SQL generation surfaces as a LookaheadError
rather than as a plausible number. Belt and braces, because the failure is silent
and the cost is a fictional backtest.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable, Sequence

from sovereign.pit.clock import AsOf, as_of as _as_of
from sovereign.pit.errors import LookaheadError, NotPointInTime, PitSchemaMismatch
from sovereign.pit.spec import FactSpec, get as get_fact

log = logging.getLogger(__name__)


def _as_dt(v: Any) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day, tzinfo=timezone.utc)
    if isinstance(v, str):
        try:
            d = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    return None


@dataclass(frozen=True)
class Observation:
    """One vintage of one fact, with the two timestamps that define it."""

    fact: str
    entity: str
    valid_time: Any
    published_ts: datetime
    data: dict[str, Any]

    def __getitem__(self, k: str) -> Any:
        return self.data[k]

    def get(self, k: str, default: Any = None) -> Any:
        return self.data.get(k, default)


class AsOfReader:
    """A frozen view of the store at one instant.

    Obtain via `sovereign.pit.view(as_of(...))`. Constructing this directly is
    possible in Python but pointless — it still requires an AsOf, which is the
    property that matters.
    """

    def __init__(self, at: AsOf, connect):
        self._at = at
        self._connect = connect

    @property
    def at(self) -> AsOf:
        return self._at

    # ── the read path ───────────────────────────────────────────────────────

    def facts(
        self,
        fact: str,
        entity: str | None = None,
        *,
        latest_only: bool = True,
        limit: int | None = None,
    ) -> list[Observation]:
        """Rows of `fact` that were knowable at this instant.

        latest_only=True (the default) collapses each identity to its most
        recent vintage AS OF this instant — i.e. what you would have believed
        then, including a restatement only if it had already been filed. Setting
        it False returns every surviving vintage, which is how you inspect a
        revision path.
        """
        spec = get_fact(fact)
        if not spec.is_point_in_time:
            raise NotPointInTime(
                f"fact {fact!r} ({spec.table}) has no publication instant and "
                f"cannot be read as-of.\n  Why: {spec.blocked_reason}"
            )

        con = self._connect()
        if con is None:
            return []

        pub, valid = spec.published_col, spec.valid_col
        where = [f"{pub} IS NOT NULL", f"{pub} < ?"]
        # Bind a NAIVE UTC datetime, not the tz-aware one.
        #
        # The fact columns are DuckDB TIMESTAMP, which is naive. Comparing a
        # naive column against a tz-AWARE parameter makes DuckDB reconcile them
        # using the SESSION'S LOCAL TIMEZONE, so the same as-of read returns
        # different rows on different machines:
        #
        #   column TIMESTAMP '2026-07-30 20:30:00', param 2026-07-31T00:00Z
        #     TimeZone=UTC              -> True   (correct)
        #     TimeZone=America/Detroit  -> False  (wrong)
        #     TimeZone=Asia/Tokyo       -> True
        #
        # That is precisely the class of silent, machine-dependent error this
        # layer exists to prevent — relocated out of as_of() (which correctly
        # forces UTC) and into the SQL comparison it triggers. Naive-vs-naive is
        # unambiguous in every session timezone.
        params: list[Any] = [self._at.ts.astimezone(timezone.utc).replace(tzinfo=None)]
        if entity is not None:
            where.append(f"{spec.entity_col} = ?")
            params.append(entity.upper())

        sql = f"SELECT * FROM {spec.table} WHERE " + " AND ".join(where)
        sql += f" ORDER BY {pub} DESC, {valid} DESC"
        if limit and not latest_only:
            sql += f" LIMIT {int(limit)}"

        try:
            with con:
                cur = con.execute(sql, params)
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            # A table that does not exist yet is benign — nothing has been
            # harvested. Anything else (a missing column, malformed SQL) means
            # the SPEC disagrees with the SCHEMA, and returning [] there would
            # report "no data was knowable" when the truth is "the query is
            # broken". That silent-empty is the precise failure this layer
            # exists to prevent, so it is raised, loudly.
            if "does not exist" in msg and "Table" in msg:
                log.info("pit: %s not present yet (nothing harvested)", spec.table)
                return []
            raise PitSchemaMismatch(
                f"fact {fact!r} cannot be read: the spec in sovereign/pit/spec.py "
                f"does not match the schema of {spec.table}.\n"
                f"  spec says published_col={spec.published_col!r}, valid_col={spec.valid_col!r}\n"
                f"  duckdb said: {msg.splitlines()[0]}\n"
                f"  Fix the spec or the schema — do NOT let this read return empty."
            ) from e

        obs = [self._to_obs(spec, dict(zip(cols, r))) for r in rows]
        obs = [o for o in obs if o is not None]

        # Re-verify in Python. If SQL generation ever regresses, this turns a
        # silent leak into a loud failure.
        self._verify(obs)

        if latest_only:
            obs = _collapse_to_latest_vintage(obs, spec)
        if limit:
            obs = obs[: int(limit)]
        return obs

    def latest(self, fact: str, entity: str) -> Observation | None:
        rows = self.facts(fact, entity, latest_only=True, limit=1)
        return rows[0] if rows else None

    def vintages(self, fact: str, entity: str) -> list[Observation]:
        """Every surviving vintage knowable at this instant, newest first.

        This is the question "what did the story of this number look like",
        which only exists because writes are append-only.
        """
        return self.facts(fact, entity, latest_only=False)

    # ── internals ───────────────────────────────────────────────────────────

    def _to_obs(self, spec: FactSpec, row: dict[str, Any]) -> Observation | None:
        pts = _as_dt(row.get(spec.published_col))
        if pts is None:
            return None
        return Observation(
            fact=spec.name,
            entity=str(row.get(spec.entity_col) or ""),
            valid_time=row.get(spec.valid_col),
            published_ts=pts,
            data=row,
        )

    def _verify(self, obs: Iterable[Observation]) -> None:
        for o in obs:
            if not self._at.knows(o.published_ts):
                raise LookaheadError(
                    f"LOOKAHEAD (reader invariant): {o.fact} row for {o.entity} "
                    f"published {o.published_ts.isoformat()} was returned for "
                    f"{self._at}. The SQL cut and the Python check disagree — "
                    f"this is a bug in sovereign/pit/reader.py, not in the data."
                )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"AsOfReader({self._at.isoformat()})"


def _collapse_to_latest_vintage(
    obs: Sequence[Observation], spec: FactSpec
) -> list[Observation]:
    """Keep the newest vintage per identity. Rows arrive published-desc."""
    seen: set[tuple] = set()
    out: list[Observation] = []
    for o in obs:
        key = tuple(o.data.get(c) for c in spec.identity)
        if key in seen:
            continue
        seen.add(key)
        out.append(o)
    return out


def view(at, connect=None) -> AsOfReader:
    """Open a point-in-time view. `at` is required and may not be None."""
    resolved = _as_of(at)
    if connect is None:
        from sovereign.pit.store import ro_connect

        connect = ro_connect
    return AsOfReader(resolved, connect)
