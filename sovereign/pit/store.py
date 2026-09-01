"""Append-only writes, so a revision never destroys the observation it revises.

This is the half of point-in-time that timestamps alone do not buy you.

The fundamentals store previously wrote every table with INSERT OR REPLACE on a
primary key that did NOT include the publication instant. So:

  - an earnings restatement overwrote the originally-reported figure
  - the normal pre-print -> post-print transition destroyed the pre-announcement
    consensus view of that quarter
  - a 13F-A amendment overwrote the original 13F IN PLACE, taking the original's
    earlier filing_date with it — after which even a correct
    `filing_date < as_of` filter UNDERSTATES what was knowable

You cannot answer "what did I know on March 3" against a table that has been
overwritten, no matter how many timestamps it carries. History has to survive
first; filtering is the easy part.

So: writes here are INSERT-only. Two rows differing only in `published_ts` are
two vintages of the same fact and both are kept. Re-observing an identical row
is a no-op rather than an error, so re-running a harvest stays idempotent.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb

from sovereign.pit.errors import NotPointInTime
from sovereign.pit.spec import FactSpec, get as get_fact

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "fundamentals.db"

#: Columns every point-in-time fact table gains.
#:
#: `observed_at` is the third timestamp and is NOT part of the temporal
#: contract — it records when WE fetched a row, which is an operational fact
#: (used to debug a harvester), never a knowability fact. Confusing it with
#: published_ts is a classic way to reintroduce leakage, so it is named
#: differently and never used in an as-of predicate.
PIT_COLUMNS = "observed_at TIMESTAMP"


def ro_connect():
    """Read-only connection, or None. Never raises.

    DuckDB is single-writer, so a read during a harvest write would otherwise
    fail. A point-in-time read that cannot reach the store must return nothing
    rather than crash a research run — but it must NEVER fall back to some other
    source, which is why this returns None instead of a degraded handle.
    """
    if not DB_PATH.exists():
        return None
    try:
        return _utc(duckdb.connect(str(DB_PATH), read_only=True))
    except duckdb.Error as e:
        log.warning("pit store unavailable (%s); as-of reads return empty", e)
        return None


def _utc(con):
    """Pin the session to UTC.

    Belt-and-braces alongside binding naive-UTC parameters in the reader. A
    DuckDB session defaults to the OS timezone, and any naive/aware timestamp
    comparison is then resolved against it — which would make an as-of read
    depend on the machine that ran it. Two independent defences, because the
    failure is silent.
    """
    try:
        con.execute("SET TimeZone='UTC'")
    except duckdb.Error:  # pragma: no cover - older builds without the setting
        pass
    return con


def rw_connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _utc(duckdb.connect(str(DB_PATH)))


def _norm(v: Any) -> Any:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return v


def ensure_pit_columns(con, spec: FactSpec) -> None:
    """Add observed_at if absent. Idempotent."""
    try:
        cols = {r[1] for r in con.execute(f"PRAGMA table_info('{spec.table}')").fetchall()}
    except duckdb.Error:
        return
    if not cols:
        return
    if "observed_at" not in cols:
        con.execute(f"ALTER TABLE {spec.table} ADD COLUMN observed_at TIMESTAMP")
        log.info("pit: added observed_at to %s", spec.table)


def append(
    fact: str,
    rows: Sequence[dict[str, Any]],
    *,
    con=None,
) -> int:
    """Append observations. Never updates, never replaces.

    Rejects any row without a publication instant: an un-timestamped row cannot
    participate in an as-of read, so storing it in a point-in-time table would
    create a silent hole. Store it in the raw layer instead, or give the
    transport a real publication time.
    """
    spec = get_fact(fact)
    if not spec.is_point_in_time:
        raise NotPointInTime(
            f"{fact!r} is not point-in-time and must not be appended here.\n"
            f"  Why: {spec.blocked_reason}"
        )
    if not rows:
        return 0

    own = con is None
    con = con or rw_connect()
    try:
        ensure_pit_columns(con, spec)

        now = datetime.now(timezone.utc)
        pub, ident = spec.published_col, spec.identity
        written = 0

        for r in rows:
            if r.get(pub) is None:
                log.warning(
                    "pit: dropping %s row for %s with no %s — an un-timestamped "
                    "row cannot be read as-of",
                    fact, r.get(spec.entity_col), pub,
                )
                continue

            # A vintage is identified by identity + publication instant. Re-seeing
            # the same vintage is a no-op; a DIFFERENT publication instant for the
            # same identity is a restatement and gets its own row.
            where = " AND ".join(f"{c} IS NOT DISTINCT FROM ?" for c in (*ident, pub))
            params = [_norm(r.get(c)) for c in (*ident, pub)]
            exists = con.execute(
                f"SELECT 1 FROM {spec.table} WHERE {where} LIMIT 1", params
            ).fetchone()
            if exists:
                continue

            payload = {k: _norm(v) for k, v in r.items()}
            payload["observed_at"] = now
            cols = list(payload)
            con.execute(
                f"INSERT INTO {spec.table} ({', '.join(cols)}) "
                f"VALUES ({', '.join('?' * len(cols))})",
                [payload[c] for c in cols],
            )
            written += 1
        return written
    finally:
        if own:
            con.close()


def vintage_count(fact: str, entity: str | None = None) -> int:
    """How many distinct vintages we hold. A restatement makes this exceed the
    number of logical facts — which is the point."""
    spec = get_fact(fact)
    con = ro_connect()
    if con is None:
        return 0
    try:
        with con:
            q = f"SELECT count(*) FROM {spec.table} WHERE {spec.published_col} IS NOT NULL"
            p: list[Any] = []
            if entity:
                q += f" AND {spec.entity_col} = ?"
                p.append(entity.upper())
            return int(con.execute(q, p).fetchone()[0])
    except duckdb.Error:
        return 0


def drop_primary_key_constraints(con, spec: FactSpec) -> bool:
    """Rebuild a table without its PRIMARY KEY so vintages can coexist.

    A PK that excludes published_ts is precisely what made revisions destructive:
    the second vintage collided with the first and INSERT OR REPLACE resolved the
    collision by deleting history. Returns True if the table was rebuilt.
    """
    try:
        info = con.execute(f"PRAGMA table_info('{spec.table}')").fetchall()
    except duckdb.Error:
        return False
    if not info:
        return False
    if not any(r[5] for r in info):  # r[5] = pk flag
        return False

    cols = [r[1] for r in info]
    types = {r[1]: r[2] for r in info}
    tmp = f"{spec.table}__pit_rebuild"
    ddl = ", ".join(f"{c} {types[c]}" for c in cols)
    con.execute(f"CREATE TABLE {tmp} ({ddl})")
    con.execute(f"INSERT INTO {tmp} SELECT {', '.join(cols)} FROM {spec.table}")
    con.execute(f"DROP TABLE {spec.table}")
    con.execute(f"ALTER TABLE {tmp} RENAME TO {spec.table}")
    log.info("pit: rebuilt %s without PRIMARY KEY so vintages can coexist", spec.table)
    return True
