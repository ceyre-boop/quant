"""The as-of instant, as a type you cannot forget to supply.

The whole point of this module is that `AsOf` has no default constructor value
and no "now" shortcut. If a caller wants data, it must state the instant it is
pretending to be at. That single constraint is what turns point-in-time from a
convention into a precondition.

Knowability rule, matching research/petrules/provenance.py exactly:

    knowable  <=>  published_ts < as_of      (STRICT)

Strict, not `<=`, and the tie goes to "not knowable". A filing timestamped at
exactly the decision instant was not usable for that decision — you cannot read
a document the moment it appears. Same-instant equality is also where clock skew
between a vendor and us lives, so the conservative direction is the only safe one.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Union

from sovereign.pit.errors import AsOfRequired, LookaheadError

AsOfLike = Union[str, date, datetime, "AsOf"]


def _to_utc(value: Union[str, date, datetime]) -> datetime:
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise AsOfRequired("as_of was an empty string")
        # A bare date means midnight UTC at the START of that day. Combined with
        # the strict `<` rule, `as_of("2026-03-03")` therefore excludes
        # everything published on the 3rd — you know only what was public when
        # the day opened. That is the conservative reading, and it is the one a
        # daily-bar backtest wants.
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError as e:
            raise AsOfRequired(f"as_of {value!r} is not an ISO date/datetime") from e
    elif isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    else:
        raise AsOfRequired(f"as_of must be a date/datetime/ISO string, got {type(value).__name__}")

    # Naive timestamps are treated as UTC rather than local. Local time would
    # make a backtest's results depend on the machine that ran it.
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


@dataclass(frozen=True, order=True)
class AsOf:
    """An instant you are pretending to be at. Immutable on purpose."""

    ts: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.ts, datetime):
            raise AsOfRequired("AsOf requires a datetime")
        if self.ts.tzinfo is None:
            raise AsOfRequired("AsOf must be timezone-aware; construct via as_of()")

    def knows(self, published_ts: datetime | None) -> bool:
        """Was something published at `published_ts` knowable at this instant?

        A None publication instant is NEVER knowable. That is not pedantry: a
        row whose publication time we failed to record could have been published
        at any moment, so treating it as usable is precisely the assumption that
        makes a backtest fiction.
        """
        if published_ts is None:
            return False
        p = published_ts
        if p.tzinfo is None:
            p = p.replace(tzinfo=timezone.utc)
        return p < self.ts

    def assert_knows(self, name: str, published_ts: datetime | None, source: str = "?") -> None:
        if not self.knows(published_ts):
            raise LookaheadError(
                f"LOOKAHEAD: {name!r} from {source} published "
                f"{published_ts.isoformat() if published_ts else 'UNKNOWN'} "
                f"is not strictly before as_of {self.ts.isoformat()}"
            )

    def isoformat(self) -> str:
        return self.ts.isoformat()

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"as_of({self.ts.isoformat()})"


def as_of(value: AsOfLike | None) -> AsOf:
    """The only way to obtain an AsOf.

    `as_of(None)` raises. There is no "now" default anywhere in this package,
    because the failure mode of a default is invisible: the query returns
    plausible data and every historical answer it gave is wrong.

    If you genuinely want the present, say so explicitly with `as_of_now()` —
    which is deliberately a different, more awkward name, and is not accepted by
    the research read paths.
    """
    if value is None:
        raise AsOfRequired(
            "as_of is required and must not be None. State the instant you are "
            "pretending to be at. For a deliberate live read, use as_of_now()."
        )
    if isinstance(value, AsOf):
        return value
    return AsOf(_to_utc(value))


def as_of_now() -> AsOf:
    """Explicitly the present instant.

    Separate from as_of() and deliberately harder to type, so that "read the
    latest" is always a visible decision in the diff rather than an omitted
    argument.
    """
    return AsOf(datetime.now(timezone.utc))

