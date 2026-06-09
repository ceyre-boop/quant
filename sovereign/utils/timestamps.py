"""
sovereign/utils/timestamps.py

Single source of truth for timestamp formatting across all decision logging.
Both systems (ICT and Forex) write to the same JSONL files — they must agree
on format or outcome back-fills silently miss their targets.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

_log = logging.getLogger(__name__)

# The ONE canonical format written to all decision log entries.
_FMT = "%Y-%m-%dT%H:%M:%S+00:00"

# Parse candidates in preference order — most specific first.
_PARSE_FMTS = [
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S+00:00",
    "%Y-%m-%dT%H:%M:%S.%f",   # ISO-T, naive (no tz) — assumed UTC
    "%Y-%m-%dT%H:%M:%S",      # ISO-T, naive (no tz) — assumed UTC; was the false-"0 new entries" bug
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
]


def canonical_timestamp() -> str:
    """Return current UTC time in the single canonical format."""
    return datetime.now(timezone.utc).strftime(_FMT)


def normalize_timestamp(ts: str) -> str:
    """
    Convert any timestamp string to canonical form.
    Handles pandas Timestamps, bare dates, space-separated datetimes,
    ISO strings with or without timezone info.
    Returns ts unchanged (with a warning) if no format matches.
    """
    if not ts:
        return ts
    ts = str(ts).strip()
    # Drop fractional seconds before trying fixed-format parsers
    ts_no_frac = ts.split(".")[0] if "." in ts and "+" not in ts.split(".")[-1] else ts

    for fmt in _PARSE_FMTS:
        for candidate in (ts, ts_no_frac):
            try:
                dt = datetime.strptime(candidate, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).strftime(_FMT)
            except ValueError:
                continue

    _log.warning("normalize_timestamp: could not parse %r", ts)
    return ts


def timestamps_match(ts1: str, ts2: str) -> bool:
    """
    Return True if both timestamps refer to the same date and hour.

    Special case: when either input is a bare date ("2026-05-26"), match
    on date only — the forensic engine passes truncated dates and any time
    on that date should be considered a match. Hour-precision matching only
    applies when both sides carry an explicit time component.
    """
    s1, s2 = ts1.strip(), ts2.strip()
    # Bare-date detection: ≤10 chars OR no time separator
    is_date_only_1 = len(s1) <= 10 or ("T" not in s1 and " " not in s1)
    is_date_only_2 = len(s2) <= 10 or ("T" not in s2 and " " not in s2)
    n1 = normalize_timestamp(s1)
    n2 = normalize_timestamp(s2)
    if is_date_only_1 or is_date_only_2:
        return bool(n1 and n2 and n1[:10] == n2[:10])  # date-only match
    return bool(n1 and n2 and n1[:13] == n2[:13])  # date+hour match
