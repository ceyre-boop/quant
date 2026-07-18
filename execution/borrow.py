"""Short-locate gating from the IB shortable snapshot.

SOURCE
------
`scripts/ib_shortable_snapshot.py` (launchd `com.alta.ib_shortable.plist`, daily
07:00 ET) fetches IB's public shortable list and writes
`data/research/gapper/ib_locate_{date}.json` with per-symbol tiers:
    EASY (>= 10,000 shares) | HARD | UNAVAILABLE | NOT_LISTED

DO NOT WIRE THE OTHER PATH. `research/yield_frontier/daily_snapshots.py:30-33`
records: "IBKR FTP unreachable from this network 2026-07-13 (ftp3 and ftp2+TLS
both timed out)"; its `borrow_snapshots/` directory is empty. That path is dead.

POLICY — deliberately conservative
----------------------------------
No snapshot for the session          -> SKIP_NO_BORROW (no_locate_snapshot)
Tier HARD / UNAVAILABLE / NOT_LISTED -> SKIP_NO_BORROW (tier_<TIER>)
Tier EASY                            -> fillable

Two rules that exist to protect the measurement:

1. NEVER fall back to a stale snapshot. Borrow availability on a parabolic
   microcap changes intraday; yesterday's EASY is not evidence about today. A
   stale-file fallback would manufacture fills that could not have happened.

2. HARD tiers skip during the measurement window. A HARD locate is a borrow you
   might not actually obtain, and counting it as filled would flatter the short
   leg — biasing the exact number this harness exists to measure honestly.

Only ONE snapshot exists so far (2026-07-16), so the short leg will emit mostly
SKIP_NO_BORROW until the 07:00 job accumulates history. That is expected and
means the two legs are NOT equally sampled; the daily summary must not be read
as if they were.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCATE_DIR = ROOT / "data" / "research" / "gapper"

FILLABLE_TIERS = frozenset({"EASY"})


def locate_path(day: date) -> Path:
    return LOCATE_DIR / f"ib_locate_{day}.json"


def load_locate(day: date) -> dict[str, str] | None:
    """Symbol -> tier for `day`, or None if no snapshot exists for that date.

    None means "unknown", never "no borrow constraint". Callers must skip.
    """
    fp = locate_path(day)
    if not fp.exists():
        return None
    try:
        raw = json.loads(fp.read_text())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"corrupt locate snapshot {fp}: {e}") from e

    # Tolerate either {"SYM": "EASY"} or {"symbols": {"SYM": {"tier": "EASY"}}}
    if isinstance(raw, dict) and "symbols" in raw and isinstance(raw["symbols"], dict):
        raw = raw["symbols"]

    out: dict[str, str] = {}
    for sym, val in raw.items():
        if isinstance(val, dict):
            tier = str(val.get("tier", "")).upper()
        else:
            tier = str(val).upper()
        if tier:
            out[str(sym).upper()] = tier
    return out


def borrow_ok(symbol: str, locate: dict[str, str] | None) -> tuple[bool, str]:
    """(fillable, reason). See module docstring for the policy rationale."""
    if locate is None:
        return False, "no_locate_snapshot"

    tier = locate.get(symbol.upper())
    if tier is None:
        return False, "tier_NOT_LISTED"
    if tier in FILLABLE_TIERS:
        return True, f"tier_{tier}"
    return False, f"tier_{tier}"


def available_locate_days() -> list[date]:
    """Session dates for which a locate snapshot exists."""
    if not LOCATE_DIR.exists():
        return []
    days: list[date] = []
    for fp in sorted(LOCATE_DIR.glob("ib_locate_*.json")):
        try:
            days.append(date.fromisoformat(fp.stem.replace("ib_locate_", "")))
        except ValueError:
            continue
    return days
