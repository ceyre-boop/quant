#!/usr/bin/env python3
"""Bring the 1-minute bar cache up to date for the continuous research universe.

The cache (data/cache/minute_bars/{TICKER}_{DATE}.parquet, Alpaca SIP feed,
adjustment=all) is what every intraday study reads. It is written per symbol-day,
so this is idempotent: a day already on disk costs nothing.

Only the CONTINUOUS symbols are refreshed — the ~23 names with 400+ cached days
that form the liquid research universe. The other ~200 symbols are gapper
event-days, fetched on demand by the study that needs them; blanket-refreshing
them would spend a lot of API calls on days no cohort references.

Weekends and holidays return no bars; that is expected, not an error, and the
day is simply skipped rather than retried.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtester.data import get_minute_bars  # noqa: E402

CACHE = ROOT / "data" / "cache" / "minute_bars"
#: Symbols with a continuous history, i.e. the liquid universe a cohort can be
#: built on. Derived from the cache itself (>=400 cached days).
CONTINUOUS_MIN_DAYS = 400


def continuous_symbols() -> list[str]:
    counts: Counter[str] = Counter()
    for p in CACHE.glob("*.parquet"):
        counts[p.stem.rsplit("_", 1)[0]] += 1
    return sorted(s for s, n in counts.items() if n >= CONTINUOUS_MIN_DAYS)


def cached_days(sym: str) -> set[str]:
    return {p.stem.rsplit("_", 1)[1] for p in CACHE.glob(f"{sym}_*.parquet")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+")
    ap.add_argument("--days", type=int, default=60, help="lookback window")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    syms = [s.upper() for s in (args.symbols or continuous_symbols())]
    today = date.today()
    window = [today - timedelta(days=i) for i in range(args.days, -1, -1)]
    # Weekends never have bars; skipping them here avoids pointless calls.
    window = [d for d in window if d.weekday() < 5]

    print(f"refresh_minute_cache: {len(syms)} symbols x {len(window)} weekdays "
          f"(back to {window[0]}), dry_run={args.dry_run}\n")

    fetched = skipped = empty = failed = 0
    for sym in syms:
        have = cached_days(sym)
        missing = [d for d in window if d.isoformat() not in have]
        if not missing:
            print(f"  {sym:6s} up to date")
            continue
        got = 0
        for d in missing:
            if args.dry_run:
                skipped += 1
                continue
            try:
                df = get_minute_bars(sym, d.isoformat())
                if df is None or len(df) == 0:
                    empty += 1          # holiday / no session
                else:
                    fetched += 1
                    got += 1
            except Exception as e:  # noqa: BLE001 — one bad day must not stop the run
                failed += 1
                print(f"    {sym} {d}: {type(e).__name__}: {str(e)[:70]}")
        print(f"  {sym:6s} {len(missing):3d} missing -> {got} fetched")

    if args.dry_run:
        print(f"\n--dry-run: {skipped} symbol-days would be fetched")
    else:
        print(f"\n{fetched} fetched, {empty} empty (holiday/weekend), {failed} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
