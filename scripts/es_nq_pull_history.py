#!/usr/bin/env python3
"""One-time + incremental NQ history pull for the ES/NQ engine.

Usage:
  python3 scripts/es_nq_pull_history.py --test-week          # 1-week probe (check billing!)
  python3 scripts/es_nq_pull_history.py --start 2018-01-01 --end 2026-01-01
  python3 scripts/es_nq_pull_history.py --update             # append from last cached bar

Writes data/es_nq/{nq_globex_1min,nq_historical_5min,nq_daily,aux_daily}.parquet
then prints the data verification report (row counts/yr, RTH bars/session, rolls,
px_0925 coverage). Fail loud everywhere.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sovereign.es_nq import data_store  # noqa: E402


def verify(df1m, df5, daily) -> None:
    print("\n=== DATA VERIFICATION ===")
    years = df1m.index.tz_convert("America/New_York").year
    import pandas as pd
    print("1-min rows/year:")
    print(pd.Series(1, index=years).groupby(level=0).sum().to_string())
    print(f"5-min RTH rows: {len(df5):,}")
    print(f"sessions: {len(daily):,}")
    print(f"RTH bars/session: mean {daily['rth_bars'].mean():.1f} "
          f"(78 = full session at 1-min×78... 5-min view below)")
    per_session_5m = df5.groupby(
        df5.index.tz_convert("America/New_York").strftime("%Y-%m-%d")).size()
    print(f"5-min bars/session: median {per_session_5m.median():.0f} (expect 78), "
          f"<70 bars on {int((per_session_5m < 70).sum())} sessions (half-days etc.)")
    rolls = daily["roll_day"].sum()
    print(f"roll days: {int(rolls)} (~{rolls / max(1, len(daily)) * 252:.1f}/yr)")
    cov = daily["px_0925"].notna().mean()
    print(f"px_0925 coverage: {cov:.2%}")
    if cov < 0.99:
        raise SystemExit(f"FATAL: px_0925 coverage {cov:.2%} < 99% — investigate before validating")
    dup = df1m.index.duplicated().sum()
    if dup:
        raise SystemExit(f"FATAL: {dup} duplicate 1-min timestamps")
    print("=== VERIFICATION PASSED ===\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--test-week", action="store_true",
                    help="pull one recent week only (billing probe)")
    ap.add_argument("--update", action="store_true",
                    help="append from the last cached timestamp to now")
    args = ap.parse_args()

    if args.test_week:
        end = datetime.now(timezone.utc) - timedelta(days=2)
        start = end - timedelta(days=7)
        df = data_store.pull_globex_history(start.strftime("%Y-%m-%d"),
                                            end.strftime("%Y-%m-%d"), chunk_days=8)
        print(f"\nTEST WEEK: {len(df):,} rows, "
              f"{df.index.min()} → {df.index.max()}, symbols: {sorted(df['symbol'].unique())}")
        print("Check the billed amount at databento.com/portal before the full pull.")
        return

    if args.update:
        existing = data_store.load_1min()
        start = (existing.index.max() + timedelta(minutes=1)).strftime("%Y-%m-%d")
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if start >= end:
            print("Cache already current.")
            df1m = existing
        else:
            fresh = data_store.pull_globex_history(start, end, chunk_days=30)
            import pandas as pd
            df1m = pd.concat([existing, fresh])
            df1m = df1m[~df1m.index.duplicated(keep="first")].sort_index()
    else:
        df1m = data_store.pull_globex_history(args.start, args.end)

    df5 = data_store.resample_5min(data_store.filter_rth(df1m))
    daily = data_store.build_daily_table(df1m)
    aux = data_store.pull_aux_daily(start="2017-06-01")

    data_store.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df1m.to_parquet(data_store.RAW_1MIN)
    df5.to_parquet(data_store.RTH_5MIN)
    daily.to_parquet(data_store.DAILY)
    aux.to_parquet(data_store.AUX_DAILY)
    print(f"Wrote {data_store.RAW_1MIN} ({len(df1m):,} rows)")
    print(f"Wrote {data_store.RTH_5MIN} ({len(df5):,} rows)")
    print(f"Wrote {data_store.DAILY} ({len(daily):,} sessions)")
    print(f"Wrote {data_store.AUX_DAILY} ({len(aux):,} days)")
    verify(df1m, df5, daily)


if __name__ == "__main__":
    main()
