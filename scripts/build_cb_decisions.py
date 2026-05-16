#!/usr/bin/env python3
"""
Build data/cache/cb_decisions.json from FRED rate series.

For each central bank:
  - Fetch monthly rate series 2014-01 to 2024-12
  - Detect every month where rate changed by ≥ 1bp
  - Estimate surprise using trailing expectation model:
      * Look at prior 3 non-zero changes
      * If all same sign → expected = median of those 3
      * If mixed or none → expected = 0
      * surprise_bps = actual_change_bps - expected_change_bps
  - Write decision record only when |actual_change_bps| >= 1

Produces records like:
  {
    "date": "2022-06-15",
    "bank": "FED",
    "country": "US",
    "actual_change_bps": 75,
    "expected_change_bps": 50,
    "surprise_bps": 25
  }
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parents[1]))
load_dotenv()

CACHE_DIR = Path(__file__).parents[1] / 'data' / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = CACHE_DIR / 'cb_decisions.json'

# FRED series IDs for policy rates (same as data_fetcher.py)
FRED_RATES = {
    'US':  ('FED',  'FEDFUNDS'),
    'EU':  ('ECB',  'ECBDFR'),
    'UK':  ('BOE',  'IUDSOIA'),
    'JP':  ('BOJ',  'IRSTCI01JPM156N'),
    'CH':  ('SNB',  'IR3TIB01CHM156N'),
    'AU':  ('RBA',  'IR3TIB01AUM156N'),
    'CA':  ('BOC',  'IR3TIB01CAM156N'),
    'NZ':  ('RBNZ', 'IR3TIB01NZM156N'),
}

START = '2014-01-01'
END   = datetime.today().strftime('%Y-%m-%d')
MIN_CHANGE_BPS = 1   # detect any change ≥ 1bp


def _surprise(actual_bps: float, prior_changes: list[float]) -> float:
    """Estimate expected change from trailing history; return surprise."""
    nonzero = [c for c in prior_changes if abs(c) >= MIN_CHANGE_BPS]
    if len(nonzero) >= 2:
        signs = [1 if c > 0 else -1 for c in nonzero]
        if len(set(signs)) == 1:
            # All same direction — continuation expected
            expected = float(median(nonzero[-3:]))
        else:
            # Mixed signals — market uncertain
            expected = 0.0
    elif len(nonzero) == 1:
        # One prior change: if same direction, half expected
        expected = nonzero[0] * 0.5
    else:
        # No recent changes — any change is a surprise
        expected = 0.0
    return round(actual_bps - expected, 1)


def build_from_fred() -> list[dict]:
    from fredapi import Fred
    fred_key = os.getenv('FRED_API_KEY')
    if not fred_key:
        raise RuntimeError('FRED_API_KEY not set')

    fred = Fred(api_key=fred_key)
    decisions = []

    for country, (bank, series_id) in FRED_RATES.items():
        print(f"  Fetching {bank} ({series_id})...", end=' ')
        try:
            s = fred.get_series(series_id, observation_start=START, observation_end=END)
            s = s.dropna().sort_index()

            # Round to nearest bp to avoid floating-point noise
            s_bps = (s * 100).round(0)

            # Monthly diff — detect changes
            changes_bps = s_bps.diff().dropna()
            significant = changes_bps[changes_bps.abs() >= MIN_CHANGE_BPS]

            print(f"{len(significant)} changes detected")

            prior_window: list[float] = []
            for date, change_bps in significant.items():
                date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
                surprise = _surprise(float(change_bps), prior_window[-6:])

                decisions.append({
                    'date':                date_str,
                    'bank':                bank,
                    'country':             country,
                    'actual_change_bps':   int(change_bps),
                    'expected_change_bps': round(float(change_bps) - surprise, 1),
                    'surprise_bps':        surprise,
                })
                prior_window.append(float(change_bps))

        except Exception as e:
            print(f"FAILED: {e}")

    decisions.sort(key=lambda d: d['date'])
    return decisions


def main():
    print(f"Building CB decisions from FRED → {OUTPUT}")
    decisions = build_from_fred()

    if not decisions:
        print("No decisions generated — check FRED key and connectivity")
        sys.exit(1)

    with open(OUTPUT, 'w') as f:
        json.dump(decisions, f, indent=2)

    print(f"\nWrote {len(decisions)} decisions to {OUTPUT}")
    print("\nSample (last 10):")
    for d in decisions[-10:]:
        print(
            f"  {d['date']}  {d['bank']:5s}  "
            f"actual={d['actual_change_bps']:+d}bp  "
            f"expected={d['expected_change_bps']:+.0f}bp  "
            f"surprise={d['surprise_bps']:+.0f}bp"
        )

    # Stats
    large_surprises = [d for d in decisions if abs(d['surprise_bps']) >= 25]
    print(f"\nTotal decisions: {len(decisions)}")
    print(f"Surprises >= 25bp: {len(large_surprises)}")
    by_bank = {}
    for d in large_surprises:
        by_bank[d['bank']] = by_bank.get(d['bank'], 0) + 1
    for bank, n in sorted(by_bank.items()):
        print(f"  {bank}: {n}")


if __name__ == '__main__':
    main()
