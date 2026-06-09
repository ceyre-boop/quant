#!/usr/bin/env python3
"""
Generate the FROZEN held-out OOS (2023-2024) trade pool the prop-challenge Monte-Carlo bootstraps.

Why this exists: sovereign/risk/monte_carlo_prop.py answers "P(pass the funded challenge)" by
bootstrap-resampling REAL trade returns. That number is only honest if the pool is HELD-OUT
(out-of-sample). The canonical logs/forex_backtest_trades.json is gitignored and gets overwritten
with the full 2015-2024 (in-sample) backtest — so we snapshot the OOS window ONCE into a tracked,
immutable file and point the sim at that, with the window enforced at load.

What it does:
  1. Back up the current canonical logs/forex_backtest_trades.json.
  2. Run the v015 backtester scoped to OOS 2023-2024 → it writes the canonical log.
  3. Filter to the 4 live portfolio pairs (AUDNZD excluded per HYP-045) and write the frozen
     data/risk/oos_trades_2023_2024.json.
  4. Restore the canonical log to exactly its prior state (other tools expect the full set).

Usage:  python3 scripts/generate_oos_pool.py
Needs network (yfinance) to fetch 2023-2024 prices.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.forex.forex_backtester import ForexBacktester

CANONICAL = ROOT / "logs" / "forex_backtest_trades.json"
FROZEN = ROOT / "data" / "risk" / "oos_trades_2023_2024.json"
OOS_START, OOS_END = "2023-01-01", "2024-12-31"
PORTFOLIO = {"EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"}  # AUDNZD excluded (HYP-045)


def main() -> int:
    backup = CANONICAL.with_suffix(".bak.json")
    had_canonical = CANONICAL.exists()
    if had_canonical:
        shutil.copy(CANONICAL, backup)
        print(f"backed up canonical log → {backup.name}")

    try:
        print(f"running v015 backtest scoped to OOS {OOS_START}..{OOS_END} …")
        ForexBacktester(start=OOS_START, end=OOS_END).backtest_all()  # writes CANONICAL

        pool = json.loads(CANONICAL.read_text())
        frozen = {p: t for p, t in pool.items() if p in PORTFOLIO}
        # sanity: every kept trade must be in-window
        bad = [(p, str(t.get("entry_date", ""))[:10]) for p, ts in frozen.items()
               for t in ts if not (OOS_START <= str(t.get("entry_date", ""))[:10] <= OOS_END)]
        if bad:
            print(f"WARNING: {len(bad)} kept trade(s) outside the OOS window, e.g. {bad[:3]}")

        FROZEN.parent.mkdir(parents=True, exist_ok=True)
        FROZEN.write_text(json.dumps(frozen, indent=2, default=str))
        counts = {p: len(ts) for p, ts in frozen.items()}
        total = sum(counts.values())
        print(f"\nfrozen OOS pool → {FROZEN.relative_to(ROOT)}")
        print(f"  per-pair: {counts}")
        print(f"  total OOS trades: {total}")
        if total < 40:
            print("  ⚠️  thin pool (<40) — pass-probabilities will carry wide uncertainty.")
    finally:
        if had_canonical:
            shutil.move(str(backup), str(CANONICAL))
            print(f"restored canonical log from backup")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
