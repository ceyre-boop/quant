#!/usr/bin/env python3
"""NQ/ES lead-lag regime classifier — thin wrapper (logic now lives in the briefing engine).

The classifier moved to `sovereign.briefing.lead_lag` (richer: rolling 1d/5d/20d spreads +
intraday 5m cross-correlation lead-lag). This wrapper preserves the original CLI and still
writes data/research/nqes_regime.json (the key reflect_cycle + oracle_session_open read).

Output is a regime INPUT (informs sizing/trust), NOT a trade signal. ES/NQ are CME futures —
data + classification only; no execution path (OANDA is forex-only).

Usage:  python3 scripts/nqes_regime.py [--lookback 20]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.briefing.lead_lag import classify, LEGACY_OUT, RICH_OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=20)
    args = ap.parse_args()
    r = classify(args.lookback)
    print(f"NQ/ES regime: {r['regime']}")
    if r["regime"] != "NO_DATA":
        ll = r.get("lead_lag", {})
        print(f"  NQ {r['nq_last']} / ES {r['es_last']} | spread_avg {r['nq_es_return_spread_avg']:+.4f} "
              f"| corr {r['contemporaneous_corr']} | leader {ll.get('leader')} "
              f"(lag {ll.get('best_lag')}, corr {ll.get('best_corr')})")
        print(f"  → {r['read']}")
    print(f"  Saved: {LEGACY_OUT.relative_to(ROOT)} + {RICH_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
