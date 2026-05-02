#!/usr/bin/env python3
"""
Alta Investments — GBPUSD Live Drift Report

Compares live fill performance vs backtest expectations.
Run daily during the 30-day drift test:

    python3 scripts/live_drift_report.py

Backtest baseline (v002, 2015-2024):
    Win rate:   58.9%
    Avg TPY:    13.2
    Sharpe:     1.09
    Max DD:     -10.4%
    PF:         1.76
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT       = Path(__file__).parents[1]
FILLS_PATH = ROOT / 'data' / 'ledger' / 'live_fills_GBPUSD.jsonl'
VETO_PATH  = ROOT / 'data' / 'ledger' / 'live_veto_ledger.jsonl'

# ── Backtest baseline (v002) ──────────────────────────────────────────
BASELINE = {
    'win_rate':  0.589,
    'sharpe':    1.09,
    'pf':        1.76,
    'max_dd':   -0.104,
    'spread_assumption_pips': 0.5,  # backtest assumed near-zero spread
}

DRIFT_THRESHOLDS = {
    'on_track':   0.20,   # live within 20% of backtest → on track
    'drifting':   0.40,   # 20-40% below → investigate slippage
    'halt':       0.40,   # >40% below → structural problem, do not scale
}

# ── Hardcoded per-pip P&L for GBPUSD ─────────────────────────────────
PIP_VALUE_PER_LOT = 10.0     # $10 per pip per standard lot
PIP_SIZE          = 0.0001


def load_fills() -> list[dict]:
    if not FILLS_PATH.exists():
        return []
    with open(FILLS_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_vetoes() -> list[dict]:
    if not VETO_PATH.exists():
        return []
    with open(VETO_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_r_multiple(fill: dict) -> float | None:
    """
    Approximate R-multiple from fill record.
    We don't have exit prices here — this is a placeholder for when
    we match fills to exits via order IDs.
    Returns None until exit is logged.
    """
    # TODO: match fill to corresponding close order once those are logged
    return None


def report(fills: list[dict], vetoes: list[dict]) -> None:
    n = len(fills)
    day_number = (datetime.now(timezone.utc) - datetime(2026, 1, 1, tzinfo=timezone.utc)).days

    print(f"\n{'═'*55}")
    print(f"  ALTA INVESTMENTS — GBPUSD LIVE DRIFT REPORT")
    print(f"  Day {day_number}  ·  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'─'*55}")

    if n == 0:
        print("  No fills recorded yet.")
        print(f"  Vetoes logged: {len(vetoes)}")
        print(f"{'═'*55}\n")
        return

    # ── Fill stats ────────────────────────────────────────────────────
    slippages = [f['slippage_pips'] for f in fills if 'slippage_pips' in f]
    spreads   = [f['spread_pips']   for f in fills if 'spread_pips'   in f]
    modes     = [f.get('mode', '?') for f in fills]

    avg_slip   = float(np.mean(slippages)) if slippages else 0.0
    avg_spread = float(np.mean(spreads))   if spreads   else 0.0
    max_slip   = float(np.max(slippages))  if slippages else 0.0

    # Conviction distribution
    convictions = [f.get('conviction', 0) for f in fills]
    avg_conv = float(np.mean(convictions)) if convictions else 0.0

    # Trajectory predictions
    p50s = [f.get('predicted_r_p50', 0) for f in fills]
    avg_p50 = float(np.mean(p50s)) if p50s else 0.0

    # Veto breakdown
    veto_reasons: dict[str, int] = {}
    for v in vetoes:
        r = v.get('reason', 'UNKNOWN')
        veto_reasons[r] = veto_reasons.get(r, 0) + 1

    # ── Slippage cost in R ────────────────────────────────────────────
    # Average stop distance approximation: 1.5× ATR on GBPUSD ≈ 40 pips
    avg_stop_pips  = 40.0
    slippage_r_cost = avg_slip / avg_stop_pips if avg_stop_pips > 0 else 0.0
    spread_r_cost   = avg_spread / avg_stop_pips if avg_stop_pips > 0 else 0.0
    total_friction  = slippage_r_cost + spread_r_cost

    # ── Status ────────────────────────────────────────────────────────
    # Without exit prices we can't compute actual Sharpe yet.
    # Use slippage + spread friction as a leading indicator:
    # If friction > 0.15R per trade, backtest Sharpe will be materially impacted.
    friction_impact_pct = total_friction / (BASELINE['sharpe'] / 10.0 + 1e-9)

    if friction_impact_pct < DRIFT_THRESHOLDS['on_track']:
        status = 'ON_TRACK'
    elif friction_impact_pct < DRIFT_THRESHOLDS['drifting']:
        status = 'DRIFTING — investigate spread/slippage'
    else:
        status = 'HALT_REQUIRED — friction too high, do not scale'

    # ── Print ─────────────────────────────────────────────────────────
    print(f"  Trades taken:         {n}")
    print(f"  Mode breakdown:       {dict((m, modes.count(m)) for m in set(modes))}")
    print(f"  Avg conviction:       {avg_conv:.2f}  (min gate: 0.60)")
    print(f"  Avg predicted p50 R:  {avg_p50:+.2f}")
    print(f"{'─'*55}")
    print(f"  Avg slippage:         {avg_slip:+.2f} pips  (max: {max_slip:+.1f})")
    print(f"  Avg spread at fill:   {avg_spread:.2f} pips")
    print(f"  Backtest spread assm: {BASELINE['spread_assumption_pips']:.1f} pips")
    print(f"  Total friction/trade: {total_friction:.3f}R  ({slippage_r_cost:.3f}R slip + {spread_r_cost:.3f}R spread)")
    print(f"{'─'*55}")
    print(f"  Backtest win rate:    {BASELINE['win_rate']:.1%}  (live: pending exit data)")
    print(f"  Backtest Sharpe:      {BASELINE['sharpe']:.2f}   (live: pending exit data)")
    print(f"  Backtest PF:          {BASELINE['pf']:.2f}   (live: pending exit data)")
    print(f"{'─'*55}")
    print(f"  Vetoes this session:  {len(vetoes)}")
    for reason, count in sorted(veto_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<28s} {count:>3d}")
    print(f"{'─'*55}")
    print(f"  STATUS: {status}")
    print(f"{'═'*55}\n")

    if 'HALT' in status:
        print("  ⚠  HALT REQUIRED — do not size up until friction is resolved.")
        print("  Likely causes: thin liquidity session, wide broker spread, poor timing.\n")


def main():
    fills  = load_fills()
    vetoes = load_vetoes()
    report(fills, vetoes)


if __name__ == '__main__':
    main()
