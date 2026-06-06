#!/usr/bin/env python3
"""
150-trade analysis for the MES/MNQ futures sandbox.

Tests two hypotheses:
  A. Does the daily bias predict direction better than chance? (binomial test)
  B. Does adaptive sizing beat flat sizing on expectancy? (bootstrap)

Runs at any trade count — use --preview for sub-150 output with appropriate caveats.

Usage:
    python3.13 scripts/futures_analysis.py             # requires >= 150 trades
    python3.13 scripts/futures_analysis.py --preview   # runs on whatever exists
    python3.13 scripts/futures_analysis.py --instrument MES
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRADE_LOG = ROOT / "data" / "futures" / "trade_log.jsonl"
VALIDATION_THRESHOLD = 150


def _load(instrument: str | None = None) -> list[dict]:
    if not TRADE_LOG.exists():
        return []
    trades = []
    with open(TRADE_LOG) as f:
        for line in f:
            try:
                t = json.loads(line)
                if instrument is None or t.get("instrument") == instrument.upper():
                    trades.append(t)
            except Exception:
                pass
    return trades


def _binomial_p(successes: int, n: int, p0: float = 0.5) -> float:
    """Two-tailed binomial p-value via normal approximation (n >= 20)."""
    import math
    if n == 0:
        return 1.0
    mu    = n * p0
    sigma = (n * p0 * (1 - p0)) ** 0.5
    z     = (successes - mu) / sigma
    # Two-tailed p via error function
    p_one_tail = 0.5 * (1 + math.erf(-abs(z) / 2 ** 0.5))
    return round(2 * p_one_tail, 4)


def _bootstrap_mean_diff(adaptive_r: list[float], flat_r: list[float],
                          n_boot: int = 5000) -> tuple[float, float]:
    """
    Bootstrap p-value for H0: mean(adaptive_r) == mean(flat_r).
    Returns (observed_diff, p_value).
    """
    import random
    n = len(adaptive_r)
    if n == 0:
        return 0.0, 1.0
    obs_diff = sum(adaptive_r) / n - sum(flat_r) / n
    combined = adaptive_r + flat_r
    count_extreme = 0
    for _ in range(n_boot):
        random.shuffle(combined)
        a_boot = combined[:n]
        f_boot = combined[n:]
        diff = sum(a_boot) / n - sum(f_boot) / n
        if abs(diff) >= abs(obs_diff):
            count_extreme += 1
    return round(obs_diff, 4), round(count_extreme / n_boot, 4)


def _analyze(trades: list[dict], preview: bool) -> None:
    n = len(trades)
    if n == 0:
        print("No trades logged yet.")
        return

    print(f"\n{'═'*60}")
    print(f"  FUTURES SANDBOX — VALIDATION ANALYSIS")
    if preview and n < VALIDATION_THRESHOLD:
        print(f"  ⚠  PREVIEW — {n}/{VALIDATION_THRESHOLD} trades  "
              f"({VALIDATION_THRESHOLD - n} remaining)")
        print(f"  Statistical conclusions are provisional at n={n}.")
    else:
        print(f"  Full dataset: n={n} trades")
    print(f"{'═'*60}")

    # ── Test A: Bias accuracy ─────────────────────────────────────────────────
    aligned_trades = [t for t in trades
                      if t.get("bias_aligned") is True
                      and t.get("bias_direction") in ("LONG", "SHORT")]
    bias_wins = sum(1 for t in aligned_trades if (t.get("r_realized") or 0) > 0)
    bias_n = len(aligned_trades)
    bias_wr = bias_wins / bias_n * 100 if bias_n else 0

    print(f"\n  ── TEST A: BIAS DIRECTIONAL ACCURACY ──────────────────────")
    print(f"  Aligned trades:   {bias_n} / {n}")
    print(f"  Win rate:         {bias_wins}/{bias_n}  ({bias_wr:.1f}%)")
    print(f"  H0 (coin flip):   50%")

    if bias_n >= 20:
        p_val = _binomial_p(bias_wins, bias_n)
        sig = "SIGNIFICANT ✓" if p_val < 0.05 else "not significant"
        print(f"  Binomial p-value: {p_val:.4f}  →  {sig}")
        if p_val < 0.05:
            if bias_wr > 50:
                print(f"  CONCLUSION: Bias predicts direction better than chance (p<0.05).")
            else:
                print(f"  CONCLUSION: Bias ANTI-predicts direction — flip it.")
        else:
            print(f"  CONCLUSION: Cannot reject H0. Bias has no proven directional edge.")
    else:
        print(f"  (Need n≥20 for binomial test; current n={bias_n})")

    # ── Below-proven-bar sub-group ────────────────────────────────────────────
    bpb = [t for t in aligned_trades if t.get("below_proven_bar")]
    proven = [t for t in aligned_trades if not t.get("below_proven_bar")]
    if bpb and proven:
        bpb_wr    = sum(1 for t in bpb if (t.get("r_realized") or 0) > 0) / len(bpb) * 100
        proven_wr = sum(1 for t in proven if (t.get("r_realized") or 0) > 0) / len(proven) * 100
        print(f"\n  Sub-group breakdown (aligned trades):")
        print(f"    BELOW_PROVEN_BAR (conviction=1):  {len(bpb):3d} trades  {bpb_wr:.1f}% WR")
        print(f"    Proven bar (conviction≥2):         {len(proven):3d} trades  {proven_wr:.1f}% WR")

    # ── Test B: Adaptive vs flat sizing ──────────────────────────────────────
    adaptive_r = [(t.get("r_realized") or 0) * (t.get("size_contracts") or 1)
                  for t in trades]
    flat_r     = [t.get("r_realized") or 0 for t in trades]

    adaptive_expectancy = sum(adaptive_r) / n
    flat_expectancy     = sum(flat_r) / n

    print(f"\n  ── TEST B: ADAPTIVE vs FLAT SIZING ────────────────────────")
    print(f"  Adaptive expectancy: {adaptive_expectancy:+.4f}R/trade  "
          f"(total: {sum(adaptive_r):+.2f}R)")
    print(f"  Flat expectancy:     {flat_expectancy:+.4f}R/trade  "
          f"(total: {sum(flat_r):+.2f}R)")

    if n >= 20:
        diff, p_val = _bootstrap_mean_diff(adaptive_r, flat_r)
        sig = "SIGNIFICANT ✓" if p_val < 0.05 else "not significant"
        print(f"  Bootstrap diff:      {diff:+.4f}R/trade  p={p_val:.4f}  →  {sig}")
        if p_val < 0.05:
            verdict = "Adaptive sizing OUTPERFORMS" if diff > 0 else "Flat sizing OUTPERFORMS"
            print(f"  CONCLUSION: {verdict} (p<0.05). n_boot=5000.")
        else:
            print(f"  CONCLUSION: Cannot reject H0. Sizing method has no proven effect yet.")
    else:
        print(f"  (Need n≥20 for bootstrap; current n={n})")

    # ── Sizing rationale breakdown ────────────────────────────────────────────
    by_rationale: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        r_key = t.get("sizing_rationale", "unknown")
        r_val = (t.get("r_realized") or 0) * (t.get("size_contracts") or 1)
        by_rationale[r_key].append(r_val)

    print(f"\n  ── SIZING RATIONALE BREAKDOWN ──────────────────────────────")
    for label in ["probe", "press", "reduce", "stand_down", "unknown"]:
        vals = by_rationale.get(label, [])
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        print(f"    {label:12s}: n={len(vals):3d}  avg={avg:+.3f}R/trade")

    # ── Session arc compliance ────────────────────────────────────────────────
    sessions: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        day = t.get("ts", "")[:10]
        sessions[day].append(t)

    max_trades_per_session = max((len(v) for v in sessions.values()), default=0)
    sessions_gt3 = sum(1 for v in sessions.values() if len(v) > 3)
    print(f"\n  ── SESSION DISCIPLINE ──────────────────────────────────────")
    print(f"    Sessions: {len(sessions)}  |  Max trades in a session: {max_trades_per_session}")
    if sessions_gt3:
        print(f"    ⚠  {sessions_gt3} session(s) had >3 trades (arc recommends max 3)")

    # ── Progress bar ──────────────────────────────────────────────────────────
    pct = min(100, n / VALIDATION_THRESHOLD * 100)
    bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
    print(f"\n  Progress: [{bar}] {n}/{VALIDATION_THRESHOLD} ({pct:.0f}%)")

    if n >= VALIDATION_THRESHOLD:
        print(f"\n  VALIDATION THRESHOLD REACHED.")
        print(f"  Review Test A and Test B conclusions above.")
        print(f"  If bias WR > 50% with p<0.05 AND adaptive R > flat R with p<0.05,")
        print(f"  the framework has earned the right to continue with full size.")

    print(f"{'═'*60}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Futures sandbox 150-trade analysis")
    ap.add_argument("--preview",    action="store_true",
                    help="Run analysis even before 150-trade threshold")
    ap.add_argument("--instrument", default=None, choices=["MES", "MNQ"],
                    help="Filter to one instrument (default: all)")
    args = ap.parse_args()

    trades = _load(args.instrument)
    n = len(trades)

    if n == 0:
        print("No trades logged yet. Log trades with: python3.13 scripts/futures_log.py")
        sys.exit(0)

    if n < VALIDATION_THRESHOLD and not args.preview:
        print(f"Only {n}/{VALIDATION_THRESHOLD} trades logged.")
        print(f"Run with --preview for interim analysis, or keep trading.")
        sys.exit(0)

    _analyze(trades, preview=args.preview)


if __name__ == "__main__":
    main()
