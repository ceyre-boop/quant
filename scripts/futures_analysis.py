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

TRADE_LOG  = ROOT / "data" / "futures" / "trade_log.jsonl"
ORACLE_LOG = ROOT / "data" / "futures" / "oracle_mornings.jsonl"
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


def _oracle_calibration(instrument: str | None = None) -> None:
    """Score oracle morning calls and display calibration curve."""
    if not ORACLE_LOG.exists():
        print("No oracle calls logged yet. Run: python3.13 scripts/futures_oracle_morning.py")
        return

    calls = []
    with open(ORACLE_LOG) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if instrument is None or rec.get("instrument") == instrument:
                    calls.append(rec)
            except Exception:
                pass

    if not calls:
        print("No oracle calls found.")
        return

    scored = [c for c in calls if c.get("outcome_scored") is not None]
    n_total = len(calls)
    n_scored = len(scored)

    print(f"\n{'═'*60}")
    print(f"  ORACLE CALIBRATION — {instrument or 'ALL'}")
    print(f"{'═'*60}")
    print(f"  Total calls: {n_total}  |  Scored: {n_scored}  |  Pending: {n_total - n_scored}")

    if n_scored == 0:
        print("\n  No scored calls yet.")
        print("  After each session, run:")
        print("    python3.13 scripts/futures_analysis.py --score-today --hit 1  (T1 reached)")
        print("    python3.13 scripts/futures_analysis.py --score-today --hit 0  (falsifier hit)")
        print(f"{'═'*60}\n")
        return

    # Brier score
    brier = sum((c["stated_probability"] - c["outcome_scored"]) ** 2 for c in scored) / n_scored
    hits  = sum(c["outcome_scored"] for c in scored)
    overall_wr = hits / n_scored * 100

    print(f"\n  Overall hit rate: {hits}/{n_scored} ({overall_wr:.1f}%)")
    print(f"  Brier score:      {brier:.4f}  {'✓ well-calibrated' if brier < 0.15 else ('~ noisy' if brier < 0.25 else '✗ overconfident')}")
    print(f"  (Brier: 0=perfect, 0.25=random, lower=better)")

    # Bucket breakdown
    buckets = {"40-50%": [], "50-60%": [], "60-70%": [], "70-80%": []}
    for c in scored:
        p = c["stated_probability"]
        if   p < 0.50: buckets["40-50%"].append(c["outcome_scored"])
        elif p < 0.60: buckets["50-60%"].append(c["outcome_scored"])
        elif p < 0.70: buckets["60-70%"].append(c["outcome_scored"])
        else:          buckets["70-80%"].append(c["outcome_scored"])

    print(f"\n  Calibration by stated probability bucket:")
    print(f"  {'Stated':10s}  {'n':>4}  {'Actual':>8}  {'Delta':>8}  Bar")
    for label, outcomes in buckets.items():
        if not outcomes:
            continue
        mid = float(label.split("-")[0].rstrip("%")) / 100 + 0.05
        actual = sum(outcomes) / len(outcomes)
        delta  = actual - mid
        bar_len = int(actual * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        sign = "+" if delta >= 0 else ""
        print(f"  {label:10s}  {len(outcomes):>4}  {actual:>7.1%}  {sign}{delta:>7.1%}  {bar}")

    # Recent calls summary
    recent = calls[-5:]
    print(f"\n  Recent calls (last {len(recent)}):")
    for c in recent:
        scored_str = ""
        if c.get("outcome_scored") is not None:
            scored_str = " → HIT" if c["outcome_scored"] == 1 else " → MISS"
        print(f"    {c.get('date','')}  {c.get('instrument','')}  "
              f"{c.get('bias','?'):7s}  {c.get('stated_probability', 0):.0%}{scored_str}")

    print(f"{'═'*60}\n")


def _score_today(instrument: str | None, hit: int) -> None:
    """Mark today's oracle call as hit (1) or miss (0)."""
    if not ORACLE_LOG.exists():
        print("No oracle log found.")
        return
    today = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%d")
    lines = []
    found = False
    with open(ORACLE_LOG) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if (rec.get("date") == today and
                        (instrument is None or rec.get("instrument") == instrument) and
                        rec.get("outcome_scored") is None):
                    p = rec.get("stated_probability", 0.5)
                    rec["outcome_scored"]    = hit
                    rec["outcome_hit_t1"]    = bool(hit)
                    rec["brier_contribution"] = round((p - hit) ** 2, 6)
                    found = True
            except Exception:
                pass
            lines.append(json.dumps(rec) if isinstance(rec, dict) else line.rstrip())
    if found:
        with open(ORACLE_LOG, "w") as f:
            f.write("\n".join(lines) + "\n")
        result = "HIT ✓" if hit else "MISS ✗"
        print(f"  Scored today's oracle call: {result}")
    else:
        print(f"  No unscored oracle call found for today ({today}).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Futures sandbox analysis + oracle calibration")
    ap.add_argument("--preview",    action="store_true",
                    help="Run trade analysis before 150-trade threshold")
    ap.add_argument("--instrument", default=None, choices=["MES", "MNQ"])
    ap.add_argument("--oracle",     action="store_true",
                    help="Show oracle calibration scores")
    ap.add_argument("--score-today", action="store_true",
                    help="Score today's oracle call (use with --hit)")
    ap.add_argument("--hit", type=int, choices=[0, 1], default=None,
                    help="1 = T1 was reached, 0 = falsifier was hit")
    args = ap.parse_args()

    if args.score_today:
        if args.hit is None:
            print("--score-today requires --hit 0 or --hit 1")
            sys.exit(1)
        _score_today(args.instrument, args.hit)
        return

    if args.oracle:
        _oracle_calibration(args.instrument)
        return

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
