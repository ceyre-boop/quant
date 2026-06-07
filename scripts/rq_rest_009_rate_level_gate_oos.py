"""
RQ-REST-009 — OOS Validation: Rate Level Gate |real_rate_diff| > 1.0%  (HYP-054)
==================================================================================

CONTEXT (from REST-003):
  In-sample analysis on 502 trades (4-pair, 2015-2024 via trade_forensics.json):
    |real_rate_diff| > 1.0%: Sharpe = 0.650, n = 326 (65% of trades pass)
    |real_rate_diff| < 1.0%: Sharpe = 0.199, n = 176
    Delta = +0.451

  This is the largest single-gate improvement found in the research cycle.

THIS SCRIPT:
  1. Load trade_forensics.json (863 trades, 2015-2024, includes real_rate_diff)
  2. Compute Sharpe per threshold sweep (0.25%, 0.50%, 0.75%, 1.00%, 1.25%, 1.50%, 2.00%)
  3. Split IS (2015-2022) vs OOS (2023-2024) to validate the 1.0% threshold holds out-of-sample
  4. Per-pair breakdown to check if effect is concentrated in one pair
  5. Year-by-year breakdown to see regime sensitivity
  6. Test combined gate: |real_rate_diff| > 1.0% AND pair_trend > 0 (HYP-052c complement)
  7. Cross-check with nom_rate_diff vs real_rate_diff

Usage:
    python3 scripts/rq_rest_009_rate_level_gate_oos.py

Output: data/research/hyp_054_rate_level_gate_oos.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

FORENSICS_PATH = ROOT / "data" / "research" / "trade_forensics.json"
OUT_PATH       = ROOT / "data" / "research" / "hyp_054_rate_level_gate_oos.json"


# ─── Sharpe from trade list ──────────────────────────────────────────────────

def sharpe_from_trades(trades: list[dict], r_col: str = "outcome_r") -> float:
    """Annualized Sharpe from per-trade R values (assuming ~10 trades/yr)."""
    rs = np.array([t[r_col] for t in trades if t.get(r_col) is not None], dtype=float)
    if len(rs) < 5:
        return float("nan")
    # Trade-level Sharpe (mean/std) — comparable to prior REST cycle calculations
    return float(np.mean(rs) / np.std(rs, ddof=1)) if np.std(rs, ddof=1) > 0 else float("nan")


def stats(trades: list[dict]) -> dict:
    rs = np.array([t["outcome_r"] for t in trades if t.get("outcome_r") is not None], dtype=float)
    if len(rs) == 0:
        return {"n": 0, "sharpe": float("nan"), "mean_r": float("nan"), "win_rate": float("nan")}
    return {
        "n": len(rs),
        "sharpe": round(float(np.mean(rs) / np.std(rs, ddof=1)) if len(rs) > 1 else float("nan"), 4),
        "mean_r": round(float(np.mean(rs)), 4),
        "win_rate": round(float(np.mean(rs > 0)), 4),
    }


def year_of(t: dict) -> int:
    return int(str(t["entry_date"])[:4])


def main():
    print("RQ-REST-009: OOS Validation — Rate Level Gate (HYP-054)")
    print(f"Data: {FORENSICS_PATH}")
    print()

    trades_all = json.load(open(FORENSICS_PATH))
    print(f"Loaded {len(trades_all)} trades (2015-2024)")

    # Annotate year
    for t in trades_all:
        t["_year"] = year_of(t)

    # Split IS / OOS
    is_trades  = [t for t in trades_all if t["_year"] <= 2022]
    oos_trades = [t for t in trades_all if t["_year"] >= 2023]
    print(f"  IS  (≤2022): {len(is_trades)} trades")
    print(f"  OOS (≥2023): {len(oos_trades)} trades")
    print()

    # ─── 1. Full period baseline ─────────────────────────────────────────────
    print("=" * 60)
    print("FULL PERIOD BASELINE")
    full_stats = stats(trades_all)
    print(f"  All {full_stats['n']} trades: Sharpe={full_stats['sharpe']:.4f}  "
          f"mean_R={full_stats['mean_r']:.4f}  WR={full_stats['win_rate']:.1%}")

    # ─── 2. Threshold sweep on full period ───────────────────────────────────
    print()
    print("=" * 60)
    print("THRESHOLD SWEEP (full 2015-2024)")
    print(f"{'Threshold':>12} {'Pass%':>6} {'N_pass':>7} {'Sharpe_pass':>12} {'N_block':>8} {'Sharpe_block':>13}")
    thresholds = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
    sweep_results = {}
    for thr in thresholds:
        passing  = [t for t in trades_all if abs(t.get("real_rate_diff", 0) or 0) >= thr]
        blocking = [t for t in trades_all if abs(t.get("real_rate_diff", 0) or 0) < thr]
        sp = stats(passing)
        sb = stats(blocking)
        pct_pass = len(passing) / len(trades_all) * 100
        print(f"  {thr:>10.2f}%  {pct_pass:>5.1f}%  {sp['n']:>6}  {sp['sharpe']:>11.4f}  "
              f"{sb['n']:>7}  {sb['sharpe']:>12.4f}")
        sweep_results[f"threshold_{thr:.2f}pct"] = {
            "threshold": thr,
            "passing":  sp,
            "blocking": sb,
            "pct_passed": round(pct_pass, 1),
        }

    # ─── 3. IS vs OOS at 1.0% threshold ─────────────────────────────────────
    print()
    print("=" * 60)
    print("IS vs OOS SPLIT at threshold=1.00%")
    for label, subset in [("IS (2015-2022)", is_trades), ("OOS (2023-2024)", oos_trades)]:
        passing  = [t for t in subset if abs(t.get("real_rate_diff", 0) or 0) >= 1.0]
        blocking = [t for t in subset if abs(t.get("real_rate_diff", 0) or 0) < 1.0]
        sp, sb   = stats(passing), stats(blocking)
        delta = (sp["sharpe"] - sb["sharpe"]) if not np.isnan(sp["sharpe"]) and not np.isnan(sb["sharpe"]) else float("nan")
        print(f"\n  {label}:")
        print(f"    Passing ({sp['n']} trades):  Sharpe={sp['sharpe']:.4f}  mean_R={sp['mean_r']:.4f}  WR={sp['win_rate']:.1%}")
        print(f"    Blocking ({sb['n']} trades): Sharpe={sb['sharpe']:.4f}  mean_R={sb['mean_r']:.4f}  WR={sb['win_rate']:.1%}")
        print(f"    Delta (pass - block): {delta:+.4f}")

    is_pass_s  = stats([t for t in is_trades  if abs(t.get("real_rate_diff", 0) or 0) >= 1.0])
    oos_pass_s = stats([t for t in oos_trades if abs(t.get("real_rate_diff", 0) or 0) >= 1.0])
    is_all_s   = stats(is_trades)
    oos_all_s  = stats(oos_trades)

    oos_split = {
        "IS":  {"all": is_all_s,  "gated_1pct": is_pass_s},
        "OOS": {"all": oos_all_s, "gated_1pct": oos_pass_s},
    }

    # ─── 4. Per-pair breakdown at 1.0% ───────────────────────────────────────
    print()
    print("=" * 60)
    print("PER-PAIR BREAKDOWN at threshold=1.00%")
    pairs = sorted(set(t["pair"] for t in trades_all))
    pair_results = {}
    for pair in pairs:
        pt = [t for t in trades_all if t["pair"] == pair]
        passing  = [t for t in pt if abs(t.get("real_rate_diff", 0) or 0) >= 1.0]
        blocking = [t for t in pt if abs(t.get("real_rate_diff", 0) or 0) < 1.0]
        sp, sb = stats(passing), stats(blocking)
        pct = len(passing) / len(pt) * 100 if pt else 0
        print(f"  {pair}: {len(pt)} trades | pass={len(passing)} ({pct:.0f}%) "
              f"Sharpe={sp['sharpe']:.3f} | block={len(blocking)} Sharpe={sb['sharpe']:.3f}")
        pair_results[pair] = {"all": stats(pt), "passing": sp, "blocking": sb, "pct_passed": round(pct, 1)}

    # ─── 5. Year-by-year breakdown ────────────────────────────────────────────
    print()
    print("=" * 60)
    print("YEAR-BY-YEAR BREAKDOWN at threshold=1.00%")
    print(f"  {'Year':>6} {'N_pass':>7} {'N_block':>8} {'Sharpe_all':>11} {'Sharpe_pass':>12} {'Delta':>7}")
    year_results = {}
    for yr in sorted(set(t["_year"] for t in trades_all)):
        yt = [t for t in trades_all if t["_year"] == yr]
        passing  = [t for t in yt if abs(t.get("real_rate_diff", 0) or 0) >= 1.0]
        blocking = [t for t in yt if abs(t.get("real_rate_diff", 0) or 0) < 1.0]
        sa, sp, sb = stats(yt), stats(passing), stats(blocking)
        delta = (sp["sharpe"] - sa["sharpe"]) if not np.isnan(sp["sharpe"]) and not np.isnan(sa["sharpe"]) else float("nan")
        oos_marker = " ← OOS" if yr >= 2023 else ""
        print(f"  {yr:>6}  {len(passing):>6}  {len(blocking):>7}  "
              f"{sa['sharpe']:>10.3f}  {sp['sharpe']:>11.3f}  {delta:>+6.3f}{oos_marker}")
        year_results[yr] = {"all": sa, "passing": sp, "blocking": sb}

    # ─── 6. Combined gate: level + trend ─────────────────────────────────────
    # Approximate pair_trend gate using nom_rate_diff momentum (30d change)
    # We don't have per-trade rate history so use nom_rate_diff as a proxy:
    # trades where nom_rate_diff is meaningful in direction of carry
    print()
    print("=" * 60)
    print("COMBINED GATE: |real_rate_diff| >= 1.0% AND direction aligns with carry")
    # macro_vs_direction == 1 means the trade direction matches the macro signal
    carry_aligned = [t for t in trades_all if t.get("macro_vs_direction", 0) == 1]
    carry_misalign = [t for t in trades_all if t.get("macro_vs_direction", 0) != 1]
    level_gated    = [t for t in trades_all if abs(t.get("real_rate_diff", 0) or 0) >= 1.0]
    combined       = [t for t in level_gated if t.get("macro_vs_direction", 0) == 1]

    print(f"  carry_aligned only:   n={len(carry_aligned)}  Sharpe={stats(carry_aligned)['sharpe']:.4f}")
    print(f"  level_gated only:     n={len(level_gated)}  Sharpe={stats(level_gated)['sharpe']:.4f}")
    print(f"  combined (both):      n={len(combined)}  Sharpe={stats(combined)['sharpe']:.4f}")

    combined_results = {
        "carry_aligned": stats(carry_aligned),
        "level_gated_1pct": stats(level_gated),
        "combined_level_and_carry": stats(combined),
    }

    # ─── 7. Nominal vs real rate diff comparison ──────────────────────────────
    print()
    print("=" * 60)
    print("NOMINAL vs REAL RATE DIFF GATE (threshold=1.0%)")
    nom_pass = [t for t in trades_all if abs(t.get("nom_rate_diff", 0) or 0) >= 1.0]
    real_pass = [t for t in trades_all if abs(t.get("real_rate_diff", 0) or 0) >= 1.0]
    print(f"  Nominal gate: n={len(nom_pass)} ({len(nom_pass)/len(trades_all)*100:.0f}%)  "
          f"Sharpe={stats(nom_pass)['sharpe']:.4f}")
    print(f"  Real gate:    n={len(real_pass)} ({len(real_pass)/len(trades_all)*100:.0f}%)  "
          f"Sharpe={stats(real_pass)['sharpe']:.4f}")

    # ─── Save results ─────────────────────────────────────────────────────────
    print()
    opt_threshold = max(sweep_results.values(), key=lambda r: r["passing"]["sharpe"] if not np.isnan(r["passing"]["sharpe"]) else -999)
    print("=" * 60)
    print("VERDICT SUMMARY")
    print(f"  Baseline Sharpe (no gate): {full_stats['sharpe']:.4f}")
    print(f"  Optimal threshold: {opt_threshold['threshold']:.2f}%  →  Sharpe={opt_threshold['passing']['sharpe']:.4f}")
    print(f"  IS  Sharpe at 1.0%: {is_pass_s['sharpe']:.4f}  (vs IS all-trades: {is_all_s['sharpe']:.4f})")
    print(f"  OOS Sharpe at 1.0%: {oos_pass_s['sharpe']:.4f}  (vs OOS all-trades: {oos_all_s['sharpe']:.4f})")
    oos_confirmed = (
        not np.isnan(oos_pass_s["sharpe"]) and
        not np.isnan(oos_all_s["sharpe"]) and
        oos_pass_s["sharpe"] > oos_all_s["sharpe"]
    )
    verdict = "OOS_CONFIRMED" if oos_confirmed else "OOS_FAILED"
    print(f"  OOS result: {verdict}")

    output = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "run_by": "REST-004",
        "hypothesis": "HYP-054",
        "n_trades_total": len(trades_all),
        "n_is": len(is_trades),
        "n_oos": len(oos_trades),
        "baseline": full_stats,
        "threshold_sweep": sweep_results,
        "is_oos_split": oos_split,
        "per_pair": pair_results,
        "year_by_year": {str(yr): v for yr, v in year_results.items()},
        "combined_gates": combined_results,
        "nom_vs_real": {
            "nom_gate_1pct": stats(nom_pass),
            "real_gate_1pct": stats(real_pass),
        },
        "verdict": verdict,
        "optimal_threshold": opt_threshold["threshold"],
        "sharpe_at_1pct_full": sweep_results["threshold_1.00pct"]["passing"]["sharpe"],
        "sharpe_at_1pct_oos": oos_pass_s["sharpe"],
        "sharpe_baseline_oos": oos_all_s["sharpe"],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {OUT_PATH}")


if __name__ == "__main__":
    main()
