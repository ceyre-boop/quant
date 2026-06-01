"""
Permutation test — ICT pattern edge WITH REAL EXITS (Loop 4, Option B).
=======================================================================

The Phase-2 ICT permutation used vanilla exits (market entry, 1×ATR stop, fixed
2R/4R) and found p=0.52 — ICT entry selection ≈ random. But ICT's actual claim is
that its FULL machinery — FVG-limit entries + structural (swept-level) stops +
regime-aware TP — produces better outcomes. This re-test honours that claim:

  • REAL leg: the live ICT trades from run_ict_backtest.backtest_pair (FVG-limit
    entry, structural stop, regime TP) — the real exit machinery.
  • NULL leg: random entries from the same eligible-bar pool, run through the
    system's DEFAULT handling for a no-structure entry (market entry, 1×ATR stop,
    SESSION-matched regime TP). Random bars have no sweep/FVG, so ATR-stop is the
    honest fallback — exactly what the live system uses when no structure is present.

This isolates the contribution of ICT's entry+stop precision (regime TP is held
session-matched so it is not the confound). p = P(null_mean_R >= real_mean_R).

  p < 0.05  → ICT's edge is real, living in the entry+exit combination.
  p >= 0.10 → ICT unproven even with its real machinery → simplify (FVG-only) or park.

Usage:  python3 scripts/permutation_test_ict_realexits.py [--perms 1000] [--pairs GBPUSD ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
for lib in ("yfinance", "peewee", "urllib3", "requests"):
    logging.getLogger(lib).setLevel(logging.ERROR)

from ict.session_classifier import SessionClassifier
from ict._atr_utils import compute_atr
from scripts.run_ict_backtest import (
    fetch, backtest_pair, simulate_outcome, _regime_tp,
    PAIRS, LONDON_PAIRS, MIN_BARS, MAX_HOLD_BARS, TP1_FRAC,
)

OUT_PATH = ROOT / "data" / "research" / "permutation_test_ict_realexits.json"


def _eligible_pool(df, clean: str) -> list:
    """Gate-only scan (NO pipeline) → list of (i, with_trend_direction, atr) bars the
    live system would consider. Mirrors backtest_pair's session/killzone/trend gates."""
    sess_clf = SessionClassifier()
    pool = []
    for i in range(MIN_BARS, len(df) - MAX_HOLD_BARS - 2):
        ts = df.index[i].to_pydatetime()
        sess = sess_clf.classify(ts)
        if not sess.should_trade:
            continue
        kz = sess.kill_zone_name
        if kz == "NY_Open":
            continue
        if kz == "London" and clean not in LONDON_PAIRS:
            continue
        window = df.iloc[i - MIN_BARS: i + 1]
        atr = compute_atr(window)
        if atr <= 0:
            continue
        sma50 = float(df["Close"].iloc[max(0, i - 50):i].mean()) if i >= 50 else None
        price_now = float(df["Close"].iloc[i])
        with_trend = ("LONG" if (sma50 and price_now > sma50)
                      else "SHORT" if (sma50 and price_now < sma50) else None)
        if with_trend is None:
            continue
        pool.append((i, with_trend, kz, atr))
    return pool


def _null_exit_R(df, i: int, direction: str, kz: str, atr: float, clean: str) -> float | None:
    """Run a random entry through the system's DEFAULT exit handling:
    market entry, 1×ATR stop, SESSION-matched regime TP. Returns R-multiple."""
    outcome, entry, exit_price, _ = simulate_outcome(df, i, direction, atr, fvg_limit=None)
    if entry == 0:
        return None
    # Session-matched regime TP (mirrors backtest_pair: GBPUSD London = TRENDING,
    # else NORMAL — score-based bump omitted since random entries have no score).
    if clean == "GBPUSD" and kz == "London":
        tp1_r, tp2_r = _regime_tp("TRENDING")
    else:
        tp1_r, tp2_r = _regime_tp("NORMAL")
    if outcome == "TP2":
        return TP1_FRAC * tp1_r + (1 - TP1_FRAC) * tp2_r
    if outcome == "TP1":
        return tp1_r * TP1_FRAC - (1 - TP1_FRAC) * 0.5
    if outcome == "STOP":
        return -1.0
    sign = 1 if direction == "LONG" else -1
    return round(sign * (exit_price - entry) / max(atr, 1e-9), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pairs", nargs="+", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    pairs = [f"{p}=X" if "=X" not in p else p for p in args.pairs] if args.pairs else PAIRS

    print("\n" + "=" * 60)
    print("PERMUTATION TEST — ICT WITH REAL EXITS (Loop 4 / Option B)")
    print(f"Permutations: {args.perms}")
    print("=" * 60)

    real_R, pools, n_by_pair = [], {}, {}
    for pair in pairs:
        clean = pair.replace("=X", "")
        print(f"\n  {clean}: real ICT trades (FVG-limit + structural stop + regime TP)...")
        real_trades = backtest_pair(pair)
        rr = [t.pnl_r for t in real_trades]
        real_R.extend(rr)
        n_by_pair[clean] = len(rr)
        print(f"    real trades={len(rr)}  meanR={np.mean(rr):+.3f}" if rr else f"    real trades=0")
        print(f"  {clean}: building eligible pool (gate-only)...")
        df = fetch(pair)
        pools[clean] = (df, _eligible_pool(df, clean))
        print(f"    eligible bars={len(pools[clean][1])}")

    if not real_R:
        raise SystemExit("No real ICT trades — cannot run test.")
    real_mean_R = float(np.mean(real_R))
    print(f"\n  REAL ICT mean R (real exits): {real_mean_R:+.4f}  (n={len(real_R)})")

    print(f"\n  Running {args.perms} permutations (random entries, real-exit handling)...")
    null_means = []
    for k in range(args.perms):
        perm_R = []
        for clean, (df, pool) in pools.items():
            n = n_by_pair.get(clean, 0)
            if n == 0 or not pool:
                continue
            pick = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
            for j in pick:
                i, direction, kz, atr = pool[j]
                r = _null_exit_R(df, i, direction, kz, atr, clean)
                if r is not None:
                    perm_R.append(r)
        if perm_R:
            null_means.append(float(np.mean(perm_R)))
        if (k + 1) % 100 == 0:
            print(f"    {k+1}/{args.perms} ...")

    null_arr = np.asarray(null_means)
    p_value = float(np.mean(null_arr >= real_mean_R))
    pct95 = float(np.percentile(null_arr, 95))
    verdict = ("REAL" if p_value < 0.05 else
               "SUGGESTIVE" if p_value < 0.10 else "NOT_PROVEN")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Real ICT mean R (real exits): {real_mean_R:+.4f}  (n={len(real_R)})")
    print(f"  Null mean R:                  {null_arr.mean():+.4f}")
    print(f"  Null 95th percentile:         {pct95:+.4f}")
    print(f"  Null max:                     {null_arr.max():+.4f}")
    print(f"  p-value:                      {p_value:.4f}")
    print(f"  VERDICT: {verdict}")
    if verdict == "REAL":
        print("  ICT's full machinery (FVG-limit entry + structural stop + regime TP) beats")
        print("  random entries with default exits — the edge is in the entry+exit combination.")
    elif verdict == "SUGGESTIVE":
        print("  Marginal (p<0.10). More data needed before trusting ICT.")
    else:
        print("  ICT is NOT proven even with its real exit machinery. Simplify to FVG-only or")
        print("  park ICT and trade the proven forex macro edge. Cap ICT at <=0.25% until validated.")

    result = {
        "test_date": datetime.now(timezone.utc).isoformat(),
        "system": "ict_real_exits",
        "test_type": "real ICT (FVG-limit+structural stop+regime TP) vs random entries + default exits",
        "n_permutations": args.perms,
        "seed": args.seed,
        "real_mean_R": round(real_mean_R, 4),
        "real_n_trades": len(real_R),
        "real_n_by_pair": n_by_pair,
        "null_mean_R": round(float(null_arr.mean()), 4),
        "null_pct95": round(pct95, 4),
        "null_max": round(float(null_arr.max()), 4),
        "p_value": round(p_value, 4),
        "verdict": verdict,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\n  Saved: {OUT_PATH.relative_to(ROOT)}")
    print("=" * 60 + "\n")
    return result


if __name__ == "__main__":
    main()
