"""
Permutation test — Forex macro edge.
====================================

Question: does the rate-differential signal *timing* beat random entries fired at
the SAME frequency, through the SAME costed engine?

Method (reuses existing infra — no new backtester method):
  • ForexBatchBacktester.preload() builds per-pair ForexArrayDataset
    (opens, closes, signals, hold_days, index) using the real signal engine.
  • REAL: run the real signals through simulate_forex_trades_arrays, apply the
    Phase-1 cost model (ForexBacktester._apply_costs), compute per-pair Sharpe via
    ForexBacktester._compute_stats; aggregate to a √n-weighted portfolio Sharpe.
  • NULL (N permutations): for each pair, place ±1 at the SAME number of randomly
    chosen bars (random sign), reuse the real hold_days, run the identical engine
    + costs. Aggregate each permutation the same way.
  • p_value = P(null_portfolio_sharpe >= real_portfolio_sharpe).

If p < 0.05: the signal timing carries real information (edge has a structural cause).
If p >= 0.10: random entries match it — the observed performance may be luck.

Usage:  python3 scripts/permutation_test_forex.py [--perms 1000] [--seed 7]
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

from sovereign.forex.batch_backtester import ForexBatchBacktester
from sovereign.forex.fast_backtester import simulate_forex_trades_arrays
from sovereign.forex.forex_backtester import ForexBacktester
from sovereign.forex.pair_universe import ALL_PAIRS

OUT_PATH = ROOT / "data" / "research" / "permutation_test_forex.json"


def _weighted_portfolio_sharpe(per_pair: dict) -> float:
    """√n-weighted mean of per-pair Sharpes (matches holdout_validation_v014)."""
    items = [(s, n) for (s, n) in per_pair.values() if n > 0 and not np.isnan(s)]
    if not items:
        return 0.0
    w = [np.sqrt(n) for _, n in items]
    return float(sum(s * wi for (s, _), wi in zip(items, w)) / sum(w))


def _run_signals(bt: ForexBatchBacktester, pair: str, ds, signals: np.ndarray):
    """Run a signals array through the engine + Phase-1 cost model; return (sharpe, n)."""
    trades = simulate_forex_trades_arrays(
        opens=ds.opens,
        closes=ds.closes,
        signals=signals.astype(np.int8),
        hold_days=ds.hold_days,
        stop_pct=bt._backtester.STOP_PCT,
        index=ds.index,
    )
    if not trades:
        return 0.0, 0
    trades = ForexBacktester._apply_costs(trades, pair)
    stats = bt._backtester._compute_stats(pair, trades, len(ds.index))
    return float(stats.sharpe), int(stats.total_trades)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("\n" + "=" * 60)
    print("PERMUTATION TEST — FOREX MACRO EDGE")
    print(f"Permutations: {args.perms}  |  costed (spread+slip+swap)")
    print("=" * 60)

    bt = ForexBatchBacktester(start="2015-01-01", end="2024-12-31")
    print("\n  Preloading pairs (downloading + building real signals)...")
    bt.preload(list(ALL_PAIRS))
    pairs = sorted(bt._array_cache.keys())
    if not pairs:
        raise SystemExit("No pairs preloaded — cannot run permutation test.")
    print(f"  Loaded {len(pairs)} pairs: {', '.join(p.replace('=X','') for p in pairs)}")

    # ── REAL ──────────────────────────────────────────────────────────────
    real_per_pair, sig_counts = {}, {}
    for pair in pairs:
        ds = bt._array_cache[pair]
        sig = np.asarray(ds.signals)
        n_sig = int(np.count_nonzero(sig))
        sig_counts[pair] = n_sig
        real_per_pair[pair] = _run_signals(bt, pair, ds, sig)
    real_portfolio = _weighted_portfolio_sharpe(real_per_pair)
    real_n = sum(n for _, n in real_per_pair.values())
    print(f"\n  REAL portfolio Sharpe (√n-weighted, costed): {real_portfolio:.3f}  "
          f"({real_n} trades)")
    for p in pairs:
        s, n = real_per_pair[p]
        print(f"    {p.replace('=X',''):8s} sharpe={s:+.3f}  n={n}  signals={sig_counts[p]}")

    # ── NULL ──────────────────────────────────────────────────────────────
    print(f"\n  Running {args.perms} permutations (random entries, same frequency)...")
    null_portfolio = []
    bar_counts = {p: len(np.asarray(bt._array_cache[p].signals)) for p in pairs}
    for k in range(args.perms):
        perm_per_pair = {}
        for pair in pairs:
            ds = bt._array_cache[pair]
            n_bars = bar_counts[pair]
            n_sig = sig_counts[pair]
            rand_sig = np.zeros(n_bars, dtype=np.int8)
            if n_sig > 0:
                idx = rng.choice(n_bars, size=min(n_sig, n_bars), replace=False)
                rand_sig[idx] = rng.choice(np.array([-1, 1], dtype=np.int8), size=len(idx))
            perm_per_pair[pair] = _run_signals(bt, pair, ds, rand_sig)
        null_portfolio.append(_weighted_portfolio_sharpe(perm_per_pair))
        if (k + 1) % 100 == 0:
            print(f"    {k+1}/{args.perms} ...")

    null_arr = np.asarray(null_portfolio)
    p_value = float(np.mean(null_arr >= real_portfolio))
    pct95 = float(np.percentile(null_arr, 95))
    verdict = ("REAL" if p_value < 0.05 else
               "SUGGESTIVE" if p_value < 0.10 else "NOT_PROVEN")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Real portfolio Sharpe:  {real_portfolio:.3f}")
    print(f"  Null mean:              {null_arr.mean():.3f}")
    print(f"  Null std:               {null_arr.std():.3f}")
    print(f"  Null 95th percentile:   {pct95:.3f}")
    print(f"  Null max:               {null_arr.max():.3f}")
    print(f"  p-value:                {p_value:.4f}")
    print(f"  VERDICT: {verdict}")
    if verdict == "REAL":
        print("  Signal timing beats random entries at the same frequency — edge has")
        print("  a structural cause (p < 0.05).")
    elif verdict == "SUGGESTIVE":
        print("  Marginal evidence (p < 0.10). More data needed.")
    else:
        print("  Random entries match the real signals — observed performance may be luck.")

    result = {
        "test_date": datetime.now(timezone.utc).isoformat(),
        "system": "forex_macro",
        "n_permutations": args.perms,
        "seed": args.seed,
        "costed": True,
        "real_portfolio_sharpe": round(real_portfolio, 4),
        "real_n_trades": real_n,
        "real_per_pair": {p.replace("=X", ""): {"sharpe": round(s, 4), "n": n}
                           for p, (s, n) in real_per_pair.items()},
        "null_mean": round(float(null_arr.mean()), 4),
        "null_std": round(float(null_arr.std()), 4),
        "null_pct95": round(pct95, 4),
        "null_max": round(float(null_arr.max()), 4),
        "p_value": round(p_value, 4),
        "edge_above_95th": bool(real_portfolio > pct95),
        "verdict": verdict,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\n  Saved: {OUT_PATH.relative_to(ROOT)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
