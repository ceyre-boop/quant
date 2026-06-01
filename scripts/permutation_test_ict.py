"""
Permutation test — ICT pattern edge.
====================================

Question: do the ICT-selected entry bars (sweep / FVG / displacement / killzone)
carry information — i.e. do they beat RANDOM entries drawn from the same eligible
bar pool, run through the identical exit machinery?

This isolates ENTRY SELECTION. Both legs use the same vanilla outcome simulation
(market entry, 1×ATR stop, 2R/4R targets via run_ict_backtest.simulate_outcome), so
the only thing that differs is *which bars* are chosen. If ICT bars don't beat random
eligible bars, the patterns add no timing information.

Method:
  • Per pair: fetch hourly data, walk the same eligible-bar gates as the live
    backtest (killzone + session + 50-SMA trend + cooldown), and record:
      - the ELIGIBLE bar pool [(i, with_trend_direction, atr)]
      - the ICT-FIRED subset (pipeline grade A/A+)
  • REAL: vanilla simulate_outcome on the ICT-fired bars → mean R.
  • NULL (N perms): sample len(fired) random bars from the eligible pool, vanilla
    simulate_outcome → mean R.
  • p_value = P(null_mean_R >= real_mean_R), pooled across pairs.

p-value is annualization-invariant (no Sharpe needed — pure mean-R comparison).

Usage:  python3 scripts/permutation_test_ict.py [--perms 1000] [--seed 7] [--pairs GBPUSD EURUSD]
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

from ict.pipeline import ICTPipeline, ICTSignal
from ict.micro_risk import MicroRiskParams
from ict.session_classifier import SessionClassifier
from ict._atr_utils import compute_atr
from scripts.run_ict_backtest import (
    fetch, simulate_outcome, PAIRS, LONDON_PAIRS,
    ACCOUNT_SIZE, MIN_BARS, MAX_HOLD_BARS,
)

OUT_PATH = ROOT / "data" / "research" / "permutation_test_ict.json"

# Vanilla exit ratios used for BOTH real and null (NORMAL regime — isolates entry).
TP1_R, TP2_R, TP1_FRAC = 2.0, 4.0, 0.5


def _pnl_r(outcome: str, entry: float, exit_price: float, atr: float, direction: str) -> float:
    """R-multiple under the vanilla 2R/4R scheme (mirrors run_ict_backtest logic)."""
    if outcome == "TP2":
        return TP1_FRAC * TP1_R + (1 - TP1_FRAC) * TP2_R          # 3.0
    if outcome == "TP1":
        return TP1_R * TP1_FRAC - (1 - TP1_FRAC) * 0.5            # 0.75
    if outcome == "STOP":
        return -1.0
    stop_dist = atr * 1.0
    sign = 1 if direction == "LONG" else -1
    return round(sign * (exit_price - entry) / max(stop_dist, 1e-9), 3)


def _scan_pair(pair: str):
    """Return (eligible_bars, fired_bars) where each is a list of (i, direction, atr).

    eligible_bars = bars passing the live backtest's session/killzone/trend gates.
    fired_bars    = eligible bars where the ICT pipeline returns a passed grade-A/A+ signal.
    """
    df = fetch(pair)
    clean = pair.replace("=X", "")
    if df.empty or len(df) < MIN_BARS:
        return df, clean, [], []

    pipeline = ICTPipeline()
    sess_clf = SessionClassifier()
    account = MicroRiskParams(account_size=ACCOUNT_SIZE)
    eligible, fired = [], []
    last_signal_bar = -10

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
        if i - last_signal_bar < 4:
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

        eligible.append((i, with_trend, atr))

        # Does the ICT pipeline actually fire here?
        try:
            result = pipeline.evaluate(symbol=clean, direction=with_trend,
                                       df=window, timestamp=ts, account=account, atr=atr)
        except Exception:
            continue
        if isinstance(result, ICTSignal) and result.passed:
            fired.append((i, result.direction, atr))
            last_signal_bar = i

    return df, clean, eligible, fired


def _trade_R(df, bars) -> list:
    """Vanilla simulate_outcome on each (i, direction, atr); return list of R-multiples."""
    out = []
    for (i, direction, atr) in bars:
        outcome, entry, exit_price, _ = simulate_outcome(df, i, direction, atr, fvg_limit=None)
        if entry == 0:
            continue
        out.append(_pnl_r(outcome, entry, exit_price, atr, direction))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pairs", nargs="+", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    pairs = [f"{p}=X" if "=X" not in p else p for p in args.pairs] if args.pairs else PAIRS

    print("\n" + "=" * 60)
    print("PERMUTATION TEST — ICT PATTERN EDGE (entry selection)")
    print(f"Permutations: {args.perms}")
    print("=" * 60)

    real_R, eligible_by_pair, n_fired_by_pair = [], {}, {}
    for pair in pairs:
        print(f"\n  Scanning {pair.replace('=X','')} (fetch + pipeline over eligible bars)...")
        df, clean, eligible, fired = _scan_pair(pair)
        if not eligible:
            print(f"    {clean}: no eligible bars — skipped")
            continue
        eligible_by_pair[clean] = (df, eligible)
        n_fired_by_pair[clean] = len(fired)
        rr = _trade_R(df, fired)
        real_R.extend(rr)
        print(f"    {clean}: eligible={len(eligible)}  fired={len(fired)}  "
              f"meanR={np.mean(rr):+.3f}" if rr else f"    {clean}: fired=0")

    if not real_R:
        raise SystemExit("No ICT trades fired — cannot run permutation test.")

    real_mean_R = float(np.mean(real_R))
    print(f"\n  REAL ICT mean R: {real_mean_R:+.4f}  (n={len(real_R)} trades)")

    print(f"\n  Running {args.perms} permutations (random eligible-bar entries)...")
    null_means = []
    for k in range(args.perms):
        perm_R = []
        for clean, (df, eligible) in eligible_by_pair.items():
            n = n_fired_by_pair[clean]
            if n == 0 or not eligible:
                continue
            pick = rng.choice(len(eligible), size=min(n, len(eligible)), replace=False)
            perm_R.extend(_trade_R(df, [eligible[j] for j in pick]))
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
    print(f"  Real ICT mean R:      {real_mean_R:+.4f}  (n={len(real_R)})")
    print(f"  Null mean R:          {null_arr.mean():+.4f}")
    print(f"  Null std:             {null_arr.std():.4f}")
    print(f"  Null 95th percentile: {pct95:+.4f}")
    print(f"  Null max:             {null_arr.max():+.4f}")
    print(f"  p-value:              {p_value:.4f}")
    print(f"  VERDICT: {verdict}")
    if verdict == "REAL":
        print("  ICT-selected bars beat random eligible bars (p < 0.05) — the patterns")
        print("  carry real entry-timing information.")
    elif verdict == "SUGGESTIVE":
        print("  Marginal (p < 0.10). More data needed.")
    else:
        print("  ICT bars do NOT beat random eligible bars — the patterns add no")
        print("  measurable timing edge. Hard truth, but better known before going live.")

    result = {
        "test_date": datetime.now(timezone.utc).isoformat(),
        "system": "ict_patterns",
        "test_type": "entry_selection (real ICT bars vs random eligible bars, same exits)",
        "n_permutations": args.perms,
        "seed": args.seed,
        "real_mean_R": round(real_mean_R, 4),
        "real_n_trades": len(real_R),
        "fired_by_pair": n_fired_by_pair,
        "null_mean_R": round(float(null_arr.mean()), 4),
        "null_std": round(float(null_arr.std()), 4),
        "null_pct95": round(pct95, 4),
        "null_max": round(float(null_arr.max()), 4),
        "p_value": round(p_value, 4),
        "edge_above_95th": bool(real_mean_R > pct95),
        "verdict": verdict,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\n  Saved: {OUT_PATH.relative_to(ROOT)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
