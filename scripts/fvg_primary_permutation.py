#!/usr/bin/env python3
"""
FVG-primary permutation test — is the Fair Value Gap a standalone edge?
======================================================================

The "71% hit-rate FVG" claim was never permutation-tested; ICT around it is disproven
(p=0.52/0.59). This isolates FVG: fire entries on FVG-TAP PRESENCE ALONE (the pipeline's
validated fvg_tap component > 0, stripped of sweep/grade/displacement/commitment) vs random
entries from the same eligible-bar pool, run through the identical exit machinery.

  p < 0.05  → FVG carries a real standalone edge → Sleeve 3 (fvg_primary) is worth building.
  p >= 0.10 → FVG-alone is indistinguishable from random → park it; the 71% claim joins ICT
              as unvalidated.

Reuses run_ict_backtest (fetch/simulate_outcome/gates) + ICTPipeline (fvg_tap component) +
permutation_test_ict (_trade_R / vanilla-exit R). Decisive and cheap.

Usage:  python3 scripts/fvg_primary_permutation.py [--perms 1000] [--pairs GBPUSD EURUSD]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
for lib in ("yfinance", "peewee", "urllib3", "requests"):
    logging.getLogger(lib).setLevel(logging.ERROR)

from ict.pipeline import ICTPipeline
from ict.micro_risk import MicroRiskParams
from ict.session_classifier import SessionClassifier
from ict._atr_utils import compute_atr
from scripts.run_ict_backtest import fetch, PAIRS, LONDON_PAIRS, ACCOUNT_SIZE, MIN_BARS, MAX_HOLD_BARS
from scripts.permutation_test_ict import _trade_R

OUT = ROOT / "data" / "research" / "fvg_primary_permutation.json"


def _scan_fvg(pair: str):
    """Return (eligible_bars, fvg_fired) — fvg_fired = bars where the pipeline's fvg_tap
    component scored > 0 (an FVG tap is present), regardless of grade/other components."""
    df = fetch(pair)
    clean = pair.replace("=X", "")
    if df.empty or len(df) < MIN_BARS:
        return df, clean, [], []
    pipeline = ICTPipeline()
    sess_clf = SessionClassifier()
    account = MicroRiskParams(account_size=ACCOUNT_SIZE)
    eligible, fired = [], []
    for i in range(MIN_BARS, len(df) - MAX_HOLD_BARS - 2):
        ts = df.index[i].to_pydatetime()
        sess = sess_clf.classify(ts)
        if not sess.should_trade:
            continue
        kz = sess.kill_zone_name
        if kz == "NY_Open" or (kz == "London" and clean not in LONDON_PAIRS):
            continue
        window = df.iloc[i - MIN_BARS: i + 1]
        atr = compute_atr(window)
        if atr <= 0:
            continue
        sma50 = float(df["Close"].iloc[max(0, i - 50):i].mean()) if i >= 50 else None
        price = float(df["Close"].iloc[i])
        wt = "LONG" if (sma50 and price > sma50) else "SHORT" if (sma50 and price < sma50) else None
        if wt is None:
            continue
        eligible.append((i, wt, atr))
        try:
            res = pipeline.evaluate(symbol=clean, direction=wt, df=window, timestamp=ts, account=account, atr=atr)
            if getattr(res, "component_scores", {}).get("fvg_tap", 0) > 0:
                fired.append((i, wt, atr))
        except Exception:
            continue
    return df, clean, eligible, fired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pairs", nargs="+", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    pairs = [f"{p}=X" if "=X" not in p else p for p in args.pairs] if args.pairs else PAIRS

    print("\n" + "=" * 60 + "\nFVG-PRIMARY PERMUTATION — is FVG a standalone edge?\n" + "=" * 60)
    real_R, pools, n_by = [], {}, {}
    for pair in pairs:
        df, clean, eligible, fired = _scan_fvg(pair)
        if not eligible:
            print(f"  {clean}: no eligible bars"); continue
        pools[clean] = (df, eligible); n_by[clean] = len(fired)
        rr = _trade_R(df, fired); real_R.extend(rr)
        print(f"  {clean}: eligible={len(eligible)} fvg_fired={len(fired)} "
              + (f"meanR={np.mean(rr):+.3f}" if rr else "(0 fired)"))

    if not real_R:
        raise SystemExit("No FVG-tap entries fired — cannot test.")
    real_mean = float(np.mean(real_R))
    print(f"\n  REAL FVG-tap mean R: {real_mean:+.4f} (n={len(real_R)})")

    print(f"\n  Running {args.perms} permutations...")
    null = []
    for k in range(args.perms):
        perm = []
        for clean, (df, eligible) in pools.items():
            n = n_by.get(clean, 0)
            if n == 0 or not eligible:
                continue
            pick = rng.choice(len(eligible), size=min(n, len(eligible)), replace=False)
            perm.extend(_trade_R(df, [eligible[j] for j in pick]))
        if perm:
            null.append(float(np.mean(perm)))
    null = np.asarray(null)
    p_value = float(np.mean(null >= real_mean))
    verdict = "REAL" if p_value < 0.05 else "SUGGESTIVE" if p_value < 0.10 else "NOT_PROVEN"

    print("\n" + "=" * 60 + "\nRESULTS\n" + "=" * 60)
    print(f"  Real FVG mean R: {real_mean:+.4f} (n={len(real_R)})")
    print(f"  Null mean R: {null.mean():+.4f}  95th pct: {np.percentile(null,95):+.4f}")
    print(f"  p-value: {p_value:.4f}  →  VERDICT: {verdict}")
    if verdict == "REAL":
        print("  FVG taps beat random eligible entries — Sleeve 3 (fvg_primary) is worth building.")
    else:
        print("  FVG-alone does NOT beat random — park Sleeve 3; the 71% claim is unvalidated.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "test_date": datetime.now(timezone.utc).isoformat(), "system": "fvg_primary",
        "real_mean_R": round(real_mean, 4), "real_n": len(real_R),
        "fired_by_pair": n_by, "null_mean_R": round(float(null.mean()), 4),
        "null_pct95": round(float(np.percentile(null, 95)), 4),
        "p_value": round(p_value, 4), "verdict": verdict,
    }, indent=2))
    print(f"  Saved: {OUT.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
