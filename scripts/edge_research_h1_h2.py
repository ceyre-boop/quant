#!/usr/bin/env python3
"""Edge research 2026-06-05 — H1 (rate-diff acceleration) + H2 (volatility regime).

Pre-registered hypotheses tested on the v015 4-pair OOS trades (n=103) with the SAME rigorous
methodology that caught the combat-rules survivorship error: NET expectancy over ALL trades (winners
+ losers), both sides of every split, 10,000-shuffle permutation, verdict logged regardless of outcome.

H1: WIDENING rate differential (real_rate_diff 10-calendar-day delta > 0) → higher mean R?
H2: MED volatility (0.4% ≤ ATR_14d < 0.7%) outperforms both tails?

Features RECONSTRUCTED per (pair, entry_date) via trade_forensics functions (flagged). Outcome metric
R = _to_r(pnl_pct, pair). Read-only: writes data/research/edge_research_h1_h2.json + appends the ledger.

Usage:  python3 scripts/edge_research_h1_h2.py [--perms 10000]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.research.trade_forensics import (
    PAIR_COUNTRIES, _atr_pct, _load_prices, _load_rate_and_cpi_cache,
    _real_rate_diff, _to_r,
)

TRADES = ROOT / "logs" / "forex_backtest_trades.json"
OUT = ROOT / "data" / "research" / "edge_research_h1_h2.json"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"


def _reconstruct():
    raw = json.loads(TRADES.read_text())
    rate_cache, cpi_cache = _load_rate_and_cpi_cache()
    price_cache, missing = {}, []
    trades = []
    for pair, lst in raw.items():
        base, quote = PAIR_COUNTRIES.get(pair, ("US", "US"))
        if pair not in price_cache:
            price_cache[pair] = _load_prices(pair)
            if price_cache[pair] is None:
                missing.append(pair)
        prices = price_cache[pair]
        for tr in lst:
            d = pd.Timestamp(tr["entry_date"])
            _, real_now, _ = _real_rate_diff(base, quote, d, rate_cache, cpi_cache)
            _, real_prev, _ = _real_rate_diff(base, quote, d - pd.Timedelta(days=10), rate_cache, cpi_cache)
            atr = _atr_pct(prices, d) if prices is not None else 0.01
            pnl = float(tr["pnl_pct"])
            trades.append({
                "pair": pair, "direction": int(tr["direction"]), "pnl_pct": pnl,
                "R": _to_r(pnl, pair), "real_rate_diff": real_now,
                "rate_diff_delta": real_now - real_prev, "atr_14d_pct": atr,
                "outcome": "WIN" if pnl > 0 else "LOSS",
            })
    return trades, missing


def _group_stats(trades, R):
    n = len(trades)
    if n == 0:
        return {"n": 0, "mean_R": None, "net_R": 0.0, "loss_only_R": 0.0}
    return {
        "n": n,
        "mean_R": round(float(np.mean(R)), 4),
        "net_R": round(float(np.sum(R)), 2),
        "loss_only_R": round(float(sum(t["R"] for t in trades if t["outcome"] == "LOSS")), 2),
    }


def _permutation(R_all, fav_mask, perms, rng):
    """One-sided permutation: d = mean(favored) − mean(other); returns (d, p_high, p_low)."""
    n_fav = int(fav_mask.sum())
    if n_fav == 0 or n_fav == len(R_all):
        return None, None, None
    d = float(R_all[fav_mask].mean() - R_all[~fav_mask].mean())
    n = len(R_all)
    null = np.empty(perms)
    for i in range(perms):
        idx = rng.permutation(n)[:n_fav]
        sel = np.zeros(n, dtype=bool); sel[idx] = True
        null[i] = R_all[sel].mean() - R_all[~sel].mean()
    p_high = float((np.sum(null >= d) + 1) / (perms + 1))   # favored significantly HIGHER
    p_low = float((np.sum(null <= d) + 1) / (perms + 1))    # favored significantly LOWER
    return d, p_high, p_low


def _verdict(p_high, p_low):
    if p_high is None:
        return "NO_DATA"
    if p_high < 0.05:
        return "VALID_EDGE"
    if p_low < 0.05:
        return "CONDITION_BACKWARDS"
    return "NOT_SIGNIFICANT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    trades, missing = _reconstruct()
    R_all = np.array([t["R"] for t in trades], dtype=float)
    rng = np.random.default_rng(args.seed)
    results = []

    # ── H1: rate differential acceleration ───────────────────────────────────
    widening = [t for t in trades if t["rate_diff_delta"] > 0]
    narrowing = [t for t in trades if t["rate_diff_delta"] <= 0]
    fav_mask = np.array([t["rate_diff_delta"] > 0 for t in trades])
    d, p_high, p_low = _permutation(R_all, fav_mask, args.perms, rng)
    h1 = {
        "id": "H1-RATE-ACCEL", "name": "Rate differential acceleration (WIDENING > NARROWING)",
        "pre_registered": "WIDENING (real_rate_diff 10-calendar-day delta > 0) has higher mean R",
        "groups": {"WIDENING": _group_stats(widening, np.array([t["R"] for t in widening])),
                   "NARROWING": _group_stats(narrowing, np.array([t["R"] for t in narrowing]))},
        "d_mean_R": round(d, 4) if d is not None else None,
        "p_value": p_high, "p_backwards": p_low,
        "low_power": len(widening) < 10 or len(narrowing) < 10,
        "verdict": _verdict(p_high, p_low),
    }
    results.append(h1)

    # ── H2: volatility regime ─────────────────────────────────────────────────
    low = [t for t in trades if t["atr_14d_pct"] < 0.004]
    med = [t for t in trades if 0.004 <= t["atr_14d_pct"] < 0.007]
    high = [t for t in trades if t["atr_14d_pct"] >= 0.007]
    fav_mask2 = np.array([0.004 <= t["atr_14d_pct"] < 0.007 for t in trades])
    d2, p_high2, p_low2 = _permutation(R_all, fav_mask2, args.perms, rng)
    h2 = {
        "id": "H2-VOL-REGIME", "name": "Volatility regime (MED outperforms LOW & HIGH tails)",
        "pre_registered": "MED_VOL (0.4% ≤ ATR_14d < 0.7%) mean R exceeds LOW and HIGH",
        "groups": {"LOW_VOL(<0.4%)": _group_stats(low, np.array([t["R"] for t in low])),
                   "MED_VOL(0.4-0.7%)": _group_stats(med, np.array([t["R"] for t in med])),
                   "HIGH_VOL(>=0.7%)": _group_stats(high, np.array([t["R"] for t in high]))},
        "d_mean_R": round(d2, 4) if d2 is not None else None,
        "p_value": p_high2, "p_backwards": p_low2,
        "low_power": len(low) < 10 or len(med) < 10 or len(high) < 10,
        "verdict": _verdict(p_high2, p_low2),
    }
    results.append(h2)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "universe": "v015 4-pair OOS (2023-2024)", "n_trades": len(trades),
            "metric": "R via trade_forensics._to_r(pnl_pct, pair)",
            "features_reconstructed": True,
            "notes": "H1 uses real_rate_diff over 10 CALENDAR days; ATR_14d_pct reconstructed. "
                     "ALL trades per partition (never conditioned on outcome).",
            "missing_price_pairs": missing, "permutations": args.perms, "seed": args.seed,
        },
        "hypotheses": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    # ── Append to the ledger (idempotent) ────────────────────────────────────
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    led = [e for e in led if e.get("id") not in ("H1-RATE-ACCEL", "H2-VOL-REGIME")]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for h in results:
        grp = "; ".join(f"{k}: n={v['n']} meanR={v['mean_R']}" for k, v in h["groups"].items())
        led.append({
            "id": h["id"], "name": h["name"], "status": h["verdict"], "date_tested": today,
            "result": f"{grp} | d={h['d_mean_R']} p={h['p_value']}"
                      + (" [LOW POWER]" if h["low_power"] else ""),
            "p_value": h["p_value"],
            "methodology_note": ("Net expectancy over ALL trades (winners+losers); 10k-shuffle "
                                 "permutation on mean-R diff; v015 4-pair OOS n=103; features "
                                 "reconstructed. Pre-registered: " + h["pre_registered"]),
        })
    LEDGER.write_text(json.dumps(led, indent=2))

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*74}\n  EDGE RESEARCH 2026-06-05 — v015 4-pair OOS (n={len(trades)}, features reconstructed)\n{'='*74}")
    for h in results:
        print(f"\n  {h['id']} — {h['name']}")
        print(f"    pre-registered: {h['pre_registered']}")
        for g, v in h["groups"].items():
            print(f"      {g:20s} n={v['n']:3d}  mean_R={str(v['mean_R']):>8s}  net_R={v['net_R']:+8.2f}  (loss-only {v['loss_only_R']:+.2f})")
        print(f"    d(fav−other)={h['d_mean_R']}  p={h['p_value']}  p_backwards={h['p_backwards']}"
              + ("  [LOW POWER, n<10]" if h["low_power"] else ""))
        print(f"    VERDICT: {h['verdict']}")
    print(f"\n  Logged to hypothesis_ledger.json. Saved: data/research/edge_research_h1_h2.json\n{'='*74}\n")


if __name__ == "__main__":
    main()
