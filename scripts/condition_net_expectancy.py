#!/usr/bin/env python3
"""Corrected Phase 0 — per-condition NET-expectancy study (read-only analysis).

Does what the combat-rules forensic should have done: measures each condition's NET expectancy
(winners + losers, never conditioning on outcome) on the CORRECT universe (v015 4-pair OOS trades),
with a permutation test. Answers: are there conditions with statistically significant NEGATIVE net
expectancy that would justify a selection veto (or POSITIVE that justify a boost)?

Features are RECONSTRUCTED per (pair, entry_date) by reusing the forensic functions (the v015 trades
file doesn't store them) — FLAGGED in the output. Outcome metric is R via the forensic's own
_to_r(pnl_pct, pair) per-pair avg-stop conversion.

Writes data/research/condition_net_expectancy.json. Touches no live code/config. Read-only.

Usage:  python3 scripts/condition_net_expectancy.py [--perms 10000]
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
    _momentum_pct, _real_rate_diff, _to_r,
)

TRADES = ROOT / "logs" / "forex_backtest_trades.json"
OUT = ROOT / "data" / "research" / "condition_net_expectancy.json"

# (name, hypothesis, predicate) — hypothesis 'veto' tests WITH-mean LOWER; 'boost' tests HIGHER.
CONDITIONS = [
    ("MACRO_AGAINST",   "veto",  lambda t: t["macro_vs_direction"] == -1),
    ("COUNTER_MOMENTUM","veto",  lambda t: t["momentum_63d"] * t["direction"] < -0.01),
    ("WEAK_RATE",       "veto",  lambda t: abs(t["real_rate_diff"]) < 0.5),
    ("LOW_VOLATILITY",  "veto",  lambda t: t["atr_14d_pct"] < 0.006),
    ("LOW_MOM_LOW_VOL", "veto",  lambda t: abs(t["momentum_63d"]) < 0.005 and t["atr_14d_pct"] < 0.007),
    ("STRONG_ALIGNED",  "boost", lambda t: abs(t["real_rate_diff"]) >= 2.0 and t["macro_vs_direction"] == 1
                                            and t["momentum_63d"] * t["direction"] > 0),
    ("STRONG_RATE",     "boost", lambda t: abs(t["real_rate_diff"]) >= 2.5),
]


def _reconstruct():
    raw = json.loads(TRADES.read_text())
    rate_cache, cpi_cache = _load_rate_and_cpi_cache()
    price_cache, missing_prices = {}, []
    trades = []
    for pair, lst in raw.items():
        base, quote = PAIR_COUNTRIES.get(pair, ("US", "US"))
        if pair not in price_cache:
            price_cache[pair] = _load_prices(pair)
            if price_cache[pair] is None:
                missing_prices.append(pair)
        prices = price_cache[pair]
        for tr in lst:
            d = pd.Timestamp(tr["entry_date"])
            direction = int(tr["direction"])
            _, real, _ = _real_rate_diff(base, quote, d, rate_cache, cpi_cache)
            mom = _momentum_pct(prices, d) if prices is not None else 0.0
            atr = _atr_pct(prices, d) if prices is not None else 0.01
            macro_sign = int(np.sign(real)) if abs(real) > 0.2 else 0
            pnl = float(tr["pnl_pct"])
            trades.append({
                "pair": pair, "direction": direction, "pnl_pct": pnl,
                "R": _to_r(pnl, pair), "real_rate_diff": real, "momentum_63d": mom,
                "atr_14d_pct": atr, "macro_vs_direction": macro_sign * direction,
                "outcome": "WIN" if pnl > 0 else "LOSS",
            })
    return trades, missing_prices


def _permutation_p(R, mask, hypothesis, perms, rng):
    """Empirical one-sided p that the WITH-group mean is lower (veto) / higher (boost) than chance."""
    n_with = int(mask.sum())
    if n_with == 0 or n_with == len(R):
        return None
    with_mean, without_mean = R[mask].mean(), R[~mask].mean()
    real_diff = with_mean - without_mean
    n = len(R)
    null = np.empty(perms)
    for i in range(perms):
        idx = rng.permutation(n)[:n_with]
        sel = np.zeros(n, dtype=bool); sel[idx] = True
        null[i] = R[sel].mean() - R[~sel].mean()
    if hypothesis == "veto":
        p = float((np.sum(null <= real_diff) + 1) / (perms + 1))
    else:
        p = float((np.sum(null >= real_diff) + 1) / (perms + 1))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    trades, missing = _reconstruct()
    R = np.array([t["R"] for t in trades], dtype=float)
    rng = np.random.default_rng(args.seed)
    per_pair = {}
    for t in trades:
        per_pair[t["pair"]] = per_pair.get(t["pair"], 0) + 1

    rows = []
    for name, hyp, pred in CONDITIONS:
        mask = np.array([bool(pred(t)) for t in trades])
        n_with = int(mask.sum())
        with_R, without_R = R[mask], R[~mask]
        mean_with = float(with_R.mean()) if n_with else None
        mean_without = float(without_R.mean()) if n_with < len(R) else None
        net_with = float(with_R.sum()) if n_with else 0.0
        loss_only = float(sum(t["R"] for t in trades if pred(t) and t["outcome"] == "LOSS"))
        p = _permutation_p(R, mask, hyp, args.perms, rng)

        if p is None:
            verdict = "NO_DATA"
        elif hyp == "veto" and net_with < 0 and p < 0.05:
            verdict = "VALID_VETO"
        elif hyp == "boost" and net_with > 0 and p < 0.05:
            verdict = "VALID_BOOST"
        elif mean_without is not None and abs(mean_with - mean_without) < 0.05:
            verdict = "NO_EFFECT"
        else:
            verdict = "NOT_SIGNIFICANT"

        rows.append({
            "condition": name, "hypothesis": hyp,
            "n_with": n_with, "mean_R_with": mean_with, "net_R_with": round(net_with, 2),
            "loss_only_R_with": round(loss_only, 2),
            "n_without": len(R) - n_with, "mean_R_without": mean_without,
            "p_value": p, "verdict": verdict,
        })

    valid_vetoes = [r["condition"] for r in rows if r["verdict"] == "VALID_VETO"]
    valid_boosts = [r["condition"] for r in rows if r["verdict"] == "VALID_BOOST"]
    n_clear = len(valid_vetoes) + len(valid_boosts)
    forensic_verdict = ("REFUTED — no condition is a statistically significant net-negative veto"
                        if not valid_vetoes else
                        f"PARTIALLY VALID — valid vetoes: {valid_vetoes}")
    summary = (
        f"{n_clear} of {len(CONDITIONS)} conditions clear the bar (p<0.05, net sign as hypothesized). "
        f"Valid vetoes: {valid_vetoes or 'NONE'}. Valid boosts: {valid_boosts or 'NONE'}. "
        f"The combat-rules forensic's veto thesis is {forensic_verdict}. "
        f"Recommendation: " + (
            "no selection gate is justified on this universe — the entry engine is geometry (Phase 1); "
            "sizing on these conditions would be sizing on noise."
            if not valid_vetoes else
            f"a veto around {valid_vetoes} is justified, but build it as a SEPARATE step held to the "
            f"same OOS/permutation discipline.") +
        f" [n={len(R)}, small sample → low power; many NOT_SIGNIFICANT verdicts are sample-size, not proof of no effect.]"
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "universe": "v015 4-pair OOS (2023-2024)", "n_trades": len(R), "per_pair": per_pair,
            "metric": "R via trade_forensics._to_r(pnl_pct, pair) (per-pair avg-stop conversion)",
            "features_reconstructed": True,
            "reconstruction_note": ("real_rate_diff/momentum_63d/atr_14d_pct/macro_vs_direction "
                                    "recomputed per (pair,entry_date) via trade_forensics functions — "
                                    "NOT stored with the v015 trades."),
            "missing_price_pairs": missing,
            "permutations": args.perms, "seed": args.seed,
            "method": "ALL trades partitioned WITH vs WITHOUT (never conditioned on outcome); "
                      "one-sided permutation test on mean-R difference.",
        },
        "conditions": rows,
        "summary": summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"\n{'='*78}\n  PER-CONDITION NET EXPECTANCY — v015 4-pair OOS (n={len(R)}, "
          f"features reconstructed)\n{'='*78}")
    print(f"  {'condition':17s} {'hyp':5s} {'n_w':>4s} {'meanR_w':>8s} {'meanR_wo':>9s} "
          f"{'netR_w':>8s} {'lossOnly':>9s} {'p':>6s}  verdict")
    for r in rows:
        mw = f"{r['mean_R_with']:+.3f}" if r["mean_R_with"] is not None else "   n/a"
        mo = f"{r['mean_R_without']:+.3f}" if r["mean_R_without"] is not None else "    n/a"
        pv = f"{r['p_value']:.3f}" if r["p_value"] is not None else " n/a"
        print(f"  {r['condition']:17s} {r['hypothesis']:5s} {r['n_with']:4d} {mw:>8s} {mo:>9s} "
              f"{r['net_R_with']:+8.2f} {r['loss_only_R_with']:+9.2f} {pv:>6s}  {r['verdict']}")
    print(f"\n  {summary}\n  Saved: data/research/condition_net_expectancy.json\n{'='*78}\n")


if __name__ == "__main__":
    main()
