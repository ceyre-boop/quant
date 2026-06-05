#!/usr/bin/env python3
"""Edge test 2026-06-05 — does currency MOMENTUM add orthogonal signal to CARRY?

Grounded in Menkhoff, Sarno, Schmeling & Schrimpf (2012), "Currency Momentum Strategies":
momentum is documented as orthogonal to carry (behavioral spot return-continuation vs carry's risk
premium), ~10%/yr cross-sectionally, but cost-sensitive and prone to OOS decay.

HONEST SCOPE: with 4 pairs / single directional carry trades, this can only test a TIME-SERIES
momentum-AGREEMENT overlay (does the carry direction agree with the pair's own 63-day spot momentum),
NOT the cross-sectional winner-minus-loser factor. Economically motivated by the paper; not a literal
replication. Prior anchor: the net-expectancy study's COUNTER_MOMENTUM group (momentum opposing dir,
n=35) had mean R 0.747 vs 0.242 — against-momentum did BETTER here — so NOT_SIGNIFICANT / backwards is
the base-rate expectation. Report whatever it is.

Methodology = the lens that caught the combat-rules survivorship error: NET expectancy over ALL trades
(winners + losers), both sides reported, 10k-shuffle permutation. Fully costed (v015 pnl_pct is net of
spread+swap). Read-only: writes data/research/edge_research_momentum.json + appends the ledger.

Usage:  python3 scripts/edge_research_momentum.py [--perms 10000]
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

from sovereign.research.trade_forensics import PAIR_COUNTRIES, _load_prices, _momentum_pct, _to_r

TRADES = ROOT / "logs" / "forex_backtest_trades.json"
OUT = ROOT / "data" / "research" / "edge_research_momentum.json"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
CITATION = "Menkhoff, Sarno, Schmeling & Schrimpf (2012), 'Currency Momentum Strategies'"


def _reconstruct():
    raw = json.loads(TRADES.read_text())
    price_cache, missing = {}, []
    trades = []
    for pair, lst in raw.items():
        if pair not in price_cache:
            price_cache[pair] = _load_prices(pair)
            if price_cache[pair] is None:
                missing.append(pair)
        prices = price_cache[pair]
        for tr in lst:
            d = pd.Timestamp(tr["entry_date"])
            direction = int(tr["direction"])
            mom = _momentum_pct(prices, d, 63) if prices is not None else 0.0
            pnl = float(tr["pnl_pct"])
            trades.append({
                "pair": pair, "entry_date": str(tr["entry_date"])[:10], "direction": direction,
                "pnl_pct": pnl, "R": _to_r(pnl, pair), "momentum_63d": mom,
                "mom_x_dir": mom * direction, "outcome": "WIN" if pnl > 0 else "LOSS",
            })
    return trades, missing


def _stats(group):
    if not group:
        return {"n": 0, "mean_R": None, "net_R": 0.0, "loss_only_R": 0.0, "sharpe": None}
    R = np.array([t["R"] for t in group], dtype=float)
    sd = float(R.std())
    return {
        "n": len(group), "mean_R": round(float(R.mean()), 4), "net_R": round(float(R.sum()), 2),
        "loss_only_R": round(float(sum(t["R"] for t in group if t["outcome"] == "LOSS")), 2),
        "sharpe": round(float(R.mean() / sd), 3) if sd > 0 else None,
    }


def _walk_forward(with_g, against_g):
    """Per-period mean_R(WITH) − mean_R(AGAINST); flags an edge concentrated in one window."""
    windows = [("2023H1", "2023-01-01", "2023-07-01"), ("2023H2", "2023-07-01", "2024-01-01"),
               ("2024H1", "2024-01-01", "2024-07-01"), ("2024H2", "2024-07-01", "2025-01-01")]
    out = []
    for name, lo, hi in windows:
        w = [t["R"] for t in with_g if lo <= t["entry_date"] < hi]
        a = [t["R"] for t in against_g if lo <= t["entry_date"] < hi]
        diff = (float(np.mean(w)) - float(np.mean(a))) if (w and a) else None
        out.append({"window": name, "n_with": len(w), "n_against": len(a),
                    "with_minus_against": round(diff, 4) if diff is not None else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    trades, missing = _reconstruct()
    with_g = [t for t in trades if t["mom_x_dir"] > 0]
    against_g = [t for t in trades if t["mom_x_dir"] < 0]
    excluded = len(trades) - len(with_g) - len(against_g)   # exact-zero momentum (negligible)

    sw, sa = _stats(with_g), _stats(against_g)

    # Permutation on d = mean_R(WITH) − mean_R(AGAINST), over the classified pool.
    pool = with_g + against_g
    R_pool = np.array([t["R"] for t in pool], dtype=float)
    n_with = len(with_g)
    p_high = p_low = d = None
    if n_with and len(against_g):
        d = float(sw["mean_R"] - sa["mean_R"])
        n = len(R_pool)
        null = np.empty(args.perms)
        for i in range(args.perms):
            idx = rng.permutation(n)[:n_with]
            sel = np.zeros(n, dtype=bool); sel[idx] = True
            null[i] = R_pool[sel].mean() - R_pool[~sel].mean()
        p_high = float((np.sum(null >= d) + 1) / (args.perms + 1))
        p_low = float((np.sum(null <= d) + 1) / (args.perms + 1))

    if p_high is None:
        verdict = "NO_DATA"
    elif p_high < 0.05:
        verdict = "VALID_EDGE"
    elif p_low < 0.05:
        verdict = "CONDITION_BACKWARDS"
    else:
        verdict = "NOT_SIGNIFICANT"
    low_power = (sw["n"] < 10) or (sa["n"] < 10)

    walk = _walk_forward(with_g, against_g) if verdict == "VALID_EDGE" else None

    hyp = {
        "id": "MOMENTUM-CARRY-ORTHO",
        "name": "Currency momentum as an orthogonal overlay on carry (WITH vs AGAINST 63d spot momentum)",
        "pre_registered": "WITH_MOMENTUM (carry agrees with 63d spot momentum sign) has higher mean R",
        "groups": {"WITH_MOMENTUM": sw, "AGAINST_MOMENTUM": sa},
        "excluded_zero_momentum": excluded,
        "d_mean_R": round(d, 4) if d is not None else None,
        "p_value": p_high, "p_backwards": p_low, "low_power": low_power,
        "verdict": verdict, "walk_forward": walk,
        "economic_rationale": CITATION + " — momentum is behavioral spot return-continuation, "
                              "orthogonal to carry's risk premium; documented cost-sensitive & OOS-decaying.",
        "construct_caveat": "Time-series momentum-AGREEMENT overlay on single carry trades, NOT the "
                            "cross-sectional winner-minus-loser factor (4 pairs can't form that). "
                            "Prior anchor: COUNTER_MOMENTUM group did better here (0.747 vs 0.242 R).",
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "universe": "v015 4-pair OOS (2023-2024)", "n_trades": len(trades),
            "metric": "R via trade_forensics._to_r(pnl_pct, pair) — FULLY COSTED (pnl_pct net of spread+swap)",
            "features_reconstructed": True, "missing_price_pairs": missing,
            "permutations": args.perms, "seed": args.seed,
            "method": "ALL trades partitioned WITH vs AGAINST (never conditioned on outcome); "
                      "one-sided 10k-shuffle permutation on mean-R difference.",
        },
        "hypothesis": hyp,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    # Ledger append (idempotent).
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    led = [e for e in led if e.get("id") != "MOMENTUM-CARRY-ORTHO"]
    led.append({
        "id": hyp["id"], "name": hyp["name"], "status": verdict,
        "date_tested": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "result": f"WITH n={sw['n']} meanR={sw['mean_R']} sharpe={sw['sharpe']}; "
                  f"AGAINST n={sa['n']} meanR={sa['mean_R']} sharpe={sa['sharpe']}; "
                  f"d={hyp['d_mean_R']} p={p_high}" + (" [LOW POWER]" if low_power else ""),
        "p_value": p_high,
        "methodology_note": ("Net expectancy over ALL trades; 10k-shuffle permutation; v015 4-pair OOS "
                             "n=103; fully costed; features reconstructed. Grounded in " + CITATION
                             + ". " + hyp["construct_caveat"]),
    })
    LEDGER.write_text(json.dumps(led, indent=2))

    # Print.
    print(f"\n{'='*76}\n  MOMENTUM × CARRY — v015 4-pair OOS (n={len(trades)}, fully costed, reconstructed)\n"
          f"  Grounded in {CITATION}\n{'='*76}")
    print(f"  pre-registered: WITH_MOMENTUM > AGAINST_MOMENTUM (carry+momentum complementary)")
    for g, v in hyp["groups"].items():
        print(f"    {g:18s} n={v['n']:3d}  mean_R={str(v['mean_R']):>8s}  net_R={v['net_R']:+8.2f}  "
              f"Sharpe={str(v['sharpe']):>7s}  (loss-only {v['loss_only_R']:+.2f})")
    print(f"    excluded (flat momentum): {excluded}")
    print(f"    d(WITH−AGAINST)={hyp['d_mean_R']}  p={p_high}  p_backwards={p_low}"
          + ("  [LOW POWER]" if low_power else ""))
    print(f"    VERDICT: {verdict}")
    if walk:
        print("    walk-forward (WITH−AGAINST per window):")
        for w in walk:
            print(f"      {w['window']}: {w['with_minus_against']}  (n_with={w['n_with']}, n_against={w['n_against']})")
    print(f"\n  Logged to hypothesis_ledger.json. Saved: data/research/edge_research_momentum.json\n{'='*76}\n")


if __name__ == "__main__":
    main()
