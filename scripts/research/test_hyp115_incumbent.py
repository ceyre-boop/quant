#!/usr/bin/env python3
"""HYP-115 — THE test. Runs once. Ladder exactly as sealed (f1b66afc75d792d1)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import prereg  # noqa: E402
from research.hyp111.date_bootstrap import stationary_block_indices, ci95  # noqa: E402
from research.incumbent.data import closes, CORE, WIDER  # noqa: E402

HYP = "HYP-115"
OUT = ROOT / "data" / "research" / "hyp115"


def sharpe(r):
    s = r.std(ddof=1); return float(r.mean() / s * np.sqrt(252)) if s > 0 else 0.0


def max_dd(r):
    eq = np.cumprod(1 + r); return float((eq / np.maximum.accumulate(eq) - 1).min())


def monthly_rebal(R: pd.DataFrame, bp: float) -> pd.Series:
    """Equal weight reset on the first session of each month; drift in between; turnover cost at rebalance."""
    w = np.full(R.shape[1], 1 / R.shape[1]); out = []; months = R.index.to_period("M")
    prev = None
    for i, (d, row) in enumerate(R.iterrows()):
        cost = 0.0
        if prev is not None and months[i] != prev:
            target = np.full(len(w), 1 / len(w)); cost = np.abs(target - w).sum() * bp; w = target
        g = w * np.exp(row.values); port = g.sum(); w = g / port
        out.append(np.log(port) - cost); prev = months[i]
    return pd.Series(out, index=R.index)


def main(argv) -> int:
    doc = prereg.gate_zero(HYP, "start")
    if "--gate-only" in argv:
        print("gate-only OK"); return 0
    P, W = doc["frozen_parameters"], doc["windows"]
    rng = np.random.default_rng(P["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    px = closes(); lr = np.log(px).diff().dropna(how="all")
    core = lr[CORE].dropna(); pool = lr[CORE + WIDER].dropna()
    def win(df, a, b): return df[(df.index >= a) & (df.index <= b)]
    IS, OOS = win(core, *W["in_sample"]), win(core, *W["out_of_sample"])
    poolIS, poolOOS = win(pool, *W["in_sample"]), win(pool, *W["out_of_sample"])
    print(f"\n{HYP} — {doc['name']}\nIS {IS.index[0].date()}→{IS.index[-1].date()} ({len(IS)})  OOS {OOS.index[0].date()}→{OOS.index[-1].date()} ({len(OOS)})  pool {pool.shape[1]} ETFs from {pool.index[0].date()}\n")

    A_is, A_oos = IS.mean(axis=1), OOS.mean(axis=1)                   # daily EW
    B_oos = monthly_rebal(OOS, P["turnover_bp"] / 1e4)
    spy_oos, spy_is = OOS["SPY"], IS["SPY"]
    sixforty = lambda df: 0.6 * df["SPY"] + 0.4 * df["TLT"]
    print(f"IS  (2015-26): basket Sharpe {sharpe(A_is.values):.3f}  total {np.expm1(A_is.sum())*100:+.1f}%  maxDD {max_dd(A_is.values)*100:.1f}%   SPY Sharpe {sharpe(spy_is.values):.3f}")
    print(f"OOS (2007-14): basket Sharpe {sharpe(A_oos.values):.3f}  total {np.expm1(A_oos.sum())*100:+.1f}%  maxDD {max_dd(A_oos.values)*100:.1f}%   SPY Sharpe {sharpe(spy_oos.values):.3f} maxDD {max_dd(spy_oos.values)*100:.1f}%   60/40 Sharpe {sharpe(sixforty(OOS).values):.3f}")
    print(f"OOS monthly-rebal 3bp: Sharpe {sharpe(B_oos.values):.3f}  total {np.expm1(B_oos.sum())*100:+.1f}%")

    # c1
    v = A_oos.values
    boot = np.array([sharpe(v[stationary_block_indices(rng, len(v), P["block_L"])]) for _ in range(P["draws"])])
    lo, hi = ci95(boot); c1 = lo > 0
    print(f"\n(c1) OOS Sharpe {sharpe(v):.3f}  95% CI [{lo:.3f}, {hi:.3f}]  {'PASS' if c1 else 'FAIL'}")

    # c2 random baskets
    def pct(poolw, target):
        cols = poolw.columns; vals = poolw.values; shs = np.empty(P["n_random_baskets"])
        for k in range(P["n_random_baskets"]):
            idx = rng.choice(len(cols), P["basket_size"], replace=False)
            shs[k] = sharpe(vals[:, idx].mean(axis=1))
        return float((shs < target).mean() * 100), shs
    pIS, shIS = pct(poolIS, sharpe(A_is.values)); pOOS, shOOS = pct(poolOOS, sharpe(A_oos.values))
    lucky = pIS >= 95 and pOOS <= 50; not_lucky = pOOS >= 50
    print(f"(c2) percentile among 10,000 random 10-ETF baskets: IS {pIS:.1f}  OOS {pOOS:.1f}   (random median Sharpe IS {np.median(shIS):.2f} OOS {np.median(shOOS):.2f})  "
          f"{'LUCKY' if lucky else ('not lucky' if not_lucky else 'below median OOS')}")

    # c3 stress
    stress = {}
    for name, (a, b) in W["stress"].items():
        seg = win(core, a, b)
        stress[name] = {"basket": max_dd(seg.mean(axis=1).values), "SPY": max_dd(seg["SPY"].values), "60/40": max_dd(sixforty(seg).values)}
    c3 = all(s["basket"] > s["SPY"] for s in stress.values())
    print("(c3) max drawdown:", {k: {kk: f"{vv*100:.1f}%" for kk, vv in s.items()} for k, s in stress.items()}, "PASS" if c3 else "FAIL")

    # descriptive
    yrs = core.groupby(core.index.year).apply(lambda g: pd.Series({"basket": np.expm1(g.mean(axis=1).sum()), "SPY": np.expm1(g["SPY"].sum()), "60/40": np.expm1(sixforty(g).sum())}))
    print("\ncalendar-year returns (%):"); print((yrs * 100).round(1).to_string())
    contrib = (OOS.sum() / 10 * 100).round(2).to_dict()
    print("OOS per-ETF contribution (% of basket log return):", contrib)

    if lucky: verdict = "LUCKY"
    elif not c1: verdict = "FRAGILE"
    elif not_lucky and c3: verdict = "VALIDATED_CORE"
    elif not_lucky: verdict = "ORDINARY"
    else: verdict = "ORDINARY"
    print(f"\n=== VERDICT: {verdict} ===\n")
    res = {"id": HYP, "hash_lock": doc["hash_lock"], "run_at": datetime.now(timezone.utc).isoformat(),
           "is": {"sharpe": sharpe(A_is.values), "total": float(np.expm1(A_is.sum())), "max_dd": max_dd(A_is.values)},
           "oos": {"sharpe": sharpe(v), "ci": [lo, hi], "total": float(np.expm1(A_oos.sum())), "max_dd": max_dd(v), "spy_sharpe": sharpe(spy_oos.values),
                   "sixforty_sharpe": sharpe(sixforty(OOS).values), "monthly_rebal_sharpe": sharpe(B_oos.values)},
           "c2": {"pct_is": pIS, "pct_oos": pOOS, "lucky": lucky}, "c3": {"stress": stress, "pass": c3},
           "calendar": yrs.to_dict(), "oos_contrib": contrib, "verdict": verdict}
    (OUT / "result.json").write_text(json.dumps(res, indent=2, default=float))
    prereg.adjudicate(HYP, verdict, verdict, {"oos_sharpe": sharpe(v), "p_value": float((boot <= 0).mean()), "result_file": "data/research/hyp115/result.json"})
    prereg.verify(HYP); return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
