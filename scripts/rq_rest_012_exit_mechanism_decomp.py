#!/usr/bin/env python3
"""
RQ-REST-012 — Exit-mechanism decomposition of the 4-pair forex edge.

Question: CLAUDE.md flags the proven forex macro edge as "regime-fragile"
(only pays in rate-trending regimes). Is that fragility seated in ENTRY
selection or in the EXIT mechanism?

Method: pure forensics on the costed v015/HYP-045 4-pair trade log
(logs/forex_backtest_trades.json, 318 trades 2015-2022, EURUSD/GBPUSD/
USDJPY/AUDUSD). Decompose net-of-cost per-trade R by exit_reason, by year,
and the year x exit interaction. Robustness: per-pair split, bootstrap CI on
the trailing-stop drag, permutation test (time vs trailing).

No network, no look-ahead in the test itself. The "zero-out" figure at the end
is an UPPER BOUND, explicitly not a strategy (you cannot select ex-ante which
trades the trailing stop will produce).

Run: python3 scripts/rq_rest_012_exit_mechanism_decomp.py
"""
import json, numpy as np
from collections import defaultdict
from pathlib import Path

RNG = np.random.default_rng(7)
SRC = Path("logs/forex_backtest_trades.json")
OUT = Path("data/agent/rq_rest_012_results.json")


def load_rows():
    d = json.load(open(SRC))
    rows = []
    for pair, tl in d.items():
        for t in tl:
            net_R = (t["pnl_pct"] - t.get("cost_spread_frac", 0.0)
                     + t.get("cost_swap_frac", 0.0)) / t["risk_pct"]
            rows.append(dict(pair=pair.replace("=X", ""), year=t["entry_date"][:4],
                             exit=t["exit_reason"], nR=net_R))
    return rows


def sharpe(a):
    a = np.asarray(a, float)
    return float(a.mean() / a.std(ddof=1)) if len(a) > 1 and a.std(ddof=1) > 0 else float("nan")


def stats(rs):
    a = np.array([r["nR"] for r in rs], float)
    return dict(n=len(a), wr=float(100 * np.mean(a > 0)), meanR=float(a.mean()),
                sharpe=sharpe(a), sumR=float(a.sum()))


def main():
    rows = load_rows()
    res = {"overall": stats(rows), "by_year": {}, "by_exit": {},
           "per_pair_trailing_vs_time": {}, "robustness": {}}

    by = defaultdict(list)
    for r in rows:
        by[("year", r["year"])].append(r)
        by[("exit", r["exit"])].append(r)
    for (k, v), rs in by.items():
        (res["by_year"] if k == "year" else res["by_exit"])[v] = stats(rs)

    for p in sorted(set(r["pair"] for r in rows)):
        tr = [r["nR"] for r in rows if r["pair"] == p and r["exit"] == "trailing_stop"]
        ti = [r["nR"] for r in rows if r["pair"] == p and r["exit"] == "time"]
        res["per_pair_trailing_vs_time"][p] = dict(
            trail_n=len(tr), trail_sumR=float(np.sum(tr)),
            trail_meanR=float(np.mean(tr)), time_sumR=float(np.sum(ti)))

    tr = np.array([r["nR"] for r in rows if r["exit"] == "trailing_stop"])
    ti = np.array([r["nR"] for r in rows if r["exit"] == "time"])
    boot = np.array([RNG.choice(tr, len(tr), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    obs = ti.mean() - tr.mean()
    pool = np.concatenate([ti, tr]); nA = len(ti); cnt = 0
    for _ in range(10000):
        RNG.shuffle(pool)
        if pool[:nA].mean() - pool[nA:].mean() >= obs:
            cnt += 1
    total = float(sum(r["nR"] for r in rows)); trail_sum = float(tr.sum())
    res["robustness"] = dict(
        trail_meanR_ci=[float(lo), float(hi)], trail_p_neg=float(np.mean(boot < 0)),
        time_minus_trail_delta=float(obs), time_gt_trail_p=cnt / 10000,
        total_net_R=total, trail_sum_R=trail_sum,
        zero_trailing_upper_bound_R=total - trail_sum,
        zero_trailing_uplift_pct=float(100 * (-trail_sum) / total),
        upper_bound_caveat="NOT a strategy — trailing exits cannot be selected ex-ante.")

    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
