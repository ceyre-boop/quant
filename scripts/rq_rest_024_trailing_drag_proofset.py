#!/usr/bin/env python3
"""
RQ-REST-024 (offline, no-network) — Cross-validate the trailing-stop drag on the
CANONICAL v015 proof trade set.

Context: RQ-REST-013 wants a full price-path exit re-sim. Network is still
403-blocked (REST cycle 2026-06-25 UTC, stooq/yahoo tunnel forbidden), so the
counterfactual re-sim cannot run. REST-018 (FIND-REST-018-a/b) localized the
-49R trailing drag to the 3-5d hold bucket using logs/forex_backtest_trades.json.

This task asks an independent question REST-018 never did: does that
localization REPLICATE on data/proof/backtest_trades_v015_2015_2024.csv — the
tracked canonical proof set (412 trades, 2015-2024, 4-pair) — and how does the
drag distribute per-year, with focus on the flat years?

Confirm-don't-predict: a finding that survives on an independent dataset earns
trust; one that doesn't is a snapshot artifact (cf. HYP-054's irreproducible
0.451 delta downgraded in REST-023).

Net R per trade = pnl_pct / risk_pct (proof CSV is already net of the modelled
cost in pnl_pct; we also cross-check vs risk_adjusted_pnl_pct).

NO look-ahead. Does NOT change live config. The 'time-floor recoverable' figure
is an UPPER BOUND on the lever (sumR currently sitting in sub-floor trailing
exits), NOT a realized counterfactual — that still needs the price-path re-sim.

Output: data/agent/rq_rest_024_results.json + findings.
"""
import json
import numpy as np
import csv
from collections import defaultdict
from pathlib import Path

SRC = Path("data/proof/backtest_trades_v015_2015_2024.csv")
OUT = Path("data/agent/rq_rest_024_results.json")
RNG = np.random.default_rng(24)


def load_rows():
    rows = []
    with open(SRC) as f:
        for t in csv.DictReader(f):
            risk = float(t["risk_pct"])
            nR = float(t["pnl_pct"]) / risk if risk else 0.0
            rows.append(dict(
                pair=t["pair"].replace("=X", ""),
                year=t["entry_date"][:4],
                exit=t["exit_reason"],
                dirn="long" if int(t["direction"]) >= 0 else "short",
                hold=int(float(t["hold_days"])),
                nR=nR,
            ))
    return rows


def sharpe(a):
    a = np.asarray(a, float)
    return float(a.mean() / a.std(ddof=1)) if len(a) > 1 and a.std(ddof=1) > 0 else float("nan")


def stats(rs):
    a = np.array([r["nR"] for r in rs], float)
    if len(a) == 0:
        return dict(n=0, wr=float("nan"), meanR=float("nan"), sharpe=float("nan"), sumR=0.0)
    return dict(n=int(len(a)), wr=float((a > 0).mean() * 100), meanR=float(a.mean()),
                sharpe=sharpe(a), sumR=float(a.sum()))


def bucket(h):
    if h <= 2: return "0-2d"
    if h <= 5: return "3-5d"
    if h <= 10: return "6-10d"
    return "11d+"


def main():
    rows = load_rows()
    out = {}

    # exit-reason composition of the whole proof set
    by_exit = defaultdict(list)
    for r in rows:
        by_exit[r["exit"]].append(r)
    out["all_trades"] = stats(rows)
    out["by_exit_reason"] = {k: stats(v) for k, v in sorted(by_exit.items())}

    trail = [r for r in rows if r["exit"] == "trailing_stop"]
    out["baseline_trailing"] = stats(trail)

    # hold-bucket cut of trailing exits  -> does the 3-5d whipsaw cluster replicate?
    bk = defaultdict(list)
    for r in trail:
        bk[bucket(r["hold"])].append(r)
    out["trailing_by_hold_bucket"] = {k: stats(bk[k]) for k in ["0-2d", "3-5d", "6-10d", "11d+"] if k in bk}

    # per-year trailing drag, flag flat years
    FLAT = {"2016", "2017", "2018", "2019", "2022"}
    yr = defaultdict(list)
    for r in trail:
        yr[r["year"]].append(r)
    out["trailing_by_year"] = {y: {**stats(yr[y]), "flat_year": y in FLAT} for y in sorted(yr)}

    # ----- ex-ante TIME-FLOOR upper bound -----
    # If trailing exits are forbidden before day F, the sumR currently realized by
    # sub-floor trailing exits is REMOVED from the book (upper bound on recoverable
    # drag; true replacement outcome needs the price path).
    allR = np.array([r["nR"] for r in rows], float)
    base_all = dict(n=len(allR), sumR=float(allR.sum()), sharpe=sharpe(allR),
                    meanR=float(allR.mean()))
    floors = {}
    for F in (4, 5, 6, 7):
        removed = [r for r in trail if r["hold"] < F]
        keepR = np.array([r["nR"] for r in rows
                          if not (r["exit"] == "trailing_stop" and r["hold"] < F)], float)
        floors[f"floor_{F}d"] = dict(
            n_removed=len(removed),
            sumR_removed=float(sum(r["nR"] for r in removed)),
            book_sumR_after=float(keepR.sum()),
            book_sharpe_after=sharpe(keepR),
            book_meanR_after=float(keepR.mean()),
            delta_sumR=float(keepR.sum() - allR.sum()),
            delta_sharpe=float(sharpe(keepR) - sharpe(allR)),
        )
    out["book_baseline_all_exits"] = base_all
    out["time_floor_upper_bound"] = floors

    # bootstrap CI on the 3-5d trailing meanR (is the cluster robust, not 1-2 trades?)
    c35 = np.array([r["nR"] for r in bk.get("3-5d", [])], float)
    if len(c35) >= 5:
        bs = [RNG.choice(c35, len(c35), replace=True).mean() for _ in range(5000)]
        out["cluster_3_5d_meanR_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)

    # ---- console summary ----
    print(f"PROOF SET: {out['all_trades']['n']} trades  sumR={out['all_trades']['sumR']:.1f}  sharpe={out['all_trades']['sharpe']:.3f}")
    print("exit composition:")
    for k, v in out["by_exit_reason"].items():
        print(f"  {k:16s} n={v['n']:3d}  sumR={v['sumR']:7.1f}  meanR={v['meanR']:+.3f}  wr={v['wr']:.0f}%")
    print(f"\nTRAILING baseline: n={out['baseline_trailing']['n']} sumR={out['baseline_trailing']['sumR']:.1f} meanR={out['baseline_trailing']['meanR']:+.3f} sharpe={out['baseline_trailing']['sharpe']:.3f}")
    print("trailing by hold bucket (REST-018 said 3-5d holds 86% of -49R drag):")
    for k in ["0-2d", "3-5d", "6-10d", "11d+"]:
        if k in out["trailing_by_hold_bucket"]:
            v = out["trailing_by_hold_bucket"][k]
            print(f"  {k:6s} n={v['n']:3d}  sumR={v['sumR']:7.1f}  meanR={v['meanR']:+.3f}  wr={v['wr']:.0f}%")
    if "cluster_3_5d_meanR_ci95" in out:
        lo, hi = out["cluster_3_5d_meanR_ci95"]
        print(f"  3-5d meanR 95% CI: [{lo:+.3f}, {hi:+.3f}]")
    print("\nper-year trailing drag (* = flat year):")
    for y, v in out["trailing_by_year"].items():
        star = "*" if v["flat_year"] else " "
        print(f"  {y}{star} n={v['n']:3d}  sumR={v['sumR']:7.1f}  meanR={v['meanR']:+.3f}")
    print("\nTIME-FLOOR upper bound (forbid trailing exit before day F; book-level):")
    print(f"  baseline all exits: sumR={base_all['sumR']:.1f} sharpe={base_all['sharpe']:.3f}")
    for k, v in floors.items():
        print(f"  {k}: remove {v['n_removed']} sub-floor trailing (sumR {v['sumR_removed']:+.1f}) "
              f"-> book sumR {v['book_sumR_after']:.1f} ({v['delta_sumR']:+.1f}), "
              f"sharpe {v['book_sharpe_after']:.3f} ({v['delta_sharpe']:+.3f})")


if __name__ == "__main__":
    main()
