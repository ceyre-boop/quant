#!/usr/bin/env python3
"""
RQ-REST-013 (PARTIAL, no-network) — Trailing-stop drag forensic.

Context: RQ-REST-013 wants a full exit RE-SIM (fixed-time, fixed TP/SL,
widened trailing) against the 4-pair price history. That needs OHLC price
paths => network. Tonight (REST cycle 2026-06-17 ET) Yahoo is 403-blocked,
so the re-sim cannot run.

What CAN run with zero network: a forensic decomposition of the realized
-49R trailing-stop cluster (HYP-059 / RQ-REST-012) using only fields already
in logs/forex_backtest_trades.json: direction, hold_days, entry_date,
exit_reason, net R. Goal: localize WHERE the trailing drag concentrates so
the eventual re-sim (RQ-REST-013) tests the right levers first, and to check
whether any sub-cut is an ex-ante-selectable gate (unlike the un-selectable
"zero out all trailing exits" upper bound).

NO look-ahead beyond what RQ-REST-012 already used. This does not change any
live config. Output: data/agent/rq_rest_013_partial_results.json + findings.

Run: python3 scripts/rq_rest_013_partial_trailing_forensic.py
"""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

RNG = np.random.default_rng(13)
SRC = Path("logs/forex_backtest_trades.json")
OUT = Path("data/agent/rq_rest_013_partial_results.json")


def load_rows():
    d = json.load(open(SRC))
    rows = []
    for pair, tl in d.items():
        for t in tl:
            net_R = (t["pnl_pct"] - t.get("cost_spread_frac", 0.0)
                     + t.get("cost_swap_frac", 0.0)) / t["risk_pct"]
            rows.append(dict(
                pair=pair.replace("=X", ""),
                year=t["entry_date"][:4],
                month=int(t["entry_date"][5:7]),
                exit=t["exit_reason"],
                dirn="long" if t.get("direction", 1) >= 0 else "short",
                hold=int(t.get("hold_days", 0)),
                nR=net_R,
            ))
    return rows


def sharpe(a):
    a = np.asarray(a, float)
    return float(a.mean() / a.std(ddof=1)) if len(a) > 1 and a.std(ddof=1) > 0 else float("nan")


def stats(rs):
    a = np.array([r["nR"] for r in rs], float)
    if len(a) == 0:
        return dict(n=0, wr=float("nan"), meanR=float("nan"), sharpe=float("nan"), sumR=0.0)
    return dict(n=int(len(a)), wr=float(100 * np.mean(a > 0)), meanR=float(a.mean()),
                sharpe=sharpe(a), sumR=float(a.sum()))


def boot_mean_ci(a, n=5000):
    a = np.asarray(a, float)
    if len(a) < 2:
        return [float("nan"), float("nan")]
    bs = [RNG.choice(a, len(a), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    rows = load_rows()
    trail = [r for r in rows if r["exit"] == "trailing_stop"]
    res = {"baseline_trailing": stats(trail), "cuts": {}}

    # --- Cut 1: direction ---------------------------------------------------
    res["cuts"]["by_direction"] = {
        d: stats([r for r in trail if r["dirn"] == d]) for d in ("long", "short")
    }

    # --- Cut 2: hold-days buckets (whipsaw hypothesis) ----------------------
    def bucket(h):
        if h <= 2:
            return "0-2d"
        if h <= 5:
            return "3-5d"
        if h <= 10:
            return "6-10d"
        return "11d+"
    hb = defaultdict(list)
    for r in trail:
        hb[bucket(r["hold"])].append(r)
    res["cuts"]["by_hold_bucket"] = {k: stats(v) for k, v in
                                     sorted(hb.items(), key=lambda kv: kv[0])}

    # --- Cut 3: quarter (seasonality of the drag) ---------------------------
    def quarter(m):
        return f"Q{(m - 1) // 3 + 1}"
    qb = defaultdict(list)
    for r in trail:
        qb[quarter(r["month"])].append(r)
    res["cuts"]["by_quarter"] = {k: stats(qb[k]) for k in ("Q1", "Q2", "Q3", "Q4") if k in qb}

    # --- Cut 4: magnitude distribution (few catastrophic vs many small) -----
    arr = np.array([r["nR"] for r in trail], float)
    arr_sorted = np.sort(arr)  # most negative first
    worst5 = arr_sorted[:5]
    res["magnitude"] = {
        "min": float(arr.min()), "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)), "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "worst5_sumR": float(worst5.sum()),
        "worst5_share_of_drag_pct": float(100 * worst5.sum() / arr.sum()),
        "losers_n": int(np.sum(arr < 0)), "winners_n": int(np.sum(arr > 0)),
        "loser_sumR": float(arr[arr < 0].sum()), "winner_sumR": float(arr[arr > 0].sum()),
    }

    # --- Selectability check: is any ex-ante cut a clean gate? --------------
    # A cut is "ex-ante selectable" only if it keys on info known at ENTRY
    # (direction, pair, quarter) — NOT on hold_days or exit_reason (only known
    # at/after exit). We report drag concentration for entry-known cuts and
    # bootstrap the worst one to see if it is a real, sign-stable sub-cluster.
    short_trail = [r["nR"] for r in trail if r["dirn"] == "short"]
    long_trail = [r["nR"] for r in trail if r["dirn"] == "long"]
    res["selectable_gate_probe"] = {
        "note": ("direction/pair/quarter are known at entry => candidate gates. "
                 "hold_bucket is NOT (only known at exit) => diagnostic only."),
        "short_trailing_meanR_ci": boot_mean_ci(short_trail),
        "long_trailing_meanR_ci": boot_mean_ci(long_trail),
        "short_minus_long_meanR": (float(np.mean(short_trail)) - float(np.mean(long_trail))
                                   if short_trail and long_trail else None),
    }

    # --- Sanity: trailing drag total reconciles with RQ-REST-012 ------------
    res["reconcile_vs_rq012"] = {
        "trailing_sumR_here": float(arr.sum()),
        "rq012_trailing_sumR": -48.972676469682625,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
