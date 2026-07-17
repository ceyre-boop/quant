"""Megascan — the largest strategy search in the repo. Daily-bar families across
the full liquid universe, per-asset AND pooled, real 12-month holdout.

Efficiency: metrics for every config are cheap; the permutation test is not, so
we compute metrics for ALL configs on the DIRTY window first, keep only those
beating the benchmark, and permutation-test just those — Bonferroni denominator
is the full config count (the honest trial count = distinct signal hypotheses;
sizing/exit permutations that don't change the hypothesis are NOT multiplied in).

Run: PYTHONPATH=~/quant python3 -m backtester.megascan
"""
import itertools
import json
import multiprocessing as _mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import daily_engine as de

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/scan_results"
OUT.mkdir(parents=True, exist_ok=True)
_CTX = _mp.get_context("fork" if os.name == "posix" else "spawn")

HOLDOUT_START = "2025-07-17"          # last 12 months = holdout; dirty is before
DIRTY_LO, DIRTY_HI = "0000", HOLDOUT_START
SIZING = 0.10
BENCH = dict(annual=0.18, sharpe=2.0, max_dd=0.15, per_year=50)
N_PERM = 200
SEED = 42


def _grids():
    """Yield (family, cfg) signal hypotheses. Each is a distinct rule."""
    fams = []
    # C — RSI mean reversion
    for direction, thrs in (("long", [15, 20, 25, 30]), ("short", [70, 75, 80, 85])):
        for thr, rn, hold, stop in itertools.product(
                thrs, [7, 14, 21], [1, 2, 3, 5, 10], [0.05, 0.10, 0.15]):
            fams.append(("rsi", dict(direction=direction, thr=thr, rsi_n=rn,
                                     hold_days=hold, stop_pct=stop)))
    # E — gap fade/follow
    for gap_dir, trade_dir in itertools.product(["up", "down"], ["fade", "follow"]):
        td = ("short" if (gap_dir == "up") == (trade_dir == "fade") else "long")
        for thr, hold, stop in itertools.product(
                [0.03, 0.05, 0.08, 0.10, 0.15], [1, 2, 3, 5], [0.05, 0.10, 0.15, 0.20]):
            fams.append(("gap", dict(gap_dir=gap_dir, trade_dir=td, thr=thr,
                                     hold_days=hold, stop_pct=stop)))
    # D — dip reversion (index-dip / HYP-095 spirit), long only
    for look, thr, hold, stop in itertools.product(
            [3, 5, 10, 20], [0.05, 0.08, 0.10, 0.15, 0.20], [1, 2, 3, 5, 10], [0.05, 0.10, 0.15]):
        fams.append(("dip", dict(look=look, thr=thr, trade_dir="long",
                                 hold_days=hold, stop_pct=stop)))
    # B/breakout — N-day channel break
    for direction, look, hold, stop in itertools.product(
            ["long", "short"], [10, 20, 50, 100], [1, 3, 5, 10, 20], [0.03, 0.05, 0.10]):
        fams.append(("breakout", dict(direction=direction, look=look,
                                      hold_days=hold, stop_pct=stop)))
    return fams


def _years(dates):
    if len(dates) < 2:
        return 0.0
    d0 = pd.to_datetime(min(dates))
    d1 = pd.to_datetime(max(dates))
    return max((d1 - d0).days / 365.25, 0.5)


def _run_ticker(args):
    """Metrics for every (family,cfg) on one ticker over the dirty window +
    the pooled per-trade returns (returned for later aggregation)."""
    ticker, grids = args
    df = de.load_daily(ticker)
    if df is None or len(df) < 400:
        return ticker, [], {}
    rows, pooled = [], {}
    for gi, (fam, cfg) in enumerate(grids):
        bt = de.backtest_daily(df, fam, cfg, DIRTY_LO, DIRTY_HI)
        rets, dts = bt["rets"], bt["dates"]
        pooled[gi] = (rets, dts)
        if len(rets) >= 8:
            m = de._metrics(rets, _years(dts), SIZING)
            rows.append((gi, ticker, fam, m))
    return ticker, rows, pooled


def _perm_p(rets, obs_sharpe, seed):
    if len(rets) < 8:
        return 1.0
    rng = np.random.default_rng(seed)
    r = np.array(rets)
    ge = 0
    for _ in range(N_PERM):
        s = rng.permutation(r)
        # sign-flip permutation: tests whether the mean edge is real, not luck
        flip = rng.choice([-1, 1], size=len(s))
        sr = s * flip
        sh = sr.mean() / sr.std() * np.sqrt(len(sr)) if sr.std() > 0 else 0.0
        if sh >= obs_sharpe:
            ge += 1
    return (ge + 1) / (N_PERM + 1)


def main():
    t0 = time.time()
    grids = _grids()
    tickers = sorted(p.stem for p in de.UNIVERSE.glob("*.parquet"))
    print(f"grids (distinct signal rules): {len(grids)} | tickers: {len(tickers)}")

    cores = max(1, (os.cpu_count() or 2) - 1)
    with _CTX.Pool(cores) as pool:
        results = pool.map(_run_ticker, [(t, grids) for t in tickers])

    # per-asset configs
    per_asset = []
    pooled_store = {gi: ([], []) for gi in range(len(grids))}
    for ticker, rows, pooled in results:
        for gi, tk, fam, m in rows:
            per_asset.append(dict(scope="asset", ticker=tk, family=fam,
                                  gi=gi, **m, **grids[gi][1]))
        for gi, (rets, dts) in pooled.items():
            pooled_store[gi][0].extend(rets)
            pooled_store[gi][1].extend(dts)

    # pooled-across-universe configs (systematic strategies)
    pooled_cfgs = []
    for gi, (rets, dts) in pooled_store.items():
        if len(rets) >= 30:
            fam = grids[gi][0]
            m = de._metrics(rets, _years(dts), SIZING)
            pooled_cfgs.append(dict(scope="pooled", ticker="UNIVERSE",
                                    family=fam, gi=gi, **m, **grids[gi][1]))

    all_cfgs = per_asset + pooled_cfgs
    n_tested = len(all_cfgs)
    df = pd.DataFrame(all_cfgs)
    print(f"configs evaluated (n_tested): {n_tested} "
          f"({len(per_asset)} per-asset + {len(pooled_cfgs)} pooled) "
          f"in {time.time()-t0:.1f}s")

    # metric survivors (beat benchmark on the dirty window)
    surv = df[(df.annual > BENCH["annual"]) & (df.sharpe > BENCH["sharpe"])
              & (df.max_dd < BENCH["max_dd"]) & (df.per_year >= BENCH["per_year"])].copy()
    print(f"metric-survivors (beat benchmark on dirty): {len(surv)}")

    # permutation + Bonferroni only on survivors
    if len(surv):
        praw = []
        for _, row in surv.iterrows():
            gi = int(row["gi"])
            if row["scope"] == "pooled":
                rets = pooled_store[gi][0]
            else:
                # recompute this ticker's rets for the config
                d = de.load_daily(row["ticker"])
                rets = de.backtest_daily(d, row["family"], grids[gi][1],
                                         DIRTY_LO, DIRTY_HI)["rets"]
            praw.append(_perm_p(rets, row["sharpe"], SEED + gi))
        surv["p_raw"] = praw
        surv["p_bonferroni"] = (surv["p_raw"] * n_tested).clip(upper=1.0)
        order = surv["p_raw"].rank(method="first")
        surv["p_holm"] = (surv["p_raw"] * (n_tested - order + 1)).clip(upper=1.0)
        surv["candidate"] = surv["p_bonferroni"] < 0.05
    else:
        surv["p_raw"] = surv["p_bonferroni"] = surv["p_holm"] = []
        surv["candidate"] = []

    df["n_tested"] = n_tested
    stamp = "20260717"
    df.to_parquet(OUT / f"megascan_{stamp}.parquet")
    surv.sort_values("p_bonferroni").to_parquet(OUT / f"megascan_survivors_{stamp}.parquet")
    n_cand = int(surv["candidate"].sum()) if len(surv) else 0
    summary = dict(n_tested=n_tested, n_metric_survivors=len(surv),
                   n_candidates_fwer=n_cand, benchmark=BENCH,
                   holdout_start=HOLDOUT_START, elapsed_s=round(time.time()-t0, 1))
    (OUT / f"megascan_summary_{stamp}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if len(surv):
        cols = ["scope", "ticker", "family", "annual", "sharpe", "max_dd",
                "per_year", "p_raw", "p_bonferroni", "candidate"]
        print(surv.sort_values("p_bonferroni")[cols].head(20).to_string())


if __name__ == "__main__":
    main()
