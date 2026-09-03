#!/usr/bin/env python3
"""THE 20-YEAR G10 CARRY MAP — Step 1 of the method (mine the dirty data, extract structure).
NOT a test. NOT a verdict. Every number here is in-sample and multiple-compared (~90 cells); the point is
to see WHERE and WHEN the carry premium was paid over 2006-06 → 2026-04 so ONE claim can be sealed after.

Definitions (fixed, not swept):
  rate_c(t)      OECD immediate short rate for currency c, published for month t-1 (no look-ahead)
  excess_c(t)    Δlog(USD value of c) over month t  +  (rate_c − rate_USD)/1200      [vs USD]
  CARRY factor   rank 10 currencies (USD included) by rate_c(t); long top 3, short bottom 3, equal weight
  pair carry     for each of the 45 pairs: long the higher-rate leg by rate(t), monthly
  states         all known at the end of month t−1
Costs: reported gross and with a 5 bp/month haircut on the factor (≈ turnover × 2 bp).
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.carry_map.data import rates, spot_usd, CCY  # noqa: E402

OUT = ROOT / "data" / "research" / "carry_map"
START, END = "2006-06-01", "2026-04-30"


def sharpe(x): s = x.std(ddof=1); return float(x.mean() / s * np.sqrt(12)) if s > 0 else 0.0
def max_dd(x): eq = np.cumprod(1 + x); return float((eq / np.maximum.accumulate(eq) - 1).min())
def stats(x: pd.Series) -> dict:
    x = x.dropna()
    return {"n": int(len(x)), "mean_m": float(x.mean()), "ann": float(x.mean() * 12), "sharpe": sharpe(x.values),
            "hit": float((x > 0).mean()), "worst": float(x.min()), "max_dd": max_dd(x.values)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    R = rates(); S = spot_usd()
    me = S.resample("ME").last()                                  # month-end levels
    vix = me["VIX"]; lv = np.log(me[CCY])
    dlog = lv.diff()                                              # month t spot log change (USD value of c)
    rt = R.copy(); rt.index = rt.index.to_period("M")
    rate_prev = rt.shift(1)                                       # rate published for t-1, aligned to month t
    rate_prev.index = rate_prev.index.to_timestamp("M")
    rate_prev = rate_prev.reindex(dlog.index).ffill()
    diff = rate_prev.sub(rate_prev["USD"], axis=0) / 1200.0
    X = (dlog + diff).loc[START:END]                              # excess return vs USD, monthly
    RP = rate_prev.loc[START:END]
    months = X.index; n = len(months)
    print(f"panel {months[0].date()} → {months[-1].date()}  {n} months  {len(CCY)} currencies\n")

    # ── CARRY factor ────────────────────────────────────────────────────────
    ranks = RP.rank(axis=1, ascending=False)
    long = (ranks <= 3).astype(float); short = (ranks >= 8).astype(float)
    factor = (X * long).sum(axis=1) / 3 - (X * short).sum(axis=1) / 3
    turnover = (long.diff().abs().sum(axis=1) + short.diff().abs().sum(axis=1)) / 6
    factor_net = factor - 0.0005
    print("CARRY FACTOR (long top-3 / short bottom-3 by rate, monthly):")
    for k, v in (("gross", factor), ("net 5bp/m", factor_net)):
        st = stats(v); print(f"  {k:10s} ann {st['ann']*100:+.2f}%  Sharpe {st['sharpe']:.2f}  hit {st['hit']:.2f}  worst {st['worst']*100:+.1f}%  maxDD {st['max_dd']*100:.1f}%")
    print(f"  mean turnover {turnover.mean():.2f} legs/month")
    yearly = factor.groupby(factor.index.year).sum()
    print("  by year (%):", {int(y): round(v * 100, 1) for y, v in yearly.items()})

    # ── per currency: how often on each side, and what it earned ────────────
    print("\nPER CURRENCY (20y): months long / short, mean excess when held, contribution to factor")
    cur = {}
    for c in CCY:
        L, Sh = long[c] > 0, short[c] > 0
        cur[c] = {"months_long": int(L.sum()), "months_short": int(Sh.sum()), "avg_rate": float(RP[c].mean()),
                  "excess_when_long_ann": float(X[c][L].mean() * 12) if L.any() else None,
                  "excess_when_short_ann": float(-X[c][Sh].mean() * 12) if Sh.any() else None,
                  "contribution_total": float((X[c] * long[c] / 3).sum() - (X[c] * short[c] / 3).sum())}
        print(f"  {c}: rate {cur[c]['avg_rate']:.2f}%  long {cur[c]['months_long']:3d}m ({(cur[c]['excess_when_long_ann'] or 0)*100:+.1f}%/yr)  "
              f"short {cur[c]['months_short']:3d}m ({(cur[c]['excess_when_short_ann'] or 0)*100:+.1f}%/yr)  contrib {cur[c]['contribution_total']*100:+.1f}%")

    # ── all 45 pairs: long the higher-rate leg each month ───────────────────
    pairs = {}
    for a, b in combinations(CCY, 2):
        sgn = np.sign(RP[a] - RP[b]); r = sgn * (X[a] - X[b])           # long a if a pays more, else long b
        st = stats(r); st["share_a_high"] = float((sgn > 0).mean()); st["avg_diff"] = float((RP[a] - RP[b]).abs().mean())
        pairs[f"{a}/{b}"] = st
    pt = pd.DataFrame(pairs).T.sort_values("sharpe", ascending=False)
    print("\nPAIR CARRY (long higher-rate leg monthly), top 12 by Sharpe and bottom 5:")
    print((pt[["ann", "sharpe", "hit", "max_dd", "avg_diff"]].head(12).assign(ann=lambda d: d.ann * 100, max_dd=lambda d: d.max_dd * 100)).round(2).to_string())
    print((pt[["ann", "sharpe", "hit", "max_dd", "avg_diff"]].tail(5).assign(ann=lambda d: d.ann * 100, max_dd=lambda d: d.max_dd * 100)).round(2).to_string())

    # ── states (known at end of t-1) ─────────────────────────────────────────
    disp = (RP.max(axis=1) - RP.min(axis=1))                       # rate dispersion
    top3 = RP.apply(lambda r: r.nlargest(3).mean(), axis=1); bot3 = RP.apply(lambda r: r.nsmallest(3).mean(), axis=1)
    spread = top3 - bot3
    vix_prev = vix.reindex(months).shift(1)
    f12 = factor.rolling(12).sum().shift(1)
    eq = (1 + factor).cumprod(); dd_prev = (eq / eq.cummax() - 1).shift(1)
    usd_mom = (-dlog[[c for c in CCY if c != "USD"]].mean(axis=1)).rolling(12).sum().shift(1).reindex(months)
    fed6 = (RP["USD"] - RP["USD"].shift(6))
    spread_chg = spread - spread.shift(12)
    def terc(s, labels=("low", "mid", "high")): return pd.qcut(s.rank(method="first"), 3, labels=labels)
    states = {
        "S1 rate spread top3-bot3 (tercile)": terc(spread),
        "S2 spread 12m change": pd.cut(spread_chg, [-99, -0.25, 0.25, 99], labels=["narrowing", "flat", "widening"]),
        "S3 VIX prior month-end": pd.cut(vix_prev, [0, 15, 25, 999], labels=["<15", "15-25", ">25"]),
        "S4 factor trailing-12m": pd.cut(f12, [-9, -0.02, 0.02, 9], labels=["neg", "flat", "pos"]),
        "S5 factor drawdown at t-1": pd.cut(dd_prev, [-1, -0.10, -0.001, 1], labels=[">10% DD", "0-10% DD", "at peak"]),
        "S6 USD 12m momentum": pd.cut(usd_mom, [-9, -0.03, 0.03, 9], labels=["USD down", "flat", "USD up"]),
        "S7 Fed 6m change": pd.cut(fed6, [-99, -0.25, 0.25, 99], labels=["cutting", "hold", "hiking"]),
    }
    print("\nSTATES → next-month CARRY factor (gross). n / ann% / Sharpe / hit / worst%")
    state_tab = {}
    for name, s in states.items():
        rows = {}
        for lvl in s.cat.categories:
            m = (s == lvl).values; x = factor[m]
            if len(x) >= 12:
                st = stats(x); rows[str(lvl)] = st
        state_tab[name] = rows
        print(f"  {name}")
        for lvl, st in rows.items():
            print(f"     {lvl:12s} n={st['n']:3d}  {st['ann']*100:+6.1f}%  Sh {st['sharpe']:+.2f}  hit {st['hit']:.2f}  worst {st['worst']*100:+.1f}%")

    # ── sequences: event studies on the factor ──────────────────────────────
    def event_path(idx_list, pre=6, post=24):
        paths = []
        for i in idx_list:
            if i - pre < 0 or i + post >= n: continue
            seg = factor.values[i - pre: i + post + 1]; paths.append(np.cumsum(seg) - np.cumsum(seg)[pre])
        return np.array(paths)
    worst5 = list(np.argsort(factor.values)[:5])
    vix_cross = [i for i in range(1, n) if vix_prev.values[i] > 30 and vix_prev.values[i - 1] <= 30]
    hikes = [i for i in range(6, n) if fed6.values[i] > 0.25 and fed6.values[i - 1] <= 0.25]
    cuts = [i for i in range(6, n) if fed6.values[i] < -0.25 and fed6.values[i - 1] >= -0.25]
    seq = {}
    print("\nSEQUENCES — cumulative factor return after the event (months +3 / +6 / +12 / +24), mean over events")
    for name, ev in (("5 worst carry months", worst5), ("VIX crosses 30", vix_cross), ("Fed first hike (6m Δ>+25bp)", hikes), ("Fed first cut (6m Δ<−25bp)", cuts)):
        P = event_path(ev)
        if len(P) == 0: continue
        m = P.mean(axis=0); seq[name] = {"n": len(P), "path": m.tolist(), "dates": [str(months[i].date()) for i in ev]}
        print(f"  {name:32s} n={len(P)}  +3m {m[6+3]*100:+.1f}%  +6m {m[6+6]*100:+.1f}%  +12m {m[6+12]*100:+.1f}%  +24m {m[6+24]*100:+.1f}%   (pre-6m {m[0]*100:+.1f}%)")
        print(f"      events: {', '.join(str(months[i].date())[:7] for i in ev)}")

    # ── holding structure: rebalance frequency ───────────────────────────────
    print("\nHOLDING STRUCTURE — same ranks, re-ranked every k months (gross)")
    hold = {}
    for k in (1, 3, 6, 12):
        Lk = long.copy(); Sk = short.copy()
        for i in range(n):
            if i % k: Lk.iloc[i] = Lk.iloc[i - i % k]; Sk.iloc[i] = Sk.iloc[i - i % k]
        fk = (X * Lk).sum(axis=1) / 3 - (X * Sk).sum(axis=1) / 3
        hold[k] = stats(fk); print(f"  every {k:2d}m: ann {hold[k]['ann']*100:+.2f}%  Sharpe {hold[k]['sharpe']:.2f}  maxDD {hold[k]['max_dd']*100:.1f}%")

    # ── long-only variants a retail account can actually hold ───────────────
    print("\nLONG-ONLY (no shorting; USD cash is the alternative): top-3 rate currencies vs USD, equal weight")
    lo = (X * long).sum(axis=1) / 3
    st = stats(lo); print(f"  ann {st['ann']*100:+.2f}%  Sharpe {st['sharpe']:.2f}  hit {st['hit']:.2f}  maxDD {st['max_dd']*100:.1f}%")
    print("  by year (%):", {int(y): round(v * 100, 1) for y, v in lo.groupby(lo.index.year).sum().items()})

    res = {"window": [str(months[0].date()), str(months[-1].date())], "n_months": n,
           "factor": {"gross": stats(factor), "net": stats(factor_net), "by_year": {int(y): float(v) for y, v in yearly.items()}},
           "currency": cur, "pairs": pairs, "states": state_tab, "sequences": seq, "holding": hold, "long_only": {"stats": st},
           "cells_compared": sum(len(v) for v in state_tab.values()) + len(pairs) + len(seq) + len(hold) + 2}
    (OUT / "map.json").write_text(json.dumps(res, indent=2, default=float))
    pd.DataFrame({"factor": factor, "factor_net": factor_net, "long_only": lo, "spread": spread, "vix_prev": vix_prev, "dd_prev": dd_prev}).to_parquet(OUT / "factor_monthly.parquet")
    print(f"\ncells compared in this map: {res['cells_compared']}  (multiplicity to declare on whatever gets sealed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
