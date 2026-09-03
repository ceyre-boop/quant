#!/usr/bin/env python3
"""LOOKBACK CEILING — what the 20 years could have paid with hindsight. Three tiers, all in-sample:
  T1  rules from the map, applied as if known (they were found on this data → lookback bias, ~76 cells)
  T2  best static choices in hindsight (best pair / best currency held 20 years)
  T3  oracle: perfect foresight each month (the absolute ceiling; not a strategy, a bound)
Not a test. The point is the size of the gap between what the map's rules could earn and the oracle."""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.carry_map.data import rates, spot_usd, CCY  # noqa: E402
from research.carry_map.map import sharpe, max_dd, START, END  # noqa: E402


def panel():
    R = rates(); S = spot_usd(); me = S.resample("ME").last(); lv = np.log(me[CCY]); dlog = lv.diff()
    rt = R.copy(); rt.index = rt.index.to_period("M"); rp = rt.shift(1); rp.index = rp.index.to_timestamp("M")
    rp = rp.reindex(dlog.index).ffill(); diff = rp.sub(rp["USD"], axis=0) / 1200.0
    return (dlog + diff).loc[START:END], rp.loc[START:END], me["VIX"].reindex(dlog.index).loc[START:END], dlog.loc[START:END]


def line(name, r, cost=0.0):
    r = r - cost
    eq = float(np.prod(1 + r)); yrs = len(r) / 12
    print(f"  {name:58s} ann {r.mean()*12*100:+6.2f}%  CAGR {(eq**(1/yrs)-1)*100:+6.2f}%  Sharpe {sharpe(r.values):5.2f}  maxDD {max_dd(r.values)*100:6.1f}%  hit {(r>0).mean():.2f}")
    return {"ann": float(r.mean() * 12), "cagr": float(eq ** (1 / yrs) - 1), "sharpe": sharpe(r.values), "max_dd": max_dd(r.values)}


def main():
    X, RP, vix, dlog = panel(); n = len(X); months = X.index
    ranks = RP.rank(axis=1, ascending=False); long = (ranks <= 3).astype(float); short = (ranks >= 8).astype(float)
    factor = (X * long).sum(axis=1) / 3 - (X * short).sum(axis=1) / 3
    eq = (1 + factor).cumprod(); dd_prev = (eq / eq.cummax() - 1).shift(1).fillna(0)
    f12 = factor.rolling(12).sum().shift(1).fillna(0)
    top3 = RP.apply(lambda r: r.nlargest(3).mean(), axis=1); bot3 = RP.apply(lambda r: r.nsmallest(3).mean(), axis=1)
    spread = top3 - bot3; spread_chg = (spread - spread.shift(12)).fillna(0)
    fed6 = (RP["USD"] - RP["USD"].shift(6)).fillna(0)
    usd_mom = (-dlog[[c for c in CCY if c != "USD"]].mean(axis=1)).rolling(12).sum().shift(1).fillna(0)
    vix_prev = vix.shift(1).bfill()

    print(f"\n20-YEAR LOOKBACK CEILING  {months[0].date()} → {months[-1].date()}  ({n} months)\n")
    print("T0  baseline, no timing")
    base = line("carry factor, always on", factor)

    print("\nT1  map rules applied with hindsight (IN-SAMPLE — these rules were found on this data)")
    rules = {
        "on only when factor >10% off peak": (dd_prev < -0.10),
        "on when >10% off peak OR at peak (skip 0-10%)": (dd_prev < -0.10) | (dd_prev > -0.001),
        "on only when trailing-12m factor < 0": (f12 < -0.02),
        "on only when rate spread narrowing (12m)": (spread_chg < -0.25),
        "off when Fed cutting (6m Δ < −25bp)": ~(fed6 < -0.25),
        "off when USD trending (|12m| > 3%)": (usd_mom.abs() <= 0.03),
        "off when rate dispersion in top tercile": (spread <= spread.quantile(2 / 3)),
        "COMBO: off-peak>10% OR narrowing, AND Fed not cutting": ((dd_prev < -0.10) | (spread_chg < -0.25)) & ~(fed6 < -0.25),
        "COMBO all five (DD>10 | narrowing) & !cut & !USDtrend & !topdisp": ((dd_prev < -0.10) | (spread_chg < -0.25)) & ~(fed6 < -0.25) & (usd_mom.abs() <= 0.03) & (spread <= spread.quantile(2 / 3)),
    }
    t1 = {}
    for name, m in rules.items():
        r = factor * m.values.astype(float); t1[name] = line(f"{name}  [{int(m.sum())}m on]", r)
    # size-up version: 2x leverage when the best state is on (retail margin allows), 0 otherwise
    best = ((dd_prev < -0.10) | (spread_chg < -0.25)) & ~(fed6 < -0.25)
    line("COMBO at 2x notional when on (margin), 0 when off", 2 * factor * best.values.astype(float))

    print("\nT2  best static choices in hindsight (held the full 20 years, monthly re-rank of the leg)")
    pairs = {}
    for a, b in combinations(CCY, 2):
        sgn = np.sign(RP[a] - RP[b]); pairs[f"{a}/{b}"] = sgn * (X[a] - X[b])
    P = pd.DataFrame(pairs); sh = P.apply(lambda c: sharpe(c.values)); ann = P.mean() * 12
    line(f"best pair by Sharpe: {sh.idxmax()}", P[sh.idxmax()]); line(f"best pair by return: {ann.idxmax()}", P[ann.idxmax()])
    top5 = sh.nlargest(5).index.tolist(); line(f"best 5 pairs EW (hindsight): {','.join(top5)}", P[top5].mean(axis=1))
    cur_long = X.mean() * 12; cur_short = -cur_long
    line(f"best single currency long vs USD: {cur_long.idxmax()}", X[cur_long.idxmax()]); line(f"best single currency short vs USD: {cur_short.idxmax()}", -X[cur_short.idxmax()])
    line("hindsight L/S: long AUD+NZD, short JPY+EUR+SEK, always on", (X["AUD"] + X["NZD"]) / 2 - (X["JPY"] + X["EUR"] + X["SEK"]) / 3)
    line("hindsight L/S above, ON only in COMBO state", ((X["AUD"] + X["NZD"]) / 2 - (X["JPY"] + X["EUR"] + X["SEK"]) / 3) * best.values)

    print("\nT3  oracle — perfect foresight each month (a bound, not a strategy)")
    line("best single pair each month, unit notional", P.abs().max(axis=1))
    line("factor with perfect sign each month (long or short the factor)", factor.abs())
    line("best currency vs USD each month, long or short", X[[c for c in CCY if c != "USD"]].abs().max(axis=1))
    print("\nread: T3 is the wall; T1/T2 are what hindsight on the MAP's own states buys; the honest expected forward value of T1 is below T1 (76 cells searched) and above T0 only if the drawdown/narrowing state is real out of sample (HYP-117).")


if __name__ == "__main__":
    main()
