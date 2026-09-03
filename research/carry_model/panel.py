"""Pair-month panel: 45 signed pairs (long a / short b convention), features at t-1, target at t.
Every feature uses data ≤ t-1 only. Cross-sectional z-scores are computed within month."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from research.carry_model.data_fred import month_end_spot, month_rates, month_cpi, guard, CCY

HAIRCUT = 0.30          # retail swap haircut on the interest differential (TICK-024 spirit; frozen)
COST_M = 0.0003 * 2     # 3 bp per leg per month, two legs


def build(unlock_holdout: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    S = month_end_spot(); R = month_rates(); C = month_cpi()
    idx = S.index.intersection(R.index)
    S, R, C = S.reindex(idx), R.reindex(idx), C.reindex(idx)
    lv = np.log(S); dlog = lv.diff()
    r_prev = R.shift(1)                                       # rate known at t-1
    # currency-level pieces
    exc_usd = dlog + (r_prev.sub(r_prev["USD"], axis=0)) / 1200.0          # gross excess vs USD
    # common (time) state variables, all at t-1
    ranks = r_prev.rank(axis=1, ascending=False)
    top = (ranks <= 3); bot = (ranks >= 8)
    factor = (exc_usd.where(top).mean(axis=1) - exc_usd.where(bot).mean(axis=1))
    eqf = (1 + factor.fillna(0)).cumprod(); factor_dd = (eqf / eqf.cummax() - 1).shift(1)
    spread = r_prev.apply(lambda x: x.nlargest(3).mean() - x.nsmallest(3).mean(), axis=1)
    spread_chg12 = spread - spread.shift(12)
    fed6 = r_prev["USD"] - r_prev["USD"].shift(6)
    usd_idx = (-dlog[[c for c in CCY if c != "USD"]].mean(axis=1))          # USD index monthly return
    rows = []
    for a, b in combinations(CCY, 2):
        spot_ab = lv[a] - lv[b]                                             # log(a/b) in USD terms
        carry = (R[a] - R[b]).shift(1)                                      # % p.a., at t-1
        # realized excess of long a / short b in month t, retail-adjusted
        gross = (dlog[a] - dlog[b]) + (carry / 1200.0)
        target = (dlog[a] - dlog[b]) + (carry / 1200.0) * (1 - HAIRCUT) - COST_M
        rp = spot_ab.diff()                                                 # pair spot return
        rvol3 = rp.rolling(3).std().shift(1) * np.sqrt(12)
        real_ab = spot_ab + np.log(C[b]) - np.log(C[a])                     # real exchange rate (a per b, CPI-adjusted)
        value = -(real_ab - real_ab.shift(60)).shift(1)                     # 5y real appreciation → negative value
        beta = rp.rolling(36).cov(usd_idx).shift(1) / usd_idx.rolling(36).var().shift(1)
        df = pd.DataFrame({
            "pair": f"{a}/{b}", "a": a, "b": b, "target": target, "gross": gross,
            "carry": carry, "carry_chg12": carry - carry.shift(12),
            "carry_d1": carry - carry.shift(1), "carry_d2": (carry - carry.shift(1)) - (carry.shift(1) - carry.shift(2)),
            "mom1": rp.shift(1), "mom3": rp.rolling(3).sum().shift(1), "mom12": rp.rolling(12).sum().shift(1),
            "mom_d1": rp.rolling(3).sum().shift(1) - rp.rolling(3).sum().shift(2),
            "value": value, "rvol3": rvol3, "rvol12": rp.rolling(12).std().shift(1) * np.sqrt(12), "dollar_beta": beta,
            "factor_dd": factor_dd, "spread_chg12": spread_chg12, "fed6": fed6,
        }, index=idx)
        rows.append(df)
    P = pd.concat(rows).reset_index().rename(columns={"index": "date"})
    P = P.dropna(subset=["target", "carry", "mom12", "rvol3"])
    if not unlock_holdout:
        P = P[P["date"] > pd.Timestamp("2005-12-31")]
    return P, guard(factor.to_frame("factor")) if not unlock_holdout else factor.to_frame("factor"), R


FEATS_BASE = ["carry", "mom12", "value"]
FEATS_ALL = ["carry", "carry_chg12", "mom1", "mom3", "mom12", "value", "rvol3", "dollar_beta"]
FEATS_INTER = FEATS_ALL + ["carry_x_dd", "carry_x_spreadchg", "carry_x_fed6"]
FEATS_CALC = FEATS_ALL + ["carry_d1", "carry_d2", "mom_d1"]


def zscore_cs(P: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    P = P.copy()
    for c in cols:
        g = P.groupby("date")[c]
        P[c + "_z"] = (P[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    P["carry_x_dd_z"] = P["carry_z"] * P["factor_dd"].fillna(0)
    P["carry_x_spreadchg_z"] = P["carry_z"] * P["spread_chg12"].fillna(0)
    P["carry_x_fed6_z"] = P["carry_z"] * P["fed6"].fillna(0)
    return P
