"""M6 — the strongest construction the ledger supports, frozen before the EM universe is read.
Keeps ONLY what survived HYP-117 (the parameter-free characteristic ranking) and adds explicit crash
management, which is mechanical, not predictive:
  score      = z(carry) + z(mom12) + z(value) − 0.5·z(rvol3)      cross-sectional z within month
  selection  = top-k by |score|, sign = sign(score)
  weights    = risk parity: w_k ∝ 1/rvol3_k, normalised to gross 1
  exposure   = min(1.5, 0.10 / σ̂_12m of the strategy's own trailing returns)   [vol target 10%, cap 1.5×]
               × 0.5 if VIX(t−1) > 30                                             [crash brake]
  M6b        = M6 × 1.5 when the universe's own carry factor is >10% below its trailing peak (map rule)
Nothing is fitted. There is no training set."""
from __future__ import annotations

import numpy as np
import pandas as pd


def zcs(P, col):
    g = P.groupby("date")[col]; return (P[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)


def score(P: pd.DataFrame) -> pd.Series:
    return (zcs(P, "carry").fillna(0) + zcs(P, "mom12").fillna(0) + zcs(P, "value").fillna(0) - 0.5 * zcs(P, "rvol3").fillna(0))


def carry_factor(P: pd.DataFrame, k: int) -> pd.Series:
    """Plain carry on the same universe: top-k positions by carry (signed), equal weight."""
    def one(g):
        g = g.assign(s=g["carry"]); g = g.reindex(g["s"].abs().sort_values(ascending=False).index[:k])
        return float((np.sign(g["s"]) * g["target"]).mean())
    return P.groupby("date").apply(one)


def run(P: pd.DataFrame, k: int, dd_boost: bool = False, vol_target: float = 0.10, cap: float = 1.5) -> dict:
    s = score(P); df = P[["date", "target", "rvol3", "vix_prev"]].assign(s=s.values)
    raw = []
    for d, g in df.groupby("date"):
        g = g.reindex(g["s"].abs().sort_values(ascending=False).index[:k])
        w = (1 / g["rvol3"].clip(lower=0.02)); w = w / w.sum()
        raw.append((d, float((w * np.sign(g["s"]) * g["target"]).sum()), float(g["vix_prev"].iloc[0]) if not np.isnan(g["vix_prev"].iloc[0]) else 20.0))
    r = pd.Series({d: v for d, v, _ in raw}); vix = pd.Series({d: x for d, _, x in raw})
    sig = r.rolling(12).std().shift(1) * np.sqrt(12)
    expo = (vol_target / sig).clip(upper=cap).fillna(1.0) * np.where(vix > 30, 0.5, 1.0)
    if dd_boost:
        fac = carry_factor(P, k); eq = (1 + fac).cumprod(); dd = (eq / eq.cummax() - 1).shift(1).reindex(r.index).fillna(0)
        expo = expo * np.where(dd < -0.10, 1.5, 1.0)
    managed = r * expo
    return {"raw": r, "managed": managed, "exposure": expo, "score": s}
