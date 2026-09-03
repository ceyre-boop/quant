"""Evaluation of a signed score on the pair-month panel: rank IC, top-k portfolios, bootstrap, permutation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.hyp111.date_bootstrap import stationary_block_indices


def sharpe_m(x): s = np.std(x, ddof=1); return float(np.mean(x) / s * np.sqrt(12)) if s > 0 else 0.0
def max_dd(x): eq = np.cumprod(1 + np.asarray(x)); return float((eq / np.maximum.accumulate(eq) - 1).min())


def monthly_ic(P: pd.DataFrame, score: pd.Series) -> pd.Series:
    df = P[["date", "target"]].assign(s=score.values).dropna()
    return df.groupby("date").apply(lambda g: spearmanr(g["s"], g["target"]).correlation if len(g) > 5 else np.nan).dropna()


def topk_returns(P: pd.DataFrame, score: pd.Series, k: int) -> pd.Series:
    """Signed positions: take the k largest |score|, position sign = sign(score), equal notional."""
    df = P[["date", "target"]].assign(s=score.values).dropna()
    def one(g):
        g = g.reindex(g["s"].abs().sort_values(ascending=False).index[:k])
        return float((np.sign(g["s"]) * g["target"]).mean())
    return df.groupby("date").apply(one)


def boot_ci(x: np.ndarray, stat, L=6, draws=5000, seed=42):
    rng = np.random.default_rng(seed); n = len(x)
    b = np.array([stat(x[stationary_block_indices(rng, n, L)]) for _ in range(draws)])
    lo, hi = np.percentile(b, [2.5, 97.5]); return float(lo), float(hi)


def perm_null_topk(P: pd.DataFrame, score: pd.Series, k: int, draws=2000, seed=42) -> float:
    """Shuffle realized returns ACROSS pairs within each month (kills skill, keeps time structure)."""
    rng = np.random.default_rng(seed)
    df = P[["date", "target"]].assign(s=score.values).dropna()
    obs = topk_returns(P, score, k).mean()
    groups = [g for _, g in df.groupby("date")]
    sel = [(np.sign(g["s"].values), g["s"].abs().values.argsort()[::-1][:k], g["target"].values) for g in groups]
    null = np.empty(draws)
    for d in range(draws):
        tot = 0.0
        for sg, ix, t in sel:
            tp = rng.permutation(t); tot += float((sg[ix] * tp[ix]).mean())
        null[d] = tot / len(sel)
    return float((null >= obs).mean())


def summarize(name: str, P: pd.DataFrame, score: pd.Series, factor: pd.Series, k: int = 5) -> dict:
    ic = monthly_ic(P, score); r5 = topk_returns(P, score, k); r1 = topk_returns(P, score, 1)
    fac = factor.reindex(r5.index).fillna(0)
    ic_lo, ic_hi = boot_ci(ic.values, np.mean)
    d = (r5 - fac).values; dlo, dhi = boot_ci(np.vstack([r5.values, fac.values]).T, lambda m: sharpe_m(m[:, 0]) - sharpe_m(m[:, 1])) if False else (None, None)
    # joint resample for the Sharpe difference
    rng = np.random.default_rng(42); n = len(r5); diffs = []
    for _ in range(5000):
        ix = stationary_block_indices(rng, n, 6); diffs.append(sharpe_m(r5.values[ix]) - sharpe_m(fac.values[ix]))
    dlo, dhi = np.percentile(diffs, [2.5, 97.5])
    years = ic.index.year; ic_by_year = ic.groupby(years).mean()
    out = {"name": name, "months": int(len(ic)), "window": [str(ic.index[0].date()), str(ic.index[-1].date())],
           "ic_mean": float(ic.mean()), "ic_ci": [ic_lo, ic_hi], "ic_years_pos": f"{int((ic_by_year > 0).sum())}/{len(ic_by_year)}",
           "top5": {"ann": float(r5.mean() * 12), "sharpe": sharpe_m(r5.values), "max_dd": max_dd(r5.values), "hit": float((r5 > 0).mean())},
           "top1": {"ann": float(r1.mean() * 12), "sharpe": sharpe_m(r1.values), "max_dd": max_dd(r1.values)},
           "carry_factor_same_window": {"ann": float(fac.mean() * 12), "sharpe": sharpe_m(fac.values)},
           "sharpe_diff_top5_vs_factor": [float(dlo), float(dhi)],
           "ic_by_year": {int(y): round(float(v), 3) for y, v in ic_by_year.items()}}
    return out
