"""Model zoo for the pair-month panel. Every fitted model is trained on months < t only (expanding
window, refit every 12 months, min 120 training months). Scores are signed: positive = long a/short b."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from research.carry_model.panel import FEATS_BASE, FEATS_ALL, FEATS_INTER, FEATS_CALC

MIN_TRAIN, REFIT = 120, 12


def _z(P, feats): return [f + "_z" for f in feats]


def score_baseline(P: pd.DataFrame) -> pd.Series:
    return P[_z(P, FEATS_BASE)].fillna(0).sum(axis=1)


def _walk_forward(P: pd.DataFrame, feats: list[str], make_model, transform=None) -> pd.Series:
    cols = _z(P, feats); X = P[cols].fillna(0).values; y = P["target"].values
    dates = np.sort(P["date"].unique()); out = pd.Series(np.nan, index=P.index)
    model = None
    for i, d in enumerate(dates):
        if i < MIN_TRAIN: continue
        if model is None or (i - MIN_TRAIN) % REFIT == 0:
            tr = (P["date"] < d).values
            model = make_model().fit(X[tr], y[tr])
        te = (P["date"] == d).values
        out[te] = model.predict(X[te])
    return out


def score_ridge(P): return _walk_forward(P, FEATS_ALL, lambda: Ridge(alpha=10.0))
def score_ridge_inter(P): return _walk_forward(P, FEATS_INTER, lambda: Ridge(alpha=10.0))


def score_gbm(P):
    mono = [1 if f in ("carry", "mom12", "value") else (-1 if f == "rvol3" else 0) for f in FEATS_ALL]
    return _walk_forward(P, FEATS_ALL, lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                                                              monotonic_cst=mono, random_state=42))


def score_shot(P):
    """Colin's shot in the dark — statistics × calculus. Statistics: ridge on the characteristics AND their
    first/second differences (carry velocity/acceleration, momentum change) gives a signed conviction.
    Calculus: the oracle's payoff is the extreme of the cross-section, so conviction is multiplied by the
    forecast magnitude (trailing 3m realized vol — the one thing HYP-109 proved persistent). Pick where a
    confident sign meets the largest expected move."""
    stat = _walk_forward(P, FEATS_CALC, lambda: Ridge(alpha=10.0))
    return stat * P["rvol3"].fillna(P["rvol3"].median())


def score_shot_pure(P):
    """No fitting: sign from the baseline composite, magnitude from realized vol."""
    return np.sign(score_baseline(P)) * P["rvol3"].fillna(P["rvol3"].median())


MODELS = {"M1 baseline z(carry)+z(mom12)+z(value)": score_baseline,
          "M2 ridge, 8 features": score_ridge,
          "M3 ridge + carry×state interactions (HYP-117 form)": score_ridge_inter,
          "M4 boosting, monotone, depth 3": score_gbm,
          "M5 SHOT: stat(ridge+derivatives) × magnitude(rvol)": score_shot,
          "M5b SHOT-pure: sign(baseline) × rvol": score_shot_pure}
