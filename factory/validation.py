"""D2 — walk-forward validation with purging + embargo (reuses discovery/cpcv).

Reports: IC (Spearman, prediction vs realized), hit rate, PnL after costs (v015 spread
convention), max drawdown, correlation-to-carry (v015 proof curve), and the
abstention-coverage curve (metric quality vs fraction of decisions taken — Article 4's
empirical face). Pure evaluation; owns no data and trains nothing itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign.discovery.cpcv import combinatorial_purged_splits

ROOT = Path(__file__).resolve().parents[1]
SPREAD_COST_R = 0.02   # per round trip, in R — v015 cost convention (documented in HYP-062)


def spearman_ic(pred: np.ndarray, real: np.ndarray) -> float | None:
    from scipy.stats import spearmanr
    if len(pred) < 3:
        return None
    rho = spearmanr(pred, real).statistic
    return None if not np.isfinite(rho) else float(rho)


def max_drawdown(rets: np.ndarray) -> float:
    eq = np.cumprod(1 + rets)
    return float((eq / np.maximum.accumulate(eq) - 1).min()) if len(eq) else 0.0


def corr_to_carry(daily_rets: pd.Series) -> float | None:
    eq_json = ROOT / "data" / "proof" / "backtest_equity_v015_2015_2024.json"
    if not eq_json.exists():
        return None
    pts = json.loads(eq_json.read_text())["points"]
    carry = pd.Series({pd.Timestamp(p["t"]): p["ret"] for p in pts}).groupby(level=0).sum()
    joined = pd.concat([daily_rets, carry], axis=1, keys=["m", "c"]).dropna()
    if len(joined) < 30:
        return None
    return float(joined["m"].corr(joined["c"]))


def abstention_coverage(proba: np.ndarray, y: np.ndarray,
                        thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75)) -> list[dict]:
    out = []
    conf = np.maximum(proba, 1 - proba)
    pred = (proba > 0.5).astype(int)
    for t in thresholds:
        take = conf >= t
        n = int(take.sum())
        out.append({"threshold": t, "coverage": float(take.mean()),
                    "hit": float((pred[take] == y[take]).mean()) if n else None, "n": n})
    return out


def walk_forward_report(model_factory, X: pd.DataFrame, y: pd.Series,
                        entry_dt: pd.Series, exit_dt: pd.Series,
                        realized_r: pd.Series, n_groups: int = 6,
                        embargo_frac: float = 0.02) -> dict:
    """CPCV evaluation: model_factory() -> fresh estimator with fit/predict_proba."""
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in combinatorial_purged_splits(entry_dt, exit_dt,
                                                           n_groups=n_groups, test_groups=1,
                                                           embargo_frac=embargo_frac):
        if not len(train_idx) or not len(test_idx):
            continue
        m = model_factory()
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds[test_idx] = m.predict_proba(X.iloc[test_idx])[:, 1]
    scored = np.isfinite(preds)
    p, yy, rr = preds[scored], y.values[scored], realized_r.values[scored]
    take = p > 0.5
    net = np.where(take, rr - SPREAD_COST_R, 0.0)
    daily = pd.Series(net, index=pd.to_datetime(entry_dt.values[scored])).groupby(level=0).sum()
    return {"n_scored": int(scored.sum()),
            "ic": spearman_ic(p, rr),
            "hit": float(((p > 0.5) == (yy == 1)).mean()) if scored.any() else None,
            "pnl_after_costs_R": float(net.sum()),
            "max_dd": max_drawdown(daily.values * 0.01),
            "corr_to_carry": corr_to_carry(daily * 0.01),
            "abstention_coverage": abstention_coverage(p, yy),
            "embargo_frac": embargo_frac, "n_groups": n_groups,
            "cost_convention": f"{SPREAD_COST_R}R/round-trip"}
