"""VRP calculator — PURE functions (no I/O, no network).

Two volatility-premium constructs, deliberately separated by CAUSALITY:

  * btz_vrp_gap  — implied_var(t) - forward_realized_var(t, t+h).  The standard
    Bollerslev-Tauchen-Zhou (2009) premium. Uses FORWARD realized variance, so it has
    look-ahead. It is used ONLY in the Stage-1 existence statistic and must never feed
    the Stage-2 correlation/Sharpe series.

  * harvest_return_causal — (IV_{t-1}/100)^2/252 - r_t^2.  The daily P&L of selling one
    unit of variance at YESTERDAY's implied level and paying TODAY's realized squared
    return. Strictly causal (IV known at prior close, r realized intraday). This is the
    ONLY series allowed into the Stage-2 orthogonality kill-gate.

Scale of the harvest return is arbitrary (variance units) but Sharpe and correlation —
the only things Stage 2 uses it for — are scale-invariant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).dropna()


def realized_variance(close: pd.Series, window: int = 21, forward: bool = False) -> pd.Series:
    """Annualized realized variance over a rolling window of daily log returns.

    forward=False: variance over the trailing [t-window+1, t] (causal, observable).
    forward=True : variance over the NEXT [t+1, t+window] (look-ahead — Stage 1 only).
    """
    r2 = log_returns(close) ** 2
    if forward:
        rv = r2.rolling(window).sum().shift(-window) * (TRADING_DAYS / window)
    else:
        rv = r2.rolling(window).sum() * (TRADING_DAYS / window)
    return rv.dropna().rename("realized_var")


def implied_variance(vol_index: pd.Series) -> pd.Series:
    """Annualized implied variance from a vol index in POINTS (VIX=20 -> 0.04)."""
    return ((vol_index / 100.0) ** 2).rename("implied_var")


def btz_vrp_gap(vol_index: pd.Series, close: pd.Series, window: int = 21) -> pd.Series:
    """LOOK-AHEAD (Stage 1 only): implied_var(t) - forward_realized_var(t, t+window).

    Positive mean => implied vol systematically exceeded subsequently realized vol => the
    volatility risk premium exists. The forward term is why this is quarantined to Stage 1.
    """
    iv = implied_variance(vol_index)
    fwd_rv = realized_variance(close, window=window, forward=True)
    idx = iv.index.intersection(fwd_rv.index)
    return (iv.loc[idx] - fwd_rv.loc[idx]).rename("btz_vrp_gap")


def harvest_return_causal(vol_index: pd.Series, close: pd.Series) -> pd.Series:
    """STRICTLY CAUSAL daily short-variance P&L: (IV_{t-1}/100)^2/252 - r_t^2.

    Information set at the open of day t: yesterday's closing implied variance (the premium
    you sold) minus today's realized squared return (what you pay). No data from > t is used.
    """
    r = log_returns(close)
    iv_daily = ((vol_index.shift(1) / 100.0) ** 2) / TRADING_DAYS   # premium sold at prior close
    iv_daily = iv_daily.dropna()
    idx = r.index.intersection(iv_daily.index)
    return (iv_daily.loc[idx] - r.loc[idx] ** 2).rename("vrp_harvest")


def regime_split(series: pd.Series, vol_index: pd.Series, threshold: float = 30.0):
    """Split a series into (calm, stressed) on the vol-index level (<= threshold vs > threshold).

    Used for the both-sides rule (NN#2): VRP is large in calm regimes and collapses/inverts
    in stress — both sides must be reported, never conditioned away.
    """
    v = vol_index.reindex(series.index).ffill()
    return series[v <= threshold], series[v > threshold]
