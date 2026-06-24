"""
sovereign/discovery/features.py
===============================
Causal feature matrix for discovery. Every column uses only information available
at or before the bar (rolling windows, no future leakage) — the backtest enters at
the next bar's open, so signals derived here are honest.

Self-contained pandas/numpy implementations (robust across environments) covering
the families in the scaffold: momentum, volatility, oscillators, trend/structure,
regime, session/time, multi-timeframe alignment, Bollinger. Where the repo already
has a richer detector (sovereign/forex/ict_engine.py) it can be layered on later;
this base set is enough to generate and gate candidates today.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder_atr(high, low, close, n=14):
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _adx(high, low, close, n=14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    atr = _wilder_atr(high, low, close, n)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0.0)


def _rolling_hurst(close, window=100):
    """Lightweight rolling Hurst proxy via rescaled-range on log returns.

    <0.5 mean-reverting, ~0.5 random, >0.5 trending. Approximate but causal.
    """
    logret = np.log(close).diff().fillna(0.0)

    def _h(x):
        x = x - x.mean()
        z = np.cumsum(x)
        r = z.max() - z.min()
        s = x.std()
        if s <= 1e-12 or r <= 0:
            return 0.5
        return float(np.log(r / s) / np.log(len(x)))

    return logret.rolling(window).apply(_h, raw=True).fillna(0.5).clip(0.0, 1.0)


def compute_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature DataFrame aligned to price_df.index (same length)."""
    df = price_df
    close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    high = df["High"] if "High" in df.columns else close
    low = df["Low"] if "Low" in df.columns else close

    f = pd.DataFrame(index=df.index)
    # momentum / returns
    f["ret1"] = close.pct_change()
    f["ret5"] = close.pct_change(5)
    f["ret20"] = close.pct_change(20)
    f["mom_sign20"] = np.sign(f["ret20"]).fillna(0.0)
    # volatility
    atr = _wilder_atr(high, low, close, 14)
    f["atr_pct"] = (atr / close).fillna(0.0)
    f["range_pct"] = ((high - low) / close).fillna(0.0)
    f["vol_z"] = (f["atr_pct"] - f["atr_pct"].rolling(100).mean()) / f["atr_pct"].rolling(100).std()
    # oscillators
    f["rsi14"] = _rsi(close, 14)
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()).fillna(0.0)
    f["adx14"] = _adx(high, low, close, 14)
    # bollinger
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    f["bb_width"] = ((4 * std20) / sma20).fillna(0.0)
    f["bb_pos"] = ((close - sma20) / (2 * std20)).clip(-3, 3).fillna(0.0)  # z within bands
    # trend / multi-timeframe alignment
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    f["dist_sma50"] = (close / sma50 - 1).fillna(0.0)
    f["above_sma200"] = (close > sma200).astype(float)
    f["mtf_align"] = (np.sign(close - sma50) + np.sign(sma50 - sma200)).fillna(0.0)  # -2..+2
    # regime
    f["hurst"] = _rolling_hurst(close, 100)
    # session / time
    f["dow"] = pd.Series(df.index.dayofweek, index=df.index).astype(float)
    f["month"] = pd.Series(df.index.month, index=df.index).astype(float)
    f["is_quarter_end"] = pd.Series(df.index.is_quarter_end, index=df.index).astype(float)

    return f.replace([np.inf, -np.inf], 0.0)


FEATURE_COLUMNS = [
    "ret1", "ret5", "ret20", "mom_sign20", "atr_pct", "range_pct", "vol_z",
    "rsi14", "macd_hist", "adx14", "bb_width", "bb_pos", "dist_sma50",
    "above_sma200", "mtf_align", "hurst", "dow", "month", "is_quarter_end",
]
