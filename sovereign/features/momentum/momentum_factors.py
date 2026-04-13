"""
Sovereign Trading Intelligence -- Momentum Factors
Phase 2: Feature Layer

Trend strength indicators computed from raw OHLCV data.
No external TA library dependency -- computed manually for
full control and auditability.

Features produced for the router:
  - adx_14:     Average Directional Index (14-bar)
  - adx_zscore: ADX relative to its own 90-day history

ADX interpretation:
  ADX < 20  = weak trend (range-bound, favor reversion)
  ADX 20-40 = developing trend (favor momentum)
  ADX > 40  = strong trend (momentum but watch for exhaustion)

ADX z-score adds context: an ADX of 25 means different things
depending on whether the asset typically trends (low z) or
is usually range-bound (high z).
"""

import numpy as np
import pandas as pd
import logging
from config.loader import params

logger = logging.getLogger(__name__)


def _compute_true_range(df: pd.DataFrame) -> pd.Series:
    """True Range = max(H-L, |H-Cprev|, |L-Cprev|)"""
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_atr(df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
    """
    Average True Range.
    """
    p = period if period is not None else 14
    tr = _compute_true_range(df)
    return tr.ewm(alpha=1.0 / p, min_periods=p).mean()


def compute_adx(df: pd.DataFrame, period: Optional[int] = None) -> pd.DataFrame:
    """
    Average Directional Index (Wilder's ADX).

    Manual implementation for full auditability.
    No external TA library dependency.

    Returns DataFrame with:
      - adx:    ADX value
      - plus_di:  +DI (positive directional indicator)
      - minus_di: -DI (negative directional indicator)
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where(
        (up_move > down_move) & (up_move > 0), up_move, 0.0
    ), index=df.index)

    minus_dm = pd.Series(np.where(
        (down_move > up_move) & (down_move > 0), down_move, 0.0
    ), index=df.index)

    # Smoothed TR and DM (Wilder's smoothing)
    alpha = 1.0 / period
    atr = _compute_true_range(df).ewm(alpha=alpha, min_periods=period).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=alpha, min_periods=period).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=alpha, min_periods=period).mean()

    # Directional Indicators
    plus_di = 100.0 * smooth_plus_dm / atr
    minus_di = 100.0 * smooth_minus_dm / atr

    # DX and ADX
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100.0 * di_diff / di_sum.replace(0, np.nan)

    adx = dx.ewm(alpha=alpha, min_periods=period).mean()

    return pd.DataFrame({
        'adx':      adx,
        'plus_di':  plus_di,
        'minus_di': minus_di,
    }, index=df.index)


def compute_rsi(price_series: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI calculation."""
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all momentum/trend-strength features needed by the Router.
    """
    close = df['close']
    pm = params['momentum']
    pr = params['regime']
    
    # ADX
    adx_df = compute_adx(df, period=14)
    adx_14 = adx_df['adx']

    # ADX z-score: how extreme is current ADX vs its own history?
    adx_mean = adx_14.rolling(90, min_periods=30).mean()
    adx_std = adx_14.rolling(90, min_periods=30).std()
    adx_zscore = (adx_14 - adx_mean) / adx_std.replace(0, np.nan)

    # ATR
    atr_14 = compute_atr(df, period=14)
    
    # Jegadeesh-Titman 12-1 Momentum
    # Standard: (Close_t1 / Close_t12) - 1. 
    # Use config window.
    window = pm['momentum_12_1_window']
    skip = 21 # 1 month skip
    jt_momentum = (close.shift(skip) / close.shift(window)) - 1
    
    # RSI
    rsi_14 = compute_rsi(close, period=pm['rsi_window'])
    
    result = pd.DataFrame({
        'adx':            adx_14,
        'adx_zscore':     adx_zscore,
        'atr':            atr_14,
        'jt_momentum':    jt_momentum,
        'rsi':            rsi_14,
        'roc_5':          close.pct_change(5),
    }, index=df.index)

    return result
