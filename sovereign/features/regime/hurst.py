"""
Sovereign Trading Intelligence -- Hurst Exponent Feature
Phase 2: Feature Layer

R/S (Rescaled Range) method, dual windows: 60 and 90 bars.
Plus Hurst velocity (rate of change) for regime transition detection.

This is the SAME R/S calculation validated in research/hurst_diagnostic.py
and deployed in sovereign/data/universe.py. Wrapped as a feature producer
to output the exact columns the Regime Router expects:
  - hurst_60:       60-bar Hurst exponent
  - hurst_90:       90-bar Hurst exponent (slower, more stable)
  - hurst_velocity: 5-bar rate of change of hurst_90

Interpretation:
  H > 0.52 = trending (momentum regime)
  H < 0.45 = mean reverting (reversion regime)
  0.45-0.52 = dead zone (both specialists lose)
  Rising velocity = regime strengthening
  Falling velocity = regime weakening (watch for transition)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _hurst_rs(ts: np.ndarray) -> float:
    """
    Single-window R/S Hurst calculation.
    Exact copy from research/hurst_diagnostic.py.
    Do not modify -- thresholds calibrated against this method.
    """
    n = len(ts)
    if n < 30:
        return 0.5  # Insufficient data -- assume random walk

    returns = np.diff(np.log(ts))
    n_ret = len(returns)

    mean = np.mean(returns)
    deviation = np.cumsum(returns - mean)
    r = np.max(deviation) - np.min(deviation)
    s = np.std(returns)

    if s == 0 or r == 0:
        return 0.5

    return np.log(r / s) / np.log(n_ret)


def compute_hurst(price_series: pd.Series, window: int = 90) -> pd.Series:
    """
    Rolling Hurst exponent via R/S analysis.

    Args:
        price_series: Close prices (must be > 0)
        window:       Rolling window size

    Returns:
        pd.Series of Hurst exponents, NaN during warm-up.
    """
    return price_series.rolling(window).apply(_hurst_rs, raw=True)


def compute_hurst_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all Hurst features needed by the Regime Router.

    Input: DataFrame with 'close' column.
    Output: DataFrame with columns:
      - hurst_60
      - hurst_90
      - hurst_velocity  (5-bar diff of hurst_90)

    All columns have the same index as the input.
    """
    close = df['close']

    hurst_60 = compute_hurst(close, window=60)
    hurst_90 = compute_hurst(close, window=90)

    # Velocity: how fast is the Hurst reading changing?
    # Rising = regime strengthening. Falling = regime weakening.
    hurst_velocity = hurst_90.diff(5)

    result = pd.DataFrame({
        'hurst_60':       hurst_60,
        'hurst_90':       hurst_90,
        'hurst_velocity': hurst_velocity,
    }, index=df.index)

    return result
