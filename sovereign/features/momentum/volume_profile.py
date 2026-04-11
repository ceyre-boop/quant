"""
Sovereign Trading Intelligence -- Volume Profile Features
Phase 2: Feature Layer

Features analyzing volume distribution, entropy, and order flow persistence.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)


def volume_zscore(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Standard volume z-score: (current_volume - rolling_mean) / rolling_std.
    Highlights abnormal institutional activity.
    """
    vol = df['volume']
    mean = vol.rolling(window).mean()
    std = vol.rolling(window).std()
    
    # Avoid division by zero
    z = (vol - mean) / std.replace(0, np.nan)
    return z.fillna(0)


def calculate_entropy(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculates Shannon Entropy of the volume distribution over a rolling window.
    Discretizes volume into bins to measure 'information' or 'surprise'.
    """
    def _entropy(x):
        if len(x) < window: return np.nan
        # Discretize into 10 bins
        counts, _ = np.histogram(x, bins=10)
        probs = counts / counts.sum()
        # Filter out zeros for entropy calculation
        return entropy(probs[probs > 0], base=2)

    return series.rolling(window).apply(_entropy, raw=True)


def ofi_decay(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Approximates Order Flow Imbalance (OFI) and measures its autocorrelation decay.
    High decay (low autocorrelation) = chaotic flow.
    Low decay (high autocorrelation) = persistent institutional flow.
    """
    # Simple OFI approximation from OHLCV
    # OFI = sum(buying_volume) - sum(selling_volume)
    # Here: up_bar_vol - down_bar_vol
    price_change = df['close'].diff()
    ofi = np.where(price_change > 0, df['volume'], np.where(price_change < 0, -df['volume'], 0))
    ofi_series = pd.Series(ofi, index=df.index)
    
    # Measure 1st order autocorrelation as proxy for persistence/decay rate
    return ofi_series.rolling(window).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False).fillna(0)


def compute_volume_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all volume profile features.
    
    Output: DataFrame with columns:
      - volume_zscore
      - volume_entropy
      - ofi_decay
    """
    vol_z = volume_zscore(df, window=20)
    vol_ent = calculate_entropy(df['volume'], window=20)
    ofi = ofi_decay(df, window=20)
    
    result = pd.DataFrame({
        'volume_zscore':  vol_z,
        'volume_entropy': vol_ent,
        'ofi_decay':      ofi,
    }, index=df.index)
    
    return result
