"""
Sovereign Trading Intelligence -- Volume Profile Features
Phase 2: Feature Layer

Features analyzing volume distribution, entropy, and order flow persistence.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
import logging
from config.loader import params
from contracts.types import MomentumFeatures

logger = logging.getLogger(__name__)


def volume_zscore(df: pd.DataFrame, window: Optional[int] = None) -> pd.Series:
    """
    Standard volume z-score.
    """
    p = params['momentum']
    w = window if window is not None else p.get('volume_window', 20)
    vol = df['volume']
    mean = vol.rolling(window).mean()
    std = vol.rolling(window).std()
    
    # Avoid division by zero
    z = (vol - mean) / std.replace(0, np.nan)
    return z.fillna(0)


def calculate_entropy(series: pd.Series, window: Optional[int] = None) -> pd.Series:
    """
    Calculates Shannon Entropy of the volume distribution over a rolling window.
    """
    p = params['momentum']
    w = window if window is not None else p.get('volume_window', 20)
    
    def _entropy(x):
        if len(x) < w: return np.nan
        # Discretize into 10 bins
        counts, _ = np.histogram(x, bins=10)
        probs = counts / counts.sum()
        # Filter out zeros for entropy calculation
        return entropy(probs[probs > 0], base=2)

    return series.rolling(window).apply(_entropy, raw=True)


def ofi_velocity(df: pd.DataFrame, window: Optional[int] = None) -> pd.Series:
    """
    Approximates Order Flow Imbalance (OFI) and measures its persistence.
    """
    p = params['momentum']
    w = window if window is not None else p.get('volume_window', 20)
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
    """
    p = params['momentum']
    w = p.get('volume_window', 20)
    
    vol_z = volume_zscore(df, window=w)
    vol_ent = calculate_entropy(df['volume'], window=w)
    ofi = ofi_velocity(df, window=w)
    
    result = pd.DataFrame({
        'volume_zscore':  vol_z,
        'volume_entropy': vol_ent,
        'ofi_velocity':   ofi,
    }, index=df.index)
    
    return result
