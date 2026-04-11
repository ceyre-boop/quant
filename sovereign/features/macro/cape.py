"""
Sovereign Trading Intelligence -- Shiller CAPE Features
Phase 2: Feature Layer

Shiller CAPE (Cyclically Adjusted P/E Ratio) normalization vs 130-year history.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def shiller_cape_zscore(macro_data: pd.DataFrame) -> pd.Series:
    """
    Normalizes CAPE vs 130-year historical parameters.
    Mean = 16.8, Std = 7.2 (Hardcoded per prompt).
    """
    # Look for 'shiller_cape' in macro data.
    # If missing, we return NaN as we cannot derive CAPE without historical earnings.
    if 'shiller_cape' in macro_data.columns:
        cape = macro_data['shiller_cape']
    else:
        # Placeholder/Proxy if unavailable: 
        # CAPE is rarely available in high frequency. 
        # Returning NaN to avoid noisy proxies.
        return pd.Series(np.nan, index=macro_data.index)
        
    zscore = (cape - 16.8) / 7.2
    return zscore


def cape_percentile(macro_data: pd.DataFrame) -> pd.Series:
    """
    Percentile in current historical distribution.
    """
    if 'shiller_cape' in macro_data.columns:
        cape = macro_data['shiller_cape']
        return cape.rolling(252*10, min_periods=252).rank(pct=True)
    return pd.Series(np.nan, index=macro_data.index)


def compute_cape_features(macro_data: pd.DataFrame) -> pd.DataFrame:
    """Compute CAPE z-score and percentile."""
    z = shiller_cape_zscore(macro_data)
    p = cape_percentile(macro_data)
    
    result = pd.DataFrame({
        'shiller_cape_zscore': z,
        'cape_percentile':     p,
    }, index=macro_data.index)
    
    return result
