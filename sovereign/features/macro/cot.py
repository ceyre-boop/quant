"""
Sovereign Trading Intelligence -- COT (Commitment of Traders) Features
Phase 2: Feature Layer

Measures institutional positioning extremes using CFTC COT data.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def cot_zscore(macro_data: pd.DataFrame, lookback: int = 156) -> pd.Series:
    """
    Z-score of net commercial positioning.
    Lookback = 156 weeks (3 years).
    
    Signal: z > +1.5 long, z < -1.5 short.
    """
    if 'net_commercial' in macro_data.columns:
        net = macro_data['net_commercial']
    else:
        return pd.Series(np.nan, index=macro_data.index)
        
    rolling_mean = net.rolling(lookback).mean()
    rolling_std = net.rolling(lookback).std()
    
    z = (net - rolling_mean) / rolling_std.replace(0, np.nan)
    return z


def compute_cot_features(macro_data: pd.DataFrame) -> pd.DataFrame:
    """Compute COT z-score."""
    z = cot_zscore(macro_data)
    
    result = pd.DataFrame({
        'cot_zscore': z,
    }, index=macro_data.index)
    
    return result
