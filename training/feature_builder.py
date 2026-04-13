"""
V6.5 Feature Builder and HYP-008 Integration
Implements the 10-year trend deviation z-score as a mathematical feature.
Validation: Requires >=0.5% Accuracy Lift on OOS data for inclusion.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_decadal_divergence_score(price_series, lookback_10yr=2520):
    """
    Measures the standard deviation of current price from its 10-year trend.
    Input: daily or hourly series (adjust lookback accordingly).
    For hourly 10yr = 252 * 7 * 10 = 17,640 bars. 
    However, we focus on daily decadal memory mapped to hourly.
    """
    if len(price_series) < 100: # Minimum data check
        return 0.0
    
    # 1. Exponential moving average as the decadal trend proxy
    # (Memory decay set to 10 years approximation)
    trend = price_series.ewm(span=lookback_10yr).mean()
    
    # 2. Distance from trend
    deviation = price_series - trend
    
    # 3. Standard Deviation of the deviation
    vol = deviation.rolling(window=lookback_10yr, min_periods=lookback_10yr//2).std()
    
    # 4. Final Z-score (Decadal Divergence Score)
    z_score = deviation / vol
    
    return z_score.ffill().fillna(0.0)

if __name__ == "__main__":
    # Internal Unit Test
    example_data = pd.Series(np.random.normal(100, 5, 20000)).cumsum()
    score = compute_decadal_divergence_score(example_data)
    print(f"Feature Unit Test: Mean Score = {score.mean():.4f}, Std dev = {score.std():.4f}")
