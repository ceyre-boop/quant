"""
Critical Slowing Down (CSD) Detector
Pillar 14: Predictive Regime Exhaustion

Based on physics of complex systems approaching phase transitions.
- Rising Variance (Destabilization)
- Rising Lag-1 Autocorreation (Critical Slowing)
- Lengthening Recovery (Structural Fragility)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CriticalSlowingDetector:
    def __init__(self, window=60, detrend_window=20):
        self.window = window
        self.detrend_window = detrend_window

    def compute(self, price_series: pd.Series):
        # 1. Detrend (Residuals from the mean)
        trend = price_series.rolling(self.detrend_window).mean()
        residuals = price_series - trend
        residuals = residuals.dropna()
        
        if len(residuals) < self.window:
            return pd.DataFrame(index=price_series.index)

        # 2. Indicator: Variance Velocity (Measuring Wobble)
        variance = residuals.rolling(self.window).var()
        variance_velocity = variance.diff(5) / variance.shift(5)

        # 3. Indicator: Lag-1 Autocorrelation (Measuring Stickiness)
        def rolling_ar1(x):
            if len(x) < 10: return 0
            # Simple manual AR1 to avoid overhead of statsmodels
            return np.corrcoef(x[:-1], x[1:])[0,1]
            
        ar1 = residuals.rolling(self.window).apply(rolling_ar1, raw=True)
        ar1_velocity = ar1.diff(5)

        # 4. Indicator: Recovery Time (Measuring Sluggishness)
        def recovery_time(x):
            mean = np.mean(x)
            # Find zero crossings of residuals
            crossings = np.where(np.diff(np.sign(x - mean)))[0]
            if len(crossings) < 2:
                return len(x) # Max sluggishness
            return np.mean(np.diff(crossings))

        shock_recovery = residuals.rolling(self.window).apply(recovery_time, raw=True)

        # 5. Composite CSD Score (Percentile Ranked)
        # We rank over the session to normalize the physics indicators
        scores = pd.DataFrame(index=residuals.index)
        scores['var_rank'] = variance_velocity.rank(pct=True)
        scores['ar1_rank'] = ar1_velocity.rank(pct=True)
        scores['rec_rank'] = shock_recovery.rank(pct=True)
        
        # Weighted Composite
        scores['csd_score'] = (
            scores['var_rank'] * 0.35 +
            scores['ar1_rank'] * 0.40 +
            scores['rec_rank'] * 0.25
        )
        
        # Merge back to original index
        final = pd.DataFrame(index=price_series.index)
        final['csd_score'] = scores['csd_score']
        final['ar1'] = ar1
        final['variance_velocity'] = variance_velocity
        final['shock_recovery'] = shock_recovery
        
        return final.ffill()
