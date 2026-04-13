"""
Sovereign Trading Intelligence -- Critical Slowing Down Detector
Phase 2: Feature Layer

Physics-based regime exhaustion detector.
Applies Scheffer et al. (2009) ecological tipping point theory
to individual asset hourly bars.

Never published in trading literature at this resolution.
This is the genuine innovation.

Signal: detects when a trend is ABOUT TO END,
not after it has ended.

Three independent signals combined into a composite score:
1. Variance Velocity  -- system destabilizing (wobble increasing)
2. AR(1) Autocorrelation -- system becoming "sticky" (critical slowing)
3. Recovery Time -- system losing resilience (sluggishness increasing)

When all three rise simultaneously, the current regime is structurally
weakening. This is the CSD exit warning that the momentum specialist
checks before entering trades.

Router features produced:
  - csd_score:        Composite score (0-1), high = regime weakening
  - csd_ar1:          Raw AR(1) autocorrelation
  - csd_variance_vel: Variance velocity (rate of destabilization)
  - csd_recovery:     Mean recovery time (resilience loss)

Implementation matches research/critical_slowing_detector.py exactly.
"""

import numpy as np
import pandas as pd
import logging
from config.loader import params

logger = logging.getLogger(__name__)


class CriticalSlowingDetector:
    """
    Detects approaching regime transitions using physics of
    complex systems near tipping points.

    Usage:
        detector = CriticalSlowingDetector(window=60, detrend_window=20)
        features = detector.compute(df['close'])
    """

    def __init__(self, window: Optional[int] = None, detrend_window: Optional[int] = None):
        p = params['regime']
        self.window = window if window is not None else p['csd_window']
        self.detrend_window = detrend_window if detrend_window is not None else p['csd_detrend_window']

    def compute(self, price_series: pd.Series) -> pd.DataFrame:
        """
        Compute all CSD indicators from a price series.

        Args:
            price_series: Close prices (pd.Series with DatetimeIndex)

        Returns:
            DataFrame with columns: csd_score, csd_ar1,
            csd_variance_vel, csd_recovery
        """
        # Detrend: residuals from rolling mean
        trend = price_series.rolling(self.detrend_window).mean()
        residuals = price_series - trend

        # Signal 1: Variance trajectory
        # Rising variance = system destabilizing
        variance = residuals.rolling(self.window).var()
        variance_velocity = variance.pct_change(5).fillna(0)

        # Signal 2: AR(1) autocorrelation
        # Rising AR1 approaching 1.0 = classic CSD signature
        # The system is becoming "sticky" -- perturbations persist longer
        ar1 = residuals.rolling(self.window).apply(
            self._rolling_ar1, raw=True
        )
        ar1_velocity = ar1.diff(5).fillna(0)

        # Signal 3: Recovery rate from shocks
        # Lengthening recovery time = system losing resilience
        recovery = residuals.rolling(self.window).apply(
            self._mean_recovery_time, raw=True
        )

        # Composite CSD score (0-1)
        # High = regime structurally weakening = exit warning
        # Weights: AR1 dominates because it's the most theoretically
        # grounded CSD indicator (Scheffer 2009)
        csd_score = (
            variance_velocity.rank(pct=True) * 0.35 +
            ar1_velocity.rank(pct=True)       * 0.40 +
            recovery.rank(pct=True)           * 0.25
        ).fillna(0)

        result = pd.DataFrame({
            'csd_score':        csd_score,
            'csd_ar1':          ar1,
            'csd_variance_vel': variance_velocity,
            'csd_recovery':     recovery,
        }, index=price_series.index)

        return result

    @staticmethod
    def _rolling_ar1(x: np.ndarray) -> float:
        """
        AR(1) autocorrelation coefficient.
        Rising toward 1.0 = classic critical slowing down.
        """
        if len(x) < 10:
            return 0.0
        return np.corrcoef(x[:-1], x[1:])[0, 1]

    @staticmethod
    def _mean_recovery_time(x: np.ndarray) -> float:
        """
        How long does the series take to cross its mean after a deviation?
        Lengthening = system losing resilience = tipping point approaching.
        """
        mean = np.mean(x)
        crossings = np.where(np.diff(np.sign(x - mean)))[0]
        if len(crossings) < 2:
            return float(len(x))  # Never recovered = maximum score
        return float(np.mean(np.diff(crossings)))


def compute_csd_features(df: pd.DataFrame,
                         window: Optional[int] = None,
                         detrend_window: Optional[int] = None) -> pd.DataFrame:
    """
    Convenience function: compute CSD features from an OHLCV DataFrame.
    
    Input: DataFrame with 'close' column.
    Output: DataFrame with csd_score, csd_ar1, csd_variance_vel, csd_recovery.
    """
    p = params['regime']
    w = window if window is not None else p['csd_window']
    dw = detrend_window if detrend_window is not None else p['csd_detrend_window']
    
    detector = CriticalSlowingDetector(window=w, detrend_window=dw)
    return detector.compute(df['close'])
