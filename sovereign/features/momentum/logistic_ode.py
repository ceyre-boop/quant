"""
Sovereign Trading Intelligence -- Logistic ODE Accumulation Model
Phase 2: Feature Layer

Models institutional accumulation/distribution as a logistic growth
process. The intuition: large players accumulate positions gradually,
creating a volume-price signature that follows S-curve dynamics.

The logistic ODE: dV/dt = k * V * (1 - V/K)
  V = cumulative volume anomaly (normalized)
  k = growth rate (accumulation speed)
  K = carrying capacity (maximum position)

Features produced for the router:
  - logistic_k:            Accumulation rate parameter
  - logistic_acceleration: d2V/dt2 -- inflection signal

When logistic_acceleration crosses zero from positive to negative,
the accumulation phase is ending. This is a leading signal --
it fires BEFORE the price move exhausts, not after.

This was discovered in the V4.1 research phase.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import logging
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def _logistic_curve(t: np.ndarray, k: float, t0: float,
                    v_max: float) -> np.ndarray:
    """
    Logistic (S-curve) function.
    V(t) = v_max / (1 + exp(-k * (t - t0)))
    """
    return v_max / (1.0 + np.exp(-k * (t - t0)))


def _fit_logistic(volume_anomaly: np.ndarray) -> dict:
    """
    Fit a logistic curve to a window of cumulative volume anomaly.

    Returns:
        Dict with 'k' (growth rate), 'success' (bool), and 'r_squared'.
        If fit fails, returns k=0.
    """
    n = len(volume_anomaly)
    if n < 20:
        return {'k': 0.0, 'success': False, 'r_squared': 0.0}

    t = np.arange(n, dtype=float)
    y = volume_anomaly.copy()

    # Normalize to [0, 1] range for fitting stability
    y_min = np.min(y)
    y_max = np.max(y)
    y_range = y_max - y_min

    if y_range < 1e-10:
        return {'k': 0.0, 'success': False, 'r_squared': 0.0}

    y_norm = (y - y_min) / y_range

    try:
        # Initial guesses
        p0 = [0.1, n / 2, 1.0]
        bounds = ([0.001, 0, 0.5], [2.0, n, 2.0])

        popt, _ = curve_fit(
            _logistic_curve, t, y_norm,
            p0=p0, bounds=bounds, maxfev=500
        )

        k = popt[0]

        # R-squared for quality check
        y_pred = _logistic_curve(t, *popt)
        ss_res = np.sum((y_norm - y_pred) ** 2)
        ss_tot = np.sum((y_norm - np.mean(y_norm)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            'k': k,
            'success': True,
            'r_squared': r_squared,
        }

    except Exception:
        return {'k': 0.0, 'success': False, 'r_squared': 0.0}


def compute_logistic_features(df: pd.DataFrame,
                              window: int = 20,
                              step: int = 5) -> pd.DataFrame:
    """
    Compute Logistic ODE features from OHLCV data.

    Args:
        df:     DataFrame with 'close' and 'volume' columns
        window: Rolling window for logistic fitting (default 20 bars)
        step:   Fit every N bars to reduce computation (default 5)

    Output columns:
      - logistic_k:            Accumulation rate from fitted logistic
      - logistic_acceleration: Second derivative of cumulative volume anomaly
    """
    close = df['close'].values
    volume = df['volume'].values
    n = len(df)

    # Volume anomaly: deviation from rolling mean volume
    vol_mean = pd.Series(volume).rolling(window, min_periods=20).mean().values
    vol_anomaly = np.where(vol_mean > 0, volume / vol_mean - 1.0, 0.0)

    # Direction-weighted volume anomaly:
    # Positive when volume is above average AND price is moving up
    # Negative when volume is above average AND price is moving down
    price_direction = np.sign(np.diff(close, prepend=close[0]))
    directed_anomaly = vol_anomaly * price_direction

    # Cumulative sum over rolling window
    cum_anomaly = pd.Series(directed_anomaly).rolling(
        window, min_periods=20
    ).sum().values

    # Fit logistic to rolling windows
    k_values = np.full(n, np.nan)

    for i in range(window, n, step):
        segment = cum_anomaly[i - window:i]
        if np.any(np.isnan(segment)):
            continue

        result = _fit_logistic(segment)
        if result['success'] and result['r_squared'] > 0.3:
            k_values[i] = result['k']

    # Forward-fill the sparse k values (since we only compute every `step` bars)
    k_series = pd.Series(k_values, index=df.index).ffill()

    # Acceleration: second derivative of cumulative anomaly
    # This is the inflection signal -- when it crosses zero from positive
    # to negative, the accumulation phase is ending
    cum_anomaly_series = pd.Series(cum_anomaly, index=df.index)
    velocity = cum_anomaly_series.diff(5)
    acceleration = velocity.diff(5)

    result = pd.DataFrame({
        'logistic_k':            k_series,
        'logistic_acceleration': acceleration,
    }, index=df.index)

    return result
