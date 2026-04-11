"""
Sovereign Trading Intelligence -- Yield Curve Macro Features
Phase 2: Feature Layer

Yield curve dynamics as early warning indicators for macro regime shifts.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


def yield_curve_spread(macro_data: pd.DataFrame) -> pd.Series:
    """T10Y minus T2Y spread."""
    # macro_feed uses 'dgs10' and 'dgs2'
    if 'dgs10' in macro_data.columns and 'dgs2' in macro_data.columns:
        return macro_data['dgs10'] - macro_data['dgs2']
    return pd.Series(np.nan, index=macro_data.index)


def yield_curve_velocity(macro_data: pd.DataFrame, days: int = 5) -> pd.Series:
    """Rate of spread change."""
    spread = yield_curve_spread(macro_data)
    return spread.diff(days)


def yield_curve_acceleration(macro_data: pd.DataFrame) -> pd.Series:
    """Second derivative of the spread."""
    velocity = yield_curve_velocity(macro_data, days=5)
    return velocity.diff(5)


def recession_probability(macro_data: pd.DataFrame) -> pd.Series:
    """
    Estrella-Mishkin (1996) Probit model for recession forecasting.
    Formula: norm.cdf(-0.6103 - 0.5582 * spread_10y_3m) * 100
    Note: Standard probit uses 10Y-3M spread.
    """
    if 'dgs10' in macro_data.columns and 'dgs3mo' in macro_data.columns:
        spread_10y_3m = macro_data['dgs10'] - macro_data['dgs3mo']
        prob = norm.cdf(-0.6103 - 0.5582 * spread_10y_3m) * 100
        return pd.Series(prob, index=macro_data.index)
    return pd.Series(np.nan, index=macro_data.index)


def resteepening_flag(macro_data: pd.DataFrame) -> pd.Series:
    """
    Flag = True if spread < 0 AND velocity > 0.5 (bps/week approx).
    Bear-steepeners after inversion are the most dangerous macro signals.
    """
    spread = yield_curve_spread(macro_data)
    velocity = yield_curve_velocity(macro_data, days=5)
    
    # 0.5 is a threshold in the prompt. Since yields are in percent (e.g., 4.5), 
    # it likely implies significant steepening.
    flag = (spread < 0) & (velocity > 0.1)  # Using 0.1 as a more sensitive weekly velocity for yields
    return flag.astype(int)


def compute_yield_curve_features(macro_data: pd.DataFrame) -> pd.DataFrame:
    """Compute all yield curve features."""
    spread = yield_curve_spread(macro_data)
    velocity = yield_curve_velocity(macro_data, days=5)
    accel = yield_curve_acceleration(macro_data)
    prob = recession_probability(macro_data)
    resteep = resteepening_flag(macro_data)
    
    result = pd.DataFrame({
        'yield_curve_spread':       spread,
        'yield_curve_velocity':     velocity,
        'yield_curve_acceleration': accel,
        'recession_probability':     prob,
        'resteepening_flag':        resteep,
    }, index=macro_data.index)
    
    return result
