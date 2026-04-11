"""
Regime Filter V2.2 - Institutional Mean Reversion Gate
Part of Pillar 3: Regime-Aware Execution.

Acts as a pre-inference gate to prevent mean-reversion signals during 
high-entropy trending or panic regimes.
"""

import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

def is_mean_reversion_regime(features: Dict[str, float]) -> Tuple[bool, str]:
    """
    Hard gate. Returns False during trending regimes where mean reversion fails.
    Must pass before XGBoost score is even requested.
    """
    # 1. ADX above 28 = directional trend = reversion dangerous
    adx = features.get('adx_14', 0.0)
    if adx > 28:
        return False, f"TRENDING: ADX={adx:.1f} > 28"

    # 2. VIX spike = panic regime = mean can keep moving
    # High macro z-score indicates systemic tail risk
    vix_zscore = features.get('vix_zscore', 0.0)
    if vix_zscore > 2.0:
        return False, f"PANIC_REGIME: VIX_Z={vix_zscore:.2f}"

    # 3. Asset-level volatility explosion
    # Rejection if realized volatility is > 2.5 standard deviations from mean
    atr_zscore = features.get('atr_zscore', 0.0)
    if atr_zscore > 2.5:
        return False, f"VOLATILITY_EXPLOSION: ATR_Z={atr_zscore:.2f}"

    # 4. Optional: Velocity Gate (RSI Exhaustion)
    rsi = features.get('rsi_14', 50.0)
    if rsi > 85 or rsi < 15:
        return False, f"EXTREME_MOMENTUM: RSI={rsi:.1f}"

    return True, "RANGE_REGIME: cleared"
