"""
LAYER C — Regime Classification
Hurst Exponent, ADX, Volatility — determines if mean-reversion signals are valid
"""

import logging
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class RegimeResult:
    score: float  # -3 to +3
    regime: str  # "mean_reverting", "random_walk", "trending"
    hurst: float
    adx: float
    vol_percentile: float
    is_valid_for_mr: bool  # Mean-reversion signals valid?


class RegimeLayer:
    """
    Layer C: Regime Classification
    A mean-reversion signal in a trending market is a death trap.
    """
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config["layer_c_regime"]
    
    async def compute(self, symbol: str, data: Dict[str, Any]) -> RegimeResult:
        """Classify current market regime."""
        self.logger.debug(f"Classifying regime for {symbol}")
        
        hurst = data.get("hurst_exponent", 0.5)
        adx = data.get("adx", 20)
        vol_pct = data.get("volatility_percentile", 50)
        
        # Determine regime from Hurst
        if hurst < self.config["hurst_exponent"]["mean_reverting_threshold"]:
            regime = "mean_reverting"
            score = self.config["regime_scores"]["mean_reverting"]
            is_valid = True
        elif hurst > self.config["hurst_exponent"]["trending_threshold"]:
            regime = "trending"
            score = self.config["regime_scores"]["trending"]
            is_valid = False
        else:
            regime = "random_walk"
            score = self.config["regime_scores"]["random_walk"]
            is_valid = False
        
        # ADX confirmation
        if adx < self.config["adx"]["weak_trend"] and vol_pct < self.config["volatility_percentile"]["low"]:
            # Compression/range environment - mean reversion plays work
            if regime != "trending":
                is_valid = True
                score = max(score, 1.0)
        
        return RegimeResult(
            score=score,
            regime=regime,
           hurst=hurst,
            adx=adx,
            vol_percentile=vol_pct,
            is_valid_for_mr=is_valid
        )
