"""
Institutional Regime Fingerprinting (Pillar 3: Modular Architecture)
Compresses Multi-Tier Macro signals into a unified S-Vector.
This vector represents the 'Market Fingerprint' used to tune risk parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from governance.policy_engine import GOVERNANCE

@dataclass
class SVector:
    """The 10-D Market Fingerprint (Pillar 3)."""
    yield_curve_slope: float       # Tier 1: Macro baseline
    momentum_regime: float         # Tier 1: Price persistence
    volatility_regime: float       # Tier 1: Risk environment
    value_spread: float            # Tier 2: Fundamental displacement
    cot_pressure: float            # Tier 2: Institutional positioning
    equity_credit_spread: float    # Tier 2: Hidden risk lead
    options_skew_intensity: float  # Tier 2: Tail risk pricing
    sector_rotation_idx: float     # Tier 2: Capital flow phase
    insider_sentiment: float       # Tier 2: Informed flow
    experimental_alpha_v1: float   # Tier 3: Hypothesis trial (e.g., FOMC NLP)
    regime_tag: str = "NEUTRAL"    # V2.2: Semantic tagging for Hard Constraints


    def to_array(self) -> np.array:
        # V2.2: Only project numerical fields for the ML array
        numerical_fields = [f.name for f in self.__dataclass_fields__.values() if f.name != 'regime_tag']
        return np.array([getattr(self, fname) for fname in numerical_fields])


class RegimeFingerprinter:
    """
    Constructs the S-Vector by aggregating data from across the 3 Macro Tiers.
    """
    def __init__(self):
        self.lookback = GOVERNANCE.parameters['bias_engine']['lookback_period']
        self.normalization_window = 252 # 1-year rolling normalization

    def generate_fingerprint(self, date_dt: datetime, macro_context: Dict[str, Any], symbol: str = "SPY") -> SVector:
        """
        Compresses current macro state into the 10-dimensional S-Vector (TEC Loop Stage 1).
        V2.2: Includes symbol-aware semantic tagging.
        """
        # Pillar 4: Defensive Coding - Ensure macro_context is valid
        if not macro_context:
            return self._get_neutral_fingerprint()

        # Tier 1: Traditional Canonical (Normalization via Z-Score)
        yield_slope = macro_context.get('hmm_regime_stress', 0.5)
        momentum = macro_context.get('returns_20d', 0.0)
        volatility = macro_context.get('realized_volatility_5d', 0.1)

        # Tier 2: Institutional (Simulated/Placeholders)
        value_spread = 0.5
        cot_pressure = 0.5
        credit_lead = 0.5
        skew = 0.5
        rotation = 0.5
        insider = 0.5
        experimental = macro_context.get('recession_prob_12m', 0.3)

        # V2.2 Regime Tagging Logic
        tag = "NEUTRAL"
        if symbol in ["GOLD", "GLD", "SILVER", "SLV", "OIL", "USO", "UNG"]:
            if volatility > 0.20:
                tag = "Commodity_MeanRevert"
        elif symbol in ["SPY", "QQQ", "DIA", "IWM"]:
            if abs(momentum) > 0.02: # Meaningful trend
                tag = "StableTrend_Index"

        return SVector(
            yield_curve_slope=yield_slope,
            momentum_regime=momentum,
            volatility_regime=volatility,
            value_spread=value_spread,
            cot_pressure=cot_pressure,
            equity_credit_spread=credit_lead,
            options_skew_intensity=skew,
            sector_rotation_idx=rotation,
            insider_sentiment=insider,
            experimental_alpha_v1=experimental,
            regime_tag=tag
        )


    def _get_neutral_fingerprint(self) -> SVector:
        return SVector(*[0.5]*10)
