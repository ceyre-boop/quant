"""
Risk Management Module for Clawd Trading

Integrates participant analysis, regime detection, and combined risk validation.
"""

from .regime_risk import (
    RegimeCluster,
    RegimeRiskLimits,
    RegimeRiskDecision,
    TradeAction,
    PatternFamily,
    get_regime_risk_limits,
    classify_regime_from_layer1,
    apply_regime_risk_to_gate12,
    get_regime_bias_adjustment,
)

from .combined_risk import (
    calculate_combined_risk_limits,
    validate_entry_with_combined_risk,
)

__all__ = [
    # Regime Risk
    "RegimeCluster",
    "RegimeRiskLimits",
    "RegimeRiskDecision",
    "TradeAction",
    "PatternFamily",
    "get_regime_risk_limits",
    "classify_regime_from_layer1",
    "apply_regime_risk_to_gate12",
    "get_regime_bias_adjustment",
    # Combined Risk
    "calculate_combined_risk_limits",
    "validate_entry_with_combined_risk",
]
