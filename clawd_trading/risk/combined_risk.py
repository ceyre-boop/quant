"""
Combined Risk System for Clawd Trading

Integrates Participant Risk + Regime Risk for unified risk management.
Used in Gate 12 (Entry Engine) for final risk validation.
"""

from typing import Any, Dict, List, Optional

from clawd_trading.participants import (
    ParticipantLikelihood,
    calculate_participant_risk_limits,
)
from clawd_trading.risk.regime_risk import (
    RegimeCluster,
    classify_regime_from_layer1,
    get_regime_risk_limits,
)


def calculate_combined_risk_limits(
    participant_likelihoods: List[ParticipantLikelihood],
    regime: RegimeCluster,
    base_limits: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate combined risk limits from participant + regime analysis.

    Multiplies participant and regime risk multipliers for conservative sizing.

    Args:
        participant_likelihoods: From participant classification
        regime: From regime classification
        base_limits: Base risk limits from Layer 3

    Returns:
        Combined risk limits with all adjustments
    """
    # Get participant risk
    participant_limits = calculate_participant_risk_limits(participant_likelihoods)

    # Get regime risk
    regime_limits = get_regime_risk_limits(regime)

    # Combine multipliers (multiplicative for conservative sizing)
    combined_size_multiplier = (
        participant_limits.max_size_multiplier * regime_limits.max_per_trade_R
    )

    combined_concurrent_multiplier = participant_limits.max_concurrent_risk * (
        regime_limits.max_concurrent_R / 3.0
    )  # Normalize to 0-1

    combined_daily_multiplier = participant_limits.max_daily_risk * (
        regime_limits.max_daily_R / 5.0
    )  # Normalize to 0-1

    # Check for trade blocks
    block_reasons = []

    if participant_limits.no_trade:
        block_reasons.append("participant_news_algo")

    if regime_limits.news_rules.get("block_fresh_entry"):
        block_reasons.append("regime_news_shock")

    # Apply to base limits
    adjusted = dict(base_limits)

    if block_reasons:
        adjusted["allow_entry"] = False
        adjusted["block_reasons"] = block_reasons
        adjusted["block_reason"] = " AND ".join(block_reasons)
    else:
        # Apply size adjustments
        if "max_position_size" in adjusted:
            adjusted["max_position_size"] *= combined_size_multiplier

        if "max_risk_per_trade_usd" in adjusted:
            adjusted["max_risk_per_trade_usd"] *= combined_size_multiplier

        if "max_concurrent_risk" in adjusted:
            adjusted["max_concurrent_risk"] *= combined_concurrent_multiplier

        if "max_daily_risk" in adjusted:
            adjusted["max_daily_risk"] *= combined_daily_multiplier

    # Add metadata
    adjusted["risk_multipliers"] = {
        "participant_size": participant_limits.max_size_multiplier,
        "regime_size": regime_limits.max_per_trade_R,
        "combined_size": combined_size_multiplier,
        "participant_concurrent": participant_limits.max_concurrent_risk,
        "regime_concurrent": regime_limits.max_concurrent_R / 3.0,
        "combined_concurrent": combined_concurrent_multiplier,
    }

    adjusted["participant_metadata"] = participant_limits.metadata
    adjusted["regime_metadata"] = {
        "regime": regime.value,
        "news_rules": regime_limits.news_rules,
    }

    return adjusted


def validate_entry_with_combined_risk(
    entry_signal: Dict[str, Any],
    layer1_output: Dict[str, Any],
    base_risk_limits: Dict[str, Any],
    current_exposure: Dict[str, float],
) -> Dict[str, Any]:
    """
    Validate entry signal with combined participant + regime risk.

    Full integration for Gate 12.

    Args:
        entry_signal: Proposed entry from Entry Engine
        layer1_output: Layer 1 analysis output
        base_risk_limits: From Layer 3 / Hard Constraints
        current_exposure: Current portfolio exposure

    Returns:
        Validation result with allow/block decision
    """
    from clawd_trading.participants import (
        extract_from_layer1_context,
        classify_participants,
    )

    # Extract participant features and classify
    participant_features = extract_from_layer1_context(layer1_output)
    participant_likelihoods = classify_participants(participant_features)

    # Classify regime
    regime = classify_regime_from_layer1(layer1_output)

    # Calculate combined risk
    combined_limits = calculate_combined_risk_limits(
        participant_likelihoods=participant_likelihoods,
        regime=regime,
        base_limits=base_risk_limits,
    )

    # Check if blocked
    if not combined_limits.get("allow_entry", True):
        return {
            "valid": False,
            "action": "BLOCK",
            "reason": combined_limits.get("block_reason", "risk_limits_exceeded"),
            "details": combined_limits,
        }

    # Check position size against combined limits
    proposed_size = entry_signal.get("position_size", 0)
    max_size = combined_limits.get("max_position_size", float("inf"))

    if proposed_size > max_size:
        return {
            "valid": False,
            "action": "BLOCK",
            "reason": "position_size_exceeds_combined_limit",
            "proposed_size": proposed_size,
            "max_allowed": max_size,
            "details": combined_limits,
        }

    # Check concurrent risk
    current_concurrent = current_exposure.get("concurrent_risk", 0)
    max_concurrent = combined_limits.get("max_concurrent_risk", float("inf"))

    if current_concurrent > max_concurrent:
        return {
            "valid": False,
            "action": "BLOCK",
            "reason": "concurrent_risk_exceeds_limit",
            "current": current_concurrent,
            "max_allowed": max_concurrent,
            "details": combined_limits,
        }

    # All checks passed
    return {
        "valid": True,
        "action": "ALLOW",
        "adjusted_limits": combined_limits,
        "risk_metadata": {
            "participant_dominant": max(
                participant_likelihoods, key=lambda x: x.probability
            ).type.name,
            "regime": regime.value,
            "size_multiplier": combined_limits["risk_multipliers"]["combined_size"],
        },
    }
