"""
Participant Risk Envelope for Clawd Trading

Adjusts position sizing and risk limits based on detected market participants.
Integrates with Layer 3 risk calculations and Entry Engine Gate 12.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from copy import deepcopy

from .participant_taxonomy import ParticipantType, get_participant_risk_multiplier
from .participant_likelihood import ParticipantLikelihood, get_dominant_participant


@dataclass(frozen=True)
class ParticipantRiskLimits:
    """
    Risk limits adjusted for participant types.

    All multipliers are relative to base/global limits.
    """

    max_size_multiplier: float
    max_concurrent_risk: float
    max_daily_risk: float
    no_trade: bool
    stop_loss_multiplier: float
    take_profit_multiplier: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def calculate_participant_risk_limits(
    likelihoods: List[ParticipantLikelihood],
) -> ParticipantRiskLimits:
    """
    Calculate risk limits based on participant likelihoods.

    Args:
        likelihoods: List of ParticipantLikelihood from classify_participants()

    Returns:
        ParticipantRiskLimits with adjusted multipliers
    """
    # Start with neutral
    max_size_multiplier = 1.0
    max_concurrent_risk = 1.0
    max_daily_risk = 1.0
    stop_loss_multiplier = 1.0
    take_profit_multiplier = 1.0
    no_trade = False
    meta = {}

    # Create probability map
    probs = {l.type: l.probability for l in likelihoods}

    # SWEEP_BOT: reduce size, tighten stops
    if probs.get(ParticipantType.SWEEP_BOT, 0.0) >= 0.15:
        max_size_multiplier *= 0.7
        max_concurrent_risk *= 0.9
        stop_loss_multiplier *= 0.85  # Tighter stops
        meta["sweep_bot_caution"] = True
    else:
        meta["sweep_bot_caution"] = False

    # LIQUIDITY_HUNTER: reduce size slightly
    if probs.get(ParticipantType.LIQUIDITY_HUNTER, 0.0) >= 0.15:
        max_size_multiplier *= 0.85
        stop_loss_multiplier *= 0.9
        meta["liquidity_hunter_caution"] = True
    else:
        meta["liquidity_hunter_caution"] = False

    # MARKET_MAKER: slightly wider stops (more noise tolerance)
    if probs.get(ParticipantType.MARKET_MAKER, 0.0) >= 0.15:
        max_concurrent_risk = min(max_concurrent_risk * 1.08, 1.1)
        stop_loss_multiplier = min(stop_loss_multiplier * 1.1, 1.15)
        meta["market_maker_wider_stops"] = True
    else:
        meta["market_maker_wider_stops"] = False

    # FUND: reduce leverage but allow continuation
    if probs.get(ParticipantType.FUND, 0.0) >= 0.15:
        max_size_multiplier *= 0.9
        take_profit_multiplier *= 1.15  # Wider targets for trend
        meta["fund_reduce_leverage"] = True
    else:
        meta["fund_reduce_leverage"] = False

    # NEWS_ALGO: no_trade during news events
    if probs.get(ParticipantType.NEWS_ALGO, 0.0) >= 0.15:
        no_trade = True
        meta["news_algo_no_trade"] = True
    else:
        meta["news_algo_no_trade"] = False

    # RETAIL: small increase in noise tolerance
    if probs.get(ParticipantType.RETAIL, 0.0) >= 0.15:
        meta["retail_noise_tolerance"] = True
    else:
        meta["retail_noise_tolerance"] = False

    return ParticipantRiskLimits(
        max_size_multiplier=max_size_multiplier,
        max_concurrent_risk=max_concurrent_risk,
        max_daily_risk=max_daily_risk,
        no_trade=no_trade,
        stop_loss_multiplier=stop_loss_multiplier,
        take_profit_multiplier=take_profit_multiplier,
        metadata=meta,
    )


def apply_participant_risk_to_gate12(
    base_risk_limits: Dict[str, Any], participant_limits: ParticipantRiskLimits
) -> Dict[str, Any]:
    """
    Apply participant risk limits to Gate 12 risk validation.

    Args:
        base_risk_limits: From Layer 3 / Hard Constraints
        participant_limits: From calculate_participant_risk_limits()

    Returns:
        Adjusted risk limits for Entry Engine
    """
    adjusted = deepcopy(base_risk_limits)

    # Apply no_trade block
    if participant_limits.no_trade:
        adjusted["no_trade"] = True
        adjusted["block_reason"] = "news_algo_detected"
        return adjusted

    # Apply size multiplier
    if "max_position_size" in adjusted:
        adjusted["max_position_size"] *= participant_limits.max_size_multiplier

    if "max_size_usd" in adjusted:
        adjusted["max_size_usd"] *= participant_limits.max_size_multiplier

    # Apply concurrent risk
    if "max_concurrent_risk" in adjusted:
        adjusted["max_concurrent_risk"] *= participant_limits.max_concurrent_risk

    # Apply daily risk
    if "max_daily_risk" in adjusted:
        adjusted["max_daily_risk"] *= participant_limits.max_daily_risk

    # Apply stop loss adjustment
    if "stop_loss_pips" in adjusted:
        adjusted["stop_loss_pips"] *= participant_limits.stop_loss_multiplier

    if "stop_loss_pct" in adjusted:
        adjusted["stop_loss_pct"] *= participant_limits.stop_loss_multiplier

    # Apply take profit adjustment
    if "take_profit_pips" in adjusted:
        adjusted["take_profit_pips"] *= participant_limits.take_profit_multiplier

    if "take_profit_pct" in adjusted:
        adjusted["take_profit_pct"] *= participant_limits.take_profit_multiplier

    # Add metadata
    adjusted["participant_risk_metadata"] = participant_limits.metadata

    return adjusted


def get_participant_bias_adjustment(
    likelihoods: List[ParticipantLikelihood],
) -> Dict[str, Any]:
    """
    Get bias adjustments based on participants.

    Returns adjustments for Layer 2 bias calculations.
    """
    probs = {l.type: l.probability for l in likelihoods}
    dominant = get_dominant_participant(likelihoods)

    adjustments = {"bias_shift": 0.0, "confidence_adjustment": 0.0, "metadata": {}}

    # SWEEP_BOT: reduce confidence, add caution
    if probs.get(ParticipantType.SWEEP_BOT, 0.0) >= 0.20:
        adjustments["confidence_adjustment"] -= 0.1
        adjustments["metadata"]["sweep_bot_caution"] = True

    # MARKET_MAKER: slightly reduce confidence (choppy)
    if probs.get(ParticipantType.MARKET_MAKER, 0.0) >= 0.25:
        adjustments["confidence_adjustment"] -= 0.05
        adjustments["metadata"]["market_maker_chop"] = True

    # FUND: increase confidence if aligned
    if probs.get(ParticipantType.FUND, 0.0) >= 0.20:
        adjustments["confidence_adjustment"] += 0.05
        adjustments["metadata"]["fund_flow"] = True

    # NEWS_ALGO: high uncertainty
    if probs.get(ParticipantType.NEWS_ALGO, 0.0) >= 0.15:
        adjustments["confidence_adjustment"] -= 0.2
        adjustments["metadata"]["news_volatility"] = True

    return adjustments
