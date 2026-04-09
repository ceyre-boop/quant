"""
Participant Likelihood Model for Clawd Trading

Classifies market participants based on microstructure features.
Returns probability distribution over participant types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .participant_taxonomy import ParticipantType
from .participant_features import ParticipantFeatureVector


@dataclass(frozen=True)
class ParticipantLikelihood:
    """Likelihood of a specific participant type being active."""

    type: ParticipantType
    probability: float
    evidence: Dict[str, Any]


def classify_participants(
    features: ParticipantFeatureVector,
) -> List[ParticipantLikelihood]:
    """
    Classify participant type likelihoods from feature vector.

    Uses deterministic rule-based scoring (no ML required).
    Returns sorted list of ParticipantLikelihood objects.

    Args:
        features: ParticipantFeatureVector from extract_participant_features()

    Returns:
        List of ParticipantLikelihood, sorted by ParticipantType name
    """
    # Initialize scores with base prior
    scores = {ptype: 1.0 for ptype in ParticipantType}
    evidence = {ptype: {} for ptype in ParticipantType}

    # SWEEP_BOT: high sweep_intensity
    if features.sweep_intensity > 4.0:
        scores[ParticipantType.SWEEP_BOT] += 3.0
        evidence[ParticipantType.SWEEP_BOT]["high_sweep_intensity"] = True
    else:
        evidence[ParticipantType.SWEEP_BOT]["high_sweep_intensity"] = False

    # MARKET_MAKER: high absorption_ratio, low sweep_intensity
    if features.absorption_ratio > 1.5 and features.sweep_intensity < 2.0:
        scores[ParticipantType.MARKET_MAKER] += 2.5
        evidence[ParticipantType.MARKET_MAKER]["high_absorption_ratio"] = True
        evidence[ParticipantType.MARKET_MAKER]["low_sweep_intensity"] = True
    else:
        evidence[ParticipantType.MARKET_MAKER]["high_absorption_ratio"] = (
            features.absorption_ratio > 1.5
        )
        evidence[ParticipantType.MARKET_MAKER]["low_sweep_intensity"] = (
            features.sweep_intensity < 2.0
        )

    # ALGO: high orderflow_velocity, high volatility_reaction
    if features.orderflow_velocity > 4.0 and features.volatility_reaction > 1.5:
        scores[ParticipantType.ALGO] += 2.0
        evidence[ParticipantType.ALGO]["high_orderflow_velocity"] = True
        evidence[ParticipantType.ALGO]["high_volatility_reaction"] = True
    else:
        evidence[ParticipantType.ALGO]["high_orderflow_velocity"] = (
            features.orderflow_velocity > 4.0
        )
        evidence[ParticipantType.ALGO]["high_volatility_reaction"] = (
            features.volatility_reaction > 1.5
        )

    # LIQUIDITY_HUNTER: high liquidity_removal_rate
    if features.liquidity_removal_rate > 3.0:
        scores[ParticipantType.LIQUIDITY_HUNTER] += 2.0
        evidence[ParticipantType.LIQUIDITY_HUNTER]["high_liquidity_removal_rate"] = True
    else:
        evidence[ParticipantType.LIQUIDITY_HUNTER][
            "high_liquidity_removal_rate"
        ] = False

    # RETAIL: open, low volatility
    if features.time_of_day_bias == "open" and features.volatility_reaction < 1.2:
        scores[ParticipantType.RETAIL] += 1.5
        evidence[ParticipantType.RETAIL]["open_and_low_volatility"] = True
    else:
        evidence[ParticipantType.RETAIL]["open_and_low_volatility"] = (
            features.time_of_day_bias == "open" and features.volatility_reaction < 1.2
        )

    # NEWS_ALGO: news window during, high volatility
    if features.news_window_behavior == "during" and features.volatility_reaction > 1.5:
        scores[ParticipantType.NEWS_ALGO] += 2.5
        evidence[ParticipantType.NEWS_ALGO]["news_during_and_high_volatility"] = True
    else:
        evidence[ParticipantType.NEWS_ALGO]["news_during_and_high_volatility"] = (
            features.news_window_behavior == "during"
            and features.volatility_reaction > 1.5
        )

    # FUND: mid, neutral vol
    if features.time_of_day_bias == "mid" and 0.8 < features.volatility_reaction < 1.3:
        scores[ParticipantType.FUND] += 1.5
        evidence[ParticipantType.FUND]["mid_neutral_vol"] = True
    else:
        evidence[ParticipantType.FUND]["mid_neutral_vol"] = (
            features.time_of_day_bias == "mid"
            and 0.8 < features.volatility_reaction < 1.3
        )

    # Normalize to probabilities
    total = sum(scores.values())
    if total == 0:
        norm_scores = {ptype: 1.0 / len(scores) for ptype in scores}
    else:
        norm_scores = {ptype: v / total for ptype, v in scores.items()}

    # Build output, sorted by ParticipantType name
    likelihoods = [
        ParticipantLikelihood(
            type=ptype, probability=norm_scores[ptype], evidence=evidence[ptype]
        )
        for ptype in sorted(ParticipantType, key=lambda x: x.name)
    ]
    return likelihoods


def get_dominant_participant(
    likelihoods: List[ParticipantLikelihood],
) -> ParticipantLikelihood:
    """
    Get the participant type with highest probability.

    Args:
        likelihoods: List from classify_participants()

    Returns:
        ParticipantLikelihood with max probability
    """
    return max(likelihoods, key=lambda x: x.probability)


def has_participant_type(
    likelihoods: List[ParticipantLikelihood],
    ptype: ParticipantType,
    threshold: float = 0.15,
) -> bool:
    """
    Check if a participant type exceeds probability threshold.

    Args:
        likelihoods: List from classify_participants()
        ptype: ParticipantType to check
        threshold: Minimum probability (default 0.15)

    Returns:
        True if participant type probability >= threshold
    """
    for likelihood in likelihoods:
        if likelihood.type == ptype:
            return likelihood.probability >= threshold
    return False
