"""
Clawd Trading - Participant Analysis Module

Market participant detection and risk adjustment.
Ported and adapted from trading-stockfish.
"""

from .participant_taxonomy import (
    ParticipantType,
    ParticipantSignature,
    get_participant_signatures,
    get_participant_risk_multiplier,
)

from .participant_features import (
    ParticipantFeatureVector,
    extract_participant_features,
    extract_from_layer1_context,
)

from .participant_likelihood import (
    ParticipantLikelihood,
    classify_participants,
    get_dominant_participant,
    has_participant_type,
)

from .participant_risk import (
    ParticipantRiskLimits,
    calculate_participant_risk_limits,
    apply_participant_risk_to_gate12,
    get_participant_bias_adjustment,
)

__all__ = [
    # Taxonomy
    'ParticipantType',
    'ParticipantSignature',
    'get_participant_signatures',
    'get_participant_risk_multiplier',
    # Features
    'ParticipantFeatureVector',
    'extract_participant_features',
    'extract_from_layer1_context',
    # Likelihood
    'ParticipantLikelihood',
    'classify_participants',
    'get_dominant_participant',
    'has_participant_type',
    # Risk
    'ParticipantRiskLimits',
    'calculate_participant_risk_limits',
    'apply_participant_risk_to_gate12',
    'get_participant_bias_adjustment',
]
