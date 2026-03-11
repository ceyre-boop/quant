"""Integration module - Firebase UI formatting and broadcasting.

Provides utilities to convert internal layer outputs to frontend-ready format
and broadcast to Firebase Realtime Database.
"""

from integration.firebase_ui_writer import (
    format_signal_for_ui,
    format_bias_for_ui,
    format_risk_for_ui,
    format_game_output_for_ui,
    format_regime_for_ui,
    format_rationale,
    format_direction,
    format_magnitude,
    format_pool_for_ui,
    format_adversarial_risk,
    # Legacy aliases
    format_layer1_for_ui,
    format_layer2_for_ui,
    format_layer3_for_ui,
)

from integration.firebase_broadcaster import FirebaseBroadcaster

__all__ = [
    # UI Formatters
    'format_signal_for_ui',
    'format_bias_for_ui',
    'format_risk_for_ui',
    'format_game_output_for_ui',
    'format_regime_for_ui',
    'format_rationale',
    'format_direction',
    'format_magnitude',
    'format_pool_for_ui',
    'format_adversarial_risk',
    # Legacy aliases
    'format_layer1_for_ui',
    'format_layer2_for_ui',
    'format_layer3_for_ui',
    # Broadcaster
    'FirebaseBroadcaster',
]
