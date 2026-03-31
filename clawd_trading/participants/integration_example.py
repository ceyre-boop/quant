"""
Integration Example: Participant Analysis with 3-Layer Architecture

Shows how to use the participant module with your existing Layer 1/2/3 system.
"""

from clawd_trading.participants import (
    ParticipantType,
    extract_participant_features,
    classify_participants,
    calculate_participant_risk_limits,
    apply_participant_risk_to_gate12,
    get_participant_bias_adjustment,
    get_dominant_participant,
)


def example_layer1_to_participant_analysis(layer1_output: dict) -> dict:
    """
    Example: Integrate participant detection into your 3-layer flow.
    
    This would be called AFTER Layer 1 (Hard Constraints) runs.
    """
    # Step 1: Extract participant features from Layer 1 output
    features = extract_participant_features(
        tick_data={
            'bids': layer1_output.get('bids', []),
            'asks': layer1_output.get('asks', []),
            'trades': layer1_output.get('trades', []),
            'time_window': 1.0,
            'short_horizon_vol': layer1_output.get('volatility', 0.0),
            'baseline_vol': layer1_output.get('baseline_volatility', 0.01),
        },
        time_of_day=layer1_output.get('session', 'all_day'),
        news_window=layer1_output.get('news_window', 'none')
    )
    
    # Step 2: Classify participants
    likelihoods = classify_participants(features)
    dominant = get_dominant_participant(likelihoods)
    
    print(f"Dominant participant: {dominant.type.name} ({dominant.probability:.2%})")
    
    # Step 3: Calculate risk adjustments
    risk_limits = calculate_participant_risk_limits(likelihoods)
    
    # Step 4: Get bias adjustments for Layer 2
    bias_adjustments = get_participant_bias_adjustment(likelihoods)
    
    return {
        'participant_likelihoods': likelihoods,
        'dominant_participant': dominant,
        'risk_limits': risk_limits,
        'bias_adjustments': bias_adjustments,
    }


def example_gate12_integration(
    base_risk_limits: dict,
    participant_analysis: dict
) -> dict:
    """
    Example: Apply participant risk to Gate 12 (Entry Engine).
    
    This would be called DURING Gate 12 validation.
    """
    risk_limits = participant_analysis['risk_limits']
    
    # Apply participant adjustments to base limits
    adjusted_limits = apply_participant_risk_to_gate12(
        base_risk_limits=base_risk_limits,
        participant_limits=risk_limits
    )
    
    # Check if trade should be blocked
    if adjusted_limits.get('no_trade'):
        return {
            'allow_entry': False,
            'reason': 'News algo detected - blocking entry',
            'adjusted_limits': adjusted_limits
        }
    
    return {
        'allow_entry': True,
        'adjusted_limits': adjusted_limits
    }


def example_full_flow():
    """
    Complete example showing the full integration.
    """
    # Simulate Layer 1 output
    layer1_output = {
        'bids': [{'price': 100.0, 'size': 100}],
        'asks': [{'price': 100.1, 'size': 200}],
        'trades': [{'price': 100.05, 'size': 50, 'aggressive': True}],
        'volatility': 0.02,
        'baseline_volatility': 0.015,
        'session': 'open',
        'news_window': 'none',
    }
    
    # Run participant analysis
    analysis = example_layer1_to_participant_analysis(layer1_output)
    
    # Simulate base risk limits from Layer 3
    base_limits = {
        'max_position_size': 1.0,
        'max_concurrent_risk': 500.0,
        'max_daily_risk': 2000.0,
        'stop_loss_pips': 20.0,
        'take_profit_pips': 40.0,
    }
    
    # Apply to Gate 12
    gate12_result = example_gate12_integration(base_limits, analysis)
    
    print(f"\nGate 12 Result: {'ALLOW' if gate12_result['allow_entry'] else 'BLOCK'}")
    print(f"Adjusted Limits: {gate12_result['adjusted_limits']}")


if __name__ == "__main__":
    example_full_flow()
