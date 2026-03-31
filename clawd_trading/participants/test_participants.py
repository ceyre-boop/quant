"""
Tests for Clawd Trading Participant Module
"""
import pytest

from clawd_trading.participants import (
    ParticipantType,
    ParticipantFeatureVector,
    ParticipantLikelihood,
    classify_participants,
    calculate_participant_risk_limits,
    get_dominant_participant,
    get_participant_risk_multiplier,
    apply_participant_risk_to_gate12,
)


def test_participant_types_exist():
    """All 7 participant types should be defined."""
    types = list(ParticipantType)
    assert len(types) == 7
    assert ParticipantType.SWEEP_BOT in types
    assert ParticipantType.MARKET_MAKER in types
    assert ParticipantType.LIQUIDITY_HUNTER in types


def test_sweep_bot_detection():
    """High sweep intensity should detect SWEEP_BOT."""
    features = ParticipantFeatureVector(
        orderflow_velocity=2.0,
        sweep_intensity=5.0,  # High
        absorption_ratio=1.0,
        spread_pressure=0.0,
        liquidity_removal_rate=1.0,
        volatility_reaction=1.0,
        time_of_day_bias='open',
        news_window_behavior='none',
        metadata={}
    )
    
    likelihoods = classify_participants(features)
    sweep_likelihood = next(l for l in likelihoods if l.type == ParticipantType.SWEEP_BOT)
    
    assert sweep_likelihood.probability > 0.2  # Should be elevated
    assert sweep_likelihood.evidence['high_sweep_intensity'] == True


def test_market_maker_detection():
    """High absorption + low sweep should detect MARKET_MAKER."""
    features = ParticipantFeatureVector(
        orderflow_velocity=2.0,
        sweep_intensity=1.0,  # Low
        absorption_ratio=2.0,  # High
        spread_pressure=0.0,
        liquidity_removal_rate=1.0,
        volatility_reaction=1.0,
        time_of_day_bias='all_day',
        news_window_behavior='none',
        metadata={}
    )
    
    likelihoods = classify_participants(features)
    mm_likelihood = next(l for l in likelihoods if l.type == ParticipantType.MARKET_MAKER)
    
    assert mm_likelihood.evidence['high_absorption_ratio'] == True
    assert mm_likelihood.evidence['low_sweep_intensity'] == True


def test_news_algo_blocks_trade():
    """NEWS_ALGO should trigger no_trade."""
    likelihoods = [
        ParticipantLikelihood(
            type=ParticipantType.NEWS_ALGO,
            probability=0.20,  # Above threshold
            evidence={'news_during_and_high_volatility': True}
        ),
        ParticipantLikelihood(
            type=ParticipantType.RETAIL,
            probability=0.80,
            evidence={}
        ),
    ]
    
    limits = calculate_participant_risk_limits(likelihoods)
    assert limits.no_trade == True
    assert limits.metadata['news_algo_no_trade'] == True


def test_sweep_bot_reduces_size():
    """SWEEP_BOT should reduce position size."""
    likelihoods = [
        ParticipantLikelihood(
            type=ParticipantType.SWEEP_BOT,
            probability=0.20,  # Above threshold
            evidence={'high_sweep_intensity': True}
        ),
        ParticipantLikelihood(
            type=ParticipantType.RETAIL,
            probability=0.80,
            evidence={}
        ),
    ]
    
    limits = calculate_participant_risk_limits(likelihoods)
    assert limits.max_size_multiplier < 1.0
    assert limits.metadata['sweep_bot_caution'] == True


def test_risk_multiplier_values():
    """Risk multipliers should be reasonable."""
    assert get_participant_risk_multiplier(ParticipantType.RETAIL) == 1.0
    assert get_participant_risk_multiplier(ParticipantType.SWEEP_BOT) < 1.0
    assert get_participant_risk_multiplier(ParticipantType.NEWS_ALGO) < 1.0


def test_gate12_integration():
    """Test integration with Gate 12."""
    base_limits = {
        'max_position_size': 1.0,
        'max_concurrent_risk': 500.0,
        'stop_loss_pips': 20.0,
    }
    
    # Create limits with SWEEP_BOT
    participant_limits = calculate_participant_risk_limits([
        ParticipantLikelihood(
            type=ParticipantType.SWEEP_BOT,
            probability=0.20,
            evidence={}
        ),
    ])
    
    adjusted = apply_participant_risk_to_gate12(base_limits, participant_limits)
    
    # Should reduce size
    assert adjusted['max_position_size'] < base_limits['max_position_size']
    assert 'participant_risk_metadata' in adjusted


def test_gate12_blocks_on_news():
    """Gate 12 should block when NEWS_ALGO detected."""
    base_limits = {'max_position_size': 1.0}
    
    participant_limits = calculate_participant_risk_limits([
        ParticipantLikelihood(
            type=ParticipantType.NEWS_ALGO,
            probability=0.20,
            evidence={}
        ),
    ])
    
    adjusted = apply_participant_risk_to_gate12(base_limits, participant_limits)
    
    assert adjusted.get('no_trade') == True
    assert adjusted.get('block_reason') == 'news_algo_detected'


def test_dominant_participant():
    """Should return participant with highest probability."""
    likelihoods = [
        ParticipantLikelihood(type=ParticipantType.RETAIL, probability=0.1, evidence={}),
        ParticipantLikelihood(type=ParticipantType.FUND, probability=0.6, evidence={}),
        ParticipantLikelihood(type=ParticipantType.ALGO, probability=0.3, evidence={}),
    ]
    
    dominant = get_dominant_participant(likelihoods)
    assert dominant.type == ParticipantType.FUND
    assert dominant.probability == 0.6
