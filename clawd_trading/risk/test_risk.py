"""
Tests for Risk Module
"""

import pytest

from clawd_trading.risk import (
    RegimeCluster,
    get_regime_risk_limits,
    classify_regime_from_layer1,
    apply_regime_risk_to_gate12,
    get_regime_bias_adjustment,
    calculate_combined_risk_limits,
)


def test_regime_risk_limits_quiet():
    """Quiet accumulation should have full risk."""
    limits = get_regime_risk_limits(RegimeCluster.QUIET_ACCUMULATION)
    assert limits.max_per_trade_R == 1.0
    assert limits.max_concurrent_R == 3.0


def test_regime_risk_limits_news_shock():
    """News shock should block all trades."""
    limits = get_regime_risk_limits(RegimeCluster.NEWS_SHOCK_EXPLOSION)
    assert limits.max_per_trade_R == 0.0
    assert limits.news_rules.get("block_fresh_entry") == True


def test_regime_risk_limits_news_pre():
    """News pre-release should reduce risk."""
    limits = get_regime_risk_limits(RegimeCluster.NEWS_PRE_RELEASE_COMPRESSION)
    assert limits.max_per_trade_R == 0.3
    assert len(limits.allowed_pattern_families) == 2


def test_classify_regime_news_shock():
    """Should detect news shock."""
    layer1 = {
        "news_minutes_to_event": 2,
        "news_impact_score": 0.8,
    }
    regime = classify_regime_from_layer1(layer1)
    assert regime == RegimeCluster.NEWS_SHOCK_EXPLOSION


def test_classify_regime_volatility_breakout():
    """Should detect volatility breakout."""
    layer1 = {
        "volatility_regime": "high",
        "trend_regime": "uptrend",
    }
    regime = classify_regime_from_layer1(layer1)
    assert regime == RegimeCluster.VOLATILITY_BREAKOUT


def test_apply_regime_risk_blocks_news():
    """Should block entry during news shock."""
    base_limits = {"max_position_size": 1.0, "allow_entry": True}
    adjusted = apply_regime_risk_to_gate12(base_limits, RegimeCluster.NEWS_SHOCK_EXPLOSION)
    assert adjusted["allow_entry"] == False
    assert "news_shock" in adjusted["block_reason"]


def test_apply_regime_risk_reduces_size():
    """Should reduce size in choppy regime."""
    base_limits = {"max_position_size": 1.0}
    adjusted = apply_regime_risk_to_gate12(base_limits, RegimeCluster.CHOPPY_MANIPULATION)
    assert adjusted["max_position_size"] < 1.0
    assert adjusted["max_position_size"] == 0.6


def test_regime_bias_adjustment():
    """Should adjust bias based on regime."""
    adj = get_regime_bias_adjustment(RegimeCluster.NEWS_SHOCK_EXPLOSION)
    assert adj["confidence_adjustment"] < 0  # Reduces confidence

    adj = get_regime_bias_adjustment(RegimeCluster.VOLATILITY_BREAKOUT)
    assert adj["confidence_adjustment"] > 0  # Increases confidence


def test_combined_risk_with_participants():
    """Test combined participant + regime risk."""
    from clawd_trading.participants import ParticipantLikelihood, ParticipantType

    likelihoods = [
        ParticipantLikelihood(type=ParticipantType.RETAIL, probability=0.8, evidence={}),
    ]

    base_limits = {
        "max_position_size": 1.0,
        "max_concurrent_risk": 3.0,
        "max_daily_risk": 5.0,
    }

    combined = calculate_combined_risk_limits(
        participant_likelihoods=likelihoods,
        regime=RegimeCluster.EXPANSIVE_TREND,
        base_limits=base_limits,
    )

    assert "risk_multipliers" in combined
    assert combined["risk_multipliers"]["combined_size"] <= 0.9


def test_combined_risk_blocks_on_news():
    """Combined risk should block during news shock."""
    from clawd_trading.participants import ParticipantLikelihood, ParticipantType

    likelihoods = [ParticipantLikelihood(type=ParticipantType.RETAIL, probability=1.0, evidence={})]

    base_limits = {"max_position_size": 1.0, "allow_entry": True}

    combined = calculate_combined_risk_limits(
        participant_likelihoods=likelihoods,
        regime=RegimeCluster.NEWS_SHOCK_EXPLOSION,
        base_limits=base_limits,
    )

    assert combined["allow_entry"] == False
    assert "regime_news_shock" in combined["block_reason"]
