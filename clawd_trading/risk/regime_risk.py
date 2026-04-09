"""
Regime Risk Envelope for Clawd Trading

Dynamic risk limits based on market regime classification.
Integrates with Layer 1 regime output and Gate 12 risk validation.
Ported and adapted from trading-stockfish.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class RegimeCluster(Enum):
    """Market regime classifications."""

    QUIET_ACCUMULATION = "quiet_accumulation"
    EXPANSIVE_TREND = "expansive_trend"
    CHOPPY_MANIPULATION = "choppy_manipulation"
    NEWS_PRE_RELEASE_COMPRESSION = "news_pre_release_compression"
    NEWS_SHOCK_EXPLOSION = "news_shock_explosion"
    NEWS_POST_DIGESTION_TREND = "news_post_digestion_trend"
    LATE_SESSION_EXHAUSTION = "late_session_exhaustion"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    LIQUIDITY_DRAIN = "liquidity_drain"


class TradeAction(Enum):
    """Allowed trade actions."""

    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE_POSITION = "CLOSE_POSITION"
    HOLD = "HOLD"
    NO_TRADE = "NO_TRADE"


class PatternFamily(Enum):
    """Trading pattern families."""

    CONTINUATION = "continuation"
    MEAN_REVERSION = "mean_reversion"
    LIQUIDITY = "liquidity"
    BREAKOUT = "breakout"
    IMBALANCE = "imbalance"
    HYBRID = "hybrid"


# Global risk caps (never exceed these)
_GLOBAL_MAX_PER_TRADE_R = 1.0
_GLOBAL_MAX_CONCURRENT_R = 3.0
_GLOBAL_MAX_DAILY_R = 5.0


@dataclass(frozen=True)
class RegimeRiskLimits:
    """Risk limits for a specific regime."""

    regime: RegimeCluster
    max_per_trade_R: float
    max_concurrent_R: float
    max_daily_R: float
    allowed_action_types: List[str]
    allowed_pattern_families: List[PatternFamily]
    news_rules: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class RegimeRiskDecision:
    """Result of regime risk check."""

    allowed: bool
    reason: str
    adjusted_size: Optional[float]
    metadata: Dict[str, Any]


def _clamp(val: float, cap: float) -> float:
    """Clamp value between 0 and cap."""
    return float(min(max(val, 0.0), cap))


def get_regime_risk_limits(cluster: RegimeCluster) -> RegimeRiskLimits:
    """
    Get risk limits for a regime cluster.

    Each regime has specific risk adjustments:
    - NEWS_SHOCK: Block all trades
    - NEWS_PRE: Reduce size, mean reversion only
    - TREND: Full size, continuation patterns
    - CHOPPY: Reduce size, restrict patterns
    """
    base = RegimeRiskLimits(
        regime=cluster,
        max_per_trade_R=_GLOBAL_MAX_PER_TRADE_R,
        max_concurrent_R=_GLOBAL_MAX_CONCURRENT_R,
        max_daily_R=_GLOBAL_MAX_DAILY_R,
        allowed_action_types=[a.value for a in TradeAction],
        allowed_pattern_families=list(PatternFamily),
        news_rules={},
        metadata={},
    )

    # Regime-specific overrides
    overrides: Dict[RegimeCluster, Dict[str, Any]] = {
        RegimeCluster.NEWS_SHOCK_EXPLOSION: {
            "max_per_trade_R": 0.0,
            "max_concurrent_R": 0.0,
            "max_daily_R": 0.0,
            "allowed_action_types": [TradeAction.NO_TRADE.value],
            "allowed_pattern_families": [],
            "news_rules": {"block_fresh_entry": True},
        },
        RegimeCluster.NEWS_PRE_RELEASE_COMPRESSION: {
            "max_per_trade_R": 0.3,  # 30% normal size
            "max_concurrent_R": 1.0,
            "max_daily_R": 1.5,
            "allowed_action_types": [
                TradeAction.OPEN_LONG.value,
                TradeAction.OPEN_SHORT.value,
            ],
            "allowed_pattern_families": [
                PatternFamily.LIQUIDITY,
                PatternFamily.MEAN_REVERSION,
            ],
            "news_rules": {"restrict_continuation": True},
        },
        RegimeCluster.NEWS_POST_DIGESTION_TREND: {
            "max_per_trade_R": 0.6,  # 60% normal size
            "max_concurrent_R": 2.0,
            "max_daily_R": 3.0,
            "allowed_action_types": [
                TradeAction.OPEN_LONG.value,
                TradeAction.OPEN_SHORT.value,
            ],
            "allowed_pattern_families": [
                PatternFamily.LIQUIDITY,
                PatternFamily.CONTINUATION,
            ],
            "news_rules": {"boost_liquidity": True},
        },
        RegimeCluster.EXPANSIVE_TREND: {
            "max_per_trade_R": 0.9,  # 90% normal size
            "allowed_pattern_families": [
                PatternFamily.CONTINUATION,
                PatternFamily.LIQUIDITY,
                PatternFamily.HYBRID,
            ],
        },
        RegimeCluster.CHOPPY_MANIPULATION: {
            "max_per_trade_R": 0.6,  # 60% normal size
            "max_concurrent_R": 2.0,
            "max_daily_R": 3.0,
            "allowed_pattern_families": [
                PatternFamily.MEAN_REVERSION,
                PatternFamily.LIQUIDITY,
            ],
        },
        RegimeCluster.VOLATILITY_BREAKOUT: {
            "max_per_trade_R": 0.7,  # 70% normal size
            "max_concurrent_R": 2.5,
            "max_daily_R": 4.0,
            "allowed_pattern_families": [
                PatternFamily.CONTINUATION,
                PatternFamily.IMBALANCE,
            ],
        },
        RegimeCluster.LIQUIDITY_DRAIN: {
            "max_per_trade_R": 0.5,  # 50% normal size
            "max_concurrent_R": 1.5,
            "max_daily_R": 2.0,
            "allowed_pattern_families": [
                PatternFamily.LIQUIDITY,
                PatternFamily.MEAN_REVERSION,
            ],
        },
        RegimeCluster.QUIET_ACCUMULATION: {
            "max_per_trade_R": 1.0,  # Full size
        },
    }

    ov = overrides.get(cluster, {})
    return RegimeRiskLimits(
        regime=cluster,
        max_per_trade_R=_clamp(
            float(ov.get("max_per_trade_R", base.max_per_trade_R)),
            _GLOBAL_MAX_PER_TRADE_R,
        ),
        max_concurrent_R=_clamp(
            float(ov.get("max_concurrent_R", base.max_concurrent_R)),
            _GLOBAL_MAX_CONCURRENT_R,
        ),
        max_daily_R=_clamp(
            float(ov.get("max_daily_R", base.max_daily_R)),
            _GLOBAL_MAX_DAILY_R,
        ),
        allowed_action_types=ov.get("allowed_action_types", base.allowed_action_types),
        allowed_pattern_families=ov.get("allowed_pattern_families", base.allowed_pattern_families),
        news_rules=ov.get("news_rules", base.news_rules),
        metadata={"source": "regime_risk_limits", "regime": cluster.value},
    )


def classify_regime_from_layer1(layer1_output: Dict[str, Any]) -> RegimeCluster:
    """
    Classify regime from Layer 1 output.

    Uses volatility, trend, session, and news data from Layer 1.
    """
    vol = layer1_output.get("volatility_regime", "normal")
    trend = layer1_output.get("trend_regime", "neutral")
    session = layer1_output.get("session", "")
    news_minutes = layer1_output.get("news_minutes_to_event")
    news_impact = layer1_output.get("news_impact_score", 0)

    # News shock detection (priority)
    if news_minutes is not None and abs(news_minutes) <= 5 and news_impact >= 0.7:
        return RegimeCluster.NEWS_SHOCK_EXPLOSION

    if news_minutes is not None and 0 <= news_minutes <= 10:
        return RegimeCluster.NEWS_PRE_RELEASE_COMPRESSION

    if news_minutes is not None and -20 <= news_minutes <= -5:
        return RegimeCluster.NEWS_POST_DIGESTION_TREND

    # Volatility regimes
    if vol in ["high", "breakout"] and trend in ["uptrend", "downtrend"]:
        return RegimeCluster.VOLATILITY_BREAKOUT

    if vol in ["elevated"] and trend in ["range", "chop"]:
        return RegimeCluster.CHOPPY_MANIPULATION

    if layer1_output.get("liquidity_state") in ["thin", "dry"]:
        return RegimeCluster.LIQUIDITY_DRAIN

    if vol in ["normal", "expansive"] and trend in ["uptrend", "downtrend"]:
        return RegimeCluster.EXPANSIVE_TREND

    if vol in ["low", "quiet"]:
        return RegimeCluster.QUIET_ACCUMULATION

    if "CLOSE" in str(session).upper() or "LATE" in str(session).upper():
        return RegimeCluster.LATE_SESSION_EXHAUSTION

    return RegimeCluster.QUIET_ACCUMULATION


def apply_regime_risk_to_gate12(
    base_limits: Dict[str, Any],
    regime: RegimeCluster,
    current_concurrent_r: float = 0.0,
    current_daily_r: float = 0.0,
) -> Dict[str, Any]:
    """
    Apply regime risk limits to Gate 12 risk validation.

    Args:
        base_limits: Base risk limits from Layer 3
        regime: Detected regime from Layer 1
        current_concurrent_r: Current concurrent risk exposure
        current_daily_r: Current daily risk exposure

    Returns:
        Adjusted limits with regime constraints
    """
    limits = get_regime_risk_limits(regime)

    # Check for trade blocks
    if limits.news_rules.get("block_fresh_entry"):
        return {
            **base_limits,
            "allow_entry": False,
            "block_reason": f"news_shock_{regime.value}",
            "regime": regime.value,
        }

    # Apply size multiplier
    adjusted = dict(base_limits)

    if "max_position_size" in adjusted:
        adjusted["max_position_size"] *= limits.max_per_trade_R

    if "max_risk_per_trade_usd" in adjusted:
        adjusted["max_risk_per_trade_usd"] *= limits.max_per_trade_R

    # Check concurrent risk
    if current_concurrent_r > limits.max_concurrent_R:
        adjusted["allow_entry"] = False
        adjusted["block_reason"] = "concurrent_risk_exceeds_regime"

    # Check daily risk
    if current_daily_r > limits.max_daily_R:
        adjusted["allow_entry"] = False
        adjusted["block_reason"] = "daily_risk_exceeds_regime"

    # Add regime metadata
    adjusted["regime"] = regime.value
    adjusted["regime_risk_multiplier"] = limits.max_per_trade_R
    adjusted["allowed_patterns"] = [p.value for p in limits.allowed_pattern_families]

    return adjusted


def get_regime_bias_adjustment(regime: RegimeCluster) -> Dict[str, Any]:
    """
    Get bias adjustments based on regime.

    Returns adjustments for Layer 2 bias calculations.
    """
    adjustments: Dict[str, Any] = {
        "bias_shift": 0.0,
        "confidence_adjustment": 0.0,
        "metadata": {"regime": regime.value},
    }

    if regime == RegimeCluster.NEWS_SHOCK_EXPLOSION:
        adjustments["confidence_adjustment"] = -0.5
        adjustments["metadata"]["high_uncertainty"] = True

    elif regime == RegimeCluster.NEWS_PRE_RELEASE_COMPRESSION:
        adjustments["confidence_adjustment"] = -0.2
        adjustments["metadata"]["reduced_confidence"] = True

    elif regime == RegimeCluster.VOLATILITY_BREAKOUT:
        adjustments["confidence_adjustment"] = 0.1
        adjustments["metadata"]["momentum_favorable"] = True

    elif regime == RegimeCluster.CHOPPY_MANIPULATION:
        adjustments["confidence_adjustment"] = -0.15
        adjustments["metadata"]["choppy_caution"] = True

    elif regime == RegimeCluster.LIQUIDITY_DRAIN:
        adjustments["confidence_adjustment"] = -0.25
        adjustments["metadata"]["liquidity_caution"] = True

    return adjustments
