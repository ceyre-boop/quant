"""Firebase UI Writer - Convert Layer1/2/3 outputs to frontend-ready format.

Transforms internal data structures to match the frontend's expected JSON format.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from contracts.types import (
    BiasOutput,
    RiskOutput,
    GameOutput,
    RegimeState,
    Direction,
    Magnitude,
    AdversarialRisk,
    VolRegime,
    TrendRegime,
    RiskAppetite,
    MomentumRegime,
    EventRisk,
    FeatureGroup,
)


def format_direction(direction: Direction) -> int:
    """Convert Direction enum to integer for frontend."""
    return direction.value


def format_magnitude(magnitude: Magnitude) -> int:
    """Convert Magnitude enum to integer for frontend."""
    return magnitude.value


def format_rationale(
    rationale: List[str], feature_snapshot: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    """Format rationale with SHAP-like values.

    Args:
        rationale: List of feature group names
        feature_snapshot: Optional snapshot with SHAP values

    Returns:
        List of rationale dicts with group and shap values
    """
    formatted = []

    # If feature_snapshot has SHAP values, use them
    shap_values = {}
    if feature_snapshot and "shap_values" in feature_snapshot:
        shap_values = feature_snapshot["shap_values"]

    for group_name in rationale:
        # Format SHAP value
        shap_val = shap_values.get(group_name, 0.0)
        shap_str = f"{shap_val:+.2f}" if shap_val != 0 else "+0.00"

        formatted.append({"group": group_name, "shap": shap_str})

    return formatted


def format_bias_for_ui(
    bias: BiasOutput, feature_snapshot: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convert Layer 1 BiasOutput to frontend format.

    Args:
        bias: BiasOutput from Layer 1
        feature_snapshot: Optional feature snapshot with additional metadata

    Returns:
        Frontend-ready layer1 dict
    """
    return {
        "direction": format_direction(bias.direction),
        "confidence": round(bias.confidence, 2),
        "magnitude": format_magnitude(bias.magnitude),
        "rationale": format_rationale(bias.rationale, feature_snapshot),
    }


def format_risk_for_ui(
    risk: RiskOutput,
    current_price: Optional[float] = None,
    bias: Optional[BiasOutput] = None,
) -> Dict[str, Any]:
    """Convert Layer 2 RiskOutput to frontend format.

    Args:
        risk: RiskOutput from Layer 2
        current_price: Current market price (for entry_price calculation)
        bias: Optional BiasOutput for direction context

    Returns:
        Frontend-ready layer2 dict
    """
    # Calculate entry price from current price and bias if needed
    entry_price = current_price
    if entry_price is None:
        # Default to stop + offset or derive from TP/stop
        if bias and bias.direction == Direction.LONG:
            entry_price = risk.stop_price + (risk.tp1_price - risk.stop_price) * 0.3
        elif bias and bias.direction == Direction.SHORT:
            entry_price = risk.stop_price - (risk.stop_price - risk.tp1_price) * 0.3
        else:
            entry_price = risk.stop_price  # Fallback

    return {
        "position_size": round(risk.position_size, 2),
        "entry_price": round(entry_price, 2) if entry_price else None,
        "stop_price": round(risk.stop_price, 2),
        "tp1_price": round(risk.tp1_price, 2),
        "tp2_price": round(risk.tp2_price, 2),
        "expected_value": round(risk.expected_value, 2),
        "kelly_fraction": round(risk.kelly_fraction, 2),
        "stop_method": risk.stop_method,
    }


def format_pool_for_ui(pool: Any) -> Optional[Dict[str, Any]]:
    """Format a liquidity pool for UI.

    Args:
        pool: LiquidityPool object or dict

    Returns:
        Frontend-ready pool dict or None
    """
    if pool is None:
        return None

    # Handle both object and dict inputs
    if hasattr(pool, "price"):
        # It's an object
        price = pool.price
        strength = pool.strength
        draw_probability = pool.draw_probability
        pool_type = pool.pool_type if hasattr(pool, "pool_type") else "unknown"
    else:
        # It's a dict
        price = pool.get("price")
        strength = pool.get("strength", 0)
        draw_probability = pool.get("draw_probability", 0.0)
        pool_type = pool.get("pool_type", "unknown")

    # Determine direction based on pool type
    if pool_type == "equal_lows":
        direction = "below"
    elif pool_type == "equal_highs":
        direction = "above"
    else:
        direction = "unknown"

    return {
        "price": round(price, 2) if price else None,
        "strength": strength,
        "direction": direction,
        "draw_probability": round(draw_probability, 2),
    }


def format_adversarial_risk(risk: AdversarialRisk) -> str:
    """Convert AdversarialRisk enum to string."""
    return risk.value if isinstance(risk, AdversarialRisk) else str(risk)


def format_game_output_for_ui(game: GameOutput) -> Dict[str, Any]:
    """Convert Layer 3 GameOutput to frontend format.

    Args:
        game: GameOutput from Layer 3

    Returns:
        Frontend-ready layer3 dict
    """
    return {
        "game_state_aligned": game.game_state_aligned,
        "adversarial_risk": format_adversarial_risk(game.adversarial_risk),
        "game_state_summary": game.game_state_summary,
        "forced_move_probability": round(game.forced_move_probability, 2),
        "nearest_unswept_pool": format_pool_for_ui(game.nearest_unswept_pool),
        "kyle_lambda": round(game.kyle_lambda, 2),
    }


def format_regime_for_ui(regime: RegimeState) -> Dict[str, str]:
    """Convert RegimeState to frontend format.

    Args:
        regime: RegimeState from regime classifier

    Returns:
        Frontend-ready regime dict
    """
    return {
        "volatility": (
            regime.volatility.value
            if isinstance(regime.volatility, VolRegime)
            else str(regime.volatility)
        ),
        "trend": (
            regime.trend.value
            if isinstance(regime.trend, TrendRegime)
            else str(regime.trend)
        ),
        "risk_appetite": (
            regime.risk_appetite.value
            if isinstance(regime.risk_appetite, RiskAppetite)
            else str(regime.risk_appetite)
        ),
        "momentum": (
            regime.momentum.value
            if isinstance(regime.momentum, MomentumRegime)
            else str(regime.momentum)
        ),
        "event_risk": (
            regime.event_risk.value
            if isinstance(regime.event_risk, EventRisk)
            else str(regime.event_risk)
        ),
    }


def format_signal_for_ui(
    symbol: str,
    bias: BiasOutput,
    risk: RiskOutput,
    game: GameOutput,
    regime: RegimeState,
    current_price: Optional[float] = None,
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Format complete signal for frontend.

    This is the main function that combines all layers into the
    frontend-expected format.

    Args:
        symbol: Trading symbol (e.g., "NAS100")
        bias: BiasOutput from Layer 1
        risk: RiskOutput from Layer 2
        game: GameOutput from Layer 3
        regime: RegimeState from regime classifier
        current_price: Current market price
        timestamp: Signal timestamp (defaults to now)

    Returns:
        Complete frontend-ready signal dict
    """
    ts = timestamp or datetime.utcnow()

    return {
        "symbol": symbol,
        "layer1": format_bias_for_ui(bias),
        "layer2": format_risk_for_ui(risk, current_price, bias),
        "layer3": format_game_output_for_ui(game),
        "regime": format_regime_for_ui(regime),
        "created_at": ts.isoformat() + "Z" if not ts.tzinfo else ts.isoformat(),
    }


# Legacy alias for compatibility
format_layer1_for_ui = format_bias_for_ui
format_layer2_for_ui = format_risk_for_ui
format_layer3_for_ui = format_game_output_for_ui
