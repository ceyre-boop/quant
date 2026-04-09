"""Example: Generate Sample Signal and Write to Firebase

This script demonstrates the complete three-layer pipeline by generating
a sample signal and writing it to Firebase Realtime Database.

Usage:
    python examples/generate_sample_signal.py

Environment Variables Required:
    FIREBASE_PROJECT_ID
    FIREBASE_SERVICE_ACCOUNT_PATH
"""

import os
import sys
import json
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.types import (
    Direction,
    BiasOutput,
    RiskOutput,
    GameOutput,
    RegimeState,
    ThreeLayerContext,
    FeatureSnapshot,
)
from integration.firebase_broadcaster import FirebaseBroadcaster
from firebase.client import FirebaseClient


def create_sample_three_layer_context() -> ThreeLayerContext:
    """Create a sample three-layer context matching the frontend structure."""

    timestamp = datetime.now(timezone.utc)

    # Layer 1: AI Bias
    feature_snapshot = FeatureSnapshot(
        raw_features={"rsi_14": 65.5, "atr_14": 42.3, "adx_14": 28.5},
        feature_group_tags={"trend": "strong", "volatility": "normal"},
        regime_at_inference={"volatility": "NORMAL", "trend": "STRONG_TREND"},
        inference_timestamp=timestamp,
    )

    bias = BiasOutput(
        direction=Direction.LONG,
        magnitude=2,  # 1-3
        confidence=0.78,
        regime_override=False,
        rationale=[
            {"group": "LIQUIDITY_SWEEP_CONFIRMED", "shap": "+0.31"},
            {"group": "MOMENTUM_ACCELERATION", "shap": "+0.18"},
            {"group": "BREAKOUT_CONFIRMED", "shap": "+0.12"},
            {"group": "VOLATILITY_SPIKE", "shap": "-0.08"},
            {"group": "BREADTH_DIVERGENCE", "shap": "-0.04"},
        ],
        model_version="v1.0",
        feature_snapshot=feature_snapshot,
        timestamp=timestamp,
    )

    # Layer 2: Risk
    risk = RiskOutput(
        position_size=1.2,
        kelly_fraction=0.42,
        stop_price=21840.0,
        stop_method="structural",
        tp1_price=21975.0,
        tp2_price=22050.0,
        trail_config={"atr_multiple": 1.5},
        expected_value=0.81,
        ev_positive=True,
        size_breakdown={"kelly": 0.42, "risk_pct": 0.005},
    )

    # Layer 3: Game Theory
    game = GameOutput(
        liquidity_map={"equal_highs": [], "equal_lows": []},
        nearest_unswept_pool={
            "price": 21820,
            "strength": 3,
            "direction": "below",
            "draw_probability": 0.72,
        },
        trapped_positions={"longs_trapped": 0, "shorts_trapped": 150},
        forced_move_probability=0.61,
        nash_zones=[],
        kyle_lambda=0.34,
        game_state_aligned=True,
        game_state_summary="SHORTS_TRAPPED_SQUEEZE_RISK",
        adversarial_risk="LOW",
        timestamp=timestamp,
    )

    regime = RegimeState(
        volatility="NORMAL",
        trend="STRONG_TREND",
        risk_appetite="RISK_ON",
        momentum="ACCELERATING",
        event_risk="CLEAR",
        composite_score=0.72,
    )

    return ThreeLayerContext(bias=bias, risk=risk, game=game, regime=regime)


def main():
    """Generate sample signal and write to Firebase."""
    print("🚀 Clawd Trading - Sample Signal Generator")
    print("=" * 50)

    # Check Firebase credentials
    project_id = os.getenv("FIREBASE_PROJECT_ID", "clawd-trading-7b8de")

    print(f"\n📡 Connecting to Firebase project: {project_id}")

    try:
        # Initialize Firebase
        firebase_client = FirebaseClient()
        broadcaster = FirebaseBroadcaster(firebase_client)

        print("✅ Firebase connected")

        # Create sample context
        print("\n🎯 Creating three-layer signal...")
        context = create_sample_three_layer_context()

        # Verify alignment
        if context.all_aligned():
            print("✅ All three layers aligned - signal is valid")
        else:
            print("⚠️  Signal blocked by three-layer agreement gate")
            return

        # Write to Firebase
        symbol = "NAS100"

        print(f"\n📤 Broadcasting signal to Firebase...")
        print(f"   Symbol: {symbol}")
        print(f"   Direction: {context.bias.direction.name}")
        print(f"   Confidence: {context.bias.confidence:.0%}")
        print(f"   EV: {context.risk.expected_value:.2f}R")

        # Broadcast signal
        broadcaster.broadcast_signal(symbol, context)

        # Set connection status
        broadcaster.set_connection_status("LIVE")

        # Update session controls
        broadcaster.broadcast_session_control(trading_enabled=True, hard_logic_status="CLEAR")

        print("\n✅ Signal successfully written to Firebase!")
        print("\n📊 Data written to:")
        print(f"   - /signals/{symbol}/latest")
        print(f"   - /live_state/{symbol}/")
        print(f"   - /session_controls/")

        print("\n🌐 View your frontend to see the live signal!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check FIREBASE_SERVICE_ACCOUNT_PATH is set correctly")
        print("2. Verify Firebase project ID: clawd-trading-7b8de")
        print("3. Ensure Firebase Realtime Database is enabled")
        raise


if __name__ == "__main__":
    main()
