"""Generate Sample Trading Signal - Demo Script

This script demonstrates the full Clawd Trading pipeline.
Usage: python examples/generate_sample_signal.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.types import (
    Direction, Magnitude, VolRegime, TrendRegime,
    RiskAppetite, MomentumRegime, EventRisk,
    RegimeState, FeatureSnapshot,
    BiasOutput, RiskOutput, GameOutput,
    LiquidityPool, TrappedPositions, NashZone, AdversarialRisk
)


def create_mock_ohlcv_data():
    return {
        'symbol': 'NAS100',
        'current_price': 21905.00,
    }


def build_features(ohlcv_data):
    print("\n[Layer 1] Building Features...")
    
    features = {
        'returns_1h': 0.0025,
        'returns_4h': 0.0085,
        'returns_daily': 0.015,
        'returns_5d': 0.032,
        'price_vs_sma_20': 0.025,
        'price_vs_sma_50': 0.045,
        'price_vs_ema_12': 0.018,
        'atr_14': 65.5,
        'atr_percent_14': 0.003,
        'bollinger_position': 0.75,
        'historical_volatility_20d': 0.145,
        'realized_volatility_5d': 0.125,
        'adx_14': 32.5,
        'dmi_plus': 35.0,
        'dmi_minus': 15.0,
        'rsi_14': 62.5,
        'macd_line': 45.2,
        'macd_histogram': 6.7,
        'volume_sma_ratio': 1.15,
        'vwap_deviation': 0.0085,
        'vix_level': 16.5,
        'market_breadth_ratio': 1.85,
        'swing_high_20': 22050.0,
        'swing_low_20': 21680.0,
        'distance_to_resistance': 145.0,
        'distance_to_support': 225.0,
    }
    
    print(f"  Built {len(features)} features")
    return features


def classify_regime(features):
    print("\n[Regime] Classifying...")
    
    vol_regime = VolRegime.NORMAL if features['vix_level'] < 20 else VolRegime.ELEVATED
    trend_regime = TrendRegime.STRONG_TREND if features['adx_14'] > 25 else TrendRegime.RANGING
    risk_appetite = RiskAppetite.RISK_ON if features['market_breadth_ratio'] > 1.5 else RiskAppetite.NEUTRAL
    momentum = MomentumRegime.ACCELERATING if features['macd_histogram'] > 0 else MomentumRegime.STEADY
    event_risk = EventRisk.CLEAR
    
    regime = RegimeState(
        volatility=vol_regime,
        trend=trend_regime,
        risk_appetite=risk_appetite,
        momentum=momentum,
        event_risk=event_risk,
        composite_score=0.72
    )
    
    print(f"  Volatility: {vol_regime.value}")
    print(f"  Trend: {trend_regime.value}")
    print(f"  Risk Appetite: {risk_appetite.value}")
    return regime


def generate_bias(features, regime, current_price):
    print("\n[Layer 1] Generating Bias...")
    
    timestamp = datetime.now(timezone.utc)
    
    feature_snapshot = FeatureSnapshot(
        raw_features=features,
        feature_group_tags={
            'LIQUIDITY_SWEEP_CONFIRMED': True,
            'MOMENTUM_ACCELERATION': True,
            'BREAKOUT_CONFIRMED': True,
        },
        regime_at_inference={
            'volatility': regime.volatility.value,
            'trend': regime.trend.value
        },
        inference_timestamp=timestamp
    )
    
    rationale = [
        {'group': 'LIQUIDITY_SWEEP_CONFIRMED', 'shap': '+0.31'},
        {'group': 'MOMENTUM_ACCELERATION', 'shap': '+0.18'},
        {'group': 'BREAKOUT_CONFIRMED', 'shap': '+0.12'},
        {'group': 'VOLATILITY_SPIKE', 'shap': '-0.08'},
        {'group': 'BREADTH_DIVERGENCE', 'shap': '-0.04'},
    ]
    
    bias = BiasOutput(
        direction=Direction.LONG,
        magnitude=Magnitude.NORMAL,
        confidence=0.78,
        regime_override=False,
        rationale=['LIQUIDITY_SWEEP_CONFIRMED', 'MOMENTUM_ACCELERATION', 'BREAKOUT_CONFIRMED'],
        model_version='v1.0',
        feature_snapshot=feature_snapshot,
        timestamp=timestamp
    )
    
    print(f"  Direction: LONG")
    print(f"  Confidence: {bias.confidence:.0%}")
    print(f"  Magnitude: {bias.magnitude.name}")
    return bias, rationale


def calculate_risk(features, bias, current_price):
    print("\n[Layer 2] Calculating Risk...")
    
    timestamp = datetime.now(timezone.utc)
    
    kelly_fraction = 0.42
    position_size = 1.2
    
    entry_price = current_price
    stop_price = current_price - 65
    tp1_price = current_price + 70
    tp2_price = current_price + 145
    
    win_prob = 0.65
    win_amount = tp1_price - entry_price
    loss_amount = entry_price - stop_price
    expected_value = (win_prob * win_amount) - ((1 - win_prob) * loss_amount)
    expected_value_pct = expected_value / entry_price
    
    risk = RiskOutput(
        timestamp=timestamp,
        position_size=position_size,
        kelly_fraction=kelly_fraction,
        stop_price=stop_price,
        stop_method='structural',
        tp1_price=tp1_price,
        tp2_price=tp2_price,
        trail_config={'atr_multiple': 1.5},
        expected_value=expected_value_pct,
        ev_positive=expected_value_pct > 0,
        size_breakdown={
            'base_size': position_size,
            'kelly_adjusted': position_size * kelly_fraction,
        }
    )
    
    print(f"  Position Size: {position_size} lots")
    print(f"  Stop: {stop_price:.0f}, TP1: {tp1_price:.0f}, TP2: {tp2_price:.0f}")
    print(f"  EV: {expected_value_pct:.2%}, Kelly: {kelly_fraction:.0%}")
    return risk


def analyze_game(features, bias, current_price):
    print("\n[Layer 3] Game Analysis...")
    
    timestamp = datetime.now(timezone.utc)
    
    liquidity_pool = LiquidityPool(
        price=21820.0,
        strength=3,
        swept=False,
        age_bars=12,
        draw_probability=0.72,
        pool_type='equal_lows'
    )
    
    liquidity_map = {
        'equal_highs': [],
        'equal_lows': [21820.0]
    }
    
    trapped = TrappedPositions(
        trapped_longs=[],
        trapped_shorts=[{'price': 21950, 'size': 150}],
        total_long_pain=0.0,
        total_short_pain=150.0,
        squeeze_probability=0.72
    )
    
    nash_zones = [
        NashZone(
            price_level=21900,
            zone_type='hvn',
            state='HOLDING',
            test_count=3,
            conviction=0.8
        )
    ]
    
    game = GameOutput(
        liquidity_map=liquidity_map,
        nearest_unswept_pool=liquidity_pool,
        trapped_positions={'shorts_above': 150},
        forced_move_probability=0.61,
        nash_zones=nash_zones,
        kyle_lambda=0.0034,
        game_state_aligned=True,
        game_state_summary='SHORTS_TRAPPED_SQUEEZE_RISK',
        adversarial_risk=AdversarialRisk.LOW,
        timestamp=timestamp
    )
    
    print(f"  Game State: ALIGNED")
    print(f"  Kyle Lambda: {game.kyle_lambda:.4f}")
    print(f"  Adversarial Risk: {game.adversarial_risk.value}")
    print(f"  Forced Move Prob: {game.forced_move_probability:.0%}")
    return game


def run_entry_gates(bias, risk, game, regime):
    print("\n[Entry Engine] 12-Gate Validation...")
    print("="*50)
    
    gates = {
        'G1_regime_volatility': regime.volatility != VolRegime.EXTREME,
        'G2_regime_event': regime.event_risk != EventRisk.EXTREME,
        'G3_bias_direction': bias.direction != Direction.NEUTRAL,
        'G4_bias_confidence': bias.confidence >= 0.55,
        'G5_bias_magnitude': bias.magnitude.value >= Magnitude.SMALL.value,
        'G6_risk_ev_positive': risk.ev_positive,
        'G7_risk_kelly_sane': 0.01 <= risk.kelly_fraction <= 0.50,
        'G8_game_aligned': game.game_state_aligned,
        'G9_game_adversarial': game.adversarial_risk not in [AdversarialRisk.HIGH, AdversarialRisk.EXTREME],
        'G10_liquidity_swept': game.nearest_unswept_pool is not None,
        'G11_nash_holding': any(z.state == 'HOLDING' for z in game.nash_zones),
        'G12_all_previous': True,
    }
    
    passed = sum(gates.values())
    failed = [k for k, v in gates.items() if not v]
    
    for gate, result in gates.items():
        status = "PASS" if result else "FAIL"
        print(f"  {gate}: {status}")
    
    print("="*50)
    print(f"GATES: {passed}/12 passed")
    
    if failed:
        print(f"Failed: {', '.join(failed)}")
    else:
        print("All gates passed!")
    
    return gates, passed == 12


def create_output_json(symbol, bias, rationale, risk, game, regime, gates, all_passed):
    timestamp = datetime.now(timezone.utc)
    
    output = {
        'id': f"demo_{timestamp.strftime('%Y%m%d_%H%M%S')}",
        'symbol': symbol,
        'ict_model_used': 'A',
        'created_at': timestamp.isoformat(),
        'layer1': {
            'direction': bias.direction.value,
            'magnitude': bias.magnitude.value,
            'confidence': bias.confidence,
            'rationale': rationale
        },
        'layer2': {
            'position_size': risk.position_size,
            'entry_price': 21905,
            'stop_price': risk.stop_price,
            'tp1_price': risk.tp1_price,
            'tp2_price': risk.tp2_price,
            'expected_value': round(risk.expected_value, 2),
            'kelly_fraction': risk.kelly_fraction,
            'stop_method': risk.stop_method
        },
        'layer3': {
            'game_state_aligned': game.game_state_aligned,
            'adversarial_risk': game.adversarial_risk.value,
            'game_state_summary': game.game_state_summary,
            'forced_move_probability': game.forced_move_probability,
            'nearest_unswept_pool': {
                'price': game.nearest_unswept_pool.price,
                'strength': game.nearest_unswept_pool.strength,
                'draw_probability': game.nearest_unswept_pool.draw_probability,
                'pool_type': game.nearest_unswept_pool.pool_type
            },
            'kyle_lambda': game.kyle_lambda
        },
        'regime': {
            'volatility': regime.volatility.value,
            'trend': regime.trend.value,
            'risk_appetite': regime.risk_appetite.value,
            'momentum': regime.momentum.value,
            'event_risk': regime.event_risk.value
        },
        'entry_gates': {
            'total': 12,
            'passed': sum(gates.values()),
            'failed': [k for k, v in gates.items() if not v],
            'all_passed': all_passed
        }
    }
    
    return output


def main():
    print("="*60)
    print("CLAWD TRADING - SAMPLE SIGNAL GENERATION")
    print("="*60)
    
    ohlcv_data = create_mock_ohlcv_data()
    symbol = ohlcv_data['symbol']
    current_price = ohlcv_data['current_price']
    print(f"\nMock data: {symbol} @ ${current_price:,.2f}")
    
    features = build_features(ohlcv_data)
    regime = classify_regime(features)
    bias, rationale = generate_bias(features, regime, current_price)
    risk = calculate_risk(features, bias, current_price)
    game = analyze_game(features, bias, current_price)
    gates, all_passed = run_entry_gates(bias, risk, game, regime)
    
    output = create_output_json(symbol, bias, rationale, risk, game, regime, gates, all_passed)
    
    output_path = Path(__file__).parent / 'sample_signal_output.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*60)
    print("OUTPUT SAVED")
    print("="*60)
    print(f"File: {output_path}")
    
    print("\nJSON Output:")
    print(json.dumps(output, indent=2))
    
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    
    layer1_agrees = bias.direction != Direction.NEUTRAL and bias.confidence >= 0.55
    layer2_agrees = risk.ev_positive
    layer3_agrees = game.game_state_aligned
    
    print(f"\nLayer 1 (Bias): {'PASS' if layer1_agrees else 'FAIL'}")
    print(f"  Direction: {bias.direction.name}, Confidence: {bias.confidence:.0%}")
    
    print(f"\nLayer 2 (Risk): {'PASS' if layer2_agrees else 'FAIL'}")
    print(f"  EV: {risk.expected_value:.2%}, Kelly: {risk.kelly_fraction:.0%}")
    
    print(f"\nLayer 3 (Game): {'PASS' if layer3_agrees else 'FAIL'}")
    print(f"  Aligned: {game.game_state_aligned}, Risk: {game.adversarial_risk.value}")
    
    print(f"\nEntry Gates: {sum(gates.values())}/12 passed")
    
    if all([layer1_agrees, layer2_agrees, layer3_agrees, all_passed]):
        print("\n*** SIGNAL VALID FOR TRADING ***")
    else:
        print("\n*** SIGNAL BLOCKED ***")
    
    print("\n" + "="*60)
    return output


if __name__ == '__main__':
    main()
