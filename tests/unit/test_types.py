"""Unit tests for contracts/types.py"""

import pytest
from datetime import datetime

from contracts.types import (
    Direction,
    Magnitude,
    VolRegime,
    TrendRegime,
    RiskAppetite,
    MomentumRegime,
    EventRisk,
    AdversarialRisk,
    FeatureGroup,
    RegimeState,
    BiasOutput,
    RiskOutput,
    LiquidityPool,
    TrappedPositions,
    NashZone,
    GameOutput,
    ThreeLayerContext,
    FeatureRecord,
    FeatureSnapshot,
    EntrySignal,
    PositionState,
    AccountState,
    MarketData,
)


class TestEnums:
    """Test enum definitions."""

    def test_direction_enum(self):
        assert Direction.SHORT.value == -1
        assert Direction.NEUTRAL.value == 0
        assert Direction.LONG.value == 1

    def test_magnitude_enum(self):
        assert Magnitude.SMALL.value == 1
        assert Magnitude.NORMAL.value == 2
        assert Magnitude.LARGE.value == 3

    def test_vol_regime_enum(self):
        assert VolRegime.LOW.value == "LOW"
        assert VolRegime.EXTREME.value == "EXTREME"

    def test_feature_group_enum(self):
        groups = [g.value for g in FeatureGroup]
        assert "VOLATILITY_SPIKE" in groups
        assert "TREND_STRENGTH" in groups


class TestRegimeState:
    """Test RegimeState dataclass."""

    def test_regime_state_creation(self):
        regime = RegimeState(
            volatility=VolRegime.NORMAL,
            trend=TrendRegime.STRONG_TREND,
            risk_appetite=RiskAppetite.RISK_ON,
            momentum=MomentumRegime.ACCELERATING,
            event_risk=EventRisk.CLEAR,
            composite_score=0.75,
        )
        assert regime.volatility == VolRegime.NORMAL
        assert regime.composite_score == 0.75

    def test_regime_state_to_dict(self):
        regime = RegimeState(
            volatility=VolRegime.ELEVATED,
            trend=TrendRegime.RANGING,
            risk_appetite=RiskAppetite.NEUTRAL,
            momentum=MomentumRegime.STEADY,
            event_risk=EventRisk.ELEVATED,
            composite_score=0.5,
        )
        d = regime.to_dict()
        assert d["volatility"] == "ELEVATED"
        assert d["composite_score"] == 0.5

    def test_regime_state_from_dict(self):
        data = {
            "volatility": "LOW",
            "trend": "WEAK_TREND",
            "risk_appetite": "RISK_OFF",
            "momentum": "REVERSING",
            "event_risk": "HIGH",
            "composite_score": 0.25,
        }
        regime = RegimeState.from_dict(data)
        assert regime.volatility == VolRegime.LOW
        assert regime.momentum == MomentumRegime.REVERSING


class TestBiasOutput:
    """Test BiasOutput dataclass."""

    def test_bias_output_creation(self):
        bias = BiasOutput(
            direction=Direction.LONG,
            magnitude=Magnitude.NORMAL,
            confidence=0.75,
            regime_override=False,
            rationale=[FeatureGroup.TREND_STRENGTH.value],
            model_version="v1.0",
            feature_snapshot={},
        )
        assert bias.direction == Direction.LONG
        assert bias.confidence == 0.75
        assert bias.timestamp is not None

    def test_bias_output_to_dict(self):
        bias = BiasOutput(
            direction=Direction.SHORT,
            magnitude=Magnitude.LARGE,
            confidence=0.85,
            regime_override=True,
            rationale=[FeatureGroup.VOLATILITY_SPIKE.value],
            model_version="v1.0",
            feature_snapshot={"raw_features": {}},
        )
        d = bias.to_dict()
        assert d["direction"] == -1
        assert d["confidence"] == 0.85
        assert d["rationale"] == ["VOLATILITY_SPIKE"]


class TestRiskOutput:
    """Test RiskOutput dataclass."""

    def test_risk_output_creation(self):
        risk = RiskOutput(
            position_size=1000.0,
            kelly_fraction=0.15,
            stop_price=19500.0,
            stop_method="atr",
            tp1_price=20000.0,
            tp2_price=20500.0,
            trail_config={"atr_multiple": 1.5},
            expected_value=0.05,
            ev_positive=True,
            size_breakdown={"base_size": 1000, "multipliers": {}},
        )
        assert risk.position_size == 1000.0
        assert risk.ev_positive
        assert risk.timestamp is not None


class TestLayer3Types:
    """Test Layer 3 types."""

    def test_liquidity_pool_creation(self):
        pool = LiquidityPool(
            price=20000.0,
            strength=3,
            swept=False,
            age_bars=10,
            draw_probability=0.75,
            pool_type="equal_highs",
        )
        assert pool.price == 20000.0
        assert not pool.swept

    def test_trapped_positions_creation(self):
        trapped = TrappedPositions(
            trapped_longs=[{"price": 19500, "size": 100}],
            trapped_shorts=[],
            total_long_pain=1000.0,
            total_short_pain=0.0,
            squeeze_probability=0.3,
        )
        assert trapped.squeeze_probability == 0.3

    def test_nash_zone_creation(self):
        zone = NashZone(
            price_level=20000.0,
            zone_type="hvn",
            state="HOLDING",
            test_count=2,
            conviction=0.8,
        )
        assert zone.state == "HOLDING"

    def test_game_output_creation(self):
        pool = LiquidityPool(
            price=20000.0,
            strength=3,
            swept=False,
            age_bars=10,
            draw_probability=0.75,
            pool_type="equal_highs",
        )
        trapped = TrappedPositions(
            trapped_longs=[],
            trapped_shorts=[],
            total_long_pain=0,
            total_short_pain=0,
            squeeze_probability=0,
        )
        game = GameOutput(
            liquidity_map={"equal_highs": [pool], "equal_lows": []},
            nearest_unswept_pool=pool,
            trapped_positions=trapped,
            forced_move_probability=0.3,
            nash_zones=[],
            kyle_lambda=0.5,
            game_state_aligned=True,
            game_state_summary="Neutral",
            adversarial_risk=AdversarialRisk.LOW,
        )
        assert game.game_state_aligned
        assert game.adversarial_risk == AdversarialRisk.LOW


class TestThreeLayerContext:
    """Test ThreeLayerContext - the three-layer agreement gate."""

    @pytest.fixture
    def valid_context(self):
        """Create a valid three-layer context."""
        bias = BiasOutput(
            direction=Direction.LONG,
            magnitude=Magnitude.NORMAL,
            confidence=0.75,
            regime_override=False,
            rationale=[FeatureGroup.TREND_STRENGTH.value],
            model_version="v1.0",
            feature_snapshot={},
        )
        risk = RiskOutput(
            position_size=1000.0,
            kelly_fraction=0.15,
            stop_price=19500.0,
            stop_method="atr",
            tp1_price=20000.0,
            tp2_price=20500.0,
            trail_config={},
            expected_value=0.05,
            ev_positive=True,
            size_breakdown={},
        )
        trapped = TrappedPositions([], [], 0, 0, 0)
        game = GameOutput(
            liquidity_map={"equal_highs": [], "equal_lows": []},
            nearest_unswept_pool=None,
            trapped_positions=trapped,
            forced_move_probability=0.3,
            nash_zones=[],
            kyle_lambda=0.5,
            game_state_aligned=True,
            game_state_summary="Aligned",
            adversarial_risk=AdversarialRisk.LOW,
        )
        regime = RegimeState(
            VolRegime.NORMAL,
            TrendRegime.STRONG_TREND,
            RiskAppetite.RISK_ON,
            MomentumRegime.ACCELERATING,
            EventRisk.CLEAR,
            0.75,
        )
        return ThreeLayerContext(bias, risk, game, regime)

    def test_all_aligned_true(self, valid_context):
        """Test that all_aligned returns True when conditions are met."""
        assert valid_context.all_aligned()

    def test_alignment_fails_on_neutral_bias(self, valid_context):
        """Test that neutral bias blocks alignment."""
        valid_context.bias.direction = Direction.NEUTRAL
        assert not valid_context.all_aligned()
        assert valid_context.block_reason() == "BIAS_NEUTRAL"

    def test_alignment_fails_on_low_confidence(self, valid_context):
        """Test that low confidence blocks alignment."""
        valid_context.bias.confidence = 0.5  # Below 0.55 threshold
        assert not valid_context.all_aligned()
        assert valid_context.block_reason() == "CONFIDENCE_TOO_LOW"

    def test_alignment_fails_on_negative_ev(self, valid_context):
        """Test that negative EV blocks alignment."""
        valid_context.risk.ev_positive = False
        assert not valid_context.all_aligned()
        assert valid_context.block_reason() == "EV_NEGATIVE"

    def test_layer3_veto_extreme_adversarial(self, valid_context):
        """Test Layer 3 veto with EXTREME adversarial risk."""
        valid_context.game.game_state_aligned = False
        valid_context.game.adversarial_risk = AdversarialRisk.EXTREME
        assert not valid_context.all_aligned()
        assert valid_context.block_reason() == "LAYER3_VETO"

    def test_layer3_no_veto_on_aligned(self, valid_context):
        """Test that aligned game state doesn't trigger veto even with EXTREME risk."""
        valid_context.game.game_state_aligned = True
        valid_context.game.adversarial_risk = AdversarialRisk.EXTREME
        # When game is aligned, EXTREME risk alone doesn't veto
        assert valid_context.all_aligned()


class TestFeatureRecord:
    """Test FeatureRecord dataclass."""

    def test_feature_record_creation(self):
        record = FeatureRecord(
            symbol="NAS100",
            timestamp=datetime.utcnow(),
            timeframe="1h",
            features={"feature_1": 1.0, "feature_2": 2.0},
            raw_data={"close": 20000.0},
            is_valid=True,
            validation_errors=[],
            metadata={},
        )
        assert record.symbol == "NAS100"
        assert record.is_valid

    def test_feature_record_to_dict(self):
        ts = datetime.utcnow()
        record = FeatureRecord(
            symbol="NAS100",
            timestamp=ts,
            timeframe="1h",
            features={"f1": 1.0},
            raw_data={},
            is_valid=True,
            validation_errors=[],
            metadata={"source": "polygon"},
        )
        d = record.to_dict()
        assert d["symbol"] == "NAS100"
        assert d["timestamp"] == ts.isoformat()


class TestExecutionTypes:
    """Test execution-related types."""

    def test_entry_signal_creation(self):
        bias = BiasOutput(
            Direction.LONG,
            Magnitude.NORMAL,
            0.75,
            False,
            [],
            "v1",
            {},
            datetime.utcnow(),
        )
        risk = RiskOutput(
            1000,
            0.15,
            19500,
            "atr",
            20000,
            20500,
            {},
            0.05,
            True,
            {},
            datetime.utcnow(),
        )
        trapped = TrappedPositions([], [], 0, 0, 0)
        game = GameOutput(
            {"equal_highs": [], "equal_lows": []},
            None,
            trapped,
            0,
            [],
            0,
            True,
            "",
            AdversarialRisk.LOW,
            datetime.utcnow(),
        )
        regime = RegimeState(
            VolRegime.NORMAL,
            TrendRegime.STRONG_TREND,
            RiskAppetite.RISK_ON,
            MomentumRegime.ACCELERATING,
            EventRisk.CLEAR,
            0.75,
        )
        context = ThreeLayerContext(bias, risk, game, regime)

        signal = EntrySignal(
            symbol="NAS100",
            direction=Direction.LONG,
            entry_price=20000.0,
            position_size=1000.0,
            stop_loss=19500.0,
            tp1=20500.0,
            tp2=21000.0,
            confidence=0.75,
            rationale=[FeatureGroup.TREND_STRENGTH.value],
            timestamp=datetime.utcnow(),
            layer_context=context,
        )
        assert signal.symbol == "NAS100"
        assert signal.direction == Direction.LONG

    def test_position_state_creation(self):
        pos = PositionState(
            trade_id="trade_001",
            symbol="NAS100",
            direction=Direction.SHORT,
            entry_price=20000.0,
            position_size=1000.0,
            stop_loss=20500.0,
            tp1=19500.0,
            tp2=19000.0,
            current_price=19900.0,
            unrealized_pnl=100.0,
            realized_pnl=0.0,
            status="OPEN",
            opened_at=datetime.utcnow(),
        )
        assert pos.status == "OPEN"
        assert pos.trade_id == "trade_001"

    def test_account_state_creation(self):
        account = AccountState(
            account_id="acc_001",
            equity=100000.0,
            balance=95000.0,
            open_positions=1,
            daily_pnl=500.0,
            daily_loss_pct=0.005,
            margin_used=10000.0,
            margin_available=40000.0,
            timestamp=datetime.utcnow(),
        )
        assert account.equity == 100000.0

    def test_market_data_creation(self):
        data = MarketData(
            symbol="NAS100",
            current_price=20000.0,
            bid=19999.5,
            ask=20000.5,
            spread=1.0,
            volume_24h=1000000,
            atr_14=150.0,
            timestamp=datetime.utcnow(),
        )
        assert data.atr_14 == 150.0
