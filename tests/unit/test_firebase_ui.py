"""Unit tests for Firebase UI writer and broadcaster."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from contracts.types import (
    Direction,
    Magnitude,
    BiasOutput,
    RiskOutput,
    GameOutput,
    RegimeState,
    PositionState,
    AccountState,
    VolRegime,
    TrendRegime,
    RiskAppetite,
    MomentumRegime,
    EventRisk,
    AdversarialRisk,
    FeatureGroup,
)

from integration.firebase_ui_writer import (
    format_signal_for_ui,
    format_bias_for_ui,
    format_risk_for_ui,
    format_game_output_for_ui,
    format_regime_for_ui,
    format_direction,
    format_magnitude,
    format_rationale,
    format_pool_for_ui,
    format_adversarial_risk,
)

from integration.firebase_broadcaster import FirebaseBroadcaster


class TestFormatDirection:
    """Test direction formatting."""

    def test_short_direction(self):
        assert format_direction(Direction.SHORT) == -1

    def test_neutral_direction(self):
        assert format_direction(Direction.NEUTRAL) == 0

    def test_long_direction(self):
        assert format_direction(Direction.LONG) == 1


class TestFormatMagnitude:
    """Test magnitude formatting."""

    def test_small_magnitude(self):
        assert format_magnitude(Magnitude.SMALL) == 1

    def test_normal_magnitude(self):
        assert format_magnitude(Magnitude.NORMAL) == 2

    def test_large_magnitude(self):
        assert format_magnitude(Magnitude.LARGE) == 3


class TestFormatRationale:
    """Test rationale formatting."""

    def test_empty_rationale(self):
        result = format_rationale([])
        assert result == []

    def test_rationale_without_shap(self):
        rationale = ["TREND_STRENGTH", "MOMENTUM_SHIFT"]
        result = format_rationale(rationale)
        assert len(result) == 2
        assert result[0]["group"] == "TREND_STRENGTH"
        assert result[0]["shap"] == "+0.00"

    def test_rationale_with_shap(self):
        rationale = ["TREND_STRENGTH", "MOMENTUM_SHIFT"]
        snapshot = {"shap_values": {"TREND_STRENGTH": 0.31, "MOMENTUM_SHIFT": -0.15}}
        result = format_rationale(rationale, snapshot)
        assert result[0]["shap"] == "+0.31"
        assert result[1]["shap"] == "-0.15"


class TestFormatBiasForUi:
    """Test Layer 1 bias formatting."""

    @pytest.fixture
    def sample_bias(self):
        return BiasOutput(
            direction=Direction.LONG,
            magnitude=Magnitude.NORMAL,
            confidence=0.78,
            regime_override=False,
            rationale=["LIQUIDITY_SWEEP_CONFIRMED", "MOMENTUM_ACCELERATION"],
            model_version="v1.0",
            feature_snapshot={
                "raw_features": {},
                "feature_group_tags": {},
                "regime_at_inference": Mock(),
                "inference_timestamp": "2024-01-15T09:30:00Z",
            },
        )

    def test_format_bias_structure(self, sample_bias):
        result = format_bias_for_ui(sample_bias)
        assert result["direction"] == 1
        assert result["confidence"] == 0.78
        assert result["magnitude"] == 2
        assert len(result["rationale"]) == 2

    def test_format_bias_rationale_groups(self, sample_bias):
        result = format_bias_for_ui(sample_bias)
        assert result["rationale"][0]["group"] == "LIQUIDITY_SWEEP_CONFIRMED"
        assert result["rationale"][1]["group"] == "MOMENTUM_ACCELERATION"


class TestFormatRiskForUi:
    """Test Layer 2 risk formatting."""

    @pytest.fixture
    def sample_risk(self):
        return RiskOutput(
            position_size=1.2,
            kelly_fraction=0.42,
            stop_price=21840,
            stop_method="structural",
            tp1_price=21975,
            tp2_price=22050,
            trail_config={"enabled": True},
            expected_value=0.81,
            ev_positive=True,
            size_breakdown={"base_size": 1.5, "kelly_applied": True},
        )

    def test_format_risk_structure(self, sample_risk):
        result = format_risk_for_ui(sample_risk, current_price=21905)
        assert result["position_size"] == 1.2
        assert result["entry_price"] == 21905
        assert result["stop_price"] == 21840
        assert result["tp1_price"] == 21975
        assert result["tp2_price"] == 22050
        assert result["expected_value"] == 0.81
        assert result["kelly_fraction"] == 0.42
        assert result["stop_method"] == "structural"

    def test_format_risk_long_direction(self, sample_risk):
        bias = BiasOutput(
            direction=Direction.LONG,
            magnitude=Magnitude.NORMAL,
            confidence=0.7,
            regime_override=False,
            rationale=[],
            model_version="v1.0",
            feature_snapshot={},
        )
        result = format_risk_for_ui(sample_risk, bias=bias)
        assert result["entry_price"] is not None

    def test_format_risk_short_direction(self, sample_risk):
        bias = BiasOutput(
            direction=Direction.SHORT,
            magnitude=Magnitude.NORMAL,
            confidence=0.7,
            regime_override=False,
            rationale=[],
            model_version="v1.0",
            feature_snapshot={},
        )
        result = format_risk_for_ui(sample_risk, bias=bias)
        assert result["entry_price"] is not None


class TestFormatPoolForUi:
    """Test liquidity pool formatting."""

    def test_format_equal_lows_pool(self):
        from contracts.types import LiquidityPool

        pool = LiquidityPool(
            price=21820,
            strength=3,
            swept=False,
            age_bars=12,
            draw_probability=0.72,
            pool_type="equal_lows",
        )
        result = format_pool_for_ui(pool)
        assert result["price"] == 21820
        assert result["strength"] == 3
        assert result["direction"] == "below"
        assert result["draw_probability"] == 0.72

    def test_format_equal_highs_pool(self):
        from contracts.types import LiquidityPool

        pool = LiquidityPool(
            price=22000,
            strength=2,
            swept=False,
            age_bars=8,
            draw_probability=0.55,
            pool_type="equal_highs",
        )
        result = format_pool_for_ui(pool)
        assert result["price"] == 22000
        assert result["direction"] == "above"

    def test_format_none_pool(self):
        result = format_pool_for_ui(None)
        assert result is None

    def test_format_dict_pool(self):
        pool = {
            "price": 21800,
            "strength": 4,
            "draw_probability": 0.80,
            "pool_type": "equal_lows",
        }
        result = format_pool_for_ui(pool)
        assert result["price"] == 21800
        assert result["direction"] == "below"


class TestFormatAdversarialRisk:
    """Test adversarial risk formatting."""

    def test_low_risk(self):
        assert format_adversarial_risk(AdversarialRisk.LOW) == "LOW"

    def test_medium_risk(self):
        assert format_adversarial_risk(AdversarialRisk.MEDIUM) == "MEDIUM"

    def test_high_risk(self):
        assert format_adversarial_risk(AdversarialRisk.HIGH) == "HIGH"

    def test_extreme_risk(self):
        assert format_adversarial_risk(AdversarialRisk.EXTREME) == "EXTREME"

    def test_string_risk(self):
        assert format_adversarial_risk("LOW") == "LOW"


class TestFormatGameOutputForUi:
    """Test Layer 3 game output formatting."""

    @pytest.fixture
    def sample_game(self):
        from contracts.types import LiquidityPool, TrappedPositions

        return GameOutput(
            liquidity_map={
                "equal_highs": [],
                "equal_lows": [
                    LiquidityPool(
                        price=21820,
                        strength=3,
                        swept=False,
                        age_bars=12,
                        draw_probability=0.72,
                        pool_type="equal_lows",
                    )
                ],
            },
            nearest_unswept_pool=LiquidityPool(
                price=21820,
                strength=3,
                swept=False,
                age_bars=12,
                draw_probability=0.72,
                pool_type="equal_lows",
            ),
            trapped_positions=TrappedPositions(
                trapped_longs=[],
                trapped_shorts=[{"price": 21950, "size": 100}],
                total_long_pain=0.0,
                total_short_pain=5000.0,
                squeeze_probability=0.61,
            ),
            forced_move_probability=0.61,
            nash_zones=[],
            kyle_lambda=0.34,
            game_state_aligned=True,
            game_state_summary="SHORTS_TRAPPED_SQUEEZE_RISK",
            adversarial_risk=AdversarialRisk.LOW,
        )

    def test_format_game_structure(self, sample_game):
        result = format_game_output_for_ui(sample_game)
        assert result["game_state_aligned"] is True
        assert result["adversarial_risk"] == "LOW"
        assert result["game_state_summary"] == "SHORTS_TRAPPED_SQUEEZE_RISK"
        assert result["forced_move_probability"] == 0.61
        assert result["kyle_lambda"] == 0.34

    def test_format_game_pool(self, sample_game):
        result = format_game_output_for_ui(sample_game)
        assert result["nearest_unswept_pool"] is not None
        assert result["nearest_unswept_pool"]["price"] == 21820
        assert result["nearest_unswept_pool"]["strength"] == 3


class TestFormatRegimeForUi:
    """Test regime state formatting."""

    @pytest.fixture
    def sample_regime(self):
        return RegimeState(
            volatility=VolRegime.NORMAL,
            trend=TrendRegime.STRONG_TREND,
            risk_appetite=RiskAppetite.RISK_ON,
            momentum=MomentumRegime.ACCELERATING,
            event_risk=EventRisk.CLEAR,
            composite_score=0.75,
        )

    def test_format_regime_structure(self, sample_regime):
        result = format_regime_for_ui(sample_regime)
        assert result["volatility"] == "NORMAL"
        assert result["trend"] == "STRONG_TREND"
        assert result["risk_appetite"] == "RISK_ON"
        assert result["momentum"] == "ACCELERATING"
        assert result["event_risk"] == "CLEAR"


class TestFormatSignalForUi:
    """Test complete signal formatting."""

    @pytest.fixture
    def sample_context(self):
        bias = BiasOutput(
            direction=Direction.LONG,
            magnitude=Magnitude.NORMAL,
            confidence=0.78,
            regime_override=False,
            rationale=["LIQUIDITY_SWEEP_CONFIRMED", "MOMENTUM_ACCELERATION"],
            model_version="v1.0",
            feature_snapshot={},
        )

        risk = RiskOutput(
            position_size=1.2,
            kelly_fraction=0.42,
            stop_price=21840,
            stop_method="structural",
            tp1_price=21975,
            tp2_price=22050,
            trail_config={},
            expected_value=0.81,
            ev_positive=True,
            size_breakdown={},
        )

        from contracts.types import LiquidityPool, TrappedPositions

        game = GameOutput(
            liquidity_map={"equal_highs": [], "equal_lows": []},
            nearest_unswept_pool=LiquidityPool(
                price=21820,
                strength=3,
                swept=False,
                age_bars=12,
                draw_probability=0.72,
                pool_type="equal_lows",
            ),
            trapped_positions=TrappedPositions(
                trapped_longs=[],
                trapped_shorts=[],
                total_long_pain=0,
                total_short_pain=0,
                squeeze_probability=0.61,
            ),
            forced_move_probability=0.61,
            nash_zones=[],
            kyle_lambda=0.34,
            game_state_aligned=True,
            game_state_summary="SHORTS_TRAPPED_SQUEEZE_RISK",
            adversarial_risk=AdversarialRisk.LOW,
        )

        regime = RegimeState(
            volatility=VolRegime.NORMAL,
            trend=TrendRegime.STRONG_TREND,
            risk_appetite=RiskAppetite.RISK_ON,
            momentum=MomentumRegime.ACCELERATING,
            event_risk=EventRisk.CLEAR,
            composite_score=0.75,
        )

        return bias, risk, game, regime

    def test_format_complete_signal(self, sample_context):
        bias, risk, game, regime = sample_context
        result = format_signal_for_ui(
            symbol="NAS100",
            bias=bias,
            risk=risk,
            game=game,
            regime=regime,
            current_price=21905,
            timestamp=datetime(2024, 1, 15, 9, 30, 0),
        )

        # Verify top-level structure
        assert result["symbol"] == "NAS100"
        assert "layer1" in result
        assert "layer2" in result
        assert "layer3" in result
        assert "regime" in result
        assert "created_at" in result

        # Verify layer1
        assert result["layer1"]["direction"] == 1
        assert result["layer1"]["confidence"] == 0.78

        # Verify layer2
        assert result["layer2"]["entry_price"] == 21905
        assert result["layer2"]["position_size"] == 1.2

        # Verify layer3
        assert result["layer3"]["game_state_aligned"] is True
        assert result["layer3"]["adversarial_risk"] == "LOW"


class TestFirebaseBroadcaster:
    """Test FirebaseBroadcaster class."""

    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.rtdb = Mock()
        client.db = Mock()
        return client

    @pytest.fixture
    def broadcaster(self, mock_client):
        return FirebaseBroadcaster(firebase_client=mock_client)

    @pytest.fixture
    def sample_regime(self):
        return RegimeState(
            volatility=VolRegime.NORMAL,
            trend=TrendRegime.STRONG_TREND,
            risk_appetite=RiskAppetite.RISK_ON,
            momentum=MomentumRegime.ACCELERATING,
            event_risk=EventRisk.CLEAR,
            composite_score=0.75,
        )

    def test_init_with_client(self, mock_client):
        broadcaster = FirebaseBroadcaster(firebase_client=mock_client)
        assert broadcaster.client == mock_client
        assert broadcaster._enabled is True

    def test_init_without_client(self):
        with patch("integration.firebase_broadcaster.FirebaseClient") as MockClient:
            mock_client = Mock()
            mock_client.rtdb = None
            MockClient.return_value = mock_client
            broadcaster = FirebaseBroadcaster()
            assert broadcaster._enabled is False

    def test_publish_signal(self, broadcaster, mock_client, sample_regime):
        bias = BiasOutput(
            direction=Direction.LONG,
            magnitude=Magnitude.NORMAL,
            confidence=0.78,
            regime_override=False,
            rationale=["TREND_STRENGTH"],
            model_version="v1.0",
            feature_snapshot={},
        )
        risk = RiskOutput(
            position_size=1.0,
            kelly_fraction=0.25,
            stop_price=100.0,
            stop_method="atr",
            tp1_price=110.0,
            tp2_price=120.0,
            trail_config={},
            expected_value=0.5,
            ev_positive=True,
            size_breakdown={},
        )
        from contracts.types import LiquidityPool, TrappedPositions

        game = GameOutput(
            liquidity_map={"equal_highs": [], "equal_lows": []},
            nearest_unswept_pool=None,
            trapped_positions=TrappedPositions(
                trapped_longs=[],
                trapped_shorts=[],
                total_long_pain=0,
                total_short_pain=0,
                squeeze_probability=0.0,
            ),
            forced_move_probability=0.0,
            nash_zones=[],
            kyle_lambda=0.0,
            game_state_aligned=True,
            game_state_summary="NEUTRAL",
            adversarial_risk=AdversarialRisk.LOW,
        )

        result = broadcaster.publish_signal("NAS100", bias, risk, game, sample_regime)
        assert result is True

        # Verify latest ref was called
        mock_client.rtdb.reference.assert_any_call("signals/NAS100/latest")

    def test_publish_regime(self, broadcaster, mock_client, sample_regime):
        result = broadcaster.publish_regime("NAS100", sample_regime)
        assert result is True
        mock_client.rtdb.reference.assert_called_with("system/regime/NAS100")

    def test_publish_health(self, broadcaster, mock_client):
        components = {"data_pipeline": "healthy", "model": "active"}
        result = broadcaster.publish_health("healthy", components)
        assert result is True
        mock_client.rtdb.reference.assert_called_with("system/health")

    def test_publish_position(self, broadcaster, mock_client):
        position = PositionState(
            trade_id="trade_001",
            symbol="NAS100",
            direction=Direction.LONG,
            entry_price=21905,
            position_size=1.2,
            stop_loss=21840,
            tp1=21975,
            tp2=22050,
            current_price=21930,
            unrealized_pnl=300.0,
            realized_pnl=0.0,
            status="OPEN",
            opened_at=datetime.utcnow(),
        )
        result = broadcaster.publish_position(position)
        assert result is True
        mock_client.rtdb.reference.assert_called_with("session/positions/NAS100")

    def test_publish_account(self, broadcaster, mock_client):
        account = AccountState(
            account_id="acc_001",
            equity=50000.0,
            balance=49700.0,
            open_positions=1,
            daily_pnl=300.0,
            daily_loss_pct=0.0,
            margin_used=1000.0,
            margin_available=49000.0,
            timestamp=datetime.utcnow(),
        )
        result = broadcaster.publish_account(account)
        assert result is True

    def test_publish_controls(self, broadcaster, mock_client):
        result = broadcaster.publish_controls(trading_enabled=True, daily_loss_pct=0.0, open_positions=1)
        assert result is True
        mock_client.rtdb.reference.assert_called_with("session/controls")

    def test_update_model_status(self, broadcaster, mock_client):
        result = broadcaster.update_model_status(model_name="bias", version="v1.0", status="active", accuracy=0.75)
        assert result is True
        mock_client.rtdb.reference.assert_called_with("system/models/bias")

    def test_delete_position(self, broadcaster, mock_client):
        mock_ref = Mock()
        mock_client.rtdb.reference.return_value = mock_ref
        result = broadcaster.delete_position("NAS100")
        assert result is True
        mock_ref.delete.assert_called_once()

    def test_get_controls(self, broadcaster, mock_client):
        mock_ref = Mock()
        mock_ref.get.return_value = {"trading_enabled": True}
        mock_client.rtdb.reference.return_value = mock_ref

        result = broadcaster.get_controls()
        assert result == {"trading_enabled": True}

    def test_get_latest_signal(self, broadcaster, mock_client):
        mock_ref = Mock()
        mock_ref.get.return_value = {"symbol": "NAS100", "direction": 1}
        mock_client.rtdb.reference.return_value = mock_ref

        result = broadcaster.get_latest_signal("NAS100")
        assert result["symbol"] == "NAS100"


class TestFirebaseBroadcasterDisabled:
    """Test broadcaster when disabled."""

    @pytest.fixture
    def disabled_broadcaster(self):
        with patch("integration.firebase_broadcaster.FirebaseClient") as MockClient:
            mock_client = Mock()
            mock_client.rtdb = None
            MockClient.return_value = mock_client
            return FirebaseBroadcaster()

    def test_publish_signal_disabled(self, disabled_broadcaster):
        result = disabled_broadcaster.publish_signal("NAS100", Mock(), Mock(), Mock(), Mock())
        assert result is False

    def test_publish_regime_disabled(self, disabled_broadcaster):
        result = disabled_broadcaster.publish_regime("NAS100", Mock())
        assert result is False

    def test_get_controls_disabled(self, disabled_broadcaster):
        result = disabled_broadcaster.get_controls()
        assert result == {}

    def test_get_latest_signal_disabled(self, disabled_broadcaster):
        result = disabled_broadcaster.get_latest_signal("NAS100")
        assert result is None
