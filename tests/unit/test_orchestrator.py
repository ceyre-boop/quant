"""Unit tests for orchestrator module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from contracts.types import (
    Direction,
    Magnitude,
    BiasOutput,
    RiskOutput,
    GameOutput,
    PositionState,
    RegimeState,
    AccountState,
    MarketData,
    VolRegime,
    TrendRegime,
    RiskAppetite,
    MomentumRegime,
    EventRisk,
    AdversarialRisk,
)

from orchestrator.daily_lifecycle import (
    DailyLifecycle,
    LifecycleConfig,
    CyclePhase,
    create_default_lifecycle,
)

from orchestrator.state_machine import (
    SymbolStateMachine,
    SymbolContext,
    SymbolState,
    PositionPhase,
    StateMachineManager,
)


class TestLifecycleConfig:
    """Test LifecycleConfig."""

    def test_default_config(self):
        config = LifecycleConfig()
        assert config.pre_market_time == "08:00"
        assert config.market_open_time == "09:30"
        assert config.market_close_time == "16:00"
        assert config.eod_cleanup_time == "16:05"
        assert config.intraday_interval_minutes == 5
        assert config.symbols == ["NAS100", "US30", "SPX500", "XAUUSD"]

    def test_custom_symbols(self):
        config = LifecycleConfig(symbols=["AAPL", "MSFT"])
        assert config.symbols == ["AAPL", "MSFT"]


class TestCyclePhase:
    """Test CyclePhase enum."""

    def test_phase_values(self):
        assert CyclePhase.PRE_MARKET.value == "pre_market"
        assert CyclePhase.OPEN.value == "open"
        assert CyclePhase.REGULAR_HOURS.value == "regular_hours"
        assert CyclePhase.CLOSE.value == "close"
        assert CyclePhase.POST_MARKET.value == "post_market"
        assert CyclePhase.CLOSED.value == "closed"


class TestDailyLifecycle:
    """Test DailyLifecycle."""

    @pytest.fixture
    def mock_broadcaster(self):
        broadcaster = Mock()
        broadcaster.publish_signal = Mock(return_value=True)
        broadcaster.publish_health = Mock(return_value=True)
        broadcaster.publish_controls = Mock(return_value=True)
        return broadcaster

    @pytest.fixture
    def lifecycle(self, mock_broadcaster):
        with patch("orchestrator.daily_lifecycle.FirebaseClient"):
            with patch(
                "orchestrator.daily_lifecycle.FirebaseBroadcaster",
                return_value=mock_broadcaster,
            ):
                config = LifecycleConfig(symbols=["TEST"])
                return DailyLifecycle(config=config)

    def test_init(self, lifecycle):
        assert lifecycle.config is not None
        assert lifecycle._current_phase == CyclePhase.CLOSED
        assert not lifecycle._is_running

    def test_get_current_phase_closed(self, lifecycle):
        # Before pre-market (assume 07:00)
        now = datetime(2024, 1, 15, 7, 0, 0)
        phase = lifecycle.get_current_phase(now)
        assert phase == CyclePhase.CLOSED

    def test_get_current_phase_pre_market(self, lifecycle):
        # During pre-market (08:30)
        now = datetime(2024, 1, 15, 8, 30, 0)
        phase = lifecycle.get_current_phase(now)
        assert phase == CyclePhase.PRE_MARKET

    def test_get_current_phase_regular_hours(self, lifecycle):
        # During regular hours (10:00)
        now = datetime(2024, 1, 15, 10, 0, 0)
        phase = lifecycle.get_current_phase(now)
        assert phase == CyclePhase.REGULAR_HOURS

    def test_get_current_phase_close(self, lifecycle):
        # During close (16:02)
        now = datetime(2024, 1, 15, 16, 2, 0)
        phase = lifecycle.get_current_phase(now)
        assert phase == CyclePhase.CLOSE

    def test_get_current_phase_post_market(self, lifecycle):
        # After EOD cleanup (17:00)
        now = datetime(2024, 1, 15, 17, 0, 0)
        phase = lifecycle.get_current_phase(now)
        assert phase == CyclePhase.POST_MARKET

    def test_register_components(self, lifecycle):
        mock_fetcher = Mock()
        mock_builder = Mock()

        lifecycle.register_components(
            data_fetcher=mock_fetcher, feature_builder=mock_builder
        )

        assert lifecycle.data_fetcher == mock_fetcher
        assert lifecycle.feature_builder == mock_builder

    def test_run_premarket(self, lifecycle, mock_broadcaster):
        # Register mock components
        lifecycle.register_components(
            data_fetcher=lambda s: MarketData(
                symbol=s,
                current_price=100.0,
                bid=99.5,
                ask=100.5,
                spread=1.0,
                volume_24h=1000.0,
                atr_14=2.0,
                timestamp=datetime.now(),
            ),
            feature_builder=lambda s, m: Mock(features={"atr": 2.0}),
            bias_engine=lambda s, f, r: BiasOutput(
                direction=Direction.LONG,
                magnitude=Magnitude.NORMAL,
                confidence=0.75,
                regime_override=False,
                rationale=["TREND"],
                model_version="v1.0",
                feature_snapshot={},
            ),
            risk_engine=lambda s, b, f, a, m: RiskOutput(
                position_size=1.0,
                kelly_fraction=0.25,
                stop_price=98.0,
                stop_method="atr",
                tp1_price=104.0,
                tp2_price=106.0,
                trail_config={},
                expected_value=0.5,
                ev_positive=True,
                size_breakdown={},
            ),
            game_engine=lambda s, m, b: Mock(
                game_state_aligned=True,
                adversarial_risk=AdversarialRisk.LOW,
                game_state_summary="NEUTRAL",
                forced_move_probability=0.5,
                nearest_unswept_pool=None,
                kyle_lambda=0.3,
            ),
            entry_validator=lambda s, b, r, g: None,
        )

        results = lifecycle.run_premarket()

        assert "error" not in results
        mock_broadcaster.publish_health.assert_called()

    def test_run_eod_cleanup(self, lifecycle, mock_broadcaster):
        results = lifecycle.run_eod_cleanup()

        assert "error" not in results
        assert "positions_closed" in results
        mock_broadcaster.publish_health.assert_called()
        mock_broadcaster.publish_controls.assert_called()

    def test_run_intraday_cycle_skipped(self, lifecycle):
        # Set last run to now
        lifecycle._last_intraday_run = datetime.now()

        results = lifecycle.run_intraday_cycle()

        assert results.get("skipped") is True

    def test_run_intraday_cycle_executes(self, lifecycle, mock_broadcaster):
        # Set last run to 10 minutes ago
        lifecycle._last_intraday_run = datetime.now() - timedelta(minutes=10)
        lifecycle.config.intraday_interval_minutes = 5

        lifecycle.register_components(
            data_fetcher=lambda s: MarketData(
                symbol=s,
                current_price=100.0,
                bid=99.5,
                ask=100.5,
                spread=1.0,
                volume_24h=1000.0,
                atr_14=2.0,
                timestamp=datetime.now(),
            ),
            feature_builder=lambda s, m: Mock(features={"atr": 2.0}),
            bias_engine=lambda s, f, r: BiasOutput(
                direction=Direction.LONG,
                magnitude=Magnitude.NORMAL,
                confidence=0.75,
                regime_override=False,
                rationale=["TREND"],
                model_version="v1.0",
                feature_snapshot={},
            ),
            risk_engine=lambda s, b, f, a, m: RiskOutput(
                position_size=1.0,
                kelly_fraction=0.25,
                stop_price=98.0,
                stop_method="atr",
                tp1_price=104.0,
                tp2_price=106.0,
                trail_config={},
                expected_value=0.5,
                ev_positive=True,
                size_breakdown={},
            ),
            game_engine=lambda s, m, b: Mock(
                game_state_aligned=True,
                adversarial_risk=AdversarialRisk.LOW,
                game_state_summary="NEUTRAL",
                forced_move_probability=0.5,
                nearest_unswept_pool=None,
                kyle_lambda=0.3,
            ),
            entry_validator=lambda s, b, r, g: None,
        )

        results = lifecycle.run_intraday_cycle()

        assert "results" in results
        assert "timestamp" in results


class TestSymbolState:
    """Test SymbolState enum."""

    def test_state_values(self):
        assert SymbolState.IDLE.value == "idle"
        assert SymbolState.SCANNING.value == "scanning"
        assert SymbolState.ENTRY_PENDING.value == "entry_pending"
        assert SymbolState.IN_POSITION.value == "in_position"
        assert SymbolState.EXIT_PENDING.value == "exit_pending"
        assert SymbolState.COOLDOWN.value == "cooldown"
        assert SymbolState.ERROR.value == "error"


class TestPositionPhase:
    """Test PositionPhase enum."""

    def test_phase_values(self):
        assert PositionPhase.NONE.value == "none"
        assert PositionPhase.OPEN.value == "open"
        assert PositionPhase.TP1_HIT.value == "tp1_hit"
        assert PositionPhase.TP2_HIT.value == "tp2_hit"
        assert PositionPhase.BE_STOP.value == "be_stop"
        assert PositionPhase.TRAILING.value == "trailing"
        assert PositionPhase.STOPPED.value == "stopped"
        assert PositionPhase.CLOSED.value == "closed"


class TestSymbolContext:
    """Test SymbolContext."""

    def test_init(self):
        ctx = SymbolContext(symbol="NAS100")
        assert ctx.symbol == "NAS100"
        assert ctx.state == SymbolState.IDLE
        assert ctx.position_phase == PositionPhase.NONE

    def test_to_dict(self):
        ctx = SymbolContext(symbol="NAS100")
        data = ctx.to_dict()
        assert data["symbol"] == "NAS100"
        assert data["state"] == "idle"
        assert "updated_at" in data


class TestSymbolStateMachine:
    """Test SymbolStateMachine."""

    @pytest.fixture
    def machine(self):
        return SymbolStateMachine()

    def test_get_context_creates_new(self, machine):
        ctx = machine.get_context("NAS100")
        assert ctx.symbol == "NAS100"
        assert ctx.state == SymbolState.IDLE

    def test_get_context_returns_existing(self, machine):
        ctx1 = machine.get_context("NAS100")
        ctx2 = machine.get_context("NAS100")
        assert ctx1 is ctx2

    def test_can_transition_valid(self, machine):
        # IDLE -> SCANNING is valid
        assert machine.can_transition("NAS100", SymbolState.SCANNING) is True

    def test_can_transition_invalid(self, machine):
        machine.get_context("NAS100").state = SymbolState.IDLE
        # IDLE -> IN_POSITION is not valid directly
        assert machine.can_transition("NAS100", SymbolState.IN_POSITION) is False

    def test_transition_success(self, machine):
        result = machine.transition("NAS100", SymbolState.SCANNING, "test")
        assert result is True

        ctx = machine.get_context("NAS100")
        assert ctx.state == SymbolState.SCANNING
        assert len(ctx.state_history) == 1

    def test_transition_failure(self, machine):
        # Try invalid transition
        machine.get_context("NAS100").state = SymbolState.IDLE
        result = machine.transition("NAS100", SymbolState.IN_POSITION)
        assert result is False

    def test_update_layer_outputs(self, machine):
        bias = BiasOutput(
            direction=Direction.LONG,
            magnitude=Magnitude.NORMAL,
            confidence=0.75,
            regime_override=False,
            rationale=["TREND"],
            model_version="v1.0",
            feature_snapshot={},
        )

        machine.update_layer_outputs("NAS100", bias=bias)

        ctx = machine.get_context("NAS100")
        assert ctx.current_bias == bias

    def test_set_position(self, machine):
        position = PositionState(
            trade_id="T001",
            symbol="NAS100",
            direction=Direction.LONG,
            entry_price=100.0,
            position_size=1.0,
            stop_loss=98.0,
            tp1=104.0,
            tp2=106.0,
            current_price=101.0,
            unrealized_pnl=100.0,
            realized_pnl=0.0,
            status="OPEN",
            opened_at=datetime.now(),
        )

        machine.set_position("NAS100", position)

        ctx = machine.get_context("NAS100")
        assert ctx.current_position == position
        assert ctx.position_phase == PositionPhase.OPEN

    def test_on_entry_signal(self, machine):
        machine.transition("NAS100", SymbolState.SCANNING)

        result = machine.on_entry_signal("NAS100", {"price": 100.0})

        assert result is True
        ctx = machine.get_context("NAS100")
        assert ctx.state == SymbolState.ENTRY_PENDING
        assert ctx.pending_entry == {"price": 100.0}

    def test_on_entry_signal_wrong_state(self, machine):
        # IN_POSITION state should not accept entry signals
        # Must go through proper transitions: IDLE -> SCANNING -> ENTRY_PENDING -> IN_POSITION
        machine.transition("NAS100", SymbolState.SCANNING)
        machine.on_entry_signal("NAS100", {"price": 100.0})  # Goes to ENTRY_PENDING
        machine.on_entry_confirmed(
            "NAS100",
            PositionState(
                trade_id="T001",
                symbol="NAS100",
                direction=Direction.LONG,
                entry_price=100.0,
                position_size=1.0,
                stop_loss=98.0,
                tp1=104.0,
                tp2=106.0,
                current_price=101.0,
                unrealized_pnl=100.0,
                realized_pnl=0.0,
                status="OPEN",
                opened_at=datetime.now(),
            ),
        )  # Now in IN_POSITION

        # Try to send entry signal while already in position
        result = machine.on_entry_signal("NAS100", {"price": 101.0})

        assert result is False

    def test_on_entry_confirmed(self, machine):
        machine.transition("NAS100", SymbolState.SCANNING)
        machine.transition("NAS100", SymbolState.ENTRY_PENDING)

        position = PositionState(
            trade_id="T001",
            symbol="NAS100",
            direction=Direction.LONG,
            entry_price=100.0,
            position_size=1.0,
            stop_loss=98.0,
            tp1=104.0,
            tp2=106.0,
            current_price=101.0,
            unrealized_pnl=100.0,
            realized_pnl=0.0,
            status="OPEN",
            opened_at=datetime.now(),
        )

        machine.on_entry_confirmed("NAS100", position)

        ctx = machine.get_context("NAS100")
        assert ctx.state == SymbolState.IN_POSITION
        assert ctx.current_position == position
        assert ctx.entry_count == 1

    def test_on_exit_confirmed(self, machine):
        # Set up a position first
        machine.transition("NAS100", SymbolState.SCANNING)
        machine.transition("NAS100", SymbolState.ENTRY_PENDING)

        position = PositionState(
            trade_id="T001",
            symbol="NAS100",
            direction=Direction.LONG,
            entry_price=100.0,
            position_size=1.0,
            stop_loss=98.0,
            tp1=104.0,
            tp2=106.0,
            current_price=105.0,
            unrealized_pnl=500.0,
            realized_pnl=0.0,
            status="OPEN",
            opened_at=datetime.now(),
        )
        machine.on_entry_confirmed("NAS100", position)

        # Now exit
        machine.on_exit_confirmed("NAS100", 500.0, 105.0)

        ctx = machine.get_context("NAS100")
        assert ctx.state == SymbolState.COOLDOWN
        assert ctx.current_position is None
        assert ctx.total_trades == 1
        assert ctx.total_pnl == 500.0
        assert ctx.winning_trades == 1

    def test_on_tp1_hit(self, machine):
        machine.transition("NAS100", SymbolState.SCANNING)
        machine.transition("NAS100", SymbolState.IN_POSITION)

        machine.on_tp1_hit("NAS100")

        ctx = machine.get_context("NAS100")
        assert ctx.position_phase == PositionPhase.TP1_HIT

    def test_on_tp2_hit(self, machine):
        machine.transition("NAS100", SymbolState.SCANNING)
        machine.transition("NAS100", SymbolState.IN_POSITION)

        machine.on_tp2_hit("NAS100")

        ctx = machine.get_context("NAS100")
        assert ctx.position_phase == PositionPhase.TP2_HIT

    def test_release_from_cooldown(self, machine):
        # Must go through proper transitions to reach COOLDOWN
        machine.transition("NAS100", SymbolState.SCANNING)
        machine.on_entry_signal("NAS100", {"price": 100.0})
        machine.on_entry_confirmed(
            "NAS100",
            PositionState(
                trade_id="T001",
                symbol="NAS100",
                direction=Direction.LONG,
                entry_price=100.0,
                position_size=1.0,
                stop_loss=98.0,
                tp1=104.0,
                tp2=106.0,
                current_price=101.0,
                unrealized_pnl=100.0,
                realized_pnl=0.0,
                status="OPEN",
                opened_at=datetime.now(),
            ),
        )
        machine.on_exit_confirmed("NAS100", 500.0, 105.0)  # Goes to COOLDOWN

        machine.release_from_cooldown("NAS100")

        ctx = machine.get_context("NAS100")
        assert ctx.state == SymbolState.SCANNING

    def test_set_error(self, machine):
        machine.set_error("NAS100", "Connection failed")

        ctx = machine.get_context("NAS100")
        assert ctx.state == SymbolState.ERROR
        assert ctx.error_message == "Connection failed"

    def test_clear_error(self, machine):
        machine.set_error("NAS100", "Connection failed")
        machine.clear_error("NAS100")

        ctx = machine.get_context("NAS100")
        assert ctx.state == SymbolState.IDLE
        assert ctx.error_message is None

    def test_get_active_positions(self, machine):
        # Create one symbol with position, one without
        # Must go through proper transitions
        machine.transition("NAS100", SymbolState.SCANNING)
        machine.on_entry_signal("NAS100", {"price": 100.0})
        machine.on_entry_confirmed(
            "NAS100",
            PositionState(
                trade_id="T001",
                symbol="NAS100",
                direction=Direction.LONG,
                entry_price=100.0,
                position_size=1.0,
                stop_loss=98.0,
                tp1=104.0,
                tp2=106.0,
                current_price=101.0,
                unrealized_pnl=100.0,
                realized_pnl=0.0,
                status="OPEN",
                opened_at=datetime.now(),
            ),
        )

        machine.transition("US30", SymbolState.SCANNING)

        active = machine.get_active_positions()

        assert len(active) == 1
        assert active[0].symbol == "NAS100"

    def test_get_symbols_in_state(self, machine):
        machine.transition("NAS100", SymbolState.SCANNING)
        machine.transition("US30", SymbolState.SCANNING)
        machine.get_context("SPX500")  # Stay in IDLE

        symbols = machine.get_symbols_in_state(SymbolState.SCANNING)

        assert set(symbols) == {"NAS100", "US30"}

    def test_reset_symbol(self, machine):
        machine.transition("NAS100", SymbolState.SCANNING)
        machine.get_context("NAS100").total_trades = 5

        machine.reset_symbol("NAS100")

        ctx = machine.get_context("NAS100")
        assert ctx.state == SymbolState.IDLE
        assert ctx.total_trades == 0

    def test_get_summary(self, machine):
        machine.transition("NAS100", SymbolState.SCANNING)

        # Put US30 in IN_POSITION through proper transitions
        machine.transition("US30", SymbolState.SCANNING)
        machine.on_entry_signal("US30", {"price": 100.0})
        machine.on_entry_confirmed(
            "US30",
            PositionState(
                trade_id="T001",
                symbol="US30",
                direction=Direction.LONG,
                entry_price=100.0,
                position_size=1.0,
                stop_loss=98.0,
                tp1=104.0,
                tp2=106.0,
                current_price=101.0,
                unrealized_pnl=100.0,
                realized_pnl=0.0,
                status="OPEN",
                opened_at=datetime.now(),
            ),
        )

        machine.get_context("SPX500")  # IDLE (by default)

        summary = machine.get_summary()

        assert summary["total_symbols"] == 3
        assert summary["state_distribution"]["scanning"] == 1
        assert summary["state_distribution"]["in_position"] == 1
        assert summary["state_distribution"]["idle"] == 1

    def test_transition_handler(self, machine):
        handler_called = [False]

        def handler(symbol, from_state, to_state):
            handler_called[0] = True
            assert symbol == "NAS100"
            assert from_state == SymbolState.IDLE
            assert to_state == SymbolState.SCANNING

        machine.register_transition_handler(
            SymbolState.IDLE, SymbolState.SCANNING, handler
        )

        machine.transition("NAS100", SymbolState.SCANNING)

        assert handler_called[0] is True


class TestStateMachineManager:
    """Test StateMachineManager."""

    @pytest.fixture
    def manager(self):
        return StateMachineManager()

    def test_get_machine_creates_new(self, manager):
        machine = manager.get_machine("strategy1")
        assert machine is not None

    def test_get_machine_returns_existing(self, manager):
        machine1 = manager.get_machine("strategy1")
        machine2 = manager.get_machine("strategy1")
        assert machine1 is machine2

    def test_get_different_machines(self, manager):
        machine1 = manager.get_machine("strategy1")
        machine2 = manager.get_machine("strategy2")
        assert machine1 is not machine2

    def test_get_all_machines(self, manager):
        manager.get_machine("strategy1")
        manager.get_machine("strategy2")

        machines = manager.get_all_machines()

        assert len(machines) == 2
        assert "strategy1" in machines
        assert "strategy2" in machines

    def test_reset_machine(self, manager):
        machine = manager.get_machine("strategy1")
        machine.transition("NAS100", SymbolState.SCANNING)

        manager.reset_machine("strategy1")

        ctx = machine.get_context("NAS100")
        assert ctx.state == SymbolState.IDLE

    def test_reset_all(self, manager):
        manager.get_machine("strategy1")
        manager.get_machine("strategy2")

        manager.reset_all()

        # After reset, getting machine creates new one
        machine = manager.get_machine("strategy1")
        ctx = machine.get_context("TEST")
        assert ctx.symbol == "TEST"


class TestCreateDefaultLifecycle:
    """Test create_default_lifecycle factory."""

    def test_creates_lifecycle(self):
        with patch("orchestrator.daily_lifecycle.FirebaseClient"):
            with patch("orchestrator.daily_lifecycle.FirebaseBroadcaster"):
                lifecycle = create_default_lifecycle()
                assert isinstance(lifecycle, DailyLifecycle)
                assert lifecycle.data_fetcher is not None
                assert lifecycle.bias_engine is not None
