"""Unit tests for backtest module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from contracts.types import (
    Direction,
    BiasOutput,
    RiskOutput,
    GameOutput,
    Magnitude,
    AdversarialRisk,
)

from backtest.backtest_runner import (
    BacktestRunner,
    BacktestConfig,
    BacktestResult,
    TradeRecord,
    BacktestMode,
    run_simple_backtest,
)
from backtest.execution_simulator import (
    ExecutionSimulator,
    ExecutionConfig,
    ExecutionResult,
    SlippageModel,
    FillModel,
    create_realistic_simulator,
    create_aggressive_simulator,
    create_conservative_simulator,
)
from backtest.report_generator import (
    ReportGenerator,
    ReportConfig,
    quick_report,
    print_trade_list,
)


class TestBacktestConfig:
    """Test BacktestConfig."""

    def test_default_config(self):
        config = BacktestConfig(
            symbol="NAS100",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
        )
        assert config.symbol == "NAS100"
        assert config.timeframe == "5m"
        assert config.mode == BacktestMode.EVENT_DRIVEN
        assert config.initial_capital == 50000.0

    def test_string_dates(self):
        config = BacktestConfig(
            symbol="NAS100", start_date="2024-01-01", end_date="2024-01-31"
        )
        assert isinstance(config.start_date, datetime)
        assert config.start_date.year == 2024


class TestTradeRecord:
    """Test TradeRecord."""

    def test_trade_record_creation(self):
        trade = TradeRecord(
            trade_id="T001",
            symbol="NAS100",
            direction=Direction.LONG,
            entry_time=datetime(2024, 1, 15, 10, 0),
            entry_price=21900.0,
            position_size=1.0,
            stop_price=21850.0,
            tp1_price=21950.0,
            tp2_price=22000.0,
        )
        assert trade.trade_id == "T001"
        assert trade.symbol == "NAS100"
        assert trade.direction == Direction.LONG

    def test_trade_record_to_dict(self):
        trade = TradeRecord(
            trade_id="T001",
            symbol="NAS100",
            direction=Direction.LONG,
            entry_time=datetime(2024, 1, 15, 10, 0),
            entry_price=21900.0,
            position_size=1.0,
            stop_price=21850.0,
            tp1_price=21950.0,
            tp2_price=22000.0,
            exit_time=datetime(2024, 1, 15, 14, 0),
            exit_price=21950.0,
            exit_reason="tp1",
            realized_pnl=500.0,
        )
        data = trade.to_dict()
        assert data["trade_id"] == "T001"
        assert data["realized_pnl"] == 500.0


class TestBacktestResult:
    """Test BacktestResult."""

    @pytest.fixture
    def sample_result(self):
        config = BacktestConfig(
            symbol="NAS100",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
        )
        return BacktestResult(config=config)

    def test_calculate_metrics_empty(self, sample_result):
        sample_result.calculate_metrics()
        assert sample_result.total_trades == 0

    def test_calculate_metrics_with_trades(self, sample_result):
        from contracts.types import LiquidityPool, TrappedPositions

        # Add winning trades
        for i in range(3):
            trade = TradeRecord(
                trade_id=f"T{i}",
                symbol="NAS100",
                direction=Direction.LONG,
                entry_time=datetime(2024, 1, i + 1, 10, 0),
                entry_price=21900.0,
                position_size=1.0,
                stop_price=21850.0,
                tp1_price=21950.0,
                tp2_price=22000.0,
                exit_time=datetime(2024, 1, i + 1, 14, 0),
                exit_price=21950.0,
                exit_reason="tp1",
                realized_pnl=500.0,
            )
            sample_result.trades.append(trade)

        # Add losing trades
        for i in range(2):
            trade = TradeRecord(
                trade_id=f"L{i}",
                symbol="NAS100",
                direction=Direction.SHORT,
                entry_time=datetime(2024, 1, i + 10, 10, 0),
                entry_price=21900.0,
                position_size=1.0,
                stop_price=21950.0,
                tp1_price=21850.0,
                tp2_price=21800.0,
                exit_time=datetime(2024, 1, i + 10, 14, 0),
                exit_price=21950.0,
                exit_reason="stop",
                realized_pnl=-500.0,
            )
            sample_result.trades.append(trade)

        sample_result.calculate_metrics()

        assert sample_result.total_trades == 5
        assert sample_result.winning_trades == 3
        assert sample_result.losing_trades == 2
        assert sample_result.win_rate == 0.6
        assert sample_result.avg_win == 500.0
        assert sample_result.avg_loss == -500.0

    def test_to_dict(self, sample_result):
        data = sample_result.to_dict()
        assert "config" in data
        assert "trades" in data
        assert "metrics" in data


class TestBacktestRunner:
    """Test BacktestRunner."""

    @pytest.fixture
    def mock_config(self):
        return BacktestConfig(
            symbol="NAS100",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
            mode=BacktestMode.EVENT_DRIVEN,
        )

    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
        np.random.seed(42)
        base_price = 21900

        returns = np.random.normal(0, 0.001, 100)
        closes = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame(
            {
                "open": closes + np.random.normal(0, 2, 100),
                "high": closes + np.abs(np.random.normal(5, 2, 100)),
                "low": closes - np.abs(np.random.normal(5, 2, 100)),
                "close": closes,
                "volume": np.random.uniform(100000, 500000, 100),
            },
            index=dates,
        )

        return df

    def test_runner_init(self, mock_config):
        runner = BacktestRunner(config=mock_config)
        assert runner.config == mock_config
        assert runner.current_capital == mock_config.initial_capital

    def test_generate_mock_data(self, mock_config):
        runner = BacktestRunner(config=mock_config)
        data = runner._generate_mock_data()

        assert isinstance(data, pd.DataFrame)
        assert "open" in data.columns
        assert "high" in data.columns
        assert "low" in data.columns
        assert "close" in data.columns
        assert len(data) > 0

    def test_run_with_mock_engines(self, mock_config, sample_data):
        def mock_bias(symbol, market_data):
            return BiasOutput(
                direction=Direction.LONG,
                magnitude=Magnitude.NORMAL,
                confidence=0.7,
                regime_override=False,
                rationale=["TREND"],
                model_version="mock",
                feature_snapshot={},
            )

        def mock_risk(symbol, bias, features, account, market_data):
            close = market_data.get("close", 21900)
            return RiskOutput(
                position_size=1.0,
                kelly_fraction=0.25,
                stop_price=close - 50,
                stop_method="atr",
                tp1_price=close + 50,
                tp2_price=close + 100,
                trail_config={},
                expected_value=0.5,
                ev_positive=True,
                size_breakdown={},
            )

        def mock_game(symbol, market_data, bias):
            from contracts.types import LiquidityPool, TrappedPositions

            return GameOutput(
                liquidity_map={"equal_highs": [], "equal_lows": []},
                nearest_unswept_pool=None,
                trapped_positions=TrappedPositions(
                    trapped_longs=[],
                    trapped_shorts=[],
                    total_long_pain=0,
                    total_short_pain=0,
                    squeeze_probability=0.0,
                ),
                forced_move_probability=0.5,
                nash_zones=[],
                kyle_lambda=0.3,
                game_state_aligned=True,
                game_state_summary="NEUTRAL",
                adversarial_risk=AdversarialRisk.LOW,
            )

        runner = BacktestRunner(
            config=mock_config,
            data_loader=lambda s, start, end, tf: sample_data,
            bias_engine=mock_bias,
            risk_engine=mock_risk,
            game_engine=mock_game,
        )

        result = runner.run()

        assert isinstance(result, BacktestResult)
        assert result.config == mock_config
        # Should have recorded equity points
        assert len(result.equity_curve) > 0

    def test_save_results(self, mock_config, tmp_path):
        runner = BacktestRunner(config=mock_config)
        runner.result.trades.append(
            TradeRecord(
                trade_id="T001",
                symbol="NAS100",
                direction=Direction.LONG,
                entry_time=datetime(2024, 1, 1, 10, 0),
                entry_price=21900.0,
                position_size=1.0,
                stop_price=21850.0,
                tp1_price=21950.0,
                tp2_price=22000.0,
                exit_time=datetime(2024, 1, 1, 14, 0),
                exit_price=21950.0,
                exit_reason="tp1",
                realized_pnl=500.0,
            )
        )

        output_file = tmp_path / "test_results.json"
        runner.save_results(str(output_file))

        assert output_file.exists()


class TestRunSimpleBacktest:
    """Test run_simple_backtest function."""

    def test_simple_backtest(self):
        result = run_simple_backtest(
            symbol="NAS100", start_date="2024-01-01", end_date="2024-01-05"
        )

        assert isinstance(result, BacktestResult)
        assert result.config.symbol == "NAS100"


class TestSlippageModel:
    """Test SlippageModel enum."""

    def test_enum_values(self):
        assert SlippageModel.NONE.value == "none"
        assert SlippageModel.FIXED.value == "fixed"
        assert SlippageModel.RANDOM.value == "random"
        assert SlippageModel.MARKET_IMPACT.value == "market_impact"
        assert SlippageModel.VOLATILITY_BASED.value == "volatility_based"


class TestFillModel:
    """Test FillModel enum."""

    def test_enum_values(self):
        assert FillModel.IMMEDIATE.value == "immediate"
        assert FillModel.PROBABILISTIC.value == "probabilistic"
        assert FillModel.PARTIAL.value == "partial"


class TestExecutionConfig:
    """Test ExecutionConfig."""

    def test_default_config(self):
        config = ExecutionConfig()
        assert config.slippage_model == SlippageModel.RANDOM
        assert config.fill_model == FillModel.IMMEDIATE
        assert config.fixed_slippage_pips == 0.5

    def test_custom_config(self):
        config = ExecutionConfig(
            slippage_model=SlippageModel.FIXED, fill_probability=0.9
        )
        assert config.slippage_model == SlippageModel.FIXED
        assert config.fill_probability == 0.9


class TestExecutionSimulator:
    """Test ExecutionSimulator."""

    @pytest.fixture
    def simulator(self):
        return ExecutionSimulator()

    @pytest.fixture
    def market_data(self):
        return {"bid": 21899.0, "ask": 21901.0, "spread": 2.0, "close": 21900.0}

    def test_init(self):
        config = ExecutionConfig(slippage_model=SlippageModel.FIXED)
        simulator = ExecutionSimulator(config)
        assert simulator.config == config

    def test_simulate_entry_long(self, simulator, market_data):
        result = simulator.simulate_entry(
            direction=Direction.LONG,
            desired_price=21900.0,
            position_size=1.0,
            market_data=market_data,
        )

        assert isinstance(result, ExecutionResult)
        assert result.filled is True
        assert result.fill_price is not None
        assert result.fill_size == 1.0

    def test_simulate_entry_short(self, simulator, market_data):
        result = simulator.simulate_entry(
            direction=Direction.SHORT,
            desired_price=21900.0,
            position_size=1.0,
            market_data=market_data,
        )

        assert isinstance(result, ExecutionResult)
        assert result.filled is True

    def test_simulate_exit_long(self, simulator, market_data):
        result = simulator.simulate_exit(
            direction=Direction.LONG,
            entry_price=21900.0,
            desired_price=21950.0,
            position_size=1.0,
            market_data=market_data,
        )

        assert isinstance(result, ExecutionResult)
        assert result.filled is True

    def test_calculate_slippage_none(self):
        config = ExecutionConfig(slippage_model=SlippageModel.NONE)
        simulator = ExecutionSimulator(config)

        slippage = simulator._calculate_slippage(
            Direction.LONG, 1.0, {"close": 100}, None
        )
        assert slippage == 0.0

    def test_calculate_slippage_fixed(self):
        config = ExecutionConfig(
            slippage_model=SlippageModel.FIXED, fixed_slippage_pips=1.0
        )
        simulator = ExecutionSimulator(config)

        slippage = simulator._calculate_slippage(
            Direction.LONG, 1.0, {"close": 100}, None
        )
        assert slippage == 1.0

    def test_calculate_commission(self, simulator):
        commission = simulator._calculate_commission(2.0)
        assert commission == 10.0  # $5 per lot

    def test_simulate_gapping_stop(self, simulator):
        result = simulator.simulate_gapping_stop(
            direction=Direction.LONG,
            stop_price=21800.0,
            gap_price=21750.0,
            position_size=1.0,
        )

        assert result.filled is True
        assert result.fill_price == 21750.0
        assert result.slippage == 50.0

    def test_estimate_market_impact(self, simulator):
        impact = simulator.estimate_market_impact(
            position_size=100, daily_volume=10000, volatility=0.02
        )
        assert impact > 0


class TestSimulatorFactories:
    """Test simulator factory functions."""

    def test_create_realistic_simulator(self):
        sim = create_realistic_simulator()
        assert sim.config.slippage_model == SlippageModel.VOLATILITY_BASED
        assert sim.config.fill_probability == 0.98

    def test_create_aggressive_simulator(self):
        sim = create_aggressive_simulator()
        assert sim.config.slippage_model == SlippageModel.MARKET_IMPACT
        assert sim.config.partial_fill_rate == 0.9

    def test_create_conservative_simulator(self):
        sim = create_conservative_simulator()
        assert sim.config.slippage_model == SlippageModel.FIXED
        assert sim.config.fixed_slippage_pips == 0.2


class TestReportConfig:
    """Test ReportConfig."""

    def test_default_config(self):
        config = ReportConfig()
        assert config.output_dir == "backtest_results"
        assert config.generate_charts is True
        assert config.generate_csv is True
        assert config.chart_format == "png"


class TestReportGenerator:
    """Test ReportGenerator."""

    @pytest.fixture
    def sample_result(self):
        config = BacktestConfig(
            symbol="NAS100",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
        )
        result = BacktestResult(config=config)

        # Add some trades
        for i in range(5):
            trade = TradeRecord(
                trade_id=f"T{i}",
                symbol="NAS100",
                direction=Direction.LONG,
                entry_time=datetime(2024, 1, i + 1, 10, 0),
                entry_price=21900.0,
                position_size=1.0,
                stop_price=21850.0,
                tp1_price=21950.0,
                tp2_price=22000.0,
                exit_time=datetime(2024, 1, i + 1, 14, 0),
                exit_price=21950.0,
                exit_reason="tp1",
                realized_pnl=500.0,
            )
            result.trades.append(trade)

        # Add equity points
        for i in range(10):
            result.equity_curve.append(
                {
                    "timestamp": datetime(2024, 1, 1, i, 0).isoformat(),
                    "equity": 50000 + i * 100,
                }
            )

        result.calculate_metrics()
        return result

    @pytest.fixture
    def generator(self, tmp_path):
        config = ReportConfig(output_dir=str(tmp_path))
        return ReportGenerator(config)

    def test_init(self, tmp_path):
        config = ReportConfig(output_dir=str(tmp_path))
        gen = ReportGenerator(config)
        assert gen.config == config
        assert gen.output_path.exists()

    def test_generate_json(self, generator, sample_result, tmp_path):
        json_path = tmp_path / "test.json"
        generator._generate_json(sample_result, json_path)
        assert json_path.exists()

    def test_generate_csv(self, generator, sample_result, tmp_path):
        csv_path = tmp_path / "test.csv"
        generator._generate_csv(sample_result, csv_path)
        assert csv_path.exists()

        # Verify CSV content
        df = pd.read_csv(csv_path)
        assert len(df) == 5

    def test_generate_text_summary(self, generator, sample_result, tmp_path):
        summary_path = tmp_path / "summary.txt"
        generator._generate_text_summary(sample_result, summary_path)
        assert summary_path.exists()

        content = summary_path.read_text()
        assert "BACKTEST REPORT" in content
        assert "NAS100" in content

    def test_generate_comparison_report(self, generator, sample_result, tmp_path):
        results = [sample_result, sample_result]
        filepath = generator.generate_comparison_report(results, "test_compare")

        assert Path(filepath).exists()
        content = Path(filepath).read_text()
        assert "Comparison Report" in content


class TestQuickReport:
    """Test quick_report function."""

    def test_quick_report(self, tmp_path):
        config = BacktestConfig(
            symbol="NAS100",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
        )
        result = BacktestResult(config=config)

        files = quick_report(result, output_dir=str(tmp_path))

        assert "json" in files
        assert "csv" in files
        assert "summary" in files
        assert Path(files["json"]).exists()


class TestPrintTradeList:
    """Test print_trade_list function."""

    def test_print_trade_list(self, capsys):
        config = BacktestConfig(
            symbol="NAS100",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
        )
        result = BacktestResult(config=config)

        # Add some trades
        for i in range(3):
            trade = TradeRecord(
                trade_id=f"T{i}",
                symbol="NAS100",
                direction=Direction.LONG,
                entry_time=datetime(2024, 1, i + 1, 10, 0),
                entry_price=21900.0,
                position_size=1.0,
                stop_price=21850.0,
                tp1_price=21950.0,
                tp2_price=22000.0,
                exit_time=datetime(2024, 1, i + 1, 14, 0),
                exit_price=21950.0,
                exit_reason="tp1",
                realized_pnl=500.0,
            )
            result.trades.append(trade)

        print_trade_list(result, limit=2)

        captured = capsys.readouterr()
        assert "TRADE LIST" in captured.out
        # print_trade_list shows last N trades, so T1 and T2 (not T0)
        assert "T1" in captured.out or "T2" in captured.out
