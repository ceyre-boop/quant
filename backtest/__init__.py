"""Backtest module - Historical replay and performance evaluation.

Provides tools for backtesting trading strategies with realistic execution simulation.
"""

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

__all__ = [
    # Backtest Runner
    "BacktestRunner",
    "BacktestConfig",
    "BacktestResult",
    "TradeRecord",
    "BacktestMode",
    "run_simple_backtest",
    # Execution Simulator
    "ExecutionSimulator",
    "ExecutionConfig",
    "ExecutionResult",
    "SlippageModel",
    "FillModel",
    "create_realistic_simulator",
    "create_aggressive_simulator",
    "create_conservative_simulator",
    # Report Generator
    "ReportGenerator",
    "ReportConfig",
    "quick_report",
    "print_trade_list",
]
