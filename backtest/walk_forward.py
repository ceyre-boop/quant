"""Walk-Forward Testing - Rolling train/test windows to prevent overfitting.

Implements walk-forward analysis:
- Train: Jan-Jun → Test: Jul
- Train: Feb-Jul → Test: Aug
- etc.

This prevents look-ahead bias and gives realistic out-of-sample performance.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np

from backtest.backtest_runner import (
    BacktestRunner,
    BacktestConfig,
    BacktestResult,
    BacktestMode,
)
from backtest.report_generator import ReportGenerator, ReportConfig

logger = logging.getLogger(__name__)


class WalkForwardMode(Enum):
    """Walk-forward window modes."""

    EXPANDING = "expanding"  # Train window grows over time
    FIXED = "fixed"  # Fixed train window size
    COMPRESSING = "compressing"  # Train window shrinks (focus on recent)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""

    symbol: str
    start_date: datetime
    end_date: datetime
    train_window_months: int = 6
    test_window_months: int = 1
    step_months: int = 1
    mode: WalkForwardMode = WalkForwardMode.FIXED
    min_train_months: int = 3

    def __post_init__(self):
        if isinstance(self.start_date, str):
            self.start_date = datetime.fromisoformat(self.start_date)
        if isinstance(self.end_date, str):
            self.end_date = datetime.fromisoformat(self.end_date)


@dataclass
class WindowResult:
    """Results from a single train/test window."""

    window_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Results
    train_result: Optional[BacktestResult] = None
    test_result: Optional[BacktestResult] = None

    # Model info (if retrained)
    model_version: Optional[str] = None
    model_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_number": self.window_number,
            "train_period": {
                "start": self.train_start.isoformat(),
                "end": self.train_end.isoformat(),
            },
            "test_period": {
                "start": self.test_start.isoformat(),
                "end": self.test_end.isoformat(),
            },
            "train_metrics": (
                self.train_result.to_dict()["metrics"] if self.train_result else None
            ),
            "test_metrics": (
                self.test_result.to_dict()["metrics"] if self.test_result else None
            ),
            "model_version": self.model_version,
        }


class WalkForwardTester:
    """Perform walk-forward optimization and testing."""

    def __init__(
        self,
        config: WalkForwardConfig,
        backtest_config_factory: Optional[Callable] = None,
        model_trainer: Optional[Callable] = None,
    ):
        """Initialize walk-forward tester.

        Args:
            config: Walk-forward configuration
            backtest_config_factory: Function to create BacktestConfig
            model_trainer: Optional function to retrain models per window
        """
        self.config = config
        self.backtest_config_factory = backtest_config_factory
        self.model_trainer = model_trainer

        self.windows: List[WindowResult] = []
        self.aggregated_result: Optional[Dict[str, Any]] = None

        logger.info(f"WalkForwardTester initialized for {config.symbol}")

    def generate_windows(self) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate train/test window pairs.

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []

        current = self.config.start_date
        window_num = 0

        while True:
            # Calculate window boundaries
            if self.config.mode == WalkForwardMode.EXPANDING:
                train_start = self.config.start_date
            else:
                train_start = current

            train_end = train_start + timedelta(
                days=30 * self.config.train_window_months
            )
            test_start = train_end
            test_end = test_start + timedelta(days=30 * self.config.test_window_months)

            # Check if we have enough data
            if test_end > self.config.end_date:
                break

            # Check minimum train period
            train_duration = (train_end - train_start).days / 30
            if train_duration >= self.config.min_train_months:
                windows.append((train_start, train_end, test_start, test_end))
                window_num += 1

            # Move forward by step
            current = current + timedelta(days=30 * self.config.step_months)

        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows

    def run(self, skip_training: bool = False) -> Dict[str, Any]:
        """Execute walk-forward testing.

        Args:
            skip_training: If True, only run test periods (no in-sample)

        Returns:
            Aggregated results across all windows
        """
        logger.info("=" * 60)
        logger.info("STARTING WALK-FORWARD TESTING")
        logger.info("=" * 60)

        windows = self.generate_windows()

        if not windows:
            logger.warning("No windows generated - check date range")
            return {"error": "No windows generated"}

        self.windows = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"\n--- Window {i+1}/{len(windows)} ---")
            logger.info(f"Train: {train_start.date()} to {train_end.date()}")
            logger.info(f"Test:  {test_start.date()} to {test_end.date()}")

            window_result = self._run_window(
                window_num=i + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                skip_training=skip_training,
            )

            self.windows.append(window_result)

        # Aggregate results
        self.aggregated_result = self._aggregate_results()

        logger.info("\n" + "=" * 60)
        logger.info("WALK-FORWARD TESTING COMPLETE")
        logger.info("=" * 60)

        return self.aggregated_result

    def _run_window(
        self,
        window_num: int,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
        skip_training: bool = False,
    ) -> WindowResult:
        """Run a single train/test window.

        Args:
            window_num: Window number
            train_start: Training start date
            train_end: Training end date
            test_start: Test start date
            test_end: Test end date
            skip_training: Skip training period backtest

        Returns:
            WindowResult with train and test results
        """
        result = WindowResult(
            window_number=window_num,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        # Optional: Retrain model on training data
        if self.model_trainer:
            logger.info(f"Training model for window {window_num}...")
            try:
                model_info = self.model_trainer(
                    self.config.symbol, train_start, train_end
                )
                result.model_version = model_info.get("version")
                result.model_metrics = model_info.get("metrics")
            except Exception as e:
                logger.error(f"Model training failed: {e}")

        # Run backtest on training period (in-sample)
        if not skip_training and self.backtest_config_factory:
            try:
                train_config = self.backtest_config_factory(
                    self.config.symbol, train_start, train_end
                )
                train_runner = BacktestRunner(train_config)
                result.train_result = train_runner.run()

                logger.info(
                    f"Train period: {result.train_result.total_trades} trades, "
                    f"{result.train_result.total_return:.2f}% return"
                )
            except Exception as e:
                logger.error(f"Training backtest failed: {e}")

        # Run backtest on test period (out-of-sample)
        if self.backtest_config_factory:
            try:
                test_config = self.backtest_config_factory(
                    self.config.symbol, test_start, test_end
                )
                test_runner = BacktestRunner(test_config)
                result.test_result = test_runner.run()

                logger.info(
                    f"Test period: {result.test_result.total_trades} trades, "
                    f"{result.test_result.total_return:.2f}% return"
                )
            except Exception as e:
                logger.error(f"Test backtest failed: {e}")

        return result

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all windows.

        Returns:
            Dict with aggregated statistics
        """
        if not self.windows:
            return {}

        # Collect test period results (out-of-sample)
        test_returns = []
        test_trades = []
        test_sharpes = []
        test_drawdowns = []

        for window in self.windows:
            if window.test_result:
                test_returns.append(window.test_result.total_return)
                test_trades.append(window.test_result.total_trades)
                test_sharpes.append(window.test_result.sharpe_ratio)
                test_drawdowns.append(window.test_result.max_drawdown_pct)

        if not test_returns:
            return {"error": "No test results to aggregate"}

        return {
            "total_windows": len(self.windows),
            "test_windows": len(test_returns),
            "out_of_sample": {
                "total_return_pct": {
                    "mean": np.mean(test_returns),
                    "std": np.std(test_returns),
                    "min": np.min(test_returns),
                    "max": np.max(test_returns),
                    "sum": np.sum(test_returns),
                },
                "trades_per_window": {
                    "mean": np.mean(test_trades),
                    "total": np.sum(test_trades),
                },
                "sharpe_ratio": {
                    "mean": np.mean(test_sharpes),
                    "std": np.std(test_sharpes),
                },
                "max_drawdown_pct": {
                    "mean": np.mean(test_drawdowns),
                    "max": np.max(test_drawdowns),
                },
            },
            "robustness_score": self._calculate_robustness_score(),
            "windows": [w.to_dict() for w in self.windows],
        }

    def _calculate_robustness_score(self) -> float:
        """Calculate a robustness score based on consistency across windows.

        Returns:
            Robustness score (0-100)
        """
        if not self.windows:
            return 0.0

        test_returns = [
            w.test_result.total_return for w in self.windows if w.test_result
        ]

        if len(test_returns) < 2:
            return 50.0  # Neutral score for insufficient data

        # Score based on:
        # 1. Consistency (low std dev)
        # 2. Positive mean return
        # 3. Win rate across windows

        consistency_score = max(0, 100 - np.std(test_returns) * 10)
        positive_return_score = 100 if np.mean(test_returns) > 0 else 0
        win_rate_score = (
            sum(1 for r in test_returns if r > 0) / len(test_returns)
        ) * 100

        return (
            consistency_score * 0.3 + positive_return_score * 0.3 + win_rate_score * 0.4
        )

    def generate_report(self, output_dir: str = "walkforward_results") -> str:
        """Generate comprehensive walk-forward report.

        Args:
            output_dir: Directory for output files

        Returns:
            Path to report file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save aggregated results
        results_file = output_path / f"wf_results_{self.config.symbol}.json"
        with open(results_file, "w") as f:
            json.dump(self.aggregated_result, f, indent=2, default=str)

        # Generate markdown report
        report_file = output_path / f"wf_report_{self.config.symbol}.md"

        lines = []
        lines.append(f"# Walk-Forward Analysis Report: {self.config.symbol}\n")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**Configuration:**\n")
        lines.append(f"- Mode: {self.config.mode.value}\n")
        lines.append(f"- Train Window: {self.config.train_window_months} months\n")
        lines.append(f"- Test Window: {self.config.test_window_months} months\n")
        lines.append(f"- Step: {self.config.step_months} month(s)\n")
        lines.append(f"\n")

        # Summary statistics
        if self.aggregated_result and "out_of_sample" in self.aggregated_result:
            oos = self.aggregated_result["out_of_sample"]
            lines.append("## Out-of-Sample Performance\n")
            lines.append(
                f"- **Total Windows:** {self.aggregated_result['total_windows']}\n"
            )
            lines.append(
                f"- **Robustness Score:** {self.aggregated_result['robustness_score']:.1f}/100\n"
            )
            lines.append(f"\n")

            lines.append("### Returns\n")
            returns = oos["total_return_pct"]
            lines.append(f"- Mean: {returns['mean']:.2f}%\n")
            lines.append(f"- Std Dev: {returns['std']:.2f}%\n")
            lines.append(f"- Min: {returns['min']:.2f}%\n")
            lines.append(f"- Max: {returns['max']:.2f}%\n")
            lines.append(f"- Total: {returns['sum']:.2f}%\n")
            lines.append(f"\n")

            lines.append("### Risk Metrics\n")
            lines.append(f"- Mean Sharpe: {oos['sharpe_ratio']['mean']:.2f}\n")
            lines.append(f"- Mean Max DD: {oos['max_drawdown_pct']['mean']:.1f}%\n")
            lines.append(f"\n")

        # Window details
        lines.append("## Window Results\n")
        lines.append("| Window | Train Period | Test Period | Test Return | Trades |\n")
        lines.append("|--------|--------------|-------------|-------------|--------|\n")

        for window in self.windows:
            train_period = f"{window.train_start.strftime('%Y-%m')} to {window.train_end.strftime('%Y-%m')}"
            test_period = f"{window.test_start.strftime('%Y-%m')} to {window.test_end.strftime('%Y-%m')}"
            test_return = (
                f"{window.test_result.total_return:.2f}%"
                if window.test_result
                else "N/A"
            )
            trades = (
                str(window.test_result.total_trades) if window.test_result else "N/A"
            )

            lines.append(
                f"| {window.window_number} | {train_period} | {test_period} | {test_return} | {trades} |\n"
            )

        with open(report_file, "w") as f:
            f.writelines(lines)

        logger.info(f"Walk-forward report saved to {report_file}")
        return str(report_file)

    def print_summary(self):
        """Print walk-forward summary to console."""
        if not self.aggregated_result:
            print("No results to display. Run walk-forward first.")
            return

        print("\n" + "=" * 70)
        print(f"WALK-FORWARD ANALYSIS: {self.config.symbol}")
        print("=" * 70)
        print(
            f"Configuration: {self.config.train_window_months}mo train, "
            f"{self.config.test_window_months}mo test, "
            f"{self.config.step_months}mo step"
        )
        print(f"Total Windows: {self.aggregated_result['total_windows']}")
        print(f"Robustness Score: {self.aggregated_result['robustness_score']:.1f}/100")
        print()

        if "out_of_sample" in self.aggregated_result:
            oos = self.aggregated_result["out_of_sample"]
            returns = oos["total_return_pct"]
            print("OUT-OF-SAMPLE PERFORMANCE:")
            print(f"  Mean Return:    {returns['mean']:+,.2f}% ± {returns['std']:.2f}%")
            print(f"  Range:          {returns['min']:+.2f}% to {returns['max']:+.2f}%")
            print(f"  Total Return:   {returns['sum']:+.2f}%")
            print(f"  Avg Sharpe:     {oos['sharpe_ratio']['mean']:.2f}")
            print(f"  Avg Max DD:     {oos['max_drawdown_pct']['mean']:.1f}%")

        print("=" * 70 + "\n")


def create_default_walkforward(
    symbol: str,
    start_date: str,
    end_date: str,
    train_months: int = 6,
    test_months: int = 1,
) -> WalkForwardTester:
    """Create a walk-forward tester with default settings.

    Args:
        symbol: Trading symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        train_months: Training window in months
        test_months: Test window in months

    Returns:
        Configured WalkForwardTester
    """
    config = WalkForwardConfig(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        train_window_months=train_months,
        test_window_months=test_months,
        step_months=test_months,
        mode=WalkForwardMode.FIXED,
    )

    def config_factory(symbol, start, end):
        return BacktestConfig(
            symbol=symbol,
            start_date=start,
            end_date=end,
            mode=BacktestMode.EVENT_DRIVEN,
        )

    return WalkForwardTester(config, backtest_config_factory=config_factory)


def run_quick_walkforward(
    symbol: str, start_date: str, end_date: str
) -> Dict[str, Any]:
    """Run a quick walk-forward analysis.

    Args:
        symbol: Trading symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Aggregated results dict
    """
    tester = create_default_walkforward(symbol, start_date, end_date)
    results = tester.run()
    tester.print_summary()
    return results
