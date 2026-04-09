"""Report Generator - Create backtest reports with equity curves and metrics.

Generates comprehensive performance reports from backtest results.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from backtest.backtest_runner import BacktestResult, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    output_dir: str = "backtest_results"
    generate_charts: bool = True
    generate_csv: bool = True
    generate_json: bool = True
    chart_format: str = "png"  # "png", "pdf", "svg"
    dpi: int = 150
    include_monthly_returns: bool = True
    include_trade_list: bool = True


class ReportGenerator:
    """Generate comprehensive backtest reports."""

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize report generator.

        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def generate(
        self, result: BacktestResult, name: Optional[str] = None
    ) -> Dict[str, str]:
        """Generate all report files.

        Args:
            result: Backtest results
            name: Optional report name (defaults to timestamp)

        Returns:
            Dict mapping file types to file paths
        """
        if name is None:
            name = f"backtest_{result.config.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        generated_files = {}

        # Generate JSON report
        if self.config.generate_json:
            json_path = self.output_path / f"{name}.json"
            self._generate_json(result, json_path)
            generated_files["json"] = str(json_path)

        # Generate CSV trade list
        if self.config.generate_csv:
            csv_path = self.output_path / f"{name}_trades.csv"
            self._generate_csv(result, csv_path)
            generated_files["csv"] = str(csv_path)

        # Generate charts
        if self.config.generate_charts and MATPLOTLIB_AVAILABLE:
            chart_path = self.output_path / f"{name}_equity.{self.config.chart_format}"
            self._generate_equity_chart(result, chart_path)
            generated_files["equity_chart"] = str(chart_path)

            # Generate drawdown chart
            dd_chart_path = (
                self.output_path / f"{name}_drawdown.{self.config.chart_format}"
            )
            self._generate_drawdown_chart(result, dd_chart_path)
            generated_files["drawdown_chart"] = str(dd_chart_path)

        # Generate text summary
        summary_path = self.output_path / f"{name}_summary.txt"
        self._generate_text_summary(result, summary_path)
        generated_files["summary"] = str(summary_path)

        logger.info(
            f"Generated {len(generated_files)} report files in {self.output_path}"
        )

        return generated_files

    def _generate_json(self, result: BacktestResult, filepath: Path):
        """Generate JSON report."""
        data = result.to_dict()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"JSON report saved to {filepath}")

    def _generate_csv(self, result: BacktestResult, filepath: Path):
        """Generate CSV trade list."""
        if not result.trades:
            # Create empty file with headers
            with open(filepath, "w") as f:
                f.write(
                    "trade_id,symbol,direction,entry_time,entry_price,exit_time,exit_price,"
                    "position_size,realized_pnl,exit_reason\n"
                )
            return

        # Convert trades to DataFrame
        trades_data = []
        for trade in result.trades:
            trades_data.append(
                {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "direction": trade.direction.name,
                    "entry_time": trade.entry_time,
                    "entry_price": trade.entry_price,
                    "exit_time": trade.exit_time,
                    "exit_price": trade.exit_price,
                    "position_size": trade.position_size,
                    "stop_price": trade.stop_price,
                    "tp1_price": trade.tp1_price,
                    "tp2_price": trade.tp2_price,
                    "realized_pnl": trade.realized_pnl,
                    "commission": trade.commission,
                    "slippage": trade.slippage,
                    "exit_reason": trade.exit_reason,
                }
            )

        df = pd.DataFrame(trades_data)
        df.to_csv(filepath, index=False)

        logger.debug(f"CSV report saved to {filepath}")

    def _generate_equity_chart(self, result: BacktestResult, filepath: Path):
        """Generate equity curve chart."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping chart generation")
            return

        if not result.equity_curve:
            logger.warning("No equity data for chart")
            return

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Parse equity data
        timestamps = []
        equities = []
        for point in result.equity_curve:
            try:
                ts = datetime.fromisoformat(point["timestamp"].replace("Z", "+00:00"))
                timestamps.append(ts)
                equities.append(point["equity"])
            except (KeyError, ValueError, AttributeError):
                continue

        if not timestamps:
            return

        # Plot equity curve
        ax1.plot(timestamps, equities, linewidth=1.5, color="#2E86AB", label="Equity")
        ax1.axhline(
            y=result.config.initial_capital,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Initial Capital",
        )

        # Formatting
        ax1.set_title(
            f"Equity Curve - {result.config.symbol}", fontsize=14, fontweight="bold"
        )
        ax1.set_ylabel("Equity ($)", fontsize=11)
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Plot drawdown
        peak = equities[0]
        drawdowns = []
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            drawdowns.append(dd)

        ax2.fill_between(timestamps, drawdowns, 0, color="#A23B72", alpha=0.3)
        ax2.plot(timestamps, drawdowns, linewidth=1, color="#A23B72")
        ax2.set_ylabel("Drawdown (%)", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Format x-axis for drawdown
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(filepath, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        logger.debug(f"Equity chart saved to {filepath}")

    def _generate_drawdown_chart(self, result: BacktestResult, filepath: Path):
        """Generate standalone drawdown chart."""
        if not MATPLOTLIB_AVAILABLE:
            return

        if not result.equity_curve:
            return

        fig, ax = plt.subplots(figsize=(12, 5))

        # Parse equity data
        timestamps = []
        equities = []
        for point in result.equity_curve:
            try:
                ts = datetime.fromisoformat(point["timestamp"].replace("Z", "+00:00"))
                timestamps.append(ts)
                equities.append(point["equity"])
            except (KeyError, ValueError, AttributeError):
                continue

        if not timestamps:
            return

        # Calculate underwater curve
        peak = equities[0]
        underwater = []
        for eq in equities:
            if eq > peak:
                peak = eq
            underwater.append((peak - eq) / peak * 100 if peak > 0 else 0)

        # Plot
        ax.fill_between(timestamps, underwater, 0, color="#E84855", alpha=0.4)
        ax.plot(timestamps, underwater, linewidth=1.5, color="#E84855")

        # Add max drawdown line
        max_dd = max(underwater) if underwater else 0
        ax.axhline(
            y=max_dd,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Max DD: {max_dd:.1f}%",
        )

        ax.set_title(
            f"Drawdown Analysis - {result.config.symbol}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylabel("Drawdown (%)", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(filepath, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        logger.debug(f"Drawdown chart saved to {filepath}")

    def _generate_text_summary(self, result: BacktestResult, filepath: Path):
        """Generate text summary report."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"BACKTEST REPORT - {result.config.symbol}")
        lines.append("=" * 70)
        lines.append("")
        lines.append("CONFIGURATION:")
        lines.append(f"  Start Date:        {result.config.start_date.date()}")
        lines.append(f"  End Date:          {result.config.end_date.date()}")
        lines.append(f"  Timeframe:         {result.config.timeframe}")
        lines.append(f"  Initial Capital:   ${result.config.initial_capital:,.2f}")
        lines.append(
            f"  Commission:        ${result.config.commission_per_trade:.2f}/trade"
        )
        lines.append("")
        lines.append("-" * 70)
        lines.append("PERFORMANCE METRICS:")
        lines.append("-" * 70)
        lines.append(f"  Total Return:      {result.total_return:+,.2f}%")
        lines.append(f"  Total Trades:      {result.total_trades}")
        lines.append(f"  Winning Trades:    {result.winning_trades}")
        lines.append(f"  Losing Trades:     {result.losing_trades}")
        lines.append(f"  Win Rate:          {result.win_rate*100:.1f}%")
        lines.append(f"  Profit Factor:     {result.profit_factor:.2f}")
        lines.append(f"  Sharpe Ratio:      {result.sharpe_ratio:.2f}")
        lines.append(
            f"  Max Drawdown:      ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.1f}%)"
        )
        lines.append("")

        if result.trades:
            lines.append("-" * 70)
            lines.append("TRADE STATISTICS:")
            lines.append("-" * 70)

            winning = [t for t in result.trades if t.realized_pnl > 0]
            losing = [t for t in result.trades if t.realized_pnl <= 0]

            if winning:
                avg_win = np.mean([t.realized_pnl for t in winning])
                max_win = max([t.realized_pnl for t in winning])
                lines.append(f"  Average Win:       ${avg_win:,.2f}")
                lines.append(f"  Largest Win:       ${max_win:,.2f}")

            if losing:
                avg_loss = np.mean([t.realized_pnl for t in losing])
                max_loss = min([t.realized_pnl for t in losing])
                lines.append(f"  Average Loss:      ${avg_loss:,.2f}")
                lines.append(f"  Largest Loss:      ${max_loss:,.2f}")

            # Exit reasons
            exit_reasons = {}
            for trade in result.trades:
                reason = trade.exit_reason
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

            lines.append("")
            lines.append("EXIT REASONS:")
            for reason, count in sorted(exit_reasons.items()):
                lines.append(f"  {reason}: {count}")

        lines.append("")
        lines.append("=" * 70)
        lines.append(
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append("=" * 70)

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        logger.debug(f"Text summary saved to {filepath}")

    def generate_comparison_report(
        self, results: List[BacktestResult], name: str = "comparison"
    ) -> str:
        """Generate comparison report for multiple backtests.

        Args:
            results: List of backtest results
            name: Report name

        Returns:
            Path to generated report
        """
        filepath = self.output_path / f"{name}.md"

        lines = []
        lines.append(f"# Backtest Comparison Report\n")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("## Summary Table\n")

        # Table header
        lines.append(
            "| Symbol | Return | Trades | Win Rate | Profit Factor | Sharpe | Max DD |"
        )
        lines.append(
            "|--------|--------|--------|----------|---------------|--------|--------|"
        )

        # Table rows
        for result in results:
            lines.append(
                f"| {result.config.symbol} | "
                f"{result.total_return:+.1f}% | "
                f"{result.total_trades} | "
                f"{result.win_rate*100:.1f}% | "
                f"{result.profit_factor:.2f} | "
                f"{result.sharpe_ratio:.2f} | "
                f"{result.max_drawdown_pct:.1f}% |"
            )

        lines.append("\n")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Comparison report saved to {filepath}")
        return str(filepath)


def quick_report(
    result: BacktestResult, output_dir: str = "backtest_results"
) -> Dict[str, str]:
    """Generate a quick report from backtest results.

    Args:
        result: Backtest results
        output_dir: Output directory

    Returns:
        Dict mapping file types to paths
    """
    config = ReportConfig(
        output_dir=output_dir,
        generate_charts=MATPLOTLIB_AVAILABLE,
        generate_csv=True,
        generate_json=True,
    )

    generator = ReportGenerator(config)
    return generator.generate(result)


def print_trade_list(result: BacktestResult, limit: int = 10):
    """Print a formatted trade list to console.

    Args:
        result: Backtest results
        limit: Maximum number of trades to display
    """
    print("\n" + "=" * 100)
    print(f"TRADE LIST - Last {limit} trades")
    print("=" * 100)
    print(
        f"{'ID':<12} {'Dir':<6} {'Entry Time':<20} {'Entry':<10} {'Exit':<10} "
        f"{'PnL':<12} {'Reason':<15}"
    )
    print("-" * 100)

    for trade in result.trades[-limit:]:
        exit_time = (
            trade.exit_time.strftime("%m-%d %H:%M") if trade.exit_time else "OPEN"
        )
        exit_price = f"{trade.exit_price:.2f}" if trade.exit_price else "---"
        pnl = f"${trade.realized_pnl:,.2f}" if trade.exit_price else "---"

        print(
            f"{trade.trade_id:<12} {trade.direction.name:<6} "
            f"{trade.entry_time.strftime('%m-%d %H:%M'):<20} "
            f"{trade.entry_price:<10.2f} {exit_price:<10} "
            f"{pnl:<12} {trade.exit_reason:<15}"
        )

    print("=" * 100 + "\n")
