"""Metrics Calculator - Financial and ML performance metrics.

Calculates trading performance metrics and model drift detection.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    # Returns
    total_return: float
    avg_daily_return: float

    # Risk
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int

    # Ratios
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_pnl: float

    # Model quality
    model_drift_detected: bool
    prediction_accuracy: float

    # Period
    start_date: str
    end_date: str


def calculate_sharpe(returns: List[float], risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """Calculate annualized Sharpe ratio.

    Args:
        returns: List of period returns (as decimals, e.g., 0.01 for 1%)
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    try:
        # Convert annual risk-free to period rate
        period_rf = risk_free_rate / periods_per_year

        # Calculate excess returns
        excess_returns = [r - period_rf for r in returns]

        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)

        if std_excess == 0:
            return 0.0

        # Annualize
        sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)

        return round(sharpe, 3)

    except Exception as e:
        logger.error(f"Error calculating Sharpe: {e}")
        return 0.0


def calculate_sortino(returns: List[float], risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """Calculate Sortino ratio (uses downside deviation only).

    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    try:
        period_rf = risk_free_rate / periods_per_year
        excess_returns = [r - period_rf for r in returns]

        mean_excess = np.mean(excess_returns)

        # Downside deviation (only negative returns)
        downside_returns = [r for r in excess_returns if r < 0]
        if not downside_returns:
            return float("inf")  # No downside

        downside_std = np.std(downside_returns, ddof=1)
        if downside_std == 0:
            return 0.0

        sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)

        return round(sortino, 3)

    except Exception as e:
        logger.error(f"Error calculating Sortino: {e}")
        return 0.0


def calculate_max_drawdown(equity_curve: List[float]) -> Tuple[float, int]:
    """Calculate maximum drawdown and its duration.

    Args:
        equity_curve: List of equity values over time

    Returns:
        (max_drawdown_pct, max_drawdown_duration_periods)
    """
    if len(equity_curve) < 2:
        return 0.0, 0

    try:
        max_dd = 0.0
        max_dd_duration = 0
        peak = equity_curve[0]
        peak_idx = 0

        for i, value in enumerate(equity_curve):
            if value > peak:
                peak = value
                peak_idx = i

            drawdown = (peak - value) / peak
            duration = i - peak_idx

            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_duration = duration

        return round(max_dd * 100, 2), max_dd_duration

    except Exception as e:
        logger.error(f"Error calculating max drawdown: {e}")
        return 0.0, 0


def calculate_win_rate(trades_pnl: List[float]) -> Tuple[float, int, int]:
    """Calculate win rate from trade P&Ls.

    Args:
        trades_pnl: List of trade P&L values

    Returns:
        (win_rate, winning_trades, total_trades)
    """
    if not trades_pnl:
        return 0.0, 0, 0

    winning = sum(1 for pnl in trades_pnl if pnl > 0)
    total = len(trades_pnl)
    win_rate = winning / total if total > 0 else 0

    return round(win_rate, 3), winning, total


def calculate_profit_factor(trades_pnl: List[float]) -> float:
    """Calculate profit factor (gross wins / gross losses).

    Args:
        trades_pnl: List of trade P&L values

    Returns:
        Profit factor (1.0 = breakeven)
    """
    if not trades_pnl:
        return 0.0

    gross_wins = sum(pnl for pnl in trades_pnl if pnl > 0)
    gross_losses = abs(sum(pnl for pnl in trades_pnl if pnl < 0))

    if gross_losses == 0:
        return float("inf") if gross_wins > 0 else 0.0

    return round(gross_wins / gross_losses, 2)


def detect_model_drift(
    recent_predictions: List[float],
    recent_actuals: List[float],
    baseline_accuracy: float,
    threshold: float = 0.1,
) -> Tuple[bool, float]:
    """Detect if model performance has degraded.

    Args:
        recent_predictions: Model predictions
        recent_actuals: Actual outcomes
        baseline_accuracy: Historical accuracy
        threshold: Drift threshold (e.g., 0.1 = 10% drop)

    Returns:
        (drift_detected, current_accuracy)
    """
    if len(recent_predictions) < 10 or len(recent_actuals) < 10:
        return False, 0.0

    try:
        # Calculate recent accuracy
        correct = sum(1 for p, a in zip(recent_predictions, recent_actuals) if (p > 0.5 and a > 0) or (p <= 0.5 and a <= 0))
        current_accuracy = correct / len(recent_predictions)

        # Check if degraded beyond threshold
        drift_detected = (baseline_accuracy - current_accuracy) > threshold

        if drift_detected:
            logger.warning(
                f"Model drift detected: accuracy dropped from " f"{baseline_accuracy:.2%} to {current_accuracy:.2%}"
            )

        return drift_detected, round(current_accuracy, 3)

    except Exception as e:
        logger.error(f"Error detecting model drift: {e}")
        return False, 0.0


def calculate_calmar(annual_return: float, max_drawdown: float) -> float:
    """Calculate Calmar ratio (annual return / max drawdown).

    Args:
        annual_return: Annual return percentage
        max_drawdown: Max drawdown percentage

    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return 0.0
    return round(abs(annual_return) / abs(max_drawdown), 2)


def calculate_all_metrics(
    trades: List[Dict[str, Any]],
    equity_curve: Optional[List[float]] = None,
    daily_returns: Optional[List[float]] = None,
    period_days: int = 7,
) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics.

    Args:
        trades: List of trade dicts with 'pnl', 'timestamp', etc.
        equity_curve: Optional equity curve
        daily_returns: Optional daily returns
        period_days: Period for metrics calculation

    Returns:
        PerformanceMetrics object
    """
    # Extract P&Ls
    trades_pnl = [t.get("pnl", 0) for t in trades if "pnl" in t]

    if not trades_pnl:
        return PerformanceMetrics(
            total_return=0.0,
            avg_daily_return=0.0,
            volatility=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            avg_trade_pnl=0.0,
            model_drift_detected=False,
            prediction_accuracy=0.0,
            start_date="",
            end_date="",
        )

    # Basic trade stats
    total_pnl = sum(trades_pnl)
    win_rate, wins, total = calculate_win_rate(trades_pnl)
    losses = total - wins

    # Average win/loss
    winning_pnls = [p for p in trades_pnl if p > 0]
    losing_pnls = [p for p in trades_pnl if p < 0]

    avg_win = statistics.mean(winning_pnls) if winning_pnls else 0.0
    avg_loss = statistics.mean(losing_pnls) if losing_pnls else 0.0

    # Profit factor
    pf = calculate_profit_factor(trades_pnl)

    # Average trade
    avg_trade = total_pnl / total if total > 0 else 0.0

    # Build equity curve if not provided
    if equity_curve is None:
        equity_curve = [100000]  # Start with $100k
        for pnl in trades_pnl:
            equity_curve.append(equity_curve[-1] + pnl)

    # Max drawdown
    max_dd, dd_duration = calculate_max_drawdown(equity_curve)

    # Returns
    total_return = ((equity_curve[-1] - equity_curve[0]) / equity_curve[0]) * 100

    # Daily returns for Sharpe
    if daily_returns is None:
        # Approximate from equity curve
        daily_returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            daily_returns.append(ret)

    avg_daily_ret = np.mean(daily_returns) if daily_returns else 0.0
    volatility = np.std(daily_returns, ddof=1) * np.sqrt(252) * 100 if daily_returns else 0.0

    # Sharpe and Sortino
    sharpe = calculate_sharpe(daily_returns)
    sortino = calculate_sortino(daily_returns)

    # Calmar (annualized return / max drawdown)
    annual_return = avg_daily_ret * 252 * 100
    calmar = calculate_calmar(annual_return, max_dd)

    # Model drift (placeholder - would need actual predictions)
    drift_detected = False
    accuracy = 0.0

    # Dates
    start_date = min(t.get("timestamp", datetime.now().isoformat()) for t in trades)
    end_date = max(t.get("timestamp", datetime.now().isoformat()) for t in trades)

    return PerformanceMetrics(
        total_return=round(total_return, 2),
        avg_daily_return=round(avg_daily_ret * 100, 3),
        volatility=round(volatility, 2),
        max_drawdown=max_dd,
        max_drawdown_duration=dd_duration,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        total_trades=total,
        winning_trades=wins,
        losing_trades=losses,
        win_rate=round(win_rate * 100, 1),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        profit_factor=pf,
        avg_trade_pnl=round(avg_trade, 2),
        model_drift_detected=drift_detected,
        prediction_accuracy=accuracy,
        start_date=(start_date[:10] if isinstance(start_date, str) else str(start_date)[:10]),
        end_date=end_date[:10] if isinstance(end_date, str) else str(end_date)[:10],
    )
