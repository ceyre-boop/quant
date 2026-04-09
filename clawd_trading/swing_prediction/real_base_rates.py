"""
Real Base Rate Calculator

Computes actual base rates from Yahoo Finance historical data.
No more fabricated probabilities - only real historical performance.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pandas as pd
import numpy as np

from data.providers import DataProvider

logger = logging.getLogger(__name__)


@dataclass
class BaseRateResult:
    """Historical base rate for a setup."""

    setup_name: str
    win_rate: float
    avg_return: float
    max_drawdown: float
    sample_size: int
    confidence_interval: tuple  # (lower, upper)
    lookback_years: int


class BaseRateCalculator:
    """
    Computes base rates from real Yahoo Finance data.

    Usage:
        calc = BaseRateCalculator()
        rate = calc.compute_rate('SPY', lookback_years=10)
    """

    def __init__(self, data_provider: Optional[DataProvider] = None):
        self.data = data_provider or DataProvider()
        self.logger = logging.getLogger(__name__)

    def compute_rate(
        self,
        symbol: str,
        setup_condition: Optional[str] = None,
        lookback_years: int = 10,
        hold_days: int = 20,
    ) -> Optional[BaseRateResult]:
        """
        Compute base rate for a symbol/setup combination.

        Args:
            symbol: Trading symbol
            setup_condition: Optional setup filter (e.g., 'rsi_oversold')
            lookback_years: Years of history to analyze
            hold_days: Days to hold position

        Returns:
            BaseRateResult or None if insufficient data
        """
        try:
            # Fetch historical data
            hist = self.data.get_historical_data(
                symbol, period=f"{lookback_years}y", interval="1d"
            )

            if hist is None or len(hist) < 252:  # Need at least 1 year
                self.logger.warning(f"Insufficient data for {symbol}")
                return None

            # Calculate forward returns
            hist["return"] = hist["close"].pct_change(hold_days).shift(-hold_days)

            # Filter by setup condition if provided
            if setup_condition:
                mask = self._apply_setup_filter(hist, setup_condition)
                trades = hist[mask]["return"].dropna()
            else:
                trades = hist["return"].dropna()

            if len(trades) < 15:  # Minimum sample size
                self.logger.warning(f"Insufficient trades for {symbol}: {len(trades)}")
                return None

            # Compute statistics
            wins = (trades > 0).sum()
            win_rate = wins / len(trades)
            avg_return = trades.mean()
            max_dd = self._calculate_max_drawdown(trades)

            # Confidence interval (95%)
            ci_lower = win_rate - 1.96 * np.sqrt(
                win_rate * (1 - win_rate) / len(trades)
            )
            ci_upper = win_rate + 1.96 * np.sqrt(
                win_rate * (1 - win_rate) / len(trades)
            )

            return BaseRateResult(
                setup_name=setup_condition or f"{symbol}_baseline",
                win_rate=win_rate,
                avg_return=avg_return,
                max_drawdown=max_dd,
                sample_size=len(trades),
                confidence_interval=(max(0, ci_lower), min(1, ci_upper)),
                lookback_years=lookback_years,
            )

        except Exception as e:
            self.logger.error(f"Error computing base rate for {symbol}: {e}")
            return None

    def _apply_setup_filter(self, df: pd.DataFrame, condition: str) -> pd.Series:
        """Apply setup condition to filter trades."""
        if condition == "rsi_oversold":
            # RSI < 30
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi < 30

        # Default: no filter
        return pd.Series([True] * len(df), index=df.index)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
