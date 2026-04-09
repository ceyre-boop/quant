"""Execution Simulator - Realistic trade execution simulation.

Simulates slippage, fill rates, and market impact for realistic backtesting.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

from contracts.types import Direction

logger = logging.getLogger(__name__)


class SlippageModel(Enum):
    """Slippage calculation models."""

    NONE = "none"
    FIXED = "fixed"
    RANDOM = "random"
    MARKET_IMPACT = "market_impact"
    VOLATILITY_BASED = "volatility_based"


class FillModel(Enum):
    """Order fill models."""

    IMMEDIATE = "immediate"  # Always fills at desired price
    PROBABILISTIC = "probabilistic"  # Probabilistic fill based on liquidity
    PARTIAL = "partial"  # May fill partially


@dataclass
class ExecutionConfig:
    """Configuration for execution simulation."""

    slippage_model: SlippageModel = SlippageModel.RANDOM
    fill_model: FillModel = FillModel.IMMEDIATE
    fixed_slippage_pips: float = 0.5
    max_slippage_pips: float = 2.0
    slippage_std_dev: float = 0.5
    fill_probability: float = 0.95  # For probabilistic fill
    partial_fill_rate: float = 1.0  # For partial fill model
    market_impact_factor: float = 0.1  # Price impact per lot
    volatility_lookback: int = 20


@dataclass
class ExecutionResult:
    """Result of simulated execution."""

    filled: bool
    fill_price: Optional[float]
    fill_size: float
    slippage: float
    slippage_pips: float
    commission: float
    remaining_size: float
    reject_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filled": self.filled,
            "fill_price": self.fill_price,
            "fill_size": self.fill_size,
            "slippage": self.slippage,
            "slippage_pips": self.slippage_pips,
            "commission": self.commission,
            "remaining_size": self.remaining_size,
            "reject_reason": self.reject_reason,
        }


class ExecutionSimulator:
    """Simulates realistic trade execution with slippage and fill uncertainty."""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Initialize execution simulator.

        Args:
            config: Execution configuration
        """
        self.config = config or ExecutionConfig()

    def simulate_entry(
        self,
        direction: Direction,
        desired_price: float,
        position_size: float,
        market_data: Dict[str, Any],
        atr: Optional[float] = None,
    ) -> ExecutionResult:
        """Simulate entry execution.

        Args:
            direction: Trade direction
            desired_price: Desired entry price
            position_size: Position size in lots
            market_data: Current market data (bid, ask, spread, etc.)
            atr: Current ATR for volatility-based slippage

        Returns:
            ExecutionResult with fill details
        """
        bid = market_data.get("bid", desired_price)
        ask = market_data.get("ask", desired_price)
        spread = market_data.get("spread", ask - bid)

        # Determine base price based on direction
        if direction == Direction.LONG:
            base_price = ask  # Buy at ask
        else:
            base_price = bid  # Sell at bid

        # Calculate slippage
        slippage_pips = self._calculate_slippage(
            direction, position_size, market_data, atr
        )

        # Apply slippage
        if direction == Direction.LONG:
            fill_price = base_price + slippage_pips
        else:
            fill_price = base_price - slippage_pips

        # Check fill probability
        fill_result = self._check_fill(position_size, market_data)

        if not fill_result["filled"]:
            return ExecutionResult(
                filled=False,
                fill_price=None,
                fill_size=0.0,
                slippage=0.0,
                slippage_pips=0.0,
                commission=0.0,
                remaining_size=position_size,
                reject_reason=fill_result.get("reason"),
            )

        actual_slippage = fill_price - desired_price

        return ExecutionResult(
            filled=True,
            fill_price=fill_price,
            fill_size=fill_result["size"],
            slippage=actual_slippage,
            slippage_pips=slippage_pips,
            commission=self._calculate_commission(fill_result["size"]),
            remaining_size=position_size - fill_result["size"],
        )

    def simulate_exit(
        self,
        direction: Direction,
        entry_price: float,
        desired_price: float,
        position_size: float,
        market_data: Dict[str, Any],
        exit_type: str = "market",
        atr: Optional[float] = None,
    ) -> ExecutionResult:
        """Simulate exit execution.

        Args:
            direction: Original trade direction
            desired_price: Desired exit price
            position_size: Position size to close
            market_data: Current market data
            exit_type: Type of exit (market, stop, limit)
            atr: Current ATR

        Returns:
            ExecutionResult with fill details
        """
        bid = market_data.get("bid", desired_price)
        ask = market_data.get("ask", desired_price)

        # Determine base price (opposite of entry)
        if direction == Direction.LONG:
            base_price = bid  # Sell at bid to close long
        else:
            base_price = ask  # Buy at ask to close short

        # Different slippage for different exit types
        if exit_type == "stop":
            # Stop orders typically have worse slippage
            slippage_multiplier = 1.5
        elif exit_type == "limit":
            # Limit orders may not fill if price gaps
            slippage_multiplier = 0.5
        else:
            slippage_multiplier = 1.0

        slippage_pips = (
            self._calculate_slippage(direction, position_size, market_data, atr)
            * slippage_multiplier
        )

        # Apply slippage (opposite direction for exit)
        if direction == Direction.LONG:
            fill_price = base_price - slippage_pips
        else:
            fill_price = base_price + slippage_pips

        # Check fill probability
        fill_result = self._check_fill(position_size, market_data, exit_type)

        if not fill_result["filled"]:
            return ExecutionResult(
                filled=False,
                fill_price=None,
                fill_size=0.0,
                slippage=0.0,
                slippage_pips=0.0,
                commission=0.0,
                remaining_size=position_size,
                reject_reason=fill_result.get("reason"),
            )

        actual_slippage = fill_price - desired_price

        return ExecutionResult(
            filled=True,
            fill_price=fill_price,
            fill_size=fill_result["size"],
            slippage=actual_slippage,
            slippage_pips=slippage_pips,
            commission=self._calculate_commission(fill_result["size"]),
            remaining_size=position_size - fill_result["size"],
        )

    def _calculate_slippage(
        self,
        direction: Direction,
        position_size: float,
        market_data: Dict[str, Any],
        atr: Optional[float] = None,
    ) -> float:
        """Calculate slippage in price units.

        Args:
            direction: Trade direction
            position_size: Position size
            market_data: Market data
            atr: Current ATR

        Returns:
            Slippage in price units
        """
        current_price = market_data.get("close", 100)

        if self.config.slippage_model == SlippageModel.NONE:
            return 0.0

        elif self.config.slippage_model == SlippageModel.FIXED:
            # Convert pips to price
            return self.config.fixed_slippage_pips

        elif self.config.slippage_model == SlippageModel.RANDOM:
            # Random slippage with normal distribution
            slippage = np.random.normal(
                self.config.fixed_slippage_pips, self.config.slippage_std_dev
            )
            return max(0, min(slippage, self.config.max_slippage_pips))

        elif self.config.slippage_model == SlippageModel.MARKET_IMPACT:
            # Market impact based on position size
            impact = position_size * self.config.market_impact_factor
            noise = np.random.normal(0, self.config.slippage_std_dev)
            return max(0, impact + noise)

        elif self.config.slippage_model == SlippageModel.VOLATILITY_BASED:
            # Slippage proportional to volatility
            if atr is None:
                atr = current_price * 0.001  # Default 0.1% ATR

            base_slippage = atr * 0.1  # 10% of ATR
            noise = np.random.normal(0, base_slippage * 0.5)
            return max(0, base_slippage + noise)

        return 0.0

    def _check_fill(
        self,
        position_size: float,
        market_data: Dict[str, Any],
        order_type: str = "market",
    ) -> Dict[str, Any]:
        """Check if order gets filled.

        Args:
            position_size: Order size
            market_data: Market data with liquidity info
            order_type: Type of order

        Returns:
            Dict with fill status and size
        """
        if self.config.fill_model == FillModel.IMMEDIATE:
            return {"filled": True, "size": position_size}

        elif self.config.fill_model == FillModel.PROBABILISTIC:
            # Check if order fills based on probability
            if np.random.random() < self.config.fill_probability:
                return {"filled": True, "size": position_size}
            else:
                return {
                    "filled": False,
                    "reason": "liquidity_insufficient",
                    "size": 0.0,
                }

        elif self.config.fill_model == FillModel.PARTIAL:
            # Partial fill based on rate
            fill_size = position_size * self.config.partial_fill_rate
            return {"filled": fill_size > 0, "size": fill_size}

        return {"filled": True, "size": position_size}

    def _calculate_commission(self, position_size: float) -> float:
        """Calculate commission for a trade.

        Args:
            position_size: Position size

        Returns:
            Commission amount
        """
        # Simplified: $5 per lot
        return 5.0 * position_size

    def simulate_gapping_stop(
        self,
        direction: Direction,
        stop_price: float,
        gap_price: float,
        position_size: float,
    ) -> ExecutionResult:
        """Simulate stop loss execution during a gap.

        Args:
            direction: Trade direction
            stop_price: Stop price level
            gap_price: Price after gap
            position_size: Position size

        Returns:
            ExecutionResult
        """
        # Stop fills at gap price (worse than stop level)
        slippage = abs(gap_price - stop_price)

        return ExecutionResult(
            filled=True,
            fill_price=gap_price,
            fill_size=position_size,
            slippage=slippage,
            slippage_pips=slippage,
            commission=self._calculate_commission(position_size),
            remaining_size=0.0,
            reject_reason=None,
        )

    def estimate_market_impact(
        self, position_size: float, daily_volume: float, volatility: float
    ) -> float:
        """Estimate market impact using square root law.

        Args:
            position_size: Order size
            daily_volume: Daily trading volume
            volatility: Daily volatility (as decimal)

        Returns:
            Expected price impact
        """
        # Square root law: impact ∝ sqrt(order_size / daily_volume)
        participation_rate = position_size / daily_volume if daily_volume > 0 else 0
        impact = (
            volatility * np.sqrt(participation_rate) * self.config.market_impact_factor
        )

        return impact


def create_realistic_simulator() -> ExecutionSimulator:
    """Create a simulator with realistic settings.

    Returns:
        ExecutionSimulator with realistic defaults
    """
    config = ExecutionConfig(
        slippage_model=SlippageModel.VOLATILITY_BASED,
        fill_model=FillModel.PROBABILISTIC,
        fill_probability=0.98,
        max_slippage_pips=5.0,
        market_impact_factor=0.05,
    )
    return ExecutionSimulator(config)


def create_aggressive_simulator() -> ExecutionSimulator:
    """Create a simulator with aggressive (worst-case) settings.

    Returns:
        ExecutionSimulator with aggressive defaults
    """
    config = ExecutionConfig(
        slippage_model=SlippageModel.MARKET_IMPACT,
        fill_model=FillModel.PARTIAL,
        partial_fill_rate=0.9,
        fixed_slippage_pips=1.0,
        max_slippage_pips=10.0,
        market_impact_factor=0.2,
    )
    return ExecutionSimulator(config)


def create_conservative_simulator() -> ExecutionSimulator:
    """Create a simulator with conservative (best-case) settings.

    Returns:
        ExecutionSimulator with conservative defaults
    """
    config = ExecutionConfig(
        slippage_model=SlippageModel.FIXED,
        fill_model=FillModel.IMMEDIATE,
        fixed_slippage_pips=0.2,
        max_slippage_pips=1.0,
    )
    return ExecutionSimulator(config)
