"""
Paper Trading Module for Clawd Trading

Simulates trading with real market data but no real money.
Tracks P&L, logs trades, compares to backtest expectations.
"""

import logging
import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from integration.production_engine import EnhancedEntrySignal
from meta_evaluator.auto_documenter import log_live_trade, compare_performance

logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


@dataclass
class PaperPosition:
    """Paper trading position."""

    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: TradeStatus = TradeStatus.PENDING
    close_reason: Optional[str] = None

    # Metadata
    entry_model: str = ""
    dominant_participant: str = ""
    regime: str = ""
    expected_r: float = 0.0


class PaperTradingEngine:
    """Paper trading engine - trades with fake money, real data."""

    def __init__(self, starting_equity: float = 100000.0):
        self.starting_equity = starting_equity
        self.current_equity = starting_equity
        self.positions: Dict[str, PaperPosition] = {}
        self.closed_positions: list = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now().date()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Paper trading initialized: ${starting_equity:,.2f}")

    def execute_signal(
        self, signal: EnhancedEntrySignal, current_price: float
    ) -> Optional[PaperPosition]:
        """
        Execute a paper trade from an entry signal.

        Args:
            signal: The entry signal from production engine
            current_price: Current market price

        Returns:
            PaperPosition if executed, None if rejected
        """
        # Check if we already have a position for this symbol
        if signal.symbol in self.positions:
            self.logger.info(f"Already have position in {signal.symbol}, skipping")
            return None

        # Generate trade ID
        trade_id = f"PAPER_{uuid.uuid4().hex[:8].upper()}"

        # Calculate position size in dollars
        position_value = (
            self.current_equity * signal.position_size * 0.02
        )  # 2% risk per trade max

        # Create position
        position = PaperPosition(
            trade_id=trade_id,
            symbol=signal.symbol,
            direction="LONG" if signal.direction.value == 1 else "SHORT",
            entry_price=current_price,  # Use current market price
            position_size=position_value,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            entry_time=datetime.now(),
            status=TradeStatus.OPEN,
            entry_model=signal.entry_model,
            dominant_participant=signal.dominant_participant,
            regime=signal.regime,
            expected_r=signal.entry_model_expected_r,
        )

        self.positions[signal.symbol] = position

        # Log the trade
        log_live_trade(
            trade_id=trade_id,
            timestamp=position.entry_time.isoformat(),
            symbol=signal.symbol,
            direction=position.direction,
            entry_price=current_price,
            exit_price=None,
            position_size=position_value,
            pnl=None,
            pnl_pct=None,
            status="OPEN",
            entry_model=signal.entry_model,
            dominant_participant=signal.dominant_participant,
            regime=signal.regime,
            gates_passed=signal.gates_passed,
        )

        self.daily_trades += 1

        self.logger.info(
            f"PAPER TRADE OPENED: {trade_id} {signal.symbol} {position.direction} "
            f"@{current_price:.2f} Size: ${position_value:,.2f} "
            f"(Model: {signal.entry_model}, Expected R: {signal.entry_model_expected_r:.2f})"
        )

        return position

    def update_positions(self, market_data: Dict[str, float]) -> list:
        """
        Update all open positions with current market prices.
        Check for stop loss / take profit hits.

        Args:
            market_data: Dict of symbol -> current_price

        Returns:
            List of closed positions
        """
        closed = []

        for symbol, price in market_data.items():
            if symbol not in self.positions:
                continue

            position = self.positions[symbol]

            # Calculate unrealized P&L
            if position.direction == "LONG":
                unrealized_pct = (price - position.entry_price) / position.entry_price
            else:
                unrealized_pct = (position.entry_price - price) / position.entry_price

            unrealized_pnl = position.position_size * unrealized_pct

            # Check for exit conditions
            exit_triggered = False
            exit_price = price
            close_reason = None

            # Stop loss hit
            if position.direction == "LONG" and price <= position.stop_loss:
                exit_triggered = True
                exit_price = position.stop_loss
                close_reason = "STOP_LOSS"
            elif position.direction == "SHORT" and price >= position.stop_loss:
                exit_triggered = True
                exit_price = position.stop_loss
                close_reason = "STOP_LOSS"

            # Take profit 1 hit
            elif position.direction == "LONG" and price >= position.take_profit_1:
                exit_triggered = True
                exit_price = position.take_profit_1
                close_reason = "TP1"
            elif position.direction == "SHORT" and price <= position.take_profit_1:
                exit_triggered = True
                exit_price = position.take_profit_1
                close_reason = "TP1"

            # Exit if triggered
            if exit_triggered:
                closed_position = self._close_position(symbol, exit_price, close_reason)
                if closed_position:
                    closed.append(closed_position)

        return closed

    def _close_position(
        self, symbol: str, exit_price: float, reason: str
    ) -> Optional[PaperPosition]:
        """Close a position and record P&L."""
        if symbol not in self.positions:
            return None

        position = self.positions.pop(symbol)
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.status = TradeStatus.CLOSED
        position.close_reason = reason

        # Calculate final P&L
        if position.direction == "LONG":
            position.pnl_pct = (
                exit_price - position.entry_price
            ) / position.entry_price
        else:
            position.pnl_pct = (
                position.entry_price - exit_price
            ) / position.entry_price

        position.pnl = position.position_size * position.pnl_pct

        # Update equity
        self.current_equity += position.pnl
        self.daily_pnl += position.pnl

        # Add to closed positions
        self.closed_positions.append(position)

        # Update logged trade
        log_live_trade(
            trade_id=position.trade_id,
            timestamp=position.entry_time.isoformat(),
            symbol=symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            position_size=position.position_size,
            pnl=position.pnl,
            pnl_pct=position.pnl_pct,
            status="CLOSED",
            entry_model=position.entry_model,
            dominant_participant=position.dominant_participant,
            regime=position.regime,
            gates_passed=0,  # Already recorded on open
        )

        # Log result
        pnl_emoji = "✅" if position.pnl > 0 else "❌"
        self.logger.info(
            f"PAPER TRADE CLOSED: {position.trade_id} {symbol} "
            f"PnL: ${position.pnl:,.2f} ({position.pnl_pct:+.2%}) "
            f"Reason: {reason} {pnl_emoji}"
        )

        return position

    def get_summary(self) -> Dict[str, Any]:
        """Get paper trading summary."""
        closed = [p for p in self.closed_positions if p.status == TradeStatus.CLOSED]

        if not closed:
            return {
                "status": "paper_trading",
                "starting_equity": self.starting_equity,
                "current_equity": self.current_equity,
                "total_pnl": 0,
                "total_return_pct": 0,
                "total_trades": 0,
                "open_positions": len(self.positions),
            }

        wins = sum(1 for p in closed if p.pnl > 0)
        total_pnl = sum(p.pnl for p in closed)

        return {
            "status": "paper_trading",
            "starting_equity": self.starting_equity,
            "current_equity": self.current_equity,
            "total_pnl": total_pnl,
            "total_return_pct": total_pnl / self.starting_equity,
            "total_trades": len(closed),
            "win_rate": wins / len(closed),
            "open_positions": len(self.positions),
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
        }

    def reset_daily(self) -> None:
        """Reset daily counters."""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset = today
            self.logger.info("Daily counters reset")

    def manual_close(
        self, symbol: str, exit_price: float, reason: str = "MANUAL"
    ) -> Optional[PaperPosition]:
        """Manually close a position."""
        return self._close_position(symbol, exit_price, reason)
