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


from execution.rr_engine import RREngine
try:
    from execution.ptj_gates import PTJCircuitBreaker, PTJGateRunner, ptj_shock_candle_gate, TradePhaseTracker
    _PTJ_AVAILABLE = True
except ImportError:
    _PTJ_AVAILABLE = False

class PaperTradingEngine:
    """
    Paper trading engine — PTJ capital protection integrated.

    PTJ gates run before every execute_signal():
      Gate 1-3: circuit breakers (weekly/monthly loss, post-loss cooldown)
      Gate 4:   max concurrent positions
      Gate 5-6: 200 SMA (requires spy_prices + asset_prices passed in)
      Gate 9:   stop-first enforcement
      Gate 10:  R:R minimum
      Gate 11:  portfolio risk cap

    Phase tracker (Principle 4) added to every position.
    Circuit breaker updated on every close.
    """

    def __init__(self, starting_equity: float = 100000.0):
        self.starting_equity = starting_equity
        self.current_equity = starting_equity
        self.positions: Dict[str, PaperPosition] = {}
        self.closed_positions: list = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now().date()
        self.rr_engine = RREngine()
        self._phase_trackers: Dict[str, Any] = {}

        # PTJ circuit breaker — tracks weekly/monthly P&L
        self._ptj_cb = PTJCircuitBreaker(starting_equity) if _PTJ_AVAILABLE else None

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Paper trading initialized: ${starting_equity:,.2f} | "
            f"PTJ gates: {'ACTIVE' if _PTJ_AVAILABLE else 'unavailable'}"
        )
    
    def execute_signal(
        self,
        signal: EnhancedEntrySignal,
        current_price: float,
        current_atr: float,
        spy_prices=None,
        asset_prices=None,
        grade: str = 'B',
    ) -> Optional[PaperPosition]:
        """
        Execute a paper trade — PTJ 12-gate filter runs first.

        PTJ sequence (defence-first, enforced in code order):
          1. Run circuit breakers and position limits
          2. Run 200 SMA gates (requires spy_prices, asset_prices)
          3. Compute stop (STEP 1 — must happen before target)
          4. Compute targets from stop (STEP 2+3)
          5. R:R gate
          6. Only then: create position
        """
        import numpy as np

        # Already in this symbol?
        if signal.symbol in self.positions:
            self.logger.info(f"Already have position in {signal.symbol}, skipping")
            return None

        direction_int = signal.direction.value  # 1=long, -1=short
        direction_str = 'long' if direction_int == 1 else 'short'

        # ── PTJ Gates (Gates 1-11) ────────────────────────────────────────── #
        if _PTJ_AVAILABLE and self._ptj_cb is not None:
            open_pos_list = [
                {'risk_pct': 0.015}  # approximate risk per open position
                for _ in self.positions
            ]
            spy_arr = (np.array(spy_prices) if spy_prices is not None
                       else np.full(200, current_price))
            asset_arr = (np.array(asset_prices) if asset_prices is not None
                         else np.full(200, current_price))

            runner = PTJGateRunner(self._ptj_cb, open_pos_list, self.current_equity)

            # Compute stop FIRST (PTJ principle: stop before target)
            stop_price = self.rr_engine.calculate_stop(
                current_price, current_atr, direction_int)
            unit_risk = abs(current_price - stop_price)
            tp1 = current_price + direction_int * unit_risk * 1.5
            new_risk_pct = (unit_risk * 1.0) / self.current_equity  # approx

            gate_result = runner.run(
                signal_direction=direction_str,
                symbol=signal.symbol,
                grade=grade,
                spy_prices=spy_arr,
                asset_prices=asset_arr,
                entry_price=current_price,
                stop_price=stop_price,
                tp1_price=tp1,
                new_trade_risk_pct=new_risk_pct,
                direction_int=direction_int,
            )

            if not gate_result:
                self.logger.info(
                    f"[PTJ_GATE] {signal.symbol} BLOCKED by {gate_result.gate}: "
                    f"{gate_result.reason}"
                )
                return None

            size_mod = gate_result.size_modifier
        else:
            # PTJ not available — compute stop for downstream use
            stop_price = self.rr_engine.calculate_stop(
                current_price, current_atr, direction_int)
            size_mod = 1.0

        # ── Compute PTJ 3-target bracket (stop already computed above) ────── #
        bracket = self.rr_engine.calculate_brackets(
            current_price, stop_price, direction_int, grade=grade)

        # Legacy single-target brackets dict for backward compat
        brackets = {
            'sl':  bracket.stop_price,
            'tp':  bracket.tp1_price,
            'tp1': bracket.tp1_price,
            'tp2': bracket.tp2_price,
            'tp3': bracket.tp3_price,
        }
        
        # Generate trade ID
        trade_id = f"PAPER_{uuid.uuid4().hex[:8].upper()}"
        
        # Position size: 2% risk × PTJ size modifier (circuit breakers may halve this)
        risk_per_share = abs(current_price - brackets['sl'])
        position_size_shares = (self.current_equity * 0.02 * size_mod) / max(risk_per_share, 1e-9)
        position_value = position_size_shares * current_price
        
        # Create position
        position = PaperPosition(
            trade_id=trade_id,
            symbol=signal.symbol,
            direction="LONG" if signal.direction.value == 1 else "SHORT",
            entry_price=current_price,
            position_size=position_value,
            stop_loss=brackets['sl'],
            take_profit_1=brackets['tp'],
            take_profit_2=brackets['tp'], # Standardizing on single dynamic target
            entry_time=datetime.now(),
            status=TradeStatus.OPEN,
            entry_model=signal.entry_model,
            dominant_participant=signal.dominant_participant,
            regime=signal.regime,
            expected_r=signal.entry_model_expected_r,
        )
        
        self.positions[signal.symbol] = position

        # PTJ Phase tracker — "every position starts wrong until proven otherwise"
        if _PTJ_AVAILABLE:
            self._phase_trackers[signal.symbol] = TradePhaseTracker(
                entry_price=current_price,
                stop_price=brackets['sl'],
                direction=direction_int,
            )

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
        return position
    
    def update_positions(self, market_data: Dict[str, float], atrs: Dict[str, float] = None) -> list:
        """
        Update all open positions — PTJ phase tracking + shock candle + trailing.
        """
        closed = []
        atrs = atrs or {}

        for symbol, price in market_data.items():
            if symbol not in self.positions:
                continue

            position = self.positions[symbol]
            direction_val = 1 if position.direction == "LONG" else -1

            # PTJ Phase tracker update
            if _PTJ_AVAILABLE and symbol in self._phase_trackers:
                tracker = self._phase_trackers[symbol]
                tracker.update(price)
                # Phase 1 early exit: moving against us before thesis proven
                if tracker.should_exit_early(price):
                    self.logger.info(
                        f"[PTJ_PHASE1] {symbol}: early exit — "
                        f"price moving against unproven position"
                    )
                    closed_pos = self._close_position(symbol, price, "PTJ_PHASE1_EARLY_EXIT")
                    if closed_pos:
                        closed.append(closed_pos)
                    continue

            # PTJ Shock candle gate (Gate 7)
            if _PTJ_AVAILABLE and symbol in atrs and atrs[symbol] > 0:
                prev_price = position.entry_price  # simplification; use last bar close ideally
                bar_ret = price - prev_price
                shock = ptj_shock_candle_gate(bar_ret, atrs[symbol], direction_val)
                if not shock:
                    self.logger.info(f"[PTJ_SHOCK] {symbol}: shock candle exit")
                    closed_pos = self._close_position(symbol, price, "PTJ_SHOCK_CANDLE")
                    if closed_pos:
                        closed.append(closed_pos)
                    continue

            # --- AUTO TRAILING LOGIC ---
            old_sl = position.stop_loss
            position.stop_loss = self.rr_engine.update_trailing_stop(
                price, position.entry_price, position.stop_loss, direction_val
            )
            if position.stop_loss != old_sl:
                self.logger.info(f"TRAILING STOP: {symbol}: {old_sl:.4f} → {position.stop_loss:.4f}")

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
            
            # Take profit hit
            elif position.direction == "LONG" and price >= position.take_profit_1:
                exit_triggered = True
                exit_price = position.take_profit_1
                close_reason = "TAKE_PROFIT"
            elif position.direction == "SHORT" and price <= position.take_profit_1:
                exit_triggered = True
                exit_price = position.take_profit_1
                close_reason = "TAKE_PROFIT"
            
            # Exit if triggered
            if exit_triggered:
                closed_position = self._close_position(symbol, exit_price, close_reason)
                if closed_position:
                    closed.append(closed_position)
        
        return closed

    
    def _close_position(self, symbol: str, exit_price: float, reason: str) -> Optional[PaperPosition]:
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
            position.pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            position.pnl_pct = (position.entry_price - exit_price) / position.entry_price
        
        position.pnl = position.position_size * position.pnl_pct
        
        # Update equity
        self.current_equity += position.pnl
        self.daily_pnl += position.pnl

        # PTJ circuit breaker: record every close so weekly/monthly tracking stays live
        if _PTJ_AVAILABLE and self._ptj_cb is not None:
            self._ptj_cb.record_trade(position.pnl)

        # Clean up phase tracker
        self._phase_trackers.pop(symbol, None)

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
    
    def manual_close(self, symbol: str, exit_price: float, reason: str = "MANUAL") -> Optional[PaperPosition]:
        """Manually close a position."""
        return self._close_position(symbol, exit_price, reason)
