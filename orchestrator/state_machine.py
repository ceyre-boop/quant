"""State Machine - Symbol state management for the Clawd Trading System.

Manages per-symbol state transitions and trading lifecycle.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from contracts.types import (
    Direction,
    BiasOutput,
    RiskOutput,
    GameOutput,
    PositionState,
    EntrySignal,
)

logger = logging.getLogger(__name__)


class SymbolState(Enum):
    """Trading states for a symbol."""

    IDLE = "idle"  # No active signal or position
    SCANNING = "scanning"  # Monitoring for entry
    ENTRY_PENDING = "entry_pending"  # Entry signal triggered, awaiting confirmation
    IN_POSITION = "in_position"  # Active position
    EXIT_PENDING = "exit_pending"  # Exit signal triggered
    COOLDOWN = "cooldown"  # Post-exit cooldown period
    ERROR = "error"  # Error state


class PositionPhase(Enum):
    """Phase within a position."""

    NONE = "none"
    OPEN = "open"
    TP1_HIT = "tp1_hit"
    TP2_HIT = "tp2_hit"
    BE_STOP = "be_stop"  # Breakeven stop active
    TRAILING = "trailing"
    STOPPED = "stopped"
    CLOSED = "closed"


@dataclass
class SymbolContext:
    """Complete context for a symbol's trading state."""

    symbol: str
    state: SymbolState = SymbolState.IDLE
    position_phase: PositionPhase = PositionPhase.NONE

    # Latest layer outputs
    current_bias: Optional[BiasOutput] = None
    current_risk: Optional[RiskOutput] = None
    current_game: Optional[GameOutput] = None

    # Position tracking
    current_position: Optional[PositionState] = None
    pending_entry: Optional[Dict[str, Any]] = None

    # State history
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    entry_count: int = 0
    last_entry_time: Optional[datetime] = None
    last_exit_time: Optional[datetime] = None

    # Performance tracking
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0

    # Metadata
    updated_at: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "state": self.state.value,
            "position_phase": self.position_phase.value,
            "current_bias": self.current_bias.to_dict() if self.current_bias else None,
            "current_risk": self.current_risk.to_dict() if self.current_risk else None,
            "current_game": self.current_game.to_dict() if self.current_game else None,
            "current_position": (
                self.current_position.to_dict() if self.current_position else None
            ),
            "pending_entry": self.pending_entry,
            "entry_count": self.entry_count,
            "last_entry_time": (
                self.last_entry_time.isoformat() if self.last_entry_time else None
            ),
            "last_exit_time": (
                self.last_exit_time.isoformat() if self.last_exit_time else None
            ),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "total_pnl": self.total_pnl,
            "updated_at": self.updated_at.isoformat(),
            "error_message": self.error_message,
        }


class SymbolStateMachine:
    """State machine for managing symbol trading lifecycle."""

    # Valid state transitions
    TRANSITIONS = {
        SymbolState.IDLE: [SymbolState.SCANNING, SymbolState.ERROR],
        SymbolState.SCANNING: [
            SymbolState.ENTRY_PENDING,
            SymbolState.IDLE,
            SymbolState.ERROR,
        ],
        SymbolState.ENTRY_PENDING: [
            SymbolState.IN_POSITION,
            SymbolState.SCANNING,
            SymbolState.ERROR,
        ],
        SymbolState.IN_POSITION: [
            SymbolState.EXIT_PENDING,
            SymbolState.COOLDOWN,
            SymbolState.ERROR,
        ],
        SymbolState.EXIT_PENDING: [SymbolState.COOLDOWN, SymbolState.ERROR],
        SymbolState.COOLDOWN: [
            SymbolState.IDLE,
            SymbolState.SCANNING,
            SymbolState.ERROR,
        ],
        SymbolState.ERROR: [SymbolState.IDLE, SymbolState.SCANNING],
    }

    def __init__(self):
        """Initialize state machine."""
        self._contexts: Dict[str, SymbolContext] = {}
        self._transition_handlers: Dict[tuple, List[callable]] = {}

    def get_context(self, symbol: str) -> SymbolContext:
        """Get or create context for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            SymbolContext for the symbol
        """
        if symbol not in self._contexts:
            self._contexts[symbol] = SymbolContext(symbol=symbol)
            logger.info(f"Created new context for {symbol}")
        return self._contexts[symbol]

    def can_transition(self, symbol: str, new_state: SymbolState) -> bool:
        """Check if transition is valid.

        Args:
            symbol: Trading symbol
            new_state: Desired new state

        Returns:
            True if transition is valid
        """
        context = self.get_context(symbol)
        return new_state in self.TRANSITIONS.get(context.state, [])

    def transition(self, symbol: str, new_state: SymbolState, reason: str = "") -> bool:
        """Attempt state transition.

        Args:
            symbol: Trading symbol
            new_state: Desired new state
            reason: Reason for transition

        Returns:
            True if transition succeeded
        """
        context = self.get_context(symbol)

        if not self.can_transition(symbol, new_state):
            logger.warning(
                f"Invalid transition for {symbol}: "
                f"{context.state.value} -> {new_state.value}"
            )
            return False

        old_state = context.state
        context.state = new_state
        context.updated_at = datetime.now()

        # Record in history
        context.state_history.append(
            {
                "from": old_state.value,
                "to": new_state.value,
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
            }
        )

        logger.info(
            f"{symbol} state: {old_state.value} -> {new_state.value} ({reason})"
        )

        # Call transition handlers
        self._call_transition_handlers(symbol, old_state, new_state)

        return True

    def update_layer_outputs(
        self,
        symbol: str,
        bias: Optional[BiasOutput] = None,
        risk: Optional[RiskOutput] = None,
        game: Optional[GameOutput] = None,
    ):
        """Update layer outputs for a symbol.

        Args:
            symbol: Trading symbol
            bias: Layer 1 bias output
            risk: Layer 2 risk output
            game: Layer 3 game output
        """
        context = self.get_context(symbol)

        if bias:
            context.current_bias = bias
        if risk:
            context.current_risk = risk
        if game:
            context.current_game = game

        context.updated_at = datetime.now()

    def set_position(self, symbol: str, position: Optional[PositionState]):
        """Set current position for symbol.

        Args:
            symbol: Trading symbol
            position: Position state or None to clear
        """
        context = self.get_context(symbol)
        context.current_position = position

        if position:
            context.position_phase = PositionPhase.OPEN
            logger.info(
                f"{symbol} position set: {position.direction.name} @ {position.entry_price}"
            )
        else:
            context.position_phase = PositionPhase.NONE
            logger.info(f"{symbol} position cleared")

        context.updated_at = datetime.now()

    def set_position_phase(self, symbol: str, phase: PositionPhase, reason: str = ""):
        """Update position phase.

        Args:
            symbol: Trading symbol
            phase: New position phase
            reason: Reason for phase change
        """
        context = self.get_context(symbol)
        old_phase = context.position_phase
        context.position_phase = phase
        context.updated_at = datetime.now()

        logger.info(f"{symbol} phase: {old_phase.value} -> {phase.value} ({reason})")

    def on_entry_signal(self, symbol: str, entry_signal: Dict[str, Any]) -> bool:
        """Handle entry signal.

        Args:
            symbol: Trading symbol
            entry_signal: Entry signal details

        Returns:
            True if signal accepted
        """
        context = self.get_context(symbol)

        if context.state not in [SymbolState.SCANNING, SymbolState.IDLE]:
            logger.warning(
                f"{symbol} ignoring entry signal in {context.state.value} state"
            )
            return False

        context.pending_entry = entry_signal
        self.transition(symbol, SymbolState.ENTRY_PENDING, "entry_signal_received")

        return True

    def on_entry_confirmed(self, symbol: str, position: PositionState):
        """Handle confirmed entry.

        Args:
            symbol: Trading symbol
            position: New position state
        """
        context = self.get_context(symbol)

        if context.state != SymbolState.ENTRY_PENDING:
            logger.warning(
                f"{symbol} unexpected entry confirmation in {context.state.value} state"
            )

        context.current_position = position
        context.entry_count += 1
        context.last_entry_time = datetime.now()
        context.pending_entry = None

        self.transition(symbol, SymbolState.IN_POSITION, "entry_confirmed")
        self.set_position_phase(symbol, PositionPhase.OPEN, "position_opened")

    def on_exit_signal(self, symbol: str, exit_reason: str):
        """Handle exit signal.

        Args:
            symbol: Trading symbol
            exit_reason: Reason for exit
        """
        context = self.get_context(symbol)

        if context.state != SymbolState.IN_POSITION:
            logger.warning(
                f"{symbol} ignoring exit signal in {context.state.value} state"
            )
            return

        self.transition(symbol, SymbolState.EXIT_PENDING, exit_reason)

    def on_exit_confirmed(self, symbol: str, realized_pnl: float, exit_price: float):
        """Handle confirmed exit.

        Args:
            symbol: Trading symbol
            realized_pnl: Realized PnL
            exit_price: Exit price
        """
        context = self.get_context(symbol)

        # Update performance tracking
        context.total_trades += 1
        context.total_pnl += realized_pnl
        if realized_pnl > 0:
            context.winning_trades += 1

        context.last_exit_time = datetime.now()
        context.current_position = None
        context.position_phase = PositionPhase.CLOSED
        context.pending_entry = None

        self.transition(
            symbol, SymbolState.COOLDOWN, f"exit_confirmed_pnl_{realized_pnl:.2f}"
        )

        logger.info(f"{symbol} exit confirmed: PnL={realized_pnl:.2f} @ {exit_price}")

    def on_tp1_hit(self, symbol: str):
        """Handle TP1 hit."""
        self.set_position_phase(symbol, PositionPhase.TP1_HIT, "tp1_hit")

    def on_tp2_hit(self, symbol: str):
        """Handle TP2 hit."""
        self.set_position_phase(symbol, PositionPhase.TP2_HIT, "tp2_hit")

        # Move to cooldown if fully exited
        context = self.get_context(symbol)
        if context.state == SymbolState.IN_POSITION:
            self.on_exit_signal(symbol, "tp2_fully_closed")

    def on_be_stop_activated(self, symbol: str):
        """Handle breakeven stop activation."""
        self.set_position_phase(
            symbol, PositionPhase.BE_STOP, "breakeven_stop_activated"
        )

    def on_trail_activated(self, symbol: str):
        """Handle trailing stop activation."""
        self.set_position_phase(
            symbol, PositionPhase.TRAILING, "trailing_stop_activated"
        )

    def release_from_cooldown(self, symbol: str):
        """Release symbol from cooldown to scanning."""
        context = self.get_context(symbol)

        if context.state == SymbolState.COOLDOWN:
            self.transition(symbol, SymbolState.SCANNING, "cooldown_expired")

    def set_error(self, symbol: str, error_message: str):
        """Set error state for symbol.

        Args:
            symbol: Trading symbol
            error_message: Error description
        """
        context = self.get_context(symbol)
        context.error_message = error_message
        self.transition(symbol, SymbolState.ERROR, error_message)

    def clear_error(self, symbol: str):
        """Clear error state and return to idle."""
        self.transition(symbol, SymbolState.IDLE, "error_cleared")
        self.get_context(symbol).error_message = None

    def register_transition_handler(
        self, from_state: SymbolState, to_state: SymbolState, handler: callable
    ):
        """Register a handler for a specific transition.

        Args:
            from_state: Source state
            to_state: Target state
            handler: Callback function(symbol, from_state, to_state)
        """
        key = (from_state, to_state)
        if key not in self._transition_handlers:
            self._transition_handlers[key] = []
        self._transition_handlers[key].append(handler)

    def _call_transition_handlers(
        self, symbol: str, from_state: SymbolState, to_state: SymbolState
    ):
        """Call all registered handlers for a transition."""
        key = (from_state, to_state)
        handlers = self._transition_handlers.get(key, [])

        for handler in handlers:
            try:
                handler(symbol, from_state, to_state)
            except Exception as e:
                logger.error(f"Transition handler error: {e}")

    def get_all_contexts(self) -> Dict[str, SymbolContext]:
        """Get all symbol contexts."""
        return self._contexts.copy()

    def get_active_positions(self) -> List[SymbolContext]:
        """Get contexts with active positions."""
        return [
            ctx
            for ctx in self._contexts.values()
            if ctx.state == SymbolState.IN_POSITION
        ]

    def get_symbols_in_state(self, state: SymbolState) -> List[str]:
        """Get all symbols in a specific state."""
        return [symbol for symbol, ctx in self._contexts.items() if ctx.state == state]

    def reset_symbol(self, symbol: str):
        """Reset a symbol to initial state."""
        self._contexts[symbol] = SymbolContext(symbol=symbol)
        logger.info(f"{symbol} context reset")

    def reset_all(self):
        """Reset all symbols."""
        self._contexts.clear()
        logger.info("All symbol contexts reset")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all symbol states."""
        state_counts = {}
        for ctx in self._contexts.values():
            state_counts[ctx.state.value] = state_counts.get(ctx.state.value, 0) + 1

        return {
            "total_symbols": len(self._contexts),
            "state_distribution": state_counts,
            "active_positions": len(self.get_active_positions()),
            "total_pnl": sum(ctx.total_pnl for ctx in self._contexts.values()),
            "total_trades": sum(ctx.total_trades for ctx in self._contexts.values()),
        }


class StateMachineManager:
    """Manager for multiple state machines (e.g., by strategy)."""

    def __init__(self):
        """Initialize manager."""
        self._machines: Dict[str, SymbolStateMachine] = {}

    def get_machine(self, name: str = "default") -> SymbolStateMachine:
        """Get or create state machine.

        Args:
            name: State machine name

        Returns:
            SymbolStateMachine instance
        """
        if name not in self._machines:
            self._machines[name] = SymbolStateMachine()
            logger.info(f"Created state machine: {name}")
        return self._machines[name]

    def get_all_machines(self) -> Dict[str, SymbolStateMachine]:
        """Get all state machines."""
        return self._machines.copy()

    def reset_machine(self, name: str):
        """Reset a specific state machine."""
        if name in self._machines:
            self._machines[name].reset_all()
            logger.info(f"Reset state machine: {name}")

    def reset_all(self):
        """Reset all state machines."""
        for machine in self._machines.values():
            machine.reset_all()
        logger.info("All state machines reset")
