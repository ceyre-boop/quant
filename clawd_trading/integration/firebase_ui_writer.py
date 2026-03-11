"""Firebase UI Writer - Formats and writes data for frontend consumption.

Ensures all Firebase writes match the frontend UI's expected structure.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum

from contracts.types import (
    EntrySignal, PositionState, AccountState,
    BiasOutput, RiskOutput as RiskStructure, GameOutput,
    Direction,
    AdversarialRisk
)

# Alias for compatibility
SignalDirection = Direction

logger = logging.getLogger(__name__)


class ConnectionStatus(str, Enum):
    """Connection status values for UI."""
    LIVE = 'live'
    DEMO = 'demo'
    ERROR = 'error'
    OFFLINE = 'offline'


class SignalStatus(str, Enum):
    """Signal status for UI."""
    ACTIVE = 'ACTIVE'
    CLOSED = 'CLOSED'
    PENDING = 'PENDING'


# Position status mapping
class PositionStatus:
    """Position status constants."""
    OPEN = 'OPEN'
    CLOSED = 'CLOSED'
    CLOSED_WIN = 'CLOSED_WIN'
    CLOSED_LOSS = 'CLOSED_LOSS'
    PENDING = 'PENDING'


def format_direction_for_ui(direction: Direction) -> str:
    """Convert Direction to UI string."""
    return 'LONG' if direction == Direction.LONG else 'SHORT'


def format_position_status_for_ui(status: str) -> str:
    """Convert position status string to UI string."""
    if status == 'OPEN':
        return SignalStatus.ACTIVE.value
    elif status in ['CLOSED', 'CLOSED_WIN', 'CLOSED_LOSS']:
        return SignalStatus.CLOSED.value
    else:
        return SignalStatus.PENDING.value


def format_signal_for_ui(
    signal: EntrySignal,
    status: SignalStatus = SignalStatus.ACTIVE
) -> Dict[str, Any]:
    """Format EntrySignal for frontend UI display.
    
    Args:
        signal: EntrySignal from execution layer
        status: Signal status (ACTIVE/CLOSED/PENDING)
    
    Returns:
        Dict matching frontend UI expected structure
    """
    return {
        'symbol': signal.symbol,
        'direction': format_direction_for_ui(signal.direction),
        'confidence': round(signal.confidence * 100, 1),  # UI shows percentage
        'entry_price': signal.entry_price,
        'stop_loss': signal.stop_loss,
        'tp1': signal.tp1,
        'tp2': signal.tp2,
        'timestamp': signal.timestamp.isoformat() if hasattr(signal.timestamp, 'isoformat') else str(signal.timestamp),
        'status': status.value,
        # Additional fields for enhanced UI
        'rationale': signal.rationale,
        'position_size': signal.position_size
    }


def format_position_for_ui(position: PositionState) -> Dict[str, Any]:
    """Format PositionState for frontend UI display.
    
    Returns:
        Dict matching frontend UI expected structure
    """
    return {
        'symbol': position.symbol,
        'direction': format_direction_for_ui(position.direction),
        'entry_price': position.entry_price,
        'stop_loss': position.stop_loss,
        'tp1': position.tp1,
        'tp2': position.tp2,
        'current_price': position.current_price,
        'unrealized_pnl': position.unrealized_pnl,
        'realized_pnl': position.realized_pnl,
        'timestamp': position.opened_at.isoformat() if hasattr(position.opened_at, 'isoformat') else str(position.opened_at),
        'status': format_position_status_for_ui(position.status),
        'trade_id': position.trade_id,
        'position_size': position.position_size
    }


def format_account_state_for_ui(account: AccountState) -> Dict[str, Any]:
    """Format AccountState for frontend UI display.
    
    Returns:
        Dict with account metrics for UI
    """
    return {
        'account_id': account.account_id,
        'equity': round(account.equity, 2),
        'balance': round(account.balance, 2),
        'open_positions': account.open_positions,
        'daily_pnl': round(account.daily_pnl, 2),
        'daily_pnl_pct': round(account.daily_loss_pct * 100, 2),  # Convert to percentage
        'margin_used': round(account.margin_used, 2),
        'margin_available': round(account.margin_available, 2),
        'timestamp': account.timestamp.isoformat() if hasattr(account.timestamp, 'isoformat') else str(account.timestamp),
        'utilization_pct': round((account.margin_used / max(account.equity, 1)) * 100, 1)
    }


def format_bias_for_ui(symbol: str, bias: BiasOutput) -> Dict[str, Any]:
    """Format BiasOutput for frontend UI display.
    
    Args:
        symbol: Trading symbol
        bias: BiasOutput
    
    Returns:
        Dict with bias metrics for UI
    """
    return {
        'symbol': symbol,
        'direction': format_direction_for_ui(bias.direction),
        'magnitude': bias.magnitude,
        'confidence': round(bias.confidence * 100, 1),
        'rationale': bias.rationale,
        'regime_override': bias.regime_override,
        'model_version': bias.model_version,
        'timestamp': bias.timestamp.isoformat() if hasattr(bias.timestamp, 'isoformat') else str(bias.timestamp),
        'feature_summary': {
            'trend_strength': bias.feature_snapshot.raw_features.get('adx_14', 0),
            'momentum': bias.feature_snapshot.raw_features.get('rsi_14', 50),
            'volatility': bias.feature_snapshot.raw_features.get('atr_percent_14', 0)
        }
    }


def format_risk_for_ui(symbol: str, risk: RiskStructure) -> Dict[str, Any]:
    """Format RiskStructure for frontend UI display.
    
    Args:
        symbol: Trading symbol
        risk: RiskOutput
    
    Returns:
        Dict with risk metrics for UI
    """
    return {
        'symbol': symbol,
        'position_size': round(risk.position_size, 2),
        'kelly_fraction': round(risk.kelly_fraction * 100, 1),  # Percentage
        'stop_price': risk.stop_price,
        'stop_method': risk.stop_method,
        'tp1_price': risk.tp1_price,
        'tp2_price': risk.tp2_price,
        'expected_value': round(risk.expected_value, 4),
        'ev_positive': risk.ev_positive,
        'trail_config': risk.trail_config,
        'timestamp': risk.timestamp.isoformat() if hasattr(risk.timestamp, 'isoformat') else str(risk.timestamp)
    }


def format_game_output_for_ui(symbol: str, game: GameOutput) -> Dict[str, Any]:
    """Format GameOutput for frontend UI display.
    
    Args:
        symbol: Trading symbol
        game: GameOutput
    
    Returns:
        Dict with game state metrics for UI
    """
    return {
        'symbol': symbol,
        'liquidity_map': game.liquidity_map,
        'nearest_unswept_pool': game.nearest_unswept_pool,
        'trapped_positions': game.trapped_positions,
        'forced_move_probability': round(game.forced_move_probability * 100, 1),
        'nash_zones': game.nash_zones,
        'kyle_lambda': round(game.kyle_lambda, 4),
        'game_state_aligned': game.game_state_aligned,
        'game_state_summary': game.game_state_summary,
        'adversarial_risk': game.adversarial_risk.value if hasattr(game.adversarial_risk, 'value') else str(game.adversarial_risk),
        'timestamp': game.timestamp.isoformat() if hasattr(game.timestamp, 'isoformat') else str(game.timestamp)
    }


def format_connection_status(status: ConnectionStatus) -> Dict[str, Any]:
    """Format connection status for UI.
    
    Returns:
        Dict with status indicator
    """
    return {
        'status': status.value,
        'is_live': status == ConnectionStatus.LIVE,
        'is_demo': status == ConnectionStatus.DEMO,
        'is_error': status == ConnectionStatus.ERROR,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }


def format_regime_for_ui(
    vol_regime: int,
    vix_level: Optional[float] = None,
    event_risk: Optional[str] = None
) -> Dict[str, Any]:
    """Format regime data for UI status indicators.
    
    Args:
        vol_regime: Volatility regime (1-4)
        vix_level: Current VIX value
        event_risk: Event risk level
    
    Returns:
        Dict with regime indicators
    """
    regime_labels = {
        1: 'LOW',
        2: 'NORMAL',
        3: 'ELEVATED',
        4: 'EXTREME'
    }
    
    regime_colors = {
        1: 'green',
        2: 'blue',
        3: 'orange',
        4: 'red'
    }
    
    return {
        'volatility_regime': regime_labels.get(vol_regime, 'UNKNOWN'),
        'volatility_regime_code': vol_regime,
        'volatility_color': regime_colors.get(vol_regime, 'gray'),
        'vix_level': round(vix_level, 2) if vix_level else None,
        'event_risk': event_risk or 'CLEAR',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }


def create_live_state_update(
    symbol: str,
    signal: Optional[EntrySignal] = None,
    position: Optional[PositionState] = None,
    bias: Optional[BiasOutput] = None,
    regime: Optional[int] = None,
    vix_level: Optional[float] = None,
    event_risk: Optional[str] = None
) -> Dict[str, Any]:
    """Create complete live state update for a symbol.
    
    This is written to `/live_state/{symbol}/` in Firebase.
    
    Args:
        symbol: Trading symbol
        signal: Current entry signal (if any)
        position: Current position (if any)
        bias: Current bias output
        regime: Volatility regime
        vix_level: VIX value
        event_risk: Event risk level
    
    Returns:
        Dict ready for Firebase write
    """
    state = {
        'symbol': symbol,
        'last_updated': datetime.now(timezone.utc).isoformat()
    }
    
    if signal:
        state['current_signal'] = format_signal_for_ui(signal)
    
    if position:
        state['current_position'] = format_position_for_ui(position)
    
    if bias:
        state['bias'] = format_bias_for_ui(bias)
    
    if regime is not None:
        state['regime'] = format_regime_for_ui(regime, vix_level, event_risk)
    
    return state


def create_session_control_update(
    trading_enabled: bool,
    hard_logic_status: str = 'ACTIVE'
) -> Dict[str, Any]:
    """Create session control update.
    
    Written to `/session_controls/` in Firebase.
    
    Args:
        trading_enabled: Whether trading is enabled
        hard_logic_status: Hard logic status string
    
    Returns:
        Dict ready for Firebase write
    """
    return {
        'trading_enabled': trading_enabled,
        'hard_logic_status': hard_logic_status,
        'last_updated': datetime.now(timezone.utc).isoformat()
    }
