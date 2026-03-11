"""Firebase Broadcaster - Writes formatted data to Firebase for frontend consumption.

Handles all Firebase Realtime Database writes with correct paths and structures.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from integration.firebase_client import FirebaseClient
from integration.firebase_ui_writer import (
    format_signal_for_ui,
    format_position_for_ui,
    format_account_state_for_ui,
    format_bias_for_ui,
    format_risk_for_ui,
    format_game_output_for_ui,
    format_connection_status,
    format_regime_for_ui,
    create_live_state_update,
    create_session_control_update,
    ConnectionStatus,
    SignalStatus
)
from contracts.types import (
    EntrySignal, PositionState, AccountState,
    BiasOutput, RiskOutput, GameOutput
)

# Alias for compatibility
RiskStructure = RiskOutput

logger = logging.getLogger(__name__)


class FirebaseBroadcaster:
    """Broadcasts trading data to Firebase for frontend consumption.
    
    Firebase paths:
    - `/live_state/{symbol}/` - Current state per symbol
    - `/session_controls/trading_enabled` - Trading enabled flag
    - `/session_controls/hard_logic_status` - Hard logic status
    - `/connection_status/` - Connection status
    - `/regime_state/` - Market regime state
    - `entry_signals` collection - Signal feed (Firestore)
    - `positions` collection - Position history (Firestore)
    """
    
    def __init__(self, firebase_client: Optional[FirebaseClient] = None):
        self.client = firebase_client or FirebaseClient()
        self._connection_status = ConnectionStatus.OFFLINE
    
    def set_connection_status(self, status: ConnectionStatus):
        """Update connection status and broadcast to Firebase."""
        self._connection_status = status
        self.broadcast_connection_status(status)
    
    def broadcast_connection_status(self, status: ConnectionStatus):
        """Broadcast connection status to Firebase.
        
        Path: `/connection_status/`
        """
        try:
            status_data = format_connection_status(status)
            self.client.rtdb_update('/connection_status', status_data)
            logger.debug(f"Broadcast connection status: {status.value}")
        except Exception as e:
            logger.error(f"Failed to broadcast connection status: {e}")
    
    def broadcast_session_control(
        self,
        trading_enabled: bool,
        hard_logic_status: str = 'ACTIVE'
    ):
        """Broadcast session control state.
        
        Path: `/session_controls/`
        """
        try:
            control_data = create_session_control_update(
                trading_enabled,
                hard_logic_status
            )
            self.client.rtdb_update('/session_controls', control_data)
            logger.debug(f"Broadcast session control: trading_enabled={trading_enabled}")
        except Exception as e:
            logger.error(f"Failed to broadcast session control: {e}")
    
    def broadcast_live_state(
        self,
        symbol: str,
        signal: Optional[EntrySignal] = None,
        position: Optional[PositionState] = None,
        bias: Optional[BiasOutput] = None,
        regime: Optional[int] = None,
        vix_level: Optional[float] = None,
        event_risk: Optional[str] = None
    ):
        """Broadcast live state for a symbol.
        
        Path: `/live_state/{symbol}/`
        """
        try:
            state_data = create_live_state_update(
                symbol=symbol,
                signal=signal,
                position=position,
                bias=bias,
                regime=regime,
                vix_level=vix_level,
                event_risk=event_risk
            )
            self.client.rtdb_update(f'/live_state/{symbol}', state_data)
            logger.debug(f"Broadcast live state for {symbol}")
        except Exception as e:
            logger.error(f"Failed to broadcast live state for {symbol}: {e}")
    
    def broadcast_regime_state(
        self,
        vol_regime: int,
        vix_level: Optional[float] = None,
        event_risk: Optional[str] = None,
        breadth_ratio: Optional[float] = None
    ):
        """Broadcast market regime state.
        
        Path: `/regime_state/`
        """
        try:
            regime_data = format_regime_for_ui(vol_regime, vix_level, event_risk)
            if breadth_ratio is not None:
                regime_data['breadth_ratio'] = round(breadth_ratio, 2)
            
            self.client.rtdb_update('/regime_state', regime_data)
            logger.debug(f"Broadcast regime state: {vol_regime}")
        except Exception as e:
            logger.error(f"Failed to broadcast regime state: {e}")
    
    def broadcast_entry_signal(
        self,
        signal: EntrySignal,
        status: SignalStatus = SignalStatus.ACTIVE
    ):
        """Broadcast new entry signal to Firestore collection.
        
        Collection: `entry_signals`
        Also updates `/live_state/{symbol}/current_signal`
        """
        try:
            # Format for UI
            signal_data = format_signal_for_ui(signal, status)
            
            # Add to Firestore collection (signal feed)
            doc_id = f"{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"
            self.client.firestore_set(f'entry_signals/{doc_id}', signal_data)
            
            # Update live state
            self.broadcast_live_state(signal.symbol, signal=signal)
            
            logger.info(f"Broadcast entry signal: {signal.symbol} {signal.direction.name}")
        except Exception as e:
            logger.error(f"Failed to broadcast entry signal: {e}")
    
    def broadcast_position_update(self, position: PositionState):
        """Broadcast position update.
        
        Collection: `positions`
        Also updates `/live_state/{symbol}/current_position`
        """
        try:
            # Format for UI
            position_data = format_position_for_ui(position)
            
            # Update in Firestore
            doc_id = position.trade_id
            self.client.firestore_set(f'positions/{doc_id}', position_data)
            
            # Update live state
            self.broadcast_live_state(position.symbol, position=position)
            
            logger.debug(f"Broadcast position update: {position.trade_id}")
        except Exception as e:
            logger.error(f"Failed to broadcast position update: {e}")
    
    def broadcast_account_state(self, account: AccountState):
        """Broadcast account state.
        
        Path: `/account_state/`
        """
        try:
            account_data = format_account_state_for_ui(account)
            self.client.rtdb_update('/account_state', account_data)
            logger.debug(f"Broadcast account state: equity={account.equity}")
        except Exception as e:
            logger.error(f"Failed to broadcast account state: {e}")
    
    def broadcast_bias(self, symbol: str, bias: BiasOutput):
        """Broadcast bias output.
        
        Collection: `bias_outputs`
        Also updates `/live_state/{symbol}/bias`
        """
        try:
            bias_data = format_bias_for_ui(symbol, bias)
            
            # Add to Firestore
            doc_id = f"{symbol}_{bias.timestamp.strftime('%Y%m%d_%H%M%S')}"
            self.client.firestore_set(f'bias_outputs/{doc_id}', bias_data)
            
            # Update live state
            self.broadcast_live_state(symbol, bias=bias)
            
            logger.debug(f"Broadcast bias for {symbol}: {bias.direction.name}")
        except Exception as e:
            logger.error(f"Failed to broadcast bias: {e}")
    
    def broadcast_risk_structure(self, symbol: str, risk: RiskStructure):
        """Broadcast risk structure.
        
        Collection: `risk_structures`
        """
        try:
            risk_data = format_risk_for_ui(symbol, risk)
            
            doc_id = f"{symbol}_{risk.timestamp.strftime('%Y%m%d_%H%M%S')}"
            self.client.firestore_set(f'risk_structures/{doc_id}', risk_data)
            
            logger.debug(f"Broadcast risk structure for {symbol}")
        except Exception as e:
            logger.error(f"Failed to broadcast risk structure: {e}")
    
    def broadcast_game_output(self, symbol: str, game: GameOutput):
        """Broadcast game output.
        
        Collection: `game_outputs`
        """
        try:
            game_data = format_game_output_for_ui(symbol, game)
            
            doc_id = f"{symbol}_{game.timestamp.strftime('%Y%m%d_%H%M%S')}"
            self.client.firestore_set(f'game_outputs/{doc_id}', game_data)
            
            logger.debug(f"Broadcast game output for {symbol}")
        except Exception as e:
            logger.error(f"Failed to broadcast game output: {e}")
    
    def broadcast_full_state(
        self,
        symbol: str,
        signal: Optional[EntrySignal] = None,
        position: Optional[PositionState] = None,
        bias: Optional[BiasOutput] = None,
        risk: Optional[RiskStructure] = None,
        game: Optional[GameOutput] = None,
        regime: Optional[int] = None,
        vix_level: Optional[float] = None,
        event_risk: Optional[str] = None
    ):
        """Broadcast complete state for a symbol.
        
        Updates all relevant Firebase paths for comprehensive UI state.
        """
        try:
            # Build complete live state
            state_data = create_live_state_update(
                symbol=symbol,
                signal=signal,
                position=position,
                bias=bias,
                regime=regime,
                vix_level=vix_level,
                event_risk=event_risk
            )
            
            # Add risk and game if present
            if risk:
                state_data['risk'] = format_risk_for_ui(risk)
            if game:
                state_data['game'] = format_game_output_for_ui(game)
            
            # Write to Realtime DB
            self.client.rtdb_update(f'/live_state/{symbol}', state_data)
            
            # Write individual collections
            if signal:
                self.broadcast_entry_signal(signal)
            if position:
                self.broadcast_position_update(position)
            if bias:
                self.broadcast_bias(symbol, bias)
            if risk:
                self.broadcast_risk_structure(symbol, risk)
            if game:
                self.broadcast_game_output(symbol, game)
            
            logger.info(f"Broadcast full state for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to broadcast full state for {symbol}: {e}")
    
    def clear_signal(self, symbol: str):
        """Clear current signal for a symbol (when closed or cancelled).
        
        Path: `/live_state/{symbol}/current_signal`
        """
        try:
            self.client.rtdb_update(f'/live_state/{symbol}/current_signal', None)
            logger.debug(f"Cleared signal for {symbol}")
        except Exception as e:
            logger.error(f"Failed to clear signal for {symbol}: {e}")
    
    def initialize_ui_state(self, symbols: List[str]):
        """Initialize UI state for all symbols.
        
        Called on startup to set initial state.
        """
        logger.info(f"Initializing UI state for {symbols}")
        
        # Set connection status
        self.set_connection_status(ConnectionStatus.DEMO)
        
        # Set session controls
        self.broadcast_session_control(trading_enabled=True, hard_logic_status='ACTIVE')
        
        # Initialize live state for each symbol
        for symbol in symbols:
            self.broadcast_live_state(symbol)
        
        # Initialize regime state
        self.broadcast_regime_state(vol_regime=2)  # NORMAL
        
        logger.info("UI state initialized")
