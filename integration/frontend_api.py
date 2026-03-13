"""Frontend API - Anthropic Maid Dashboard Connector

Formats three-layer signals for the frontend dashboard.
Provides REST API endpoints for the dashboard to consume.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict

from contracts.types import (
    BiasOutput, RiskOutput, GameOutput, RegimeState,
    EntrySignal, ThreeLayerContext, Direction
)
from firebase.client import FirebaseClient

logger = logging.getLogger(__name__)


@dataclass
class DashboardSignal:
    """Signal formatted for frontend dashboard."""
    symbol: str
    direction: str  # 'LONG', 'SHORT', or 'NEUTRAL'
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    position_size: float
    confidence: float
    risk_reward_ratio: float
    expected_value: float
    rationale: List[str]
    timestamp: str
    status: str  # 'PENDING', 'ACTIVE', 'CLOSED'
    
    # Layer breakdown
    layer1_bias: Dict[str, Any]
    layer2_risk: Dict[str, Any]
    layer3_game: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FrontendAPI:
    """API layer for frontend dashboard integration."""
    
    def __init__(self, firebase_client: Optional[FirebaseClient] = None):
        self.firebase = firebase_client or FirebaseClient()
        self.logger = logging.getLogger(__name__)
    
    def get_latest_signal(self, symbol: str) -> Optional[DashboardSignal]:
        """Get latest signal for a symbol formatted for dashboard.
        
        Args:
            symbol: Trading symbol (e.g., 'NAS100')
        
        Returns:
            DashboardSignal or None if no signal
        """
        try:
            # Fetch latest signal from Firebase Realtime DB
            latest = self.firebase.read_realtime(f'/signals/{symbol}/latest')
            
            if not latest:
                return None
            
            # Fetch full context from Firestore
            signal_id = latest.get('signal_id')
            if signal_id:
                context = self.firebase.read('entry_signals', signal_id)
            else:
                context = None
            
            return self._format_dashboard_signal(symbol, latest, context)
            
        except Exception as e:
            self.logger.error(f"Failed to get latest signal: {e}")
            return None
    
    def get_all_active_signals(self) -> List[DashboardSignal]:
        """Get all active signals for dashboard."""
        symbols = ['NAS100', 'US30', 'SPX500', 'XAUUSD']
        signals = []
        
        for symbol in symbols:
            signal = self.get_latest_signal(symbol)
            if signal and signal.status == 'PENDING':
                signals.append(signal)
        
        return signals
    
    def get_signal_history(self, symbol: str, limit: int = 10) -> List[DashboardSignal]:
        """Get signal history for a symbol."""
        try:
            # Query Firestore for recent signals
            signals_data = self.firebase.query(
                collection='entry_signals',
                filters=[{'field': 'symbol', 'op': '==', 'value': symbol}],
                order_by='timestamp',
                direction='DESCENDING',
                limit=limit
            )
            
            signals = []
            for data in signals_data:
                signal = self._format_dashboard_signal(symbol, data, data)
                if signal:
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Failed to get signal history: {e}")
            return []
    
    def get_layer_breakdown(self, symbol: str) -> Dict[str, Any]:
        """Get detailed three-layer breakdown for dashboard.
        
        Returns:
            Dict with Layer 1, 2, 3 details
        """
        try:
            # Fetch latest from each layer
            bias = self.firebase.read_realtime(f'/live_state/{symbol}/bias')
            risk = self.firebase.read_realtime(f'/live_state/{symbol}/risk')
            game = self.firebase.read_realtime(f'/live_state/{symbol}/game')
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'layer1': bias or {},
                'layer2': risk or {},
                'layer3': game or {},
                'agreement': self._check_agreement(bias, risk, game)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get layer breakdown: {e}")
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status for dashboard."""
        try:
            health = self.firebase.read_realtime('/health')
            controls = self.firebase.read_realtime('/session_controls')
            
            return {
                'status': health.get('status', 'unknown') if health else 'unknown',
                'trading_enabled': controls.get('trading_enabled', False) if controls else False,
                'open_positions': controls.get('open_positions', 0) if controls else 0,
                'daily_pnl': controls.get('daily_pnl', 0.0) if controls else 0.0,
                'hard_logic_status': controls.get('hard_logic_status', 'unknown') if controls else 'unknown',
                'last_update': health.get('timestamp') if health else None,
                'components': health.get('components', {}) if health else {}
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _format_dashboard_signal(
        self,
        symbol: str,
        latest: Dict,
        context: Optional[Dict]
    ) -> Optional[DashboardSignal]:
        """Format raw signal data for dashboard."""
        try:
            # Calculate risk/reward
            entry = latest.get('entry_price', 0)
            stop = latest.get('stop_loss', 0)
            tp1 = latest.get('tp1', 0)
            
            if entry > 0 and stop > 0:
                risk = abs(entry - stop)
                reward = abs(tp1 - entry)
                rr_ratio = reward / risk if risk > 0 else 0.0
            else:
                rr_ratio = 0.0
            
            # Get expected value from context
            ev = 0.0
            if context and 'risk' in context:
                ev = context['risk'].get('expected_value', 0.0)
            
            # Determine direction string
            direction_val = latest.get('direction', 0)
            direction_str = 'LONG' if direction_val == 1 else 'SHORT' if direction_val == -1 else 'NEUTRAL'
            
            return DashboardSignal(
                symbol=symbol,
                direction=direction_str,
                entry_price=entry,
                stop_loss=stop,
                take_profit_1=tp1,
                take_profit_2=latest.get('tp2', 0),
                position_size=latest.get('position_size', 0),
                confidence=latest.get('confidence', 0.0),
                risk_reward_ratio=rr_ratio,
                expected_value=ev,
                rationale=latest.get('rationale', []),
                timestamp=latest.get('timestamp', datetime.now().isoformat()),
                status=latest.get('status', 'PENDING'),
                layer1_bias=context.get('bias', {}) if context else {},
                layer2_risk=context.get('risk', {}) if context else {},
                layer3_game=context.get('game', {}) if context else {}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to format dashboard signal: {e}")
            return None
    
    def _check_agreement(
        self,
        bias: Optional[Dict],
        risk: Optional[Dict],
        game: Optional[Dict]
    ) -> Dict[str, Any]:
        """Check three-layer agreement."""
        if not bias or not risk or not game:
            return {'aligned': False, 'reason': 'Missing layer data'}
        
        # Check Layer 1
        l1_ok = (
            bias.get('direction', 0) != 0 and
            bias.get('confidence', 0) >= 0.55
        )
        
        # Check Layer 2
        l2_ok = risk.get('ev_positive', False)
        
        # Check Layer 3
        l3_ok = (
            game.get('game_state_aligned', False) or
            game.get('adversarial_risk', 'LOW') != 'EXTREME'
        )
        
        aligned = l1_ok and l2_ok and l3_ok
        
        return {
            'aligned': aligned,
            'layer1_ok': l1_ok,
            'layer2_ok': l2_ok,
            'layer3_ok': l3_ok,
            'reason': 'All layers aligned' if aligned else 'Layer mismatch'
        }


class DashboardWebSocket:
    """WebSocket handler for real-time dashboard updates."""
    
    def __init__(self, firebase_client: FirebaseClient):
        self.firebase = firebase_client
        self.clients = set()
        self.logger = logging.getLogger(__name__)
    
    def add_client(self, client) -> None:
        """Add WebSocket client."""
        self.clients.add(client)
        self.logger.info(f"Client connected. Total: {len(self.clients)}")
    
    def remove_client(self, client) -> None:
        """Remove WebSocket client."""
        self.clients.discard(client)
        self.logger.info(f"Client disconnected. Total: {len(self.clients)}")
    
    def broadcast_signal(self, signal: DashboardSignal) -> None:
        """Broadcast new signal to all connected clients."""
        message = {
            'type': 'NEW_SIGNAL',
            'data': signal.to_dict()
        }
        
        for client in self.clients:
            try:
                client.send_json(message)
            except Exception as e:
                self.logger.error(f"Failed to send to client: {e}")
    
    def broadcast_status_update(self, status: Dict[str, Any]) -> None:
        """Broadcast system status update."""
        message = {
            'type': 'STATUS_UPDATE',
            'data': status
        }
        
        for client in self.clients:
            try:
                client.send_json(message)
            except Exception as e:
                self.logger.error(f"Failed to send status to client: {e}")


def create_frontend_api() -> FrontendAPI:
    """Factory function to create frontend API."""
    return FrontendAPI()


# Flask/Express route handlers (for reference)
"""
# Python Flask example:
from flask import Flask, jsonify
app = Flask(__name__)
api = create_frontend_api()

@app.route('/api/signals/<symbol>')
def get_signal(symbol):
    signal = api.get_latest_signal(symbol)
    if signal:
        return jsonify(signal.to_dict())
    return jsonify({'error': 'No signal'}), 404

@app.route('/api/signals')
def get_all_signals():
    signals = api.get_all_active_signals()
    return jsonify([s.to_dict() for s in signals])

@app.route('/api/layers/<symbol>')
def get_layers(symbol):
    breakdown = api.get_layer_breakdown(symbol)
    return jsonify(breakdown)

@app.route('/api/status')
def get_status():
    status = api.get_system_status()
    return jsonify(status)
"""
