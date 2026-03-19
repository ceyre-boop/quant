"""Feature Tracker - Captures intermediate feature values.

Tracks features at each layer for debugging and explainability.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from integration.firebase_client import FirebaseClient

logger = logging.getLogger(__name__)


@dataclass
class FeatureSnapshot:
    """Snapshot of features at a point in time."""
    timestamp: datetime
    symbol: str
    price: float
    
    # Layer 1: Bias features
    volatility_regime: str
    momentum_score: float
    trend_strength: float
    rsi_14: float
    macd_signal: float
    
    # Layer 2: Risk features
    liquidity_score: float
    spread_pct: float
    volume_profile: str
    adr_pct: float  # Average daily range
    
    # Layer 3: Game features
    microstructure_score: float
    order_imbalance: float
    trapped_positions: int
    absorption_level: float
    
    # Meta
    vix_level: float
    market_regime: int


class FeatureTracker:
    """Tracks and broadcasts intermediate feature values.
    
    Pushes features to Firebase at /features/{symbol}/ for live monitoring.
    Also archives for later analysis.
    
    Usage:
        tracker = FeatureTracker()
        
        # Record features
        tracker.capture_features(
            symbol='NQ',
            layer1_features={'volatility_regime': 'normal', ...},
            layer2_features={'liquidity_score': 0.85, ...},
            ...
        )
    """
    
    def __init__(self, firebase_client: Optional[FirebaseClient] = None):
        self.firebase = firebase_client or FirebaseClient()
    
    def capture_features(
        self,
        symbol: str,
        price: float,
        layer1_features: Optional[Dict[str, Any]] = None,
        layer2_features: Optional[Dict[str, Any]] = None,
        layer3_features: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None
    ):
        """Capture and broadcast feature snapshot.
        
        Args:
            symbol: Trading symbol
            price: Current price
            layer1_features: Bias model features
            layer2_features: Risk model features
            layer3_features: Game theory features
            market_context: VIX, regime, etc.
        """
        timestamp = datetime.now()
        
        # Build feature snapshot
        snapshot = {
            'timestamp': timestamp.isoformat(),
            'symbol': symbol,
            'price': price,
            'captured_at': timestamp.strftime('%H:%M:%S')
        }
        
        # Layer 1 features
        if layer1_features:
            snapshot['layer1_bias'] = {
                'volatility_regime': layer1_features.get('volatility_regime', 'unknown'),
                'momentum_score': round(layer1_features.get('momentum_score', 0), 4),
                'trend_strength': round(layer1_features.get('trend_strength', 0), 4),
                'rsi_14': round(layer1_features.get('rsi_14', 50), 2),
                'macd_signal': round(layer1_features.get('macd_signal', 0), 4)
            }
        
        # Layer 2 features
        if layer2_features:
            snapshot['layer2_risk'] = {
                'liquidity_score': round(layer2_features.get('liquidity_score', 0), 4),
                'spread_pct': round(layer2_features.get('spread_pct', 0), 4),
                'volume_profile': layer2_features.get('volume_profile', 'normal'),
                'adr_pct': round(layer2_features.get('adr_pct', 0), 4)
            }
        
        # Layer 3 features
        if layer3_features:
            snapshot['layer3_game'] = {
                'microstructure_score': round(layer3_features.get('microstructure_score', 0), 4),
                'order_imbalance': round(layer3_features.get('order_imbalance', 0), 4),
                'trapped_positions': layer3_features.get('trapped_positions', 0),
                'absorption_level': round(layer3_features.get('absorption_level', 0), 4)
            }
        
        # Market context
        if market_context:
            snapshot['market_context'] = {
                'vix': market_context.get('vix', 0),
                'market_regime': market_context.get('market_regime', 0),
                'correlation_regime': market_context.get('correlation_regime', 'normal')
            }
        
        # Broadcast to Firebase
        self._broadcast(symbol, snapshot)
        
        return snapshot
    
    def _broadcast(self, symbol: str, snapshot: Dict[str, Any]):
        """Push snapshot to Firebase."""
        try:
            # Current features
            path = f"/features/{symbol}/current"
            self.firebase.rtdb_update(path, snapshot)
            
            # History (last 100)
            history_path = f"/features/{symbol}/history"
            self.firebase.rtdb_push(history_path, snapshot)
            
            logger.debug(f"Features broadcast for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to broadcast features: {e}")
    
    def capture_raw_features(
        self,
        symbol: str,
        features: Dict[str, float],
        feature_names: Optional[list] = None
    ):
        """Capture raw 43-feature vector.
        
        Args:
            symbol: Trading symbol
            features: Dict of feature_name -> value
            feature_names: Optional list of all 43 feature names
        """
        # Store raw features for debugging
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'raw_features': features,
            'feature_count': len(features)
        }
        
        try:
            path = f"/features/{symbol}/raw"
            self.firebase.rtdb_update(path, snapshot)
        except Exception as e:
            logger.error(f"Failed to capture raw features: {e}")


class LiveFeatureMonitor:
    """Live monitoring dashboard for features.
    
    Provides real-time view of what the models are "seeing".
    """
    
    def __init__(self, firebase_client: Optional[FirebaseClient] = None):
        self.firebase = firebase_client or FirebaseClient()
    
    def get_current_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current feature snapshot for a symbol."""
        try:
            return self.firebase.rtdb_get(f"/features/{symbol}/current")
        except Exception as e:
            logger.error(f"Failed to get features: {e}")
            return None
    
    def get_feature_history(
        self,
        symbol: str,
        limit: int = 100
    ) -> list:
        """Get recent feature history."""
        try:
            history = self.firebase.rtdb_get(f"/features/{symbol}/history")
            if isinstance(history, dict):
                # Convert to list
                items = list(history.values())
                return items[-limit:]
            return []
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []
    
    def get_feature_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary of current feature state."""
        current = self.get_current_features(symbol)
        
        if not current:
            return {'error': 'No feature data available'}
        
        # Extract key indicators
        summary = {
            'symbol': symbol,
            'timestamp': current.get('timestamp'),
            'price': current.get('price'),
            'indicators': {}
        }
        
        layer1 = current.get('layer1_bias', {})
        layer2 = current.get('layer2_risk', {})
        layer3 = current.get('layer3_game', {})
        market = current.get('market_context', {})
        
        # Key indicators
        summary['indicators'] = {
            'volatility_regime': layer1.get('volatility_regime'),
            'momentum': 'bullish' if layer1.get('momentum_score', 0) > 0 else 'bearish',
            'liquidity': 'high' if layer2.get('liquidity_score', 0) > 0.7 else 'low',
            'microstructure': 'favorable' if layer3.get('microstructure_score', 0) > 0.6 else 'unfavorable',
            'vix': market.get('vix'),
            'market_regime': market.get('market_regime')
        }
        
        return summary
