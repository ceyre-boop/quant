"""ICT AMD Strategy Wrapper - Phase 13 Implementation

Wraps the existing ICT AMD Swing NAS100 strategy to consume three-layer context
and output frontend-compatible signals.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from contracts.types import (
    Direction, BiasOutput, RiskOutput, GameOutput, RegimeState,
    AccountState, EntrySignal, ThreeLayerContext
)
from firebase.client import FirebaseClient

logger = logging.getLogger(__name__)


@dataclass
class ICTPattern:
    """ICT pattern detection result."""
    pattern_type: str  # 'AMD', 'FVG', 'OB', 'Breaker'
    direction: str  # 'long', 'short'
    valid: bool
    entry_price: float
    stop_loss: float
    confidence: float
    metadata: Dict[str, Any]


class ICTAMDWrapper:
    """Wrapper for ICT AMD Swing NAS100 strategy.
    
    Integrates the existing ICT AMD strategy with the three-layer system.
    Does NOT modify the original strategy - only wraps it.
    """
    
    def __init__(self, firebase_client: Optional[FirebaseClient] = None):
        self.firebase = firebase_client or FirebaseClient()
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters (from ict_amd_params.json)
        self.params = self._load_strategy_params()
    
    def _load_strategy_params(self) -> Dict[str, Any]:
        """Load ICT AMD strategy parameters."""
        return {
            'risk_per_trade': 0.005,  # 0.5% default (overridden by Layer 2)
            'max_concurrent_trades': 1,
            'time_stop_candles': 15,
            'min_confidence': 0.55,
            'session': 'NY',
            'use_amd': True,
            'use_fvg': True,
            'use_ob': True,
            'use_breaker': True
        }
    
    def on_bar(
        self,
        h4_data: Any,
        daily_data: Any,
        three_layer_context: ThreeLayerContext,
        current_price: float,
        ict_setup: Optional[Dict] = None
    ) -> Optional[EntrySignal]:
        """Process bar with three-layer context.
        
        Args:
            h4_data: 4-hour OHLCV data
            daily_data: Daily OHLCV data
            three_layer_context: Aggregated context from all three layers
            current_price: Current market price
            ict_setup: Optional ICT pattern detection results
        
        Returns:
            EntrySignal if all conditions met, None otherwise
        """
        # Gate 1: Three-layer agreement check
        if not three_layer_context.all_aligned():
            reason = three_layer_context.block_reason()
            self._log_blocked_signal('THREE_LAYER_MISMATCH', reason, three_layer_context)
            self.logger.info(f"Strategy blocked: {reason}")
            return None
        
        # Gate 2: Minimum confidence check
        if three_layer_context.bias.confidence < self.params['min_confidence']:
            self._log_blocked_signal(
                'CONFIDENCE_TOO_LOW',
                f"Confidence {three_layer_context.bias.confidence:.2f} < {self.params['min_confidence']}",
                three_layer_context
            )
            return None
        
        # Gate 3: ICT pattern validation (if provided)
        if ict_setup and not ict_setup.get('valid', False):
            self._log_blocked_signal('ICT_PATTERN_INVALID', 'Pattern validation failed', three_layer_context)
            return None
        
        # Gate 4: Layer 3 adversarial check (EXTREME risk blocks)
        if (three_layer_context.game.adversarial_risk == 'EXTREME' and 
            not three_layer_context.game.game_state_aligned):
            self._log_blocked_signal('LAYER3_VETO', 'EXTREME adversarial risk', three_layer_context)
            return None
        
        # All gates passed - build entry signal
        signal = self._build_entry_signal(
            h4_data, daily_data, three_layer_context, current_price, ict_setup
        )
        
        # Write to Firebase
        self._write_signal_to_firebase(signal)
        
        self.logger.info(f"ENTRY SIGNAL GENERATED: {signal.symbol} {signal.direction.name}")
        return signal
    
    def _build_entry_signal(
        self,
        h4_data: Any,
        daily_data: Any,
        context: ThreeLayerContext,
        current_price: float,
        ict_setup: Optional[Dict]
    ) -> EntrySignal:
        """Build entry signal with Layer 2 overrides."""
        
        # Use Layer 2 risk structure (overrides strategy defaults)
        position_size = context.risk.position_size
        stop_loss = context.risk.stop_price
        tp1 = context.risk.tp1_price
        tp2 = context.risk.tp2_price
        
        # If ICT setup provides better structural levels, compare and use best
        if ict_setup:
            ict_stop = ict_setup.get('stop_loss')
            ict_tp1 = ict_setup.get('tp1')
            ict_tp2 = ict_setup.get('tp2')
            
            # Use whichever stop is tighter (more conservative)
            if ict_stop:
                if context.bias.direction == Direction.LONG:
                    stop_loss = max(stop_loss, ict_stop)  # Higher stop for longs
                else:
                    stop_loss = min(stop_loss, ict_stop)  # Lower stop for shorts
            
            # TP1: Use Layer 2's 1R calculation (mathematically optimal)
            # But cap at structural daily high/low if ICT provides it
            if ict_tp2 and context.risk.tp2_price > ict_tp2:
                tp2 = ict_tp2  # Structural cap
        
        # Build rationale
        rationale = self._build_rationale(context, ict_setup)
        
        return EntrySignal(
            symbol=getattr(h4_data, 'symbol', 'NAS100'),
            direction=context.bias.direction,
            entry_price=current_price,
            position_size=position_size,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            confidence=context.bias.confidence,
            rationale=rationale,
            timestamp=datetime.utcnow(),
            layer_context=context
        )
    
    def _build_rationale(
        self,
        context: ThreeLayerContext,
        ict_setup: Optional[Dict]
    ) -> List[str]:
        """Build comprehensive rationale for the signal."""
        rationale = []
        
        # Layer 1 rationale
        rationale.extend(context.bias.rationale)
        
        # Layer 2 rationale
        rationale.append(f"EV_{context.risk.expected_value:.2f}")
        rationale.append(f"KELLY_{context.risk.kelly_fraction:.2f}")
        rationale.append(f"STOP_{context.risk.stop_method.upper()}")
        
        # Layer 3 rationale
        if context.game.game_state_aligned:
            rationale.append("GAME_ALIGNED")
        
        if context.game.nearest_unswept_pool:
            pool = context.game.nearest_unswept_pool
            rationale.append(f"LIQUIDITY_{pool.pool_type.upper()}")
        
        if context.game.trapped_positions.trapped_shorts:
            rationale.append("SHORTS_TRAPPED")
        if context.game.trapped_positions.trapped_longs:
            rationale.append("LONGS_TRAPPED")
        
        # ICT rationale
        if ict_setup:
            pattern = ict_setup.get('pattern', 'UNKNOWN')
            rationale.append(f"ICT_{pattern.upper()}")
        
        return rationale
    
    def _write_signal_to_firebase(self, signal: EntrySignal) -> None:
        """Write entry signal to Firebase."""
        try:
            signal_data = signal.to_dict()
            timestamp = signal.timestamp
            
            # Write to entry_signals collection
            doc_id = f"{signal.symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            self.firebase.write('entry_signals', doc_id, signal_data)
            
            # Archive in signal history (for later analysis)
            history_path = f"signals_history/{timestamp.strftime('%Y-%m')}/{timestamp.strftime('%Y-%m-%d')}"
            history_id = f"signal_{timestamp.strftime('%H%M%S')}_{signal.symbol}"
            self.firebase.write(history_path, history_id, {
                **signal_data,
                'outcome': 'PENDING',
                'realized_pnl': None,
                'closed_at': None
            })
            
            # Also update latest signal in Realtime DB
            self.firebase.update_realtime(f'/signals/{signal.symbol}/latest', {
                'direction': signal.direction.value,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'tp1': signal.tp1,
                'tp2': signal.tp2,
                'position_size': signal.position_size,
                'confidence': signal.confidence,
                'rationale': signal.rationale,
                'timestamp': timestamp.isoformat(),
                'status': 'PENDING_EXECUTION',
                'signal_id': doc_id
            })
            
            # Update signal count
            current_count = self.firebase.read_realtime(f'/signals/{signal.symbol}/count_today') or 0
            self.firebase.update_realtime(f'/signals/{signal.symbol}/count_today', current_count + 1)
            
            self.logger.debug(f"Signal written to Firebase: {doc_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to write signal to Firebase: {e}")
    
    def _log_blocked_signal(
        self,
        reason_code: str,
        reason_detail: str,
        context: ThreeLayerContext
    ) -> None:
        """Log blocked signal to Firebase for analysis."""
        try:
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'reason_code': reason_code,
                'reason_detail': reason_detail,
                'bias_direction': context.bias.direction.name,
                'bias_confidence': context.bias.confidence,
                'ev_positive': context.risk.ev_positive,
                'game_aligned': context.game.game_state_aligned,
                'adversarial_risk': context.game.adversarial_risk
            }
            
            self.firebase.write('blocked_signals', None, log_entry)
            
        except Exception as e:
            self.logger.error(f"Failed to log blocked signal: {e}")


class StrategyIntegration:
    """High-level strategy integration coordinator.
    
    Connects the three-layer system with the ICT AMD strategy
    and the frontend dashboard.
    """
    
    def __init__(self):
        self.wrapper = ICTAMDWrapper()
        self.logger = logging.getLogger(__name__)
    
    def process_signal(
        self,
        symbol: str,
        context: ThreeLayerContext,
        market_data: Any,
        ict_setup: Optional[Dict] = None
    ) -> Optional[EntrySignal]:
        """Process a complete signal through the integration layer.
        
        This is the main entry point for signal generation.
        """
        current_price = getattr(market_data, 'current_price', 0)
        
        return self.wrapper.on_bar(
            h4_data=market_data,
            daily_data=market_data,
            three_layer_context=context,
            current_price=current_price,
            ict_setup=ict_setup
        )
    
    def get_signal_for_dashboard(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest signal formatted for frontend dashboard.
        
        Returns:
            Dict with signal data formatted for the Anthropic Maid frontend
        """
        try:
            # Fetch latest signal from Firebase
            latest = self.wrapper.firebase.read_realtime(f'/signals/{symbol}/latest')
            
            if not latest:
                return None
            
            # Format for dashboard
            return {
                'symbol': symbol,
                'direction': 'LONG' if latest.get('direction') == 1 else 'SHORT',
                'entry_price': latest.get('entry_price'),
                'stop_loss': latest.get('stop_loss'),
                'take_profit_1': latest.get('tp1'),
                'take_profit_2': latest.get('tp2'),
                'position_size': latest.get('position_size'),
                'confidence': latest.get('confidence'),
                'rationale': latest.get('rationale', []),
                'status': latest.get('status'),
                'timestamp': latest.get('timestamp'),
                'risk_reward': self._calculate_rr(latest),
                'expected_value': self._get_ev(latest)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard signal: {e}")
            return None
    
    def _calculate_rr(self, signal_data: Dict) -> float:
        """Calculate risk/reward ratio."""
        entry = signal_data.get('entry_price', 0)
        stop = signal_data.get('stop_loss', 0)
        tp1 = signal_data.get('tp1', 0)
        
        if entry == 0 or stop == 0:
            return 0.0
        
        risk = abs(entry - stop)
        reward = abs(tp1 - entry)
        
        return reward / risk if risk > 0 else 0.0
    
    def _get_ev(self, signal_data: Dict) -> float:
        """Get expected value from signal context."""
        # EV would be stored in the layer_context
        # This is a simplified version
        return signal_data.get('expected_value', 0.0)


def create_integration_layer() -> StrategyIntegration:
    """Factory function to create the strategy integration layer."""
    return StrategyIntegration()
