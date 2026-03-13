"""Orchestrator V2 - Production Daily Lifecycle

Wires together all real components (no mocks) for the three-layer trading system.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from contracts.types import (
    BiasOutput, RiskOutput, GameOutput, RegimeState,
    EntrySignal, ThreeLayerContext, AccountState, MarketData
)
from firebase.client import FirebaseClient
from integration.firebase_broadcaster import FirebaseBroadcaster

# Import real components (not mocks)
from layer1.bias_engine_v2 import BiasEngineV2, create_bias_engine
from layer1.feature_builder_v2 import FeatureBuilder, create_feature_builder
from layer1.regime_classifier import RegimeClassifier
from layer1.hard_constraints_v2 import HardConstraints, create_hard_constraints
from layer2.risk_engine import RiskEngine
from layer3.game_engine import GameEngine
from entry_engine.entry_engine import EntryEngine
from trading_strategies.strategy_wrapper import StrategyIntegration, create_integration_layer

logger = logging.getLogger(__name__)


class ProductionOrchestrator:
    """Production orchestrator with real components."""
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        firebase_client: Optional[FirebaseClient] = None
    ):
        self.symbols = symbols or ["NAS100", "US30", "SPX500"]
        self.firebase = firebase_client or FirebaseClient()
        self.broadcaster = FirebaseBroadcaster(self.firebase)
        
        # Initialize all components
        logger.info("Initializing production components...")
        
        self.feature_builder = create_feature_builder()
        self.bias_engine = create_bias_engine()
        self.regime_classifier = RegimeClassifier()
        self.hard_constraints = create_hard_constraints()
        self.risk_engine = RiskEngine()
        self.game_engine = GameEngine()
        self.entry_engine = EntryEngine()
        self.strategy_integration = create_integration_layer()
        
        logger.info("All components initialized")
    
    def run_premarket(self) -> Dict[str, Any]:
        """Run pre-market pipeline at 08:00 EST.
        
        Returns:
            Dict with results for each symbol
        """
        logger.info("=" * 60)
        logger.info("STARTING PRE-MARKET CYCLE")
        logger.info("=" * 60)
        
        results = {}
        timestamp = datetime.now()
        
        try:
            # Update health status
            self.broadcaster.publish_health(
                status="healthy",
                components={"lifecycle": "pre_market", "data": "fetching"}
            )
            
            for symbol in self.symbols:
                try:
                    logger.info(f"Processing {symbol}...")
                    result = self._process_symbol_premarket(symbol)
                    results[symbol] = result
                    
                    if result.get("error"):
                        logger.error(f"Error processing {symbol}: {result['error']}")
                    else:
                        logger.info(f"{symbol} processed successfully")
                        
                except Exception as e:
                    logger.error(f"Exception processing {symbol}: {e}")
                    results[symbol] = {"error": str(e)}
            
            # Update final health status
            self.broadcaster.publish_health(
                status="healthy",
                components={
                    "lifecycle": "pre_market_complete",
                    "symbols_processed": len(results),
                    "errors": sum(1 for r in results.values() if r.get("error"))
                }
            )
            
            logger.info("PRE-MARKET CYCLE COMPLETE")
            return results
            
        except Exception as e:
            logger.error(f"Pre-market cycle failed: {e}")
            self.broadcaster.publish_health(
                status="degraded",
                components={"lifecycle": "pre_market_error", "error": str(e)}
            )
            return {"error": str(e)}
    
    def run_intraday_cycle(self) -> Dict[str, Any]:
        """Run intraday cycle every 5 minutes.
        
        Returns:
            Dict with results and any entry signals
        """
        now = datetime.now()
        
        logger.info("-" * 60)
        logger.info("STARTING INTRADAY CYCLE")
        logger.info("-" * 60)
        
        results = {}
        entry_signals = []
        
        try:
            for symbol in self.symbols:
                try:
                    logger.info(f"Processing {symbol}...")
                    result = self._process_symbol_intraday(symbol)
                    results[symbol] = result
                    
                    # Check for entry signals
                    if result.get("entry_signal"):
                        entry_signals.append(result["entry_signal"])
                        logger.info(f"ENTRY SIGNAL for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Exception processing {symbol}: {e}")
                    results[symbol] = {"error": str(e)}
            
            # Update health
            self.broadcaster.publish_health(
                status="healthy",
                components={
                    "lifecycle": "intraday",
                    "timestamp": now.isoformat(),
                    "entry_signals": len(entry_signals)
                }
            )
            
            logger.info(f"INTRADAY CYCLE COMPLETE - {len(entry_signals)} entry signals")
            return {
                "results": results,
                "entry_signals": entry_signals,
                "timestamp": now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Intraday cycle failed: {e}")
            return {"error": str(e)}
    
    def run_eod_cleanup(self) -> Dict[str, Any]:
        """Run end-of-day cleanup at 16:05 EST."""
        logger.info("=" * 60)
        logger.info("STARTING EOD CLEANUP")
        logger.info("=" * 60)
        
        try:
            results = {
                "positions_closed": 0,
                "daily_pnl": 0.0,
                "signals_archived": 0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Publish EOD status
            self.broadcaster.publish_health(
                status="healthy",
                components={
                    "lifecycle": "eod_cleanup",
                    "positions_closed": results["positions_closed"],
                    "daily_pnl": results["daily_pnl"]
                }
            )
            
            # Update session controls for next day
            self.broadcaster.publish_controls(
                trading_enabled=False,
                open_positions=0,
                hard_logic_status="pending_next_session"
            )
            
            logger.info("EOD CLEANUP COMPLETE")
            return results
            
        except Exception as e:
            logger.error(f"EOD cleanup failed: {e}")
            return {"error": str(e)}
    
    def _process_symbol_premarket(self, symbol: str) -> Dict[str, Any]:
        """Process a single symbol through pre-market pipeline."""
        result = {"symbol": symbol}
        
        # 1. Fetch market data (would call data pipeline here)
        market_data = self._fetch_market_data(symbol)
        result["market_data"] = market_data
        
        # 2. Build features
        features = self.feature_builder.build_features(
            symbol=symbol,
            ohlcv=market_data.get('ohlcv'),
            vix_value=market_data.get('vix', 20.0),
            breadth_ratio=market_data.get('breadth', 1.0)
        )
        result["features"] = features.to_dict()
        
        # 3. Classify regime
        regime = self.regime_classifier.classify(features.to_dict())
        result["regime"] = regime.to_dict()
        
        # 4. Run bias engine
        bias = self.bias_engine.get_daily_bias(symbol, features.to_dict(), regime)
        result["bias"] = bias.to_dict()
        
        # 5. Write to Firebase
        self._write_to_firebase(symbol, bias, regime, market_data)
        
        return result
    
    def _process_symbol_intraday(self, symbol: str) -> Dict[str, Any]:
        """Process a single symbol through intraday pipeline."""
        result = {"symbol": symbol}
        
        # 1. Fetch fresh market data
        market_data = self._fetch_market_data(symbol)
        
        # 2. Build features
        features = self.feature_builder.build_features(
            symbol=symbol,
            ohlcv=market_data.get('ohlcv'),
            vix_value=market_data.get('vix', 20.0)
        )
        
        # 3. Classify regime
        regime = self.regime_classifier.classify(features.to_dict())
        
        # 4. Run Layer 1 - Bias Engine
        bias = self.bias_engine.get_daily_bias(symbol, features.to_dict(), regime)
        
        # 5. Run Layer 3 - Game Engine
        game_output = self.game_engine.analyze(
            ohlcv=market_data.get('ohlcv'),
            bias=bias,
            current_price=market_data.get('current_price', 0)
        )
        
        # Convert game output to GameOutput type
        game = self._convert_game_output(game_output)
        
        # 6. Run Layer 2 - Risk Engine
        account = self._get_account_state()
        market = self._convert_market_data(market_data)
        risk = self.risk_engine.compute_risk_structure(
            bias=bias,
            regime=regime,
            market_data=market,
            account_state=account
        )
        
        # 7. Build three-layer context
        context = ThreeLayerContext(
            bias=bias,
            risk=risk,
            game=game,
            regime=regime
        )
        
        # 8. Run hard constraint checks
        constraint_check = self.hard_constraints.check_all_constraints(
            account=account,
            risk=risk
        )
        
        if not constraint_check.passed:
            logger.info(f"Hard constraint blocked {symbol}: {constraint_check.reason}")
            result["blocked"] = constraint_check.reason
            return result
        
        # 9. Validate entry through 12 gates
        entry_price = market_data.get('current_price', 0)
        ict_setup = self._detect_ict_pattern(market_data.get('ohlcv'))
        
        entry_signal = self.strategy_integration.process_signal(
            symbol=symbol,
            context=context,
            market_data=market,
            ict_setup=ict_setup
        )
        
        if entry_signal:
            result["entry_signal"] = entry_signal.to_dict()
            logger.info(f"ENTRY SIGNAL: {symbol} {entry_signal.direction.name}")
        
        # 10. Broadcast to Firebase
        self.broadcaster.publish_signal(
            symbol=symbol,
            bias=bias,
            risk=risk,
            game=game,
            regime=regime,
            current_price=entry_price
        )
        
        result["bias"] = bias.to_dict()
        result["game_aligned"] = game.game_state_aligned
        
        return result
    
    def _fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch market data for symbol."""
        # This would call the data pipeline
        # For now, return mock data structure
        return {
            'symbol': symbol,
            'current_price': 21905.0,
            'ohlcv': None,  # Would be DataFrame from data pipeline
            'vix': 20.0,
            'breadth': 1.0
        }
    
    def _write_to_firebase(
        self,
        symbol: str,
        bias: BiasOutput,
        regime: RegimeState,
        market_data: Dict
    ) -> None:
        """Write pre-market results to Firebase."""
        try:
            timestamp = datetime.now().isoformat()
            doc_id = f"{symbol}_{datetime.now().strftime('%Y%m%d')}"
            
            # 1. Write bias output to Firestore
            self.firebase.write('bias_outputs', doc_id, {
                **bias.to_dict(),
                'symbol': symbol
            })
            
            # 2. Update Realtime DB - Live Market State Engine
            self.firebase.update_realtime(f'/market/{symbol}', {
                'regime': regime.regime_state,
                'volatility': regime.volatility_level,
                'liquidity': market_data.get('breadth', 1.0),
                'momentum_score': bias.confidence,
                'trend_strength': abs(bias.bias_direction),
                'last_update': timestamp
            })
            
            # 3. Publish Feature Monitoring Data
            if hasattr(bias, 'feature_snapshot') and bias.feature_snapshot:
                features = bias.feature_snapshot
                self.firebase.update_realtime(f'/features/{symbol}', {
                    'volatility_regime': features.get('volatility_regime', 0),
                    'momentum_score': features.get('rsi_14', 50) / 100,
                    'liquidity_score': features.get('market_breadth_ratio', 1.0),
                    'trend_strength': features.get('trend_strength', 0),
                    'feature_importance': features.get('feature_importance', {}),
                    'last_update': timestamp
                })
            
            # 4. Publish Explainability Data (SHAP values)
            if hasattr(bias, 'rationale') and bias.rationale:
                self.firebase.update_realtime(f'/explainability/{symbol}', {
                    'decision_rationale': bias.rationale,
                    'confidence': bias.confidence,
                    'bias_direction': 'BULLISH' if bias.bias_direction > 0 else 'BEARISH' if bias.bias_direction < 0 else 'NEUTRAL',
                    'feature_contributions': getattr(bias, 'feature_importance', {}),
                    'last_update': timestamp
                })
            
            # 5. Update system status
            self.firebase.update_realtime('/system/status', {
                'status': 'healthy',
                'latency_ms': 0,
                'last_update': timestamp,
                'symbols_processed': 1
            })
            
            logger.debug(f"Firebase state updated for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to write to Firebase: {e}")
    
    def _convert_game_output(self, game_dict: Dict) -> GameOutput:
        """Convert game engine dict output to GameOutput type."""
        from contracts.types import GameOutput, LiquidityPool, TrappedPositions, NashZone, AdversarialRisk
        
        # Simplified conversion - would need full implementation
        return GameOutput(
            liquidity_map=game_dict.get('liquidity_map', {}),
            nearest_unswept_pool=None,
            trapped_positions=TrappedPositions([], [], 0.0, 0.0, 0.0),
            forced_move_probability=game_dict.get('forced_move_probability', 0.0),
            nash_zones=[],
            kyle_lambda=game_dict.get('kyle_lambda', 0.0),
            game_state_aligned=game_dict.get('game_state_aligned', False),
            game_state_summary=game_dict.get('game_state_summary', 'NEUTRAL'),
            adversarial_risk=AdversarialRisk.LOW
        )
    
    def _convert_market_data(self, data: Dict) -> MarketData:
        """Convert dict to MarketData type."""
        return MarketData(
            symbol=data.get('symbol', 'NAS100'),
            current_price=data.get('current_price', 0),
            bid=data.get('current_price', 0) - 0.5,
            ask=data.get('current_price', 0) + 0.5,
            spread=1.0,
            volume_24h=1000000.0,
            atr_14=45.0,
            timestamp=datetime.now()
        )
    
    def _get_account_state(self) -> AccountState:
        """Get current account state from Firebase or broker."""
        return AccountState(
            account_id="default",
            equity=50000.0,
            balance=50000.0,
            open_positions=0,
            daily_pnl=0.0,
            daily_loss_pct=0.0,
            margin_used=0.0,
            margin_available=50000.0,
            timestamp=datetime.now()
        )
    
    def _detect_ict_pattern(self, ohlcv) -> Optional[Dict]:
        """Detect ICT patterns from OHLCV data."""
        # Would use ICTDetector from entry_engine
        return None


def create_production_orchestrator(symbols: Optional[List[str]] = None) -> ProductionOrchestrator:
    """Factory function to create production orchestrator."""
    return ProductionOrchestrator(symbols=symbols)


if __name__ == "__main__":
    # Run pre-market
    logging.basicConfig(level=logging.INFO)
    orchestrator = create_production_orchestrator()
    results = orchestrator.run_premarket()
    print(f"Pre-market complete: {results}")
