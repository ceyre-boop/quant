"""Main Trading Engine - Entry point for the Clawd Trading System.

This module serves as the main entry point for the trading engine,
supporting both local development and cloud deployment (Railway/Heroku).
"""

import os
import sys
import json
import time
import logging
import signal
import threading
from typing import Dict, List, Optional
from datetime import datetime, timezone
from pathlib import Path

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Handle imports gracefully
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Environment loaded from .env file")
except ImportError:
    logger.info("dotenv not installed, using system environment")

from integration.firebase_client import FirebaseClient
from integration.firebase_broadcaster import FirebaseBroadcaster
from data.pipeline import DataPipeline
from contracts.types import (
    BiasOutput, RiskOutput, GameOutput, RegimeState,
    Direction, Magnitude, VolRegime, TrendRegime,
    RiskAppetite, MomentumRegime, EventRisk, AdversarialRisk,
    LiquidityPool, TrappedPositions, NashZone
)


class TradingEngine:
    """Main trading engine that orchestrates data, signals, and execution."""
    
    def __init__(self):
        self.running = False
        self.symbols = os.getenv('SYMBOLS', 'NAS100,SPY').split(',')
        self.trading_mode = os.getenv('TRADING_MODE', 'paper')
        
        # Initialize components
        logger.info("Initializing Firebase...")
        try:
            self.firebase = FirebaseClient(demo_mode=False)  # Force production mode
            self.broadcaster = FirebaseBroadcaster(self.firebase)
            logger.info("Firebase initialized successfully")
        except Exception as e:
            logger.error(f"Firebase initialization failed: {e}")
            raise
        
        logger.info("Initializing Data Pipeline...")
        try:
            self.data_pipeline = DataPipeline()
            logger.info("Data Pipeline initialized")
        except Exception as e:
            logger.error(f"Data Pipeline initialization failed: {e}")
            self.data_pipeline = None
        
        # Trading state
        self.current_positions: Dict[str, dict] = {}
        self.daily_stats = {
            'trades_today': 0,
            'daily_pnl': 0.0,
            'start_of_day_equity': 10000.0
        }
        
        logger.info(f"Trading Engine initialized for symbols: {self.symbols}")
        logger.info(f"Trading mode: {self.trading_mode}")
    
    def generate_demo_bias(self, symbol: str) -> BiasOutput:
        """Generate demo bias output for testing."""
        import random
        
        direction = random.choice([Direction.LONG, Direction.SHORT])
        confidence = random.uniform(0.55, 0.85)
        
        return BiasOutput(
            direction=direction,
            magnitude=Magnitude.NORMAL,
            confidence=confidence,
            regime_override=False,
            rationale=['MOMENTUM_SHIFT', 'LIQUIDITY_DRAW', 'TREND_STRUCTURE'],
            model_version='v1.0-demo',
            feature_snapshot={
                'raw_features': {'rsi_14': random.uniform(30, 70), 'atr_14': random.uniform(10, 50)},
                'feature_group_tags': {},
                'regime_at_inference': self.generate_demo_regime(),
                'inference_timestamp': datetime.now(timezone.utc).isoformat()
            }
        )
    
    def generate_demo_risk(self, symbol: str) -> RiskOutput:
        """Generate demo risk output for testing."""
        import random
        
        current_price = 18300.0 if symbol == 'NAS100' else 450.0
        
        return RiskOutput(
            position_size=1.2,
            kelly_fraction=0.35,
            stop_price=current_price - 70,
            stop_method='atr',
            tp1_price=current_price + 85,
            tp2_price=current_price + 150,
            trail_config={'trail_pct': 0.02},
            expected_value=1.84,
            ev_positive=True,
            size_breakdown={'base': 1.0, 'kelly': 0.35, 'regime': 1.0}
        )
    
    def generate_demo_game(self, symbol: str) -> GameOutput:
        """Generate demo game output for testing."""
        import random
        
        return GameOutput(
            liquidity_map={
                'equal_highs': [
                    LiquidityPool(18600, 4, False, 5, 0.91, 'equal_highs'),
                    LiquidityPool(18540, 3, False, 3, 0.67, 'equal_highs'),
                ],
                'equal_lows': [
                    LiquidityPool(18200, 4, False, 8, 0.91, 'equal_lows'),
                    LiquidityPool(18260, 3, False, 4, 0.58, 'equal_lows'),
                ]
            },
            nearest_unswept_pool=LiquidityPool(18600, 4, False, 5, 0.91, 'equal_highs'),
            trapped_positions=TrappedPositions([], [], 0, 0, 0.2),
            forced_move_probability=0.25,
            nash_zones=[
                NashZone(18300, 'hvn', 'HOLDING', 3, 0.75)
            ],
            kyle_lambda=0.0043,
            game_state_aligned=True,
            game_state_summary='NORMAL_CONDUCTION',
            adversarial_risk=AdversarialRisk.LOW
        )
    
    def generate_demo_regime(self) -> RegimeState:
        """Generate demo regime state."""
        return RegimeState(
            volatility=VolRegime.NORMAL,
            trend=TrendRegime.STRONG_TREND,
            risk_appetite=RiskAppetite.RISK_ON,
            momentum=MomentumRegime.ACCELERATING,
            event_risk=EventRisk.CLEAR,
            composite_score=0.72
        )
    
    def run_trading_cycle(self):
        """Run one complete trading cycle."""
        try:
            logger.info("Running trading cycle...")
            
            for symbol in self.symbols:
                logger.info(f"Processing {symbol}...")
                
                # Generate signals (demo mode for now)
                bias = self.generate_demo_bias(symbol)
                risk = self.generate_demo_risk(symbol)
                game = self.generate_demo_game(symbol)
                regime = self.generate_demo_regime()
                
                # Broadcast to Firebase
                self.broadcaster.broadcast_full_state(
                    symbol=symbol,
                    bias=bias,
                    risk=risk,
                    game=game,
                    regime=2,  # NORMAL
                    vix_level=18.5,
                    event_risk='CLEAR'
                )
                
                # Broadcast liquidity map
                self.broadcaster.client.rtdb_update(
                    f'liquidity_map/{symbol}',
                    {
                        'price_18600': 0.91,
                        'price_18540': 0.67,
                        'price_18480': 0.44,
                        'price_18395': 0.29,
                        'price_18310': 0.0,
                        'price_18260': 0.35,
                        'price_18190': 0.58,
                        'price_18100': 0.82,
                        'price_18020': 0.91,
                    }
                )
                
                logger.info(f"Broadcast complete for {symbol}")
            
            # Update session controls
            self.broadcaster.broadcast_session_control(
                trading_enabled=True,
                hard_logic_status='CLEAR'
            )
            
            # Update performance snapshot
            self.broadcaster.broadcast_performance_snapshot(
                daily_pnl=self.daily_stats['daily_pnl'],
                open_positions=len(self.current_positions),
                win_rate=0.587,
                sharpe=1.42
            )
            
            logger.info("Trading cycle complete")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
    
    def run(self):
        """Main run loop."""
        logger.info("=" * 60)
        logger.info("CLAWD TRADING ENGINE STARTING")
        logger.info("=" * 60)
        
        self.running = True
        
        # Initialize UI state
        try:
            self.broadcaster.initialize_ui_state(self.symbols)
        except Exception as e:
            logger.error(f"Failed to initialize UI state: {e}")
        
        # Signal handler for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Shutdown signal received...")
            self.running = False
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Main loop
        cycle_count = 0
        while self.running:
            try:
                self.run_trading_cycle()
                cycle_count += 1
                logger.info(f"Completed cycle {cycle_count}")
                
                # Sleep for 60 seconds between cycles
                for _ in range(60):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)  # Brief pause before retry
        
        logger.info("Trading engine stopped")


# Flask app for health checks (Railway/Heroku requirement)
flask_app = None
engine = None

def create_flask_app():
    """Create Flask app for health checks."""
    global flask_app, engine
    
    try:
        from flask import Flask, jsonify
        
        app = Flask(__name__)
        
        @app.route('/')
        def health_check():
            return jsonify({
                'status': 'healthy',
                'service': 'clawd-trading-engine',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'mode': os.getenv('TRADING_MODE', 'paper'),
                'firebase_connected': engine.broadcaster.client.is_connected() if engine else False
            })
        
        @app.route('/status')
        def status():
            return jsonify({
                'engine_running': engine.running if engine else False,
                'symbols': os.getenv('SYMBOLS', 'NAS100,SPY').split(','),
                'firebase_connected': engine.broadcaster.client.is_connected() if engine else False,
                'trading_mode': os.getenv('TRADING_MODE', 'paper')
            })
        
        flask_app = app
        return app
        
    except ImportError:
        logger.warning("Flask not installed, web server disabled")
        return None


def main():
    """Main entry point."""
    global engine
    
    # Check for required environment variables
    logger.info("Checking environment...")
    
    # Initialize trading engine
    engine = TradingEngine()
    
    # Check if we should run web server (for Railway/Heroku)
    port = os.getenv('PORT')
    if port:
        logger.info(f"Web server mode - port {port}")
        app = create_flask_app()
        if app:
            # Run trading engine in background thread
            trading_thread = threading.Thread(target=engine.run)
            trading_thread.daemon = True
            trading_thread.start()
            
            # Run Flask app
            try:
                from gunicorn.app.base import BaseApplication
                
                class GunicornApp(BaseApplication):
                    def __init__(self, app, options=None):
                        self.options = options or {}
                        self.application = app
                        super().__init__()
                    
                    def load_config(self):
                        for key, value in self.options.items():
                            self.cfg.set(key, value)
                    
                    def load(self):
                        return self.application
                
                options = {
                    'bind': f'0.0.0.0:{port}',
                    'workers': 1,
                    'threads': 4,
                    'timeout': 120
                }
                GunicornApp(app, options).run()
                
            except ImportError:
                # Fallback to Flask dev server
                logger.warning("Gunicorn not installed, using Flask dev server")
                app.run(host='0.0.0.0', port=int(port), threaded=True)
        else:
            # No Flask, run engine directly
            engine.run()
    else:
        # Run trading engine directly
        engine.run()


if __name__ == '__main__':
    main()
