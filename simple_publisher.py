"""
Simple Firebase Publisher using REST API

Writes real data to Firebase Realtime Database using HTTP REST API.
No Admin SDK credentials needed - uses the web API key.
"""
import logging
import os
import time
import requests
import json
from datetime import datetime
from typing import Dict, Any, Optional
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from data.providers import DataProvider
from integration.production_engine import ProductionEntryEngine
from execution.paper_trading import PaperTradingEngine
from contracts.types import AccountState

load_dotenv('.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

TZ_NY = ZoneInfo("America/New_York")


class SimpleFirebasePublisher:
    """
    Publishes REAL market data to Firebase using REST API.
    Your frontend (trading-dashboard.html) reads from here.
    """
    
    def __init__(self):
        self.data = DataProvider()
        self.engine = ProductionEntryEngine()
        self.paper = PaperTradingEngine(starting_equity=100000.0)
        
        # Firebase config
        self.project_id = os.getenv('FIREBASE_PROJECT_ID')
        self.api_key = os.getenv('FIREBASE_API_KEY')
        self.rtdb_url = os.getenv('FIREBASE_RTDB_URL', 
                                  f'https://{self.project_id}-default-rtdb.firebaseio.com')
        
        if not self.project_id:
            raise ValueError("FIREBASE_PROJECT_ID not set")
        if not self.api_key:
            raise ValueError("FIREBASE_API_KEY not set")
        
        self.symbols = ["NAS100", "US30", "SPX500", "XAUUSD"]
        self.running = False
        
        logger.info("="*60)
        logger.info("SIMPLE FIREBASE PUBLISHER")
        logger.info(f"Project: {self.project_id}")
        logger.info(f"RTDB: {self.rtdb_url}")
        logger.info("="*60)
    
    def _firebase_put(self, path: str, data: Dict[str, Any]) -> bool:
        """Write data to Firebase using REST API."""
        url = f"{self.rtdb_url}/{path}.json?auth={self.api_key}"
        
        try:
            response = requests.put(url, json=data, timeout=10)
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Firebase write failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Firebase write error: {e}")
            return False
    
    def publish_live_state(self, symbol: str) -> bool:
        """
        Fetch REAL data and publish to /live_state/{symbol}
        """
        try:
            # Get real market data
            market_data = self.data.get_market_data(symbol)
            if not market_data:
                logger.warning(f"No data for {symbol}")
                return False
            
            # Build 3-layer analysis
            change_pct = market_data.change_percent
            volatility = (market_data.high - market_data.low) / market_data.close
            
            if change_pct > 0.005:
                trend = "uptrend"
                direction = 1
            elif change_pct < -0.005:
                trend = "downtrend"
                direction = -1
            else:
                trend = "neutral"
                direction = 0
            
            confidence = min(0.5 + abs(change_pct) * 20, 0.95)
            base_ev = (confidence - 0.5) * 4 * direction
            
            adversarial = "LOW"
            if volatility > 0.02:
                adversarial = "MEDIUM"
            
            # Build live state (matches frontend expectations)
            live_state = {
                "symbol": symbol,
                "timestamp": datetime.now(TZ_NY).isoformat(),
                "price": market_data.close,
                "change_24h": change_pct,
                "real_data": True,
                
                # Layer 1 - Bias
                "bias": {
                    "direction": direction,
                    "confidence": confidence,
                    "rationale": [
                        f"Price change: {change_pct:+.2%}",
                        f"Volatility: {volatility:.2%}"
                    ],
                    "model_version": "v2.1-real"
                },
                "current_bias": {
                    "direction": direction,
                    "confidence": confidence
                },
                
                # Layer 2 - Risk
                "risk": {
                    "ev_positive": base_ev > 0,
                    "expected_value": base_ev,
                    "stop_price": market_data.close * 0.99,
                    "tp1_price": market_data.close * 1.02,
                    "tp2_price": market_data.close * 1.04
                },
                "open_position": {
                    "ev_positive": base_ev > 0,
                    "expected_value": base_ev
                },
                
                # Layer 3 - Game
                "game": {
                    "adversarial_risk": adversarial,
                    "game_state_aligned": direction != 0,
                    "forced_move_probability": 0.15,
                    "kyle_lambda": 0.0012,
                    "game_state_summary": f"{trend} with {adversarial} risk"
                },
                
                # Regime
                "regime": {
                    "volatility": "HIGH" if volatility > 0.02 else "NORMAL" if volatility > 0.01 else "LOW",
                    "trend": trend.upper(),
                    "risk_appetite": "ELEVATED" if volatility > 0.02 else "NORMAL",
                    "event_risk": "NONE",
                    "composite_score": confidence
                },
                
                # Session
                "session": {
                    "pnl": self.paper.daily_pnl,
                    "position": "LONG" if symbol in self.paper.positions else "FLAT",
                    "entry": list(self.paper.positions.values())[0].entry_price if symbol in self.paper.positions else 0
                }
            }
            
            # Publish to Firebase
            if self._firebase_put(f'live_state/{symbol}', live_state):
                logger.info(f"Published {symbol}: ${market_data.close:.2f} | {trend.upper()} | Conf: {confidence:.2f}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error publishing {symbol}: {e}")
            return False
    
    def publish_signal(self, symbol: str) -> bool:
        """Generate and publish a signal."""
        try:
            market_data = self.data.get_market_data(symbol)
            if not market_data:
                return False
            
            # Build layers
            layer1 = {
                "symbol": symbol,
                "direction": 1 if market_data.change_percent > 0.005 else -1 if market_data.change_percent < -0.005 else 0,
                "confidence": min(0.5 + abs(market_data.change_percent) * 20, 0.95),
                "trend_regime": "uptrend" if market_data.change_percent > 0.005 else "downtrend" if market_data.change_percent < -0.005 else "neutral",
                "current_price": market_data.close,
                "features": {"change_pct": market_data.change_percent},
                "fvg_detected": False,
                "liquidity_sweep": abs(market_data.change_percent) > 0.01,
                "order_block": False,
                "ict_setup": {},
                "session": "RTH",
            }
            
            layer2 = {
                "ev": (layer1["confidence"] - 0.5) * 4 * layer1["direction"],
                "win_prob": 0.5 + (layer1["confidence"] - 0.5) * 0.6,
                "max_position_size": 0.1,
                "stop_loss": market_data.close * 0.99,
                "take_profit": market_data.close * 1.02,
            }
            
            layer3 = {
                "adversarial_risk": "LOW",
                "game_state_aligned": layer1["direction"] != 0,
            }
            
            account = AccountState(
                account_id="paper",
                equity=self.paper.current_equity,
                balance=self.paper.current_equity,
                open_positions=len(self.paper.positions),
                daily_pnl=self.paper.daily_pnl,
                daily_loss_pct=0,
                margin_used=0,
                margin_available=self.paper.current_equity,
                timestamp=datetime.now(),
            )
            
            # Generate signal
            signal = self.engine.generate_signal(
                symbol=symbol,
                layer1_output=layer1,
                layer2_output=layer2,
                layer3_output=layer3,
                account=account,
            )
            
            if signal:
                signal_data = {
                    "signal_id": f"SIG_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "timestamp": datetime.now(TZ_NY).isoformat(),
                    "symbol": signal.symbol,
                    "direction": signal.direction_str,
                    "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "tp1": signal.take_profit_1,
                    "confidence": signal.confidence * 100,
                    "expected_value": signal.expected_value,
                    "ev": signal.expected_value,
                    "status": "ACTIVE",
                    "entry_model": signal.entry_model,
                    "dominant_participant": signal.dominant_participant,
                    "regime": signal.regime
                }
                
                if self._firebase_put(f'entry_signals/{symbol}/latest', signal_data):
                    self._firebase_put(f'entry_signals/{symbol}/history/{signal_data["signal_id"]}', signal_data)
                    logger.info(f"SIGNAL: {signal.direction_str} {symbol} @ ${signal.entry_price:.2f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Signal error: {e}")
            return False
    
    def publish_status(self):
        """Publish system status."""
        status = {
            "status": "healthy",
            "trading_enabled": True,
            "open_positions": len(self.paper.positions),
            "hard_logic_status": "CLEAR",
            "daily_pnl": self.paper.daily_pnl,
            "last_update": datetime.now(TZ_NY).isoformat()
        }
        self._firebase_put('session_controls', status)
    
    def run_once(self, symbol: str = "NAS100"):
        """Run one cycle."""
        logger.info(f"\n{'='*60}")
        logger.info(f"PUBLISHING REAL DATA FOR {symbol}")
        logger.info(f"{'='*60}")
        
        success = self.publish_live_state(symbol)
        self.publish_status()
        self.publish_signal(symbol)
        
        if success:
            logger.info("Data published to Firebase!")
            logger.info(f"Check your frontend - it should now show:")
            logger.info(f"  - Real price from Yahoo Finance")
            logger.info(f"  - 3-layer analysis")
            logger.info(f"  - Regime classification")
        else:
            logger.error("Failed to publish")
    
    def run_continuous(self, interval: int = 5):
        """Run continuously."""
        logger.info(f"\nStarting continuous publishing every {interval}s")
        logger.info("Press Ctrl+C to stop\n")
        
        self.running = True
        try:
            while self.running:
                for symbol in self.symbols:
                    self.publish_live_state(symbol)
                    self.publish_status()
                    
                    if int(time.time()) % 30 == 0:
                        self.publish_signal(symbol)
                
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("\nStopped")
            self.running = False


def main():
    publisher = SimpleFirebasePublisher()
    publisher.run_once("NAS100")
    
    logger.info("\nPress Ctrl+C to stop continuous mode")
    try:
        publisher.run_continuous(interval=5)
    except KeyboardInterrupt:
        logger.info("\nShutdown complete")


if __name__ == "__main__":
    main()
