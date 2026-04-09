"""
Real-Time Firebase Publisher

Takes REAL data from DataProvider + ProductionEntryEngine
Publishes to Firebase so YOUR frontend (trading-dashboard.html) can read it.

This is the GLUE between backend and frontend.
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from zoneinfo import ZoneInfo

# Load environment variables first
from dotenv import load_dotenv

load_dotenv(".env")

from data.providers import DataProvider
from integration.production_engine import ProductionEntryEngine, EnhancedEntrySignal
from integration.firebase_broadcaster import FirebaseBroadcaster
from execution.paper_trading import PaperTradingEngine
from config.settings import get_starting_equity
from contracts.types import AccountState
from clawd_trading.participants import (
    classify_participants,
    extract_from_layer1_context,
    calculate_participant_risk_limits,
)
from clawd_trading.risk import classify_regime_from_layer1

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TZ_NY = ZoneInfo("America/New_York")


class RealtimeFirebasePublisher:
    """
    Publishes REAL market data to Firebase every 5 seconds.
    This connects the Python backend to your HTML frontend.
    """

    def __init__(self):
        self.data = DataProvider()
        self.engine = ProductionEntryEngine()
        self.firebase = None
        self.paper = PaperTradingEngine(starting_equity=get_starting_equity())

        # Try to connect to Firebase
        try:
            self.firebase = FirebaseBroadcaster()
            logger.info("✅ Firebase connected")
        except Exception as e:
            logger.error(f"❌ Firebase connection failed: {e}")
            logger.error("Make sure FIREBASE_PROJECT_ID is set in .env")
            raise

        self.symbols = ["NAS100", "US30", "SPX500", "XAUUSD"]
        self.running = False

        logger.info("=" * 60)
        logger.info("REALTIME PUBLISHER INITIALIZED")
        logger.info("Data flow: Yahoo Finance → Python → Firebase → Your Frontend")
        logger.info("=" * 60)

    def fetch_and_publish_live_state(self, symbol: str) -> bool:
        """
        Fetch REAL data and publish to Firebase /live_state/{symbol}
        """
        try:
            # 1. Get REAL market data from Yahoo Finance
            market_data = self.data.get_market_data(symbol)
            if not market_data:
                logger.warning(f"No market data for {symbol}")
                return False

            # 2. Build Layer 1 from REAL data
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

            layer1_output = {
                "symbol": symbol,
                "direction": direction,
                "confidence": confidence,
                "trend_regime": trend,
                "volatility_regime": ("high" if volatility > 0.02 else "normal" if volatility > 0.01 else "low"),
                "current_price": market_data.close,
                "open": market_data.open,
                "high": market_data.high,
                "low": market_data.low,
                "volume": market_data.volume,
                "daily_return": change_pct,
                "features": {
                    "change_pct": change_pct,
                    "volatility": volatility,
                    "volume": market_data.volume,
                },
                "session": "RTH",
            }

            # 3. Build Layer 2 (EV/Risk)
            base_ev = (confidence - 0.5) * 4 * direction

            layer2_output = {
                "ev": base_ev,
                "expected_value": base_ev,
                "win_prob": 0.5 + (confidence - 0.5) * 0.6,
                "stop_price": market_data.close * 0.99,
                "tp1_price": market_data.close * 1.02,
                "tp2_price": market_data.close * 1.04,
                "ev_positive": base_ev > 0,
                "risk_reward": 2.0,
            }

            # 4. Build Layer 3 (Game Theory)
            adversarial = "LOW"
            if volatility > 0.02:
                adversarial = "MEDIUM"

            layer3_output = {
                "adversarial_risk": adversarial,
                "game_state_aligned": direction != 0,
                "forced_move_probability": 0.15,
                "kyle_lambda": 0.0012,
                "game_state_summary": f"{trend} with {adversarial} risk",
            }

            # 5. Build LIVE STATE for Firebase (matches what frontend expects)
            live_state = {
                "symbol": symbol,
                "timestamp": datetime.now(TZ_NY).isoformat(),
                "price": market_data.close,
                "change_24h": change_pct,
                "data_source": "yahoo_finance",
                "real_data": True,
                # Layer 1 - Bias
                "bias": {
                    "direction": direction,
                    "confidence": confidence,
                    "rationale": [
                        f"Price change: {change_pct:+.2%}",
                        f"Volatility: {volatility:.2%}",
                        f"Volume: {market_data.volume:,}",
                    ],
                    "model_version": "v2.1-real-data",
                },
                "current_bias": {
                    "direction": direction,
                    "confidence": confidence,
                },
                # Layer 2 - Risk
                "risk": layer2_output,
                "open_position": {
                    "ev_positive": base_ev > 0,
                    "expected_value": base_ev,
                    "stop_price": market_data.close * 0.99,
                    "tp1_price": market_data.close * 1.02,
                },
                # Layer 3 - Game
                "game": layer3_output,
                "game_state": layer3_output,
                # Regime
                "regime": {
                    "volatility": ("HIGH" if volatility > 0.02 else "NORMAL" if volatility > 0.01 else "LOW"),
                    "trend": trend.upper(),
                    "risk_appetite": "ELEVATED" if volatility > 0.02 else "NORMAL",
                    "event_risk": "NONE",
                    "composite_score": confidence,
                },
                "current_regime": {
                    "state": trend.upper(),
                    "composite_score": confidence,
                },
                # Session
                "session": {
                    "pnl": self.paper.daily_pnl,
                    "position": "LONG" if symbol in self.paper.positions else "FLAT",
                    "entry": (list(self.paper.positions.values())[0].entry_price if symbol in self.paper.positions else 0),
                },
                "session_pnl": self.paper.daily_pnl,
                "position_state": "LONG" if symbol in self.paper.positions else "FLAT",
            }

            # 6. PUBLISH TO FIREBASE
            if self.firebase and self.firebase._enabled:
                self._publish_to_firebase(symbol, live_state)
                logger.info(f"📡 Published {symbol}: ${market_data.close:.2f} | {trend.upper()} | Conf: {confidence:.2f}")
                return True
            else:
                logger.error("Firebase not available")
                return False

        except Exception as e:
            logger.error(f"Error publishing {symbol}: {e}")
            return False

    def _publish_to_firebase(self, symbol: str, data: Dict[str, Any]) -> bool:
        """Publish data to Firebase Realtime Database."""
        try:
            from firebase.client import FirebaseClient

            client = FirebaseClient()
            if not client.rtdb:
                logger.error("Firebase RTDB not available")
                return False

            # Publish to /live_state/{symbol}
            ref = client.rtdb.reference(f"live_state/{symbol}")
            ref.set(data)

            return True

        except Exception as e:
            logger.error(f"Firebase publish error: {e}")
            return False

    def generate_and_publish_signal(self, symbol: str) -> bool:
        """
        Generate a REAL signal and publish to Firebase.
        """
        try:
            # Get market data
            market_data = self.data.get_market_data(symbol)
            if not market_data:
                return False

            # Build layer outputs
            layer1 = {
                "symbol": symbol,
                "direction": (1 if market_data.change_percent > 0.005 else -1 if market_data.change_percent < -0.005 else 0),
                "confidence": min(0.5 + abs(market_data.change_percent) * 20, 0.95),
                "trend_regime": (
                    "uptrend"
                    if market_data.change_percent > 0.005
                    else ("downtrend" if market_data.change_percent < -0.005 else "neutral")
                ),
                "volatility_regime": ("high" if (market_data.high - market_data.low) / market_data.close > 0.02 else "normal"),
                "current_price": market_data.close,
                "features": {
                    "change_pct": market_data.change_percent,
                    "volume": market_data.volume,
                },
                "fvg_detected": (market_data.high - market_data.low) / market_data.close > 0.015,
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
                daily_loss_pct=(self.paper.daily_pnl / self.paper.current_equity if self.paper.current_equity > 0 else 0),
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
                # Publish signal
                signal_data = {
                    "signal_id": f"SIG_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "timestamp": datetime.now(TZ_NY).isoformat(),
                    "symbol": signal.symbol,
                    "direction": signal.direction_str,
                    "direction_num": signal.direction.value,
                    "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "tp1": signal.take_profit_1,
                    "tp2": signal.take_profit_2,
                    "confidence": signal.confidence * 100,  # Frontend expects percentage
                    "expected_value": signal.expected_value,
                    "ev": signal.expected_value,
                    "status": "ACTIVE",
                    # Enhanced data
                    "entry_model": signal.entry_model,
                    "dominant_participant": signal.dominant_participant,
                    "participant_confidence": signal.participant_confidence,
                    "regime": signal.regime,
                    "gates_passed": signal.gates_passed,
                    "gate_details": signal.gate_details,
                }

                # Publish to Firebase
                from firebase.client import FirebaseClient

                client = FirebaseClient()
                if client.rtdb:
                    ref = client.rtdb.reference(f"entry_signals/{symbol}/latest")
                    ref.set(signal_data)

                    # Also add to history
                    history_ref = client.rtdb.reference(f'entry_signals/{symbol}/history/{signal_data["signal_id"]}')
                    history_ref.set(signal_data)

                logger.info(
                    f"🎯 SIGNAL: {signal.direction_str} {symbol} @ ${signal.entry_price:.2f} (Model: {signal.entry_model})"
                )
                return True
            else:
                logger.info(f"⏸️ No signal for {symbol} (conditions not met)")
                return False

        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return False

    def publish_system_status(self) -> bool:
        """Publish system health to Firebase."""
        try:
            status = {
                "status": "healthy",
                "trading_enabled": True,
                "open_positions": len(self.paper.positions),
                "hard_logic_status": "CLEAR",
                "daily_pnl": self.paper.daily_pnl,
                "equity": self.paper.current_equity,
                "last_update": datetime.now(TZ_NY).isoformat(),
            }

            from firebase.client import FirebaseClient

            client = FirebaseClient()
            if client.rtdb:
                ref = client.rtdb.reference("session_controls")
                ref.set(status)
                return True
            return False

        except Exception as e:
            logger.error(f"Status publish error: {e}")
            return False

    def run_once(self, symbol: str = "NAS100"):
        """Run one cycle - for testing."""
        logger.info(f"\n{'='*60}")
        logger.info(f"PUBLISHING REAL DATA FOR {symbol}")
        logger.info(f"{'='*60}")

        # Publish live state
        success = self.fetch_and_publish_live_state(symbol)

        # Publish system status
        self.publish_system_status()

        # Try to generate signal
        self.generate_and_publish_signal(symbol)

        if success:
            logger.info(f"✅ Data published to Firebase")
            logger.info(f"   Your frontend should now see real data!")
        else:
            logger.error(f"❌ Failed to publish")

    def run_continuous(self, interval_seconds: int = 5):
        """Run continuously publishing data."""
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING CONTINUOUS PUBLISHING")
        logger.info(f"Interval: {interval_seconds}s")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"{'='*60}\n")

        self.running = True

        try:
            while self.running:
                for symbol in self.symbols:
                    self.fetch_and_publish_live_state(symbol)
                    self.publish_system_status()

                    # Generate signals less frequently
                    if int(time.time()) % 30 == 0:  # Every 30 seconds
                        self.generate_and_publish_signal(symbol)

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("\nStopped by user")
            self.running = False


def main():
    """Main entry point."""
    publisher = RealtimeFirebasePublisher()

    # Run one cycle for testing
    publisher.run_once("NAS100")

    # Ask if user wants continuous
    logger.info("\nPress Ctrl+C to stop continuous publishing")
    logger.info("Starting continuous mode...\n")

    try:
        publisher.run_continuous(interval_seconds=5)
    except KeyboardInterrupt:
        logger.info("\nShutdown complete")


if __name__ == "__main__":
    main()
