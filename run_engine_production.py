"""
Clawd Trading Engine - Production Runner

Runs the full 3-layer trading system with:
- Layer 1: Hard Constraints + Regime Classification + Data Pipeline
- Layer 2: Bias Engine + EV Calculation
- Layer 3: Game Theory + Adversarial Analysis
- Entry Engine: 12-gate validation + Entry Scoring
- Participant Analysis: Microstructure detection
- Regime Risk: Dynamic risk limits
- Combined Risk: Participant × Regime
- Firebase Broadcast: Real-time signal distribution
"""
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from config.settings import get_starting_equity

# Import production components
from integration.production_engine import ProductionEntryEngine, EnhancedEntrySignal
from integration.firebase_broadcaster import FirebaseBroadcaster
from contracts.types import AccountState
from layer1.hard_constraints_v2 import HardConstraints

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
TZ_NY = ZoneInfo("America/New_York")
SLEEP_SECONDS = 20
ENGINE_VERSION = os.getenv("ENGINE_VERSION", "2.0.0-production")
LOG_PATH = "logs/engine.log"

PREMARKET_TIME = dt_time(hour=8, minute=0)
MARKET_OPEN_TIME = dt_time(hour=9, minute=30)
MARKET_CLOSE_TIME = dt_time(hour=16, minute=0)
EOD_SHUTDOWN_TIME = dt_time(hour=16, minute=5)
WEEKLY_REFIT_TIME = dt_time(hour=16, minute=10)
SESSION_END_TIME = dt_time(hour=18, minute=0)
SIGNAL_INTERVAL = timedelta(minutes=5)

DEFAULT_SYMBOLS = ["NAS100", "US30", "SPX500", "XAUUSD"]


@dataclass
class RuntimeState:
    run_date: date
    premarket_done: bool = False
    market_open_logged: bool = False
    eod_done: bool = False
    next_signal_due: Optional[datetime] = None
    weekly_refit_week_key: Optional[str] = None


def configure_logging() -> None:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    formatter = logging.Formatter(LOG_FORMAT)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_environment() -> None:
    load_dotenv(".env")
    load_dotenv()
    
    required_vars = ["POLYGON_API_KEY", "FIREBASE_PROJECT_ID"]
    for var in required_vars:
        if not os.getenv(var):
            raise RuntimeError(f"Missing environment variable: {var}")


def get_symbols() -> list[str]:
    raw = os.getenv("TRADE_SYMBOLS", "").strip()
    if not raw:
        return DEFAULT_SYMBOLS
    parsed = [part.strip().upper() for part in raw.split(",") if part.strip()]
    return parsed or DEFAULT_SYMBOLS


def is_trading_day(now_et: datetime) -> bool:
    return now_et.weekday() < 5


def time_in_range(now_et: datetime, start: dt_time, end: dt_time) -> bool:
    now_t = now_et.time()
    return start <= now_t < end


def combine_today(now_et: datetime, t: dt_time) -> datetime:
    return now_et.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)


class ProductionTradingEngine:
    """Production trading engine with full component integration."""
    
    def __init__(self):
        self.constraints = HardConstraints()
        self.entry_engine = ProductionEntryEngine(hard_constraints=self.constraints)
        self.firebase = FirebaseBroadcaster()
        self.logger = logging.getLogger(__name__)
        self.symbols = get_symbols()
    
    def fetch_layer1_analysis(self, symbol: str) -> Dict[str, Any]:
        """Fetch Layer 1 analysis (bias, regime, features)."""
        try:
            from data.pipeline import DataPipeline
            
            pipeline = DataPipeline()
            record = pipeline.get_latest(symbol)
            
            if not record or not getattr(record, "is_valid", False):
                self.logger.warning(f"No valid Layer 1 data for {symbol}")
                return self._generate_fallback_layer1(symbol)
            
            return {
                "symbol": symbol,
                "direction": 1 if record.bias > 0 else -1 if record.bias < 0 else 0,
                "confidence": abs(record.bias),
                "trend_regime": record.trend_regime,
                "volatility_regime": record.vol_regime,
                "current_price": record.close,
                "features": record.features,
                "fvg_detected": getattr(record, "fvg_detected", False),
                "fvg_level": getattr(record, "fvg_level", record.close),
                "liquidity_sweep": getattr(record, "liquidity_sweep", False),
                "sweep_level": getattr(record, "sweep_level", record.close),
                "order_block": getattr(record, "order_block", False),
                "order_block_level": getattr(record, "order_block_level", record.close),
                "ict_setup": getattr(record, "ict_setup", {}),
                "session": getattr(record, "session", "RTH"),
                "news_minutes_to_event": getattr(record, "news_minutes_to_event", None),
                "news_impact_score": getattr(record, "news_impact_score", 0),
            }
            
        except Exception as e:
            self.logger.error(f"Layer 1 fetch failed for {symbol}: {e}")
            return self._generate_fallback_layer1(symbol)
    
    def fetch_layer2_analysis(self, symbol: str, layer1: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch Layer 2 analysis (EV, risk, game)."""
        try:
            # Calculate EV based on Layer 1 features
            confidence = layer1.get("confidence", 0.5)
            direction = layer1.get("direction", 0)
            
            # Base EV calculation
            base_ev = (confidence - 0.5) * 2 * direction  # Range: -1 to 1
            
            # Adjust for trend alignment
            trend = layer1.get("trend_regime", "neutral")
            if trend in ["uptrend", "downtrend"]:
                base_ev *= 1.2
            
            return {
                "ev": base_ev,
                "expected_value": base_ev,
                "win_prob": 0.5 + (confidence - 0.5) * 0.8,
                "risk_reward": 2.0,
                "max_position_size": 1.0,
                "stop_loss": layer1.get("current_price", 0) * 0.99,
                "take_profit": layer1.get("current_price", 0) * 1.02,
                "bias_aligned": trend in ["uptrend", "downtrend"],
            }
            
        except Exception as e:
            self.logger.error(f"Layer 2 fetch failed for {symbol}: {e}")
            return {
                "ev": 0,
                "win_prob": 0.5,
                "risk_reward": 1.5,
                "max_position_size": 0.5,
            }
    
    def fetch_layer3_analysis(self, symbol: str, layer1: Dict[str, Any], layer2: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch Layer 3 analysis (game theory)."""
        try:
            # Check adversarial conditions
            vol_regime = layer1.get("volatility_regime", "normal")
            session = layer1.get("session", "RTH")
            
            adversarial_risk = "LOW"
            if vol_regime == "high":
                adversarial_risk = "MEDIUM"
            if session in ["OPEN", "CLOSE"] and vol_regime == "high":
                adversarial_risk = "HIGH"
            
            return {
                "adversarial_risk": adversarial_risk,
                "game_state_aligned": layer2.get("ev", 0) > 0.2,
                "pool_position": {},
            }
            
        except Exception as e:
            self.logger.error(f"Layer 3 fetch failed for {symbol}: {e}")
            return {
                "adversarial_risk": "LOW",
                "game_state_aligned": True,
            }
    
    def get_account_state(self) -> AccountState:
        """Get current account state."""
        try:
            from firebase.client import FirebaseClient
            
            client = FirebaseClient()
            account_data = client.read_realtime("/account")
            
            if account_data:
                return AccountState(
                    equity=account_data.get("equity", get_starting_equity()),
                    daily_loss_pct=account_data.get("daily_loss_pct", 0),
                    open_positions=account_data.get("open_positions", 0),
                )
        except Exception as e:
            self.logger.warning(f"Could not fetch account state: {e}")
        
        # Default account state
        return AccountState(
            equity=get_starting_equity(),
            daily_loss_pct=0,
            open_positions=0,
        )
    
    def run_signal_cycle(self, symbol: str) -> Optional[EnhancedEntrySignal]:
        """Run one full signal generation cycle."""
        self.logger.info(f"Running signal cycle for {symbol}")
        
        # Fetch all three layers
        layer1 = self.fetch_layer1_analysis(symbol)
        layer2 = self.fetch_layer2_analysis(symbol, layer1)
        layer3 = self.fetch_layer3_analysis(symbol, layer1, layer2)
        account = self.get_account_state()
        
        # Generate signal with production engine
        signal = self.entry_engine.generate_signal(
            symbol=symbol,
            layer1_output=layer1,
            layer2_output=layer2,
            layer3_output=layer3,
            account=account,
        )
        
        if signal:
            # Broadcast to Firebase
            self._broadcast_signal(signal)
            return signal
        
        return None
    
    def _broadcast_signal(self, signal: EnhancedEntrySignal) -> bool:
        """Broadcast signal to Firebase."""
        try:
            # Publish to Realtime DB for dashboard
            self.firebase.publish_signal_realtime(
                symbol=signal.symbol,
                signal_data=signal.to_firebase_dict()
            )
            
            # Publish to Firestore for history
            self.firebase.publish_signal(
                symbol=signal.symbol,
                bias={"direction": signal.direction, "confidence": signal.confidence},
                risk={
                    "expected_value": signal.expected_value,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit_1,
                },
                game={
                    "entry_model": signal.entry_model,
                    "dominant_participant": signal.dominant_participant,
                    "regime": signal.regime,
                },
                regime={"state": signal.regime},
            )
            
            self.logger.info(f"Signal broadcast for {signal.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to broadcast signal: {e}")
            return False
    
    def _generate_fallback_layer1(self, symbol: str) -> Dict[str, Any]:
        """Generate fallback Layer 1 data when pipeline fails."""
        self.logger.warning(f"Using fallback Layer 1 data for {symbol}")
        return {
            "symbol": symbol,
            "direction": 0,
            "confidence": 0.5,
            "trend_regime": "neutral",
            "volatility_regime": "normal",
            "current_price": 0,
            "features": {},
        }


def run_premarket_build(symbols: list[str]) -> None:
    """Run 08:00 premarket feature build."""
    logging.info("08:00 premarket: building features for %s", symbols)
    
    try:
        from data.pipeline import DataPipeline
        
        pipeline = DataPipeline()
        records = pipeline.run_premarket(symbols=symbols, include_all_features=True)
        valid_count = sum(1 for record in records.values() if getattr(record, "is_valid", False))
        logging.info("Premarket feature build complete: %s/%s valid records", valid_count, len(records))
    except Exception as exc:
        logging.exception("Premarket feature build failed: %s", exc)


def run_eod_shutdown(firebase: FirebaseBroadcaster) -> None:
    """Run 16:05 shutdown tasks."""
    logging.info("16:05 shutdown: finalizing session")
    
    try:
        firebase.publish_health(
            status="healthy",
            components={"lifecycle": "eod_shutdown", "engine_version": ENGINE_VERSION}
        )
    except Exception as exc:
        logging.warning("EOD health publish skipped: %s", exc)
    
    logging.info("Session shutdown complete")


def run_weekly_refit_check(now_et: datetime) -> None:
    """Run weekly model refit evaluation."""
    logging.info("Weekly refit check triggered")
    
    try:
        from meta_evaluator.refit_scheduler import RefitScheduler
        
        scheduler = RefitScheduler()
        
        metrics = {
            "win_rate": float(os.getenv("WEEKLY_WIN_RATE", "0.55")),
            "sharpe": float(os.getenv("WEEKLY_SHARPE", "0.8")),
            "total_trades": int(os.getenv("WEEKLY_TOTAL_TRADES", "0")),
        }
        drift_detected = os.getenv("WEEKLY_DRIFT_DETECTED", "false").lower() == "true"
        
        evaluation = scheduler.evaluate_refit_need(metrics, drift_detected=drift_detected)
        
        if evaluation.get("should_refit"):
            schedule = scheduler.schedule_refit(
                evaluation,
                model_version=f"weekly-{now_et.date().isoformat()}"
            )
            logging.info("Weekly refit scheduled: %s", schedule)
        else:
            logging.info("No weekly refit scheduled: %s", evaluation.get("reasons", []))
    except Exception as exc:
        logging.exception("Weekly refit check failed: %s", exc)


def push_heartbeat(status: str, engine: Optional[ProductionTradingEngine] = None) -> None:
    """Push engine heartbeat to Firebase."""
    payload = {
        "status": status,
        "last_cycle": datetime.now(TZ_NY).isoformat(),
        "engine_version": ENGINE_VERSION,
        "components": {
            "production_engine": "active",
            "participant_analysis": "active",
            "regime_risk": "active",
            "entry_scoring": "active",
        }
    }
    
    try:
        from firebase.client import FirebaseClient
        
        client = FirebaseClient()
        if client.rtdb is None:
            logging.debug("Heartbeat skipped: RTDB unavailable")
            return
        
        client.rtdb.reference("system/health").set(payload)
        logging.info("Heartbeat updated")
    except Exception as exc:
        logging.warning("Heartbeat push failed: %s", exc)


def reset_daily_state(state: RuntimeState, now_et: datetime) -> RuntimeState:
    """Reset daily markers at start of new day."""
    if state.run_date == now_et.date():
        return state
    return RuntimeState(run_date=now_et.date(), weekly_refit_week_key=state.weekly_refit_week_key)


def main() -> None:
    configure_logging()
    logging.info("=" * 60)
    logging.info("Starting CLAWD Trading Engine v%s", ENGINE_VERSION)
    logging.info("=" * 60)
    
    load_environment()
    symbols = get_symbols()
    primary_symbol = symbols[0] if symbols else "NAS100"
    
    # Initialize production engine
    engine = ProductionTradingEngine()
    firebase = FirebaseBroadcaster()
    
    now_et = datetime.now(TZ_NY)
    state = RuntimeState(run_date=now_et.date())
    
    logging.info("Environment loaded")
    logging.info("Symbols: %s", symbols)
    logging.info("Schedule: 08:00 premarket | 09:30-16:00 live signals | 16:05 shutdown")
    logging.info("Components: Participant Analysis | Regime Risk | Entry Scoring | 12-Gate Validation")
    
    push_heartbeat(status="initialized", engine=engine)
    
    try:
        while True:
            now_et = datetime.now(TZ_NY)
            state = reset_daily_state(state, now_et)
            
            try:
                if is_trading_day(now_et):
                    premarket_dt = combine_today(now_et, PREMARKET_TIME)
                    market_open_dt = combine_today(now_et, MARKET_OPEN_TIME)
                    eod_shutdown_dt = combine_today(now_et, EOD_SHUTDOWN_TIME)
                    weekly_refit_dt = combine_today(now_et, WEEKLY_REFIT_TIME)
                    
                    # 08:00 - Premarket build
                    if time_in_range(now_et, PREMARKET_TIME, MARKET_OPEN_TIME) and not state.premarket_done:
                        run_premarket_build(symbols)
                        state.premarket_done = True
                    
                    # 09:30 - Market open
                    if time_in_range(now_et, MARKET_OPEN_TIME, MARKET_CLOSE_TIME) and not state.market_open_logged:
                        logging.info("09:30 market open: production engine starting")
                        state.market_open_logged = True
                        state.next_signal_due = market_open_dt
                    
                    # Live trading hours - Generate signals
                    if time_in_range(now_et, MARKET_OPEN_TIME, MARKET_CLOSE_TIME):
                        if state.next_signal_due is None:
                            state.next_signal_due = market_open_dt
                        
                        if now_et >= state.next_signal_due:
                            signal = engine.run_signal_cycle(symbol=primary_symbol)
                            if signal:
                                logging.info(
                                    f"Signal generated: {signal.direction_str} {signal.symbol} "
                                    f"@ {signal.entry_price:.2f} "
                                    f"(Model: {signal.entry_model}, "
                                    f"Participant: {signal.dominant_participant}, "
                                    f"Regime: {signal.regime})"
                                )
                            else:
                                logging.info("No signal generated this cycle")
                            
                            state.next_signal_due = state.next_signal_due + SIGNAL_INTERVAL
                    
                    # 16:05 - EOD shutdown
                    if time_in_range(now_et, EOD_SHUTDOWN_TIME, SESSION_END_TIME) and not state.eod_done:
                        run_eod_shutdown(firebase)
                        state.eod_done = True
                    
                    # Weekly refit (Fridays at 16:10)
                    week_key = f"{now_et.isocalendar().year}-W{now_et.isocalendar().week:02d}"
                    if (
                        now_et.weekday() == 4
                        and time_in_range(now_et, WEEKLY_REFIT_TIME, SESSION_END_TIME)
                        and state.weekly_refit_week_key != week_key
                    ):
                        run_weekly_refit_check(now_et)
                        state.weekly_refit_week_key = week_key
                
                time.sleep(SLEEP_SECONDS)
                
            except Exception as exc:
                logging.exception("Engine loop iteration failed: %s", exc)
                time.sleep(SLEEP_SECONDS)
                
    except KeyboardInterrupt:
        logging.info("Engine stopped by user")
        push_heartbeat(status="stopped")


if __name__ == "__main__":
    main()
