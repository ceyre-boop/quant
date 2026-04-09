import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
TZ_NY = ZoneInfo("America/New_York")
SLEEP_SECONDS = 20
ENGINE_VERSION = os.getenv("ENGINE_VERSION", "1.0.0")
LOG_PATH = "logs/engine.log"

PREMARKET_TIME = dt_time(hour=8, minute=0)
MARKET_OPEN_TIME = dt_time(hour=9, minute=30)
MARKET_CLOSE_TIME = dt_time(hour=16, minute=0)
EOD_SHUTDOWN_TIME = dt_time(hour=16, minute=5)
WEEKLY_REFIT_TIME = dt_time(hour=16, minute=10)
SESSION_END_TIME = dt_time(hour=18, minute=0)
SIGNAL_INTERVAL = timedelta(minutes=5)

DEFAULT_SYMBOL = "NAS100"
DEFAULT_SYMBOLS = ["NAS100", "US30", "SPX500", "XAUUSD"]


@dataclass
class RuntimeState:
    run_date: date
    premarket_done: bool = False
    market_open_logged: bool = False
    eod_done: bool = False
    next_signal_due: datetime | None = None
    weekly_refit_week_key: str | None = None


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
    """Load environment variables and verify required configuration."""
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


def generate_signal():
    """Generate a signal by adapting to available example module APIs."""
    from examples import generate_sample_signal as sample_module

    if hasattr(sample_module, "generate_sample_signal"):
        return sample_module.generate_sample_signal()
    if hasattr(sample_module, "create_sample_three_layer_context"):
        return sample_module.create_sample_three_layer_context()

    raise RuntimeError(
        "examples.generate_sample_signal has neither " "generate_sample_signal() nor create_sample_three_layer_context()"
    )


def broadcast_to_firebase(signal, symbol: str = DEFAULT_SYMBOL) -> bool:
    """Broadcast signal using whichever broadcaster API is available."""
    try:
        from integration import firebase_broadcaster as broadcaster_module
    except Exception as exc:
        logging.warning("Firebase broadcaster unavailable: %s", exc)
        return False

    if hasattr(broadcaster_module, "broadcast_signal"):
        broadcaster_module.broadcast_signal(signal)
        return True

    if hasattr(broadcaster_module, "FirebaseBroadcaster"):
        broadcaster = broadcaster_module.FirebaseBroadcaster()
        return broadcaster.publish_signal(
            symbol=symbol,
            bias=signal.bias,
            risk=signal.risk,
            game=signal.game,
            regime=signal.regime,
        )

    logging.warning("No supported Firebase broadcast API found")
    return False


def run_premarket_build(symbols: list[str]) -> None:
    """Run 08:00 premarket feature build using data pipeline."""
    logging.info("08:00 premarket: building features for %s", symbols)

    try:
        from data.pipeline import DataPipeline

        pipeline = DataPipeline()
        records = pipeline.run_premarket(symbols=symbols, include_all_features=True)
        valid_count = sum(1 for record in records.values() if getattr(record, "is_valid", False))
        logging.info(
            "Premarket feature build complete: %s/%s valid records",
            valid_count,
            len(records),
        )
    except Exception as exc:
        logging.exception("Premarket feature build failed: %s", exc)


def run_signal_cycle(symbol: str = DEFAULT_SYMBOL) -> None:
    """Run one full signal cycle and broadcast result."""
    logging.info("Running intraday signal cycle")
    signal = generate_signal()
    logging.info("Signal generated")

    if broadcast_to_firebase(signal, symbol=symbol):
        logging.info("Signal sent to Firebase")
    else:
        logging.info("Firebase broadcast skipped")

    push_heartbeat(status="running")


def run_eod_shutdown() -> None:
    """Run 16:05 shutdown tasks and logging."""
    logging.info("16:05 shutdown: finalizing session and publishing health")
    try:
        from integration.firebase_broadcaster import FirebaseBroadcaster

        broadcaster = FirebaseBroadcaster()
        broadcaster.publish_health(status="healthy", components={"lifecycle": "eod_shutdown"})
    except Exception as exc:
        logging.warning("EOD health publish skipped: %s", exc)

    logging.info("Session shutdown complete")
    push_heartbeat(status="shutdown")


def push_heartbeat(status: str) -> None:
    """Push engine heartbeat to Firebase /system/health."""
    payload = {
        "status": status,
        "last_cycle": datetime.now(TZ_NY).isoformat(),
        "engine_version": ENGINE_VERSION,
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


def run_weekly_refit_check(now_et: datetime) -> None:
    """Run weekly model refit evaluation using existing scheduler class."""
    logging.info("Weekly refit check triggered")
    try:
        from meta_evaluator.refit_scheduler import RefitScheduler

        scheduler = RefitScheduler()

        # Prefer environment-provided metrics; defaults avoid false-positive refits.
        metrics = {
            "win_rate": float(os.getenv("WEEKLY_WIN_RATE", "0.55")),
            "sharpe": float(os.getenv("WEEKLY_SHARPE", "0.8")),
            "total_trades": int(os.getenv("WEEKLY_TOTAL_TRADES", "0")),
        }
        drift_detected = os.getenv("WEEKLY_DRIFT_DETECTED", "false").lower() == "true"

        evaluation = scheduler.evaluate_refit_need(metrics, drift_detected=drift_detected)

        if evaluation.get("should_refit"):
            schedule = scheduler.schedule_refit(evaluation, model_version=f"weekly-{now_et.date().isoformat()}")
            logging.info("Weekly refit scheduled: %s", schedule)
        else:
            logging.info("No weekly refit scheduled: %s", evaluation.get("reasons", []))
    except Exception as exc:
        logging.exception("Weekly refit check failed: %s", exc)


def reset_daily_state(state: RuntimeState, now_et: datetime) -> RuntimeState:
    """Reset one-time daily markers at the start of a new calendar day."""
    if state.run_date == now_et.date():
        return state
    return RuntimeState(run_date=now_et.date(), weekly_refit_week_key=state.weekly_refit_week_key)


def main() -> None:
    configure_logging()
    logging.info("Starting CLAWD Trading Engine...")

    load_environment()
    symbols = get_symbols()
    primary_symbol = symbols[0] if symbols else DEFAULT_SYMBOL

    now_et = datetime.now(TZ_NY)
    state = RuntimeState(run_date=now_et.date())

    logging.info("Environment loaded")
    logging.info("Engine initialized for timezone: America/New_York")
    logging.info("Symbols: %s", symbols)
    logging.info("Schedule: 08:00 premarket | 09:30-16:00 loop | 16:05 shutdown | weekly refit")
    push_heartbeat(status="initialized")

    while True:
        now_et = datetime.now(TZ_NY)
        state = reset_daily_state(state, now_et)

        try:
            if is_trading_day(now_et):
                premarket_dt = combine_today(now_et, PREMARKET_TIME)
                market_open_dt = combine_today(now_et, MARKET_OPEN_TIME)
                eod_shutdown_dt = combine_today(now_et, EOD_SHUTDOWN_TIME)
                weekly_refit_dt = combine_today(now_et, WEEKLY_REFIT_TIME)

                if time_in_range(now_et, PREMARKET_TIME, MARKET_OPEN_TIME) and not state.premarket_done:
                    run_premarket_build(symbols)
                    state.premarket_done = True

                if time_in_range(now_et, MARKET_OPEN_TIME, MARKET_CLOSE_TIME) and not state.market_open_logged:
                    logging.info("09:30 market open: starting live engine")
                    state.market_open_logged = True
                    state.next_signal_due = market_open_dt

                if time_in_range(now_et, MARKET_OPEN_TIME, MARKET_CLOSE_TIME):
                    if state.next_signal_due is None:
                        state.next_signal_due = market_open_dt

                    if now_et >= state.next_signal_due:
                        run_signal_cycle(symbol=primary_symbol)
                        state.next_signal_due = state.next_signal_due + SIGNAL_INTERVAL

                if time_in_range(now_et, EOD_SHUTDOWN_TIME, SESSION_END_TIME) and not state.eod_done:
                    run_eod_shutdown()
                    state.eod_done = True

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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Engine stopped by user")
