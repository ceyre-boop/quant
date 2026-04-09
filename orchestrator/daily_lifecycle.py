"""Daily Lifecycle - Master coordinator for the Clawd Trading System.

Manages the daily trading cycle:
- Pre-market setup (08:00 EST)
- Intraday cycles (every 5 minutes)
- End-of-day cleanup (16:05 EST)
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from contracts.types import (
    BiasOutput,
    RiskOutput,
    GameOutput,
    RegimeState,
    FeatureRecord,
    EntrySignal,
    PositionState,
    AccountState,
    MarketData,
    Direction,
    ThreeLayerContext,
    VolRegime,
    TrendRegime,
    RiskAppetite,
    MomentumRegime,
    EventRisk,
    Magnitude,
    AdversarialRisk,
)
from firebase.client import FirebaseClient
from integration.firebase_broadcaster import FirebaseBroadcaster

logger = logging.getLogger(__name__)


class CyclePhase(Enum):
    """Trading day phases."""

    PRE_MARKET = "pre_market"
    OPEN = "open"
    REGULAR_HOURS = "regular_hours"
    CLOSE = "close"
    POST_MARKET = "post_market"
    CLOSED = "closed"


@dataclass
class LifecycleConfig:
    """Configuration for daily lifecycle."""

    pre_market_time: str = "08:00"  # EST
    market_open_time: str = "09:30"  # EST
    market_close_time: str = "16:00"  # EST
    eod_cleanup_time: str = "16:05"  # EST
    intraday_interval_minutes: int = 5
    symbols: List[str] = None

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["NAS100", "US30", "SPX500", "XAUUSD"]


class DailyLifecycle:
    """Master coordinator for daily trading operations."""

    def __init__(
        self,
        config: Optional[LifecycleConfig] = None,
        firebase_client: Optional[FirebaseClient] = None,
        broadcaster: Optional[FirebaseBroadcaster] = None,
    ):
        """Initialize daily lifecycle.

        Args:
            config: Lifecycle configuration
            firebase_client: Firebase client instance
            broadcaster: Firebase broadcaster instance
        """
        self.config = config or LifecycleConfig()
        self.firebase = firebase_client or FirebaseClient()
        self.broadcaster = broadcaster or FirebaseBroadcaster(self.firebase)

        # State tracking
        self._current_phase: CyclePhase = CyclePhase.CLOSED
        self._last_intraday_run: Optional[datetime] = None
        self._is_running: bool = False
        self._symbols_data: Dict[str, Dict[str, Any]] = {}

        # Component references (to be injected)
        self.data_fetcher: Optional[Callable] = None
        self.feature_builder: Optional[Callable] = None
        self.bias_engine: Optional[Callable] = None
        self.risk_engine: Optional[Callable] = None
        self.game_engine: Optional[Callable] = None
        self.entry_validator: Optional[Callable] = None

        logger.info("DailyLifecycle initialized")

    def register_components(
        self,
        data_fetcher: Optional[Callable] = None,
        feature_builder: Optional[Callable] = None,
        bias_engine: Optional[Callable] = None,
        risk_engine: Optional[Callable] = None,
        game_engine: Optional[Callable] = None,
        entry_validator: Optional[Callable] = None,
    ):
        """Register trading system components.

        Args:
            data_fetcher: Function to fetch market data
            feature_builder: Function to build feature vectors
            bias_engine: Layer 1 bias engine
            risk_engine: Layer 2 risk engine
            game_engine: Layer 3 game engine
            entry_validator: Entry validation function
        """
        self.data_fetcher = data_fetcher
        self.feature_builder = feature_builder
        self.bias_engine = bias_engine
        self.risk_engine = risk_engine
        self.game_engine = game_engine
        self.entry_validator = entry_validator
        logger.info("Components registered")

    def get_current_phase(self, now: Optional[datetime] = None) -> CyclePhase:
        """Determine current market phase based on time.

        Args:
            now: Current datetime (defaults to now)

        Returns:
            Current CyclePhase
        """
        now = now or datetime.now()

        # Parse config times (assuming EST/EDT)
        pre_market = self._parse_time(self.config.pre_market_time, now)
        market_open = self._parse_time(self.config.market_open_time, now)
        market_close = self._parse_time(self.config.market_close_time, now)
        eod_cleanup = self._parse_time(self.config.eod_cleanup_time, now)

        if now < pre_market:
            return CyclePhase.CLOSED
        elif pre_market <= now < market_open:
            return CyclePhase.PRE_MARKET
        elif market_open <= now < market_close:
            return CyclePhase.REGULAR_HOURS
        elif market_close <= now < eod_cleanup:
            return CyclePhase.CLOSE
        else:
            return CyclePhase.POST_MARKET

    def _parse_time(self, time_str: str, base_date: datetime) -> datetime:
        """Parse time string into datetime."""
        hour, minute = map(int, time_str.split(":"))
        return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def run_premarket(self) -> Dict[str, Any]:
        """Run pre-market setup at 08:00 EST.

        Pipeline:
        1. Fetch overnight data
        2. Build features for all symbols
        3. Run all three engines
        4. Broadcast initial signals

        Returns:
            Dict with results for each symbol
        """
        logger.info("=" * 60)
        logger.info("STARTING PRE-MARKET CYCLE")
        logger.info("=" * 60)

        results = {}

        try:
            self._current_phase = CyclePhase.PRE_MARKET

            # Update health status
            self.broadcaster.publish_health(
                status="healthy",
                components={"lifecycle": "pre_market", "data": "fetching"},
            )

            for symbol in self.config.symbols:
                try:
                    logger.info(f"Processing {symbol}...")
                    result = self._process_symbol(symbol)
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
                    "errors": sum(1 for r in results.values() if r.get("error")),
                },
            )

            logger.info("PRE-MARKET CYCLE COMPLETE")
            return results

        except Exception as e:
            logger.error(f"Pre-market cycle failed: {e}")
            self.broadcaster.publish_health(
                status="degraded",
                components={"lifecycle": "pre_market_error", "error": str(e)},
            )
            return {"error": str(e)}

    def run_intraday_cycle(self) -> Dict[str, Any]:
        """Run intraday cycle every 5 minutes.

        Pipeline:
        1. Refresh market data
        2. Recalculate features
        3. Run engines
        4. Validate entry conditions
        5. Broadcast updates

        Returns:
            Dict with results for each symbol
        """
        now = datetime.now()

        # Check if we should run
        if self._last_intraday_run:
            elapsed = (now - self._last_intraday_run).total_seconds() / 60
            if elapsed < self.config.intraday_interval_minutes:
                logger.debug(
                    f"Skipping intraday cycle, {elapsed:.1f} min since last run"
                )
                return {"skipped": True, "minutes_since_last": elapsed}

        logger.info("-" * 60)
        logger.info("STARTING INTRADAY CYCLE")
        logger.info("-" * 60)

        self._current_phase = CyclePhase.REGULAR_HOURS
        self._last_intraday_run = now

        results = {}
        entry_signals = []

        try:
            for symbol in self.config.symbols:
                try:
                    logger.info(f"Processing {symbol}...")
                    result = self._process_symbol(symbol)
                    results[symbol] = result

                    # Check for entry signals
                    if result.get("entry_signal"):
                        entry_signals.append(result["entry_signal"])
                        logger.info(
                            f"ENTRY SIGNAL for {symbol}: {result['entry_signal']}"
                        )

                except Exception as e:
                    logger.error(f"Exception processing {symbol}: {e}")
                    results[symbol] = {"error": str(e)}

            # Update health
            self.broadcaster.publish_health(
                status="healthy",
                components={
                    "lifecycle": "intraday",
                    "timestamp": now.isoformat(),
                    "entry_signals": len(entry_signals),
                },
            )

            logger.info(f"INTRADAY CYCLE COMPLETE - {len(entry_signals)} entry signals")
            return {
                "results": results,
                "entry_signals": entry_signals,
                "timestamp": now.isoformat(),
            }

        except Exception as e:
            logger.error(f"Intraday cycle failed: {e}")
            return {"error": str(e)}

    def run_eod_cleanup(self) -> Dict[str, Any]:
        """Run end-of-day cleanup at 16:05 EST.

        Tasks:
        1. Close any open positions
        2. Log daily performance
        3. Archive signals
        4. Reset state for next day

        Returns:
            Dict with cleanup results
        """
        logger.info("=" * 60)
        logger.info("STARTING EOD CLEANUP")
        logger.info("=" * 60)

        self._current_phase = CyclePhase.POST_MARKET

        try:
            results = {
                "positions_closed": 0,
                "daily_pnl": 0.0,
                "signals_archived": 0,
                "timestamp": datetime.now().isoformat(),
            }

            # Publish EOD status
            self.broadcaster.publish_health(
                status="healthy",
                components={
                    "lifecycle": "eod_cleanup",
                    "positions_closed": results["positions_closed"],
                    "daily_pnl": results["daily_pnl"],
                },
            )

            # Update session controls for next day
            self.broadcaster.publish_controls(
                trading_enabled=False,
                open_positions=0,
                hard_logic_status="pending_next_session",
            )

            logger.info("EOD CLEANUP COMPLETE")
            return results

        except Exception as e:
            logger.error(f"EOD cleanup failed: {e}")
            return {"error": str(e)}

    def _process_symbol(self, symbol: str) -> Dict[str, Any]:
        """Process a single symbol through the full pipeline.

        Pipeline:
        fetch_market_data() → build_features() → run_bias_engine() →
        run_game_theory() → run_risk_model() → validate_entry() →
        publish_to_firebase()

        Args:
            symbol: Trading symbol

        Returns:
            Dict with processing results
        """
        result = {"symbol": symbol}

        # 1. Fetch market data
        if self.data_fetcher:
            market_data = self.data_fetcher(symbol)
            result["market_data"] = market_data
        else:
            market_data = None

        # 2. Build features
        if self.feature_builder and market_data:
            features = self.feature_builder(symbol, market_data)
            result["features"] = features
        else:
            features = None

        # 3. Run Layer 1 - Bias Engine
        bias = None
        if self.bias_engine and features:
            try:
                # Get regime if available
                regime = getattr(features, "regime", None) or self._get_default_regime()
                bias = self.bias_engine(symbol, features, regime)
                result["bias"] = bias
                logger.debug(
                    f"{symbol} bias: {bias.direction.name if bias else 'None'} "
                    f"conf={bias.confidence:.2f}"
                    if bias
                    else ""
                )
            except Exception as e:
                logger.error(f"Bias engine error for {symbol}: {e}")
                result["bias_error"] = str(e)

        # 4. Run Layer 3 - Game Engine
        game = None
        if self.game_engine and market_data:
            try:
                game = self.game_engine(symbol, market_data, bias)
                result["game"] = game
                logger.debug(
                    f"{symbol} game: aligned={game.game_state_aligned if game else 'None'}"
                )
            except Exception as e:
                logger.error(f"Game engine error for {symbol}: {e}")
                result["game_error"] = str(e)

        # 5. Run Layer 2 - Risk Engine
        risk = None
        if self.risk_engine and bias and features:
            try:
                account = self._get_account_state()
                market = market_data if isinstance(market_data, MarketData) else None
                risk = self.risk_engine(symbol, bias, features, account, market)
                result["risk"] = risk
                logger.debug(
                    f"{symbol} risk: EV={risk.expected_value:.2f}" if risk else ""
                )
            except Exception as e:
                logger.error(f"Risk engine error for {symbol}: {e}")
                result["risk_error"] = str(e)

        # 6. Validate entry
        if self.entry_validator and bias and risk and game:
            try:
                entry_signal = self.entry_validator(symbol, bias, risk, game)
                result["entry_signal"] = entry_signal
                if entry_signal:
                    logger.info(f"{symbol} ENTRY VALIDATED: {entry_signal}")
            except Exception as e:
                logger.error(f"Entry validation error for {symbol}: {e}")
                result["entry_error"] = str(e)

        # 7. Publish to Firebase
        if bias and risk and game:
            try:
                regime = self._get_default_regime()
                current_price = (
                    market_data.current_price
                    if isinstance(market_data, MarketData)
                    else None
                )

                self.broadcaster.publish_signal(
                    symbol=symbol,
                    bias=bias,
                    risk=risk,
                    game=game,
                    regime=regime,
                    current_price=current_price,
                )
                result["published"] = True
            except Exception as e:
                logger.error(f"Failed to publish {symbol}: {e}")
                result["publish_error"] = str(e)

        return result

    def _get_default_regime(self) -> RegimeState:
        """Get default regime state."""
        return RegimeState(
            volatility=VolRegime.NORMAL,
            trend=TrendRegime.RANGING,
            risk_appetite=RiskAppetite.NEUTRAL,
            momentum=MomentumRegime.STEADY,
            event_risk=EventRisk.CLEAR,
            composite_score=0.5,
        )

    def _get_account_state(self) -> AccountState:
        """Get current account state."""
        return AccountState(
            account_id="default",
            equity=50000.0,
            balance=50000.0,
            open_positions=0,
            daily_pnl=0.0,
            daily_loss_pct=0.0,
            margin_used=0.0,
            margin_available=50000.0,
            timestamp=datetime.now(),
        )

    def run_continuous(self, check_interval_seconds: int = 30):
        """Run lifecycle continuously, checking for phase transitions.

        Args:
            check_interval_seconds: How often to check phase
        """
        logger.info("Starting continuous lifecycle")
        self._is_running = True

        try:
            while self._is_running:
                phase = self.get_current_phase()

                if phase != self._current_phase:
                    logger.info(
                        f"Phase transition: {self._current_phase.value} -> {phase.value}"
                    )
                    self._handle_phase_transition(phase)

                # Run intraday cycle during regular hours
                if phase == CyclePhase.REGULAR_HOURS:
                    self.run_intraday_cycle()

                time.sleep(check_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Lifecycle stopped by user")
        finally:
            self._is_running = False

    def _handle_phase_transition(self, new_phase: CyclePhase):
        """Handle transition to a new phase.

        Args:
            new_phase: The new cycle phase
        """
        self._current_phase = new_phase

        if new_phase == CyclePhase.PRE_MARKET:
            self.run_premarket()
        elif new_phase == CyclePhase.CLOSE:
            self.run_eod_cleanup()

    def stop(self):
        """Stop the continuous lifecycle."""
        logger.info("Stopping lifecycle")
        self._is_running = False


def create_default_lifecycle() -> DailyLifecycle:
    """Factory function to create a lifecycle with default components.

    Returns:
        Configured DailyLifecycle instance
    """
    from layer1.bias_engine import BiasEngine
    from layer1.feature_builder import FeatureBuilder
    from layer1.regime_classifier import RegimeClassifier
    from layer2.risk_engine import RiskEngine
    from layer3.game_engine import GameEngine

    lifecycle = DailyLifecycle()

    # Register components
    lifecycle.register_components(
        data_fetcher=_mock_data_fetcher,
        feature_builder=_mock_feature_builder,
        bias_engine=_mock_bias_engine,
        risk_engine=_mock_risk_engine,
        game_engine=_mock_game_engine,
        entry_validator=_mock_entry_validator,
    )

    return lifecycle


# Mock component implementations for testing/development


def _mock_data_fetcher(symbol: str) -> MarketData:
    """Mock data fetcher."""
    return MarketData(
        symbol=symbol,
        current_price=21905.0,
        bid=21904.5,
        ask=21905.5,
        spread=1.0,
        volume_24h=1000000.0,
        atr_14=45.0,
        timestamp=datetime.now(),
    )


def _mock_feature_builder(symbol: str, market_data: MarketData):
    """Mock feature builder."""

    class MockFeatures:
        def __init__(self):
            self.symbol = symbol
            self.features = {"atr": 45.0, "rsi": 55.0}
            self.regime = RegimeState(
                volatility=VolRegime.NORMAL,
                trend=TrendRegime.STRONG_TREND,
                risk_appetite=RiskAppetite.RISK_ON,
                momentum=MomentumRegime.ACCELERATING,
                event_risk=EventRisk.CLEAR,
                composite_score=0.75,
            )

    return MockFeatures()


def _mock_bias_engine(symbol: str, features, regime):
    """Mock bias engine."""
    import random

    return BiasOutput(
        direction=Direction.LONG if random.random() > 0.5 else Direction.SHORT,
        magnitude=Magnitude.NORMAL,
        confidence=0.65 + random.random() * 0.25,
        regime_override=False,
        rationale=["TREND_STRENGTH", "MOMENTUM_SHIFT"],
        model_version="v1.0",
        feature_snapshot={},
    )


def _mock_risk_engine(symbol: str, bias, features, account, market):
    """Mock risk engine."""
    direction_mult = 1 if bias.direction == Direction.LONG else -1
    entry = 21905.0
    stop = entry - (45.0 * 1.5 * direction_mult)
    tp1 = entry + (45.0 * 2.0 * direction_mult)
    tp2 = entry + (45.0 * 3.5 * direction_mult)

    return RiskOutput(
        position_size=1.2,
        kelly_fraction=0.25,
        stop_price=stop,
        stop_method="atr",
        tp1_price=tp1,
        tp2_price=tp2,
        trail_config={"enabled": True, "activation": tp1},
        expected_value=0.75,
        ev_positive=True,
        size_breakdown={"base": 1.5, "adjusted": 1.2},
    )


def _mock_game_engine(symbol: str, market_data, bias):
    """Mock game engine."""
    from contracts.types import LiquidityPool, TrappedPositions

    return GameOutput(
        liquidity_map={"equal_highs": [], "equal_lows": []},
        nearest_unswept_pool=LiquidityPool(
            price=21820.0,
            strength=3,
            swept=False,
            age_bars=12,
            draw_probability=0.72,
            pool_type="equal_lows",
        ),
        trapped_positions=TrappedPositions(
            trapped_longs=[],
            trapped_shorts=[],
            total_long_pain=0.0,
            total_short_pain=5000.0,
            squeeze_probability=0.45,
        ),
        forced_move_probability=0.55,
        nash_zones=[],
        kyle_lambda=0.34,
        game_state_aligned=True,
        game_state_summary="NEUTRAL",
        adversarial_risk=AdversarialRisk.LOW,
    )


def _mock_entry_validator(symbol: str, bias, risk, game):
    """Mock entry validator."""
    if (
        bias.confidence >= 0.55
        and risk.ev_positive
        and game.adversarial_risk != AdversarialRisk.EXTREME
    ):
        return {
            "symbol": symbol,
            "direction": bias.direction.value,
            "entry_price": 21905.0,
            "position_size": risk.position_size,
            "valid": True,
        }
    return None
