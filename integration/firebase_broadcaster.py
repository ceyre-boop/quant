"""Firebase Broadcaster - Push trading data to Firebase Realtime Database.

Manages all writes to Firebase RTDB with structured paths for frontend consumption.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from firebase.client import FirebaseClient
from integration.firebase_ui_writer import (
    format_signal_for_ui,
    format_bias_for_ui,
    format_risk_for_ui,
    format_game_output_for_ui,
    format_regime_for_ui,
)
from contracts.types import (
    BiasOutput,
    RiskOutput,
    GameOutput,
    RegimeState,
    PositionState,
    AccountState,
    EntrySignal,
)

logger = logging.getLogger(__name__)


class FirebaseBroadcaster:
    """Broadcast trading signals and system state to Firebase RTDB."""

    def __init__(self, firebase_client: Optional[FirebaseClient] = None):
        """Initialize broadcaster.

        Args:
            firebase_client: Optional FirebaseClient instance (creates new if None)
        """
        self.client = firebase_client or FirebaseClient()
        self._enabled = self.client.rtdb is not None

        if not self._enabled:
            logger.warning("FirebaseBroadcaster initialized without RTDB access")

    def _get_rtdb_ref(self, path: str):
        """Get RTDB reference for a path."""
        if not self._enabled or self.client.rtdb is None:
            return None
        return self.client.rtdb.reference(path)

    def publish_signal(
        self,
        symbol: str,
        bias: BiasOutput,
        risk: RiskOutput,
        game: GameOutput,
        regime: RegimeState,
        current_price: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Publish complete signal to /signals/{symbol}/latest and /signals/{symbol}/history/{timestamp}.

        Args:
            symbol: Trading symbol
            bias: Layer 1 bias output
            risk: Layer 2 risk output
            game: Layer 3 game output
            regime: Current regime state
            current_price: Current market price
            timestamp: Signal timestamp

        Returns:
            True if published successfully
        """
        if not self._enabled:
            logger.debug("Broadcaster disabled, skipping signal publish")
            return False

        try:
            # Format signal for UI
            signal = format_signal_for_ui(
                symbol=symbol,
                bias=bias,
                risk=risk,
                game=game,
                regime=regime,
                current_price=current_price,
                timestamp=timestamp,
            )

            # Publish to /signals/{symbol}/latest
            latest_ref = self._get_rtdb_ref(f"signals/{symbol}/latest")
            if latest_ref:
                latest_ref.set(signal)
                logger.info(f"Published latest signal for {symbol}")

            # Publish to /signals/{symbol}/history/{timestamp}
            ts = timestamp or datetime.utcnow()
            history_key = ts.strftime("%Y%m%d_%H%M%S")
            history_ref = self._get_rtdb_ref(f"signals/{symbol}/history/{history_key}")
            if history_ref:
                history_ref.set(signal)
                logger.debug(f"Published history signal for {symbol} at {history_key}")

            return True

        except Exception as e:
            logger.error(f"Failed to publish signal for {symbol}: {e}")
            return False

    def publish_regime(self, symbol: str, regime: RegimeState, timestamp: Optional[datetime] = None) -> bool:
        """Publish regime state to /system/regime/{symbol}.

        Args:
            symbol: Trading symbol
            regime: Current regime state
            timestamp: Optional timestamp

        Returns:
            True if published successfully
        """
        if not self._enabled:
            return False

        try:
            ts = timestamp or datetime.utcnow()
            regime_data = {
                **format_regime_for_ui(regime),
                "composite_score": round(regime.composite_score, 2),
                "updated_at": ts.isoformat() + "Z",
            }

            ref = self._get_rtdb_ref(f"system/regime/{symbol}")
            if ref:
                ref.set(regime_data)
                logger.debug(f"Published regime for {symbol}")

            return True

        except Exception as e:
            logger.error(f"Failed to publish regime for {symbol}: {e}")
            return False

    def publish_signal_realtime(self, symbol: str, signal_data: Dict[str, Any]) -> bool:
        """Publish signal directly to Realtime Database for dashboard.

        Args:
            symbol: Trading symbol
            signal_data: Raw signal data dict

        Returns:
            True if published successfully
        """
        if not self._enabled:
            logger.debug("Broadcaster disabled, skipping realtime publish")
            return False

        try:
            # Publish to /signals/{symbol}/latest
            latest_ref = self._get_rtdb_ref(f"signals/{symbol}/latest")
            if latest_ref:
                latest_ref.set(signal_data)
                logger.info(f"Published realtime signal for {symbol}")

            # Also publish to history
            ts = datetime.utcnow()
            history_key = ts.strftime("%Y%m%d_%H%M%S")
            history_ref = self._get_rtdb_ref(f"signals/{symbol}/history/{history_key}")
            if history_ref:
                history_ref.set(signal_data)

            return True

        except Exception as e:
            logger.error(f"Failed to publish realtime signal for {symbol}: {e}")
            return False

    def publish_health(
        self,
        status: str = "healthy",
        components: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Publish system health to /system/health.

        Args:
            status: Overall health status ("healthy", "degraded", "down")
            components: Dict of component health states
            timestamp: Optional timestamp

        Returns:
            True if published successfully
        """
        if not self._enabled:
            return False

        try:
            ts = timestamp or datetime.utcnow()
            health_data = {
                "status": status,
                "components": components or {},
                "updated_at": ts.isoformat() + "Z",
                "timestamp_ms": int(ts.timestamp() * 1000),
            }

            ref = self._get_rtdb_ref("system/health")
            if ref:
                ref.set(health_data)
                logger.debug(f"Published health status: {status}")

            return True

        except Exception as e:
            logger.error(f"Failed to publish health: {e}")
            return False

    def publish_position(self, position: PositionState, timestamp: Optional[datetime] = None) -> bool:
        """Publish position state to /session/positions/{symbol}.

        Args:
            position: Current position state
            timestamp: Optional timestamp

        Returns:
            True if published successfully
        """
        if not self._enabled:
            return False

        try:
            ts = timestamp or datetime.utcnow()
            position_data = {
                "trade_id": position.trade_id,
                "symbol": position.symbol,
                "direction": position.direction.value,
                "entry_price": position.entry_price,
                "position_size": position.position_size,
                "stop_loss": position.stop_loss,
                "tp1": position.tp1,
                "tp2": position.tp2,
                "current_price": position.current_price,
                "unrealized_pnl": position.unrealized_pnl,
                "realized_pnl": position.realized_pnl,
                "status": position.status,
                "opened_at": (position.opened_at.isoformat() if position.opened_at else None),
                "updated_at": ts.isoformat() + "Z",
            }

            ref = self._get_rtdb_ref(f"session/positions/{position.symbol}")
            if ref:
                ref.set(position_data)
                logger.info(f"Published position for {position.symbol}: {position.status}")

            return True

        except Exception as e:
            logger.error(f"Failed to publish position: {e}")
            return False

    def publish_account(self, account: AccountState, timestamp: Optional[datetime] = None) -> bool:
        """Publish account state to /account/equity and /account/pnl.

        Args:
            account: Current account state
            timestamp: Optional timestamp

        Returns:
            True if published successfully
        """
        if not self._enabled:
            return False

        try:
            ts = timestamp or datetime.utcnow()

            # Equity info
            equity_data = {
                "equity": account.equity,
                "balance": account.balance,
                "margin_used": account.margin_used,
                "margin_available": account.margin_available,
                "open_positions": account.open_positions,
                "updated_at": ts.isoformat() + "Z",
            }

            # PnL info
            pnl_data = {
                "daily_pnl": account.daily_pnl,
                "daily_loss_pct": account.daily_loss_pct,
                "updated_at": ts.isoformat() + "Z",
            }

            equity_ref = self._get_rtdb_ref("account/equity")
            pnl_ref = self._get_rtdb_ref("account/pnl")

            if equity_ref:
                equity_ref.set(equity_data)
            if pnl_ref:
                pnl_ref.set(pnl_data)

            logger.debug(f"Published account state for {account.account_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish account: {e}")
            return False

    def publish_controls(
        self,
        trading_enabled: Optional[bool] = None,
        daily_loss_pct: Optional[float] = None,
        open_positions: Optional[int] = None,
        hard_logic_status: Optional[str] = None,
        manual_override: Optional[bool] = None,
    ) -> bool:
        """Publish session controls to /session/controls.

        Args:
            trading_enabled: Whether trading is enabled
            daily_loss_pct: Current daily loss percentage
            open_positions: Number of open positions
            hard_logic_status: Status of hard logic checks
            manual_override: Whether manual override is active

        Returns:
            True if published successfully
        """
        if not self._enabled:
            return False

        try:
            controls_data = {"updated_at": datetime.utcnow().isoformat() + "Z"}

            if trading_enabled is not None:
                controls_data["trading_enabled"] = trading_enabled
            if daily_loss_pct is not None:
                controls_data["daily_loss_pct"] = round(daily_loss_pct, 4)
            if open_positions is not None:
                controls_data["open_positions"] = open_positions
            if hard_logic_status is not None:
                controls_data["hard_logic_status"] = hard_logic_status
            if manual_override is not None:
                controls_data["manual_override"] = manual_override

            ref = self._get_rtdb_ref("session/controls")
            if ref:
                ref.update(controls_data)
                logger.debug("Published session controls")

            return True

        except Exception as e:
            logger.error(f"Failed to publish controls: {e}")
            return False

    def update_model_status(
        self,
        model_name: str,
        version: str,
        status: str = "active",
        last_prediction: Optional[datetime] = None,
        accuracy: Optional[float] = None,
    ) -> bool:
        """Update model status in /system/models/{model_name}.

        Args:
            model_name: Name of the model ("bias", "risk", "game")
            version: Model version string
            status: Model status ("active", "stale", "error")
            last_prediction: Timestamp of last prediction
            accuracy: Current model accuracy

        Returns:
            True if published successfully
        """
        if not self._enabled:
            return False

        try:
            ts = datetime.utcnow()
            model_data = {
                "version": version,
                "status": status,
                "updated_at": ts.isoformat() + "Z",
            }

            if last_prediction:
                model_data["last_prediction"] = last_prediction.isoformat() + "Z"
            if accuracy is not None:
                model_data["accuracy"] = round(accuracy, 4)

            ref = self._get_rtdb_ref(f"system/models/{model_name}")
            if ref:
                ref.set(model_data)
                logger.debug(f"Published model status for {model_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to publish model status: {e}")
            return False

    def delete_position(self, symbol: str) -> bool:
        """Delete position from /session/positions/{symbol}.

        Args:
            symbol: Trading symbol

        Returns:
            True if deleted successfully
        """
        if not self._enabled:
            return False

        try:
            ref = self._get_rtdb_ref(f"session/positions/{symbol}")
            if ref:
                ref.delete()
                logger.info(f"Deleted position for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete position: {e}")
            return False

    def get_controls(self) -> Dict[str, Any]:
        """Get current session controls.

        Returns:
            Current controls dict or empty dict
        """
        if not self._enabled:
            return {}

        ref = self._get_rtdb_ref("session/controls")
        if ref:
            return ref.get() or {}
        return {}

    def get_latest_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest signal for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Latest signal dict or None
        """
        if not self._enabled:
            return None

        ref = self._get_rtdb_ref(f"signals/{symbol}/latest")
        if ref:
            return ref.get()
        return None
