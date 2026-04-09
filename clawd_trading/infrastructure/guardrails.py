"""Execution Guardrails - System-wide safety controls for trading operations.

Reads trading controls from Firebase at `/system_controls/` and provides
an interface for enforcing trading limits and emergency stops.

Critical Safety Features:
- Trading enable/disable toggle
- Emergency stop mechanism
- Daily loss limits
- Position size limits
- Symbol whitelist
"""

import logging
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from integration.firebase_client import FirebaseClient

logger = logging.getLogger(__name__)


class GuardrailStatus(Enum):
    """Status of guardrail checks."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass
class SystemControls:
    """Trading system control parameters from Firebase.

    Path: `/system_controls/`
    """

    trading_enabled: bool = False
    emergency_stop: bool = False
    max_daily_loss: float = 0.05  # 5% default
    max_position_size: float = 100000.0  # $100k default
    allowed_symbols: List[str] = field(default_factory=list)
    max_open_positions: int = 5
    max_positions_per_symbol: int = 1
    updated_at: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemControls":
        """Create SystemControls from Firebase data dictionary."""
        return cls(
            trading_enabled=data.get("trading_enabled", False),
            emergency_stop=data.get("emergency_stop", False),
            max_daily_loss=data.get("max_daily_loss", 0.05),
            max_position_size=data.get("max_position_size", 100000.0),
            allowed_symbols=data.get("allowed_symbols", []),
            max_open_positions=data.get("max_open_positions", 5),
            max_positions_per_symbol=data.get("max_positions_per_symbol", 1),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data.get("updated_at")
                else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Firebase storage."""
        return {
            "trading_enabled": self.trading_enabled,
            "emergency_stop": self.emergency_stop,
            "max_daily_loss": self.max_daily_loss,
            "max_position_size": self.max_position_size,
            "allowed_symbols": self.allowed_symbols,
            "max_open_positions": self.max_open_positions,
            "max_positions_per_symbol": self.max_positions_per_symbol,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    status: GuardrailStatus
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ExecutionGuardrails:
    """Execution guardrail system for trading safety.

    Reads system controls from Firebase and enforces trading limits.
    All trade executions should pass through this system.

    Usage:
        guardrails = ExecutionGuardrails()
        result = guardrails.check_trading_allowed()
        if not result.passed:
            logger.error(f"Trading blocked: {result.message}")
            return
    """

    # Default allowed symbols
    DEFAULT_SYMBOLS = ["NAS100", "US30", "SPX500", "EURUSD", "GBPUSD", "XAUUSD"]

    def __init__(
        self, firebase_client: Optional[FirebaseClient] = None, demo_mode: bool = True
    ):
        """Initialize execution guardrails.

        Args:
            firebase_client: Firebase client for reading controls
            demo_mode: If True, uses default controls without Firebase
        """
        self.client = firebase_client or FirebaseClient(demo_mode=demo_mode)
        self._controls: Optional[SystemControls] = None
        self._last_refresh: Optional[datetime] = None
        self._cache_ttl_seconds = 30  # Refresh controls every 30 seconds

        logger.info("ExecutionGuardrails initialized")

    def _refresh_controls(self) -> SystemControls:
        """Refresh system controls from Firebase.

        Returns:
            Current system controls
        """
        try:
            data = self.client.rtdb_get("/system_controls")

            if data:
                self._controls = SystemControls.from_dict(data)
                self._last_refresh = datetime.utcnow()
                logger.debug(
                    f"Refreshed system controls: trading_enabled={self._controls.trading_enabled}"
                )
            else:
                # Use defaults if no data in Firebase
                if self._controls is None:
                    self._controls = SystemControls(
                        trading_enabled=False,  # Default to OFF for safety
                        allowed_symbols=self.DEFAULT_SYMBOLS,
                    )
                    logger.warning(
                        "No system controls found in Firebase, using safe defaults"
                    )

        except Exception as e:
            logger.error(f"Failed to refresh system controls: {e}")
            # Keep existing controls or use defaults
            if self._controls is None:
                self._controls = SystemControls(
                    trading_enabled=False, allowed_symbols=self.DEFAULT_SYMBOLS
                )

        return self._controls

    def get_controls(self) -> SystemControls:
        """Get current system controls, refreshing if needed."""
        if (
            self._controls is None
            or self._last_refresh is None
            or (datetime.utcnow() - self._last_refresh).total_seconds()
            > self._cache_ttl_seconds
        ):
            return self._refresh_controls()
        return self._controls

    def check_trading_allowed(self) -> GuardrailResult:
        """Check if trading is globally enabled.

        Checks:
        - trading_enabled flag
        - emergency_stop flag

        Returns:
            GuardrailResult with status and message
        """
        controls = self.get_controls()

        # Check emergency stop first (highest priority)
        if controls.emergency_stop:
            return GuardrailResult(
                status=GuardrailStatus.EMERGENCY_STOP,
                passed=False,
                message="EMERGENCY STOP ACTIVE - All trading halted",
                details={"emergency_stop": True},
            )

        # Check trading enabled
        if not controls.trading_enabled:
            return GuardrailResult(
                status=GuardrailStatus.FAIL,
                passed=False,
                message="Trading is currently disabled",
                details={"trading_enabled": False},
            )

        return GuardrailResult(
            status=GuardrailStatus.PASS,
            passed=True,
            message="Trading is enabled",
            details={"trading_enabled": True},
        )

    def check_symbol_allowed(self, symbol: str) -> GuardrailResult:
        """Check if a symbol is allowed for trading.

        Args:
            symbol: Trading symbol to check

        Returns:
            GuardrailResult with status and message
        """
        controls = self.get_controls()

        # If no allowed symbols defined, allow all (backward compatible)
        if not controls.allowed_symbols:
            return GuardrailResult(
                status=GuardrailStatus.PASS,
                passed=True,
                message=f"Symbol {symbol} allowed (no whitelist configured)",
                details={"symbol": symbol, "whitelisted": True},
            )

        # Check if symbol is in allowed list
        allowed_set = set(s.upper() for s in controls.allowed_symbols)
        if symbol.upper() in allowed_set:
            return GuardrailResult(
                status=GuardrailStatus.PASS,
                passed=True,
                message=f"Symbol {symbol} is whitelisted",
                details={"symbol": symbol, "whitelisted": True},
            )

        return GuardrailResult(
            status=GuardrailStatus.FAIL,
            passed=False,
            message=f"Symbol {symbol} is not in allowed_symbols list",
            details={
                "symbol": symbol,
                "whitelisted": False,
                "allowed_symbols": controls.allowed_symbols,
            },
        )

    def check_position_limits(
        self,
        position_size: float,
        symbol: str,
        current_positions: Optional[Dict[str, Any]] = None,
    ) -> GuardrailResult:
        """Check if position size is within limits.

        Args:
            position_size: Proposed position size in base currency
            symbol: Trading symbol
            current_positions: Optional dict of current positions for limit checks

        Returns:
            GuardrailResult with status and message
        """
        controls = self.get_controls()
        details = {
            "position_size": position_size,
            "max_position_size": controls.max_position_size,
            "symbol": symbol,
        }

        # Check max position size
        if position_size > controls.max_position_size:
            return GuardrailResult(
                status=GuardrailStatus.FAIL,
                passed=False,
                message=f"Position size {position_size} exceeds maximum {controls.max_position_size}",
                details=details,
            )

        # Check position size is positive
        if position_size <= 0:
            return GuardrailResult(
                status=GuardrailStatus.FAIL,
                passed=False,
                message=f"Invalid position size: {position_size}",
                details=details,
            )

        # Check max open positions if current positions provided
        if current_positions is not None:
            open_count = len(
                [p for p in current_positions.values() if p.get("status") == "OPEN"]
            )
            if open_count >= controls.max_open_positions:
                return GuardrailResult(
                    status=GuardrailStatus.FAIL,
                    passed=False,
                    message=f"Max open positions ({controls.max_open_positions}) reached",
                    details={**details, "current_open": open_count},
                )
            details["current_open_positions"] = open_count

        return GuardrailResult(
            status=GuardrailStatus.PASS,
            passed=True,
            message=f"Position size {position_size} within limits",
            details=details,
        )

    def check_daily_loss(
        self, daily_pnl: float, starting_equity: float
    ) -> GuardrailResult:
        """Check if daily loss limit has been reached.

        Args:
            daily_pnl: Current day's profit/loss
            starting_equity: Account equity at start of day

        Returns:
            GuardrailResult with status and message
        """
        controls = self.get_controls()

        if starting_equity <= 0:
            return GuardrailResult(
                status=GuardrailStatus.FAIL,
                passed=False,
                message="Invalid starting equity",
                details={"starting_equity": starting_equity},
            )

        daily_loss_pct = abs(min(0, daily_pnl)) / starting_equity
        details = {
            "daily_pnl": daily_pnl,
            "daily_loss_pct": daily_loss_pct,
            "max_daily_loss": controls.max_daily_loss,
            "starting_equity": starting_equity,
        }

        # Check if daily loss limit exceeded
        if daily_loss_pct >= controls.max_daily_loss:
            return GuardrailResult(
                status=GuardrailStatus.FAIL,
                passed=False,
                message=f"Daily loss limit exceeded: {daily_loss_pct:.2%} >= {controls.max_daily_loss:.2%}",
                details=details,
            )

        # Warning at 80% of limit
        if daily_loss_pct >= controls.max_daily_loss * 0.8:
            return GuardrailResult(
                status=GuardrailStatus.WARNING,
                passed=True,
                message=f"Daily loss at {daily_loss_pct:.2%} (80% of limit)",
                details=details,
            )

        return GuardrailResult(
            status=GuardrailStatus.PASS,
            passed=True,
            message=f"Daily loss {daily_loss_pct:.2%} within limits",
            details=details,
        )

    def validate_trade(
        self,
        symbol: str,
        position_size: float,
        daily_pnl: float,
        starting_equity: float,
        current_positions: Optional[Dict[str, Any]] = None,
    ) -> GuardrailResult:
        """Perform full trade validation against all guardrails.

        Args:
            symbol: Trading symbol
            position_size: Proposed position size
            daily_pnl: Current daily P&L
            starting_equity: Starting equity for loss calculation
            current_positions: Current open positions

        Returns:
            GuardrailResult - passed=True only if ALL checks pass
        """
        checks = [
            ("trading_allowed", self.check_trading_allowed()),
            ("symbol_allowed", self.check_symbol_allowed(symbol)),
            (
                "position_limits",
                self.check_position_limits(position_size, symbol, current_positions),
            ),
            ("daily_loss", self.check_daily_loss(daily_pnl, starting_equity)),
        ]

        failed_checks = [(name, result) for name, result in checks if not result.passed]

        if failed_checks:
            failed_names = [name for name, _ in failed_checks]
            failed_messages = [
                f"{name}: {result.message}" for name, result in failed_checks
            ]

            # Check if any is emergency stop
            is_emergency = any(
                result.status == GuardrailStatus.EMERGENCY_STOP
                for _, result in failed_checks
            )

            return GuardrailResult(
                status=(
                    GuardrailStatus.EMERGENCY_STOP
                    if is_emergency
                    else GuardrailStatus.FAIL
                ),
                passed=False,
                message=f"Trade validation failed: {'; '.join(failed_names)}",
                details={
                    "failed_checks": failed_names,
                    "messages": failed_messages,
                    "all_results": {
                        name: (
                            result.to_dict()
                            if hasattr(result, "to_dict")
                            else str(result)
                        )
                        for name, result in checks
                    },
                },
            )

        # Aggregate warnings
        warnings = [
            name for name, result in checks if result.status == GuardrailStatus.WARNING
        ]
        status = GuardrailStatus.WARNING if warnings else GuardrailStatus.PASS

        return GuardrailResult(
            status=status,
            passed=True,
            message=f"Trade validation passed{' with warnings: ' + ', '.join(warnings) if warnings else ''}",
            details={
                "checks_passed": [name for name, _ in checks],
                "warnings": warnings,
                "all_results": {
                    name: (
                        result.to_dict() if hasattr(result, "to_dict") else str(result)
                    )
                    for name, result in checks
                },
            },
        )

    def update_controls(self, controls: SystemControls) -> bool:
        """Update system controls in Firebase.

        Args:
            controls: New system controls

        Returns:
            True if update successful
        """
        try:
            controls.updated_at = datetime.utcnow()
            self.client.rtdb_set("/system_controls", controls.to_dict())
            self._controls = controls
            self._last_refresh = controls.updated_at
            logger.info(
                f"Updated system controls: trading_enabled={controls.trading_enabled}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update system controls: {e}")
            return False

    def trigger_emergency_stop(self, reason: str) -> bool:
        """Trigger emergency stop.

        Args:
            reason: Reason for emergency stop

        Returns:
            True if stop was triggered
        """
        try:
            controls = self.get_controls()
            controls.emergency_stop = True
            controls.trading_enabled = False
            controls.updated_at = datetime.utcnow()

            self.client.rtdb_set("/system_controls", controls.to_dict())
            self._controls = controls

            logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
            return True
        except Exception as e:
            logger.error(f"Failed to trigger emergency stop: {e}")
            return False

    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop (requires manual intervention).

        Returns:
            True if reset was successful
        """
        try:
            controls = self.get_controls()
            controls.emergency_stop = False
            controls.updated_at = datetime.utcnow()

            self.client.rtdb_set("/system_controls", controls.to_dict())
            self._controls = controls

            logger.warning("Emergency stop has been reset")
            return True
        except Exception as e:
            logger.error(f"Failed to reset emergency stop: {e}")
            return False


def create_default_controls(
    trading_enabled: bool = False,
    max_daily_loss: float = 0.05,
    allowed_symbols: Optional[List[str]] = None,
) -> SystemControls:
    """Factory function to create default system controls.

    Args:
        trading_enabled: Whether trading is enabled
        max_daily_loss: Maximum daily loss as decimal
        allowed_symbols: List of allowed trading symbols

    Returns:
        SystemControls with defaults
    """
    return SystemControls(
        trading_enabled=trading_enabled,
        emergency_stop=False,
        max_daily_loss=max_daily_loss,
        max_position_size=100000.0,
        allowed_symbols=allowed_symbols or ExecutionGuardrails.DEFAULT_SYMBOLS,
        max_open_positions=5,
        max_positions_per_symbol=1,
        updated_at=datetime.utcnow(),
    )
