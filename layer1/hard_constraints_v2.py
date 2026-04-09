"""Layer 1: Hard Constraints - Control Layer Rules

These rules CANNOT be bypassed by any model output.
Implements the control layer from Clawdbot v4.1 Section 1.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, time
from dataclasses import dataclass

from contracts.types import RiskOutput, AccountState, Direction

logger = logging.getLogger(__name__)


@dataclass
class ConstraintCheck:
    """Result of a constraint check."""

    passed: bool
    reason: Optional[str] = None
    severity: str = "error"  # 'error' blocks, 'warning' logs only


class HardConstraints:
    """Hard-logic control layer that cannot be bypassed.

    These rules are enforced regardless of model confidence or signal quality.
    """

    def __init__(
        self,
        daily_loss_limit_pct: float = 0.03,  # 3% daily loss limit
        max_positions: int = 5,
        max_position_size_pct: float = 0.02,  # 2% max risk per trade
        trading_hours_start: str = "09:35",
        trading_hours_end: str = "15:55",
        min_account_equity: float = 10000.0,
        firebase_client=None,
    ):
        self.firebase = firebase_client
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_positions = max_positions
        self.max_position_size_pct = max_position_size_pct
        self.trading_hours_start = time.fromisoformat(trading_hours_start)
        self.trading_hours_end = time.fromisoformat(trading_hours_end)
        self.min_account_equity = min_account_equity
        self.logger = logging.getLogger(__name__)

    def check_all_constraints(
        self,
        account: AccountState,
        risk: Optional[RiskOutput] = None,
        timestamp: Optional[datetime] = None,
    ) -> ConstraintCheck:
        """Run all hard constraint checks.

        Returns:
            ConstraintCheck - fails if ANY constraint is violated
        """
        # First check Firebase remote controls (emergency override)
        firebase_check = self.check_firebase_controls()
        if not firebase_check.passed:
            return firebase_check

        checks = [
            self.check_daily_loss_limit(account),
            self.check_max_positions(account),
            self.check_trading_hours(timestamp or datetime.now()),
            self.check_min_equity(account),
        ]

        if risk:
            checks.append(self.check_position_size(risk, account))

        # Return first failure
        for check in checks:
            if not check.passed:
                return check

        return ConstraintCheck(passed=True)

    def check_daily_loss_limit(self, account: AccountState) -> ConstraintCheck:
        """Check if daily loss limit has been hit.

        This rule CANNOT be bypassed - blocks all trading if hit.
        """
        if account.daily_loss_pct >= self.daily_loss_limit_pct:
            return ConstraintCheck(
                passed=False,
                reason=f"Daily loss limit hit: {account.daily_loss_pct:.2%} >= {self.daily_loss_limit_pct:.2%}",
                severity="error",
            )

        # Warning at 80% of limit
        if account.daily_loss_pct >= self.daily_loss_limit_pct * 0.8:
            self.logger.warning(
                f"Approaching daily loss limit: {account.daily_loss_pct:.2%}"
            )

        return ConstraintCheck(passed=True)

    def check_max_positions(self, account: AccountState) -> ConstraintCheck:
        """Check if max concurrent positions reached."""
        if account.open_positions >= self.max_positions:
            return ConstraintCheck(
                passed=False,
                reason=f"Max positions reached: {account.open_positions} >= {self.max_positions}",
                severity="error",
            )

        return ConstraintCheck(passed=True)

    def check_position_size(
        self, risk: RiskOutput, account: AccountState
    ) -> ConstraintCheck:
        """Check if position size exceeds maximum allowed."""
        if account.equity <= 0:
            return ConstraintCheck(
                passed=False, reason="Invalid account equity", severity="error"
            )

        position_risk_pct = risk.position_size * abs(account.equity / account.equity)

        if position_risk_pct > self.max_position_size_pct:
            return ConstraintCheck(
                passed=False,
                reason=f"Position size too large: {position_risk_pct:.2%} > {self.max_position_size_pct:.2%}",
                severity="error",
            )

        return ConstraintCheck(passed=True)

    def check_trading_hours(self, timestamp: datetime) -> ConstraintCheck:
        """Check if current time is within trading hours.

        Default: 09:35 - 15:55 EST (avoids open/close volatility)
        """
        current_time = timestamp.time()

        # Check if before market open
        if current_time < self.trading_hours_start:
            return ConstraintCheck(
                passed=False,
                reason=f"Before trading hours: {current_time} < {self.trading_hours_start}",
                severity="error",
            )

        # Check if after market close
        if current_time > self.trading_hours_end:
            return ConstraintCheck(
                passed=False,
                reason=f"After trading hours: {current_time} > {self.trading_hours_end}",
                severity="error",
            )

        return ConstraintCheck(passed=True)

    def check_min_equity(self, account: AccountState) -> ConstraintCheck:
        """Check if account has minimum required equity."""
        if account.equity < self.min_account_equity:
            return ConstraintCheck(
                passed=False,
                reason=f"Account equity too low: ${account.equity:.2f} < ${self.min_account_equity:.2f}",
                severity="error",
            )

        return ConstraintCheck(passed=True)

    def check_weekend_trading(self, timestamp: datetime) -> ConstraintCheck:
        """Block trading on weekends."""
        if timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return ConstraintCheck(
                passed=False, reason="Weekend trading blocked", severity="error"
            )

        return ConstraintCheck(passed=True)

    def check_holiday(
        self, timestamp: datetime, holidays: Optional[list] = None
    ) -> ConstraintCheck:
        """Block trading on market holidays."""
        if holidays is None:
            # Default US market holidays (simplified)
            holidays = [
                "2026-01-01",  # New Year's Day
                "2026-01-19",  # MLK Day
                "2026-02-16",  # Presidents Day
                "2026-04-03",  # Good Friday
                "2026-05-25",  # Memorial Day
                "2026-07-03",  # Independence Day (observed)
                "2026-09-07",  # Labor Day
                "2026-11-26",  # Thanksgiving
                "2026-12-25",  # Christmas
            ]

        date_str = timestamp.strftime("%Y-%m-%d")
        if date_str in holidays:
            return ConstraintCheck(
                passed=False, reason=f"Market holiday: {date_str}", severity="error"
            )

        return ConstraintCheck(passed=True)

    def check_firebase_controls(self) -> ConstraintCheck:
        """Check Firebase remote control switches.

        Allows instant shutdown from dashboard without code changes.
        """
        if self.firebase is None:
            return ConstraintCheck(passed=True)  # No Firebase, skip this check

        try:
            controls = self.firebase.read_realtime("/system_controls")

            if not controls:
                return ConstraintCheck(passed=True)  # No controls set, allow trading

            # Check emergency stop
            if controls.get("emergency_stop", False):
                return ConstraintCheck(
                    passed=False,
                    reason="Emergency stop activated via Firebase",
                    severity="error",
                )

            # Check trading enabled flag
            if not controls.get("trading_enabled", True):
                return ConstraintCheck(
                    passed=False,
                    reason="Trading disabled via Firebase control",
                    severity="error",
                )

            # Check max daily loss from Firebase (overrides default)
            firebase_max_loss = controls.get("max_daily_loss")
            if firebase_max_loss is not None:
                # Would need account balance to check this
                pass

            return ConstraintCheck(passed=True)

        except Exception as e:
            # If Firebase check fails, log but don't block trading
            self.logger.error(f"Firebase control check failed: {e}")
            return ConstraintCheck(passed=True)

    def get_constraint_status(self, account: AccountState) -> Dict[str, Any]:
        """Get status of all constraints."""
        return {
            "daily_loss_limit": {
                "current": account.daily_loss_pct,
                "limit": self.daily_loss_limit_pct,
                "remaining": max(0, self.daily_loss_limit_pct - account.daily_loss_pct),
                "status": (
                    "BLOCKED"
                    if account.daily_loss_pct >= self.daily_loss_limit_pct
                    else "OK"
                ),
            },
            "max_positions": {
                "current": account.open_positions,
                "limit": self.max_positions,
                "remaining": self.max_positions - account.open_positions,
                "status": (
                    "BLOCKED" if account.open_positions >= self.max_positions else "OK"
                ),
            },
            "trading_hours": {
                "start": self.trading_hours_start.isoformat(),
                "end": self.trading_hours_end.isoformat(),
                "status": "OK",  # Would need current time to evaluate
            },
            "equity": {
                "current": account.equity,
                "minimum": self.min_account_equity,
                "status": (
                    "OK" if account.equity >= self.min_account_equity else "BLOCKED"
                ),
            },
        }


def create_hard_constraints() -> HardConstraints:
    """Factory function to create hard constraints."""
    return HardConstraints()
