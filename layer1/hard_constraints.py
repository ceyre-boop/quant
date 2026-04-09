"""Hard Constraints - Control layer rules that cannot be overridden.

These are the hard-logic safety rules from Clawdbot v4.1 Section 1.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from contracts.types import (
    Direction,
    BiasOutput,
    RiskOutput,
    GameOutput,
    RegimeState,
    EventRisk,
    VolRegime,
    AccountState,
)

logger = logging.getLogger(__name__)


@dataclass
class ConstraintCheck:
    """Result of a constraint check."""

    passed: bool
    reason: Optional[str] = None
    severity: str = "INFO"  # INFO, WARNING, BLOCK


class HardConstraints:
    """Hard-logic control layer - cannot be overridden by model output."""

    # Risk limits
    MAX_DAILY_LOSS_PCT = 0.03  # 3% max daily loss
    MAX_POSITION_SIZE_PCT = 0.05  # 5% max position size
    MAX_CONCURRENT_POSITIONS = 5

    # Market condition blocks
    VIX_EXTREME_THRESHOLD = 40
    MAX_SPREAD_PCT = 0.002  # 0.2% max spread

    # Time-based blocks
    PRE_MARKET_MINUTES = 30  # Block first 30 min
    LAST_HOUR_CUTOFF = 15  # No new positions after 3 PM

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.block_history: List[Dict[str, Any]] = []

    def check_all_constraints(
        self,
        bias: BiasOutput,
        risk: RiskOutput,
        regime: RegimeState,
        account: AccountState,
        current_time: Optional[datetime] = None,
    ) -> ConstraintCheck:
        """Check all hard constraints.

        Returns:
            ConstraintCheck with pass/fail status
        """
        current_time = current_time or datetime.utcnow()

        checks = [
            self.check_daily_loss_limit(account),
            self.check_event_risk(regime),
            self.check_volatility_extreme(regime),
            self.check_max_positions(account),
            self.check_position_size(risk, account),
            self.check_trading_hours(current_time),
            self.check_positive_ev(risk),
            self.check_confidence_threshold(bias),
        ]

        failed = [c for c in checks if not c.passed]

        if failed:
            # Return the most severe failure
            block_failures = [f for f in failed if f.severity == "BLOCK"]
            if block_failures:
                result = block_failures[0]
            else:
                result = failed[0]

            self._log_block(result)
            return result

        return ConstraintCheck(passed=True)

    def check_daily_loss_limit(self, account: AccountState) -> ConstraintCheck:
        """Check if daily loss limit has been hit."""
        if account.daily_loss_pct >= self.MAX_DAILY_LOSS_PCT:
            return ConstraintCheck(
                passed=False,
                reason=f"Daily loss limit hit: {account.daily_loss_pct:.2%} >= {self.MAX_DAILY_LOSS_PCT:.2%}",
                severity="BLOCK",
            )
        return ConstraintCheck(passed=True)

    def check_event_risk(self, regime: RegimeState) -> ConstraintCheck:
        """Check for high-impact event risk."""
        if regime.event_risk == EventRisk.EXTREME:
            return ConstraintCheck(
                passed=False,
                reason="EXTREME event risk - FOMC or major announcement",
                severity="BLOCK",
            )
        elif regime.event_risk == EventRisk.HIGH:
            return ConstraintCheck(
                passed=False,
                reason="HIGH event risk - reduced position sizes",
                severity="WARNING",
            )
        return ConstraintCheck(passed=True)

    def check_volatility_extreme(self, regime: RegimeState) -> ConstraintCheck:
        """Check for extreme volatility conditions."""
        if regime.volatility == VolRegime.EXTREME:
            return ConstraintCheck(
                passed=False,
                reason="EXTREME volatility regime - VIX spike",
                severity="BLOCK",
            )
        return ConstraintCheck(passed=True)

    def check_max_positions(self, account: AccountState) -> ConstraintCheck:
        """Check maximum concurrent positions."""
        if account.open_positions >= self.MAX_CONCURRENT_POSITIONS:
            return ConstraintCheck(
                passed=False,
                reason=f"Max positions reached: {account.open_positions}",
                severity="BLOCK",
            )
        return ConstraintCheck(passed=True)

    def check_position_size(
        self, risk: RiskOutput, account: AccountState
    ) -> ConstraintCheck:
        """Check position size limits."""
        max_position = account.equity * self.MAX_POSITION_SIZE_PCT

        if risk.position_size > max_position:
            return ConstraintCheck(
                passed=False,
                reason=f"Position size {risk.position_size:,.0f} exceeds max {max_position:,.0f}",
                severity="BLOCK",
            )
        return ConstraintCheck(passed=True)

    def check_trading_hours(self, current_time: datetime) -> ConstraintCheck:
        """Check trading hours constraints.

        Blocks:
        - Before 9:30 AM (pre-market)
        - After 3:00 PM (no new positions)
        """
        hour = current_time.hour
        minute = current_time.minute

        # Block pre-market (before 9:30 AM EST)
        if hour < 9 or (hour == 9 and minute < 30):
            return ConstraintCheck(
                passed=False, reason="Pre-market - no new positions", severity="BLOCK"
            )

        # Block after 3 PM
        if hour >= 15:
            return ConstraintCheck(
                passed=False, reason="After 3 PM - no new positions", severity="BLOCK"
            )

        return ConstraintCheck(passed=True)

    def check_positive_ev(self, risk: RiskOutput) -> ConstraintCheck:
        """Check that expected value is positive."""
        if not risk.ev_positive:
            return ConstraintCheck(
                passed=False,
                reason=f"Negative EV: {risk.expected_value:.4f}",
                severity="BLOCK",
            )
        return ConstraintCheck(passed=True)

    def check_confidence_threshold(self, bias: BiasOutput) -> ConstraintCheck:
        """Check minimum confidence threshold."""
        MIN_CONFIDENCE = 0.55

        if bias.confidence < MIN_CONFIDENCE:
            return ConstraintCheck(
                passed=False,
                reason=f"Confidence {bias.confidence:.2f} below threshold {MIN_CONFIDENCE}",
                severity="BLOCK",
            )
        return ConstraintCheck(passed=True)

    def check_bias_neutral(self, bias: BiasOutput) -> ConstraintCheck:
        """Check that bias is not neutral."""
        if bias.direction == Direction.NEUTRAL:
            return ConstraintCheck(
                passed=False, reason="Bias is NEUTRAL", severity="BLOCK"
            )
        return ConstraintCheck(passed=True)

    def _log_block(self, check: ConstraintCheck):
        """Log a constraint block."""
        self.block_history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "reason": check.reason,
                "severity": check.severity,
            }
        )

        if check.severity == "BLOCK":
            self.logger.warning(f"HARD CONSTRAINT BLOCK: {check.reason}")
        else:
            self.logger.info(f"Constraint warning: {check.reason}")

    def get_block_history(self) -> List[Dict[str, Any]]:
        """Get history of constraint blocks."""
        return self.block_history

    def clear_history(self):
        """Clear block history."""
        self.block_history = []


# Convenience function for quick checks
def check_constraints(
    bias: BiasOutput,
    risk: RiskOutput,
    regime: RegimeState,
    account: AccountState,
    current_time: Optional[datetime] = None,
) -> ConstraintCheck:
    """Quick constraint check."""
    constraints = HardConstraints()
    return constraints.check_all_constraints(bias, risk, regime, account, current_time)
