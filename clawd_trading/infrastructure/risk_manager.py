"""Risk Manager - Integrates guardrails with trading execution.

Provides a high-level interface for risk-checking trades before execution.
Coordinates with ExecutionGuardrails for system-level controls and adds
position-level risk calculations.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from infrastructure.guardrails import (
    ExecutionGuardrails, 
    GuardrailResult, 
    GuardrailStatus,
    SystemControls
)
from contracts.types import Direction, PositionState, AccountState

logger = logging.getLogger(__name__)


class RiskCheckType(Enum):
    """Types of risk checks performed."""
    SYSTEM = "SYSTEM"
    POSITION_SIZE = "POSITION_SIZE"
    DAILY_LOSS = "DAILY_LOSS"
    CORRELATION = "CORRELATION"
    VOLATILITY = "VOLATILITY"


@dataclass
class RiskCheck:
    """Individual risk check result."""
    check_type: RiskCheckType
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    """Complete risk assessment for a proposed trade."""
    can_execute: bool
    checks: List[RiskCheck]
    total_risk_pct: float
    position_risk_pct: float
    margin_required: float
    warnings: List[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'can_execute': self.can_execute,
            'checks': [
                {
                    'check_type': c.check_type.value,
                    'passed': c.passed,
                    'message': c.message,
                    'details': c.details
                } for c in self.checks
            ],
            'total_risk_pct': self.total_risk_pct,
            'position_risk_pct': self.position_risk_pct,
            'margin_required': self.margin_required,
            'warnings': self.warnings,
            'blocked_reason': self.blocked_reason,
            'timestamp': self.timestamp.isoformat()
        }


class RiskManager:
    """Risk manager for trading operations.
    
    Integrates guardrails with position-level risk calculations.
    All trade executions should flow through this manager.
    
    Usage:
        risk_manager = RiskManager()
        
        # Check if trading is allowed
        if not risk_manager.check_trading_allowed():
            return
        
        # Full risk assessment
        assessment = risk_manager.assess_trade(
            symbol='NAS100',
            direction=Direction.LONG,
            entry_price=15000.0,
            stop_loss=14950.0,
            position_size=1.0,
            account_state=account_state
        )
        
        if assessment.can_execute:
            execute_trade(...)
    """
    
    def __init__(
        self,
        guardrails: Optional[ExecutionGuardrails] = None,
        firebase_client: Optional[Any] = None,
        max_total_risk_pct: float = 0.02,  # 2% max risk per trade
        max_account_risk_pct: float = 0.06,  # 6% total account risk
        demo_mode: bool = True
    ):
        """Initialize risk manager.
        
        Args:
            guardrails: ExecutionGuardrails instance
            firebase_client: Firebase client
            max_total_risk_pct: Maximum risk per trade as % of equity
            max_account_risk_pct: Maximum total account risk
            demo_mode: Whether to run in demo mode
        """
        self.guardrails = guardrails or ExecutionGuardrails(
            firebase_client=firebase_client,
            demo_mode=demo_mode
        )
        self.max_total_risk_pct = max_total_risk_pct
        self.max_account_risk_pct = max_account_risk_pct
        
        # Track daily stats
        self._daily_trades: List[Dict[str, Any]] = []
        self._last_reset: Optional[datetime] = None
        
        logger.info(f"RiskManager initialized: max_risk={max_total_risk_pct:.2%}")
    
    def check_trading_allowed(self) -> bool:
        """Quick check if trading is globally allowed.
        
        Returns:
            True if trading is enabled and no emergency stop
        """
        result = self.guardrails.check_trading_allowed()
        return result.passed
    
    def check_position_limits(
        self,
        position_size: float,
        symbol: str,
        current_positions: Optional[Dict[str, PositionState]] = None
    ) -> GuardrailResult:
        """Check position against guardrail limits.
        
        Args:
            position_size: Proposed position size
            symbol: Trading symbol
            current_positions: Current open positions
            
        Returns:
            GuardrailResult
        """
        positions_dict = None
        if current_positions:
            positions_dict = {
                k: v.to_dict() if hasattr(v, 'to_dict') else v 
                for k, v in current_positions.items()
            }
        
        return self.guardrails.check_position_limits(
            position_size=position_size,
            symbol=symbol,
            current_positions=positions_dict
        )
    
    def check_daily_loss(self, account_state: AccountState) -> GuardrailResult:
        """Check daily loss limits.
        
        Args:
            account_state: Current account state
            
        Returns:
            GuardrailResult
        """
        # Calculate starting equity
        starting_equity = account_state.equity - account_state.daily_pnl
        if starting_equity <= 0:
            starting_equity = account_state.equity
        
        return self.guardrails.check_daily_loss(
            daily_pnl=account_state.daily_pnl,
            starting_equity=starting_equity
        )
    
    def calculate_position_risk(
        self,
        entry_price: float,
        stop_loss: float,
        direction: Direction,
        position_size: float
    ) -> float:
        """Calculate the risk amount for a position.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: Trade direction
            position_size: Position size
            
        Returns:
            Risk amount in currency
        """
        if direction == Direction.LONG:
            risk_per_unit = entry_price - stop_loss
        else:
            risk_per_unit = stop_loss - entry_price
        
        # Risk amount
        risk_amount = risk_per_unit * position_size
        
        return max(0, risk_amount)
    
    def calculate_risk_percentage(
        self,
        entry_price: float,
        stop_loss: float,
        direction: Direction,
        position_size: float,
        equity: float
    ) -> float:
        """Calculate position risk as percentage of equity.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: Trade direction
            position_size: Position size
            equity: Account equity
            
        Returns:
            Risk as percentage of equity
        """
        if equity <= 0:
            return float('inf')
        
        risk_amount = self.calculate_position_risk(
            entry_price, stop_loss, direction, position_size
        )
        
        return risk_amount / equity
    
    def assess_trade(
        self,
        symbol: str,
        direction: Direction,
        entry_price: float,
        stop_loss: float,
        position_size: float,
        account_state: AccountState,
        current_positions: Optional[Dict[str, PositionState]] = None,
        take_profit: Optional[float] = None
    ) -> RiskAssessment:
        """Perform full risk assessment for a proposed trade.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction (LONG/SHORT)
            entry_price: Proposed entry price
            stop_loss: Stop loss price
            position_size: Position size
            account_state: Current account state
            current_positions: Currently open positions
            take_profit: Optional take profit price
            
        Returns:
            RiskAssessment with execution decision
        """
        checks: List[RiskCheck] = []
        warnings: List[str] = []
        
        # 1. System-level checks (guardrails)
        guardrail_result = self.guardrails.validate_trade(
            symbol=symbol,
            position_size=position_size,
            daily_pnl=account_state.daily_pnl,
            starting_equity=account_state.equity - account_state.daily_pnl,
            current_positions={
                k: v.to_dict() if hasattr(v, 'to_dict') else v 
                for k, v in (current_positions or {}).items()
            }
        )
        
        system_check = RiskCheck(
            check_type=RiskCheckType.SYSTEM,
            passed=guardrail_result.passed,
            message=guardrail_result.message,
            details=guardrail_result.details
        )
        checks.append(system_check)
        
        if not guardrail_result.passed:
            return RiskAssessment(
                can_execute=False,
                checks=checks,
                total_risk_pct=0.0,
                position_risk_pct=0.0,
                margin_required=0.0,
                blocked_reason=guardrail_result.message,
                warnings=warnings
            )
        
        # Collect warnings from guardrails
        if guardrail_result.status == GuardrailStatus.WARNING:
            warnings.append(f"System warning: {guardrail_result.message}")
        
        # 2. Position size risk check
        position_risk_pct = self.calculate_risk_percentage(
            entry_price=entry_price,
            stop_loss=stop_loss,
            direction=direction,
            position_size=position_size,
            equity=account_state.equity
        )
        
        position_check_passed = position_risk_pct <= self.max_total_risk_pct
        position_check = RiskCheck(
            check_type=RiskCheckType.POSITION_SIZE,
            passed=position_check_passed,
            message=(
                f"Position risk {position_risk_pct:.2%} "
                f"{'within' if position_check_passed else 'exceeds'} "
                f"limit {self.max_total_risk_pct:.2%}"
            ),
            details={
                'position_risk_pct': position_risk_pct,
                'max_risk_pct': self.max_total_risk_pct
            }
        )
        checks.append(position_check)
        
        if not position_check_passed:
            return RiskAssessment(
                can_execute=False,
                checks=checks,
                total_risk_pct=position_risk_pct,
                position_risk_pct=position_risk_pct,
                margin_required=entry_price * position_size,
                blocked_reason=f"Position risk {position_risk_pct:.2%} exceeds limit",
                warnings=warnings
            )
        
        # 3. Total account risk check
        total_open_risk = position_risk_pct
        if current_positions:
            for pos in current_positions.values():
                if pos.status == 'OPEN':
                    pos_risk = self.calculate_risk_percentage(
                        entry_price=pos.entry_price,
                        stop_loss=pos.stop_loss,
                        direction=pos.direction,
                        position_size=pos.position_size,
                        equity=account_state.equity
                    )
                    total_open_risk += pos_risk
        
        account_check_passed = total_open_risk <= self.max_account_risk_pct
        account_check = RiskCheck(
            check_type=RiskCheckType.CORRELATION,
            passed=account_check_passed,
            message=(
                f"Total account risk {total_open_risk:.2%} "
                f"{'within' if account_check_passed else 'exceeds'} "
                f"limit {self.max_account_risk_pct:.2%}"
            ),
            details={
                'total_risk_pct': total_open_risk,
                'max_account_risk_pct': self.max_account_risk_pct,
                'new_position_risk': position_risk_pct
            }
        )
        checks.append(account_check)
        
        if not account_check_passed:
            return RiskAssessment(
                can_execute=False,
                checks=checks,
                total_risk_pct=total_open_risk,
                position_risk_pct=position_risk_pct,
                margin_required=entry_price * position_size,
                blocked_reason=f"Total account risk {total_open_risk:.2%} exceeds limit",
                warnings=warnings
            )
        
        # 4. Risk/reward check if take profit provided
        if take_profit:
            if direction == Direction.LONG:
                reward = take_profit - entry_price
                risk = entry_price - stop_loss
            else:
                reward = entry_price - take_profit
                risk = stop_loss - entry_price
            
            if risk > 0:
                rr_ratio = reward / risk
                if rr_ratio < 1.0:
                    warnings.append(f"Risk/reward ratio {rr_ratio:.2f} < 1.0")
        
        # Calculate margin required
        margin_required = entry_price * position_size
        
        return RiskAssessment(
            can_execute=True,
            checks=checks,
            total_risk_pct=total_open_risk,
            position_risk_pct=position_risk_pct,
            margin_required=margin_required,
            warnings=warnings
        )
    
    def validate_signal_risk(
        self,
        signal: Any,  # EntrySignal
        account_state: AccountState,
        current_positions: Optional[Dict[str, PositionState]] = None
    ) -> RiskAssessment:
        """Validate an entry signal against risk rules.
        
        Args:
            signal: EntrySignal to validate
            account_state: Current account state
            current_positions: Current open positions
            
        Returns:
            RiskAssessment
        """
        return self.assess_trade(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            position_size=signal.position_size,
            account_state=account_state,
            current_positions=current_positions,
            take_profit=signal.tp1
        )
    
    def record_trade_execution(
        self,
        symbol: str,
        direction: Direction,
        position_size: float,
        risk_pct: float,
        entry_price: float
    ):
        """Record a trade execution for tracking.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction
            position_size: Position size
            risk_pct: Risk as percentage of equity
            entry_price: Entry price
        """
        # Check if we need to reset daily stats
        now = datetime.utcnow()
        if self._last_reset is None or now.date() != self._last_reset.date():
            self._daily_trades = []
            self._last_reset = now
        
        self._daily_trades.append({
            'timestamp': now.isoformat(),
            'symbol': symbol,
            'direction': direction.name,
            'position_size': position_size,
            'risk_pct': risk_pct,
            'entry_price': entry_price
        })
        
        logger.info(f"Recorded trade: {symbol} {direction.name} @ {entry_price}")
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily trading statistics.
        
        Returns:
            Dictionary with daily stats
        """
        now = datetime.utcnow()
        if self._last_reset is None or now.date() != self._last_reset.date():
            return {
                'trade_count': 0,
                'total_risk_pct': 0.0,
                'symbols_traded': [],
                'last_reset': None
            }
        
        total_risk = sum(t['risk_pct'] for t in self._daily_trades)
        symbols = list(set(t['symbol'] for t in self._daily_trades))
        
        return {
            'trade_count': len(self._daily_trades),
            'total_risk_pct': total_risk,
            'symbols_traded': symbols,
            'last_reset': self._last_reset.isoformat()
        }
    
    def get_system_controls(self) -> SystemControls:
        """Get current system controls from guardrails."""
        return self.guardrails.get_controls()
    
    def trigger_emergency_stop(self, reason: str) -> bool:
        """Trigger emergency stop via guardrails."""
        return self.guardrails.trigger_emergency_stop(reason)
