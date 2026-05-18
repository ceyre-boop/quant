"""
Prop firm rules engine — exact simulation of EOD trailing drawdown structure.

Supports Lucid LucidFlex and MyFundedFutures Flex (same mechanics, different
default parameters). The EOD drawdown rule is the critical insight: intraday
moves don't affect the floor. Only the closing balance matters.

Usage:
    rules = PropFirmRules(account_size=100_000, profit_target=0.08, max_dd=0.08)
    rules.open_challenge()

    # Each trade:
    max_risk = rules.max_position_risk()
    rules.apply_trade_pnl(pnl_dollars)
    if rules.is_bust(): ...

    # Each market close:
    rules.update_eod()
    if rules.is_passed(): ...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional


# ── Preset configurations ─────────────────────────────────────────────────

LUCID_LUCID_FLEX = {
    "profit_target": 0.08,
    "max_dd": 0.08,
    "min_trading_days": 2,
    "daily_loss_limit": None,   # no separate daily limit
    "dd_trail_stops_at_starting": True,
}

MYFUNDEDFUTURES_FLEX = {
    "profit_target": 0.08,
    "max_dd": 0.10,
    "min_trading_days": 2,
    "daily_loss_limit": 0.04,  # 4% daily hard stop
    "dd_trail_stops_at_starting": True,
}


@dataclass
class TradeRecord:
    trade_num: int
    pair: str
    direction: str
    r_multiple: float
    risk_dollars: float
    pnl_dollars: float
    balance_before: float
    balance_after: float
    floor_at_time: float
    buffer_at_time: float
    blocked: bool = False
    size_reduced: bool = False
    original_risk_pct: float = 0.0
    actual_risk_pct: float = 0.0
    note: str = ""


@dataclass
class DayRecord:
    day_num: int
    date_str: str
    open_balance: float
    close_balance: float
    floor: float
    buffer: float
    buffer_pct_of_starting_dd: float
    trades_today: int
    status: str   # ON_TRACK / CAUTION / DANGER / PASSED / BUST


class PropFirmRules:
    """
    Exact simulation of Lucid LucidFlex / MyFundedFutures Flex prop firm rules.
    """

    def __init__(
        self,
        account_size: float = 100_000,
        profit_target: float = 0.08,
        max_dd: float = 0.08,
        min_trading_days: int = 2,
        daily_loss_limit: Optional[float] = None,
        dd_trail_stops_at_starting: bool = True,
        risk_per_trade_pct: float = 0.0075,
        max_risk_buffer_fraction: float = 0.25,
    ):
        self.account_size = account_size
        self.profit_target = profit_target
        self.max_dd = max_dd
        self.min_trading_days = min_trading_days
        self.daily_loss_limit = daily_loss_limit
        self.dd_trail_stops_at_starting = dd_trail_stops_at_starting
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_risk_buffer_fraction = max_risk_buffer_fraction

        # State — reset on open_challenge()
        self.balance: float = 0.0
        self.drawdown_floor: float = 0.0
        self.peak_eod_balance: float = 0.0
        self.trading_days: int = 0
        self.day_open_balance: float = 0.0
        self.trade_count: int = 0
        self.is_active: bool = False
        self.outcome: Optional[str] = None  # PASSED / BUST / TIMEOUT

        self.trade_log: List[TradeRecord] = []
        self.day_log: List[DayRecord] = []

    # ── Challenge lifecycle ───────────────────────────────────────────────

    def open_challenge(self) -> None:
        self.balance = float(self.account_size)
        self.drawdown_floor = self.account_size * (1 - self.max_dd)
        self.peak_eod_balance = float(self.account_size)
        self.trading_days = 0
        self.day_open_balance = float(self.account_size)
        self.trade_count = 0
        self.is_active = True
        self.outcome = None
        self.trade_log = []
        self.day_log = []

    def update_eod(self, day_str: str = "") -> DayRecord:
        """
        Call at market close every day. Updates the trailing floor.
        Floor only moves UP — never down. Stops trailing once account
        exceeds starting balance (at that point floor is locked).
        """
        if not self.is_active:
            raise RuntimeError("Challenge not active")

        if self.balance > self.peak_eod_balance:
            self.peak_eod_balance = self.balance

        # New floor candidate: peak_eod - max_drawdown_dollars
        max_dd_dollars = self.account_size * self.max_dd
        new_floor = self.peak_eod_balance - max_dd_dollars

        # Floor only moves up
        self.drawdown_floor = max(self.drawdown_floor, new_floor)

        # If dd_trail_stops_at_starting: once peak >= starting balance,
        # floor locks — no further tightening even if account keeps growing.
        # (Lucid / MFF both implement this — saves funded traders from
        # having the floor chase them to breakeven on huge wins.)
        if self.dd_trail_stops_at_starting and self.peak_eod_balance >= self.account_size:
            self.drawdown_floor = min(
                self.drawdown_floor,
                self.account_size - max_dd_dollars
            )

        # Count this as a trading day (even if no trades — firm counts calendar days)
        self.trading_days += 1

        buffer = self.balance - self.drawdown_floor
        buffer_pct = buffer / (self.account_size * self.max_dd)

        if buffer_pct >= 0.40:
            status = "ON_TRACK"
        elif buffer_pct >= 0.20:
            status = "CAUTION"
        else:
            status = "DANGER"

        if self.is_passed():
            status = "PASSED"
        if self.is_bust():
            status = "BUST"

        rec = DayRecord(
            day_num=self.trading_days,
            date_str=day_str,
            open_balance=self.day_open_balance,
            close_balance=self.balance,
            floor=self.drawdown_floor,
            buffer=buffer,
            buffer_pct_of_starting_dd=round(buffer_pct, 4),
            trades_today=sum(1 for t in self.trade_log if not hasattr(t, '_day_counted')),
            status=status,
        )
        self.day_log.append(rec)
        self.day_open_balance = self.balance
        return rec

    # ── Risk gating ───────────────────────────────────────────────────────

    def max_position_risk(self) -> float:
        """
        Maximum dollar risk for the next trade, given current floor proximity.
        Never risk more than max_risk_buffer_fraction of remaining drawdown buffer.
        """
        available_buffer = self.balance - self.drawdown_floor
        buffer_cap = available_buffer * self.max_risk_buffer_fraction
        default_risk = self.balance * self.risk_per_trade_pct
        return min(default_risk, buffer_cap)

    def apply_trade_pnl(
        self,
        r_multiple: float,
        pair: str = "",
        direction: str = "",
        requested_risk_pct: float = None,
    ) -> TradeRecord:
        """
        Execute a trade. Sizes down automatically if too close to floor.
        Returns a TradeRecord describing what happened.
        """
        if not self.is_active:
            raise RuntimeError("Challenge not active — call open_challenge() first")

        self.trade_count += 1
        requested_risk_pct = requested_risk_pct or self.risk_per_trade_pct
        default_risk_dollars = self.balance * requested_risk_pct
        max_risk = self.max_position_risk()

        blocked = False
        size_reduced = False
        actual_risk_dollars = default_risk_dollars

        if max_risk <= 0:
            blocked = True
            actual_risk_dollars = 0.0
            note = "BLOCKED: no buffer remaining"
        elif default_risk_dollars > max_risk:
            size_reduced = True
            actual_risk_dollars = max_risk
            note = f"SIZE_REDUCED: {default_risk_dollars:.0f} → {max_risk:.0f}"
        else:
            note = ""

        pnl = actual_risk_dollars * r_multiple if not blocked else 0.0
        balance_before = self.balance
        self.balance += pnl

        rec = TradeRecord(
            trade_num=self.trade_count,
            pair=pair,
            direction=direction,
            r_multiple=r_multiple,
            risk_dollars=actual_risk_dollars,
            pnl_dollars=pnl,
            balance_before=balance_before,
            balance_after=self.balance,
            floor_at_time=self.drawdown_floor,
            buffer_at_time=self.balance - self.drawdown_floor,
            blocked=blocked,
            size_reduced=size_reduced,
            original_risk_pct=requested_risk_pct,
            actual_risk_pct=actual_risk_dollars / self.balance if self.balance > 0 else 0,
            note=note,
        )
        self.trade_log.append(rec)

        # Check bust intraday (intraday moves do matter for bust — only
        # EOD floor UPDATE is based on closing balance, not the bust check)
        if self.is_bust():
            self.is_active = False
            self.outcome = "BUST"

        return rec

    # ── State checks ──────────────────────────────────────────────────────

    def is_bust(self) -> bool:
        return self.balance <= self.drawdown_floor

    def is_passed(self) -> bool:
        target = self.account_size * (1 + self.profit_target)
        return self.balance >= target and self.trading_days >= self.min_trading_days

    def buffer_pct(self) -> float:
        """Remaining DD buffer as fraction of starting max drawdown dollars."""
        starting_dd = self.account_size * self.max_dd
        return (self.balance - self.drawdown_floor) / starting_dd

    def summary(self) -> dict:
        trades_blocked  = sum(1 for t in self.trade_log if t.blocked)
        trades_reduced  = sum(1 for t in self.trade_log if t.size_reduced)
        return {
            "outcome": self.outcome or ("PASSED" if self.is_passed() else "IN_PROGRESS"),
            "balance": round(self.balance, 2),
            "drawdown_floor": round(self.drawdown_floor, 2),
            "buffer": round(self.balance - self.drawdown_floor, 2),
            "buffer_pct_of_max_dd": round(self.buffer_pct(), 4),
            "trading_days": self.trading_days,
            "trade_count": self.trade_count,
            "trades_blocked": trades_blocked,
            "trades_reduced": trades_reduced,
            "blocked_pct": round(trades_blocked / max(self.trade_count, 1), 4),
            "total_pnl": round(self.balance - self.account_size, 2),
            "return_pct": round((self.balance / self.account_size - 1) * 100, 2),
        }

    @classmethod
    def lucid(cls, account_size: float = 100_000, **kwargs) -> "PropFirmRules":
        return cls(account_size=account_size, **{**LUCID_LUCID_FLEX, **kwargs})

    @classmethod
    def mff(cls, account_size: float = 100_000, **kwargs) -> "PropFirmRules":
        return cls(account_size=account_size, **{**MYFUNDEDFUTURES_FLEX, **kwargs})
