"""
Forex position sizer.

Hard rules (from ICT Hard Rules):
  - Never risk more than 2% of account on a single trade
  - Max 2 concurrent positions
  - Daily loss limit: 3%; weekly: 6%
  - Size scales with ICT checklist score (size_modifier from entry engine)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionSize:
    pair: str
    units: float            # units to trade (1 standard lot = 100,000 base)
    risk_pct: float         # actual risk as % of account
    risk_usd: float
    stop_distance: float    # in price terms
    entry: float
    stop: float
    t1: float
    t2: float
    t3: float
    size_modifier: float    # from ICT score (0.5, 0.75, 1.0)
    rejected: bool = False
    reject_reason: str = ''


class ForexPositionSizer:

    MAX_RISK_PCT = 0.02       # 2% hard max per trade
    DAILY_LIMIT_PCT = 0.03    # 3% daily stop
    WEEKLY_LIMIT_PCT = 0.06   # 6% weekly stop
    MAX_CONCURRENT = 2
    LOT_SIZE = 100_000        # 1 standard lot base currency units

    def __init__(self, account_balance: float = 10_000.0):
        self.account_balance = account_balance
        self._daily_loss = 0.0
        self._weekly_loss = 0.0
        self._open_trades = 0

    def size(
        self,
        pair: str,
        entry: float,
        stop: float,
        t1: float,
        t2: float,
        t3: float,
        size_modifier: float = 1.0,    # from ICT score
        pip_value_usd: Optional[float] = None,  # per unit per pip; None = auto-estimate
    ) -> PositionSize:

        # ── Guard rails ──────────────────────────────────────────────── #
        if self._open_trades >= self.MAX_CONCURRENT:
            return self._reject(pair, entry, stop, t1, t2, t3, size_modifier,
                                f'At max concurrent positions ({self.MAX_CONCURRENT})')

        if self._daily_loss >= self.DAILY_LIMIT_PCT * self.account_balance:
            return self._reject(pair, entry, stop, t1, t2, t3, size_modifier,
                                f'Daily loss limit hit ({self.DAILY_LIMIT_PCT:.0%})')

        if self._weekly_loss >= self.WEEKLY_LIMIT_PCT * self.account_balance:
            return self._reject(pair, entry, stop, t1, t2, t3, size_modifier,
                                f'Weekly loss limit hit ({self.WEEKLY_LIMIT_PCT:.0%})')

        stop_dist = abs(entry - stop)
        if stop_dist == 0:
            return self._reject(pair, entry, stop, t1, t2, t3, size_modifier,
                                'Zero stop distance')

        # ── Risk calculation ─────────────────────────────────────────── #
        risk_pct = self.MAX_RISK_PCT * size_modifier   # e.g. 2% × 0.5 = 1%
        risk_usd = risk_pct * self.account_balance

        # Units = risk_usd / stop_distance_in_price_per_unit
        # For USD-quoted pairs: stop_dist is already in USD per unit
        # For JPY-quoted: need to account for ~150 JPY/USD — use pip_value override
        if pip_value_usd is None:
            pip_value_usd = self._estimate_pip_value(pair, entry)

        # stop_dist is in native price terms; convert to USD per unit
        # For XXX/USD pairs: 1 unit of base moved stop_dist USD
        # For USD/XXX pairs: need to invert
        stop_dist_usd = stop_dist * pip_value_usd * self.LOT_SIZE / self._pip_size(pair)
        units_lots = risk_usd / stop_dist_usd if stop_dist_usd > 0 else 0
        units = units_lots * self.LOT_SIZE

        return PositionSize(
            pair=pair,
            units=round(units),
            risk_pct=round(risk_pct, 4),
            risk_usd=round(risk_usd, 2),
            stop_distance=round(stop_dist, 5),
            entry=entry,
            stop=stop,
            t1=t1,
            t2=t2,
            t3=t3,
            size_modifier=size_modifier,
        )

    def record_loss(self, usd_loss: float) -> None:
        self._daily_loss += usd_loss
        self._weekly_loss += usd_loss

    def open_trade(self) -> None:
        self._open_trades = min(self._open_trades + 1, self.MAX_CONCURRENT)

    def close_trade(self) -> None:
        self._open_trades = max(self._open_trades - 1, 0)

    def reset_daily(self) -> None:
        self._daily_loss = 0.0

    def reset_weekly(self) -> None:
        self._weekly_loss = 0.0
        self._daily_loss = 0.0

    # ── Helpers ───────────────────────────────────────────────────────── #

    @staticmethod
    def _pip_size(pair: str) -> float:
        """Return pip size in price terms."""
        if 'JPY' in pair:
            return 0.01
        return 0.0001

    @staticmethod
    def _estimate_pip_value(pair: str, price: float) -> float:
        """
        Rough USD pip value per lot.
        XXX/USD: 1 pip = $10 per standard lot (0.0001 * 100,000 = $10)
        USD/XXX: 1 pip ≈ $10 / price (approximate)
        XXX/YYY crosses: rough approximation via price
        """
        if pair.startswith('USD'):
            return 10.0 / price
        return 10.0  # default for USD-quoted pairs

    def _reject(
        self, pair, entry, stop, t1, t2, t3, size_modifier, reason
    ) -> PositionSize:
        return PositionSize(
            pair=pair, units=0, risk_pct=0, risk_usd=0,
            stop_distance=abs(entry - stop), entry=entry, stop=stop,
            t1=t1, t2=t2, t3=t3, size_modifier=size_modifier,
            rejected=True, reject_reason=reason,
        )
