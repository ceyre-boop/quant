"""
Calendar-based signal layer — pure date math, no network required.

Edge 6 — Quarter-End Rebalancing (55–58% win rate, 5–10 day hold):
  Mechanism: Corporates hedge prior-quarter FX gains at QE.
             Fade the prior quarter's trend in the first 5 days of
             the new quarter.
  Signal:    Opposite to prior-quarter pair return.

Seasonal — March JPY Repatriation (>65% historical hit rate):
  Mechanism: Japanese fiscal year ends March 31. Corporations and
             pension funds repatriate foreign profits → buy JPY.
  Signal:    SHORT JPY pairs (USDJPY, EURJPY, GBPJPY, AUDJPY, NZDJPY)
             during March 10–31.
  Exit:      April 5 (repatriation flow exhausted after FY close).

Seasonal — January USD Weakness:
  Mechanism: Foreign investors deploy new mandates, buying non-USD.
  Signal:    Mild SHORT USD pairs (EURUSD, GBPUSD, AUDUSD) in Jan 1–15.

Seasonal — August Low-Liquidity Warning:
  No new positions in August 1–31 — flash crash risk elevated.
  Returns size_modifier=0.5 for any pair during this window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# Quarter-end window: last N trading days of quarter + first M of new quarter
QE_FADE_LOOKBACK_DAYS = 63   # prior quarter = ~63 trading days
QE_ENTRY_WINDOW = 5          # enter day 1–5 of new quarter
QE_HOLD_DAYS = 8
QE_MIN_MOVE_PCT = 0.04       # prior quarter must have moved >4% to avoid noise

# JPY pairs affected by March repatriation (SHORT these = long JPY)
JPY_PAIRS_REPATRIATION = {'USDJPY=X', 'EURJPY=X', 'GBPJPY=X', 'AUDNZD=X'}
MARCH_REPAT_START_DAY = 10   # March 10
MARCH_REPAT_END_DAY   = 31
MARCH_REPAT_HOLD_DAYS = 15
MARCH_REPAT_CONVICTION = 0.60

# August warning
AUGUST_SIZE_PENALTY = 0.5


@dataclass
class CalendarSignal:
    source: str           # 'quarter_end' / 'march_jpy' / 'january_usd' / 'august_warning'
    direction: str        # 'LONG' / 'SHORT' / 'REDUCE'
    conviction: float     # 0..1
    hold_days: int
    note: str


class CalendarSignalEngine:

    def get_signal(
        self,
        pair: str,
        as_of: pd.Timestamp,
        price_history: Optional[pd.Series] = None,
    ) -> Optional[CalendarSignal]:
        """
        Returns the strongest active calendar signal for this pair on as_of,
        or None if no calendar edge is active.
        """
        month = as_of.month
        day   = as_of.day

        # ── August warning ─────────────────────────────────────────────── #
        if month == 8:
            return CalendarSignal(
                source='august_warning',
                direction='REDUCE',
                conviction=0.0,
                hold_days=0,
                note='August: low liquidity, reduce size 50%',
            )

        # ── March JPY repatriation ─────────────────────────────────────── #
        if month == 3 and MARCH_REPAT_START_DAY <= day <= MARCH_REPAT_END_DAY:
            if pair in JPY_PAIRS_REPATRIATION:
                return CalendarSignal(
                    source='march_jpy',
                    direction='SHORT',    # short JPY pair = long JPY
                    conviction=MARCH_REPAT_CONVICTION,
                    hold_days=MARCH_REPAT_HOLD_DAYS,
                    note='March JPY fiscal year-end repatriation flow',
                )

        # ── Quarter-end rebalancing ────────────────────────────────────── #
        qe_signal = self._quarter_end_signal(pair, as_of, price_history)
        if qe_signal:
            return qe_signal

        return None

    def get_historical_signals(
        self,
        pair: str,
        price_history: pd.Series,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[dict]:
        """
        Returns one entry per distinct calendar event (not per day).
        Used by the backtester.

        Each calendar event fires exactly once on its first qualifying date —
        not on every day within the window.
        """
        results = []
        dates = pd.date_range(start, end, freq='B')

        seen_events: set[str] = set()  # (source, year, month) dedup key

        for d in dates:
            sig = self.get_signal(pair, d, price_history)
            if sig is None or sig.direction == 'REDUCE':
                continue

            # One event per (source, year, month-window)
            if sig.source == 'quarter_end':
                key = f'qe_{d.year}_{d.quarter}'
            elif sig.source == 'march_jpy':
                key = f'march_jpy_{d.year}'
            elif sig.source == 'january_usd':
                key = f'jan_usd_{d.year}'
            else:
                key = f'{sig.source}_{d.year}_{d.month}'

            if key in seen_events:
                continue
            seen_events.add(key)

            results.append({
                'signal_date':  d,
                'direction':    sig.direction,
                'conviction':   sig.conviction,
                'hold_days':    sig.hold_days,
                'source':       sig.source,
            })

        return results

    # ── Quarter-end internals ─────────────────────────────────────────── #

    def _quarter_end_signal(
        self,
        pair: str,
        as_of: pd.Timestamp,
        prices: Optional[pd.Series],
    ) -> Optional[CalendarSignal]:
        """
        Fires in the first QE_ENTRY_WINDOW business days of a new quarter.
        Direction = FADE of prior quarter's move.
        Requires price history to measure prior quarter return.
        """
        if prices is None or len(prices) < QE_FADE_LOOKBACK_DAYS + 5:
            return None

        # Are we in the entry window? (first 5 bdays of Jan/Apr/Jul/Oct)
        quarter_start = self._quarter_start(as_of)
        if quarter_start is None:
            return None

        bdays_into_quarter = self._bdays_since(quarter_start, as_of)
        if not (1 <= bdays_into_quarter <= QE_ENTRY_WINDOW):
            return None

        # Prior quarter return on this pair
        prices_aligned = prices[prices.index <= as_of]
        if len(prices_aligned) < QE_FADE_LOOKBACK_DAYS:
            return None

        prior_qtr_return = float(
            prices_aligned.iloc[-1] / prices_aligned.iloc[-QE_FADE_LOOKBACK_DAYS] - 1
        )
        if abs(prior_qtr_return) < QE_MIN_MOVE_PCT:
            return None  # move too small to bother fading

        # Fade direction: prior quarter up → sell; prior quarter down → buy
        fade_direction = 'SHORT' if prior_qtr_return > 0 else 'LONG'
        conviction = min(0.40 + abs(prior_qtr_return) * 2, 0.65)

        return CalendarSignal(
            source='quarter_end',
            direction=fade_direction,
            conviction=round(conviction, 3),
            hold_days=QE_HOLD_DAYS,
            note=f'QE rebalancing: prior qtr {prior_qtr_return:+.1%}, fade {fade_direction}',
        )

    @staticmethod
    def _quarter_start(dt: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Return the first calendar day of the current quarter, or None if not in entry month."""
        qstart_months = {1: 1, 4: 4, 7: 7, 10: 10}
        if dt.month not in qstart_months:
            return None
        return pd.Timestamp(dt.year, dt.month, 1)

    @staticmethod
    def _bdays_since(start: pd.Timestamp, end: pd.Timestamp) -> int:
        """Business days between start and end (inclusive start, exclusive end)."""
        return int(np.busday_count(
            start.date(),
            end.date(),
        ))
