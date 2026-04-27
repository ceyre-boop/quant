"""
Shared forex signal generation for live scan and backtest.

Three layers, merged in priority order (highest priority wins):
  1. CB event trigger   — post-decision drift, 10–20 day hold
  2. Calendar signals   — quarter-end rebalancing, March JPY, seasonal
  3. Macro (IRP + RRD)  — monthly, 60 day hold

The goal is one canonical signal definition consumed by both live scan
and historical backtest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from sovereign.forex.data_fetcher import FALLBACK_CPI, FALLBACK_RATES
from sovereign.forex.calendar_signals import CalendarSignalEngine


@dataclass(frozen=True)
class SignalConfig:
    hold_days: int = 60
    signal_threshold: float = 0.15
    cb_entry_window_days: int = 5
    cb_surprise_threshold: int = 20


class ForexSignalEngine:
    def __init__(self, fetcher, cb_trigger, config: Optional[SignalConfig] = None):
        self._fetcher = fetcher
        self._cb = cb_trigger
        self._calendar = CalendarSignalEngine()
        self.config = config or SignalConfig()

    def build_signal_frame(
        self,
        prices: pd.DataFrame,
        base_country: str,
        quote_country: str,
        start: str,
        end: str,
        pair: str = '',
    ) -> pd.DataFrame:
        close = prices['Close'] if 'Close' in prices.columns else prices.iloc[:, 0]
        signals, hold_days = self.build_signal_arrays(
            close=close,
            base_country=base_country,
            quote_country=quote_country,
            start=start,
            end=end,
            pair=pair,
        )
        sig_df = pd.DataFrame(
            {
                'signal': signals.astype(float, copy=False),
                'hold_days': hold_days.astype(int, copy=False),
            },
            index=close.index,
        )
        return sig_df

    def build_signal_arrays(
        self,
        close: pd.Series,
        base_country: str,
        quote_country: str,
        start: str,
        end: str,
        pair: str = '',
    ) -> tuple[np.ndarray, np.ndarray]:
        all_dates = pd.DatetimeIndex(close.index)
        signals = np.zeros(len(all_dates), dtype=np.int8)
        hold_days = np.full(len(all_dates), self.config.hold_days, dtype=np.int32)

        # ── Layer 3: Macro (lowest priority, 60-day hold) ─────────────── #
        base_rates  = self._fetcher.get_rate_history(base_country, start='2014-01-01')
        quote_rates = self._fetcher.get_rate_history(quote_country, start='2014-01-01')
        base_cpi_h  = self._fetcher.get_cpi_history(base_country, start='2014-01-01')
        quote_cpi_h = self._fetcher.get_cpi_history(quote_country, start='2014-01-01')

        for date in close.resample('BMS').first().index:
            macro_sign = self._macro_signal_for_date(
                close=close, date=date,
                base_country=base_country, quote_country=quote_country,
                base_rates=base_rates, quote_rates=quote_rates,
                base_cpi_h=base_cpi_h, quote_cpi_h=quote_cpi_h,
            )
            if macro_sign != 0:
                idx = all_dates.get_indexer([date])[0]
                if idx >= 0:
                    signals[idx] = np.int8(macro_sign)
                    hold_days[idx] = np.int32(self.config.hold_days)

        # ── Layer 2: Calendar signals (quarter-end, seasonal) ─────────── #
        if pair:
            cal_events = self._calendar.get_historical_signals(
                pair=pair,
                price_history=close,
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
            )
            for ev in cal_events:
                idx = int(all_dates.searchsorted(ev['signal_date'], side='left'))
                if idx >= len(all_dates):
                    continue
                signal_val = np.int8(1 if ev['direction'] == 'LONG' else -1)
                ev_hold = np.int32(ev['hold_days'])
                existing = signals[idx]
                # Calendar overrides macro; CB event will override calendar below
                if existing == 0 or existing == signal_val:
                    signals[idx] = signal_val
                    hold_days[idx] = ev_hold

        # ── Layer 1: CB event trigger (highest priority, 10-20 day hold) ─ #
        cb_events = self._cb.check_historical(
            base_country=base_country,
            quote_country=quote_country,
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
            min_surprise_bps=self.config.cb_surprise_threshold,
        )
        for ev in cb_events:
            entry_start = ev['entry_start']
            entry_end   = ev['entry_end']
            signal_val  = np.int8(1 if ev['direction'] == 'LONG' else -1)
            ev_hold     = np.int32(ev['hold_days'])
            start_idx = int(all_dates.searchsorted(entry_start, side='left'))
            end_idx   = int(all_dates.searchsorted(entry_end,   side='right'))
            if start_idx >= len(all_dates) or start_idx >= end_idx:
                continue
            force_overlay = abs(ev['surprise_bps']) >= 50
            for idx in range(start_idx, min(end_idx, len(all_dates))):
                existing = signals[idx]
                if existing == 0 or existing == signal_val or force_overlay:
                    signals[idx] = signal_val
                    hold_days[idx] = ev_hold

        return signals, hold_days

    def _macro_signal_for_date(
        self,
        close: pd.Series,
        date: pd.Timestamp,
        base_country: str,
        quote_country: str,
        base_rates: pd.Series,
        quote_rates: pd.Series,
        base_cpi_h: pd.Series,
        quote_cpi_h: pd.Series,
    ) -> int:
        spot = float(close.asof(date))
        hist = close.loc[:date]

        b_rate = float(base_rates.asof(date)) if len(base_rates) and date >= base_rates.index[0] else FALLBACK_RATES.get(base_country, 2.0)
        q_rate = float(quote_rates.asof(date)) if len(quote_rates) and date >= quote_rates.index[0] else FALLBACK_RATES.get(quote_country, 2.0)
        b_cpi = float(base_cpi_h.asof(date)) if len(base_cpi_h) and date >= base_cpi_h.index[0] else FALLBACK_CPI.get(base_country, 2.0)
        q_cpi = float(quote_cpi_h.asof(date)) if len(quote_cpi_h) and date >= quote_cpi_h.index[0] else FALLBACK_CPI.get(quote_country, 2.0)

        real_rate_diff = (b_rate - b_cpi) - (q_rate - q_cpi)
        irp_fv = spot * (1 + q_rate / 100) / (1 + b_rate / 100)
        irp_dev = (spot - irp_fv) / irp_fv if irp_fv != 0 else 0.0
        irp_z = (irp_dev / (hist.pct_change().std() * np.sqrt(252) + 1e-8)
                 if len(hist) > 252 else 0.0)

        macro_score = (
            0.50 * np.clip(-irp_z / 1.5, -1, 1) +
            0.50 * np.clip(real_rate_diff / 4.0, -1, 1)
        )

        mom_sign = 0
        if len(hist) > 63:
            mom = float(hist.iloc[-1] / hist.iloc[-63] - 1)
            mom_sign = int(np.sign(mom)) if abs(mom) > 0.005 else 0

        macro_sign = int(np.sign(macro_score)) if abs(macro_score) > self.config.signal_threshold else 0
        if macro_sign != 0 and (mom_sign == 0 or mom_sign == macro_sign):
            return macro_sign
        return 0
