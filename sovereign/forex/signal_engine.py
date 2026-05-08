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

from sovereign.forex.data_fetcher import FALLBACK_CPI, FALLBACK_RATES, ForexDataFetcher
from sovereign.forex.calendar_signals import CalendarSignalEngine
from sovereign.forex.cpi_engine import CPISurpriseEngine
from sovereign.forex.entry_engine import CBEventTrigger

HOLD_DAYS = 60
SIGNAL_THRESHOLD = 0.15
CB_SURPRISE_THRESHOLD = 20
DONCHIAN_FAST_ENTRY_DAYS = 20
DONCHIAN_SLOW_ENTRY_DAYS = 55


@dataclass(frozen=True)
class SignalConfig:
    hold_days: int = HOLD_DAYS
    signal_threshold: float = SIGNAL_THRESHOLD
    cb_entry_window_days: int = 5
    cb_surprise_threshold: int = CB_SURPRISE_THRESHOLD
    strict_mode: bool = False
    use_macro_overlay: bool = False
    donchian_fast_entry_days: int = DONCHIAN_FAST_ENTRY_DAYS
    donchian_slow_entry_days: int = DONCHIAN_SLOW_ENTRY_DAYS


class ForexSignalEngine:
    def __init__(self, fetcher, cb_trigger, config: Optional[SignalConfig] = None):
        self._fetcher = fetcher
        self._cb = cb_trigger
        self._calendar = CalendarSignalEngine()
        self._cpi = CPISurpriseEngine(fetcher=fetcher)
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
            prices_df=prices,
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
        prices_df: pd.DataFrame = None,   # full OHLCV for ATR computation
    ) -> tuple[np.ndarray, np.ndarray]:
        all_dates = pd.DatetimeIndex(close.index)
        signals = np.zeros(len(all_dates), dtype=np.int8)
        hold_days = np.full(len(all_dates), self.config.hold_days, dtype=np.int32)

        # ── Pre-compute rolling ATR for the ATR filter ────────────────── #
        # Trajectory model: atr_pct < 2.2% = worst R outcomes (77% importance).
        atr_series = self._compute_atr_pct(close, prices_df)

        # ── Pre-fetch BOJ rate history for JPY gate ───────────────────── #
        boj_rates = None
        if 'JPY' in pair:
            boj_rates = self._fetcher.get_rate_history('JP', start='2014-01-01')

        # ── Layer 3: Macro (lowest priority, 60-day hold) ─────────────── #
        base_rates  = self._fetcher.get_rate_history(base_country, start='2014-01-01')
        quote_rates = self._fetcher.get_rate_history(quote_country, start='2014-01-01')
        base_cpi_h  = self._fetcher.get_cpi_history(base_country, start='2014-01-01')
        quote_cpi_h = self._fetcher.get_cpi_history(quote_country, start='2014-01-01')

        for date in close.resample('BMS').first().index:
            # ── ATR filter ────────────────────────────────────────────── #
            if atr_series is not None and date in atr_series.index:
                atr_pct_now = float(atr_series.asof(date))
                if atr_pct_now < 0.022:   # 2.2% median from attribution training
                    continue

            # ── USDCHF: only trade when CHF safe-haven flows are active ── #
            # SNB suppression dominates in low-VIX risk-on environments.
            # Use the macro_engine RiskSentimentEngine result via a proxy:
            # approximate VIX from VXX/SPY returns (already in spy_5d_return).
            # Simpler: skip USDCHF when the CHF rate is moving (SNB signalling).
            # Best available signal in the data: USDCHF has near-zero real rate
            # differential. Gate it on CHF rate movement (same as BOJ gate).
            if pair == 'USDCHF=X':
                ch_rates = self._fetcher.get_rate_history('CH', start='2014-01-01')
                if ch_rates is not None and len(ch_rates) >= 63:
                    ch_recent = ch_rates.loc[:date].iloc[-63:] if len(ch_rates.loc[:date]) >= 63 else None
                    if ch_recent is not None:
                        ch_change = abs(float(ch_recent.iloc[-1]) - float(ch_recent.iloc[0]))
                        if ch_change < 0.10:   # SNB hasn't moved in 3 months
                            continue

            # ── EURJPY/JPY: BOJ activity gate ────────────────────────── #
            if 'JPY' in pair and boj_rates is not None and len(boj_rates) >= 90:
                boj_hist = boj_rates.loc[:date]
                if len(boj_hist) >= 90:
                    boj_change_90d = abs(float(boj_hist.iloc[-1]) - float(boj_hist.iloc[-90]))
                    if boj_change_90d < 0.001:   # BOJ frozen — no edge
                        continue

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

        # ── Layer 1.5: CPI surprise fade (Edge 2, 5-day hold) ────────────── #
        for country in (base_country, quote_country):
            cpi_events = self._cpi.get_historical_surprises(
                country=country,
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
            )
            for ev in cpi_events:
                # Fade direction is relative to the currency in the pair
                if country == base_country:
                    signal_val = np.int8(1 if ev['direction'] == 'LONG' else -1)
                else:
                    # Quote currency: flip direction
                    signal_val = np.int8(-1 if ev['direction'] == 'LONG' else 1)
                ev_hold = np.int32(ev['hold_days'])
                idx = int(all_dates.searchsorted(ev['signal_date'], side='left'))
                if idx >= len(all_dates):
                    continue
                existing = signals[idx]
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

        if self.config.strict_mode:
            donchian_signals = self._build_donchian_signals(
                close=close,
                fast_days=self.config.donchian_fast_entry_days,
                slow_days=self.config.donchian_slow_entry_days,
            )
            if self.config.use_macro_overlay:
                # In strict mode, conflicting macro direction cancels the Donchian entry.
                overlay_ok = (signals == 0) | (signals == donchian_signals)
                signals = np.where(overlay_ok, donchian_signals, 0).astype(np.int8, copy=False)
            else:
                signals = donchian_signals.astype(np.int8, copy=False)
            hold_days = np.full(len(all_dates), self.config.hold_days, dtype=np.int32)

        return signals, hold_days

    @staticmethod
    def _build_donchian_signals(
        close: pd.Series,
        fast_days: int,
        slow_days: int,
    ) -> np.ndarray:
        """Donchian breakout entries: long on fast/slow highs, short on fast/slow lows."""
        if close.empty:
            return np.zeros(0, dtype=np.int8)
        fast_high = close.rolling(fast_days).max().shift(1)
        fast_low = close.rolling(fast_days).min().shift(1)
        slow_high = close.rolling(slow_days).max().shift(1)
        slow_low = close.rolling(slow_days).min().shift(1)

        long_break = (close > fast_high) | (close > slow_high)
        short_break = (close < fast_low) | (close < slow_low)

        signals = np.zeros(len(close), dtype=np.int8)
        signals[(long_break & ~short_break).fillna(False).to_numpy()] = 1
        signals[(short_break & ~long_break).fillna(False).to_numpy()] = -1
        return signals

    @staticmethod
    def _compute_atr_pct(
        close: pd.Series,
        prices_df: pd.DataFrame = None,
        period: int = 14,
    ) -> pd.Series:
        """Rolling ATR as % of price. Uses True Range if OHLCV available."""
        try:
            if prices_df is not None and 'High' in prices_df.columns:
                h = prices_df['High']
                l = prices_df['Low']
                c = prices_df['Close'] if 'Close' in prices_df.columns else close
                tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
            else:
                # Fallback: use close-to-close range as ATR proxy
                tr = close.diff().abs()
            atr = tr.rolling(period).mean()
            atr_pct = atr / close
            return atr_pct.fillna(method='bfill')
        except Exception:
            return None

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


def build_signal_arrays(
    pair: str,
    prices: pd.DataFrame,
    base_country: str,
    quote_country: str,
    *,
    fetcher=None,
    cb_trigger=None,
    config: Optional[SignalConfig] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    fetcher = fetcher or ForexDataFetcher()
    cb_trigger = cb_trigger or CBEventTrigger()
    engine = ForexSignalEngine(fetcher=fetcher, cb_trigger=cb_trigger, config=config)
    if start is None:
        start = str(prices.index.min().date()) if len(prices.index) else '2015-01-01'
    if end is None:
        end = str(prices.index.max().date()) if len(prices.index) else '2015-01-01'
    close = prices['Close'] if 'Close' in prices.columns else prices.iloc[:, 0]
    return engine.build_signal_arrays(
        close=close,
        base_country=base_country,
        quote_country=quote_country,
        start=start,
        end=end,
        pair=pair,
        prices_df=prices,
    )


def build_signal_frame(
    pair: str,
    prices: pd.DataFrame,
    base_country: str,
    quote_country: str,
    *,
    fetcher=None,
    cb_trigger=None,
    config: Optional[SignalConfig] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    signal_arr, hold_arr = build_signal_arrays(
        pair=pair,
        prices=prices,
        base_country=base_country,
        quote_country=quote_country,
        fetcher=fetcher,
        cb_trigger=cb_trigger,
        config=config,
        start=start,
        end=end,
    )
    return pd.DataFrame(
        {
            'signal': signal_arr.astype(float, copy=False),
            'hold': hold_arr.astype(int, copy=False),
        },
        index=prices.index,
    )
