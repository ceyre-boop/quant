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
    # Factor weights for the backtest macro signal (parameterized for seed-library
    # factor replication). DEFAULTS = the historical hardcoded literals → zero behavior
    # change. A pure-carry seed sets irp_weight=0, rate_weight=1, use_momentum_filter=False.
    irp_weight: float = 0.50            # IRP mean-reversion (value-ish) component
    rate_weight: float = 0.50           # real rate differential (carry) component
    use_momentum_filter: bool = True    # 63-day momentum confirmation gate


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
        # Latent feature sizing modifiers (Session 3 findings, 2026-05-19):
        # Counter-momentum entries produce 3× better avgR (0.331 vs 0.107).
        # VIX slope [0,1) is best environment; >3 reduces follow-through.
        sig_df['size_mult'] = self._compute_size_multipliers(
            close=close, signals=sig_df['signal'], prices_df=prices, pair=pair
        )

        # ── Bull+VIX regime gate (v011, 2026-05-22) ──────────────────────
        # Universal finding: macro rate differential signals degrade in bull market
        # + elevated VIX as fear flows compete with rate signals.
        # Tiered thresholds: JPY/cross pairs VIX>15, USD macro pairs VIX>20.
        _VIX_GATES = {
            # HYP-044 rolled back — VIX 13→15 for USDJPY/AUDNZD: delta 0.000 OOS,
            # confirmed in-sample noise (holdout 2026-05-31).
            'USDJPY=X': 15.0, 'AUDNZD=X': 15.0,
            'EURUSD=X': 18.0, 'GBPUSD=X': 18.0, 'AUDUSD=X': 20.0,
        }
        if pair in _VIX_GATES:
            sig_df = self._apply_vix_regime_gate(sig_df, close.index, _VIX_GATES[pair])

        # ── WTI term structure gate for USDCAD (HYP-030, 2026-05-23) ─────
        # Confirmed: WTI term structure removes ~71% of USDCAD signals,
        # WR lifts 41.8%→51.2%, Sharpe 0.326→0.701 (+0.375).
        # Gate: zero USDCAD signals when WTI is in contango (slope_20 < slope_60)
        # AND OVX (oil vol) outside fear zone [15, 40].
        # Rationale: CAD is oil-linked; only trade when oil curve confirms direction.
        if pair == 'USDCAD=X':
            sig_df = self._apply_wti_term_structure_gate(sig_df, close.index)

        return sig_df

    def _apply_vix_regime_gate(
        self, sig_df: 'pd.DataFrame', date_index: 'pd.DatetimeIndex', vix_threshold: float
    ) -> 'pd.DataFrame':
        """Zero signals when SPY > 200 SMA AND VIX > vix_threshold."""
        try:
            import yfinance as yf
            start = str(date_index[0].date())
            end   = str(date_index[-1].date())
            spy = yf.download('SPY', start=start, end=end, progress=False)
            vix = yf.download('^VIX', start=start, end=end, progress=False)
            for df_ in (spy, vix):
                if hasattr(df_.columns, 'get_level_values'):
                    df_.columns = df_.columns.get_level_values(0)
                df_.index = pd.to_datetime(df_.index).tz_localize(None)
            spy['sma200'] = spy['Close'].rolling(200).mean()
            spy['is_bull'] = spy['Close'] > spy['sma200']
            sig_df = sig_df.copy()
            for date in sig_df[sig_df['signal'] != 0].index:
                try:
                    if bool(spy['is_bull'].asof(date)) and float(vix['Close'].asof(date)) > vix_threshold:
                        sig_df.loc[date, 'signal'] = 0.0
                except Exception:
                    pass
        except Exception:
            pass
        return sig_df

    def _apply_wti_term_structure_gate(
        self, sig_df: 'pd.DataFrame', date_index: 'pd.DatetimeIndex'
    ) -> 'pd.DataFrame':
        """
        HYP-030: Zero USDCAD signals when WTI term structure is NOT confirming.
        Confirmed: WR 41.8%→51.2%, Sharpe +0.375 (2026-05-23).

        Gate fires (zeroes signal) when EITHER:
          - WTI slope_20 >= slope_60 (backwardation: supply fear, bad for CAD longs)
          - OVX outside [15, 40] fear zone (extreme calm or panic — edge disappears)
        """
        try:
            import yfinance as yf
            start = str(date_index[0].date())
            end   = str(date_index[-1].date())
            # CL=F is WTI spot; we use it as a proxy for term structure slope
            # Slope approximation: 20d momentum vs 60d momentum of WTI
            wti = yf.download('CL=F', start=start, end=end, progress=False)
            ovx = yf.download('^OVX', start=start, end=end, progress=False)
            for df_ in (wti, ovx):
                if isinstance(df_.columns, pd.MultiIndex):
                    df_.columns = df_.columns.get_level_values(0)
                df_.index = pd.to_datetime(df_.index).tz_localize(None)

            wti_close = wti['Close'] if 'Close' in wti.columns else wti.iloc[:, 0]
            ovx_close = ovx['Close'] if 'Close' in ovx.columns else ovx.iloc[:, 0]

            for date in sig_df.index:
                if sig_df.loc[date, 'signal'] == 0:
                    continue
                try:
                    wti_hist = wti_close.loc[:date].dropna().tail(65)
                    ovx_val  = float(ovx_close.loc[:date].dropna().tail(1).iloc[-1])
                    if len(wti_hist) < 60:
                        continue
                    slope_20 = float(wti_hist.iloc[-1] / wti_hist.iloc[-20] - 1)
                    slope_60 = float(wti_hist.iloc[-1] / wti_hist.iloc[-60] - 1)
                    # Gate: backwardation (short faster than long) OR OVX out of fear zone
                    if slope_20 >= slope_60 or not (15 <= ovx_val <= 40):
                        sig_df.loc[date, 'signal'] = 0
                        sig_df.loc[date, 'size_mult'] = 0.0
                except Exception:
                    pass
        except Exception:
            pass
        return sig_df

    def _compute_size_multipliers(
        self,
        close: pd.Series,
        signals: pd.Series,
        prices_df: pd.DataFrame,
        pair: str = '',
    ) -> pd.Series:
        """
        Size modifiers based on latent feature search (Session 3).
        Counter-momentum (5d): 1.25× | Flat: 1.0× | Aligned: 0.75×
        VIX slope > 3 (extreme contango): additional 0.85× discount
        Combined effect: pullback entry in calm VIX regime gets 1.25×,
                         chasing entry in high-VIX-contango gets 0.64×.
        """
        size_mult = pd.Series(1.0, index=close.index)
        sig_fired = signals[signals != 0].index

        if len(sig_fired) == 0:
            return size_mult

        # Pre-load VIX data (once)
        vix_df = None
        try:
            import yfinance as yf
            vix_df = yf.download('^VIX', start=str(close.index[0].date()),
                                  end=str(close.index[-1].date()), progress=False)
            vix3m_df = yf.download('^VIX3M', start=str(close.index[0].date()),
                                    end=str(close.index[-1].date()), progress=False)
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = vix_df.columns.get_level_values(0)
            if isinstance(vix3m_df.columns, pd.MultiIndex):
                vix3m_df.columns = vix3m_df.columns.get_level_values(0)
            vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None)
            vix3m_df.index = pd.to_datetime(vix3m_df.index).tz_localize(None)
        except Exception:
            vix_df = None
            vix3m_df = None

        for date in sig_fired:
            sig_dir = int(signals.loc[date])
            mult = 1.0

            # 5-day momentum modifier
            hist = close.loc[:date].tail(8)
            if len(hist) >= 6:
                mom_5d = float(hist.iloc[-1] / hist.iloc[-6] - 1) * sig_dir
                if mom_5d < -0.002:   # counter-momentum (pullback entry)
                    mult *= 1.25
                elif mom_5d > 0.002:  # aligned (chasing)
                    mult *= 0.75

            # VIX slope modifier
            if vix_df is not None and vix3m_df is not None:
                try:
                    v   = float(vix_df.loc[:date].tail(1)['Close'].iloc[-1]
                                if 'Close' in vix_df.columns
                                else vix_df.loc[:date].tail(1).iloc[-1, 0])
                    v3  = float(vix3m_df.loc[:date].tail(1)['Close'].iloc[-1]
                                if 'Close' in vix3m_df.columns
                                else vix3m_df.loc[:date].tail(1).iloc[-1, 0])
                    slope = v3 - v
                    if slope > 3.0:    # extreme contango — poor follow-through
                        mult *= 0.85
                    elif slope < 0.0:  # inverted — fear regime, tighten
                        mult *= 0.90
                except Exception:
                    pass

            size_mult.loc[date] = round(mult, 4)

        # ── HYP-028: US10Y divergence boost for EUR/USD (2026-05-22) ────────
        # When US10Y rises >20bps in 10 days AND EUR/USD hasn't responded yet,
        # the spot price historically follows: 70% hit rate n=125 p<0.0001.
        # IC = 0.086 (below 0.15 standalone gate) → deploy as 1.25× size boost
        # only when existing macro signal already aligned with rate move.
        if pair == 'EURUSD=X':
            try:
                us10y_df = yf.download('^TNX', start=str(close.index[0].date()),
                                       end=str(close.index[-1].date()), progress=False)
                if hasattr(us10y_df.columns, 'get_level_values'):
                    us10y_df.columns = us10y_df.columns.get_level_values(0)
                us10y_df.index = pd.to_datetime(us10y_df.index).tz_localize(None)
                for date in sig_fired:
                    if signals.loc[date] == 0:
                        continue
                    try:
                        rate_now = float(us10y_df['Close'].asof(date))
                        rate_10d_ago = float(us10y_df['Close'].loc[:date].iloc[-11])
                        eur_now  = float(close.loc[date])
                        eur_10d  = float(close.loc[:date].iloc[-11])
                        rate_chg_bps = (rate_now - rate_10d_ago) * 100
                        eur_chg_pct  = (eur_now / eur_10d - 1) * 100
                        # Signal long: rate fell (bullish EUR) but EUR didn't rise yet
                        # Signal short: rate rose (bearish EUR) but EUR didn't fall yet
                        sig_dir = int(signals.loc[date])
                        is_divergence = (
                            (sig_dir > 0 and rate_chg_bps < -20 and eur_chg_pct < 0.5) or
                            (sig_dir < 0 and rate_chg_bps >  20 and eur_chg_pct > -0.5)
                        )
                        if is_divergence:
                            size_mult.loc[date] = round(size_mult.loc[date] * 1.25, 4)
                    except Exception:
                        pass
            except Exception:
                pass

        # Apply forex allocation weight from regime engine (continuous dimmer)
        try:
            from sovereign.intelligence.allocation_engine import read_allocation as _read_alloc
            _alloc_weight = _read_alloc().forex_weight
            if _alloc_weight < 1.0:
                size_mult = (size_mult * _alloc_weight).round(4)
        except Exception:
            pass

        return size_mult

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

            # ── USDCHF: require confirmed momentum alignment ──────────── #
            # v003 gate (SNB rate change > 0.10) was broken: SNB held flat
            # -0.75% for 7 years → nearly all signals filtered → noise.
            # v004 fix: remove rate gate. Instead require BOTH macro sign AND
            # 3-month momentum in the same direction. CHF safe-haven reversals
            # are fast and macro-confirmed; pure IRP signals are unreliable.
            # This doubles the confirmation bar without blocking all signals.
            _usdchf_mom_gate = True   # assume pass unless pair-specific check fails
            if pair == 'USDCHF=X':
                hist_check = close.loc[:date]
                if len(hist_check) >= 63:
                    mom_63 = float(hist_check.iloc[-1] / hist_check.iloc[-63] - 1)
                    # Only allow signal if 3-month momentum is meaningful
                    _usdchf_mom_gate = abs(mom_63) > 0.015  # 1.5% move in 3 months

            if pair == 'USDCHF=X' and not _usdchf_mom_gate:
                continue

            macro_sign = self._macro_signal_for_date(
                close=close, date=date,
                base_country=base_country, quote_country=quote_country,
                base_rates=base_rates, quote_rates=quote_rates,
                base_cpi_h=base_cpi_h, quote_cpi_h=quote_cpi_h,
            )

            # ── EURGBP: require 6-month momentum confirmation ─────────── #
            # v003: ECB and BOE hiked in lockstep 2022-2024 → IRP differential
            # near zero → random macro signals → -0.04 Sharpe.
            # v004 fix: require 6-month price momentum to confirm macro direction.
            # EURGBP is a cross with low carry — macro alone is insufficient.
            if pair == 'EURGBP=X' and macro_sign != 0:
                hist_check = close.loc[:date]
                if len(hist_check) >= 126:
                    mom_126 = float(hist_check.iloc[-1] / hist_check.iloc[-126] - 1)
                    # Only enter if 6-month momentum agrees with macro direction
                    if int(np.sign(mom_126)) != macro_sign:
                        macro_sign = 0  # suppress signal: macro and momentum conflict

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
            return atr_pct.bfill()
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
            self.config.irp_weight * np.clip(-irp_z / 1.5, -1, 1) +
            self.config.rate_weight * np.clip(real_rate_diff / 4.0, -1, 1)
        )

        macro_sign = int(np.sign(macro_score)) if abs(macro_score) > self.config.signal_threshold else 0
        if not self.config.use_momentum_filter:
            return macro_sign

        mom_sign = 0
        if len(hist) > 63:
            mom = float(hist.iloc[-1] / hist.iloc[-63] - 1)
            mom_sign = int(np.sign(mom)) if abs(mom) > 0.005 else 0

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
