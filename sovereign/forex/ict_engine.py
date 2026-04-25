"""
ICT price-action analysis engine for forex.

Detects on OHLCV DataFrames (daily or intraday):
  - Fair Value Gaps (FVG) — 3-candle imbalance
  - Order Blocks (OB) — last opposing candle before impulse
  - Breaker Blocks — invalidated OBs
  - Market Structure: HH/HL/LH/LL, BOS, CHOCH
  - Liquidity Sweeps — stop-hunt + reversal
  - Kill Zone membership (ET time)
  - Optimal Trade Entry (OTE) Fibonacci zones (62–79%)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timezone, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Kill Zone windows (ET = UTC-4 in summer, UTC-5 in winter) ──────────── #
# We store as (hh_start, mm_start, hh_end, mm_end) in UTC offsets applied below.
_KZ: dict = {
    'London':    (2, 0,  5, 0),
    'NY_Open':   (7, 0, 10, 0),
    'NY_PM':     (13, 30, 16, 0),
    'Asia':      (20, 0, 23, 59),
}
_NY_LUNCH_START = time(12, 0)
_NY_LUNCH_END   = time(13, 30)


@dataclass
class FVG:
    kind: str           # BULLISH / BEARISH
    top: float          # upper boundary
    bottom: float       # lower boundary
    midpoint: float
    formed_at: pd.Timestamp
    filled: bool = False

    @property
    def size(self) -> float:
        return self.top - self.bottom


@dataclass
class OrderBlock:
    kind: str           # BULLISH / BEARISH
    high: float
    low: float
    midpoint: float
    formed_at: pd.Timestamp
    valid: bool = True
    is_breaker: bool = False


@dataclass
class LiquiditySweep:
    direction: str      # BULLISH_SWEEP (swept SSL → expect up) / BEARISH_SWEEP
    swept_level: float
    sweep_candle_low: float
    sweep_candle_high: float
    formed_at: pd.Timestamp


@dataclass
class MarketStructure:
    trend: str          # BULLISH / BEARISH / RANGING
    last_bos: Optional[str] = None      # 'BULLISH' / 'BEARISH'
    last_choch: Optional[str] = None    # 'BULLISH' / 'BEARISH'
    swing_highs: List[float] = field(default_factory=list)
    swing_lows: List[float] = field(default_factory=list)


@dataclass
class ICTAnalysis:
    pair: str
    as_of: pd.Timestamp
    market_structure: MarketStructure
    active_fvgs: List[FVG]
    active_obs: List[OrderBlock]
    recent_sweeps: List[LiquiditySweep]
    in_kill_zone: bool
    kill_zone_name: Optional[str]
    in_ny_lunch: bool
    # Price context
    current_price: float
    atr_daily: float
    # Nearest actionable zones
    nearest_bullish_ob: Optional[OrderBlock]
    nearest_bearish_ob: Optional[OrderBlock]
    nearest_bullish_fvg: Optional[FVG]
    nearest_bearish_fvg: Optional[FVG]


class ICTEngine:

    SWING_PERIOD = 5          # bars each side to define swing high/low
    OB_LOOKBACK = 20          # how far back to look for OBs
    FVG_LOOKBACK = 30
    SWEEP_LOOKBACK = 10
    IMPULSE_MIN_MULTIPLE = 1.5  # impulse must be 1.5× ATR to qualify

    def analyse(self, pair: str, df: pd.DataFrame, as_of: Optional[pd.Timestamp] = None) -> ICTAnalysis:
        """
        Main entry point.
        df must have OHLCV columns: Open, High, Low, Close, Volume (Volume optional).
        """
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Open', 'High', 'Low', 'Close']].dropna()
        if len(df) < 50:
            raise ValueError(f"Need at least 50 bars for ICT analysis, got {len(df)}")

        as_of = as_of or df.index[-1]
        price = float(df['Close'].iloc[-1])
        atr = self._atr(df)

        ms = self._market_structure(df)
        fvgs = self._find_fvgs(df, atr)
        obs = self._find_order_blocks(df, atr)
        sweeps = self._find_sweeps(df, atr)

        in_kz, kz_name = self._kill_zone(as_of)
        in_lunch = self._ny_lunch(as_of)

        # Nearest actionable OBs
        bull_obs = [o for o in obs if o.kind == 'BULLISH' and o.valid and not o.is_breaker and o.high < price]
        bear_obs = [o for o in obs if o.kind == 'BEARISH' and o.valid and not o.is_breaker and o.low > price]
        nearest_bull_ob = max(bull_obs, key=lambda o: o.high) if bull_obs else None
        nearest_bear_ob = min(bear_obs, key=lambda o: o.low) if bear_obs else None

        bull_fvgs = [f for f in fvgs if f.kind == 'BULLISH' and not f.filled and f.top < price]
        bear_fvgs = [f for f in fvgs if f.kind == 'BEARISH' and not f.filled and f.bottom > price]
        nearest_bull_fvg = max(bull_fvgs, key=lambda f: f.top) if bull_fvgs else None
        nearest_bear_fvg = min(bear_fvgs, key=lambda f: f.bottom) if bear_fvgs else None

        return ICTAnalysis(
            pair=pair,
            as_of=as_of,
            market_structure=ms,
            active_fvgs=fvgs,
            active_obs=obs,
            recent_sweeps=sweeps,
            in_kill_zone=in_kz,
            kill_zone_name=kz_name,
            in_ny_lunch=in_lunch,
            current_price=price,
            atr_daily=atr,
            nearest_bullish_ob=nearest_bull_ob,
            nearest_bearish_ob=nearest_bear_ob,
            nearest_bullish_fvg=nearest_bull_fvg,
            nearest_bearish_fvg=nearest_bear_fvg,
        )

    # ── Market Structure ──────────────────────────────────────────────── #

    def _market_structure(self, df: pd.DataFrame) -> MarketStructure:
        highs = df['High'].values
        lows = df['Low'].values
        n = len(df)
        p = self.SWING_PERIOD

        swing_high_idx = [
            i for i in range(p, n - p)
            if highs[i] == max(highs[i - p:i + p + 1])
        ]
        swing_low_idx = [
            i for i in range(p, n - p)
            if lows[i] == min(lows[i - p:i + p + 1])
        ]

        sh_vals = [highs[i] for i in swing_high_idx[-6:]]
        sl_vals = [lows[i] for i in swing_low_idx[-6:]]

        trend = 'RANGING'
        last_bos = None
        last_choch = None

        if len(sh_vals) >= 2 and len(sl_vals) >= 2:
            higher_highs = sh_vals[-1] > sh_vals[-2]
            higher_lows  = sl_vals[-1] > sl_vals[-2]
            lower_highs  = sh_vals[-1] < sh_vals[-2]
            lower_lows   = sl_vals[-1] < sl_vals[-2]

            if higher_highs and higher_lows:
                trend = 'BULLISH'
                last_bos = 'BULLISH'
            elif lower_highs and lower_lows:
                trend = 'BEARISH'
                last_bos = 'BEARISH'
            elif higher_lows and lower_highs:
                # First break of prior bearish structure = bullish CHOCH
                last_choch = 'BULLISH'
                trend = 'BULLISH'
            elif lower_highs and higher_lows:
                last_choch = 'BEARISH'
                trend = 'BEARISH'

        return MarketStructure(
            trend=trend,
            last_bos=last_bos,
            last_choch=last_choch,
            swing_highs=sh_vals,
            swing_lows=sl_vals,
        )

    # ── Fair Value Gaps ───────────────────────────────────────────────── #

    def _find_fvgs(self, df: pd.DataFrame, atr: float) -> List[FVG]:
        fvgs = []
        price_now = float(df['Close'].iloc[-1])
        lookback = min(self.FVG_LOOKBACK, len(df) - 2)

        for i in range(len(df) - lookback, len(df) - 2):
            c1 = df.iloc[i]
            c2 = df.iloc[i + 1]
            c3 = df.iloc[i + 2]

            # Bullish FVG: c1 high < c3 low (gap between c1 top and c3 bottom)
            if c1['High'] < c3['Low']:
                top = c3['Low']
                bottom = c1['High']
                if (top - bottom) >= 0.3 * atr:
                    filled = price_now < bottom
                    fvgs.append(FVG(
                        kind='BULLISH',
                        top=top,
                        bottom=bottom,
                        midpoint=(top + bottom) / 2,
                        formed_at=df.index[i + 1],
                        filled=filled,
                    ))

            # Bearish FVG: c1 low > c3 high
            if c1['Low'] > c3['High']:
                top = c1['Low']
                bottom = c3['High']
                if (top - bottom) >= 0.3 * atr:
                    filled = price_now > top
                    fvgs.append(FVG(
                        kind='BEARISH',
                        top=top,
                        bottom=bottom,
                        midpoint=(top + bottom) / 2,
                        formed_at=df.index[i + 1],
                        filled=filled,
                    ))

        return fvgs

    # ── Order Blocks ──────────────────────────────────────────────────── #

    def _find_order_blocks(self, df: pd.DataFrame, atr: float) -> List[OrderBlock]:
        obs = []
        price_now = float(df['Close'].iloc[-1])
        lookback = min(self.OB_LOOKBACK, len(df) - 2)
        threshold = self.IMPULSE_MIN_MULTIPLE * atr

        for i in range(len(df) - lookback, len(df) - 1):
            c_ob = df.iloc[i]
            c_impulse = df.iloc[i + 1]

            impulse_size = abs(c_impulse['Close'] - c_impulse['Open'])

            # Bullish OB: last bearish candle before strong bullish impulse
            if (c_ob['Close'] < c_ob['Open'] and
                    c_impulse['Close'] > c_impulse['Open'] and
                    impulse_size >= threshold):
                ob = OrderBlock(
                    kind='BULLISH',
                    high=c_ob['High'],
                    low=c_ob['Low'],
                    midpoint=(c_ob['High'] + c_ob['Low']) / 2,
                    formed_at=df.index[i],
                    valid=price_now > c_ob['Low'],
                    is_breaker=price_now < c_ob['Low'],  # price broke below → breaker
                )
                obs.append(ob)

            # Bearish OB: last bullish candle before strong bearish impulse
            if (c_ob['Close'] > c_ob['Open'] and
                    c_impulse['Close'] < c_impulse['Open'] and
                    impulse_size >= threshold):
                ob = OrderBlock(
                    kind='BEARISH',
                    high=c_ob['High'],
                    low=c_ob['Low'],
                    midpoint=(c_ob['High'] + c_ob['Low']) / 2,
                    formed_at=df.index[i],
                    valid=price_now < c_ob['High'],
                    is_breaker=price_now > c_ob['High'],
                )
                obs.append(ob)

        return obs

    # ── Liquidity Sweeps ─────────────────────────────────────────────── #

    def _find_sweeps(self, df: pd.DataFrame, atr: float) -> List[LiquiditySweep]:
        sweeps = []
        lookback = min(self.SWEEP_LOOKBACK, len(df) - 2)
        p = self.SWING_PERIOD

        for i in range(max(p, len(df) - lookback), len(df) - 1):
            c = df.iloc[i]
            c_next = df.iloc[i + 1]

            # Get recent swing levels to check if they were swept
            prior_highs = df['High'].iloc[max(0, i - 20):i]
            prior_lows = df['Low'].iloc[max(0, i - 20):i]
            if prior_highs.empty or prior_lows.empty:
                continue

            ssl = float(prior_lows.min())
            bsl = float(prior_highs.max())

            # Bullish sweep: wicked below SSL then closed back above
            if c['Low'] < ssl and c['Close'] > ssl and c_next['Close'] > c['Close']:
                sweeps.append(LiquiditySweep(
                    direction='BULLISH_SWEEP',
                    swept_level=ssl,
                    sweep_candle_low=c['Low'],
                    sweep_candle_high=c['High'],
                    formed_at=df.index[i],
                ))

            # Bearish sweep: wicked above BSL then closed back below
            if c['High'] > bsl and c['Close'] < bsl and c_next['Close'] < c['Close']:
                sweeps.append(LiquiditySweep(
                    direction='BEARISH_SWEEP',
                    swept_level=bsl,
                    sweep_candle_low=c['Low'],
                    sweep_candle_high=c['High'],
                    formed_at=df.index[i],
                ))

        return sweeps

    # ── Kill Zones ────────────────────────────────────────────────────── #

    @staticmethod
    def _et_time(ts: pd.Timestamp) -> time:
        """Convert timestamp to Eastern Time (approximation: UTC-4 summer, UTC-5 winter)."""
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        # DST approximation: EDT (UTC-4) from Mar to Nov, EST (UTC-5) otherwise
        month = ts.month
        offset_hours = -4 if 3 <= month <= 11 else -5
        et = ts + pd.Timedelta(hours=offset_hours)
        return et.time()

    def _kill_zone(self, ts: pd.Timestamp) -> Tuple[bool, Optional[str]]:
        t = self._et_time(ts)
        for name, (sh, sm, eh, em) in _KZ.items():
            start = time(sh, sm)
            end = time(eh, em)
            if start <= t <= end:
                return True, name
        return False, None

    @staticmethod
    def _ny_lunch(ts: pd.Timestamp) -> bool:
        t = ICTEngine._et_time(ts)
        return _NY_LUNCH_START <= t <= _NY_LUNCH_END

    # ── Helpers ───────────────────────────────────────────────────────── #

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float:
        h, l, c = df['High'], df['Low'], df['Close']
        tr = pd.concat([
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    @staticmethod
    def ote_levels(impulse_low: float, impulse_high: float) -> Tuple[float, float]:
        """Return (ote_62, ote_79) retracement levels for a bullish impulse."""
        span = impulse_high - impulse_low
        return (
            impulse_high - 0.62 * span,
            impulse_high - 0.79 * span,
        )
