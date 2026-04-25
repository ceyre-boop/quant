"""
Risk sentiment engine — VIX + carry unwind detector.

Edge 4 (from cause-effect map): VIX > 25 + backwardation = short carry, long safe havens.
Historical win rate: 65–70% — highest conviction edge in the system.

VIX term structure:
  Contango  (VIX3M > VIX) = normal, risk-on
  Backwardation (VIX > VIX3M) = risk-off already underway

Safe havens: JPY, CHF, USD
Risk currencies: AUD, NZD (carry)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Pairs to short in risk-off (sell the carry, buy the safe haven)
CARRY_UNWIND_SHORTS = ['AUDJPY=X', 'NZDJPY=X', 'AUDUSD=X', 'NZDUSD=X']
SAFE_HAVEN_LONGS    = ['USDJPY=X']   # SHORT USD/JPY = long JPY

VIX_RISK_OFF_THRESHOLD    = 25.0
VIX_RISK_ON_THRESHOLD     = 20.0
VIX_EXTREME_THRESHOLD     = 35.0   # carry unwinds are violent above this


@dataclass
class RiskSentimentReading:
    vix: float
    vix_3m: float
    regime: str             # RISK_ON / RISK_OFF / EXTREME_RISK_OFF / NEUTRAL
    term_structure: str     # CONTANGO / BACKWARDATION / FLAT
    carry_unwind_active: bool
    vix_zscore: float       # vs trailing 1-year z-score
    # Override signals: pair → forced direction
    # These override macro signals when carry unwind is active
    forced_shorts: list[str]   # pairs to short regardless of macro
    forced_longs: list[str]    # pairs to long regardless of macro


class RiskSentimentEngine:

    VIX_TICKER  = '^VIX'
    VIX3M_TICKER = '^VIX3M'    # yfinance ticker for 3-month VIX

    def __init__(self):
        self._cache: Optional[RiskSentimentReading] = None
        self._cache_ts: Optional[pd.Timestamp] = None

    def get_reading(self, refresh: bool = False) -> RiskSentimentReading:
        # Cache for the trading day
        now = pd.Timestamp.utcnow()
        if (not refresh and self._cache is not None and
                self._cache_ts is not None and
                (now - self._cache_ts).total_seconds() < 3600):
            return self._cache

        reading = self._fetch()
        self._cache = reading
        self._cache_ts = now
        return reading

    def is_risk_off(self) -> bool:
        r = self.get_reading()
        return r.regime in ('RISK_OFF', 'EXTREME_RISK_OFF')

    def override_for_pair(self, pair: str) -> Optional[str]:
        """
        Returns 'SHORT', 'LONG', or None.
        During carry unwinds, certain pairs get forced direction regardless of macro.
        """
        r = self.get_reading()
        if not r.carry_unwind_active:
            return None
        if pair in r.forced_shorts:
            return 'SHORT'
        if pair in r.forced_longs:
            return 'LONG'
        return None

    # ── Internal ──────────────────────────────────────────────────────── #

    def _fetch(self) -> RiskSentimentReading:
        import yfinance as yf

        vix = self._latest(self.VIX_TICKER, fallback=18.0)
        vix_3m = self._latest(self.VIX3M_TICKER, fallback=19.0)

        # Z-score vs trailing 252-day VIX
        vix_z = self._vix_zscore(vix)

        # Term structure
        spread = vix_3m - vix
        if spread > 1.0:
            term_structure = 'CONTANGO'
        elif spread < -1.0:
            term_structure = 'BACKWARDATION'
        else:
            term_structure = 'FLAT'

        # Regime
        if vix >= VIX_EXTREME_THRESHOLD and term_structure == 'BACKWARDATION':
            regime = 'EXTREME_RISK_OFF'
        elif vix >= VIX_RISK_OFF_THRESHOLD and term_structure == 'BACKWARDATION':
            regime = 'RISK_OFF'
        elif vix <= VIX_RISK_ON_THRESHOLD and term_structure == 'CONTANGO':
            regime = 'RISK_ON'
        else:
            regime = 'NEUTRAL'

        carry_active = regime in ('RISK_OFF', 'EXTREME_RISK_OFF')

        # During risk-off: short carry currencies vs safe havens
        forced_shorts = CARRY_UNWIND_SHORTS if carry_active else []
        # USD/JPY = short = long JPY safe haven
        forced_longs_raw = []
        if carry_active:
            # We express JPY strength as: USDJPY short (dollar weakens vs yen)
            forced_longs_raw = []
            forced_shorts = CARRY_UNWIND_SHORTS + ['USDJPY=X']  # short USD/JPY = long JPY

        return RiskSentimentReading(
            vix=round(vix, 2),
            vix_3m=round(vix_3m, 2),
            regime=regime,
            term_structure=term_structure,
            carry_unwind_active=carry_active,
            vix_zscore=round(vix_z, 2),
            forced_shorts=forced_shorts,
            forced_longs=forced_longs_raw,
        )

    @staticmethod
    def _latest(ticker: str, fallback: float) -> float:
        import yfinance as yf
        try:
            df = yf.download(ticker, period='5d', progress=False, auto_adjust=True)
            if df.empty:
                return fallback
            if hasattr(df.columns, 'get_level_values'):
                df.columns = df.columns.get_level_values(0)
            val = df['Close'].dropna().iloc[-1]
            return float(val)
        except Exception:
            return fallback

    @staticmethod
    def _vix_zscore(current_vix: float) -> float:
        import yfinance as yf
        try:
            df = yf.download('^VIX', period='1y', progress=False, auto_adjust=True)
            if df.empty or len(df) < 50:
                return 0.0
            if hasattr(df.columns, 'get_level_values'):
                df.columns = df.columns.get_level_values(0)
            s = df['Close'].dropna()
            mu, sigma = s.mean(), s.std()
            return float((current_vix - mu) / sigma) if sigma > 0 else 0.0
        except Exception:
            return 0.0
