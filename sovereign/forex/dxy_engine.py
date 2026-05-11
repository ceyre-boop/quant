"""
DXY Trend Overlay — Dollar Smile Theory.

The dollar strengthens in TWO opposite conditions:
  1. US economy STRONG → capital attracted to growth (risk-on bull)
  2. Global CRISIS → flight to safety (risk-off bull)

It weakens only when the US is mediocre — not strong enough to attract
capital, not scary enough to trigger safe-haven flows.

Without this overlay, the macro engine can generate USD signals that
fight the structural dollar trend.

Smile detection: use VIX to distinguish growth vs safety regimes.
  DXY trending up + VIX < 20 → GROWTH_DRIVEN smile → boost USD-long
  DXY trending up + VIX > 25 → SAFETY_DRIVEN smile → boost safe-haven (JPY, CHF)
  DXY flat / down → WEAK → reduce USD-long signals
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_PATH  = Path(__file__).parents[2] / 'data' / 'cache' / 'dxy.parquet'
CACHE_DAYS  = 1

DXY_TICKER  = 'DX-Y.NYB'
VIX_TICKER  = '^VIX'

# Pairs where USD is the base (USD strengthening → pair goes UP)
USD_BASE_PAIRS  = {'USDCHF=X', 'USDCAD=X', 'USDJPY=X'}
# Pairs where USD is the quote (USD strengthening → pair goes DOWN)
USD_QUOTE_PAIRS = {'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X'}
# Safe-haven pairs (strong in SAFETY_DRIVEN smile)
SAFE_HAVEN_PAIRS = {'USDJPY=X', 'USDCHF=X'}


class DXYEngine:

    def __init__(self):
        self._cache: Optional[dict] = None
        self._cache_ts: Optional[pd.Timestamp] = None

    def get_trend(self) -> dict:
        """
        Returns:
            trend:         'STRONG_BULL' / 'BULL' / 'NEUTRAL' / 'BEAR' / 'STRONG_BEAR'
            z_score:       20-day price vs 252-day rolling mean, normalised by std
            smile_regime:  'GROWTH_DRIVEN' / 'SAFETY_DRIVEN' / 'WEAK'
            vix:           current VIX level
        """
        now = pd.Timestamp.utcnow()
        if (self._cache is not None and self._cache_ts is not None and
                (now - self._cache_ts).total_seconds() < 3600):
            return self._cache

        result = self._compute()
        self._cache = result
        self._cache_ts = now
        return result

    def get_modifier(self, pair: str, direction: str) -> float:
        """
        Returns a size multiplier (0.7 – 1.3) based on DXY trend and smile regime.

        Rules:
          STRONG_BULL + GROWTH_DRIVEN:
            → USD-long signals (USD base LONG, USD quote SHORT): ×1.3
            → USD-short signals: ×0.7
          STRONG_BULL + SAFETY_DRIVEN:
            → Safe-haven pairs (USDJPY, USDCHF) matching trend: ×1.2
            → Carry/risk pairs: ×0.8
          STRONG_BEAR:
            → USD-short signals: ×1.2
            → USD-long signals: ×0.7
          Otherwise: ×1.0
        """
        trend_data = self.get_trend()
        trend  = trend_data['trend']
        smile  = trend_data['smile_regime']

        usd_long  = (pair in USD_BASE_PAIRS  and direction == 'LONG') or \
                    (pair in USD_QUOTE_PAIRS and direction == 'SHORT')
        usd_short = (pair in USD_BASE_PAIRS  and direction == 'SHORT') or \
                    (pair in USD_QUOTE_PAIRS and direction == 'LONG')
        safe_haven_long = pair in SAFE_HAVEN_PAIRS and direction == 'SHORT'  # short USDJPY = long JPY

        if trend == 'STRONG_BULL' and smile == 'GROWTH_DRIVEN':
            if usd_long:   return 1.3
            if usd_short:  return 0.7
        elif trend == 'STRONG_BULL' and smile == 'SAFETY_DRIVEN':
            if safe_haven_long: return 1.2
            if pair in {'AUDUSD=X', 'NZDUSD=X'} and direction == 'LONG': return 0.8
        elif trend in ('BULL',):
            if usd_long:  return 1.1
            if usd_short: return 0.9
        elif trend == 'STRONG_BEAR':
            if usd_short: return 1.2
            if usd_long:  return 0.7
        elif trend == 'BEAR':
            if usd_short: return 1.1
            if usd_long:  return 0.9

        return 1.0

    # ── Internals ─────────────────────────────────────────────────────── #

    def _compute(self) -> dict:
        dxy = self._load_dxy()
        vix = self._latest_vix()

        if dxy is None or len(dxy) < 30:
            return {'trend': 'NEUTRAL', 'z_score': 0.0,
                    'smile_regime': 'WEAK', 'vix': vix}

        price  = float(dxy.iloc[-1])
        mu20   = float(dxy.tail(20).mean())
        mu252  = float(dxy.tail(252).mean())
        std252 = float(dxy.tail(252).std())
        z = (price - mu252) / std252 if std252 > 0 else 0.0

        if z > 1.5:
            trend = 'STRONG_BULL'
        elif z > 0.5:
            trend = 'BULL'
        elif z < -1.5:
            trend = 'STRONG_BEAR'
        elif z < -0.5:
            trend = 'BEAR'
        else:
            trend = 'NEUTRAL'

        if trend in ('STRONG_BULL', 'BULL'):
            smile = 'GROWTH_DRIVEN' if vix < 20 else 'SAFETY_DRIVEN'
        else:
            smile = 'WEAK'

        return {
            'trend':        trend,
            'z_score':      round(z, 3),
            'smile_regime': smile,
            'vix':          round(vix, 2),
        }

    def _load_dxy(self) -> Optional[pd.Series]:
        if CACHE_PATH.exists():
            age = (datetime.now() - datetime.fromtimestamp(
                CACHE_PATH.stat().st_mtime)).days
            if age < CACHE_DAYS:
                try:
                    return pd.read_parquet(CACHE_PATH).squeeze()
                except Exception:
                    logger.debug("DXY cache read failed, refetching")
        try:
            df = yf.download(DXY_TICKER, period='3y', progress=False, auto_adjust=True)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            s = df['Close'].dropna()
            s.name = 'DXY'
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            s.to_frame('DXY').to_parquet(CACHE_PATH)
            return s
        except Exception as e:
            logger.warning(f"DXY download failed: {e}")
            return None

    @staticmethod
    def _latest_vix() -> float:
        try:
            df = yf.download(VIX_TICKER, period='5d', progress=False, auto_adjust=True)
            if df.empty:
                return 18.0
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return float(df['Close'].dropna().iloc[-1])
        except Exception:
            return 18.0
