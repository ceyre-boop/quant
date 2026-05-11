"""
CPI Surprise Fade — Edge 2 from the cause-effect map.

Mechanism: CPI beats/misses consensus → initial FX move overshoots →
fade it 24-48h later for a 3-10 day mean-reversion trade.
Win rate: 55-60% historically.

Consensus proxy: trailing 12-month average MoM change (no paid data needed).
Surprise = actual MoM change - consensus proxy.
Signal fires the day AFTER the release (fade, not chase).

Source: FRED CPI series (same as data_fetcher.py).
Cache: uses existing data/cache/macro/{country}_cpi.parquet.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum surprise to generate a signal (in percentage points MoM)
MIN_SURPRISE_PCT = 0.15

# Countries and pairs affected — base/quote determines fade direction
# 'base_usd': True means USD is the base currency in the pair notation
CPI_AFFECTED_PAIRS: dict[str, str] = {
    'US':  'FED',
    'EU':  'ECB',
    'UK':  'BOE',
    'JP':  'BOJ',
    'AU':  'RBA',
    'CA':  'BOC',
    'NZ':  'RBNZ',
}

FADE_HOLD_DAYS = 5


class CPISurpriseEngine:

    def __init__(self, fetcher=None):
        self._fetcher = fetcher   # ForexDataFetcher, optional — uses cache if None

    def get_latest_surprise(self, country: str) -> Optional[dict]:
        """
        Returns the most recent CPI surprise for a country, or None.

        Return:
            {'country': str, 'surprise_pct': float,
             'direction': 'LONG'|'SHORT',  # direction to FADE (opposite to initial move)
             'conviction': float, 'hold_days': int,
             'release_date': str}

        FADE logic:
            CPI beat (surprise > 0) → initial FX move UP for base currency
            → FADE = SHORT base currency (expect reversal)
            CPI miss (surprise < 0) → initial FX move DOWN
            → FADE = LONG base currency
        """
        # The cached series is already YoY % — compare latest reading to
        # trailing 12-month average as consensus proxy.
        yoy = self._get_cpi_series(country)
        if yoy is None or len(yoy) < 14:
            return None

        yoy = yoy.dropna()
        actual    = float(yoy.iloc[-1])
        consensus = float(yoy.iloc[-13:-1].mean())
        surprise  = actual - consensus   # in percentage points YoY

        if abs(surprise) < MIN_SURPRISE_PCT:
            return None

        direction  = 'SHORT' if surprise > 0 else 'LONG'
        conviction = min(0.40 + abs(surprise) * 1.5, 0.65)

        return {
            'country':      country,
            'surprise_pct': round(surprise, 3),
            'direction':    direction,
            'conviction':   round(conviction, 3),
            'hold_days':    FADE_HOLD_DAYS,
            'release_date': str(yoy.index[-1].date()),
        }

    def get_historical_surprises(
        self,
        country: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[dict]:
        """
        Returns all CPI surprise events in [start, end] for backtesting.
        Each event has 'signal_date' = release_date + 1 business day (enter next day).
        """
        cpi = self._get_cpi_series(country)
        if cpi is None or len(cpi) < 14:
            return []

        yoy = self._get_cpi_series(country)
        if yoy is None:
            return []
        yoy = yoy.dropna()
        # Resample to monthly — CPI releases are monthly (or quarterly)
        # This prevents duplicate signals from daily forward-fill noise
        monthly = yoy.resample('MS').last().dropna()
        results = []

        for i in range(12, len(monthly)):
            d = monthly.index[i]
            if not (start <= d <= end):
                continue

            actual    = float(monthly.iloc[i])
            consensus = float(monthly.iloc[i - 12:i].mean())
            surprise  = actual - consensus

            if abs(surprise) < MIN_SURPRISE_PCT:
                continue

            direction  = 'SHORT' if surprise > 0 else 'LONG'
            conviction = min(0.40 + abs(surprise) * 1.5, 0.65)

            signal_date = d + pd.offsets.BDay(1)

            results.append({
                'signal_date':   signal_date,
                'release_date':  d,
                'country':       country,
                'surprise_pct':  round(surprise, 3),
                'direction':     direction,
                'conviction':    round(conviction, 3),
                'hold_days':     FADE_HOLD_DAYS,
            })

        return results

    def _get_cpi_series(self, country: str) -> Optional[pd.Series]:
        cache_path = (Path(__file__).parents[2] / 'data' / 'cache' / 'macro'
                      / f'{country}_cpi.parquet')
        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path).squeeze()
            except Exception:
                logger.debug("CPI cache read failed for %s, trying fetcher", country)

        if self._fetcher:
            try:
                return self._fetcher.get_cpi_history(country, start='2010-01-01')
            except Exception as e:
                logger.warning(f"CPI fetch failed for {country}: {e}")
        return None
