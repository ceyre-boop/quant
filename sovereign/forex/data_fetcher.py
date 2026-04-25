"""
Forex macro data fetcher.

Primary: FRED API (if FRED_API_KEY in env)
Fallback: yfinance proxies + hardcoded known rates

Cached to data/cache/macro/{country}.parquet — updates monthly.
"""
from __future__ import annotations

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parents[3] / 'data' / 'cache' / 'macro'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# FRED series IDs per country/metric
FRED_RATES: Dict[str, str] = {
    'US': 'FEDFUNDS',
    'EU': 'ECBDFR',
    'UK': 'IUDSOIA',
    'JP': 'IRSTCI01JPM156N',
    'CH': 'IR3TIB01CHM156N',
    'AU': 'IR3TIB01AUM156N',   # Australia 3-month interbank rate (RBA proxy)
    'CA': 'IR3TIB01CAM156N',   # Canada 3-month T-bill rate (BOC proxy)
    'NZ': 'IR3TIB01NZM156N',   # New Zealand 3-month rate (RBNZ proxy)
}

FRED_CPI: Dict[str, str] = {
    'US': 'CPIAUCSL',
    'EU': 'CP0000EZ19M086NEST',
    'UK': 'GBRCPIALLMINMEI',
    'JP': '',   # No reliable current FRED series — uses fallback 3.2%
    'CH': 'CHECPIALLMINMEI',
    'AU': 'AUSCPIALLQINMEI',
    'CA': 'CANCPIALLMINMEI',
    'NZ': 'NZLCPIALLQINMEI',
}

# CPI series that are quarterly (4 periods = 1 year, not 12)
QUARTERLY_CPI = {'AU', 'NZ'}

DIRECT_YOY_CPI: set = set()  # reserved for future direct-% series

FRED_GDP: Dict[str, str] = {
    'US': 'GDP',
    'EU': 'EUNNGDP',
    'UK': 'UKNGDP',
    'JP': 'JPNNGDP',
}

# yfinance rate proxies (short-term gov yields as central bank rate proxies)
# These are close enough for signal generation
YF_RATE_PROXIES: Dict[str, Tuple[str, float]] = {
    'US': ('^IRX', 0.01),     # 13-week T-bill, already in %
    'EU': ('^TNX', 0.01),     # fallback: use US 10Y as proxy if EU unavailable
    'UK': ('^TNX', 0.01),
    'JP': ('^TNX', 0.01),
    'AU': ('^TNX', 0.01),
    'CA': ('^TNX', 0.01),
    'CH': ('^TNX', 0.01),
    'NZ': ('^TNX', 0.01),
}

# Known current approximate rates (2026-04) as last-resort fallback
# These change slowly — update quarterly
FALLBACK_RATES: Dict[str, float] = {
    'US': 4.33, 'EU': 2.50, 'UK': 4.50, 'JP': 0.50,
    'CH': 0.50, 'AU': 4.10, 'CA': 2.75, 'NZ': 3.50,
}

FALLBACK_CPI: Dict[str, float] = {
    'US': 2.4, 'EU': 2.2, 'UK': 2.8, 'JP': 3.2,
    'CH': 0.3, 'AU': 2.4, 'CA': 2.3, 'NZ': 2.2,
}

FALLBACK_GDP_GROWTH: Dict[str, float] = {
    'US': 2.5, 'EU': 1.2, 'UK': 0.8, 'JP': -0.2,
    'CH': 1.0, 'AU': 1.5, 'CA': 1.8, 'NZ': 0.5,
}

# Rate trajectory from last 3 decisions (1=hike, 0=hold, -1=cut)
# Update monthly from central bank announcements
RATE_TRAJECTORY: Dict[str, list] = {
    'US': [-1, -1, 0],   # FED cut twice, now holding
    'EU': [-1, -1, -1],  # ECB cutting cycle
    'UK': [-1, -1, 0],
    'JP': [1, 1, 0],     # BOJ hiking from ZIRP
    'CH': [-1, -1, -1],  # SNB cutting
    'AU': [-1, 0, 0],
    'CA': [-1, -1, -1],
    'NZ': [-1, -1, -1],
}


class ForexDataFetcher:
    """
    Fetches and caches macro fundamentals for all 8 forex economies.
    """

    CACHE_DAYS = 30

    def __init__(self):
        self._fred = None
        self._fred_ok = False
        fred_key = os.getenv('FRED_API_KEY')
        if fred_key:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=fred_key)
                self._fred_ok = True
                logger.info("ForexDataFetcher: FRED API ready")
            except Exception as e:
                logger.warning(f"FRED init failed: {e}")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_country_macro(self, country: str, refresh: bool = False) -> dict:
        """
        Returns current macro snapshot for a country.
        Keys: rate, cpi_yoy, gdp_growth, real_rate, rate_trajectory
        """
        cache_path = CACHE_DIR / f'{country}_macro.json'

        if not refresh and cache_path.exists():
            age_days = (datetime.now() - datetime.fromtimestamp(
                cache_path.stat().st_mtime
            )).days
            if age_days < self.CACHE_DAYS:
                with open(cache_path) as f:
                    return json.load(f)

        macro = self._fetch_macro(country)
        with open(cache_path, 'w') as f:
            json.dump(macro, f, indent=2, default=str)
        return macro

    def get_rate_history(
        self, country: str, start: str = '2015-01-01'
    ) -> pd.Series:
        """
        Returns historical rate series for backtesting.
        Cached as parquet.
        """
        cache_path = CACHE_DIR / f'{country}_rates.parquet'
        if cache_path.exists():
            age = (datetime.now() - datetime.fromtimestamp(
                cache_path.stat().st_mtime
            )).days
            if age < self.CACHE_DAYS:
                return pd.read_parquet(cache_path).squeeze()

        series = self._fetch_rate_history(country, start)
        if series is not None and not series.empty:
            series.to_frame('rate').to_parquet(cache_path)
        return series if series is not None else pd.Series(dtype=float)

    def get_cpi_history(
        self, country: str, start: str = '2015-01-01'
    ) -> pd.Series:
        cache_path = CACHE_DIR / f'{country}_cpi.parquet'
        if cache_path.exists():
            age = (datetime.now() - datetime.fromtimestamp(
                cache_path.stat().st_mtime
            )).days
            if age < self.CACHE_DAYS:
                return pd.read_parquet(cache_path).squeeze()

        series = self._fetch_cpi_history(country, start)
        if series is not None and not series.empty:
            series.to_frame('cpi').to_parquet(cache_path)
        return series if series is not None else pd.Series(dtype=float)

    def get_pair_differentials(
        self, base_country: str, quote_country: str, start: str = '2015-01-01'
    ) -> pd.DataFrame:
        """
        Returns daily DataFrame with rate/cpi differentials for a pair.
        Used by the backtester for historical signal generation.
        """
        base_rates = self.get_rate_history(base_country, start)
        quote_rates = self.get_rate_history(quote_country, start)
        base_cpi = self.get_cpi_history(base_country, start)
        quote_cpi = self.get_cpi_history(quote_country, start)

        df = pd.DataFrame(index=pd.date_range(start, datetime.now(), freq='B'))
        df['base_rate'] = base_rates.reindex(df.index).ffill()
        df['quote_rate'] = quote_rates.reindex(df.index).ffill()
        df['base_cpi'] = base_cpi.reindex(df.index).ffill()
        df['quote_cpi'] = quote_cpi.reindex(df.index).ffill()

        df['rate_differential'] = df['base_rate'] - df['quote_rate']
        df['cpi_differential'] = df['base_cpi'] - df['quote_cpi']
        df['real_rate_base'] = df['base_rate'] - df['base_cpi']
        df['real_rate_quote'] = df['quote_rate'] - df['quote_cpi']
        df['real_rate_differential'] = df['real_rate_base'] - df['real_rate_quote']

        # Rate differential momentum: 1-month change
        df['rate_diff_momentum'] = df['rate_differential'].diff(21)

        return df.dropna(subset=['rate_differential'])

    # ------------------------------------------------------------------ #
    # Internal fetch                                                       #
    # ------------------------------------------------------------------ #

    def _fetch_macro(self, country: str) -> dict:
        rate = FALLBACK_RATES.get(country, 2.0)
        cpi = FALLBACK_CPI.get(country, 2.0)
        gdp = FALLBACK_GDP_GROWTH.get(country, 1.0)
        trajectory = RATE_TRAJECTORY.get(country, [0, 0, 0])

        if self._fred_ok:
            try:
                rate = self._fred_latest(FRED_RATES.get(country, ''), rate)
                cpi = self._fred_yoy(FRED_CPI.get(country, ''), cpi, country)
                if country in FRED_GDP:
                    gdp = self._fred_qoq(FRED_GDP[country], gdp)
            except Exception as e:
                logger.warning(f"FRED fetch for {country}: {e}")
        else:
            # Try yfinance rate proxy for US only (most reliable)
            if country == 'US':
                try:
                    import yfinance as yf
                    data = yf.download('^IRX', period='5d', progress=False, auto_adjust=True)
                    if not data.empty:
                        close = data['Close'].iloc[-1]
                        if hasattr(close, 'item'):
                            close = close.item()
                        rate = float(close)
                except Exception:
                    pass

        return {
            'country': country,
            'rate': rate,
            'cpi_yoy': cpi,
            'gdp_growth': gdp,
            'real_rate': rate - cpi,
            'rate_trajectory': trajectory,
            'as_of': datetime.now().strftime('%Y-%m-%d'),
        }

    def _fred_latest(self, series_id: str, fallback: float) -> float:
        if not series_id or not self._fred:
            return fallback
        try:
            s = self._fred.get_series(series_id)
            val = s.dropna().iloc[-1]
            return float(val)
        except Exception:
            return fallback

    def _fred_yoy(self, series_id: str, fallback: float, country: str = '') -> float:
        """Year-over-year % change. Handles monthly, quarterly, and pre-computed YoY series."""
        if not series_id or not self._fred:
            return fallback
        try:
            s = self._fred.get_series(series_id)
            s = s.dropna()
            if country in DIRECT_YOY_CPI:
                return float(s.iloc[-1])
            periods = 4 if country in QUARTERLY_CPI else 12
            if len(s) >= periods + 1:
                yoy = (s.iloc[-1] / s.iloc[-(periods + 1)] - 1) * 100
                return float(yoy)
        except Exception:
            pass
        return fallback

    def _fred_qoq(self, series_id: str, fallback: float) -> float:
        """Quarter-over-quarter annualized GDP growth."""
        if not series_id or not self._fred:
            return fallback
        try:
            s = self._fred.get_series(series_id)
            s = s.dropna()
            if len(s) >= 5:
                qoq = (s.iloc[-1] / s.iloc[-5] - 1) * 100
                return float(qoq)
        except Exception:
            pass
        return fallback

    def _fetch_rate_history(
        self, country: str, start: str
    ) -> Optional[pd.Series]:
        if self._fred_ok and country in FRED_RATES:
            try:
                s = self._fred.get_series(
                    FRED_RATES[country], observation_start=start
                )
                s = s.dropna().asfreq('B').ffill()
                s.name = f'{country}_rate'
                return s
            except Exception as e:
                logger.warning(f"FRED rate history {country}: {e}")

        # Fallback: flat series at current rate
        logger.info(f"Using fallback flat rate history for {country}")
        idx = pd.date_range(start, datetime.now(), freq='B')
        current = FALLBACK_RATES.get(country, 2.0)
        return pd.Series(current, index=idx, name=f'{country}_rate')

    def _fetch_cpi_history(
        self, country: str, start: str
    ) -> Optional[pd.Series]:
        if self._fred_ok and country in FRED_CPI:
            try:
                s = self._fred.get_series(
                    FRED_CPI[country], observation_start=start
                )
                s = s.dropna()
                if country in DIRECT_YOY_CPI:
                    yoy = s  # already YoY %
                else:
                    periods = 4 if country in QUARTERLY_CPI else 12
                    yoy = s.pct_change(periods) * 100
                    yoy = yoy.dropna()
                yoy = yoy.asfreq('B').ffill()
                yoy.name = f'{country}_cpi_yoy'
                return yoy
            except Exception as e:
                logger.warning(f"FRED CPI history {country}: {e}")

        idx = pd.date_range(start, datetime.now(), freq='B')
        current = FALLBACK_CPI.get(country, 2.0)
        return pd.Series(current, index=idx, name=f'{country}_cpi_yoy')
