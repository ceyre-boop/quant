"""
Sovereign Trading Intelligence — Macro Data Feed
Phase 1: Data Foundation

FRED API (primary) with yfinance fallback for:
- Yield Curve: T10Y2Y, T10Y3M, DGS10, DGS2
- Volatility: VIXCLS (FRED) / ^VIX (yfinance)
- Credit: BAMLH0A0HYM2 (FRED) / HYG spread proxy (yfinance)

Returns daily series aligned to trading calendar.
Must support: get_macro_snapshot(date) → dict of current readings.

Design decisions (from research history):
- Yield curve VELOCITY and ACCELERATION matter more than level
  (Estrella-Mishkin 1996 recession probability)
- HYG-SPY divergence is a credit stress signal — need both series
- VIX z-score requires 252-day lookback for normalization
- M2 velocity is a Quantity Theory check for structural inflation
- FRED data has publication lag — must handle missing recent values
  gracefully via forward fill, NOT interpolation
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FRED series IDs
# ---------------------------------------------------------------------------
FRED_SERIES = {
    # Yield Curve
    't10y2y':       'T10Y2Y',       # 10Y-2Y Treasury spread
    't10y3m':       'T10Y3M',       # 10Y-3M Treasury spread
    'dgs10':        'DGS10',        # 10-Year Treasury rate
    'dgs2':         'DGS2',         # 2-Year Treasury rate
    
    # Volatility
    'vixcls':       'VIXCLS',       # CBOE VIX daily close
    
    # Credit
    'hyg_oas':      'BAMLH0A0HYM2', # ICE BofA US High Yield OAS
    
    # Monetary
    'm2':           'WM2NS',        # M2 Money Stock (weekly → daily via ffill)
    
    # Inflation
    't10yie':       'T10YIE',       # 10Y Breakeven Inflation Rate
}

# yfinance fallback symbols
YFINANCE_FALLBACKS = {
    'vix':  '^VIX',
    'tnx':  '^TNX',     # 10Y Treasury Yield * 10
    'hyg':  'HYG',      # High Yield Corporate Bond ETF
    'tlt':  'TLT',      # 20+ Year Treasury ETF
    'spy':  'SPY',      # For HYG-SPY divergence calculation
}


class MacroFeed:
    """
    Macro data feed for Sovereign.
    
    Primary: FRED API (requires FRED_API_KEY in .env)
    Fallback: yfinance (no key required, less precise)
    
    All series returned as daily, forward-filled to trading calendar.
    """
    
    # Cache duration — macro data changes once per day at most
    CACHE_HOURS = 6
    
    def __init__(self, fred_api_key: Optional[str] = None):
        load_dotenv()
        
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.fred = None
        self._fred_available = False
        
        if self.fred_api_key:
            try:
                from fredapi import Fred
                self.fred = Fred(api_key=self.fred_api_key)
                self._fred_available = True
                logger.info("MacroFeed: FRED API initialized")
            except Exception as e:
                logger.warning(f"MacroFeed: FRED init failed — {e}")
        else:
            logger.warning(
                "MacroFeed: No FRED_API_KEY found. "
                "Using yfinance fallback for all macro data. "
                "Add FRED_API_KEY to .env for production quality."
            )
        
        # Local cache
        self._cache: Dict[str, pd.Series] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Cache directory for parquet persistence
        self._cache_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'cache', 'macro'
        )
        self._cache_dir = os.path.abspath(self._cache_dir)
        os.makedirs(self._cache_dir, exist_ok=True)
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def get_series(
        self,
        series_key: str,
        start: Optional[str] = '2021-01-01',
        end: Optional[str] = None,
    ) -> pd.Series:
        """
        Get a single macro series by key.
        
        Args:
            series_key: Key from FRED_SERIES dict (e.g. 't10y2y')
            start:      Start date string 'YYYY-MM-DD'
            end:        End date string (default: today)
            
        Returns:
            pd.Series with DatetimeIndex, forward-filled to trading days.
        """
        cache_key = f"{series_key}_{start}_{end}"
        
        # Check memory cache
        if cache_key in self._cache:
            cache_time = self._cache_timestamps.get(cache_key)
            if cache_time and (
                datetime.now() - cache_time
            ).total_seconds() < self.CACHE_HOURS * 3600:
                return self._cache[cache_key]
        
        # Try FRED first
        if self._fred_available and series_key in FRED_SERIES:
            data = self._fetch_from_fred(series_key, start, end)
            if data is not None and not data.empty:
                self._cache[cache_key] = data
                self._cache_timestamps[cache_key] = datetime.now()
                return data
        
        # Fallback to yfinance
        data = self._fetch_from_yfinance(series_key, start, end)
        if data is not None and not data.empty:
            self._cache[cache_key] = data
            self._cache_timestamps[cache_key] = datetime.now()
        
        return data if data is not None else pd.Series(dtype=float)
    
    def get_all_series(
        self,
        start: str = '2021-01-01',
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get all macro series as a single aligned DataFrame.
        Forward-filled to common trading calendar.
        """
        all_series = {}
        
        for key in FRED_SERIES:
            try:
                s = self.get_series(key, start=start, end=end)
                if s is not None and not s.empty:
                    all_series[key] = s
            except Exception as e:
                logger.warning(f"Failed to fetch {key}: {e}")
        
        if not all_series:
            logger.error("No macro series retrieved")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_series)
        df = df.ffill()  # Forward fill — NEVER interpolate macro data
        
        return df
    
    def get_macro_snapshot(self, date: Optional[datetime] = None) -> dict:
        """
        Get a snapshot of all macro readings as of a specific date.
        
        This is the input to the Kimi/Petroulas fault detector.
        
        Returns dict with all current macro readings:
        - Yield curve: spread, velocity, acceleration
        - VIX: level and z-score
        - Credit: OAS spread or HYG proxy
        - Monetary: M2 level and velocity
        - Inflation: 10Y breakeven rate
        """
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else str(date)
        
        # Fetch all series with enough lookback for z-scores
        lookback_start = (
            pd.Timestamp(date_str) - timedelta(days=400)
        ).strftime('%Y-%m-%d')
        
        macro_df = self.get_all_series(start=lookback_start, end=date_str)
        
        if macro_df.empty:
            logger.warning("No macro data available for snapshot")
            return {}
        
        # Get the row closest to (but not after) the requested date
        target = pd.Timestamp(date_str)
        available = macro_df.index[macro_df.index <= target]
        if available.empty:
            logger.warning(f"No macro data available on or before {date_str}")
            return {}
        
        latest_date = available[-1]
        latest = macro_df.loc[latest_date]
        
        snapshot = {}
        
        # ── Yield Curve ──────────────────────────────────────────────
        t10y2y = latest.get('t10y2y', np.nan)
        snapshot['t10y_t2y_spread'] = t10y2y
        
        # Velocity: 5-day rate of change
        if 't10y2y' in macro_df.columns and len(macro_df) >= 10:
            t10y2y_series = macro_df['t10y2y'].dropna()
            if len(t10y2y_series) >= 6:
                snapshot['t10y_t2y_spread_5d'] = t10y2y_series.iloc[-6]
            if len(t10y2y_series) >= 21:
                snapshot['t10y_t2y_spread_20d'] = t10y2y_series.iloc[-21]
        
        snapshot['dgs10'] = latest.get('dgs10', np.nan)
        snapshot['dgs2'] = latest.get('dgs2', np.nan)
        snapshot['t10y3m'] = latest.get('t10y3m', np.nan)
        
        # ── VIX ──────────────────────────────────────────────────────
        vix = latest.get('vixcls', np.nan)
        snapshot['vix_level'] = vix
        
        if 'vixcls' in macro_df.columns:
            vix_series = macro_df['vixcls'].dropna()
            if len(vix_series) >= 252:
                vix_mean = vix_series.tail(252).mean()
                vix_std = vix_series.tail(252).std()
                if vix_std > 0:
                    snapshot['vix_zscore'] = (vix - vix_mean) / vix_std
                else:
                    snapshot['vix_zscore'] = 0.0
            else:
                snapshot['vix_zscore'] = 0.0
        
        # ── Credit ───────────────────────────────────────────────────
        snapshot['hyg_oas'] = latest.get('hyg_oas', np.nan)
        
        # ── Monetary ─────────────────────────────────────────────────
        snapshot['m2'] = latest.get('m2', np.nan)
        if 'm2' in macro_df.columns:
            m2_series = macro_df['m2'].dropna()
            if len(m2_series) >= 52:
                # Year-over-year M2 growth rate
                snapshot['m2_yoy'] = (
                    m2_series.iloc[-1] / m2_series.iloc[-52] - 1
                ) * 100
        
        # ── Inflation Breakeven ──────────────────────────────────────
        snapshot['inflation_10y_breakeven'] = latest.get('t10yie', np.nan)
        
        # ── Computed Metrics ─────────────────────────────────────────
        dgs10 = snapshot.get('dgs10', np.nan)
        breakeven = snapshot.get('inflation_10y_breakeven', np.nan)
        if not np.isnan(dgs10) and not np.isnan(breakeven):
            snapshot['real_10y_yield'] = dgs10 - breakeven
        
        snapshot['snapshot_date'] = str(latest_date.date())
        snapshot['data_source'] = 'FRED' if self._fred_available else 'yfinance'
        
        return snapshot
    
    # ------------------------------------------------------------------
    # Internal: FRED fetching
    # ------------------------------------------------------------------
    
    def _fetch_from_fred(
        self,
        series_key: str,
        start: Optional[str],
        end: Optional[str],
    ) -> Optional[pd.Series]:
        """Fetch a series from FRED API."""
        fred_id = FRED_SERIES.get(series_key)
        if not fred_id or not self.fred:
            return None
        
        try:
            data = self.fred.get_series(
                fred_id,
                observation_start=start,
                observation_end=end,
            )
            
            if data is not None and not data.empty:
                data = data.dropna()
                data.name = series_key
                data.index = pd.DatetimeIndex(data.index)
                # Forward fill to daily — FRED has gaps on weekends/holidays
                # and some series are weekly (e.g. M2)
                data = data.asfreq('B').ffill()
                logger.info(
                    f"FRED {fred_id}: {len(data)} observations "
                    f"({data.index.min().date()} to {data.index.max().date()})"
                )
                return data
                
        except Exception as e:
            logger.warning(f"FRED fetch failed for {fred_id}: {e}")
        
        return None
    
    # ------------------------------------------------------------------
    # Internal: yfinance fallback
    # ------------------------------------------------------------------
    
    def _fetch_from_yfinance(
        self,
        series_key: str,
        start: Optional[str],
        end: Optional[str],
    ) -> Optional[pd.Series]:
        """
        Fetch approximation of a macro series from yfinance.
        
        Mappings:
        - vixcls   → ^VIX close
        - dgs10    → ^TNX / 10 (TNX is yield * 10)
        - hyg_oas  → HYG-SPY return divergence (proxy only)
        - t10y2y   → computed from ^TNX and ^TWO if available
        
        These are APPROXIMATIONS. FRED is the ground truth.
        """
        import yfinance as yf
        
        end_str = end or datetime.now().strftime('%Y-%m-%d')
        start_str = start or '2021-01-01'
        
        try:
            if series_key == 'vixcls':
                data = yf.download(
                    '^VIX', start=start_str, end=end_str,
                    progress=False, auto_adjust=True
                )
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    result = data['Close'].rename('vixcls')
                    logger.info(f"yfinance ^VIX: {len(result)} observations")
                    return result
            
            elif series_key == 'dgs10':
                data = yf.download(
                    '^TNX', start=start_str, end=end_str,
                    progress=False, auto_adjust=True
                )
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    # ^TNX is yield * 10 — but as of recent versions
                    # yfinance returns the actual yield
                    result = data['Close'].rename('dgs10')
                    # If values are > 50, they're in the old TNX format
                    if result.mean() > 50:
                        result = result / 10
                    logger.info(f"yfinance ^TNX: {len(result)} observations")
                    return result
            
            elif series_key in ('hyg_oas', 'hyg'):
                # Use HYG price as a proxy — rising HYG = tightening spreads
                data = yf.download(
                    'HYG', start=start_str, end=end_str,
                    progress=False, auto_adjust=True
                )
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    result = data['Close'].rename('hyg_oas')
                    logger.info(f"yfinance HYG: {len(result)} observations")
                    return result
            
            elif series_key == 't10y2y':
                # Approximate from TNX - need 2Y yield too
                # yfinance doesn't have a good 2Y symbol — return empty
                logger.warning(
                    "t10y2y not available via yfinance fallback. "
                    "Set FRED_API_KEY for yield curve data."
                )
                return None
            
            elif series_key == 'm2':
                logger.warning(
                    "M2 not available via yfinance fallback. "
                    "Set FRED_API_KEY for monetary data."
                )
                return None
            
            elif series_key == 't10yie':
                logger.warning(
                    "10Y breakeven not available via yfinance fallback. "
                    "Set FRED_API_KEY for inflation data."
                )
                return None
            
            else:
                logger.debug(
                    f"No yfinance fallback defined for {series_key}"
                )
                return None
                
        except Exception as e:
            logger.warning(f"yfinance fallback failed for {series_key}: {e}")
        
        return None
    
    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    
    def save_snapshot(self, snapshot: dict, path: Optional[str] = None):
        """Save a macro snapshot to disk for reproducibility."""
        if path is None:
            date_str = snapshot.get('snapshot_date', 'unknown')
            path = os.path.join(self._cache_dir, f"snapshot_{date_str}.json")
        
        import json
        
        # Convert numpy types for JSON serialization
        clean = {}
        for k, v in snapshot.items():
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, (np.ndarray,)):
                clean[k] = v.tolist()
            else:
                clean[k] = v
        
        with open(path, 'w') as f:
            json.dump(clean, f, indent=2, default=str)
        
        logger.info(f"Macro snapshot saved: {path}")


# ======================================================================
# CLI entry point
# ======================================================================

def main():
    """Fetch and display current macro snapshot."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    
    feed = MacroFeed()
    
    print("\n" + "=" * 70)
    print("SOVEREIGN DATA FOUNDATION — MACRO FEED")
    print("=" * 70 + "\n")
    
    # Fetch all series
    print("Fetching all macro series from 2021-01-01...")
    df = feed.get_all_series(start='2021-01-01')
    
    if not df.empty:
        print(f"\nSeries retrieved: {len(df.columns)}")
        print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"Total observations: {len(df)}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nLatest values:")
        print(df.tail(3).to_string())
    else:
        print("⚠ No macro data retrieved")
    
    # Get snapshot
    print("\n" + "-" * 70)
    print("CURRENT MACRO SNAPSHOT")
    print("-" * 70)
    
    snapshot = feed.get_macro_snapshot()
    for k, v in sorted(snapshot.items()):
        if isinstance(v, float):
            print(f"  {k:30s}: {v:>10.4f}")
        else:
            print(f"  {k:30s}: {v}")
    
    # Save snapshot
    feed.save_snapshot(snapshot)
    
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
