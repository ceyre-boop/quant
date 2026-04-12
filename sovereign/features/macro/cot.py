"""
Sovereign Trading Intelligence -- COT (Commitment of Traders) Features
Phase 1 Fix: Real CFTC data source added.

Primary:  CFTC public reporting API (Disaggregated futures)
Fallback: Nasdaq Data Link (requires NASDAQ_API_KEY)
Cache:    data/cache/cot_{symbol}.parquet — refresh weekly
"""

import os
import logging
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# CFTC public reporting OData endpoint
CFTC_BASE = "https://publicreporting.cftc.gov/api/odata/v1/"

# Map our symbols to CFTC market names (partial match)
SYMBOL_MAP = {
    "NQ": "E-MINI NASDAQ-100",
    "ES": "E-MINI S&P 500",
    "ZN": "10-YEAR U.S. TREASURY NOTES",
    "GC": "GOLD",
    "CL": "CRUDE OIL",
}

COT_CACHE_DIR = Path("data/cache")


def _cache_path(symbol: str) -> Path:
    return COT_CACHE_DIR / f"cot_{symbol}.parquet"


def _is_cache_fresh(symbol: str, max_age_days: int = 7) -> bool:
    p = _cache_path(symbol)
    if not p.exists():
        return False
    age = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).days
    return age < max_age_days


def _fetch_cftc_cot(symbol: str) -> pd.DataFrame:
    """
    Fetch COT disaggregated futures data from CFTC public API.
    Parses commercial long/short positions.
    """
    market_name = SYMBOL_MAP.get(symbol.upper(), symbol)
    logger.info(f"Fetching CFTC COT for {symbol} ({market_name})...")

    try:
        url = (
            CFTC_BASE
            + "TriWeeklyReports/ExcelLayoutCombined"
            + f"?$filter=contains(Market_and_Exchange_Names,'{market_name}')"
            + "&$select=Report_Date_as_MM_DD_YYYY,Comm_Positions_Long_All,Comm_Positions_Short_All"
            + "&$orderby=Report_Date_as_MM_DD_YYYY desc"
            + "&$top=300"
            + "&$format=json"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("value", [])

        if not data:
            logger.warning(f"CFTC returned empty data for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.rename(columns={
            "Report_Date_as_MM_DD_YYYY": "date",
            "Comm_Positions_Long_All": "commercial_longs",
            "Comm_Positions_Short_All": "commercial_shorts",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df[["commercial_longs", "commercial_shorts"]] = df[
            ["commercial_longs", "commercial_shorts"]
        ].apply(pd.to_numeric, errors="coerce")
        df["net_commercial"] = df["commercial_longs"] - df["commercial_shorts"]

        COT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_cache_path(symbol))
        logger.info(f"COT cached for {symbol}: {len(df)} observations")
        return df

    except Exception as e:
        logger.warning(f"CFTC API fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def _fetch_nasdaq_cot(symbol: str) -> pd.DataFrame:
    """Fallback: Nasdaq Data Link CFTC dataset."""
    try:
        api_key = os.getenv("NASDAQ_API_KEY")
        if not api_key:
            return pd.DataFrame()
        import nasdaqdatalink
        nasdaqdatalink.ApiConfig.api_key = api_key
        dataset_map = {"NQ": "CFTC/NQ_FO_ALL", "ES": "CFTC/ES1_F_ALL"}
        dataset = dataset_map.get(symbol)
        if not dataset:
            return pd.DataFrame()
        df = nasdaqdatalink.get(dataset, rows=300)
        df = df[["Commercial Long", "Commercial Short"]].rename(columns={
            "Commercial Long": "commercial_longs",
            "Commercial Short": "commercial_shorts",
        })
        df["net_commercial"] = df["commercial_longs"] - df["commercial_shorts"]
        return df
    except Exception as e:
        logger.warning(f"Nasdaq Data Link COT fallback failed: {e}")
        return pd.DataFrame()


def _load_cot(symbol: str) -> pd.DataFrame:
    """Load from cache or fetch."""
    if _is_cache_fresh(symbol):
        return pd.read_parquet(_cache_path(symbol))
    df = _fetch_cftc_cot(symbol)
    if df.empty:
        df = _fetch_nasdaq_cot(symbol)
    if df.empty:
        logger.warning(f"COT: no data available for {symbol}")
    return df


def get_cot_zscore(symbol: str = "NQ", lookback_weeks: int = 156) -> float:
    """
    Z-score of net commercial positioning vs trailing 3-year window.
    Returns NaN if data unavailable.
    """
    df = _load_cot(symbol)
    if df.empty or "net_commercial" not in df.columns:
        return float("nan")

    net = df["net_commercial"].dropna()
    if len(net) < 20:
        return float("nan")

    window = net.tail(lookback_weeks)
    current = net.iloc[-1]
    mean = window.mean()
    std = window.std()
    if std == 0:
        return float("nan")
    return float((current - mean) / std)


def cot_zscore(macro_data: pd.DataFrame, lookback: int = 156) -> pd.Series:
    """Vectorised version for factor scanner — uses 'net_commercial' column if present."""
    try:
        from config.loader import params
    except Exception:
        pass

    if "net_commercial" in macro_data.columns:
        net = macro_data["net_commercial"]
        rolling_mean = net.rolling(lookback).mean()
        rolling_std = net.rolling(lookback).std()
        return (net - rolling_mean) / rolling_std.replace(0, float("nan"))
    return pd.Series(float("nan"), index=macro_data.index)


def compute_cot_features(macro_data: pd.DataFrame) -> pd.DataFrame:
    """Compute COT z-score for the factor scanner."""
    z = cot_zscore(macro_data)
    return pd.DataFrame({"cot_zscore": z}, index=macro_data.index)
