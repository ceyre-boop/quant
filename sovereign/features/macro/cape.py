"""
Sovereign Trading Intelligence -- Shiller CAPE Features
Phase 1 Fix: Real data source added.

Primary:  Robert Shiller's public dataset (monthly, cached locally)
Fallback: SPY trailing P/E via yfinance
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import io

logger = logging.getLogger(__name__)

SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
CAPE_CACHE = Path("data/cache/shiller_cape.parquet")


def _fetch_shiller_cape() -> pd.DataFrame:
    """Download Shiller IE data and extract CAPE column."""
    logger.info("Fetching Shiller CAPE from Yale dataset...")
    try:
        resp = requests.get(SHILLER_URL, timeout=30)
        resp.raise_for_status()
        # Sheet "Data" — CAPE is column index 15 (0-based) in the xls
        xls = pd.read_excel(
            io.BytesIO(resp.content),
            sheet_name="Data",
            header=7,  # data starts row 8
            usecols=[0, 15],  # Date, CAPE
            names=["date", "CAPE"],
        )
        xls = xls.dropna(subset=["CAPE"])
        # Date column is fractional year e.g. 1871.01
        xls["date"] = pd.to_datetime(
            xls["date"].astype(str).str[:7].str.replace(".", "-", regex=False) + "-01",
            errors="coerce",
            format="%Y-%m-%d",
        )
        xls = xls.dropna(subset=["date"]).set_index("date").sort_index()
        CAPE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        xls.to_parquet(CAPE_CACHE)
        logger.info(f"Shiller CAPE cached: {len(xls)} observations")
        return xls
    except Exception as e:
        logger.warning(f"Shiller CAPE fetch failed: {e}")
        return pd.DataFrame()


def _load_cape_data() -> pd.DataFrame:
    """Load from cache or fetch fresh."""
    if CAPE_CACHE.exists():
        age_days = (
            pd.Timestamp.now() - pd.Timestamp(CAPE_CACHE.stat().st_mtime, unit="s")
        ).days
        if age_days < 30:
            return pd.read_parquet(CAPE_CACHE)
    return _fetch_shiller_cape()


def get_cape_zscore(lookback_years: int = 10) -> float:
    """
    Returns the z-score of the current CAPE vs. its own trailing history.
    Primary: Shiller dataset. Fallback: SPY trailing PE vs hardcoded 130yr params.
    """
    try:
        from config.loader import params
        cape_mean = params["petroulas"]["cape_mean"]
        cape_std = params["petroulas"]["cape_std"]
    except Exception:
        cape_mean, cape_std = 16.8, 6.5

    try:
        cape_df = _load_cape_data()
        if not cape_df.empty and "CAPE" in cape_df.columns:
            cape = cape_df["CAPE"]
            window = lookback_years * 12
            mean = cape.rolling(window).mean().iloc[-1]
            std = cape.rolling(window).std().iloc[-1]
            current = cape.iloc[-1]
            if pd.notna(current) and pd.notna(std) and std > 0:
                return float((current - mean) / std)
            # Fallback to 130yr params
            return float((current - cape_mean) / cape_std)
    except Exception as e:
        logger.warning(f"CAPE from Shiller failed: {e}. Trying yfinance fallback.")

    # yfinance fallback
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        pe = spy.info.get("trailingPE", float("nan"))
        if not pd.isna(pe):
            return float((pe - cape_mean) / cape_std)
    except Exception as e:
        logger.warning(f"CAPE yfinance fallback failed: {e}")

    return float("nan")


def shiller_cape_zscore(macro_data: pd.DataFrame) -> pd.Series:
    """
    Vectorised version for the factor scanner.
    Uses hardcoded 130-year params as per MASTER_BUILD_PLAN.
    """
    try:
        from config.loader import params
        cape_mean = params["petroulas"]["cape_mean"]
        cape_std = params["petroulas"]["cape_std"]
    except Exception:
        cape_mean, cape_std = 16.8, 6.5

    if "shiller_cape" in macro_data.columns:
        cape = macro_data["shiller_cape"]
        return (cape - cape_mean) / cape_std
    return pd.Series(float("nan"), index=macro_data.index)


def cape_percentile(macro_data: pd.DataFrame) -> pd.Series:
    """Percentile rank in rolling 10-year window."""
    if "shiller_cape" in macro_data.columns:
        cape = macro_data["shiller_cape"]
        return cape.rolling(252 * 10, min_periods=252).rank(pct=True)
    return pd.Series(float("nan"), index=macro_data.index)


def compute_cape_features(macro_data: pd.DataFrame) -> pd.DataFrame:
    """Compute CAPE z-score and percentile for the factor scanner."""
    z = shiller_cape_zscore(macro_data)
    p = cape_percentile(macro_data)
    return pd.DataFrame(
        {"shiller_cape_zscore": z, "cape_percentile": p},
        index=macro_data.index,
    )
