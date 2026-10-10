"""Daily adjusted closes with a parquet cache. yfinance is imported lazily so importing
this module stays network-free (pattern: sovereign/research/vrp/data_loader.py)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "yf_daily"


def _fetch(ticker: str) -> pd.Series:
    import yfinance as yf  # lazy
    h = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
    if h.empty:
        raise RuntimeError(f"yfinance returned nothing for {ticker}")
    s = h["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s.name = ticker
    return s.astype(float)


def load_daily(ticker: str, cache: Path = CACHE, max_age_hours: float = 20.0) -> pd.Series:
    """Adjusted close series indexed by naive date. Refreshes when the cache is stale."""
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{ticker.upper()}.parquet"
    if f.exists():
        age = datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)
        if age < timedelta(hours=max_age_hours):
            return pd.read_parquet(f)["close"]
    s = _fetch(ticker)
    s.to_frame("close").to_parquet(f)
    return s


def price_on(series: pd.Series, d: str | date, how: str = "ffill") -> float:
    """Price on date d, or the last prior close if d is not a trading day."""
    ts = pd.Timestamp(d).normalize()
    s = series[series.index <= ts] if how == "ffill" else series[series.index >= ts]
    if s.empty:
        raise KeyError(f"no price for {series.name} at {d}")
    return float(s.iloc[-1] if how == "ffill" else s.iloc[0])
