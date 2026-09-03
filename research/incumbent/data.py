"""Daily adjusted closes for HYP-115/116 from yfinance (auto_adjust), cached once. The desk's
daily_universe cache (2014+) and Alpaca (2016+) cannot reach the 2007-2014 out-of-sample window."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "yf_daily_close_2005_2026.parquet"
CORE = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "EFA", "EEM", "XLF", "XLE"]
WIDER = ["XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "HYG", "LQD", "USO", "SLV", "VNQ", "XBI", "SMH", "KRE", "FXI", "EWZ", "EWJ", "GDX", "UNG"]


def closes(start: str = "2005-01-01", end: str = "2026-07-17") -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import yfinance as yf
    d = yf.download(CORE + WIDER, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    d.to_parquet(CACHE)
    return d
