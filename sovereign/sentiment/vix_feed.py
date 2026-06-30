"""sovereign/sentiment/vix_feed.py — VIX daily close + momentum + regime for the sentiment board.

Downloads ^VIX daily (config sentiment.vix.ticker) from sentiment.macro_start, computes
vix_momentum = close − close_(N)d_ago (N = sentiment.vix.momentum_lookback_days), classifies the regime
from the config cutoffs, and upserts to sentiment_vix_daily. Idempotent (INSERT OR REPLACE).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from config.loader import params
from sovereign.sentiment.store import connect, upsert


def classify_regime(level, cfg: dict | None = None) -> str | None:
    """VIX regime from the configured cutoffs: LOW <low_max, NORMAL [low_max,normal_max],
    HIGH (normal_max,spike_min], SPIKE >spike_min. Returns None for a missing level."""
    if level is None or (isinstance(level, float) and math.isnan(level)):
        return None
    cfg = cfg or params["sentiment"]["vix_regime"]
    if level < cfg["low_max"]:
        return "LOW"
    if level <= cfg["normal_max"]:
        return "NORMAL"
    if level <= cfg["spike_min"]:
        return "HIGH"
    return "SPIKE"


def update(con=None, start: str | None = None) -> dict:
    """Download ^VIX history and upsert daily close/momentum/regime. Returns coverage."""
    cfg = params["sentiment"]
    ticker = cfg["vix"]["ticker"]
    lookback = int(cfg["vix"].get("momentum_lookback_days", 5))
    start = start or cfg.get("macro_start", "2015-01-01")
    own = con is None
    con = con or connect()
    now = datetime.now(timezone.utc)
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, interval="1d", auto_adjust=True, progress=False)
    except Exception as exc:
        print(f"  [vix] {ticker}: FETCH FAILED ({type(exc).__name__}: {exc})")
        if own:
            con.close()
        return {"rows": 0}
    if df is None or df.empty:
        if own:
            con.close()
        return {"rows": 0}
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    close = df["Close"].astype(float)
    out = pd.DataFrame({
        "date": pd.to_datetime(close.index).date,
        "vix_close": close.to_numpy(dtype=float),
        "vix_5d_ago": close.shift(lookback).to_numpy(dtype=float),
    })
    out["vix_momentum"] = out["vix_close"] - out["vix_5d_ago"]
    out["vix_regime"] = [classify_regime(x) for x in out["vix_close"]]
    out["fetched_at"] = now
    upsert(con, "sentiment_vix_daily", out, ["date"])
    if own:
        con.close()
    return {"rows": int(len(out)), "start": str(out["date"].min()), "end": str(out["date"].max())}
