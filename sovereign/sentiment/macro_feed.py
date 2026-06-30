"""sovereign/sentiment/macro_feed.py — FRED macro ingestion for the sentiment board.

Pulls the configured FRED series (config/parameters.yml :: sentiment.fred.series) from the configured
start (sentiment.macro_start, default 2015-01-01), computes trailing 1d & 5d deltas (no forward-look),
and upserts long-format rows to sentiment_macro_daily. Idempotent (INSERT OR REPLACE).

Series the board consumes downstream: T10Y2Y (yield curve), BAMLH0A0HYM2 (HY spread), T10YIE
(inflation expectations); DFF and VIXCLS are stored as macro context.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from config.loader import params
from sovereign.sentiment.store import connect, env_key, upsert


def fetch_series(series_id: str, start: str) -> pd.Series:
    """One FRED series from `start` to latest, as a float Series indexed by date. Empty on failure."""
    try:
        from fredapi import Fred
        fred = Fred(api_key=env_key("FRED_API_KEY"))
        raw = fred.get_series(series_id, observation_start=start)
        if raw is None or len(raw) == 0:
            return pd.Series(dtype=float)
        return pd.Series(raw).astype(float).dropna().sort_index()
    except Exception as exc:  # network / key / unknown series — skip, don't crash the run
        print(f"  [macro] {series_id}: FETCH FAILED ({type(exc).__name__}: {exc})")
        return pd.Series(dtype=float)


def update(con=None, start: str | None = None) -> dict:
    """Fetch all configured FRED series and upsert to sentiment_macro_daily. Returns per-series coverage."""
    cfg = params["sentiment"]
    series_ids = cfg["fred"]["series"]
    w1, w5 = cfg["fred"].get("delta_windows", [1, 5])
    start = start or cfg.get("macro_start", "2015-01-01")
    own = con is None
    con = con or connect()
    now = datetime.now(timezone.utc)
    frames, coverage = [], {}
    for sid in series_ids:
        s = fetch_series(sid, start)
        if s.empty:
            coverage[sid] = {"rows": 0}
            continue
        df = pd.DataFrame({
            "date": pd.to_datetime(s.index).date,
            "series": sid,
            "value": s.to_numpy(dtype=float),
            "delta_1d": s.diff(w1).to_numpy(dtype=float),
            "delta_5d": s.diff(w5).to_numpy(dtype=float),
            "fetched_at": now,
        })
        frames.append(df)
        coverage[sid] = {"rows": int(len(s)), "start": str(s.index.min().date()), "end": str(s.index.max().date())}
    if frames:
        macro_df = pd.concat(frames, ignore_index=True)
        upsert(con, "sentiment_macro_daily", macro_df, ["date", "series"])
    if own:
        con.close()
    return coverage
