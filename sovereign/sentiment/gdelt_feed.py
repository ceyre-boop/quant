"""sovereign/sentiment/gdelt_feed.py — GDELT 2.0 texture feeder (free, keyless, full history).

The GDELT DOC API (mode=timelinetone / timelinevolraw) returns a daily series of average article TONE
[-100,+100] and raw article VOLUME for a free-text query, aggregated across thousands of sources/day —
robust, low-noise, ~2017→present, no API key. This replaces the tier-limited NewsAPI as the board's
news dimension (a full-history "texture": tone level + 5d tone momentum + attention/volume).

RATE LIMIT: GDELT throttles to ~1 request / 5s and returns a PLAIN-TEXT message (not JSON) when exceeded.
This feeder spaces every call by `rate_limit_sec`, detects the non-JSON throttle body, and backs off.
Free-data, ingestion-only: pulls public tone IN, writes to the local DuckDB. No key, no egress.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config.loader import params
from sovereign.sentiment.store import connect, upsert

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def normalize_tone(tone_raw: float) -> float:
    """Map GDELT tone [-100, 100] → [-1, 1]."""
    return float(tone_raw) / 100.0


def parse_timeline(payload: dict) -> list:
    """Extract the [{date, value}] points from a GDELT timeline* JSON payload ([] if absent/malformed)."""
    if not isinstance(payload, dict):
        return []
    tl = payload.get("timeline") or []
    if tl and isinstance(tl[0], dict) and isinstance(tl[0].get("data"), list):
        return tl[0]["data"]
    if isinstance(payload.get("data"), list):     # tolerate a flat shape
        return payload["data"]
    return []


def _gdelt_get(query: str, mode: str, sd: str, ed: str, cfg: dict) -> list:
    """One DOC API call → list of {date, value}. Spaces by rate_limit_sec, retries on the throttle text."""
    import requests
    rate = float(cfg.get("rate_limit_sec", 5))
    for attempt in range(int(cfg.get("max_retries", 3)) + 1):
        time.sleep(rate)                          # always space ≥ the 5s limit (bounded; no escalation)
        try:
            r = requests.get(DOC_API, timeout=float(cfg.get("timeout_sec", 30)), params={
                "query": query, "mode": mode, "format": "json",
                "startdatetime": sd, "enddatetime": ed,
            })
            body = (r.text or "").strip()
            if not body.startswith("{"):          # GDELT throttle / error → plain text, retry
                continue
            return parse_timeline(r.json())
        except Exception as exc:
            print(f"  [gdelt] {mode} '{query[:24]}': {type(exc).__name__}: {exc}")
    print(f"  [gdelt] {mode} '{query[:24]}': throttled/failed after retries")
    return []


def _timeline_to_df(points: list, value_col: str) -> pd.DataFrame:
    if not points:
        return pd.DataFrame(columns=["date", value_col])
    df = pd.DataFrame(points)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"]).rename(columns={"value": value_col})
    return df.groupby("date", as_index=False)[value_col].mean()     # dedupe sub-daily → one row/day


def update(con=None, start: str | None = None) -> dict:
    """Fetch tone + volume for every configured pair and upsert sentiment_gdelt_daily. Returns coverage."""
    cfg = params["sentiment"]["gdelt"]
    start = start or cfg.get("start", "2017-01-01")
    own = con is None
    con = con or connect()
    now = datetime.now(timezone.utc)
    sd = pd.Timestamp(start).strftime("%Y%m%d%H%M%S")
    ed = now.strftime("%Y%m%d%H%M%S")
    mom = int(cfg.get("tone_momentum_days", 5))
    frames, coverage = [], {}
    for pair, query in cfg["pairs"].items():
        tone = _timeline_to_df(_gdelt_get(query, "timelinetone", sd, ed, cfg), "tone_raw")
        vol = _timeline_to_df(_gdelt_get(query, "timelinevolraw", sd, ed, cfg), "vol_raw")
        if tone.empty:
            coverage[pair] = {"rows": 0}
            continue
        df = tone.merge(vol, on="date", how="left").sort_values("date").reset_index(drop=True)
        df["pair"] = pair
        df["tone_score"] = df["tone_raw"] / 100.0
        df["tone_5d"] = df["tone_score"] - df["tone_score"].shift(mom)
        df["volume"] = np.log1p(df["vol_raw"].clip(lower=0)) if "vol_raw" in df else np.nan
        df["fetched_at"] = now
        frames.append(df[["date", "pair", "tone_raw", "tone_score", "tone_5d", "volume", "fetched_at"]])
        coverage[pair] = {"rows": int(len(df)), "start": str(df["date"].min()), "end": str(df["date"].max())}
    if frames:
        gdelt_df = pd.concat(frames, ignore_index=True)
        upsert(con, "sentiment_gdelt_daily", gdelt_df, ["date", "pair"])
    if own:
        con.close()
    return coverage
