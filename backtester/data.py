"""Unified data layer — minute & daily bars from Alpaca / Polygon / local cache.

Design note (verified 2026-07-17): yfinance 1-minute history is limited to the
last ~7 days, so it CANNOT serve the 2025-26 event minute bars this repo studies.
The real historical minute source is Alpaca SIP, already cached under
data/research/gapper/cache/alpaca/YYYY-MM-DD.json.gz. `source="auto"` therefore
tries: local gz cache -> parquet cache -> Alpaca -> Polygon -> yfinance(recent).
Every fetched (ticker, date) is memoised to data/cache/minute_bars/*.parquet so
the same bar is never pulled twice.

Bars are timezone-aware in US/Eastern with a 'time' column of 'HH:MM' strings for
cheap entry/exit alignment, plus float o/h/l/c and int volume.
"""
from __future__ import annotations

import gzip
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GZ_CACHE = REPO / "data/research/gapper/cache/alpaca"
PARQUET_CACHE = REPO / "data/cache/minute_bars"
ET = ZoneInfo("America/New_York")
PARQUET_CACHE.mkdir(parents=True, exist_ok=True)

_BAR_COLS = ["time", "open", "high", "low", "close", "volume"]


def _env(key: str) -> str | None:
    env_path = REPO / ".env"
    if not env_path.exists():
        return os.environ.get(key)
    for line in env_path.read_text().splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip('"')
    return os.environ.get(key)


def _bars_from_alpaca_records(records: list[dict]) -> pd.DataFrame:
    """Alpaca bar dicts (o/h/l/c/v/t ISO-UTC) -> normalised ET DataFrame."""
    rows = []
    for b in records:
        t_utc = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
        t_et = t_utc.astimezone(ET)
        rows.append((t_et.strftime("%H:%M"), float(b["o"]), float(b["h"]),
                     float(b["l"]), float(b["c"]), int(b.get("v", 0))))
    df = pd.DataFrame(rows, columns=_BAR_COLS)
    return df.sort_values("time").reset_index(drop=True)


def _from_gz_cache(ticker: str, date: str) -> pd.DataFrame | None:
    fp = GZ_CACHE / f"{date}.json.gz"
    if not fp.exists():
        return None
    day = json.load(gzip.open(fp))
    recs = (day.get("intraday") or {}).get(ticker)
    if not recs:
        return None
    return _bars_from_alpaca_records(recs)


def _parquet_path(ticker: str, date: str) -> Path:
    return PARQUET_CACHE / f"{ticker}_{date}.parquet"


def _from_parquet(ticker: str, date: str) -> pd.DataFrame | None:
    p = _parquet_path(ticker, date)
    if p.exists():
        return pd.read_parquet(p)
    return None


def _alpaca_fetch(ticker: str, date: str) -> pd.DataFrame | None:
    kid, sec = _env("ALPACA_API_KEY"), _env("ALPACA_SECRET_KEY")
    if not kid or not sec:
        return None
    start = datetime.fromisoformat(f"{date}T09:30:00").replace(tzinfo=ET) \
        .astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = datetime.fromisoformat(f"{date}T16:00:00").replace(tzinfo=ET) \
        .astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    q = urllib.parse.urlencode({"symbols": ticker, "timeframe": "1Min",
                                "start": start, "end": end, "adjustment": "all",
                                "feed": "sip", "limit": 10000})
    req = urllib.request.Request(
        f"https://data.alpaca.markets/v2/stocks/bars?{q}",
        headers={"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                recs = (json.loads(r.read()).get("bars") or {}).get(ticker) or []
            return _bars_from_alpaca_records(recs) if recs else None
        except Exception:
            time.sleep(1 + attempt)
    return None


def get_minute_bars(ticker: str, date: str, source: str = "auto") -> pd.DataFrame:
    """Return normalised minute bars for (ticker, date). Empty df if unavailable.

    date: 'YYYY-MM-DD'. Caches every successful non-cache fetch to parquet.
    """
    if source in ("auto", "csv"):
        for loader in (_from_parquet, _from_gz_cache):
            df = loader(ticker, date)
            if df is not None and len(df):
                return df
        if source == "csv":
            return pd.DataFrame(columns=_BAR_COLS)

    if source in ("auto", "alpaca"):
        df = _alpaca_fetch(ticker, date)
        if df is not None and len(df):
            df.to_parquet(_parquet_path(ticker, date))
            return df

    # Vendor-agnostic last resort (TICK-043). Ordering is deliberate: parquet ->
    # gz -> direct Alpaca -> adapter. Every day already covered by the first
    # three paths returns byte-identical bars to pre-TICK-043, so the v015
    # 0.6886 reconcile anchor cannot move. The adapter only fills days that
    # previously returned an EMPTY frame — it can add coverage, never restate
    # history.
    if source in ("auto", "adapter"):
        df = _adapter_fetch(ticker, date)
        if df is not None and len(df):
            df.to_parquet(_parquet_path(ticker, date))
            return df

    return pd.DataFrame(columns=_BAR_COLS)


def _adapter_fetch(ticker: str, date: str) -> pd.DataFrame | None:
    """Fetch via MarketDataAdapter and coerce to this module's frame contract.

    The adapter speaks UTC timestamps over a full window; this module speaks ET
    'HH:MM' strings over the RTH session. Converting here — rather than changing
    either contract — is what keeps the swap behaviour-preserving.
    """
    try:
        from sovereign.data.adapter import MarketDataAdapter
    except Exception:                                   # noqa: BLE001
        return None

    try:
        nxt = (datetime.fromisoformat(date).date().toordinal() + 1)
        end_date = datetime.fromordinal(nxt).date().isoformat()
        # use_cache=False: this module owns its own parquet cache above, and two
        # caches over one fetch is how stale data gets served twice.
        raw = MarketDataAdapter(use_cache=False).get_bars(
            ticker, date, end_date, timeframe="1min")
    except Exception:                                   # noqa: BLE001
        return None

    if raw is None or raw.empty:
        return None

    ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(ET)
    out = pd.DataFrame({
        "time": ts.dt.strftime("%H:%M"),
        "open": raw["open"].astype(float),
        "high": raw["high"].astype(float),
        "low": raw["low"].astype(float),
        "close": raw["close"].astype(float),
        "volume": raw["volume"].astype(float),
    })
    # Match the RTH window the direct Alpaca path requests (09:30–16:00 ET).
    out = out[(out["time"] >= "09:30") & (out["time"] < "16:00")]
    return out.reset_index(drop=True) if len(out) else None


def get_minute_range(ticker: str, start: str, end: str) -> dict:
    """Bulk-fetch 1-min bars for [start, end] (dates), paginated, and cache each
    day to parquet. Returns {date: DataFrame}. Days already cached are skipped;
    only the missing span is pulled from Alpaca. One request covers many days.
    """
    import pandas as _pd
    from datetime import datetime as _dt
    kid, sec = _env("ALPACA_API_KEY"), _env("ALPACA_SECRET_KEY")
    out: dict[str, pd.DataFrame] = {}
    if not kid or not sec:
        return out
    s_utc = datetime.fromisoformat(f"{start}T09:30:00").replace(tzinfo=ET) \
        .astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    e_utc = datetime.fromisoformat(f"{end}T16:00:00").replace(tzinfo=ET) \
        .astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    token = None
    recs: list[dict] = []
    for _ in range(500):  # page cap
        params = {"symbols": ticker, "timeframe": "1Min", "start": s_utc,
                  "end": e_utc, "adjustment": "all", "feed": "sip",
                  "limit": 10000}
        if token:
            params["page_token"] = token
        req = urllib.request.Request(
            f"https://data.alpaca.markets/v2/stocks/bars?"
            f"{urllib.parse.urlencode(params)}",
            headers={"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = json.loads(r.read())
        except Exception:
            time.sleep(1.0)
            continue
        recs.extend((payload.get("bars") or {}).get(ticker) or [])
        token = payload.get("next_page_token")
        if not token:
            break
    if not recs:
        return out
    # group by ET date, normalise, cache per day
    by_day: dict[str, list] = {}
    for b in recs:
        t_et = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET)
        by_day.setdefault(t_et.strftime("%Y-%m-%d"), []).append(b)
    for d, day_recs in by_day.items():
        df = _bars_from_alpaca_records(day_recs)
        if len(df):
            df.to_parquet(_parquet_path(ticker, d))
            out[d] = df
    return out


def get_daily_bars(ticker: str, start: str, end: str,
                   source: str = "auto") -> pd.DataFrame:
    """Daily OHLCV via yfinance (daily history is not 7-day-capped)."""
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame(columns=["date", *_BAR_COLS[1:]])
    df = yf.download(ticker, start=start, end=end, progress=False,
                     auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", *_BAR_COLS[1:]])
    df = df.reset_index()
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d"),
        "open": df["Open"].to_numpy().ravel(),
        "high": df["High"].to_numpy().ravel(),
        "low": df["Low"].to_numpy().ravel(),
        "close": df["Close"].to_numpy().ravel(),
        "volume": df["Volume"].to_numpy().ravel(),
    })
    return out
