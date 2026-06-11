"""ES/NQ data store — Databento GLBX.MDP3 pulls, parquet caches, daily session table.

Sandbox-local: imports only sovereign.futures plumbing + own config. Fail loud always:
a missing key, package, cache, or symbol column raises — never silently degrades
(silent failures hid forex bugs for weeks; non-negotiable #7).

Caches (data/es_nq/):
  nq_globex_1min.parquet      full ~23h Globex 1-min bars, mapped contract symbol kept
  nq_historical_5min.parquet  RTH-only 5-min bars (the brief's named cache)
  nq_daily.parquet            one row per US session date (levels, overnight, rolls)
  aux_daily.parquet           ^VIX / ^N225 / ^GDAXI daily closes (yfinance)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from sovereign.es_nq.config import es_nq_params

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "es_nq"
RAW_1MIN = DATA_DIR / "nq_globex_1min.parquet"
RTH_5MIN = DATA_DIR / "nq_historical_5min.parquet"
DAILY = DATA_DIR / "nq_daily.parquet"
AUX_DAILY = DATA_DIR / "aux_daily.parquet"

AUX_TICKERS = {"vix": "^VIX", "nikkei": "^N225", "dax": "^GDAXI"}
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _databento_key() -> str:
    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if not key:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("DATABENTO_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("FATAL: DATABENTO_API_KEY not set (env or .env). "
                         "Sign up at databento.com ($125 free credit) and add it.")
    return key


def pull_globex_history(start: str, end: str, symbol: Optional[str] = None,
                        chunk_days: int = 180) -> pd.DataFrame:
    """Full-session (Globex) 1-min OHLCV for the continuous NQ front month.

    Chunked requests (Databento bills per request size; chunks keep each one
    bounded and restartable). Keeps the mapped raw-contract `symbol` column —
    the roll detector depends on it and FAILS LOUD if it is missing.
    """
    try:
        import databento as db
    except ImportError:
        raise SystemExit("FATAL: databento not installed. Run: pip3 install databento")
    sym = symbol or es_nq_params()["meta"]["research_symbol"]
    client = db.Historical(_databento_key())

    frames: list[pd.DataFrame] = []
    cur = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    stop = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    while cur < stop:
        nxt = min(cur + timedelta(days=chunk_days), stop)
        store = client.timeseries.get_range(
            dataset="GLBX.MDP3", schema="ohlcv-1m", stype_in="continuous",
            symbols=[sym], start=cur.strftime("%Y-%m-%dT%H:%M:%S"),
            end=nxt.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        df = store.to_df()
        if df is not None and len(df):
            frames.append(df)
        print(f"  pulled {cur.date()} → {nxt.date()}: "
              f"{0 if df is None else len(df):,} rows")
        cur = nxt
    if not frames:
        raise SystemExit(f"FATAL: Databento returned zero rows for {sym} {start}→{end}")

    df = pd.concat(frames)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    # Continuous-symbol requests map the 'symbol' column to the alias (NQ.v.0),
    # so the roll detector keys on instrument_id — it changes when the front
    # contract rolls (verified databento 0.79.0, 2026-06-10).
    if "instrument_id" not in df.columns:
        raise SystemExit("FATAL: Databento to_df() lacks 'instrument_id' column — roll "
                         "detection impossible. Check databento version.")
    df["symbol"] = df["instrument_id"].astype(str)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df[OHLCV + ["symbol"]]


def resample_5min(df1m: pd.DataFrame) -> pd.DataFrame:
    """1-min → 5-min bars. Bar stamped 09:30 covers 09:30:00–09:34:59 ET
    (label='left', closed='left' — PRE-REGISTERED convention). Empty bins dropped."""
    if df1m is None or len(df1m) == 0:
        raise ValueError("resample_5min: empty input")
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df1m[OHLCV].resample("5min", label="left", closed="left").agg(agg)
    return out.dropna(subset=["Open"])


def filter_rth(df: pd.DataFrame, day: Optional[str] = None) -> pd.DataFrame:
    """Keep only 09:30–16:00 ET bars (optionally a single YYYY-MM-DD ET day).
    Same mask logic as sovereign/futures/bar_feed._filter_rth."""
    if df is None or len(df) == 0:
        return df
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    et = idx.tz_convert(ET)
    mask = ((et.hour > 9) | ((et.hour == 9) & (et.minute >= 30))) & (et.hour < 16)
    if day is not None:
        mask = mask & (et.strftime("%Y-%m-%d") == day)
    out = df.copy()
    out.index = idx
    return out[mask]


def overnight_slice(df: pd.DataFrame, us_session_date: str) -> pd.DataFrame:
    """Globex overnight bars for a US session: prior calendar day 18:00 ET → 09:30 ET."""
    d = datetime.fromisoformat(us_session_date)
    start = datetime(d.year, d.month, d.day, 18, 0, tzinfo=ET) - timedelta(days=1)
    end = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET)
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    et = idx.tz_convert(ET)
    out = df.copy()
    out.index = idx
    return out[(et >= start) & (et < end)]


def build_daily_table(df1m: pd.DataFrame) -> pd.DataFrame:
    """One row per US session date from Globex 1-min bars.

    Columns: rth_open, rth_close, rth_high, rth_low (→ next session's PDH/PDL),
    onh, onl (overnight 18:00→09:30 high/low), px_0925 (last 1-min Close ≤ 09:25 ET),
    prior_rth_close, overnight_ret, symbol (front contract during RTH), roll_day.

    Roll handling (pre-registered): Databento continuous is SPLICED, not
    back-adjusted. overnight_ret on a roll day mixes contracts → the bias
    engine zeroes the overnight score there, and structure trading skips
    roll-day sessions entirely.
    """
    if "symbol" not in df1m.columns:
        raise SystemExit("FATAL: build_daily_table needs the 'symbol' column")
    df = df1m.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    et = df.index.tz_convert(ET)
    df["_et_date"] = et.strftime("%Y-%m-%d")
    df["_et_minutes"] = et.hour * 60 + et.minute

    rth = df[(df["_et_minutes"] >= 570) & (df["_et_minutes"] < 960)]  # 09:30–16:00
    if len(rth) == 0:
        raise SystemExit("FATAL: no RTH bars in input")

    rows = []
    grouped = rth.groupby("_et_date", sort=True)
    for date, g in grouped:
        rows.append({
            "date": date,
            "rth_open": float(g["Open"].iloc[0]),
            "rth_close": float(g["Close"].iloc[-1]),
            "rth_high": float(g["High"].max()),
            "rth_low": float(g["Low"].min()),
            "rth_bars": int(len(g)),
            "symbol": str(g["symbol"].iloc[-1]),
        })
    daily = pd.DataFrame(rows).set_index("date").sort_index()

    # Overnight stats per session: prior calendar day 18:00 ET → 09:30 ET.
    onh, onl, px0925 = [], [], []
    for date in daily.index:
        d = datetime.fromisoformat(date)
        start = datetime(d.year, d.month, d.day, 18, 0, tzinfo=ET) - timedelta(days=1)
        cutoff_0930 = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET)
        cutoff_0925 = datetime(d.year, d.month, d.day, 9, 25, tzinfo=ET)
        sl = df[(et >= start) & (et < cutoff_0930)]
        if len(sl) == 0:
            onh.append(float("nan")); onl.append(float("nan")); px0925.append(float("nan"))
            continue
        onh.append(float(sl["High"].max()))
        onl.append(float(sl["Low"].min()))
        sl_et = sl.index.tz_convert(ET)
        pre = sl[sl_et <= cutoff_0925]
        px0925.append(float(pre["Close"].iloc[-1]) if len(pre) else float("nan"))
    daily["onh"], daily["onl"], daily["px_0925"] = onh, onl, px0925

    daily["prior_rth_close"] = daily["rth_close"].shift(1)
    daily["overnight_ret"] = daily["px_0925"] / daily["prior_rth_close"] - 1.0
    daily["roll_day"] = daily["symbol"].ne(daily["symbol"].shift(1))
    daily.iloc[0, daily.columns.get_loc("roll_day")] = False  # first row: unknown, not a roll
    return daily


def pull_aux_daily(start: str = "2017-06-01", end: Optional[str] = None) -> pd.DataFrame:
    """Daily closes for ^VIX/^N225/^GDAXI via yfinance → one frame, columns vix/nikkei/dax."""
    import yfinance as yf
    frames = {}
    for name, ticker in AUX_TICKERS.items():
        df = yf.download(ticker, start=start, end=end, interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise SystemExit(f"FATAL: yfinance returned nothing for {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        frames[name] = df["Close"]
    out = pd.DataFrame(frames)
    out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d")
    out.index.name = "date"
    return out


def _require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"FATAL: missing cache {path} — run scripts/es_nq_pull_history.py first")
    return path


def load_5min(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_parquet(_require(RTH_5MIN))
    return _slice(df, start, end)


def load_1min(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_parquet(_require(RAW_1MIN))
    return _slice(df, start, end)


def load_daily(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_parquet(_require(DAILY))
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    return df


def load_aux_daily() -> pd.DataFrame:
    return pd.read_parquet(_require(AUX_DAILY))


def _slice(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    et = df.index.tz_convert(ET)
    if start:
        df = df[et >= datetime.fromisoformat(start).replace(tzinfo=ET)]
        et = df.index.tz_convert(ET)
    if end:
        df = df[et < datetime.fromisoformat(end).replace(tzinfo=ET) + timedelta(days=1)]
    return df
