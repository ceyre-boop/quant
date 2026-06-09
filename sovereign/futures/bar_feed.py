"""Bar feeds for the futures sandbox — one abstraction, three sources.

Sandbox-local: no forex/ICT/intelligence imports.

  - ReplayBarFeed: drives a stored 1-min DataFrame bar-by-bar, yielding the
    session-to-date slice at each step (so VWAP/RSI replay exactly as they would
    accumulate live). This is what makes "backtest like it's live tonight" honest.
  - load_history(): fetch 1-min RTH bars for a day/lookback from IB (preferred) or
    yfinance (fallback so the replay runs tonight even if IB Gateway is down).

Decision (per plan): IB bars now, Databento later. yfinance is the tonight-fallback.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterator, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

TICKER_MAP = {"MES": "ES=F", "MNQ": "NQ=F", "ES": "ES=F", "NQ": "NQ=F"}


class ReplayBarFeed:
    """Replays a 1-min OHLCV DataFrame as a live session would see it.

    Each step yields (bar_timestamp, bars_so_far) where bars_so_far is the slice
    from session open through the current bar — the exact view compute_indicators
    gets live. `warmup` skips yielding until at least N bars exist.
    """

    def __init__(self, df, warmup: int = 2):
        if df is None or len(df) == 0:
            raise ValueError("ReplayBarFeed needs a non-empty DataFrame")
        self.df = df
        self.warmup = max(1, warmup)

    def __len__(self) -> int:
        return len(self.df)

    def stream(self) -> Iterator[tuple[datetime, object]]:
        for i in range(self.warmup, len(self.df) + 1):
            window = self.df.iloc[:i]
            ts = window.index[-1]
            yield ts, window


def _filter_rth(df, day: Optional[str] = None):
    """Keep only 09:30–16:00 ET bars (optionally for a single YYYY-MM-DD ET day)."""
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


def _ib_duration(lookback: str) -> str:
    """Convert a '20d'/'2w'/'1m' lookback into an IB durationStr ('20 D'/'2 W'/'1 M').
    IB serves 1-min bars up to ~1 month per request; falls back to '20 D' if unparseable."""
    s = (lookback or "").strip().lower()
    num = "".join(ch for ch in s if ch.isdigit()) or "20"
    unit = {"d": "D", "w": "W", "m": "M", "y": "Y"}.get(s[-1:], "D")
    return f"{num} {unit}"


def load_history(instrument: str, source: str = "yf",
                 day: Optional[str] = None, lookback: str = "5d"):
    """1-min RTH OHLCV for `instrument` (MES/MNQ). source 'ib' | 'yf'.

    'yf': last `lookback` of 1-min bars (yfinance caps 1m at ~7 days), RTH-filtered,
          optionally narrowed to a single ET `day`.
    'ib': pulls via the IB bridge (Gateway must be running).
    Returns a DataFrame with Open/High/Low/Close/Volume indexed by tz-aware datetime.
    """
    inst = instrument.upper()
    if source == "ib":
        from sovereign.futures.ib_bridge import IBBridge
        bridge = IBBridge()
        bridge.connect()
        try:
            contract = bridge.mes_contract() if inst in ("MES", "ES") else bridge.mnq_contract()
            end = "" if day is None else f"{day.replace('-', '')} 16:00:00 US/Eastern"
            duration = "1 D" if day else _ib_duration(lookback)  # honor --lookback for IB
            df = bridge.historical_bars(contract, duration=duration,
                                        bar_size="1 min", rth=True, end=end)
        finally:
            bridge.disconnect()
        return _filter_rth(df, day)

    # yfinance fallback
    import yfinance as yf
    import pandas as pd
    ticker = TICKER_MAP.get(inst, "ES=F")
    df = yf.download(ticker, period=lookback, interval="1m", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return _filter_rth(df[["Open", "High", "Low", "Close", "Volume"]], day)


def live_session_bars(bridge, contract):
    """Current RTH session 1-min OHLCV from an already-connected IB bridge.
    Thin wrapper so the live monitor reuses its persistent connection (no reconnect
    per poll) and stays testable."""
    return bridge.historical_bars(contract, duration="1 D", bar_size="1 min", rth=True)


def session_days(df) -> list[str]:
    """Distinct ET trading days present in a bar DataFrame (sorted)."""
    if df is None or len(df) == 0:
        return []
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return sorted(set(idx.tz_convert(ET).strftime("%Y-%m-%d")))
