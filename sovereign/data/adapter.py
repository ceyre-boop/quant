"""Unified market data interface — one seam in front of every vendor.

Callers ask for bars; they do not ask Polygon or Alpaca or Yahoo for bars. The
vendor is chosen by ``DATA_PRIMARY`` / ``DATA_FALLBACK`` in the environment, and
if the primary raises the fallback runs transparently (logged, never silent).

This CONSOLIDATES three pre-existing part-adapters rather than adding a fourth:

  - ``data/providers.py``            (yfinance primary + polygon fallback, SYMBOL_MAP)
  - ``sovereign/data/feeds/alpaca_feed.py``  (Alpaca bars + parquet cache)
  - ``data/alpaca_client.py``        (a second Alpaca bar fetcher)

Those modules remain as the vendor-specific transport underneath; the backends
here wrap them. No new vendor SDK code is introduced.

    from sovereign.data.adapter import MarketDataAdapter
    adapter = MarketDataAdapter()
    bars = adapter.get_bars("SPY", "2026-01-02", "2026-01-10", timeframe="1min")

Isolation: imports nothing from ``ict/`` (NN#1).

SCOPE BOUNDARY — read before extending:
This adapter is a *transport* seam. It decides where bytes come from. It does
NOT model fills, spreads, slippage or costs, and it must never be given that
job. ``execution/quotes.py`` (real captured quotes) and
``backtester/realistic_fills.py`` (the model) are deliberately two independent
pricing paths whose difference IS the measurement recorded as
``vs_backtest_delta``. Routing both through one pricing implementation would
drive that delta to zero by construction and destroy the instrument. See
NEXT.md 2026-07-20.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd

from sovereign.data.cache import DataCache

logger = logging.getLogger(__name__)

BAR_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

_DEFAULT_PRIMARY = "alpaca"
_DEFAULT_FALLBACK = "yfinance"


class DataUnavailable(RuntimeError):
    """Raised when the primary and the fallback both fail."""


class VendorNotSupported(NotImplementedError):
    """Raised when a vendor cannot serve a given call (e.g. yfinance options)."""


# ── normalisation ────────────────────────────────────────────────────────────

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce any vendor frame into the BAR_COLUMNS contract.

    Vendors disagree on everything: Alpaca returns a MultiIndex, yfinance
    returns capitalised columns and sometimes a (Field, Symbol) MultiIndex,
    Polygon returns short keys (t/o/h/l/c/v). Every one of those variations has
    already bitten this repo at least once, so normalisation is centralised
    here rather than repeated per caller.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=BAR_COLUMNS)

    df = df.copy()

    # yfinance >= 0.2 returns a (Field, Symbol) MultiIndex even for one ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()

    rename = {
        "t": "timestamp", "date": "timestamp", "Date": "timestamp",
        "datetime": "timestamp", "Datetime": "timestamp", "index": "timestamp",
        "o": "open", "Open": "open",
        "h": "high", "High": "high",
        "l": "low", "Low": "low",
        "c": "close", "Close": "close",
        "v": "volume", "Volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Duplicate labels can survive the rename (e.g. both 'Close' and 'close').
    df = df.loc[:, ~df.columns.duplicated()]

    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"vendor frame missing required columns {missing}; "
                         f"got {list(df.columns)}")

    df = df[BAR_COLUMNS]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ── backends ─────────────────────────────────────────────────────────────────

class _Backend:
    """A vendor. Any method may raise VendorNotSupported; the adapter falls back."""

    name = "base"

    def get_bars(self, symbol: str, start: str, end: str, timeframe: str) -> pd.DataFrame:
        raise VendorNotSupported(f"{self.name}: get_bars")

    def get_snapshot(self, symbols: list[str]) -> dict:
        raise VendorNotSupported(f"{self.name}: get_snapshot")

    def get_top_movers(self, n: int, min_gap_pct: float) -> list[dict]:
        raise VendorNotSupported(f"{self.name}: get_top_movers")

    def get_options_chain(self, symbol: str, expiry: str | None) -> pd.DataFrame:
        raise VendorNotSupported(f"{self.name}: get_options_chain")


class AlpacaBackend(_Backend):
    name = "alpaca"

    # (amount, unit) — resolved against alpaca's TimeFrameUnit at call time.
    _TF = {"1min": (1, "Minute"), "5min": (5, "Minute"), "15min": (15, "Minute"),
           "1h": (1, "Hour"), "1hour": (1, "Hour"),
           "1d": (1, "Day"), "1day": (1, "Day")}

    def __init__(self):
        from data.alpaca_client import AlpacaDataClient
        self._client = AlpacaDataClient()

    def get_bars(self, symbol, start, end, timeframe):
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed

        spec = self._TF.get(timeframe.lower())
        if spec is None:
            raise VendorNotSupported(f"alpaca: timeframe {timeframe!r}")
        amount, unit = spec
        tf = TimeFrame(amount, getattr(TimeFrameUnit, unit))

        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=tf,
            start=pd.Timestamp(start, tz="UTC").to_pydatetime(),
            end=pd.Timestamp(end, tz="UTC").to_pydatetime(),
            feed=DataFeed.SIP,
        )
        bars = self._client.client.get_stock_bars(req)
        if not bars or not hasattr(bars, "df") or bars.df.empty:
            return pd.DataFrame(columns=BAR_COLUMNS)
        return _normalise(bars.df.reset_index())

    def get_snapshot(self, symbols):
        out = {}
        for s in symbols:
            price = self._client.get_latest_price(s)
            if price is not None:
                out[s] = {"symbol": s, "price": float(price), "source": self.name}
        return out


class PolygonBackend(_Backend):
    name = "polygon"

    _TF = {"1min": (1, "minute"), "5min": (5, "minute"), "15min": (15, "minute"),
           "1h": (1, "hour"), "1hour": (1, "hour"), "1d": (1, "day"), "1day": (1, "day")}

    def __init__(self):
        from data.polygon_client import PolygonRestClient
        self._client = PolygonRestClient()

    def get_bars(self, symbol, start, end, timeframe):
        spec = self._TF.get(timeframe.lower())
        if spec is None:
            raise VendorNotSupported(f"polygon: timeframe {timeframe!r}")
        mult, span = spec
        resp = self._client.get_aggregates(symbol, mult, span, start, end)
        results = (resp or {}).get("results") or []
        if not results:
            return pd.DataFrame(columns=BAR_COLUMNS)
        df = pd.DataFrame(results)
        df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        return _normalise(df)

    def get_snapshot(self, symbols):
        out = {}
        for s in symbols:
            snap = self._client.get_snapshot(s) or {}
            ticker = snap.get("ticker") or {}
            day = ticker.get("day") or {}
            prev = ticker.get("prevDay") or {}
            price = day.get("c") or prev.get("c")
            if price:
                out[s] = {"symbol": s, "price": float(price),
                          "prev_close": prev.get("c"), "source": self.name}
        return out

    def get_top_movers(self, n, min_gap_pct):
        """Pre-market gainers from the full-market grouped snapshot."""
        resp = self._client._get(
            "/v2/snapshot/locale/us/markets/stocks/gainers") or {}
        movers = []
        for t in (resp.get("tickers") or []):
            gap = t.get("todaysChangePerc")
            if gap is None or gap < min_gap_pct:
                continue
            day, prev = t.get("day") or {}, t.get("prevDay") or {}
            movers.append({
                "symbol": t.get("ticker"),
                "gap_pct": float(gap),
                "price": day.get("c") or prev.get("c"),
                "prev_close": prev.get("c"),
                "volume": day.get("v"),
                "source": self.name,
            })
        movers.sort(key=lambda m: m["gap_pct"], reverse=True)
        return movers[:n]


class YFinanceBackend(_Backend):
    name = "yfinance"

    _TF = {"1min": "1m", "5min": "5m", "15min": "15m",
           "1h": "1h", "1hour": "1h", "1d": "1d", "1day": "1d"}

    def get_bars(self, symbol, start, end, timeframe):
        import yfinance as yf

        interval = self._TF.get(timeframe.lower())
        if interval is None:
            raise VendorNotSupported(f"yfinance: timeframe {timeframe!r}")
        df = yf.download(symbol, start=start, end=end, interval=interval,
                         progress=False, auto_adjust=False)
        return _normalise(df)

    def get_snapshot(self, symbols):
        import yfinance as yf

        out = {}
        for s in symbols:
            try:
                fi = yf.Ticker(s).fast_info
                price = fi.get("lastPrice") if hasattr(fi, "get") else fi.last_price
                if price:
                    out[s] = {"symbol": s, "price": float(price), "source": self.name}
            except Exception as e:  # noqa: BLE001 - per-symbol failure is not fatal
                logger.debug("yfinance snapshot failed for %s: %s", s, e)
        return out

    def get_options_chain(self, symbol, expiry=None):
        import yfinance as yf

        tk = yf.Ticker(symbol)
        expiries = tk.options
        if not expiries:
            return pd.DataFrame()
        target = expiry or expiries[0]
        chain = tk.option_chain(target)
        calls, puts = chain.calls.copy(), chain.puts.copy()
        calls["option_type"], puts["option_type"] = "call", "put"
        out = pd.concat([calls, puts], ignore_index=True)
        out["expiry"] = target
        out["symbol"] = symbol
        return out


_BACKENDS: dict[str, type[_Backend]] = {
    "alpaca": AlpacaBackend,
    "polygon": PolygonBackend,
    "yfinance": YFinanceBackend,
}


# ── adapter ──────────────────────────────────────────────────────────────────

class MarketDataAdapter:
    """Unified market data interface. Vendor is configured in .env, not in calling code."""

    def __init__(self, primary: str | None = None, fallback: str | None = None,
                 cache: DataCache | None = None, use_cache: bool = True):
        self.primary_name = (primary or os.getenv("DATA_PRIMARY", _DEFAULT_PRIMARY)).lower()
        self.fallback_name = (fallback or os.getenv("DATA_FALLBACK", _DEFAULT_FALLBACK)).lower()
        for name in (self.primary_name, self.fallback_name):
            if name and name not in _BACKENDS:
                raise ValueError(f"unknown vendor {name!r}; have {sorted(_BACKENDS)}")
        self.use_cache = use_cache
        self.cache = cache or DataCache()
        self._instances: dict[str, _Backend] = {}

    # ── vendor plumbing ──────────────────────────────────────────────────────

    def _backend(self, name: str) -> _Backend:
        """Backends are constructed lazily — a missing Polygon key must not break
        an Alpaca-only caller at import time."""
        if name not in self._instances:
            self._instances[name] = _BACKENDS[name]()
        return self._instances[name]

    def _with_fallback(self, op: str, call: Callable[[_Backend], Any]) -> Any:
        try:
            return call(self._backend(self.primary_name))
        except Exception as primary_err:  # noqa: BLE001 - fallback is the point
            if not self.fallback_name or self.fallback_name == self.primary_name:
                raise DataUnavailable(
                    f"{op}: primary {self.primary_name} failed and no fallback "
                    f"configured: {primary_err}") from primary_err
            logger.warning("%s: primary %s failed (%s: %s) -> falling back to %s",
                           op, self.primary_name, type(primary_err).__name__,
                           primary_err, self.fallback_name)
            try:
                return call(self._backend(self.fallback_name))
            except Exception as fb_err:  # noqa: BLE001
                raise DataUnavailable(
                    f"{op}: primary {self.primary_name} failed ({primary_err}); "
                    f"fallback {self.fallback_name} failed ({fb_err})") from fb_err

    # ── interface ────────────────────────────────────────────────────────────

    def get_bars(self, symbol: str, start: str, end: str,
                 timeframe: str = "1min") -> pd.DataFrame:
        """Returns OHLCV bars. Columns: timestamp, open, high, low, close, volume"""
        symbol = symbol.upper().strip()

        if not self.use_cache:
            return self._fetch_bars(symbol, start, end, timeframe)

        # Cache granularity is the symbol-day; the key carries the timeframe so
        # 1min and 1d pulls for the same day never collide.
        from sovereign.data.cache import date_range

        frames = []
        for day in date_range(start, end):
            key = f"{day}_{timeframe}"
            nxt = (pd.Timestamp(day) + timedelta(days=1)).date().isoformat()
            frames.append(self.cache.get_or_fetch(
                symbol, key,
                lambda d=day, n=nxt: self._fetch_bars(symbol, d, n, timeframe)))

        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return pd.DataFrame(columns=BAR_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values("timestamp").reset_index(drop=True)

    def _fetch_bars(self, symbol, start, end, timeframe) -> pd.DataFrame:
        return _normalise(self._with_fallback(
            f"get_bars({symbol})",
            lambda b: b.get_bars(symbol, start, end, timeframe)))

    def get_snapshot(self, symbols: list[str]) -> dict:
        """Current quote snapshot for a list of symbols"""
        symbols = [s.upper().strip() for s in symbols]
        return self._with_fallback("get_snapshot",
                                   lambda b: b.get_snapshot(symbols))

    def get_top_movers(self, n: int = 20, min_gap_pct: float = 0.5) -> list[dict]:
        """Pre-market top gainers by gap percentage"""
        return self._with_fallback("get_top_movers",
                                   lambda b: b.get_top_movers(n, min_gap_pct))

    def get_options_chain(self, symbol: str, expiry: str | None = None) -> pd.DataFrame:
        """Options chain for a symbol"""
        symbol = symbol.upper().strip()
        return self._with_fallback("get_options_chain",
                                   lambda b: b.get_options_chain(symbol, expiry))

    # ── diagnostics ──────────────────────────────────────────────────────────

    def health(self) -> dict:
        return {"primary": self.primary_name, "fallback": self.fallback_name,
                "cache_dir": str(self.cache.cache_dir),
                "cache_stats": self.cache.stats.summary()}
