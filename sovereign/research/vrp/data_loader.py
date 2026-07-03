"""VRP data loader — the ONLY impure module in the vrp package.

Pulls free yfinance series (no historical option chains exist in this system; see
strategy_simulator for the DATA_INSUFFICIENT boundary). It also READS the OUTPUT of the
live forex backtest (logs/forex_backtest_trades.json) for the recent-window carry
secondary — it imports NO forex/ict modules (NN#1; enforced by test_vrp_isolation.py).

yfinance is imported lazily inside functions so importing this module stays light and
network-free (the isolation/causality tests import it without hitting the network).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]   # vrp -> research -> sovereign -> repo root
logging.getLogger("yfinance").setLevel(logging.ERROR)

FOREX_TRADES = ROOT / "logs" / "forex_backtest_trades.json"


def _hist(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    h = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
    if h is None or len(h) == 0:
        return pd.DataFrame()
    h.index = h.index.tz_localize(None)
    return h


def load_underlying(symbol: str) -> pd.DataFrame:
    """Daily OHLC for SPY/QQQ, corporate-action adjusted (auto_adjust)."""
    h = _hist(symbol)
    if len(h) < 500:
        raise SystemExit(f"FATAL: {symbol} daily history unavailable/too short ({len(h)}) — refusing thin data.")
    return h[["Open", "High", "Low", "Close"]]


def load_vol_index(ticker: str) -> pd.Series:
    """Implied-vol index close (^VIX, ^VXN, ...) in vol POINTS (e.g. 20.0 = 20%)."""
    h = _hist(ticker)
    if len(h) < 500:
        raise SystemExit(f"FATAL: {ticker} history unavailable/too short ({len(h)}) — refusing thin data.")
    return h["Close"].rename(ticker)


def load_carry_proxy() -> pd.Series:
    """DBV (Invesco G10 carry-factor ETF) daily returns, 2006->2023-03. The standard
    tradeable carry proxy — covers GFC/COVID/2022 (the v015 forex log is recent/OOS-only)."""
    h = _hist("DBV")
    if len(h) < 250:
        return pd.Series(dtype=float)
    return h["Close"].pct_change().dropna().rename("carry_dbv")


def load_forex_log_carry() -> pd.Series:
    """Recent-window secondary: daily summed pnl_pct from the live forex backtest OUTPUT.
    Reads a JSON artifact only — imports no forex module (NN#1)."""
    try:
        raw = json.loads(FOREX_TRADES.read_text())
    except Exception:
        return pd.Series(dtype=float)
    by_date: dict = {}
    for lst in raw.values():
        if not isinstance(lst, list):
            continue
        for t in lst:
            stamp = str(t.get("entry_date", t.get("entry", "")))[:10]
            d = pd.Timestamp(stamp) if stamp else pd.NaT
            if pd.isna(d):
                continue
            by_date[d] = by_date.get(d, 0.0) + float(t.get("pnl_pct", 0.0))
    if not by_date:
        return pd.Series(dtype=float)
    return pd.Series(by_date).sort_index().rename("carry_v015")


def load_overnight_qqq(qqq: pd.DataFrame | None = None) -> pd.Series:
    """Re-derive the overnight-QQQ return (open/prior-close) from QQQ OHLC. Mirrors the
    live edge's definition WITHOUT importing it (NN#1)."""
    if qqq is None:
        qqq = load_underlying("QQQ")
    return (qqq["Open"] / qqq["Close"].shift(1) - 1.0).dropna().rename("overnight_qqq")


# ════════════════════════════════════════════════════════════════════════════════
# ThetaDataLoader — historical SPY option chains via ThetaData V3 (local ThetaTerminal).
#
# LIVE (schema verified 2026-06-16 against ThetaTerminal v3 on 127.0.0.1:25503, Options
# VALUE tier). V3 renamed `root`->`symbol`, responses are CSV, dates YYYY-MM-DD, strikes
# in decimal dollars. The EOD chain endpoint returns one row per (strike,right) with
# bid/ask/close/volume — greeks/iv/open_interest are NOT included (-> NaN; optional, the
# backtest needs only strike + bid/ask/mid). Stock history is FREE-tier-gated (403), so the
# underlying close is NOT sourced here — the backtest uses the free yfinance spy_daily series.
#
# The simulator depends only on the normalized contract (OPTION_CHAIN_COLUMNS), never the
# source — so MockThetaDataLoader (tests) and this loader are interchangeable.
# ════════════════════════════════════════════════════════════════════════════════

# Normalized chain columns the simulator relies on. Any source must map to exactly these.
OPTION_CHAIN_COLUMNS = [
    "strike",
    "call_bid", "call_ask", "call_mid", "call_iv", "call_delta",
    "put_bid", "put_ask", "put_mid", "put_iv", "put_delta",
    "volume", "open_interest",
]

VRP_DATA_CACHE = ROOT / "data" / "research" / "vrp_data_cache"


class ThetaDataLoader:
    """Historical SPY option chains via ThetaData V3 (local ThetaTerminal on 25503).

    Contract (source-agnostic): every chain frame has columns == OPTION_CHAIN_COLUMNS.
    `get_chain_for_dte_range` additionally carries `expiration` (Timestamp) and `dte` (int).

    Live V3 surface (verified 2026-06-16; CSV responses, dates YYYY-MM-DD, strikes in $):
        expirations  GET /v3/option/list/expirations ?symbol=SPY
        eod chain    GET /v3/option/history/eod      ?symbol=SPY&expiration=YYYY-MM-DD
                                                       &start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
                     -> one row per (strike,right): symbol,expiration,strike,right,...,close,
                        volume,count,bid_size,bid_exchange,bid,bid_condition,ask_size,...,ask,...
    Localhost needs no auth (the terminal authenticates itself); a non-local base adds a
    Bearer header. greeks/iv/open_interest are absent from EOD -> NaN.
    """

    def __init__(self, api_key: str | None = None,
                 base_url: str = "http://127.0.0.1:25503",
                 cache_dir: Path = VRP_DATA_CACHE) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self._exp_cache: dict[str, list[str]] = {}

    # ── transport ──
    def _is_local(self) -> bool:
        return "127.0.0.1" in self.base_url or "localhost" in self.base_url

    def _get(self, path: str, timeout: int = 60) -> str:
        import time
        import urllib.error
        import urllib.request
        headers = {} if self._is_local() else {"Authorization": f"Bearer {self.api_key}"}
        req = urllib.request.Request(self.base_url + path, headers=headers)
        # ThetaTerminal rate-limits burst backfills with transient 403s (observed
        # 2026-07-02 mid-backfill; the same request succeeds after a cooldown).
        # Bounded backoff — anything still failing after 3 waits raises as before.
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read().decode()
            except urllib.error.HTTPError as exc:
                if exc.code == 403 and attempt < 3:
                    time.sleep(15 * (attempt + 1))
                    continue
                raise
        raise RuntimeError("unreachable")

    @staticmethod
    def _csv(text: str) -> pd.DataFrame:
        from io import StringIO
        if not text or not text.strip():
            return pd.DataFrame()
        return pd.read_csv(StringIO(text))

    # ── cache layer (per symbol/date/expiration parquet) ──
    def _cache_path(self, symbol: str, date, expiration) -> Path:
        return self.cache_dir / str(symbol) / f"{str(date)[:10]}_{str(expiration)[:10]}.parquet"

    def _cached_or_fetch(self, path: Path, fetch_fn):
        if path.exists():
            return pd.read_parquet(path)
        df = fetch_fn()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return df

    # ── normalization: v3 EOD rows (strike x right) -> one row per strike ──
    @staticmethod
    def _pivot_chain(raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty or "strike" not in raw.columns:
            return pd.DataFrame(columns=OPTION_CHAIN_COLUMNS)
        raw = raw.copy()
        raw["side"] = raw["right"].astype(str).str.upper().str[0]   # 'C' / 'P'
        rows = []
        for strike, g in raw.groupby("strike"):
            row = dict.fromkeys(OPTION_CHAIN_COLUMNS, float("nan"))
            row["strike"] = float(strike)
            row["volume"] = 0.0
            for _, r in g.iterrows():
                side = "call" if r["side"] == "C" else "put"
                bid = float(r.get("bid", float("nan")))
                ask = float(r.get("ask", float("nan")))
                row[f"{side}_bid"] = bid
                row[f"{side}_ask"] = ask
                row[f"{side}_mid"] = (bid + ask) / 2.0
                row["volume"] += float(r.get("volume", 0) or 0)
            rows.append(row)
        return (pd.DataFrame(rows)[OPTION_CHAIN_COLUMNS]
                .sort_values("strike").reset_index(drop=True))

    # ── data accessors (V3) ──
    def list_expirations(self, symbol: str = "SPY") -> list[str]:
        if symbol not in self._exp_cache:
            df = self._csv(self._get(f"/v3/option/list/expirations?symbol={symbol}"))
            exps = sorted(df["expiration"].astype(str).str[:10].unique().tolist()) if not df.empty else []
            self._exp_cache[symbol] = exps
        return self._exp_cache[symbol]

    def earliest_available_date(self, symbol: str = "SPY") -> str:
        exps = self.list_expirations(symbol)
        return exps[0] if exps else ""

    def get_underlying_close(self, symbol: str, date) -> float:
        raise NotImplementedError(
            "ThetaData stock history requires a Stock VALUE subscription (current: FREE -> HTTP 403). "
            "The VRP backtest sources SPY spot + realized vol from the free yfinance series "
            "(spy_daily), so this method is not on the backtest path.")

    def get_option_chain(self, symbol: str, date, expiration) -> pd.DataFrame:
        """Full chain for one (date, expiration), normalized to OPTION_CHAIN_COLUMNS. Cached.
        ThetaData HTTP 472 = no data for the request (e.g. no EOD for that expiry on that day)
        -> return (and cache) an empty chain rather than abort the backtest."""
        import urllib.error
        d, e = str(date)[:10], str(expiration)[:10]

        def _fetch() -> pd.DataFrame:
            try:
                raw = self._csv(self._get(
                    f"/v3/option/history/eod?symbol={symbol}&expiration={e}&start_date={d}&end_date={d}"))
            except urllib.error.HTTPError as ex:
                if ex.code == 472:                      # NO_DATA
                    return pd.DataFrame(columns=OPTION_CHAIN_COLUMNS)
                raise
            return self._pivot_chain(raw)
        return self._cached_or_fetch(self._cache_path(symbol, d, e), _fetch)

    def get_chain_for_dte_range(self, symbol: str, date, dte_min: int, dte_max: int) -> pd.DataFrame:
        """Chains for every expiration `dte_min..dte_max` calendar days out, with expiration + dte."""
        d = pd.Timestamp(str(date)[:10])
        frames = []
        for exp in self.list_expirations(symbol):
            dte = (pd.Timestamp(exp) - d).days
            if dte_min <= dte <= dte_max:
                ch = self.get_option_chain(symbol, str(d.date()), exp)
                if not ch.empty:
                    ch = ch.copy()
                    ch["expiration"] = pd.Timestamp(exp)
                    ch["dte"] = int(dte)
                    frames.append(ch)
        if not frames:
            return pd.DataFrame(columns=OPTION_CHAIN_COLUMNS + ["expiration", "dte"])
        return pd.concat(frames, ignore_index=True)
