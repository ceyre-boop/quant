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
# ThetaDataLoader — historical option chains (Phase II, awaiting subscription).
#
# DESIGN-ONLY: the RETURN CONTRACT (normalized columns) is committed; the API request
# bodies are TODO stubs. The schema is UNVERIFIED until the ThetaData Options Value
# account is active — run scripts/vrp_schema_verify.py once on activation, then fill the
# bodies marked `# VERIFY SCHEMA AGAINST LIVE RESPONSE BEFORE IMPLEMENTING`.
#
# The simulator depends only on this contract, never on the source — so MockThetaDataLoader
# (tests) and the real loader are interchangeable.
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
    """Historical SPY/QQQ option chains via ThetaData (local ThetaTerminal REST gateway).

    Contract (source-agnostic): every chain frame has columns == OPTION_CHAIN_COLUMNS.
    `get_chain_for_dte_range` additionally carries `expiration` (date) and `dte` (int).

    Assumed API surface (DOCUMENTED, NOT CALLED — verify on activation):
        base_url        http://127.0.0.1:25510   (ThetaTerminal local gateway)
        eod chain       GET /v2/hist/option/eod      ?root=&exp=YYYYMMDD&start_date=&end_date=
        quote           GET /v2/hist/option/quote    ?root=&exp=&right=C|P&strike=<1/10 cent>&...
        expirations     GET /v2/list/expirations     ?root=
        strikes         GET /v2/list/strikes         ?root=&exp=
    Strikes are expressed in 1/10-cent integers in the raw API; the loader normalizes to
    dollars. None of the above is exercised until the schema is verified live.
    """

    def __init__(self, api_key: str | None = None,
                 base_url: str = "http://127.0.0.1:25510",
                 cache_dir: Path = VRP_DATA_CACHE) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.cache_dir = Path(cache_dir)

    # ── cache layer (REAL — source-agnostic; works the moment the fetchers are filled) ──
    def _cache_path(self, symbol: str, date, expiration) -> Path:
        d = str(date)[:10]
        e = str(expiration)[:10]
        return self.cache_dir / str(symbol) / f"{d}_{e}.parquet"

    def _cached_or_fetch(self, path: Path, fetch_fn):
        """Read the parquet cache if present, else fetch once and persist. Each historical
        (symbol, date, expiration) costs one API call, ever."""
        if path.exists():
            return pd.read_parquet(path)
        df = fetch_fn()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return df

    # ── data accessors (TODO bodies — contract is fixed, parsing is not) ──
    def get_underlying_close(self, symbol: str, date) -> float:
        """Adjusted close for the underlying on `date`.
        # VERIFY SCHEMA AGAINST LIVE RESPONSE BEFORE IMPLEMENTING
        # endpoint: /v2/hist/stock/eod ?root=symbol&start_date=date&end_date=date -> parse close."""
        raise NotImplementedError("ThetaData get_underlying_close — fill after schema_verify")

    def get_option_chain(self, symbol: str, date, expiration) -> pd.DataFrame:
        """Full chain (all strikes) for one (date, expiration), normalized to OPTION_CHAIN_COLUMNS.
        # VERIFY SCHEMA AGAINST LIVE RESPONSE BEFORE IMPLEMENTING
        # endpoint: /v2/hist/option/eod ?root=symbol&exp=YYYYMMDD&start_date=date&end_date=date
        # map raw call/put bid/ask/iv/delta + volume/open_interest onto OPTION_CHAIN_COLUMNS;
        # mid = (bid+ask)/2; strike = raw_strike / 1000.0 (1/10-cent -> dollars)."""
        def _fetch() -> pd.DataFrame:
            raise NotImplementedError("ThetaData get_option_chain — fill after schema_verify")
        return self._cached_or_fetch(self._cache_path(symbol, date, expiration), _fetch)

    def get_chain_for_dte_range(self, symbol: str, date, dte_min: int, dte_max: int) -> pd.DataFrame:
        """All chains whose expiration is `dte_min..dte_max` calendar days out from `date`,
        concatenated with `expiration` and `dte` columns added.
        # VERIFY SCHEMA AGAINST LIVE RESPONSE BEFORE IMPLEMENTING
        # 1) /v2/list/expirations ?root=symbol -> expirations; keep those with dte in [min,max]
        # 2) for each, self.get_option_chain(...); assign expiration + dte; concat."""
        raise NotImplementedError("ThetaData get_chain_for_dte_range — fill after schema_verify")

    def earliest_available_date(self, symbol: str) -> str:
        """Earliest date ThetaData serves for `symbol` (the IS-boundary check).
        # VERIFY SCHEMA AGAINST LIVE RESPONSE BEFORE IMPLEMENTING
        # derive from /v2/list/expirations or a probe request; return 'YYYY-MM-DD'."""
        raise NotImplementedError("ThetaData earliest_available_date — fill after schema_verify")
