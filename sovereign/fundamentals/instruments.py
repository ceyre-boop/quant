"""What kind of instrument a ticker is, and therefore what it can ever have.

This exists so the panel can tell the difference between two very different
statements:

    "no earnings data cached yet"        -> run the harvester
    "this instrument has no earnings"    -> nothing will ever fill it

An ETF has no earnings, no Form 4 and no 13F filed under its own ticker, and it
never will. Rendering "not harvested" there sends the reader to look for a
problem that does not exist.

The list is explicit rather than inferred: there is no free reference-data call
that reliably classifies a ticker, and guessing from name patterns misfires on
real operating companies. Unlisted tickers are treated as equities, which is the
safe default -- worst case we fetch and get an honest empty answer back.
"""
from __future__ import annotations

KNOWN_ETFS: frozenset[str] = frozenset({
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "GLD", "SLV", "USO", "UNG",
    "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "AGG", "BND",
    "VIXY", "UVXY", "SVXY", "SQQQ", "TQQQ", "SPXL", "SPXS", "SOXL", "SOXS",
    "XLE", "XLF", "XLK", "XLV", "XLY", "XLI", "XLU", "XLP", "XLB", "XLRE", "XLC",
    "SMH", "SOXX", "ARKK", "ARKG", "EEM", "EFA", "VEA", "VWO", "IEMG",
    "KWEB", "SLX", "GDX", "GDXJ", "XBI", "IBB", "ITB", "XHB", "KRE", "IYR",
})

# Futures (=F) and FX pairs (=X) in yfinance notation. No filings of any kind.
_DERIVATIVE_SUFFIXES = ("=F", "=X")


def is_etf(ticker: str) -> bool:
    return ticker.upper().strip() in KNOWN_ETFS


def is_derivative(ticker: str) -> bool:
    return ticker.upper().strip().endswith(_DERIVATIVE_SUFFIXES)


def files_with_sec(ticker: str) -> bool:
    """True when the issuer files earnings, Form 4 and 13F under this ticker."""
    return not (is_etf(ticker) or is_derivative(ticker))


def no_filings_reason(ticker: str) -> str | None:
    """Why this ticker can never have issuer filings, or None if it can."""
    if is_derivative(ticker) or ticker.upper().strip() in NON_SEC:
        return f"{ticker.upper()} is an FX/futures symbol — no issuer files with the SEC"
    if is_etf(ticker):
        return (f"{ticker.upper()} is an ETF — it has no earnings, no Form 4 and "
                "no 13F filed under its own ticker")
    return None


# ── Non-SEC tradeable universe ──────────────────────────────────────────────
# FX pairs and futures are absent from SEC company_tickers.json by definition —
# no issuer, no filings. They are still things the desk trades and charts every
# day, so they must be first-class in the symbol search rather than falling
# through to "not in the SEC ticker map", which reads as a broken tool.
#
# This is the single source of truth: the harvester emits it into
# data/fundamentals/symbol_index.json so the front end never keeps a second copy
# that can drift.
#
# `tv` is the TradingView symbol the chart embed should load.
NON_SEC: dict[str, dict[str, str]] = {
    # FX majors + the live v015 carry pairs
    "EURUSD": {"name": "Euro / US Dollar",          "kind": "fx", "tv": "FX:EURUSD"},
    "GBPUSD": {"name": "British Pound / US Dollar", "kind": "fx", "tv": "FX:GBPUSD"},
    "AUDUSD": {"name": "Australian Dollar / US Dollar", "kind": "fx", "tv": "FX:AUDUSD"},
    "NZDUSD": {"name": "New Zealand Dollar / US Dollar", "kind": "fx", "tv": "FX:NZDUSD"},
    "USDJPY": {"name": "US Dollar / Japanese Yen",  "kind": "fx", "tv": "FX:USDJPY"},
    "USDCHF": {"name": "US Dollar / Swiss Franc",   "kind": "fx", "tv": "FX:USDCHF"},
    "USDCAD": {"name": "US Dollar / Canadian Dollar", "kind": "fx", "tv": "FX:USDCAD"},
    "GBPJPY": {"name": "British Pound / Japanese Yen", "kind": "fx", "tv": "FX:GBPJPY"},
    "EURGBP": {"name": "Euro / British Pound",      "kind": "fx", "tv": "FX:EURGBP"},
    "EURJPY": {"name": "Euro / Japanese Yen",       "kind": "fx", "tv": "FX:EURJPY"},
    "AUDNZD": {"name": "Australian Dollar / NZ Dollar", "kind": "fx", "tv": "FX:AUDNZD"},
    "AUDJPY": {"name": "Australian Dollar / Yen",   "kind": "fx", "tv": "FX:AUDJPY"},
    # Index futures
    "NQ":  {"name": "Nasdaq 100 E-mini",   "kind": "future", "tv": "CME_MINI:NQ1!"},
    "MNQ": {"name": "Nasdaq 100 Micro",    "kind": "future", "tv": "CME_MINI:MNQ1!"},
    "ES":  {"name": "S&P 500 E-mini",      "kind": "future", "tv": "CME_MINI:ES1!"},
    "MES": {"name": "S&P 500 Micro",       "kind": "future", "tv": "CME_MINI:MES1!"},
    "YM":  {"name": "Dow E-mini",          "kind": "future", "tv": "CBOT_MINI:YM1!"},
    "RTY": {"name": "Russell 2000 E-mini", "kind": "future", "tv": "CME_MINI:RTY1!"},
    # Commodity + rates futures
    "CL": {"name": "Crude Oil WTI",   "kind": "future", "tv": "NYMEX:CL1!"},
    "NG": {"name": "Natural Gas",     "kind": "future", "tv": "NYMEX:NG1!"},
    "GC": {"name": "Gold",            "kind": "future", "tv": "COMEX:GC1!"},
    "SI": {"name": "Silver",          "kind": "future", "tv": "COMEX:SI1!"},
    "ZB": {"name": "30-Year T-Bond",  "kind": "future", "tv": "CBOT:ZB1!"},
    "ZN": {"name": "10-Year T-Note",  "kind": "future", "tv": "CBOT:ZN1!"},
    # Volatility / crypto
    "VIX":     {"name": "CBOE Volatility Index", "kind": "index",  "tv": "CBOE:VIX"},
    "BTCUSD":  {"name": "Bitcoin / US Dollar",   "kind": "crypto", "tv": "COINBASE:BTCUSD"},
    "ETHUSD":  {"name": "Ethereum / US Dollar",  "kind": "crypto", "tv": "COINBASE:ETHUSD"},
}


def tradingview_symbol(ticker: str) -> str | None:
    """TradingView symbol for a non-SEC instrument, or None for equities
    (where the bare ticker is what TradingView expects)."""
    return (NON_SEC.get(ticker.upper().strip()) or {}).get("tv")
