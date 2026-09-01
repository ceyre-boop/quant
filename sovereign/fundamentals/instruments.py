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
    if is_derivative(ticker):
        return f"{ticker.upper()} is a futures/FX symbol — no issuer filings exist"
    if is_etf(ticker):
        return (f"{ticker.upper()} is an ETF — it has no earnings, no Form 4 and "
                "no 13F filed under its own ticker")
    return None
