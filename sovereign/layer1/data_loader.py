"""sovereign/layer1/data_loader.py — historical raw data fetchers for the Layer-1 matrix.

Range-bounded to [TRAIN_START, TRAIN_END] = 2015-01-01 .. 2023-12-31. The 2024+ HOLDOUT is
NEVER fetched here: every fetch uses an exclusive upper bound of 2024-01-01, so holdout data
cannot enter the matrix or the forward label by construction.

FAIL-LOUD CONTRACT: every fetcher returns (data_or_None, FetchStatus). On any failure —
network error, missing series, empty history, missing dependency — it records the reason and
returns None. It NEVER returns zeros and NEVER silently substitutes. The LoadReport aggregates
per-source status so the pipeline can report exactly what loaded and what did not. Silent
zero-fill is how a data pipeline lies (a model "learns COT doesn't matter" when it never saw
COT); this module refuses to do that.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# In-process memo so repeated fetches of the same ticker/series don't re-hit the network
# (avoids yfinance/FRED rate-limiting when many features share a source). Caches failures
# (None) too, so a dead source isn't retried 50 times. Keyed by (kind, id).
_CACHE: dict = {}

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"          # holdout is 2024-01-01+ — never fetched
_YF_END_EXCL = "2024-01-01"       # yfinance end is exclusive → includes through 2023-12-31

# Global (cross-pair) sources.
FRED_SERIES = {
    "fed_funds": "DFF",           # daily effective fed funds
    "us_2y": "DGS2",
    "us_5y": "DGS5",
    "us_10y": "DGS10",
    "breakeven_10y": "T10YIE",    # 10y inflation expectations
    "us_2s10s": "T10Y2Y",         # FRED publishes this spread directly
}
YF_TICKERS = {
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "vix9d": "^VIX9D",
    "vix3m": "^VIX3M",
    "spy": "SPY",
    "oil": "CL=F",
    "gold": "GC=F",
    "crb": "^TRCCRB",
    "hyg": "HYG",
    "ief": "IEF",
    "tlt": "TLT",
}
PAIR_YF = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X",
}
# CFTC currency per pair (the non-USD leg) for COT positioning.
PAIR_COT_CCY = {
    "EURUSD": "EUR", "GBPUSD": "GBP", "USDJPY": "JPY",
    "AUDUSD": "AUD", "USDCAD": "CAD",
}
# data_fetcher country codes per pair (base, quote).
PAIR_COUNTRIES = {
    "EURUSD": ("EU", "US"), "GBPUSD": ("UK", "US"), "USDJPY": ("US", "JP"),
    "AUDUSD": ("AU", "US"), "USDCAD": ("US", "CA"),
}


@dataclass
class FetchStatus:
    source: str
    ok: bool
    rows: int = 0
    start: Optional[str] = None
    end: Optional[str] = None
    nan_pct: Optional[float] = None
    reason: Optional[str] = None   # populated on failure — never None on a failed fetch


class LoadReport:
    """Accumulates per-source fetch status. The anti-silent-drop record."""

    def __init__(self) -> None:
        self.items: list[FetchStatus] = []

    def record(self, status: FetchStatus) -> FetchStatus:
        self.items.append(status)
        return status

    def ok(self, source: str, series: pd.Series) -> FetchStatus:
        nan_pct = round(float(series.isna().mean()) * 100, 2) if len(series) else 100.0
        return self.record(FetchStatus(
            source=source, ok=True, rows=int(len(series)),
            start=str(series.index.min())[:10] if len(series) else None,
            end=str(series.index.max())[:10] if len(series) else None,
            nan_pct=nan_pct,
        ))

    def fail(self, source: str, reason: str) -> FetchStatus:
        return self.record(FetchStatus(source=source, ok=False, reason=reason))

    def failures(self) -> list[FetchStatus]:
        return [s for s in self.items if not s.ok]

    def to_dict(self) -> dict:
        return {
            "generated_for": "layer1 features_v1",
            "train_window": [TRAIN_START, TRAIN_END],
            "holdout_fetched": False,
            "n_sources": len(self.items),
            "n_ok": sum(1 for s in self.items if s.ok),
            "n_failed": len(self.failures()),
            "sources": [asdict(s) for s in self.items],
        }


# ─── helpers ───────────────────────────────────────────────────────────────────

def fred_api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip().startswith("FRED_API_KEY"):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise RuntimeError("FRED_API_KEY not found in environment or .env")
    return key


def _bound(s: pd.Series) -> pd.Series:
    """Clip a series to the training window — defensive belt-and-suspenders vs holdout."""
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.loc[(s.index >= pd.Timestamp(TRAIN_START)) & (s.index <= pd.Timestamp(TRAIN_END))]


# ─── fetchers (each fail-loud) ──────────────────────────────────────────────────

def get_fred_series(name: str, report: LoadReport) -> Optional[pd.Series]:
    series_id = FRED_SERIES.get(name, name)
    src = f"fred:{name}({series_id})"
    ckey = ("fred", series_id)
    if ckey in _CACHE:
        return _CACHE[ckey]
    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_api_key())
        raw = fred.get_series(series_id, observation_start=TRAIN_START, observation_end=TRAIN_END)
        if raw is None or len(raw) == 0:
            report.fail(src, "empty series from FRED")
            _CACHE[ckey] = None
            return None
        s = _bound(pd.Series(raw).astype(float))
        s.name = name
        if s.empty:
            report.fail(src, "no rows in training window after bounding")
            _CACHE[ckey] = None
            return None
        report.ok(src, s)
        _CACHE[ckey] = s
        return s
    except Exception as e:  # noqa: BLE001 — fail loud, record the reason
        report.fail(src, f"{type(e).__name__}: {e}")
        _CACHE[ckey] = None
        return None


def get_yf_close(name: str, ticker: Optional[str], report: LoadReport) -> Optional[pd.Series]:
    tk = ticker or YF_TICKERS.get(name, name)
    src = f"yfinance:{name}({tk})"
    ckey = ("yf", tk)
    if ckey in _CACHE:
        return _CACHE[ckey]
    try:
        import yfinance as yf
        df = yf.download(tk, start=TRAIN_START, end=_YF_END_EXCL,
                         auto_adjust=True, progress=False)
        if df is None or df.empty or "Close" not in df.columns:
            report.fail(src, "empty/invalid yfinance frame")
            _CACHE[ckey] = None
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):       # multiindex columns → squeeze
            close = close.iloc[:, 0]
        s = _bound(close.astype(float))
        s.name = name
        if s.empty:
            report.fail(src, "no rows in training window")
            _CACHE[ckey] = None
            return None
        report.ok(src, s)
        _CACHE[ckey] = s
        return s
    except Exception as e:  # noqa: BLE001
        report.fail(src, f"{type(e).__name__}: {e}")
        _CACHE[ckey] = None
        return None


def get_pair_prices(pair: str, report: LoadReport) -> Optional[pd.Series]:
    """Daily close for a pair (used for technical features AND labels)."""
    return get_yf_close(f"px:{pair}", PAIR_YF[pair], report)


def get_pair_differentials(pair: str, report: LoadReport) -> Optional[pd.DataFrame]:
    """Reuse ForexDataFetcher for clean historical rate/real-rate/momentum differentials."""
    src = f"data_fetcher:diffs({pair})"
    base, quote = PAIR_COUNTRIES[pair]
    try:
        from sovereign.forex.data_fetcher import ForexDataFetcher
        df = ForexDataFetcher().get_pair_differentials(base, quote, start=TRAIN_START)
        if df is None or df.empty:
            report.fail(src, "empty differentials frame")
            return None
        df = df[(df.index >= pd.Timestamp(TRAIN_START)) & (df.index <= pd.Timestamp(TRAIN_END))]
        if df.empty:
            report.fail(src, "no rows in training window")
            return None
        report.record(FetchStatus(src, True, rows=len(df),
                                  start=str(df.index.min())[:10], end=str(df.index.max())[:10],
                                  nan_pct=round(float(df.isna().mean().mean()) * 100, 2)))
        return df
    except Exception as e:  # noqa: BLE001
        report.fail(src, f"{type(e).__name__}: {e}")
        return None


def get_cot_net(pair: str, report: LoadReport) -> Optional[pd.Series]:
    """Weekly COT large-spec net position for the pair's non-USD currency.

    CAVEAT (disclosed, not hidden): cot_engine uses current CFTC category definitions applied
    to history, not point-in-time definitions — a known survivorship/redefinition limitation
    documented in docs/layer1/features.md. Recorded here so the limitation travels with the data.
    """
    ccy = PAIR_COT_CCY[pair]
    src = f"cot_engine:net({pair}/{ccy})"
    try:
        from sovereign.forex.cot_engine import COTEngine
        eng = COTEngine()
        s = None
        for meth in ("_load_or_fetch", "_load_history", "get_history"):
            fn = getattr(eng, meth, None)
            if callable(fn):
                s = fn(ccy)
                break
        if s is None or len(s) == 0:
            report.fail(src, (
                "no COT history: cot_engine._fetch_cftc downloads only the single hardcoded "
                "2024 file (which is HOLDOUT) and FX positioning is not in the disaggregated "
                "(commodity) report — it lives in the legacy/TFF report. Needs a dedicated "
                "historical COT fetcher (per-year 2015-2023 legacy zips, survivorship-aware). "
                "DEFERRED, not zero-filled."))
            return None
        s = _bound(pd.Series(s).astype(float))
        s.name = f"cot_net_{ccy}"
        if s.empty:
            report.fail(src, "no rows in training window")
            return None
        st = report.ok(src, s)
        st.reason = "OK; CAVEAT: current-category defs, not point-in-time (survivorship)"
        return s
    except Exception as e:  # noqa: BLE001
        report.fail(src, f"{type(e).__name__}: {e}")
        return None


# ─── historical event calendar (replaces snapshot event_calendar.py) ────────────

# Public, pre-announced FOMC meeting end-dates 2015–2023 (8/yr). Known at t0, not market data.
FOMC_DATES = [
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17","2015-07-29","2015-09-17","2015-10-28","2015-12-16",
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15","2016-07-27","2016-09-21","2016-11-02","2016-12-14",
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14","2017-07-26","2017-09-20","2017-11-01","2017-12-13",
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13","2018-08-01","2018-09-26","2018-11-08","2018-12-19",
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31","2019-09-18","2019-10-30","2019-12-11",
    "2020-01-29","2020-03-15","2020-04-29","2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28","2021-09-22","2021-11-03","2021-12-15",
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27","2022-09-21","2022-11-02","2022-12-14",
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26","2023-09-20","2023-11-01","2023-12-13",
]
# ECB Governing Council monetary-policy meeting dates 2015–2023 (approx; public schedule).
ECB_DATES = [
    "2015-01-22","2015-03-05","2015-04-15","2015-06-03","2015-07-16","2015-09-03","2015-10-22","2015-12-03",
    "2016-01-21","2016-03-10","2016-04-21","2016-06-02","2016-07-21","2016-09-08","2016-10-20","2016-12-08",
    "2017-01-19","2017-03-09","2017-04-27","2017-06-08","2017-07-20","2017-09-07","2017-10-26","2017-12-14",
    "2018-01-25","2018-03-08","2018-04-26","2018-06-14","2018-07-26","2018-09-13","2018-10-25","2018-12-13",
    "2019-01-24","2019-03-07","2019-04-10","2019-06-06","2019-07-25","2019-09-12","2019-10-24","2019-12-12",
    "2020-01-23","2020-03-12","2020-04-30","2020-06-04","2020-07-16","2020-09-10","2020-10-29","2020-12-10",
    "2021-01-21","2021-03-11","2021-04-22","2021-06-10","2021-07-22","2021-09-09","2021-10-28","2021-12-16",
    "2022-02-03","2022-03-10","2022-04-14","2022-06-09","2022-07-21","2022-09-08","2022-10-27","2022-12-15",
    "2023-02-02","2023-03-16","2023-05-04","2023-06-15","2023-07-27","2023-09-14","2023-10-26","2023-12-14",
]


def nfp_dates(start: str = TRAIN_START, end: str = TRAIN_END) -> list[pd.Timestamp]:
    """US Non-Farm Payrolls release ≈ first Friday of each month (BLS convention)."""
    out = []
    for ts in pd.date_range(start, end, freq="MS"):
        d = ts
        while d.weekday() != 4:          # 4 = Friday
            d += pd.Timedelta(days=1)
        out.append(d)
    return out


def historical_event_dates() -> dict[str, list[pd.Timestamp]]:
    return {
        "fomc": [pd.Timestamp(d) for d in FOMC_DATES],
        "ecb": [pd.Timestamp(d) for d in ECB_DATES],
        "nfp": nfp_dates(),
    }
