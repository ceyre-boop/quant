"""yfinance transport: earnings history and forward-estimate snapshots.

yfinance wraps Yahoo Finance's undocumented endpoints; its surface for
estimate revisions (eps_trend / eps_revisions) has shifted across releases,
so those calls are guarded and degrade to [] rather than raising.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time as dt_time
from typing import Optional

import pandas as pd
import yfinance as yf

from sovereign.fundamentals.types import EarningsEvent, EstimateSnapshot

log = logging.getLogger(__name__)

_SOURCE = "yahoo"

_ET_AMC_HOUR = 16     # >= 16:00 ET => after market close
_ET_BMO_HOUR = 9.5    # < 09:30 ET => before market open


def _report_time(ts: pd.Timestamp) -> str:
    """yfinance's Earnings Date is tz-aware in America/New_York; the clock time
    itself (not a separate flag) is the only signal Yahoo gives us for BMO/AMC."""
    if ts is None or pd.isna(ts):
        return "unknown"
    hour_frac = ts.hour + ts.minute / 60.0
    if hour_frac >= _ET_AMC_HOUR:
        return "amc"
    if hour_frac < _ET_BMO_HOUR:
        return "bmo"
    return "unknown"


def _to_date(ts: pd.Timestamp) -> Optional[date]:
    if ts is None or pd.isna(ts):
        return None
    return ts.date()


def _to_float(v) -> Optional[float]:
    if v is None or pd.isna(v):
        return None
    return float(v)


def earnings_history(ticker: str, limit: int = 24) -> list[EarningsEvent]:
    """Historical + upcoming earnings prints from yfinance.get_earnings_dates.

    Rows with NaN Reported EPS are future prints (estimate known, actual not
    yet reported) — included, not dropped, so callers can see what's ahead;
    eps_surprise / eps_surprise_pct simply stay None for those rows.
    """
    t = yf.Ticker(ticker)
    df = t.get_earnings_dates(limit=limit)
    if df is None or df.empty:
        return []

    out: list[EarningsEvent] = []
    for ts, row in df.iterrows():
        eps_est = _to_float(row.get("EPS Estimate"))
        eps_act = _to_float(row.get("Reported EPS"))
        surprise_pct = _to_float(row.get("Surprise(%)"))

        eps_surprise = None
        if eps_est is not None and eps_act is not None:
            eps_surprise = eps_act - eps_est

        published_ts: Optional[datetime] = ts.to_pydatetime() if not pd.isna(ts) else None

        out.append(EarningsEvent(
            source=_SOURCE,
            published_ts=published_ts,
            ticker=ticker.upper(),
            report_date=_to_date(ts),
            report_time=_report_time(ts),
            eps_estimate=eps_est,
            eps_actual=eps_act,
            eps_surprise=eps_surprise,
            eps_surprise_pct=surprise_pct,
        ))
    return out


def estimate_snapshot(ticker: str) -> list[EstimateSnapshot]:
    """Forward consensus snapshot from yfinance eps_trend/eps_revisions.

    yfinance's API for these has moved across releases (Ticker.eps_trend has
    appeared, been renamed, and disappeared entirely between versions), so this
    is guarded end to end. Any miss returns [] with a logged reason rather than
    raising — this is exactly the "gap the panel can show as absent" case, not
    a SectionUnavailable, since the free tier never reliably has this section.
    """
    t = yf.Ticker(ticker)

    trend_df = None
    try:
        if hasattr(t, "eps_trend"):
            trend_df = t.eps_trend
    except Exception as e:  # yfinance surfaces arbitrary exceptions from Yahoo's JSON shape
        log.info("yahoo estimate_snapshot: eps_trend unavailable for %s: %s", ticker, e)
        trend_df = None

    if trend_df is None or not isinstance(trend_df, pd.DataFrame) or trend_df.empty:
        log.info("yahoo estimate_snapshot: no eps_trend data for %s", ticker)
        return []

    revisions_df = None
    try:
        if hasattr(t, "eps_revisions"):
            revisions_df = t.eps_revisions
    except Exception as e:
        log.info("yahoo estimate_snapshot: eps_revisions unavailable for %s: %s", ticker, e)
        revisions_df = None

    snapshot_date = date.today()
    out: list[EstimateSnapshot] = []
    for period, row in trend_df.iterrows():
        period_str = str(period)
        up_30d = down_30d = None
        if isinstance(revisions_df, pd.DataFrame) and period_str in getattr(revisions_df, "index", []):
            rev_row = revisions_df.loc[period_str]
            up_30d = int(rev_row["upLast30days"]) if "upLast30days" in rev_row and not pd.isna(rev_row["upLast30days"]) else None
            down_30d = int(rev_row["downLast30days"]) if "downLast30days" in rev_row and not pd.isna(rev_row["downLast30days"]) else None

        out.append(EstimateSnapshot(
            source=_SOURCE,
            published_ts=datetime.combine(snapshot_date, dt_time.min),
            ticker=ticker.upper(),
            snapshot_date=snapshot_date,
            period=period_str,
            eps_avg=_to_float(row.get("current")) if "current" in row else None,
            n_analysts=int(row["numberOfAnalysts"]) if "numberOfAnalysts" in row and not pd.isna(row["numberOfAnalysts"]) else None,
            up_30d=up_30d,
            down_30d=down_30d,
        ))
    return out
