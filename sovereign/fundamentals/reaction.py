"""sovereign/fundamentals/reaction.py — price-reaction join.

Derived, never fetched: takes EarningsEvent rows already sourced from a
provider and joins them onto daily bars pulled through
``sovereign.data.adapter.MarketDataAdapter`` (the one seam in front of every
vendor — see that module's docstring). Nothing here talks to a vendor SDK
directly.

Missing bars is NOT a SectionUnavailable condition: the price-reaction row is
a computed convenience, not an independently-sourced fact. If bars can't be
pulled (vendor outage, delisted ticker, insufficient history), the honest
response is an empty reaction list, letting the earnings section render
without the "reaction" sub-object rather than the whole panel section
erroring out.
"""
from __future__ import annotations

import bisect
import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from sovereign.data.adapter import DataUnavailable, MarketDataAdapter
from sovereign.fundamentals.types import EarningsEvent, PriceReaction

log = logging.getLogger(__name__)

_SOURCE = "computed_reaction"
_ATR_WINDOW = 20
_BMO = "bmo"
_AMC = "amc"


def _session_map(df: pd.DataFrame) -> tuple[list[date], dict[date, int]]:
    """Trading-session dates in order + a date->row-index lookup."""
    dates = [ts.date() for ts in df["timestamp"]]
    return dates, {d: i for i, d in enumerate(dates)}


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]), (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def _react_index(sessions: list[date], report_date: date, report_time: str) -> Optional[int]:
    """First session on/after report_date for bmo (pre-market print, same session
    absorbs it); first session strictly after report_date for amc/unknown (print
    lands after close, the NEXT session absorbs it). unknown is treated like amc
    — see the module docstring's two-day-ambiguity note on the returned flag."""
    idx = bisect.bisect_left(sessions, report_date)
    if report_time == _BMO:
        react_idx = idx
    else:
        # amc or unknown: strictly after report_date.
        if idx < len(sessions) and sessions[idx] == report_date:
            react_idx = idx + 1
        else:
            react_idx = idx  # sessions[idx] (if any) is already > report_date
    if react_idx < 0 or react_idx >= len(sessions):
        return None
    return react_idx


def compute_reactions(
    ticker: str,
    events: list[EarningsEvent],
    bars_source: Optional[MarketDataAdapter] = None,
) -> list[PriceReaction]:
    """One PriceReaction per event that has already reported (report_date +
    eps_actual both set) and lands inside fetchable bar history. Returns []
    if bars are unavailable — see module docstring."""
    reported = [e for e in events if e.report_date is not None and e.eps_actual is not None]
    if not reported:
        return []

    # use_cache=False is deliberate and load-bearing.
    #
    # MarketDataAdapter's cache is keyed on the symbol-DAY and, on a miss, it
    # issues one fetch PER CALENDAR DAY in the range (see adapter.get_bars).
    # That is the right shape for its intended 1-min intraday pulls, but here
    # the range spans years of DAILY bars: ~1,100 days x 2 symbols became ~2,200
    # sequential HTTP calls and build_panel('AAPL') did not finish inside 600s.
    #
    # Disabling the cache takes the adapter's single ranged _fetch_bars path
    # instead: exactly one request per symbol. We do NOT change the adapter --
    # its docstring carries an explicit SCOPE BOUNDARY against extension, and
    # its per-day granularity is correct for the callers it was built for.
    adapter = bars_source or MarketDataAdapter(use_cache=False)

    earliest = min(e.report_date for e in reported)
    latest = max(e.report_date for e in reported)
    # Buffer: ATR20_pre needs ~20 trading sessions (~30 calendar days incl.
    # weekends/holidays) before the earliest report, plus one more session for
    # the shift(1) prev_close in the TR calc; d5_pct needs ~7 calendar days
    # after the latest report to cover 5 trading sessions.
    start = (earliest - timedelta(days=55)).isoformat()
    end = (latest + timedelta(days=12)).isoformat()

    try:
        tkr_bars = adapter.get_bars(ticker, start, end, timeframe="1d")
        spy_bars = adapter.get_bars("SPY", start, end, timeframe="1d")
    except DataUnavailable as e:
        log.info("compute_reactions(%s): bars unavailable, returning []: %s", ticker, e)
        return []

    if tkr_bars is None or tkr_bars.empty:
        log.info("compute_reactions(%s): no bars in [%s, %s]", ticker, start, end)
        return []

    sessions, idx_by_date = _session_map(tkr_bars)
    tr = _true_range(tkr_bars)
    atr20 = tr.rolling(_ATR_WINDOW, min_periods=_ATR_WINDOW).mean()

    spy_sessions, spy_idx_by_date = (_session_map(spy_bars) if spy_bars is not None and not spy_bars.empty else ([], {}))

    out: list[PriceReaction] = []
    for ev in reported:
        react_idx = _react_index(sessions, ev.report_date, ev.report_time)
        if react_idx is None or react_idx == 0:
            continue  # no prior session to anchor prev_close on, or no session found at all
        prev_idx = react_idx - 1
        prev_close = tkr_bars["close"].iloc[prev_idx]
        if prev_close in (None, 0) or pd.isna(prev_close):
            continue

        react_row = tkr_bars.iloc[react_idx]
        react_date = sessions[react_idx]
        gap_pct = (react_row["open"] / prev_close - 1) * 100 if not pd.isna(react_row["open"]) else None
        d0_pct = (react_row["close"] / prev_close - 1) * 100 if not pd.isna(react_row["close"]) else None

        d1_idx = react_idx + 1
        d1_pct = (
            (tkr_bars["close"].iloc[d1_idx] / prev_close - 1) * 100
            if d1_idx < len(tkr_bars) and not pd.isna(tkr_bars["close"].iloc[d1_idx])
            else None
        )
        d5_idx = react_idx + 5
        d5_pct = (
            (tkr_bars["close"].iloc[d5_idx] / prev_close - 1) * 100
            if d5_idx < len(tkr_bars) and not pd.isna(tkr_bars["close"].iloc[d5_idx])
            else None
        )

        d0_excess_spy = None
        if d0_pct is not None and react_date in spy_idx_by_date:
            spy_react_idx = spy_idx_by_date[react_date]
            if spy_react_idx > 0:
                spy_prev_close = spy_bars["close"].iloc[spy_react_idx - 1]
                spy_close = spy_bars["close"].iloc[spy_react_idx]
                if spy_prev_close not in (None, 0) and not pd.isna(spy_prev_close) and not pd.isna(spy_close):
                    spy_d0_pct = (spy_close / spy_prev_close - 1) * 100
                    d0_excess_spy = d0_pct - spy_d0_pct

        atr20_pre_val = atr20.iloc[prev_idx]
        atr20_pre = float(atr20_pre_val) if not pd.isna(atr20_pre_val) else None
        gap_over_atr = None
        if gap_pct is not None and atr20_pre is not None and prev_close:
            atr20_pre_pct = (atr20_pre / prev_close) * 100
            if atr20_pre_pct:
                gap_over_atr = gap_pct / atr20_pre_pct

        out.append(PriceReaction(
            source=_SOURCE,
            published_ts=None,  # computed, not sourced — carries no independent publish instant
            ticker=ticker.upper(),
            report_date=ev.report_date,
            react_date=react_date,
            gap_pct=gap_pct,
            d0_pct=d0_pct,
            d1_pct=d1_pct,
            d5_pct=d5_pct,
            d0_excess_spy=d0_excess_spy,
            atr20_pre=atr20_pre,
            gap_over_atr=gap_over_atr,
        ))

    return out
