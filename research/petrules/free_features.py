"""Free features + the earnings-surprise LABEL — all with strict filing/publication gating.

Every value leaves this module wrapped in a Provenanced (value, source, published_ts). The
publication timestamp is ALWAYS the filing/print date, never the transaction date or the
period-of-report date. Downstream, assert_knowable() enforces published_ts < freeze_ts.

What is built here (free / public):
  LABEL   earnings_surprise           — AV EARNINGS (reportedEPS - estimatedEPS), ts=reportedDate
  T2 feat earnings_surprise_history   — prior-quarter surprises knowable before freeze
  T1 feat disclosed_form4_cluster     — >=3 Form-4 filings within 30d, gated by filing date
  T1 feat activist_disclosure_recent  — 13D/G filed within 90d, gated by filing date
  T1 feat institutional_accumulation  — latest 13F, gated by FILING date (not period end)

Consensus baseline (options implied move / skew) and revision-path features are PAID —
they live as stubs in paid_stubs.py and are never approximated here.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

from .provenance import Provenanced
from . import sources


# ---------------------------------------------------------------- time helpers

def _d(s: str) -> Optional[date]:
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _dt(d: date) -> datetime:
    return datetime(d.year, d.month, d.day)


def freeze_ts_for(reported: date, report_time: str) -> datetime:
    """The conservative freeze instant BEFORE an earnings print becomes public.

    post-market : print lands after the close of `reported` → freeze at 00:00 of `reported`
                  (all of reported-1 and earlier is knowable; strict `<` excludes same-day).
    pre-market  : print lands before the open of `reported` → freeze at 00:00 of `reported`
                  is still before the ~pre-open print, and strict `<` excludes any same-day
                  filing. (Conservative: we never let `reported`-dated data into the example.)
    """
    return _dt(reported)


# ---------------------------------------------------------------- earnings events + label

def earnings_events(ticker: str):
    """List of normalized earnings events for a ticker (newest first), or [] if unavailable.

    Each event: {ticker, fiscal_end(date), reported(date), reported_time,
                 estimated_eps(float|None), reported_eps(float|None), surprise(float|None)}.
    """
    payload = sources.av_earnings(ticker)
    if not payload:
        return []
    out = []
    for q in payload.get("quarterlyEarnings", []):
        reported = _d(q.get("reportedDate"))
        if reported is None:
            continue
        def _f(x):
            try:
                return float(x)
            except (ValueError, TypeError):
                return None
        out.append({
            "ticker": ticker,
            "fiscal_end": _d(q.get("fiscalDateEnding")),
            "reported": reported,
            "reported_time": q.get("reportTime", ""),
            "estimated_eps": _f(q.get("estimatedEPS")),
            "reported_eps": _f(q.get("reportedEPS")),
            "surprise": _f(q.get("surprise")),
        })
    return out


def label_earnings_surprise(event: dict) -> Provenanced:
    """The divergence LABEL for an earnings event: reportedEPS - estimatedEPS.

    published_ts = reportedDate (the print). By construction this is AT/AFTER freeze — the
    label is forward. The audit asserts label_ts >= freeze_ts (a same-bar label would fail).
    """
    surprise = event.get("surprise")
    if surprise is None and event.get("reported_eps") is not None and event.get("estimated_eps") is not None:
        surprise = event["reported_eps"] - event["estimated_eps"]
    return Provenanced(value=surprise, source="alpha_vantage.EARNINGS",
                       published_ts=_dt(event["reported"]))


# ---------------------------------------------------------------- T2: earnings surprise history

def feature_earnings_surprise_history(all_events: list, freeze_ts: datetime,
                                      n_quarters: int = 4) -> Provenanced:
    """Prior-quarters' beat/miss knowable strictly before freeze.

    Contributors are prior earnings whose reportedDate < freeze_ts. The value is a compact
    summary (count of prior beats, mean surprise) over up to n_quarters most-recent priors.
    published_ts = the MAX contributing reportedDate (the latest public moment that fed the
    feature) — always < freeze by the filter, so the whole feature is knowable-at-freeze.
    """
    priors = sorted(
        (e for e in all_events if _dt(e["reported"]) < freeze_ts and e.get("surprise") is not None),
        key=lambda e: e["reported"], reverse=True,
    )[:n_quarters]
    if not priors:
        return Provenanced(value=None, source="alpha_vantage.EARNINGS", published_ts=None)
    beats = sum(1 for e in priors if e["surprise"] > 0)
    mean_surprise = sum(e["surprise"] for e in priors) / len(priors)
    latest_contrib = max(_dt(e["reported"]) for e in priors)
    return Provenanced(
        value={"n_priors": len(priors), "prior_beats": beats, "mean_surprise": round(mean_surprise, 4)},
        source="alpha_vantage.EARNINGS",
        published_ts=latest_contrib,
    )


# ---------------------------------------------------------------- T1: disclosed-flow (EDGAR)

def _filings_of(sub: dict, form_prefixes):
    out = []
    if not sub:
        return out
    for form, fd, rd, acc in sources.edgar_recent_filings(sub):
        if any(form == p or form.startswith(p) for p in form_prefixes):
            f = _d(fd)
            if f is not None:
                out.append((form, f, _d(rd)))
    return out


def feature_form4_cluster(sub: Optional[dict], freeze_ts: datetime,
                          window_days: int = 30, min_filings: int = 3) -> Provenanced:
    """Form-4 cluster: >=`min_filings` Form-4 filings within `window_days`, ALL gated by
    filing date < freeze. published_ts = the last cluster filing date (its public moment).

    Note: direction (buy/sell) requires parsing the Form-4 XML body — a Phase-1+ extension
    (see PHASE1_GROUNDWORK.md). This builder measures cluster PRESENCE with correct gating.
    """
    if sub is None:
        return Provenanced(None, "sec_edgar.form4", None)
    dates = sorted(f for _, f, _ in _filings_of(sub, ["4"]) if _dt(f) < freeze_ts)
    if len(dates) < min_filings:
        return Provenanced({"cluster": False}, "sec_edgar.form4",
                           _dt(dates[-1]) if dates else None)
    # any trailing window ending at a filing date holding >= min_filings filings
    best_end = None
    for i in range(len(dates)):
        w = [d for d in dates if 0 <= (dates[i] - d).days <= window_days]
        if len(w) >= min_filings:
            best_end = dates[i]
    present = best_end is not None
    return Provenanced(
        value={"cluster": present, "window_days": window_days, "min_filings": min_filings},
        source="sec_edgar.form4",
        published_ts=_dt(best_end) if present else _dt(dates[-1]),
    )


def feature_activist_disclosure_recent(sub: Optional[dict], freeze_ts: datetime,
                                       lookback_days: int = 90) -> Provenanced:
    """13D/G filed within `lookback_days` before freeze (the filing IS the catalyst)."""
    if sub is None:
        return Provenanced(None, "sec_edgar.13dg", None)
    horizon = freeze_ts - timedelta(days=lookback_days)
    dates = sorted(
        f for form, f, _ in _filings_of(sub, ["SC 13D", "SC 13G", "SCHEDULE 13D", "SCHEDULE 13G"])
        if _dt(f) < freeze_ts and _dt(f) >= horizon
    )
    present = bool(dates)
    return Provenanced(
        value={"activist_recent": present, "n": len(dates)},
        source="sec_edgar.13dg",
        published_ts=_dt(dates[-1]) if present else None,
    )


def feature_institutional_accumulation(sub: Optional[dict], freeze_ts: datetime) -> Provenanced:
    """Latest 13F-HR knowable at freeze, gated by FILING date — never the period-of-report.

    The Phase-0 audit's 13F trap: a Q1 (period end Mar-31) 13F is filed ~45 days later
    (~May-15). Using the period end as the timestamp would leak ~45 days of hindsight. This
    builder times the value by filingDate; strict `<` freeze enforces it structurally.
    """
    if sub is None:
        return Provenanced(None, "sec_edgar.13f", None)
    filings = [(f, rd) for form, f, rd in _filings_of(sub, ["13F-HR"]) if _dt(f) < freeze_ts]
    if not filings:
        return Provenanced({"has_13f": False}, "sec_edgar.13f", None)
    latest_filing, period = max(filings, key=lambda x: x[0])
    return Provenanced(
        value={"has_13f": True, "period_of_report": period.isoformat() if period else None},
        source="sec_edgar.13f",
        published_ts=_dt(latest_filing),  # FILING date, not `period`
    )
