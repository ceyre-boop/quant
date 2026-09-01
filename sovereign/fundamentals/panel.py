"""sovereign/fundamentals/panel.py — the fundamentals portability seam.

``build_panel(ticker)`` returns EXACTLY the JSON shape app/src/lib/fundamentals.ts
expects (that file is the wire contract; this module is the only writer of it).
Every other caller — the CLI harvester's ``--emit-json``, a future http.server
route, a future Vercel function — is a thin wrapper around this one function.
Nothing downstream should ever build the panel shape itself.

Section discipline (mirrors errors.py's contract): every section in
``sections`` ALWAYS has ``as_of`` / ``staleness_days`` / ``sources`` / ``gaps``,
even when the underlying provider raised ``SectionUnavailable`` — the section
is still present with an explanatory string in ``gaps`` and empty rows. A
section is never omitted and never silently empty; "we tried and there is
nothing" and "we could not try" must stay visually distinguishable in the UI.

Store-first: each section reads sovereign.fundamentals.store first (populated
by scripts/harvest_fundamentals.py / scripts/harvest_13f_bulk.py) and only
reaches for a live provider fetch when the store has nothing AND
``warm_only`` is False. ``warm_only=True`` (used by static-artifact builds
that must not touch the network) serves whatever is cached and reports a gap
for anything that isn't.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional

from sovereign.fundamentals.errors import SectionUnavailable, TickerUnresolved
from sovereign.fundamentals.reaction import compute_reactions
from sovereign.fundamentals.registry import get_provider
from sovereign.fundamentals.store import DB_PATH, connect
from sovereign.fundamentals.transports import sec
from sovereign.fundamentals.types import EarningsEvent, InsiderTxn, InstitutionalPosition, PriceReaction


def _ro_connect():
    """read-only connect() that degrades to None instead of raising when the
    fundamentals DB hasn't been created yet (a fresh checkout before the first
    harvest run) — DuckDB refuses to open a nonexistent file read-only."""
    if not DB_PATH.exists():
        return None
    return connect(read_only=True)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
EARNINGS_CAP = 20
INSIDER_LOOKBACK_DAYS = 180
INSIDER_CAP = 200
HOLDERS_CAP = 25
SHORT_VOLUME_DAYS = 90

# ── Request-path cost budget ────────────────────────────────────────────────
# build_panel() backs an HTTP route, so it must answer in seconds, not minutes.
# Two sections are expensive enough to blow that budget and are therefore
# STORE-ONLY here; the nightly harvester is what populates them:
#
#   short_volume  FINRA publishes one file PER DAY containing the WHOLE market.
#                 A 90-day live pull is ~62 multi-MB downloads for one ticker.
#                 The harvester fetches each day-file once for every watchlist
#                 ticker at the same time, which is the only sane shape.
#   institutions  13F arrives as multi-GB quarterly bulk ZIPs.
#
# Insider IS fetched live but tightly bounded: each Form 4 is a separate XML
# under www.sec.gov/Archives and SEC fair-access caps us at 5 req/s, so 40
# filings is ~8s of pure waiting. 12 covers a recent window; the harvester
# still walks the full 40 for warm tickers.
#
# Measured: an unbounded build_panel('AAPL') did not finish inside 600s.
LIVE_INSIDER_MAX_FILINGS = 12
HARVEST_INSIDER_MAX_FILINGS = 40


def _now() -> datetime:
    return datetime.utcnow()


def _staleness(as_of: Optional[date]) -> Optional[int]:
    if as_of is None:
        return None
    return (date.today() - as_of).days


def _empty_section(gap: str) -> dict:
    return {"as_of": None, "staleness_days": None, "sources": [], "gaps": [gap], "rows": []}


# ── store readers (each returns None when the store has nothing, [] is a ─────
#    valid "empty but present" result the caller must not treat as absence) ──

def _store_earnings(ticker: str) -> Optional[list[EarningsEvent]]:
    con = _ro_connect()
    if con is None:
        return None
    with con:
        rows = con.execute(
            """
            SELECT ticker, fiscal_end, report_date, report_time, eps_estimate, eps_actual,
                   eps_surprise, eps_surprise_pct, rev_estimate, rev_actual, guide_eps_low,
                   guide_eps_high, eps_actual_gaap, source, published_ts
            FROM fund_earnings_event WHERE ticker = ? ORDER BY report_date DESC LIMIT ?
            """,
            [ticker.upper(), EARNINGS_CAP],
        ).fetchall()
    if not rows:
        return None
    out = []
    for r in rows:
        out.append(EarningsEvent(
            source=r[13], published_ts=r[14], ticker=r[0], fiscal_end=r[1], report_date=r[2],
            report_time=r[3] or "unknown", eps_estimate=r[4], eps_actual=r[5], eps_surprise=r[6],
            eps_surprise_pct=r[7], rev_estimate=r[8], rev_actual=r[9], guide_eps_low=r[10],
            guide_eps_high=r[11], eps_actual_gaap=r[12],
        ))
    return out


def _store_insider(ticker: str, since: date) -> Optional[list[InsiderTxn]]:
    # published_ts is aliased FROM filing_date rather than stored twice: for a
    # Form 4 the filing date IS the moment the transaction became knowable, and
    # a second column holding the same fact is a divergence waiting to happen.
    con = _ro_connect()
    if con is None:
        return None
    with con:
        rows = con.execute(
            """
            SELECT accession, line_no, ticker, issuer_cik, owner_name, owner_title, is_director,
                   is_officer, is_ten_pct, txn_date, filing_date, code, shares, price, value_usd,
                   shares_after, source, filing_date AS published_ts
            FROM fund_insider_txn WHERE ticker = ? AND filing_date >= ?
            ORDER BY filing_date DESC LIMIT ?
            """,
            [ticker.upper(), since, INSIDER_CAP],
        ).fetchall()
    if not rows:
        return None
    out = []
    for r in rows:
        out.append(InsiderTxn(
            source=r[16], published_ts=r[17], ticker=r[2], issuer_cik=r[3] or 0, accession=r[0],
            line_no=r[1], owner_name=r[4] or "", owner_title=r[5] or "", is_director=bool(r[6]),
            is_officer=bool(r[7]), is_ten_pct=bool(r[8]), txn_date=r[9], filing_date=r[10],
            code=r[11] or "", shares=r[12], price=r[13], value_usd=r[14], shares_after=r[15],
        ))
    return out


def _store_reactions(ticker: str) -> dict[date, PriceReaction]:
    con = _ro_connect()
    if con is None:
        return {}
    with con:
        rows = con.execute(
            """
            SELECT report_date, react_date, gap_pct, d0_pct, d1_pct, d5_pct, d0_excess_spy,
                   atr20_pre, gap_over_atr, bars_source
            FROM fund_price_reaction WHERE ticker = ?
            """,
            [ticker.upper()],
        ).fetchall()
    out: dict[date, PriceReaction] = {}
    for r in rows:
        out[r[0]] = PriceReaction(
            source=r[9] or "computed_reaction", published_ts=None, ticker=ticker.upper(),
            report_date=r[0], react_date=r[1], gap_pct=r[2], d0_pct=r[3], d1_pct=r[4], d5_pct=r[5],
            d0_excess_spy=r[6], atr20_pre=r[7], gap_over_atr=r[8],
        )
    return out


def _store_short_interest(ticker: str, since: date):
    con = _ro_connect()
    if con is None:
        return []
    with con:
        return con.execute(
            """
            SELECT settlement_date, shares_short, days_to_cover, source
            FROM fund_short_interest WHERE ticker = ? AND settlement_date >= ?
            ORDER BY settlement_date DESC
            """,
            [ticker.upper(), since],
        ).fetchall()


def _store_short_volume(ticker: str, since: date):
    con = _ro_connect()
    if con is None:
        return []
    with con:
        return con.execute(
            """
            SELECT date, short_pct, source FROM fund_short_volume_daily
            WHERE ticker = ? AND date >= ? ORDER BY date DESC LIMIT ?
            """,
            [ticker.upper(), since, SHORT_VOLUME_DAYS],
        ).fetchall()


def _store_borrow(ticker: str):
    con = _ro_connect()
    if con is None:
        return None
    with con:
        return con.execute(
            """
            SELECT date, tier, fee_rate, source FROM fund_borrow
            WHERE ticker = ? ORDER BY date DESC LIMIT 1
            """,
            [ticker.upper()],
        ).fetchone()


def _store_name(ticker: str) -> Optional[str]:
    con = _ro_connect()
    if con is None:
        return None
    with con:
        row = con.execute(
            "SELECT name FROM fund_symbol WHERE ticker = ?", [ticker.upper()]
        ).fetchone()
    return row[0] if row and row[0] else None


# ── ticker identity ───────────────────────────────────────────────────────

def _resolve_cik(ticker: str) -> Optional[int]:
    try:
        return sec.resolve_cik(ticker)
    except TickerUnresolved:
        return None


def _ticker_name(ticker: str, cik: Optional[int], warm_only: bool) -> Optional[str]:
    stored = _store_name(ticker)
    if stored:
        return stored
    if warm_only or cik is None:
        return None
    try:
        return sec.submissions(cik).get("name")
    except SectionUnavailable:
        return None


# ── earnings section ──────────────────────────────────────────────────────

def _build_earnings(ticker: str, provider, warm_only: bool) -> dict:
    events = _store_earnings(ticker)
    gap: Optional[str] = None
    if events is None:
        if warm_only:
            return _empty_section("warm_only: no cached earnings data")
        try:
            events = provider.earnings_history(ticker, limit=EARNINGS_CAP)
        except SectionUnavailable as e:
            return _empty_section(str(e))

    events = events[:EARNINGS_CAP]

    # Reaction join is a live/cached bars fetch (MarketDataAdapter) — skipped
    # under warm_only so a static-artifact build never touches the network.
    reactions_by_report_date = _store_reactions(ticker)
    if not warm_only:
        try:
            computed = compute_reactions(ticker, events)
            for r in computed:
                if r.report_date is not None:
                    reactions_by_report_date[r.report_date] = r
        except Exception as e:  # noqa: BLE001 - reaction is derived; never break the panel over it
            log.info("panel(%s): reaction computation failed, continuing without it: %s", ticker, e)

    rows = []
    sources = set()
    for e in events:
        sources.add(e.source)
        reaction = reactions_by_report_date.get(e.report_date) if e.report_date else None
        rows.append({
            "fiscal_end": e.fiscal_end.isoformat() if e.fiscal_end else None,
            "report_date": e.report_date.isoformat() if e.report_date else None,
            "report_time": e.report_time,
            "eps_estimate": e.eps_estimate,
            "eps_actual": e.eps_actual,
            "eps_surprise": e.eps_surprise,
            "eps_surprise_pct": e.eps_surprise_pct,
            "rev_estimate": e.rev_estimate,
            "rev_actual": e.rev_actual,
            "guide_eps_low": e.guide_eps_low,
            "guide_eps_high": e.guide_eps_high,
            "reaction": ({
                "gap_pct": reaction.gap_pct, "d0_pct": reaction.d0_pct, "d1_pct": reaction.d1_pct,
                "d5_pct": reaction.d5_pct, "d0_excess_spy": reaction.d0_excess_spy,
                "gap_over_atr": reaction.gap_over_atr,
            } if reaction else None),
            "source": e.source,
        })

    # as_of must be the most recent REPORTED print, not the scheduled next one.
    # Including the future date produced 'as of 2026-10-29 -58d' -- negative
    # staleness, which reads as nonsense. Freshness here means "how long since
    # this company last told us something", and a date that has not happened yet
    # cannot answer that.
    as_of = max((e.report_date for e in events
                 if e.report_date and e.eps_actual is not None), default=None)
    return {
        "as_of": as_of.isoformat() if as_of else None,
        "staleness_days": _staleness(as_of),
        "sources": sorted(sources),
        "gaps": [gap] if gap else [],
        "rows": rows,
    }


# ── insider section ─────────────────────────────────────────────────────

def _build_insider(ticker: str, cik: Optional[int], provider, warm_only: bool,
                   max_filings: int = LIVE_INSIDER_MAX_FILINGS) -> dict:
    since = date.today() - timedelta(days=INSIDER_LOOKBACK_DAYS)
    txns = _store_insider(ticker, since)
    if txns is None:
        if warm_only:
            return {**_empty_section("warm_only: no cached insider data"),
                    "summary": None}
        if cik is None:
            return {**_empty_section(f"{ticker} could not be resolved to a CIK"),
                    "summary": None}
        try:
            txns = provider.insider_transactions(ticker, cik, since, max_filings=max_filings)
        except SectionUnavailable as e:
            return {**_empty_section(str(e)), "summary": None}

    txns = [t for t in txns if t.filing_date is None or t.filing_date >= since][:INSIDER_CAP]

    rows = []
    sources = set()
    buys = sells = 0
    net_shares = 0.0
    net_value = 0.0
    any_shares = any_value = False
    for t in txns:
        sources.add(t.source)
        rows.append({
            "owner_name": t.owner_name, "owner_title": t.owner_title,
            "txn_date": t.txn_date.isoformat() if t.txn_date else None,
            "filing_date": t.filing_date.isoformat() if t.filing_date else None,
            "code": t.code, "shares": t.shares, "price": t.price, "value_usd": t.value_usd,
            "is_open_market": t.is_open_market,
        })
        if not t.is_open_market:
            continue
        if t.code == "P":
            buys += 1
        elif t.code == "S":
            sells += 1
        if t.shares is not None:
            net_shares += t.shares
            any_shares = True
        if t.value_usd is not None:
            # shares is already signed (negative for dispositions) in the
            # transport layer; value_usd = shares*price inherits that sign.
            net_value += t.value_usd
            any_value = True

    as_of = max((t.filing_date for t in txns if t.filing_date), default=None)
    return {
        "as_of": as_of.isoformat() if as_of else None,
        "staleness_days": _staleness(as_of),
        "sources": sorted(sources),
        "gaps": [],
        "summary": {
            "buys_180d": buys, "sells_180d": sells,
            "net_shares_180d": net_shares if any_shares else None,
            "net_value_usd_180d": net_value if any_value else None,
            "counts_only": False,
        },
        "rows": rows,
    }


# ── institutions section ────────────────────────────────────────────────

def _build_institutions(ticker: str, provider, warm_only: bool) -> dict:
    base = {
        "as_of": None, "staleness_days": None, "sources": [], "gaps": [],
        "period_end": None, "filing_date_max": None, "n_holders": None,
        "d_holders_qoq": None, "total_shares": None, "d_shares_qoq": None,
        "top_buyers": [], "top_sellers": [], "rows": [],
    }
    try:
        positions: list[InstitutionalPosition] = provider.institutional_positions(ticker, quarters=4)
    except SectionUnavailable as e:
        base["gaps"] = [str(e)]
        return base

    if not positions:
        base["gaps"] = ["no institutional holders on file for this ticker"]
        return base

    periods = sorted({p.period_end for p in positions if p.period_end}, reverse=True)
    if not periods:
        base["gaps"] = ["institutional rows present but none carry a period_end"]
        return base

    current_period = periods[0]
    prior_period = periods[1] if len(periods) > 1 else None

    current = {p.filer_cik: p for p in positions if p.period_end == current_period}
    prior = {p.filer_cik: p for p in positions if p.period_end == prior_period} if prior_period else {}

    n_holders = len(current)
    total_shares = sum(p.shares or 0 for p in current.values())
    filing_date_max = max((p.filing_date for p in current.values() if p.filing_date), default=None)
    sources = sorted({p.source for p in current.values()})

    d_holders_qoq = d_shares_qoq = None
    top_buyers: list[dict] = []
    top_sellers: list[dict] = []
    if prior_period is not None:
        d_holders_qoq = n_holders - len(prior)
        prior_total = sum(p.shares or 0 for p in prior.values())
        d_shares_qoq = total_shares - prior_total

        deltas = []
        for cik_key, p in current.items():
            prior_shares = (prior[cik_key].shares or 0) if cik_key in prior else 0
            d_shares = (p.shares or 0) - prior_shares
            deltas.append((p.filer_name, p.shares, d_shares))
        # Closed positions (in prior, absent from current) are sellers of their full stake.
        for cik_key, pp in prior.items():
            if cik_key not in current:
                deltas.append((pp.filer_name, 0, -(pp.shares or 0)))

        buyers = sorted([d for d in deltas if d[2] > 0], key=lambda d: d[2], reverse=True)
        sellers = sorted([d for d in deltas if d[2] < 0], key=lambda d: d[2])
        top_buyers = [{"filer_name": n, "shares": s, "d_shares": d} for n, s, d in buyers[:HOLDERS_CAP]]
        top_sellers = [{"filer_name": n, "shares": s, "d_shares": d} for n, s, d in sellers[:HOLDERS_CAP]]
    else:
        top_buyers = [
            {"filer_name": p.filer_name, "shares": p.shares, "d_shares": None}
            for p in sorted(current.values(), key=lambda p: p.shares or 0, reverse=True)[:HOLDERS_CAP]
        ]

    base.update({
        "as_of": current_period.isoformat() if current_period else None,
        "staleness_days": _staleness(current_period),
        "sources": sources,
        "gaps": ["no prior quarter on file — QoQ deltas unavailable"] if prior_period is None else [],
        "period_end": current_period.isoformat() if current_period else None,
        "filing_date_max": filing_date_max.isoformat() if filing_date_max else None,
        "n_holders": n_holders,
        "d_holders_qoq": d_holders_qoq,
        "total_shares": total_shares,
        "d_shares_qoq": d_shares_qoq,
        "top_buyers": top_buyers,
        "top_sellers": top_sellers,
        "rows": [],
    })
    return base


# ── short section ───────────────────────────────────────────────────────

def _build_short(ticker: str, provider, warm_only: bool) -> dict:
    since_interest = date.today() - timedelta(days=400)  # bimonthly prints: keep well over a year
    since_volume = date.today() - timedelta(days=SHORT_VOLUME_DAYS)

    gaps: list[str] = []
    sources: set[str] = set()

    interest_rows = _store_short_interest(ticker, since_interest)
    if not interest_rows and not warm_only:
        try:
            live = provider.short_interest(ticker, since_interest)
            interest_rows = [(r.settlement_date, r.shares_short, r.days_to_cover, r.source) for r in live]
        except SectionUnavailable as e:
            gaps.append(f"short_interest: {e}")
            interest_rows = []
    interest_rows = interest_rows or []

    # STORE-ONLY by design — see the cost budget above. Live-pulling 90 daily
    # whole-market FINRA files to answer one request is what made build_panel
    # exceed 600s. Absence here is an un-harvested ticker, and we say so.
    volume_rows = _store_short_volume(ticker, since_volume) or []
    if not volume_rows:
        gaps.append(
            "short_volume: not harvested for this ticker — FINRA publishes one "
            "whole-market file per day, so it is fetched in batch by "
            "scripts/harvest_fundamentals.py, never per request"
        )

    borrow_row = _store_borrow(ticker)
    if borrow_row is None and not warm_only:
        try:
            live = provider.borrow(ticker)
            if live:
                b = live[0]
                borrow_row = (b.date, b.tier, b.fee_rate, b.source)
        except SectionUnavailable as e:
            gaps.append(f"borrow: {e}")

    interest = [{"settlement_date": d.isoformat() if d else None, "shares_short": s, "days_to_cover": dtc}
                for d, s, dtc, _src in interest_rows]
    for _d, _s, _dtc, src in interest_rows:
        sources.add(src)

    short_volume = [{"date": d.isoformat() if d else None, "short_pct": pct}
                     for d, pct, _src in volume_rows]
    for _d, _pct, src in volume_rows:
        sources.add(src)

    borrow = None
    if borrow_row is not None:
        b_date, tier, fee_rate, src = borrow_row
        borrow = {"date": b_date.isoformat() if b_date else None, "tier": tier, "fee_rate": fee_rate}
        sources.add(src)

    as_of_candidates = [r[0] for r in interest_rows] + [r[0] for r in volume_rows]
    if borrow_row is not None and borrow_row[0] is not None:
        as_of_candidates.append(borrow_row[0])
    as_of = max((d for d in as_of_candidates if d), default=None)

    return {
        "as_of": as_of.isoformat() if as_of else None,
        "staleness_days": _staleness(as_of),
        "sources": sorted(sources),
        "gaps": gaps,
        "interest": interest,
        "short_volume": short_volume,
        "borrow": borrow,
    }


# ── top-level ────────────────────────────────────────────────────────────

def build_panel(ticker: str, warm_only: bool = False) -> dict:
    ticker = ticker.upper().strip()
    provider = get_provider()
    cik = _resolve_cik(ticker)
    name = _ticker_name(ticker, cik, warm_only)

    sections: dict[str, Any] = {
        "earnings": _build_earnings(ticker, provider, warm_only),
        "insider": _build_insider(ticker, cik, provider, warm_only),
        "institutions": _build_institutions(ticker, provider, warm_only),
        "short": _build_short(ticker, provider, warm_only),
    }

    partial = any(s.get("gaps") for s in sections.values())

    return {
        "schema_version": SCHEMA_VERSION,
        "ticker": ticker,
        "cik": cik,
        "name": name,
        "generated_at": _now().isoformat() + "Z",
        "capabilities": sorted(provider.capabilities()),
        "partial": partial,
        "sections": sections,
    }
