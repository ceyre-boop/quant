"""Wire contract for the fundamentals layer.

Every row carries ``source`` and ``published_ts``. ``published_ts`` is the
FILING or PUBLICATION instant — never the transaction date, never the
period-of-report date. This is the same look-ahead discipline enforced in
``research/petrules_audit/probe_sources.py``: a Form 4 transaction on the 3rd
that is filed on the 5th was not knowable until the 5th, and keying it on the
3rd silently poisons any backtest that later consumes this data.

These dataclasses are the only thing the store, the panel builder, the harvester
and the front end all agree on. A paid provider slots in by returning these same
types; nothing downstream changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Any, Optional

# Sections a provider may advertise in capabilities().
SECTIONS = frozenset({
    "earnings", "guidance", "estimates", "revisions",
    "insider", "institutions", "short_interest", "short_volume", "borrow",
})


def _d(o: Any) -> Any:
    """JSON-safe conversion for dataclass -> dict."""
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, dict):
        return {k: _d(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_d(v) for v in o]
    return o


@dataclass(frozen=True)
class Row:
    """Provenance carried by every fetched row."""
    source: str
    published_ts: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {k: _d(v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class EarningsEvent(Row):
    ticker: str = ""
    fiscal_end: Optional[date] = None
    report_date: Optional[date] = None
    report_time: str = "unknown"        # bmo | amc | unknown
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    eps_surprise: Optional[float] = None
    eps_surprise_pct: Optional[float] = None
    rev_estimate: Optional[float] = None
    rev_actual: Optional[float] = None
    # Free tier cannot source forward guidance. These stay None by design, and
    # the panel renders the column greyed rather than omitting it, so the gap is
    # visible instead of implied. See paid_stubs.
    guide_eps_low: Optional[float] = None
    guide_eps_high: Optional[float] = None
    eps_actual_gaap: Optional[float] = None   # SEC XBRL, independent cross-check


@dataclass(frozen=True)
class EstimateSnapshot(Row):
    """One night's view of forward consensus. Accumulating these builds the
    point-in-time revision path no free source will sell us."""
    ticker: str = ""
    snapshot_date: Optional[date] = None
    period: str = ""                    # 0q | +1q | 0y | +1y
    period_end: Optional[date] = None
    eps_avg: Optional[float] = None
    eps_low: Optional[float] = None
    eps_high: Optional[float] = None
    n_analysts: Optional[int] = None
    up_30d: Optional[int] = None
    down_30d: Optional[int] = None


@dataclass(frozen=True)
class PriceReaction(Row):
    """Computed, never fetched — a join of earnings dates onto bars we already have."""
    ticker: str = ""
    report_date: Optional[date] = None
    react_date: Optional[date] = None   # the session that actually absorbed the print
    gap_pct: Optional[float] = None
    d0_pct: Optional[float] = None
    d1_pct: Optional[float] = None
    d5_pct: Optional[float] = None
    d0_excess_spy: Optional[float] = None
    atr20_pre: Optional[float] = None
    gap_over_atr: Optional[float] = None


@dataclass(frozen=True)
class InsiderTxn(Row):
    ticker: str = ""
    issuer_cik: int = 0
    accession: str = ""
    line_no: int = 0
    owner_name: str = ""
    owner_title: str = ""
    is_director: bool = False
    is_officer: bool = False
    is_ten_pct: bool = False
    txn_date: Optional[date] = None
    filing_date: Optional[date] = None  # the only date that gates knowability
    code: str = ""                      # P purchase, S sale, A grant, M option, F tax
    shares: Optional[float] = None
    price: Optional[float] = None
    value_usd: Optional[float] = None
    shares_after: Optional[float] = None

    @property
    def is_open_market(self) -> bool:
        """Grants (A) and tax withholding (F) are not decisions to buy or sell.
        Counting them as insider buying/selling makes the panel lie."""
        return self.code in ("P", "S")


@dataclass(frozen=True)
class InstitutionalPosition(Row):
    ticker: str = ""
    period_end: Optional[date] = None
    filer_cik: int = 0
    filer_name: str = ""
    filing_date: Optional[date] = None
    cusip: str = ""
    shares: Optional[float] = None
    value_usd: Optional[float] = None
    is_amendment: bool = False


@dataclass(frozen=True)
class ShortInterestPoint(Row):
    """Official bimonthly short interest. ~8 days stale at publication by
    regulation — that is a fact about the world, not a sourcing failure."""
    ticker: str = ""
    settlement_date: Optional[date] = None
    shares_short: Optional[float] = None
    avg_daily_volume: Optional[float] = None
    days_to_cover: Optional[float] = None
    pct_float: Optional[float] = None   # no free float source; stays None


@dataclass(frozen=True)
class ShortVolumePoint(Row):
    """Daily short VOLUME from FINRA. This is not short interest and must never
    be merged into it — it includes market-maker facilitation."""
    ticker: str = ""
    date: Optional[date] = None
    short_volume: Optional[float] = None
    short_exempt_volume: Optional[float] = None
    total_volume: Optional[float] = None
    short_pct: Optional[float] = None


@dataclass(frozen=True)
class BorrowPoint(Row):
    """IB's book only, not the street."""
    ticker: str = ""
    date: Optional[date] = None
    tier: str = ""
    available_shares: Optional[float] = None
    fee_rate: Optional[float] = None
