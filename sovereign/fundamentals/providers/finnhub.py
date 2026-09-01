"""Finnhub provider — STUBBED.

Same idiom as fmp.py and research/petrules/paid_stubs.py: NotImplementedError
naming the source, cost, coverage window, and required published_ts semantics.
Finnhub is the second paid candidate — kept as a distinct stub (rather than
merging with fmp.py) so registry.py's FUNDAMENTALS_FALLBACK can name either
independently once one is wired.
"""
from __future__ import annotations

from datetime import date

from sovereign.fundamentals.providers.base import FundamentalsProvider
from sovereign.fundamentals.types import (
    BorrowPoint,
    EarningsEvent,
    EstimateSnapshot,
    InsiderTxn,
    InstitutionalPosition,
    SECTIONS,
    ShortInterestPoint,
    ShortVolumePoint,
)

# Source : Finnhub.io REST API (finnhub.io)
# Cost   : free tier is rate-capped (60 calls/min) and excludes /stock/revenue-estimate,
#          /stock/eps-estimate revision history, and forward guidance — those need the
#          paid "All-In-One" plan (~$50+/mo, quote varies).
# PIT    : /stock/earnings gives `period` (fiscal period end) and no separate publish
#          timestamp for the print itself; /calendar/earnings gives a `date` that IS the
#          report date. Revision endpoints (paid tier) carry their own per-row `period`,
#          which the live implementation must map to a true revision-publish instant before
#          it can serve consensus_revision_momentum-style features — do not fabricate one.
_FINNHUB = ("Finnhub.io — paid All-In-One tier for guidance/estimate-revision history "
            "(free tier excludes those endpoints and rate-caps at 60 calls/min). "
            "Attach published_ts = the true revision/publish instant per row, never a period end.")


class FinnhubProvider(FundamentalsProvider):
    name = "finnhub"

    def capabilities(self) -> frozenset[str]:
        return SECTIONS

    def earnings_history(self, ticker: str, cik: int | None = None, limit: int = 20) -> list[EarningsEvent]:
        raise NotImplementedError(f"PAID STUB FinnhubProvider.earnings_history({ticker}): {_FINNHUB}")

    def estimate_snapshot(self, ticker: str) -> list[EstimateSnapshot]:
        raise NotImplementedError(f"PAID STUB FinnhubProvider.estimate_snapshot({ticker}): {_FINNHUB}")

    def insider_transactions(self, ticker: str, cik: int, since: date, max_filings: int = 40) -> list[InsiderTxn]:
        raise NotImplementedError(f"PAID STUB FinnhubProvider.insider_transactions({ticker}): {_FINNHUB}")

    def institutional_positions(self, ticker: str, cusip: str | None = None, quarters: int = 4) -> list[InstitutionalPosition]:
        raise NotImplementedError(f"PAID STUB FinnhubProvider.institutional_positions({ticker}): {_FINNHUB}")

    def short_interest(self, ticker: str, since: date) -> list[ShortInterestPoint]:
        raise NotImplementedError(f"PAID STUB FinnhubProvider.short_interest({ticker}): {_FINNHUB}")

    def short_volume(self, ticker: str, since: date) -> list[ShortVolumePoint]:
        raise NotImplementedError(f"PAID STUB FinnhubProvider.short_volume({ticker}): {_FINNHUB}")

    def borrow(self, ticker: str) -> list[BorrowPoint]:
        raise NotImplementedError(f"PAID STUB FinnhubProvider.borrow({ticker}): {_FINNHUB}")
