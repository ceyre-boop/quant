"""Financial Modeling Prep provider — STUBBED.

Wiring this is a one-file change once a key is bought: replace the
NotImplementedError bodies with real HTTP calls behind the same signatures,
routed through httpcache the way the free transports are.

Per the same discipline as research/petrules/paid_stubs.py: a stub raises
NotImplementedError naming the source, its cost, its coverage window, and the
required published_ts semantics, rather than silently degrading to a free
approximation. FMP is the one candidate free.py cannot match on two sections:
forward guidance and the point-in-time analyst-revision path (capabilities()
below is the reason those two land in SECTIONS at all).
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

# Source : Financial Modeling Prep REST API (financialmodelingprep.com)
# Cost   : Starter ~$29/mo (250 calls/day) up to Ultimate for full guidance/estimate history
# PIT    : /earning_calendar and /analyst-estimates carry a `date` field that is the ESTIMATE
#          period, not a publication instant — the live implementation MUST resolve the true
#          filing/publish timestamp per row (FMP does not expose one directly for guidance;
#          the /earning-call-transcript endpoint's `date` is the closest true publish time for
#          guidance language) or published_ts must stay None rather than be faked from the
#          period date.
_FMP = ("Financial Modeling Prep — paid, starting ~$29/mo (Starter tier, 250 calls/day); "
        "guidance/estimates history needs Premium+ for point-in-time revision vintages. "
        "Attach published_ts = the true filing/publish instant per row, never the period date.")


class FMPProvider(FundamentalsProvider):
    name = "fmp"

    def capabilities(self) -> frozenset[str]:
        return SECTIONS  # a paid key buys everything free.py cannot serve, incl. guidance/revisions

    def earnings_history(self, ticker: str, cik: int | None = None, limit: int = 20) -> list[EarningsEvent]:
        raise NotImplementedError(f"PAID STUB FMPProvider.earnings_history({ticker}): {_FMP}")

    def estimate_snapshot(self, ticker: str) -> list[EstimateSnapshot]:
        raise NotImplementedError(f"PAID STUB FMPProvider.estimate_snapshot({ticker}): {_FMP}")

    def insider_transactions(self, ticker: str, cik: int, since: date, max_filings: int = 40) -> list[InsiderTxn]:
        raise NotImplementedError(f"PAID STUB FMPProvider.insider_transactions({ticker}): {_FMP}")

    def institutional_positions(self, ticker: str, cusip: str | None = None, quarters: int = 4) -> list[InstitutionalPosition]:
        raise NotImplementedError(f"PAID STUB FMPProvider.institutional_positions({ticker}): {_FMP}")

    def short_interest(self, ticker: str, since: date) -> list[ShortInterestPoint]:
        raise NotImplementedError(f"PAID STUB FMPProvider.short_interest({ticker}): {_FMP}")

    def short_volume(self, ticker: str, since: date) -> list[ShortVolumePoint]:
        raise NotImplementedError(f"PAID STUB FMPProvider.short_volume({ticker}): {_FMP}")

    def borrow(self, ticker: str) -> list[BorrowPoint]:
        raise NotImplementedError(f"PAID STUB FMPProvider.borrow({ticker}): {_FMP}")
