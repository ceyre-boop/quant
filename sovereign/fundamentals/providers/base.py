"""Abstract provider interface for the fundamentals layer.

Every concrete provider (``free.py``, and the paid stubs ``fmp.py`` /
``finnhub.py``) implements this ABC. ``registry.py`` is the only place that
picks a concrete class; everything else — ``reaction.py``, ``panel.py``, the
harvester — talks to a ``FundamentalsProvider`` and never imports a concrete
provider directly.

CRITICAL CONTRACT (repeated on every method below, not just here, because it is
the one rule this whole package exists to enforce): a method returns ``[]``
ONLY when it successfully reached the source and the source genuinely has
nothing to say — e.g. a company with zero insider Form 4s in the lookback
window, or a ticker with no official short-interest print yet. If the method
could not reach or parse the source (network failure, missing credentials, no
dataset ingested, throttled response, unresolvable ticker) it raises
``sovereign.fundamentals.errors.SectionUnavailable`` instead. Conflating the
two — returning [] for a failure — is exactly the silent-emptiness bug
``errors.py``'s docstring warns about: the panel would render "no insider buys"
when the true state is "we don't know", and a consumer downstream (a backtest,
an Oracle reflection, a trader) would treat absence-of-evidence as
evidence-of-absence.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from sovereign.fundamentals.types import (
    BorrowPoint,
    EarningsEvent,
    EstimateSnapshot,
    InsiderTxn,
    InstitutionalPosition,
    ShortInterestPoint,
    ShortVolumePoint,
)


class FundamentalsProvider(ABC):
    """One fundamentals data source (composed of transports, or a paid vendor)."""

    name: str

    @abstractmethod
    def capabilities(self) -> frozenset[str]:
        """Subset of ``sovereign.fundamentals.types.SECTIONS`` this provider can
        actually serve. ``registry.py`` and ``panel.py`` use this to populate the
        panel's top-level ``capabilities`` list and to decide, before calling a
        method, whether a gap should read "not supported by this provider" versus
        "supported but the fetch failed"."""
        raise NotImplementedError

    @abstractmethod
    def earnings_history(
        self, ticker: str, cik: int | None = None, limit: int = 20
    ) -> list[EarningsEvent]:
        """Past (and near-future, estimate-only) earnings prints, most recent first.
        [] means the ticker genuinely has no earnings history reachable from this
        source (e.g. a brand-new listing). Raises SectionUnavailable on fetch
        failure."""
        raise NotImplementedError

    @abstractmethod
    def estimate_snapshot(self, ticker: str) -> list[EstimateSnapshot]:
        """Tonight's forward-consensus snapshot (one row per period bucket:
        0q/+1q/0y/+1y). [] means no forward estimates exist for this ticker from
        this source (small/illiquid names are commonly like this). Raises
        SectionUnavailable on fetch failure."""
        raise NotImplementedError

    @abstractmethod
    def insider_transactions(
        self, ticker: str, cik: int, since: date, max_filings: int = 40
    ) -> list[InsiderTxn]:
        """Form 4 open-market and administrative transactions since ``since``.
        [] means zero Form 4s were filed in the window — a real, common state for
        a quiet issuer. Raises SectionUnavailable if the filing index or filing
        bodies could not be fetched."""
        raise NotImplementedError

    @abstractmethod
    def institutional_positions(
        self, ticker: str, cusip: str | None = None, quarters: int = 4
    ) -> list[InstitutionalPosition]:
        """13F holdings across the last ``quarters`` reporting periods. []
        means the dataset was reachable but this ticker has no institutional
        holders on file (implausible for a liquid name — more likely signals a
        CUSIP-resolution gap upstream, which callers should treat as suspicious).
        Raises SectionUnavailable when the underlying 13F dataset itself has not
        been ingested at all — that is a "we don't know" state, not a "zero
        holders" state."""
        raise NotImplementedError

    @abstractmethod
    def short_interest(self, ticker: str, since: date) -> list[ShortInterestPoint]:
        """Official bimonthly short-interest prints since ``since``. []
        means no prints exist in the window for this ticker (thinly-traded or
        newly-listed names sometimes have none). Raises SectionUnavailable on
        fetch failure."""
        raise NotImplementedError

    @abstractmethod
    def short_volume(self, ticker: str, since: date) -> list[ShortVolumePoint]:
        """Daily FINRA Reg SHO short-volume rows since ``since``. []
        means the ticker had zero matching rows across the fetched day files
        (e.g. it didn't trade). Raises SectionUnavailable on fetch failure."""
        raise NotImplementedError

    @abstractmethod
    def borrow(self, ticker: str) -> list[BorrowPoint]:
        """Most recent IB locate/borrow snapshot for this ticker, as a
        single-element list (or [] if the ticker is simply absent from the most
        recent snapshot file — NOT_LISTED is a real, common tier). Raises
        SectionUnavailable if no snapshot file exists to read at all."""
        raise NotImplementedError
