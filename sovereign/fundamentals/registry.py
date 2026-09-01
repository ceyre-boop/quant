"""Provider selection for the fundamentals layer.

Mirrors the env-selection idiom in sovereign/data/adapter.py's
``MarketDataAdapter`` (``DATA_PRIMARY``/``DATA_FALLBACK`` + logged, never-silent
fallback) — but does NOT import from that module. adapter.py's own docstring
states an explicit SCOPE BOUNDARY: it is a bar-transport seam and must not be
extended for other purposes. Fundamentals providers are a different domain
(filings, not bars) with a different contract (SectionUnavailable per-section,
not a whole-adapter DataUnavailable), so the pattern is replicated here rather
than the class reused.

``FUNDAMENTALS_PRIMARY`` defaults to "free" (the only provider with a real
implementation today). ``FUNDAMENTALS_FALLBACK`` defaults to "" (no fallback) —
until a paid key is bought, chaining to fmp/finnhub would just turn every
free-provider gap into a guaranteed NotImplementedError, which is strictly
worse than the honest SectionUnavailable the primary already raises.
"""
from __future__ import annotations

import logging
import os
from datetime import date
from typing import Optional

from sovereign.fundamentals.errors import SectionUnavailable
from sovereign.fundamentals.providers.base import FundamentalsProvider
from sovereign.fundamentals.transports.alphavantage import CallBudget
from sovereign.fundamentals.types import (
    BorrowPoint,
    EarningsEvent,
    EstimateSnapshot,
    InsiderTxn,
    InstitutionalPosition,
    ShortInterestPoint,
    ShortVolumePoint,
)

log = logging.getLogger(__name__)

_DEFAULT_PRIMARY = "free"
_DEFAULT_FALLBACK = ""

_PROVIDERS: dict[str, str] = {
    "free": "sovereign.fundamentals.providers.free.FreeProvider",
    "fmp": "sovereign.fundamentals.providers.fmp.FMPProvider",
    "finnhub": "sovereign.fundamentals.providers.finnhub.FinnhubProvider",
}


def _import(dotted: str):
    module_path, cls_name = dotted.rsplit(".", 1)
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _build(name: str, av_budget: Optional[CallBudget]) -> FundamentalsProvider:
    if name not in _PROVIDERS:
        raise ValueError(f"unknown fundamentals provider {name!r}; have {sorted(_PROVIDERS)}")
    cls = _import(_PROVIDERS[name])
    if name == "free":
        return cls(av_budget=av_budget)
    return cls()


class _ChainedProvider(FundamentalsProvider):
    """Primary, falling back to a secondary on any exception — logged loudly.

    Every method below follows the same shape: try primary, and on ANY
    exception (not just SectionUnavailable — a provider bug should not take
    the whole chain down silently either) log a warning naming the real
    exception, then try the fallback. If both fail, the fallback's exception
    propagates (it is usually the more specific one, e.g. a paid stub's
    NotImplementedError naming exactly what would unlock the section)."""

    def __init__(self, primary: FundamentalsProvider, fallback: FundamentalsProvider):
        self._primary = primary
        self._fallback = fallback
        self.name = f"{primary.name}+{fallback.name}"

    def capabilities(self) -> frozenset[str]:
        return self._primary.capabilities() | self._fallback.capabilities()

    def _try(self, op: str, primary_call, fallback_call):
        try:
            return primary_call()
        except Exception as e:  # noqa: BLE001 - fallback is the point; never silent
            log.warning(
                "fundamentals %s: primary %s failed (%s: %s) -> falling back to %s",
                op, self._primary.name, type(e).__name__, e, self._fallback.name,
            )
            return fallback_call()

    def earnings_history(self, ticker, cik=None, limit=20):
        return self._try(
            "earnings_history",
            lambda: self._primary.earnings_history(ticker, cik, limit),
            lambda: self._fallback.earnings_history(ticker, cik, limit),
        )

    def estimate_snapshot(self, ticker):
        return self._try(
            "estimate_snapshot",
            lambda: self._primary.estimate_snapshot(ticker),
            lambda: self._fallback.estimate_snapshot(ticker),
        )

    def insider_transactions(self, ticker, cik, since, max_filings=40):
        return self._try(
            "insider_transactions",
            lambda: self._primary.insider_transactions(ticker, cik, since, max_filings),
            lambda: self._fallback.insider_transactions(ticker, cik, since, max_filings),
        )

    def institutional_positions(self, ticker, cusip=None, quarters=4):
        return self._try(
            "institutional_positions",
            lambda: self._primary.institutional_positions(ticker, cusip, quarters),
            lambda: self._fallback.institutional_positions(ticker, cusip, quarters),
        )

    def short_interest(self, ticker, since):
        return self._try(
            "short_interest",
            lambda: self._primary.short_interest(ticker, since),
            lambda: self._fallback.short_interest(ticker, since),
        )

    def short_volume(self, ticker, since):
        return self._try(
            "short_volume",
            lambda: self._primary.short_volume(ticker, since),
            lambda: self._fallback.short_volume(ticker, since),
        )

    def borrow(self, ticker):
        return self._try(
            "borrow",
            lambda: self._primary.borrow(ticker),
            lambda: self._fallback.borrow(ticker),
        )


def get_provider(
    primary: str | None = None,
    fallback: str | None = None,
    av_budget: Optional[CallBudget] = None,
) -> FundamentalsProvider:
    """Return the configured provider (or a chained primary->fallback pair).

    ``av_budget`` is threaded through to FreeProvider so callers (the harvester)
    can hand it a single shared CallBudget across every ticker in a run — the
    25-call/day Alpha Vantage ceiling is per RUN, not per ticker.
    """
    primary_name = (primary or os.getenv("FUNDAMENTALS_PRIMARY", _DEFAULT_PRIMARY)).lower()
    fallback_name = (fallback or os.getenv("FUNDAMENTALS_FALLBACK", _DEFAULT_FALLBACK)).lower()

    primary_provider = _build(primary_name, av_budget)
    if not fallback_name or fallback_name == primary_name:
        return primary_provider

    fallback_provider = _build(fallback_name, av_budget)
    return _ChainedProvider(primary_provider, fallback_provider)
