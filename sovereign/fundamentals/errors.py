"""Error types for the fundamentals layer.

The distinction that matters here is between "I fetched and there is genuinely
nothing" and "I could not fetch". A provider method returns an empty list ONLY
for the first case; the second raises ``SectionUnavailable``. That is what lets
the panel say "no insider buys in 180 days" rather than showing an empty chart
that silently means "the source broke three weeks ago".

Mirrors the contract in ``research/petrules/free_features.py``, which records an
ABSENT provenanced value rather than fabricating one.
"""
from __future__ import annotations


class FundamentalsError(Exception):
    """Base for everything in this package."""


class SectionUnavailable(FundamentalsError):
    """A named section could not be fetched. Never means 'no data exists'."""

    def __init__(self, section: str, provider: str, reason: str):
        self.section = section
        self.provider = provider
        self.reason = reason
        super().__init__(f"{section} unavailable from {provider}: {reason}")


class TickerUnresolved(FundamentalsError):
    """Ticker could not be mapped to a CIK."""


class BudgetExhausted(FundamentalsError):
    """A hard per-run call budget was hit. Raised by us, not by the API.

    Alpha Vantage's free tier is 25 requests/day. We refuse the 21st call in
    code rather than discovering the ceiling from a 200 response containing an
    error string.
    """
