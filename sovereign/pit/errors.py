"""Failure modes of the point-in-time layer.

These are deliberately loud. A point-in-time store that degrades quietly is
worse than none at all: it produces a backtest that looks fine and is fiction.
"""
from __future__ import annotations


class PitError(Exception):
    """Base for everything in sovereign.pit."""


class LookaheadError(PitError, AssertionError):
    """A value was used at an instant before it was knowable.

    Subclasses AssertionError to match research/petrules/provenance.py, so the
    two layers raise something a caller can catch uniformly.
    """


class AsOfRequired(PitError):
    """A read was attempted without an as-of instant.

    There is no default and there is deliberately no "now". Defaulting to now is
    how a research query silently becomes a live query: it works on your desk,
    passes review, and is wrong for every historical bar it touches.

    Mirrors backtester/holdout_guard.py, where an unbounded end date is treated
    as a violation rather than as "everything".
    """


class NotPointInTime(PitError):
    """This fact has no publication instant, so it cannot be read as-of.

    Raised rather than returning rows, because the alternative is to hand back
    data whose knowability is unknown — which is exactly the silent leak this
    layer exists to make impossible. The fix is to give the source a real
    publication timestamp, never to relax this.
    """


class UnknownFact(PitError):
    """No such fact is registered in the spec.

    Facts must be declared (sovereign/pit/spec.py) before they can be read, so
    an ad-hoc table name cannot slip past the temporal contract.
    """


class PitSchemaMismatch(PitError):
    """The declared spec does not match the table on disk.

    Raised instead of returning no rows. An as-of read that answers "nothing was
    knowable" when the real answer is "the query is malformed" is indistinguishable
    from a true negative, and would quietly hollow out a backtest.
    """
