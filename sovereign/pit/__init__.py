"""Point-in-time storage: what was knowable, when.

Every record carries two timestamps — when the event happened, and when it
became knowable — and every read states the instant it is pretending to be at.

    from sovereign.pit import as_of, view

    v = view(as_of("2026-03-03"))
    rows = v.facts("earnings", "AAPL")     # only what was public on March 3

There is no default as-of and no implicit "now": `as_of(None)` raises. A fact
with no publication instant cannot be read at all rather than being read
optimistically. Writes are append-only, so a restatement adds a vintage instead
of destroying the observation it revises — without that, no amount of filtering
can answer "what did I know then".

See spec.py for the per-fact temporal contract, including the facts that are
currently BLOCKED because their source gives us no publication instant.
"""
from __future__ import annotations

from sovereign.pit.clock import AsOf, as_of, as_of_now
from sovereign.pit.errors import (
    AsOfRequired,
    LookaheadError,
    NotPointInTime,
    PitError,
    UnknownFact,
)
from sovereign.pit.reader import AsOfReader, Observation, view
from sovereign.pit.spec import FACTS, FactSpec, blocked_facts, point_in_time_facts

__all__ = [
    "AsOf",
    "AsOfReader",
    "AsOfRequired",
    "FACTS",
    "FactSpec",
    "LookaheadError",
    "NotPointInTime",
    "Observation",
    "PitError",
    "UnknownFact",
    "as_of",
    "as_of_now",
    "blocked_facts",
    "point_in_time_facts",
    "view",
]
