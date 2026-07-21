"""Provenance core — the structural mechanism that makes lookahead IMPOSSIBLE.

The Phase-0 audit's single biggest finding was that revision-path features were a silent
lookahead leak: a value that *looks* like it was knowable at time T but was actually only
published later. The defense is to refuse to carry any feature value without an attached
publication timestamp, and to gate every value against the event's freeze time with a
single, strict rule.

Rule (one line, no exceptions):
    a value is knowable at freeze iff  published_ts  <  freeze_ts   (STRICT)

Everything downstream (replay engine, feature builders, the leakage-audit test) routes
through `Provenanced` and `knowable_at`. A feature builder physically cannot emit a value
into an example without a publication timestamp, so an un-audited value cannot exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass(frozen=True)
class Provenanced:
    """A single value tagged with where it came from and WHEN it became public.

    published_ts is the moment the value became publicly knowable:
      - Form 4 / 13D / 13F : the EDGAR *filing* datetime  (never the transaction or
        period-of-report date — that is the exact leak the audit exists to catch)
      - earnings surprise  : the earnings *reportedDate*  (the print moment)
      - options implied move: the quote datetime of the chain snapshot

    published_ts is None ONLY for a genuinely-absent value (source unavailable / paid-stub
    not wired). A None-ts value is never treated as knowable and is skipped by the audit —
    it is not fabricated and it is not silently trusted.
    """
    value: Any
    source: str
    published_ts: Optional[datetime]

    def knowable_at(self, freeze_ts: datetime) -> bool:
        return knowable_at(self.published_ts, freeze_ts)

    @property
    def is_present(self) -> bool:
        return self.value is not None and self.published_ts is not None


def knowable_at(published_ts: Optional[datetime], freeze_ts: datetime) -> bool:
    """True iff a value published at `published_ts` was public STRICTLY before `freeze_ts`.

    Strict `<` is deliberate: a filing dated the same instant as the freeze is treated as
    NOT-yet-knowable (excluded), the conservative direction. A value with no publication
    timestamp is never knowable.
    """
    if published_ts is None:
        return False
    return published_ts < freeze_ts


class LookaheadError(AssertionError):
    """Raised when a feature value's publication timestamp is not strictly before freeze."""


def assert_knowable(name: str, pv: Provenanced, freeze_ts: datetime) -> None:
    """Hard guard used at build time so a leak fails loudly rather than silently training."""
    if not pv.is_present:
        return  # absent value — nothing knowable to check, and it is not used as a feature
    if not pv.knowable_at(freeze_ts):
        raise LookaheadError(
            f"LOOKAHEAD: feature '{name}' from {pv.source} published {pv.published_ts.isoformat()} "
            f"is NOT strictly before freeze {freeze_ts.isoformat()}"
        )
