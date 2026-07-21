"""Paid-data interfaces — STUBBED. Wiring each is a one-file change (this file).

Per the spec's public-data-only hard constraint and the Phase-0 NARROW verdict, the
consensus baseline (options implied move / skew) and the analyst revision path are paid.
They are NOT approximated with a looser free source — a fake number in the organ whose
product is confidence is exactly the failure mode the Gate exists to avoid.

Each stub raises NotImplementedError naming the source, cost, coverage window, and the
exact provenance timestamp the live implementation MUST attach so the leakage audit keeps
its teeth once the source is live.
"""
from __future__ import annotations

from datetime import datetime

from .provenance import Provenanced

# --------------------------------------------------------------------------- ThetaData
# Source : ThetaData options EOD (ThetaTerminal local API, :25503/v3)
# Cost   : VALUE tier already on file → 2020-01→present. STANDARD (~$/mo) extends to 2012+.
# PIT    : event-dated chains carry bid/ask AS OF date → published_ts = that quote date.
_THETA = ("ThetaData options EOD — VALUE tier (2020-01→present) on file; STANDARD extends "
          "history to 2012. Attach published_ts = chain snapshot (quote) date at T-1 close.")


def options_implied_move(ticker: str, freeze_ts: datetime) -> Provenanced:
    """CONSENSUS BASELINE: ATM straddle / spot at T-1 → priced-in expected move.

    Unlocks: the consensus_move_pct baseline for BOTH earnings and non-earnings setups
    (the Phase-0 fallback after the free revision path died). Without this the Gate has no
    priced-in reference and cannot compute divergence.
    """
    raise NotImplementedError(f"PAID STUB options_implied_move({ticker}): {_THETA}")


def options_skew_direction(ticker: str, freeze_ts: datetime) -> Provenanced:
    """T1 feature: put/call skew sign vs consensus direction (ThetaData chain at T-1)."""
    raise NotImplementedError(f"PAID STUB options_skew_direction({ticker}): {_THETA}")


def implied_move_change_30d(ticker: str, freeze_ts: datetime) -> Provenanced:
    """T1 feature (pre-reg REPLACEMENT for the dead consensus_revision_momentum): change in
    options-implied move over T-30..T-1 — the only honest 'expectation dynamics' signal that
    survived the Phase-0 audit. ThetaData two-date chain read; published_ts = later quote date.
    """
    raise NotImplementedError(f"PAID STUB implied_move_change_30d({ticker}): {_THETA}")


# --------------------------------------------------------------------------- Revision vintages
# Source : Refinitiv I/B/E/S or Zacks (Nasdaq Data Link premium)
# Cost   : out of current budget (Refinitiv enterprise / Zacks premium dataset)
# PIT    : each estimate revision carries its own publication date → published_ts = revision date.
_REVISION = ("Refinitiv I/B/E/S or Zacks (NDL premium) — paid, out of budget. No free source "
             "exposes the point-in-time revision PATH (Phase-0 Q1). Attach published_ts = each "
             "estimate's revision publication date.")


def consensus_revision_momentum(ticker: str, freeze_ts: datetime) -> Provenanced:
    """DROPPED from Tier 1 by the Phase-0 narrowing (no free PIT revision history). Kept as a
    stub so a future paid pre-reg update can wire it without restructuring the builder.

    Unlocks: analyst-revision velocity/direction over 30/60/90d — the original Tier-1 signal
    the audit disqualified as a silent lookahead when approximated from today's snapshot.
    """
    raise NotImplementedError(f"PAID STUB consensus_revision_momentum({ticker}): {_REVISION}")
