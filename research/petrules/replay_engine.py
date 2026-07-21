"""Replay engine — enumerate earnings events and freeze the knowable-then state.

For each earnings event it builds a FrozenEvent: a freeze timestamp (before the print) and a
dict of Provenanced features + a Provenanced label. The engine NEVER passes a value into an
example without a publication timestamp, so an un-audited (potentially leaked) value cannot
enter the dataset — lookahead is structurally impossible, not merely tested-for.

Free features are populated. Paid features (options implied move/skew, revision momentum) are
recorded as ABSENT slots naming their stub, so wiring them later fills the slot without
changing the event shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .provenance import Provenanced, assert_knowable
from . import free_features as ff
from . import sources


# Paid feature slots recorded as ABSENT until their stub is wired (see paid_stubs.py).
PAID_FEATURE_SLOTS = {
    "options_implied_move": "paid_stubs.options_implied_move",
    "options_skew_direction": "paid_stubs.options_skew_direction",
    "implied_move_change_30d": "paid_stubs.implied_move_change_30d",
    "consensus_revision_momentum": "paid_stubs.consensus_revision_momentum",
}


@dataclass
class FrozenEvent:
    ticker: str
    event_id: str
    freeze_ts: datetime
    features: Dict[str, Provenanced]
    label: Provenanced
    meta: dict = field(default_factory=dict)

    def present_features(self) -> Dict[str, Provenanced]:
        return {k: v for k, v in self.features.items() if v.is_present}

    def audit(self) -> None:
        """Hard build-time guard: every present feature knowable-at-freeze; label forward."""
        for name, pv in self.features.items():
            assert_knowable(name, pv, self.freeze_ts)
        if self.label.is_present and self.label.published_ts < self.freeze_ts:
            from .provenance import LookaheadError
            raise LookaheadError(
                f"LABEL '{self.event_id}' published {self.label.published_ts.isoformat()} is "
                f"BEFORE freeze {self.freeze_ts.isoformat()} — label is not forward-looking"
            )


def build_events_for_ticker(ticker: str, min_priors: int = 4) -> List[FrozenEvent]:
    """Build one FrozenEvent per earnings print that has >=min_priors of prior history.

    Disclosed-flow features use live EDGAR when reachable; offline they are ABSENT (not faked).
    """
    events = ff.earnings_events(ticker)
    if not events:
        return []
    cik = sources.TICKERS_CIK.get(ticker)
    sub = sources.edgar_submissions(cik) if cik is not None else None  # None when offline

    frozen: List[FrozenEvent] = []
    for ev in events:
        freeze_ts = ff.freeze_ts_for(ev["reported"], ev["reported_time"])
        hist = ff.feature_earnings_surprise_history(events, freeze_ts, n_quarters=min_priors)
        if not hist.is_present:
            continue  # need prior history to form a real example

        features: Dict[str, Provenanced] = {
            "earnings_surprise_history": hist,
            "disclosed_form4_cluster": ff.feature_form4_cluster(sub, freeze_ts),
            "activist_disclosure_recent": ff.feature_activist_disclosure_recent(sub, freeze_ts),
            "institutional_accumulation": ff.feature_institutional_accumulation(sub, freeze_ts),
        }
        # paid slots — explicit ABSENT placeholders naming their stub
        for slot, stub in PAID_FEATURE_SLOTS.items():
            features[slot] = Provenanced(value=None, source=stub, published_ts=None)

        fe = FrozenEvent(
            ticker=ticker,
            event_id=f"{ticker}:{ev['reported'].isoformat()}",
            freeze_ts=freeze_ts,
            features=features,
            label=ff.label_earnings_surprise(ev),
            meta={"fiscal_end": ev["fiscal_end"].isoformat() if ev["fiscal_end"] else None,
                  "reported_time": ev["reported_time"],
                  "edgar_online": sub is not None},
        )
        fe.audit()  # fail loud at build time if anything leaks
        frozen.append(fe)
    return frozen


def build_sample(tickers: Optional[List[str]] = None) -> List[FrozenEvent]:
    """Build the small proof-of-pipeline sample. Offline default = the fixture tickers."""
    if tickers is None:
        tickers = ["AAPL", "DKS"]  # committed real fixtures → always builds offline
    out: List[FrozenEvent] = []
    for t in tickers:
        out.extend(build_events_for_ticker(t))
    return out
