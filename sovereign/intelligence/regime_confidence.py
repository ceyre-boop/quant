"""
sovereign/intelligence/regime_confidence.py

Merges internal regime detection (cross_system_state.json) with external
TradingView regime signals (data/agent/tv_regime_signals.json).

Pure read-only. No side effects. Called by:
  - sovereign/oracle/pulse_check.py   (_check_regime_alignment anomaly)
  - sovereign/oracle/reflect_cycle.py  (regime section added to Oracle prompt)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRIDGE_STATE_PATH = ROOT / "data" / "forensics" / "cross_system_state.json"
TV_SIGNALS_PATH   = ROOT / "data" / "agent" / "tv_regime_signals.json"

_TV_WINDOW_HOURS = 2


@dataclass
class RegimeConfidence:
    internal_regime: str     # MOMENTUM | REVERSION | FLAT | UNKNOWN
    external_regime: str     # TRENDING | RANGING | UNKNOWN
    agreement: bool
    confidence: float        # 0.0–1.0
    sizing_multiplier: float # 0.5 | 0.75 | 1.0
    tv_signal_count: int
    tv_signal_age_min: float # minutes since newest signal; -1 if none
    reason: str


def _internal_from_state(state: dict) -> str:
    primary = (state.get("library_primary_regime") or "").upper()
    threat  = float(state.get("library_threat_score", 0.0))
    ict     = (state.get("ict_mode") or "NORMAL").upper()

    if ict == "HALT_NEW" or threat >= 0.80:
        return "FLAT"
    if primary in ("WARNING", "DANGER"):
        return "REVERSION"
    if primary == "CRITICAL":
        return "FLAT"
    if threat >= 0.60:
        return "REVERSION"
    return "MOMENTUM"


def _external_from_signals(signals: list[dict]) -> tuple[str, int, float]:
    """Return (external_regime, count_recent, age_min_newest)."""
    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=_TV_WINDOW_HOURS)

    recent: list[tuple[datetime, dict]] = []
    for s in signals:
        try:
            ts = datetime.fromisoformat(s.get("timestamp", "").replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                recent.append((ts, s))
        except Exception:
            continue

    if not recent:
        return "UNKNOWN", 0, -1.0

    recent.sort(key=lambda x: x[0], reverse=True)
    age_min = (now - recent[0][0]).total_seconds() / 60.0

    trending = sum(1 for _, s in recent if s.get("regime", "").startswith("TRENDING"))
    ranging  = sum(1 for _, s in recent if s.get("regime", "") in ("RANGING", "CHoCH"))

    if trending > ranging:
        return "TRENDING", len(recent), age_min
    if ranging > trending:
        return "RANGING", len(recent), age_min

    # Tie — use most recent
    latest = recent[0][1].get("regime", "UNKNOWN")
    if latest.startswith("TRENDING"):
        return "TRENDING", len(recent), age_min
    if latest in ("RANGING", "CHoCH"):
        return "RANGING", len(recent), age_min
    return "UNKNOWN", len(recent), age_min


# (internal, external) → (agreement, confidence, sizing_multiplier)
_MATRIX: dict[tuple[str, str], tuple[bool, float, float]] = {
    ("MOMENTUM",  "TRENDING"): (True,  0.90, 1.00),
    ("REVERSION", "RANGING"):  (True,  0.85, 1.00),
    ("FLAT",      "TRENDING"): (True,  0.75, 0.75),
    ("FLAT",      "RANGING"):  (True,  0.75, 0.75),
    ("FLAT",      "UNKNOWN"):  (True,  0.70, 0.75),
    ("MOMENTUM",  "RANGING"):  (False, 0.40, 0.50),
    ("REVERSION", "TRENDING"): (False, 0.40, 0.50),
    ("MOMENTUM",  "UNKNOWN"):  (True,  0.65, 0.75),
    ("REVERSION", "UNKNOWN"):  (True,  0.60, 0.75),
}


def score_regime_confidence() -> RegimeConfidence:
    """Read bridge state + TV signals and return regime alignment. Never raises."""
    try:
        state = json.loads(BRIDGE_STATE_PATH.read_text()) if BRIDGE_STATE_PATH.exists() else {}
    except Exception:
        state = {}

    try:
        raw = json.loads(TV_SIGNALS_PATH.read_text()) if TV_SIGNALS_PATH.exists() else []
        signals = raw if isinstance(raw, list) else []
    except Exception:
        signals = []

    internal = _internal_from_state(state)
    external, count, age_min = _external_from_signals(signals)

    agreement, confidence, mult = _MATRIX.get((internal, external), (True, 0.60, 0.75))

    if external == "UNKNOWN":
        reason = (
            f"No TV signals in last {_TV_WINDOW_HOURS}h — "
            f"internal-only ({internal}), sizing at {mult:.0%}"
        )
    elif agreement:
        reason = (
            f"Internal ({internal}) and TradingView ({external}) agree — "
            f"confidence {confidence:.0%}, sizing at {mult:.0%}"
        )
    else:
        reason = (
            f"REGIME CONFLICT: internal={internal} vs TradingView={external}. "
            f"Reducing to {mult:.0%} size until regimes align."
        )

    return RegimeConfidence(
        internal_regime=internal,
        external_regime=external,
        agreement=agreement,
        confidence=confidence,
        sizing_multiplier=mult,
        tv_signal_count=count,
        tv_signal_age_min=round(age_min, 1),
        reason=reason,
    )
