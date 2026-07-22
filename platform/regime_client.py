"""regime_client — the tiny reader every strategy calls before it sizes.

Usage (importable from BOTH ict/ and sovereign/ — platform imports neither):

    from platform.regime_client import get_regime
    r = get_regime("carry")               # reads system_regime_state.json
    if r.stale or r.verdict == "STAND_ASIDE":
        skip()                            # never trade a stale/adverse regime
    size *= r.size_multiplier             # Connected Edge x Regime, enforced

Design contract (matches scripts/obsidian_sync.py discipline):
  * Fail loud, never fake. If the contract file is missing, unreadable, stale,
    or the section is absent/degraded, the returned RegimeRead is STAND_ASIDE
    with ``favorable=False``, ``size_multiplier=0.0``, ``stale=True`` and a
    human-readable reason. A missing regime read must NEVER present as a
    favorable one.
  * This module never raises to its caller. A strategy asking "what's the
    regime?" always gets an answer it can act on safely.

The reader imports only the standard library.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Repo root is two levels up from this file (platform/regime_client.py -> repo/).
REPO = Path(__file__).resolve().parent.parent

# The one canonical contract file, written by scripts/build_system_regime.py.
CONTRACT_PATH = Path(
    os.environ.get(
        "ALTA_REGIME_STATE",
        str(REPO / "data" / "agent" / "system_regime_state.json"),
    )
)

# How old the whole contract may be before every read is forced STALE. The
# writer runs every 30 min; 90 min (3 missed runs) is the safety cut. This is
# a client-side staleness guard, independent of the writer's per-section ages.
CONTRACT_MAX_AGE_HOURS = float(os.environ.get("ALTA_REGIME_MAX_AGE_HOURS", "1.5"))

STAND_ASIDE = "STAND_ASIDE"


@dataclass
class RegimeRead:
    """What a strategy gets back. Safe-by-default: an empty/unknown read is
    STAND_ASIDE, not GO."""

    strategy: str
    verdict: str = STAND_ASIDE
    favorable: bool = False
    size_multiplier: float = 0.0
    stale: bool = True
    reason: str = "not yet read"
    source: str | None = None
    source_age_hours: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"RegimeRead(strategy={self.strategy!r}, verdict={self.verdict!r}, "
            f"favorable={self.favorable}, size_multiplier={self.size_multiplier}, "
            f"stale={self.stale}, reason={self.reason!r})"
        )


def _stand_aside(strategy: str, reason: str) -> RegimeRead:
    return RegimeRead(
        strategy=strategy,
        verdict=STAND_ASIDE,
        favorable=False,
        size_multiplier=0.0,
        stale=True,
        reason=reason,
    )


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_contract(path: Path | None = None) -> dict[str, Any] | None:
    """Read the raw contract JSON. Returns None on any failure (never raises)."""
    p = Path(path) if path is not None else CONTRACT_PATH
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def get_regime(strategy: str, path: Path | None = None) -> RegimeRead:
    """Return the current regime read for ``strategy``.

    Safe-by-default: any missing/stale/degraded condition yields a STAND_ASIDE
    read with size_multiplier 0.0 and stale=True. Never raises.
    """
    contract = load_contract(path)
    if contract is None:
        return _stand_aside(
            strategy,
            f"contract file unreadable or missing at {CONTRACT_PATH}",
        )

    # Whole-file staleness guard.
    generated = _parse_ts(contract.get("generated_at"))
    if generated is None:
        return _stand_aside(strategy, "contract has no valid generated_at timestamp")
    age_h = (datetime.now(timezone.utc) - generated).total_seconds() / 3600.0
    contract_stale = age_h > CONTRACT_MAX_AGE_HOURS

    strategies = contract.get("strategies") or {}
    section = strategies.get(strategy)
    if not isinstance(section, dict):
        return _stand_aside(
            strategy, f"no '{strategy}' section in contract (known: {sorted(strategies)})"
        )

    status = str(section.get("status", "UNAVAILABLE")).upper()
    verdict = str(section.get("verdict", STAND_ASIDE)).upper()
    reason = str(section.get("reason", "no reason given"))
    source = section.get("source")
    src_age = section.get("source_age_hours")
    detail = section.get("detail") or {}

    section_stale = status in {"STALE", "UNAVAILABLE", "DEGRADED"}
    stale = contract_stale or section_stale

    if stale:
        why = reason
        if contract_stale:
            why = f"contract stale ({age_h:.1f}h old > {CONTRACT_MAX_AGE_HOURS}h); {reason}"
        return RegimeRead(
            strategy=strategy,
            verdict=STAND_ASIDE,
            favorable=False,
            size_multiplier=0.0,
            stale=True,
            reason=why,
            source=source,
            source_age_hours=src_age,
            detail=detail,
        )

    # Fresh, non-degraded section: trust its verdict/favorable/size, but clamp
    # size to 0.0 if the verdict is STAND_ASIDE regardless of what was written.
    favorable = bool(section.get("favorable", False))
    size_multiplier = float(section.get("size_multiplier", 0.0) or 0.0)
    if verdict == STAND_ASIDE:
        favorable = False
        size_multiplier = 0.0

    return RegimeRead(
        strategy=strategy,
        verdict=verdict,
        favorable=favorable,
        size_multiplier=size_multiplier,
        stale=False,
        reason=reason,
        source=source,
        source_age_hours=src_age,
        detail=detail,
    )
