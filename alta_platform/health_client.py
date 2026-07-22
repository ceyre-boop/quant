"""health_client — the tiny reader every strategy calls alongside get_regime().

Usage (importable from BOTH ict/ and sovereign/ — alta_platform imports neither):

    from alta_platform.health_client import get_health
    h = get_health("ict_equities")        # reads system_health_verdict.json
    if h.kill_switch == "HALT" or h.stale:
        skip()                            # never trade on a HALT/stale conscience
    # REDUCE → size down; TRADE → full. The regime organ is the other gate.

Design contract (matches regime_client.py discipline):
  * Fail loud, never fake, fail SAFE. If the verdict file is missing, unreadable,
    stale, or the section is absent/degraded, the returned HealthRead is HALT with
    ``stale=True`` and a human-readable reason. Absent evidence is NEVER a TRADE.
  * An unproven edge (edge_health status INSUFFICIENT_DATA) surfaces as
    ``edge_divergence=False`` but a REDUCE kill switch — small, never full.
  * This module never raises to its caller.

Imports only the standard library.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

VERDICT_PATH = Path(
    os.environ.get(
        "ALTA_HEALTH_VERDICT",
        str(REPO / "data" / "agent" / "system_health_verdict.json"),
    )
)

# Whole-file staleness guard (writer runs every 30 min; 1.5h = 3 missed runs).
VERDICT_MAX_AGE_HOURS = float(os.environ.get("ALTA_HEALTH_MAX_AGE_HOURS", "1.5"))

TRADE = "TRADE"
REDUCE = "REDUCE"
HALT = "HALT"


@dataclass
class HealthRead:
    """What a strategy gets back. Safe-by-default: an empty/unknown read is HALT."""

    strategy: str
    kill_switch: str = HALT
    edge_divergence: bool = False
    stale: bool = True
    reason: str = "not yet read"
    edge_status: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"HealthRead(strategy={self.strategy!r}, kill_switch={self.kill_switch!r}, "
            f"edge_divergence={self.edge_divergence}, stale={self.stale}, "
            f"reason={self.reason!r})"
        )


def _halt(strategy: str, reason: str) -> HealthRead:
    return HealthRead(
        strategy=strategy,
        kill_switch=HALT,
        edge_divergence=False,
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


def load_verdict(path: Path | None = None) -> dict[str, Any] | None:
    """Read the raw verdict JSON. Returns None on any failure (never raises)."""
    p = Path(path) if path is not None else VERDICT_PATH
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def get_health(strategy: str, path: Path | None = None) -> HealthRead:
    """Return the current health read for ``strategy``.

    Safe-by-default: any missing/stale/degraded condition yields a HALT read with
    stale=True. Never raises.
    """
    verdict = load_verdict(path)
    if verdict is None:
        return _halt(strategy, f"health verdict unreadable or missing at {VERDICT_PATH}")

    generated = _parse_ts(verdict.get("generated_at"))
    if generated is None:
        return _halt(strategy, "health verdict has no valid generated_at timestamp")
    age_h = (datetime.now(timezone.utc) - generated).total_seconds() / 3600.0
    file_stale = age_h > VERDICT_MAX_AGE_HOURS

    strategies = verdict.get("strategies") or {}
    section = strategies.get(strategy)
    if not isinstance(section, dict):
        return _halt(
            strategy,
            f"no '{strategy}' section in verdict (known: {sorted(strategies)})",
        )

    kill_switch = str(section.get("kill_switch", HALT)).upper()
    reason = str(section.get("reason", "no reason given"))
    edge = section.get("edge_health") or {}
    edge_status = str(edge.get("status")) if edge.get("status") is not None else None
    edge_divergence = bool(edge.get("divergence_flag", False))

    if file_stale:
        # A stale conscience cannot vouch for anything — force HALT.
        return HealthRead(
            strategy=strategy,
            kill_switch=HALT,
            edge_divergence=edge_divergence,
            stale=True,
            reason=f"health verdict stale ({age_h:.1f}h old > {VERDICT_MAX_AGE_HOURS}h); {reason}",
            edge_status=edge_status,
            detail=section,
        )

    # Guard: an unknown kill-switch value is treated as HALT (fail safe).
    if kill_switch not in {TRADE, REDUCE, HALT}:
        return HealthRead(
            strategy=strategy,
            kill_switch=HALT,
            edge_divergence=edge_divergence,
            stale=False,
            reason=f"unknown kill_switch {kill_switch!r} — treated as HALT (fail-safe)",
            edge_status=edge_status,
            detail=section,
        )

    return HealthRead(
        strategy=strategy,
        kill_switch=kill_switch,
        edge_divergence=edge_divergence,
        stale=False,
        reason=reason,
        edge_status=edge_status,
        detail=section,
    )
