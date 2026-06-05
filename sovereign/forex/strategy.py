"""
Shared forex strategy constants and helpers.

This module exists to keep live scan and backtest logic anchored to the
same thresholds instead of letting them drift in separate files.
"""
from __future__ import annotations

from dataclasses import dataclass


CONVICTION_NEUTRAL_THRESHOLD = 0.10  # authorized 2026-06-05 — see data/agent/param_change_log.jsonl
CONVICTION_FULL_SIZE = 0.70
CONVICTION_MAX_SIZE = 0.85

TARGET_RR_T1 = 2.0
TARGET_RR_T2 = 3.0
TARGET_RR_T3 = 5.0


@dataclass(frozen=True)
class MacroScoreBreakdown:
    raw_score: float
    conviction: float
    direction: str


def direction_from_score(raw_score: float) -> str:
    if raw_score > 0:
        return 'LONG'
    if raw_score < 0:
        return 'SHORT'
    return 'NEUTRAL'


def conviction_from_score(raw_score: float) -> float:
    return min(abs(raw_score), 1.0)


def grade_from_signal(rate_differential: float, conviction: float) -> str:
    """
    Grade assignment from combat-rules analysis. Rate diff in percent-points (e.g. 3.60).

    A+ : |rate_diff| >= 2.0 AND conviction >= 0.60  (B-001: strong diff + momentum confirms)
    A  : |rate_diff| >= 1.5                          (strong differential)
    B  : |rate_diff| >= 0.5                          (baseline — above C-005 weak-signal zone)
    C  : |rate_diff| <  0.5                          (weak rate signal — risk engine sizes at 0.25%)
    """
    abs_diff = abs(rate_differential)
    if abs_diff >= 2.0 and conviction >= 0.60:
        return 'A+'
    if abs_diff >= 1.5:
        return 'A'
    if abs_diff >= 0.5:
        return 'B'
    return 'C'
