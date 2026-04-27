"""
Shared forex strategy constants and helpers.

This module exists to keep live scan and backtest logic anchored to the
same thresholds instead of letting them drift in separate files.
"""
from __future__ import annotations

from dataclasses import dataclass


CONVICTION_NEUTRAL_THRESHOLD = 0.35
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
