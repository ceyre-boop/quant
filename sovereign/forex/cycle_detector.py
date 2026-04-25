"""
Dalio debt cycle detector for forex economies.

Four phases based on rate/inflation/GDP trajectory:
  EARLY_EXP  → rates low, growth accelerating, currency still weak
  MID_EXP    → rates rising, growth solid, currency STRENGTHENING
  LATE_EXP   → rates peaked, growth slowing, currency AT_PEAK
  CONTRACTION → rates falling, growth negative, currency WEAKENING

The edge is in pair DIVERGENCE: long MID_EXP vs CONTRACTION.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CycleState:
    country: str
    phase: str            # EARLY_EXP / MID_EXP / LATE_EXP / CONTRACTION
    rate_trajectory: str  # HIKING / HOLDING / CUTTING
    inflation_trajectory: str  # RISING / STABLE / FALLING
    gdp_trajectory: str   # ACCELERATING / STABLE / DECELERATING
    phase_score: float    # -1 (contraction) to +1 (mid expansion)


@dataclass
class PairCycleSignal:
    pair: str
    base_cycle: CycleState
    quote_cycle: CycleState
    divergence_score: float  # base_score - quote_score, [-2, +2]
    direction: str           # LONG / SHORT / NEUTRAL


# Phase scores for arithmetic: higher = currency tailwind
_PHASE_SCORE = {
    'EARLY_EXP':   0.0,
    'MID_EXP':     1.0,
    'LATE_EXP':    0.3,
    'CONTRACTION': -1.0,
}

_TRAJECTORY_SCORE = {
    'HIKING':      0.5,
    'HOLDING':     0.0,
    'CUTTING':    -0.5,
    'ACCELERATING': 0.3,
    'STABLE':      0.0,
    'DECELERATING': -0.3,
    'RISING':      0.1,
    'FALLING':    -0.1,
}


class CycleDetector:

    def classify_economy(self, macro: dict) -> CycleState:
        """
        Classify an economy based on current macro snapshot.
        macro: dict with keys rate, cpi_yoy, gdp_growth, rate_trajectory
        """
        country = macro['country']
        rate = macro.get('rate', 2.0)
        cpi = macro.get('cpi_yoy', 2.0)
        gdp = macro.get('gdp_growth', 1.5)
        real_rate = macro.get('real_rate', rate - cpi)
        trajectory = macro.get('rate_trajectory', [0, 0, 0])

        rate_traj = self._rate_trajectory(trajectory)
        inflation_traj = self._inflation_trajectory(cpi)
        gdp_traj = self._gdp_trajectory(gdp)
        phase = self._classify_phase(rate, cpi, gdp, real_rate, rate_traj)

        phase_score = (
            _PHASE_SCORE[phase]
            + _TRAJECTORY_SCORE[rate_traj]
            + _TRAJECTORY_SCORE[gdp_traj]
        )

        return CycleState(
            country=country,
            phase=phase,
            rate_trajectory=rate_traj,
            inflation_trajectory=inflation_traj,
            gdp_trajectory=gdp_traj,
            phase_score=round(phase_score, 3),
        )

    def score_pair(
        self, pair: str, base_macro: dict, quote_macro: dict
    ) -> PairCycleSignal:
        base_cycle = self.classify_economy(base_macro)
        quote_cycle = self.classify_economy(quote_macro)

        div = base_cycle.phase_score - quote_cycle.phase_score

        if div > 0.5:
            direction = 'LONG'
        elif div < -0.5:
            direction = 'SHORT'
        else:
            direction = 'NEUTRAL'

        return PairCycleSignal(
            pair=pair,
            base_cycle=base_cycle,
            quote_cycle=quote_cycle,
            divergence_score=round(div, 3),
            direction=direction,
        )

    # ── Trajectory classifiers ──────────────────────────────────────── #

    @staticmethod
    def _rate_trajectory(decisions: List[int]) -> str:
        """decisions: list of recent CB moves [1=hike, 0=hold, -1=cut]"""
        if not decisions:
            return 'HOLDING'
        net = sum(decisions)
        if net > 0:
            return 'HIKING'
        if net < 0:
            return 'CUTTING'
        return 'HOLDING'

    @staticmethod
    def _inflation_trajectory(cpi: float) -> str:
        if cpi > 3.0:
            return 'RISING'
        if cpi < 1.5:
            return 'FALLING'
        return 'STABLE'

    @staticmethod
    def _gdp_trajectory(gdp: float) -> str:
        if gdp > 2.5:
            return 'ACCELERATING'
        if gdp < 0.5:
            return 'DECELERATING'
        return 'STABLE'

    @staticmethod
    def _classify_phase(
        rate: float, cpi: float, gdp: float,
        real_rate: float, rate_traj: str
    ) -> str:
        if gdp < 0 or (gdp < 0.5 and rate_traj == 'CUTTING'):
            return 'CONTRACTION'
        if rate_traj == 'HIKING' and gdp > 1.5:
            return 'MID_EXP'
        if rate > 3.5 and rate_traj != 'HIKING' and gdp < 1.5:
            return 'LATE_EXP'
        return 'EARLY_EXP'
