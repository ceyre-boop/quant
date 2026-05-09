"""
execution/rr_engine.py
======================
PTJ-Updated Risk/Reward Engine

Paul Tudor Jones principle embedded in code sequence:
  STEP 1: Compute stop (structural + ATR)   ← ALWAYS FIRST
  STEP 2: Verify stop is acceptable risk    ← GATE CHECK
  STEP 3: Compute targets from stop         ← ONLY THEN

Three-target structure (PTJ minimum 5:1 on runner):
  TP1 = 1.5R (40% of position — get something)
  TP2 = 3.0R (35% of position — main move)
  TP3 = 5.0R (25% of position — PTJ runner; 7R for A+ grade)

"If you win 1 in 5 trades at 1:5 R:R, you break even.
 At 70% win rate and 5:1 R:R: EV = 3.2R per trade."
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]

def _load_ptj_config() -> dict:
    p = ROOT / 'config' / 'ptj_philosophy.json'
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

_PTJ = _load_ptj_config()
_RR  = _PTJ.get('rr_targets', {})
_AW  = _PTJ.get('assume_wrong', {})


# ── PTJ Bracket (replaces single tp) ─────────────────────────────────────── #

@dataclass
class PTJBracket:
    """
    Three-target bracket from a single stop distance.
    Encodes PTJ principle: stop is the anchor, targets are derived.
    """
    entry_price:  float
    stop_price:   float
    tp1_price:    float    # 1.5R — partial exit (40%)
    tp2_price:    float    # 3.0R — main move   (35%)
    tp3_price:    float    # 5.0R — PTJ runner   (25%)
    unit_risk:    float    # |entry - stop| in price
    rr_at_tp1:    float    # always 1.5
    rr_at_tp3:    float    # 5.0 standard; 7.0 for A+
    grade:        str = 'B'
    direction:    int = 1  # 1=long, -1=short

    @property
    def tp1_size_pct(self) -> float: return _RR.get('tp1_size_pct', 0.40)
    @property
    def tp2_size_pct(self) -> float: return _RR.get('tp2_size_pct', 0.35)
    @property
    def tp3_size_pct(self) -> float: return _RR.get('tp3_size_pct', 0.25)

    def as_dict(self) -> dict:
        return {
            'entry': self.entry_price,
            'stop': self.stop_price,
            'tp1': self.tp1_price,
            'tp2': self.tp2_price,
            'tp3': self.tp3_price,
            'unit_risk': self.unit_risk,
            'rr_tp1': self.rr_at_tp1,
            'rr_tp3': self.rr_at_tp3,
            'grade': self.grade,
            'tp1_pct': self.tp1_size_pct,
            'tp2_pct': self.tp2_size_pct,
            'tp3_pct': self.tp3_size_pct,
        }


class RREngine:
    """
    PTJ-compliant Risk/Reward Engine.

    Code sequence enforces defence-first:
      1. calculate_stop() must be called before calculate_brackets()
      2. calculate_brackets() raises if stop is zero or invalid
      3. Only after stop is validated are targets computed

    PTJ: "Define where to exit THEN consider profit."
    """

    def __init__(self, atr_multiplier: float = 1.5):
        self.atr_multiplier = atr_multiplier
        self._tp1_r  = _RR.get('tp1_r',       1.5)
        self._tp2_r  = _RR.get('tp2_r',       3.0)
        self._tp3_r  = _RR.get('tp3_r',       5.0)
        self._tp3_r_aplus = _RR.get('tp3_aplus_r', 7.0)
        self._min_rr = _RR.get('minimum_rr_to_enter', 2.0)
        self._trail_r = _RR.get('trailing_activates_at_r', 1.5)
        self._trail_lock_r = _RR.get('trailing_stop_lock_r', 0.5)

    # ── Step 1: Stop FIRST ────────────────────────────────────────────────── #

    def calculate_stop(
        self,
        entry_price: float,
        current_atr: float,
        direction: int,
        structural_level: Optional[float] = None,
    ) -> float:
        """
        STEP 1 — compute stop before any target.
        PTJ: "Define where to exit first."

        Uses ATR as primary stop distance.
        If structural_level provided and tighter than ATR stop → use structure.
        Structural_level must be on the correct side of price.
        """
        atr_stop_dist = current_atr * self.atr_multiplier
        atr_stop = entry_price - direction * atr_stop_dist

        if structural_level is not None:
            # Verify structural level is on the correct side
            struct_dist = abs(entry_price - structural_level)
            struct_valid = (
                (direction == 1 and structural_level < entry_price) or
                (direction == -1 and structural_level > entry_price)
            )
            if struct_valid and struct_dist < atr_stop_dist:
                logger.debug(f"PTJ stop: using structural level {structural_level:.4f} "
                             f"(tighter than ATR stop {atr_stop:.4f})")
                return float(structural_level)

        return float(atr_stop)

    # ── Step 2 + 3: Gate check then targets ───────────────────────────────── #

    def calculate_brackets(
        self,
        entry_price: float,
        stop_price: float,
        direction: int,
        grade: str = 'B',
    ) -> PTJBracket:
        """
        STEP 2+3 — validate stop then compute PTJ three-target bracket.

        Must be called AFTER calculate_stop(). The stop_price argument
        enforces this dependency at the API level.

        PTJ targets:
          TP1 = 1.5R (40% of position — partial, get something)
          TP2 = 3.0R (35% of position — main expected move)
          TP3 = 5.0R (25% — runner; 7R for A+ grade)
        """
        unit_risk = abs(entry_price - stop_price)
        if unit_risk == 0:
            raise ValueError(
                "PTJ RULE VIOLATION: stop_price == entry_price. "
                "No valid stop means no trade. Call calculate_stop() first."
            )

        # Select TP3 based on grade
        tp3_r = self._tp3_r_aplus if grade == 'A+' else self._tp3_r

        # Compute targets (stop is the anchor — all targets derive from it)
        tp1 = entry_price + direction * unit_risk * self._tp1_r
        tp2 = entry_price + direction * unit_risk * self._tp2_r
        tp3 = entry_price + direction * unit_risk * tp3_r

        bracket = PTJBracket(
            entry_price=entry_price,
            stop_price=stop_price,
            tp1_price=tp1,
            tp2_price=tp2,
            tp3_price=tp3,
            unit_risk=unit_risk,
            rr_at_tp1=self._tp1_r,
            rr_at_tp3=tp3_r,
            grade=grade,
            direction=direction,
        )

        logger.info(
            f"PTJ bracket | grade={grade} | "
            f"entry={entry_price:.4f} stop={stop_price:.4f} "
            f"TP1={tp1:.4f}(1.5R) TP2={tp2:.4f}(3R) TP3={tp3:.4f}({tp3_r}R)"
        )
        return bracket

    def update_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        current_sl: float,
        direction: int,
    ) -> float:
        """
        PTJ trailing stop: activates at 1.5R, locks to entry+0.5R.
        Never moves stop further from entry (one-direction tightening only).
        """
        unit_risk = abs(entry_price - current_sl)
        if unit_risk == 0:
            return current_sl

        current_profit = (current_price - entry_price) * direction

        if current_profit >= unit_risk * self._trail_r:
            locked_stop = entry_price + direction * unit_risk * self._trail_lock_r
            if direction == 1:
                return max(current_sl, locked_stop)
            else:
                return min(current_sl, locked_stop)

        return current_sl

    # ── Legacy compatibility ───────────────────────────────────────────────── #

    def calculate_brackets_legacy(
        self,
        entry_price: float,
        current_atr: float,
        direction: int,
        grade: str = 'B',
    ) -> dict:
        """
        Single-call convenience (stop computed internally then targets).
        Returns legacy dict with sl, tp1, tp2, tp3 for backward compat.
        """
        stop = self.calculate_stop(entry_price, current_atr, direction)
        bracket = self.calculate_brackets(entry_price, stop, direction, grade)
        return {
            'sl':  bracket.stop_price,
            'tp':  bracket.tp1_price,    # legacy tp → tp1
            'tp1': bracket.tp1_price,
            'tp2': bracket.tp2_price,
            'tp3': bracket.tp3_price,
            'unit_risk': bracket.unit_risk,
            'rr_tp3': bracket.rr_at_tp3,
        }


# ── Phase tracker ─────────────────────────────────────────────────────────── #

class TradePhaseTracker:
    """
    PTJ Principle 4: "Every position starts wrong until proven otherwise."

    Phase 1 (0→0.5R): UNPROVEN — maximum vigilance, exit on any structural break
    Phase 2 (0.5R→1R): PROVING  — move stop toward entry
    Phase 3 (2R+):     VALIDATED — market proved thesis; let runner run
    """

    def __init__(self, entry_price: float, stop_price: float, direction: int):
        self.entry = entry_price
        self.stop = stop_price
        self.direction = direction
        self.unit_risk = abs(entry_price - stop_price)
        self.phase = 'PHASE1_UNPROVEN'
        self.peak_r = 0.0
        self._p1 = _AW.get('phase1_threshold_r', 0.5)
        self._p2 = _AW.get('phase2_threshold_r', 1.0)
        self._p3 = _AW.get('phase3_threshold_r', 2.0)

    def current_r(self, price: float) -> float:
        if self.unit_risk == 0:
            return 0.0
        return (price - self.entry) * self.direction / self.unit_risk

    def update(self, price: float) -> str:
        r = self.current_r(price)
        self.peak_r = max(self.peak_r, r)
        if r >= self._p3:
            self.phase = 'PHASE3_VALIDATED'
        elif r >= self._p1:
            self.phase = 'PHASE2_PROVING'
        else:
            self.phase = 'PHASE1_UNPROVEN'
        return self.phase

    def should_exit_early(self, price: float) -> bool:
        """Phase 1: if price moves 25% of risk back against position → exit."""
        if self.phase != 'PHASE1_UNPROVEN':
            return False
        r = self.current_r(price)
        return r < -0.25   # 25% of risk unit moving against us in unproven phase


if __name__ == '__main__':
    engine = RREngine()
    entry, atr = 100.0, 2.0

    # PTJ demo: stop first, THEN targets
    stop = engine.calculate_stop(entry, atr, direction=1)
    bracket = engine.calculate_brackets(entry, stop, direction=1, grade='A+')
    print(f"Entry: {entry} | Stop: {stop:.2f} | unit_risk: {bracket.unit_risk:.2f}")
    print(f"TP1 (1.5R): {bracket.tp1_price:.2f}  ← 40% of position")
    print(f"TP2 (3.0R): {bracket.tp2_price:.2f}  ← 35% of position")
    print(f"TP3 (7.0R): {bracket.tp3_price:.2f}  ← 25% PTJ runner (A+ grade)")
    print(f"EV at 70% win, 5:1 R:R: {0.70*5 - 0.30*1:.1f}R per trade")
