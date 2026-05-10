"""
ict/pipeline.py
===============
Standalone ICT Micro-Edge entry/exit pipeline.

Orchestrates the four detection stages in sequence:
  1. SessionClassifier  → is it a Kill Zone?
  2. SweepDetector      → is there a recent liquidity sweep?
  3. FVGDetector        → is price near an unfilled FVG / OB?
  4. Market structure   → BOS / CHOCH alignment (simple swing check)
  5. PD array           → is price in discount (for longs) or premium (for shorts)?
  6. MicroRiskEngine    → compute sizing or veto

Produces an `ICTSignal` (passed) or an `ICTVeto` (blocked with score breakdown).

ISOLATION RULE
--------------
This module MUST NOT import from:
  sovereign/, layer1/, layer2/, layer3/, orchestrator/

It may import from:
  ict.*            — all other ICT subsystem modules
  contracts/types  — shared pure-data types (Direction only)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
import yaml

from ict.session_classifier import SessionClassifier, KillZoneStatus
from ict.sweep_detector import SweepDetector, SweepResult
from ict.fvg_detector import FVGDetector, FVGResult, OrderBlockResult
from ict.micro_risk import MicroRiskEngine, MicroRiskParams, PositionSizing, RiskVeto

logger = logging.getLogger(__name__)

_DEFAULT_MIN_SCORE = 6.5
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "kill_zone":        2.0,
    "sweep":            2.5,
    "fvg_tap":          2.0,
    "market_structure": 2.0,
    "pd_alignment":     1.5,
}


# ── Enums & data classes ─────────────────────────────────────────────────── #

class ICTGrade(str, Enum):
    A_PLUS = "A+"
    A      = "A"
    B      = "B"    # logged, not executed
    C      = "C"    # rejected
    VETOED = "VETOED"


@dataclass
class ICTSignal:
    """An approved ICT entry signal with full context."""
    symbol: str
    direction: str                          # 'LONG' | 'SHORT'
    timestamp: datetime
    score: float
    grade: ICTGrade
    sizing: PositionSizing

    # Detection artifacts
    session_status: KillZoneStatus
    sweep: Optional[SweepResult]
    nearest_fvg: Optional[FVGResult]
    nearest_ob: Optional[OrderBlockResult]

    # Breakdown
    component_scores: Dict[str, float] = field(default_factory=dict)
    confirmations: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.grade in (ICTGrade.A_PLUS, ICTGrade.A)


@dataclass
class ICTVeto:
    """A rejected signal with the reason (score too low, risk gate, etc.)."""
    symbol: str
    direction: str
    timestamp: datetime
    score: float
    grade: ICTGrade
    reason: str
    component_scores: Dict[str, float] = field(default_factory=dict)
    confirmations: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)


# ── Pipeline ─────────────────────────────────────────────────────────────── #

class ICTPipeline:
    """
    Full ICT micro-edge entry pipeline.

    Usage::

        pipe = ICTPipeline()
        account = MicroRiskParams(account_size=10_000)
        result = pipe.evaluate(
            symbol="GBPUSD",
            direction="LONG",
            df=df_5min,        # intraday OHLCV
            timestamp=datetime.utcnow(),
            account=account,
        )
        if isinstance(result, ICTSignal) and result.passed:
            execute(result.sizing)
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        cfg = self._load_config(config_path)
        scoring_cfg = cfg.get("scoring", {})
        self._min_score: float = scoring_cfg.get("min_score_to_trade", _DEFAULT_MIN_SCORE)
        self._weights: Dict[str, float] = scoring_cfg.get("weights", _DEFAULT_WEIGHTS)

        self.session_clf = SessionClassifier(config_path)
        self.sweep_det = SweepDetector(config_path)
        self.fvg_det = FVGDetector(config_path)
        self.risk_engine = MicroRiskEngine(config_path)

    # ── Public API ─────────────────────────────────────────────────────── #

    def evaluate(
        self,
        symbol: str,
        direction: str,
        df: pd.DataFrame,
        timestamp: datetime,
        account: MicroRiskParams,
        atr: Optional[float] = None,
    ) -> "ICTSignal | ICTVeto":
        """
        Run the full 5-stage ICT pipeline.

        Args:
            symbol: e.g. 'GBPUSD'
            direction: 'LONG' or 'SHORT'
            df: intraday OHLCV (at least 60 bars recommended)
            timestamp: current datetime (UTC or tz-aware)
            account: live account state for risk engine
            atr: override ATR; computed from df if None

        Returns:
            ICTSignal if score ≥ threshold and risk gates pass.
            ICTVeto  otherwise.
        """
        if direction not in ("LONG", "SHORT"):
            return ICTVeto(
                symbol=symbol, direction=direction, timestamp=timestamp,
                score=0.0, grade=ICTGrade.C,
                reason=f"Invalid direction '{direction}' — must be LONG or SHORT.",
            )

        df = self._normalise(df)
        if atr is None:
            atr = self._atr(df)

        price = float(df["Close"].iloc[-1])
        scores: Dict[str, float] = {}
        confirmations: List[str] = []
        missing: List[str] = []

        # ── Stage 1: Kill Zone / session ────────────────────────────────
        session = self.session_clf.classify(timestamp)
        if session.should_trade:
            scores["kill_zone"] = self._weights.get("kill_zone", 2.0)
            confirmations.append(f"Kill Zone: {session.kill_zone_name}")
        else:
            scores["kill_zone"] = 0.0
            reason = "NY Lunch" if session.in_ny_lunch else (
                f"Session={session.kill_zone_name or 'Off-Hours'} (not HP)"
            )
            missing.append(f"Kill Zone ({reason})")

        # ── Stage 2: Liquidity sweep ─────────────────────────────────────
        expected_sweep_dir = "BULLISH_SWEEP" if direction == "LONG" else "BEARISH_SWEEP"
        sweeps = self.sweep_det.detect(df)
        relevant_sweeps = [s for s in sweeps if s.direction == expected_sweep_dir]
        recent_sweep = relevant_sweeps[0] if relevant_sweeps else None

        if recent_sweep is not None:
            scores["sweep"] = self._weights.get("sweep", 2.5)
            confirmations.append(
                f"Sweep: {recent_sweep.direction} @ {recent_sweep.swept_level:.5f}"
                + (" (reversal confirmed)" if recent_sweep.reversal_confirmed else "")
            )
        else:
            scores["sweep"] = 0.0
            missing.append(f"Liquidity sweep ({expected_sweep_dir})")

        # ── Stage 3: FVG tap + OB alignment ──────────────────────────────
        bull_fvg, bear_fvg, bull_ob, bear_ob = self.fvg_det.nearest_actionable(df)
        tap_fvg = bull_fvg if direction == "LONG" else bear_fvg
        tap_ob = bull_ob if direction == "LONG" else bear_ob

        # Score FVG tap
        fvg_score = 0.0
        if tap_fvg is not None and tap_fvg.price_tapping(price, proximity_fraction=0.5):
            fvg_score = self._weights.get("fvg_tap", 2.0)
            confirmations.append(
                f"FVG tap: {tap_fvg.kind} [{tap_fvg.bottom:.5f}–{tap_fvg.top:.5f}]"
            )
        elif tap_ob is not None:
            fvg_score = self._weights.get("fvg_tap", 2.0) * 0.6  # partial credit for OB only
            confirmations.append(
                f"OB (no FVG): {tap_ob.kind} [{tap_ob.low:.5f}–{tap_ob.high:.5f}]"
            )
        else:
            missing.append("FVG / Order Block alignment")
        scores["fvg_tap"] = fvg_score

        # ── Stage 4: Market structure (BOS / CHOCH) ──────────────────────
        ms_score, ms_label = self._market_structure_score(df, direction)
        scores["market_structure"] = ms_score * self._weights.get("market_structure", 2.0)
        if ms_score > 0:
            confirmations.append(f"Market structure: {ms_label}")
        else:
            missing.append(f"Market structure confirmation ({ms_label})")

        # ── Stage 5: Premium / Discount alignment ────────────────────────
        pd_score, pd_label = self._pd_alignment_score(df, price, direction)
        scores["pd_alignment"] = pd_score * self._weights.get("pd_alignment", 1.5)
        if pd_score > 0:
            confirmations.append(f"PD array: {pd_label}")
        else:
            missing.append(f"PD array ({pd_label})")

        total_score = sum(scores.values())
        grade = self._grade(total_score)

        # ── Stage 6: Risk engine gate ─────────────────────────────────────
        if grade in (ICTGrade.A_PLUS, ICTGrade.A):
            stop = self.risk_engine.suggest_stop(price, direction, atr)
            sizing_result = self.risk_engine.size(
                direction=direction,
                entry=price,
                stop_loss=stop,
                atr=atr,
                params=account,
            )
            if isinstance(sizing_result, RiskVeto):
                return ICTVeto(
                    symbol=symbol, direction=direction, timestamp=timestamp,
                    score=total_score, grade=ICTGrade.VETOED,
                    reason=f"Risk gate: {sizing_result.reason} — {sizing_result.detail}",
                    component_scores=scores,
                    confirmations=confirmations,
                    missing=missing,
                )
            return ICTSignal(
                symbol=symbol, direction=direction, timestamp=timestamp,
                score=total_score, grade=grade, sizing=sizing_result,
                session_status=session,
                sweep=recent_sweep,
                nearest_fvg=tap_fvg,
                nearest_ob=tap_ob,
                component_scores=scores,
                confirmations=confirmations,
                missing=missing,
            )

        # Grade B / C → veto
        return ICTVeto(
            symbol=symbol, direction=direction, timestamp=timestamp,
            score=total_score, grade=grade,
            reason=f"Score {total_score:.1f} < threshold {self._min_score:.1f} (grade {grade})",
            component_scores=scores,
            confirmations=confirmations,
            missing=missing,
        )

    # ── Market-structure helper ────────────────────────────────────────── #

    @staticmethod
    def _market_structure_score(df: pd.DataFrame, direction: str) -> tuple[float, str]:
        """
        Simple swing-based BOS / CHOCH check.

        Returns (score_fraction 0–1, label string).
        """
        if len(df) < 15:
            return 0.0, "Not enough bars"

        highs = df["High"].values
        lows = df["Low"].values
        p = 5  # swing period

        sh_idx = [i for i in range(p, len(df) - p) if highs[i] == max(highs[i - p:i + p + 1])]
        sl_idx = [i for i in range(p, len(df) - p) if lows[i] == min(lows[i - p:i + p + 1])]

        if len(sh_idx) < 2 or len(sl_idx) < 2:
            return 0.0, "Insufficient swing data"

        sh_vals = [highs[i] for i in sh_idx[-3:]]
        sl_vals = [lows[i] for i in sl_idx[-3:]]

        higher_highs = sh_vals[-1] > sh_vals[-2]
        higher_lows = sl_vals[-1] > sl_vals[-2]
        lower_highs = sh_vals[-1] < sh_vals[-2]
        lower_lows = sl_vals[-1] < sl_vals[-2]

        if direction == "LONG":
            if higher_highs and higher_lows:
                return 1.0, "BOS Bullish (HH+HL)"
            if higher_lows and lower_highs:
                return 0.8, "CHOCH Bullish (HL holds)"
            return 0.0, "Bearish structure — no LONG alignment"
        else:
            if lower_highs and lower_lows:
                return 1.0, "BOS Bearish (LH+LL)"
            if lower_highs and higher_lows:
                return 0.8, "CHOCH Bearish (LH confirmed)"
            return 0.0, "Bullish structure — no SHORT alignment"

    # ── PD array helper ────────────────────────────────────────────────── #

    @staticmethod
    def _pd_alignment_score(df: pd.DataFrame, price: float, direction: str) -> tuple[float, str]:
        """
        Check whether price is in the correct Premium/Discount zone.

        Returns (score_fraction 0–1, label).
        """
        high = float(df["High"].tail(20).max())
        low = float(df["Low"].tail(20).min())
        if high <= low:
            return 0.0, "Invalid range"

        eq = (high + low) / 2
        pct = (price - low) / (high - low)

        if direction == "LONG":
            if price < eq:
                discount_depth = (eq - price) / (eq - low)
                label = f"Discount {pct:.0%} of range (deeper = better)"
                score = 0.5 + 0.5 * discount_depth  # 0.5 if at eq, 1.0 if at low
                return min(score, 1.0), label
            return 0.0, f"Premium zone {pct:.0%} — wrong side for LONG"
        else:
            if price > eq:
                premium_depth = (price - eq) / (high - eq)
                label = f"Premium {pct:.0%} of range"
                score = 0.5 + 0.5 * premium_depth
                return min(score, 1.0), label
            return 0.0, f"Discount zone {pct:.0%} — wrong side for SHORT"

    # ── Grading ────────────────────────────────────────────────────────── #

    def _grade(self, score: float) -> ICTGrade:
        max_score = sum(self._weights.values())
        pct = score / max_score if max_score > 0 else 0
        if score >= 8.5:
            return ICTGrade.A_PLUS
        if score >= self._min_score:
            return ICTGrade.A
        if score >= 5.0:
            return ICTGrade.B
        return ICTGrade.C

    # ── Static helpers ─────────────────────────────────────────────────── #

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        rename = {c: c.capitalize()
                  for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
        return df.rename(columns=rename)[["Open", "High", "Low", "Close"]].dropna()

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float:
        h, l, c = df["High"], df["Low"], df["Close"]
        tr = pd.concat([
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs(),
        ], axis=1).max(axis=1)
        val = float(tr.rolling(period).mean().iloc[-1])
        return val if val > 0 else 1e-6

    @staticmethod
    def _load_config(config_path: Optional[str]) -> dict:
        path = config_path or _default_config_path()
        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("ICT config not found at %s — using defaults", path)
            return {}


def _default_config_path() -> str:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "config", "ict_params.yml")
