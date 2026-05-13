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
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from ict._atr_utils import compute_atr
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

# Stop buffer expressed as a fraction of ATR added beyond the structural anchor.
_STRUCTURE_STOP_BUFFER_ATR = 0.2

# Minimum stop distance as a fraction of ATR (prevents stops so tight they
# are immediately hit by spread/noise even when the structure anchor is close).
_MIN_STOP_ATR_FRACTION = 0.5

# Volatility-adaptive threshold offsets
# ATR / price > HIGH_VOL_THRESHOLD → lower bar (easier to trade)
# ATR / price < LOW_VOL_THRESHOLD  → raise bar (avoid dead markets)
_HIGH_VOL_THRESHOLD = 0.008   # 0.8% ATR/price
_LOW_VOL_THRESHOLD  = 0.002   # 0.2% ATR/price
_ATR_SPIKE_MULT     = 3.0     # veto if ATR spikes 3× recent average


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
        # Grade thresholds (configurable so they can be tuned without code changes)
        self._grade_a_plus_threshold: float = scoring_cfg.get("grade_a_plus_threshold", 8.5)
        self._grade_b_threshold: float = scoring_cfg.get("grade_b_threshold", 5.0)

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
        Run the full ICT pipeline (6 stages + risk engine gate).

        Stages 1–5 collect component scores.
        Stages 1, 2, and 5 are HARD GATES (Phase 3): failing any one returns an
        immediate ICTVeto regardless of the other scores, because these components
        are not independent evidence — they describe the same institutional move.
        Stage 6 is the risk engine gate that checks account constraints.

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
            atr = compute_atr(df)

        price = float(df["Close"].iloc[-1])

        # ── ATR spike veto (robustness gate) ─────────────────────────────
        if len(df) >= 20:
            recent_atr = compute_atr(df.iloc[-20:])
            if recent_atr > 0 and atr > recent_atr * _ATR_SPIKE_MULT:
                return ICTVeto(
                    symbol=symbol, direction=direction, timestamp=timestamp,
                    score=0.0, grade=ICTGrade.VETOED,
                    reason=f"ATR_SPIKE: current ATR {atr:.5f} > {_ATR_SPIKE_MULT}× recent avg {recent_atr:.5f}",
                )

        # ── Volatility-adaptive threshold ─────────────────────────────────
        vol_ratio = atr / price if price > 0 else 0.0
        if vol_ratio > _HIGH_VOL_THRESHOLD:
            effective_threshold = self._min_score - 0.5
            vol_regime = "HIGH_VOL"
        elif vol_ratio < _LOW_VOL_THRESHOLD:
            effective_threshold = self._min_score + 0.5
            vol_regime = "LOW_VOL"
        else:
            effective_threshold = self._min_score
            vol_regime = "NORMAL_VOL"

        scores: Dict[str, float] = {}
        confirmations: List[str] = [f"Vol regime: {vol_regime} (ATR/price={vol_ratio:.4f})"]
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

        # ── HARD GATE: kill zone required ────────────────────────────────
        if scores["kill_zone"] == 0.0:
            return ICTVeto(
                symbol=symbol, direction=direction, timestamp=timestamp,
                score=scores["kill_zone"], grade=ICTGrade.VETOED,
                reason="GATE_KILL_ZONE: must be in a high-probability kill zone — off-hours = no ICT trade",
                component_scores=scores,
                confirmations=confirmations,
                missing=missing,
            )

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
                + (" [displacement]" if recent_sweep.displacement_confirmed else "")
            )
        else:
            scores["sweep"] = 0.0
            missing.append(f"Liquidity sweep ({expected_sweep_dir})")

        # ── HARD GATE: sweep required ────────────────────────────────────
        if scores["sweep"] == 0.0:
            return ICTVeto(
                symbol=symbol, direction=direction, timestamp=timestamp,
                score=sum(scores.values()), grade=ICTGrade.VETOED,
                reason="GATE_SWEEP: liquidity sweep required — no ICT trade without institutional fuel",
                component_scores=scores,
                confirmations=confirmations,
                missing=missing,
            )

        # ── Stage 3: FVG tap + OB alignment ──────────────────────────────
        bull_fvg, bear_fvg, bull_ob, bear_ob = self.fvg_det.nearest_actionable(df)
        tap_fvg = bull_fvg if direction == "LONG" else bear_fvg
        tap_ob = bull_ob if direction == "LONG" else bear_ob

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

        # ── HARD GATE: PD array alignment required ───────────────────────
        if scores["pd_alignment"] == 0.0:
            return ICTVeto(
                symbol=symbol, direction=direction, timestamp=timestamp,
                score=sum(scores.values()), grade=ICTGrade.VETOED,
                reason="GATE_PD_ARRAY: price must be in the correct PD array zone — longs from discount, shorts from premium",
                component_scores=scores,
                confirmations=confirmations,
                missing=missing,
            )

        total_score = sum(scores.values())
        grade = self._grade(total_score, threshold=effective_threshold)

        # ── Stage 6: Risk engine gate ─────────────────────────────────────
        if grade in (ICTGrade.A_PLUS, ICTGrade.A):
            # Phase 1: use ICT-native structure stop instead of generic ATR stop
            stop = self.compute_structure_stop(
                direction=direction,
                sweep=recent_sweep,
                nearest_ob=tap_ob,
                nearest_fvg=tap_fvg,
                df=df,
                atr=atr,
            )

            # Phase 1: compute liquidity-based TP levels
            liq_targets = self.compute_liquidity_targets(
                direction=direction,
                df=df,
                price=price,
                atr=atr,
            )
            tp1_override = liq_targets[0] if len(liq_targets) >= 1 else None
            tp2_override = liq_targets[1] if len(liq_targets) >= 2 else None

            sizing_result = self.risk_engine.size(
                direction=direction,
                entry=price,
                stop_loss=stop,
                atr=atr,
                params=account,
                tp1_override=tp1_override,
                tp2_override=tp2_override,
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

    # ── ICT-native stop computation (Phase 1) ─────────────────────────── #

    @staticmethod
    def compute_structure_stop(
        direction: str,
        sweep: Optional[SweepResult],
        nearest_ob: Optional[OrderBlockResult],
        nearest_fvg: Optional[FVGResult],
        df: pd.DataFrame,
        atr: float,
    ) -> float:
        """
        Compute an ICT-native stop-loss using structure invalidation anchors.

        Priority (highest to lowest):
          1. Sweep anchor  — below the sweep candle's wick low (LONG) / above wick high (SHORT).
             This is the exact level where the setup is proved wrong: if price returns
             to the sweep extreme, the smart-money absorption failed.
          2. Order Block   — below OB.low (LONG) / above OB.high (SHORT).
          3. FVG anchor    — below FVG.bottom (LONG) / above FVG.top (SHORT).
          4. Swing fallback — below 20-bar swing low (LONG) / above 20-bar swing high (SHORT).

        A small ATR-derived buffer is always added to avoid stop-hunting on the spread.
        If the resulting stop distance is smaller than ``_MIN_STOP_ATR_FRACTION × atr``,
        the stop is pushed out to that minimum to prevent noise-hits.
        """
        buffer = _STRUCTURE_STOP_BUFFER_ATR * atr

        if direction == "LONG":
            if sweep is not None:
                raw_stop = sweep.wick_low - buffer
            elif nearest_ob is not None:
                raw_stop = nearest_ob.low - buffer
            elif nearest_fvg is not None:
                raw_stop = nearest_fvg.bottom - buffer
            else:
                raw_stop = float(df["Low"].tail(20).min()) - buffer
        else:  # SHORT
            if sweep is not None:
                raw_stop = sweep.wick_high + buffer
            elif nearest_ob is not None:
                raw_stop = nearest_ob.high + buffer
            elif nearest_fvg is not None:
                raw_stop = nearest_fvg.top + buffer
            else:
                raw_stop = float(df["High"].tail(20).max()) + buffer

        # Sanity: ensure the stop distance meets the minimum ATR fraction
        price = float(df["Close"].iloc[-1])
        min_distance = _MIN_STOP_ATR_FRACTION * atr
        actual_distance = abs(price - raw_stop)
        if actual_distance < min_distance:
            if direction == "LONG":
                raw_stop = price - min_distance
            else:
                raw_stop = price + min_distance

        return raw_stop

    @staticmethod
    def compute_liquidity_targets(
        direction: str,
        df: pd.DataFrame,
        price: float,
        atr: float,
        n_targets: int = 2,
    ) -> List[float]:
        """
        Compute ICT-style liquidity-based take-profit targets.

        Scans for:
          • Prior swing highs (LONG) / swing lows (SHORT) — BSL / SSL pools.
          • Session high/low (last 4 hours of 5-minute bars ≈ 48 bars).
          • Clusters equal highs / equal lows (levels within 0.5 × ATR)
            and uses their centroid as a single target.

        Returns up to ``n_targets`` levels sorted nearest-first.
        Falls back to an empty list when no valid targets can be identified;
        the risk engine will then use R-multiple TPs as a fallback.
        """
        highs = df["High"].values
        lows = df["Low"].values
        n = len(df)
        p = 5  # swing-detection period

        raw_targets: List[float] = []

        if direction == "LONG":
            # Collect swing highs above current price
            for i in range(p, n - p):
                if highs[i] == max(highs[max(0, i - p): i + p + 1]) and highs[i] > price:
                    raw_targets.append(float(highs[i]))
            # Session high (last 48 bars ≈ 4 hours at 5m)
            session_high = float(df["High"].tail(48).max())
            if session_high > price:
                raw_targets.append(session_high)
        else:
            # Collect swing lows below current price
            for i in range(p, n - p):
                if lows[i] == min(lows[max(0, i - p): i + p + 1]) and lows[i] < price:
                    raw_targets.append(float(lows[i]))
            # Session low
            session_low = float(df["Low"].tail(48).min())
            if session_low < price:
                raw_targets.append(session_low)

        if not raw_targets:
            return []

        # Cluster equal levels within 0.5 × ATR (equal highs / equal lows)
        cluster_threshold = 0.5 * atr
        sorted_levels = sorted(raw_targets)
        clusters: List[List[float]] = [[sorted_levels[0]]]
        for level in sorted_levels[1:]:
            if level - clusters[-1][-1] <= cluster_threshold:
                clusters[-1].append(level)
            else:
                clusters.append([level])
        clustered = [sum(c) / len(c) for c in clusters]

        # Sort nearest-first
        if direction == "LONG":
            clustered.sort()
        else:
            clustered.sort(reverse=True)

        return clustered[:n_targets]

    # ── Market-structure helper ────────────────────────────────────────── #

    @staticmethod
    def _market_structure_score(df: pd.DataFrame, direction: str) -> tuple[float, str]:
        """
        Vectorized swing-based BOS / CHOCH check.

        Uses pandas rolling max/min (O(n)) rather than a Python loop (O(n²)).
        Returns (score_fraction 0–1, label string).
        """
        if len(df) < 15:
            return 0.0, "Not enough bars"

        p = 5  # swing period
        highs = df["High"]
        lows = df["Low"]

        # A bar is a swing high if its High equals the rolling max over [i-p, i+p]
        roll_max = highs.rolling(2 * p + 1, center=True).max()
        roll_min = lows.rolling(2 * p + 1, center=True).min()

        sh_vals = highs[highs == roll_max].dropna().values
        sl_vals = lows[lows == roll_min].dropna().values

        if len(sh_vals) < 2 or len(sl_vals) < 2:
            return 0.0, "Insufficient swing data"

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

    def _grade(self, score: float, threshold: Optional[float] = None) -> ICTGrade:
        t = threshold if threshold is not None else self._min_score
        if score >= self._grade_a_plus_threshold:
            return ICTGrade.A_PLUS
        if score >= t:
            return ICTGrade.A
        if score >= self._grade_b_threshold:
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
