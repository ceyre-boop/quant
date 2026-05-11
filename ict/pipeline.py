"""
ict/pipeline.py
===============
ICT Micro-Edge entry pipeline — enforcing the causal sequence.

The entry sequence is NOT a checklist. It is a causal chain:

  SWEEP → DISPLACEMENT → FVG (created by displacement)
        → RETRACEMENT TO FVG → ENTRY AT FVG MIDPOINT

Every stage gates the next. Skipping any stage produces entries
that fight the market rather than joining institutional order flow.

Stages:
  1. Session classifier  — is it a Kill Zone?
  2. Liquidity sweep     — CLOSE-confirmed reversal at a swing level
  2.5 Displacement       — strong impulse body after the sweep (SMT)
  3. Post-sweep FVG      — imbalance CREATED by the displacement
  4. Market structure    — BOS / CHoCH alignment on the swing
  5. PD array            — price in the correct premium / discount zone
  6. Risk engine gate    — sizing, leverage, kill-switch

ISOLATION RULE
--------------
No imports from: sovereign/, layer1/, layer2/, layer3/
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
import yaml

from ict._atr_utils import compute_atr
from ict.session_classifier import SessionClassifier, KillZoneStatus
from ict.sweep_detector import SweepDetector, SweepResult
from ict.fvg_detector import FVGDetector, FVGResult, OrderBlockResult
from ict.micro_risk import MicroRiskEngine, MicroRiskParams, PositionSizing, RiskVeto

logger = logging.getLogger(__name__)

# ── Scoring weights (configurable via ict_params.yml) ─────────────────────── #
_DEFAULT_MIN_SCORE = 7.5          # raised from 6.5 — A-grade now requires more confluence
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "kill_zone":        2.0,
    "sweep":            2.5,      # only confirmed sweeps score here
    "displacement":     2.0,      # new stage — strong body after sweep
    "fvg_tap":          2.0,      # post-sweep FVG retest only
    "market_structure": 1.5,
    "pd_alignment":     1.0,
}

# Volatility-adaptive threshold adjustments
_HIGH_VOL_ATR_RATIO = 0.008   # ATR/price > 0.8% → lower bar by 0.5
_LOW_VOL_ATR_RATIO  = 0.002   # ATR/price < 0.2% → raise bar by 0.5
_ATR_SPIKE_MULT     = 3.0     # instant veto if ATR > 3× recent avg

# Displacement: minimum candle body size relative to ATR
_DISPLACEMENT_BODY_ATR = 0.55  # body must be ≥ 55% of ATR to count
_DISPLACEMENT_LOOKBACK = 5     # bars after sweep to find displacement


# ── Enums & dataclasses ────────────────────────────────────────────────────── #

class ICTGrade(str, Enum):
    A_PLUS = "A+"
    A      = "A"
    B      = "B"
    C      = "C"
    VETOED = "VETOED"


@dataclass
class ICTSignal:
    symbol:         str
    direction:      str
    timestamp:      datetime
    score:          float
    grade:          ICTGrade
    sizing:         PositionSizing
    session_status: KillZoneStatus
    sweep:          Optional[SweepResult]
    nearest_fvg:    Optional[FVGResult]
    nearest_ob:     Optional[OrderBlockResult]
    entry_level:    Optional[float] = None   # FVG midpoint limit price
    component_scores: Dict[str, float] = field(default_factory=dict)
    confirmations:  List[str] = field(default_factory=list)
    missing:        List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.grade in (ICTGrade.A_PLUS, ICTGrade.A)


@dataclass
class ICTVeto:
    symbol:           str
    direction:        str
    timestamp:        datetime
    score:            float
    grade:            ICTGrade
    reason:           str
    component_scores: Dict[str, float] = field(default_factory=dict)
    confirmations:    List[str] = field(default_factory=list)
    missing:          List[str] = field(default_factory=list)


# ── Pipeline ───────────────────────────────────────────────────────────────── #

class ICTPipeline:
    """
    Full ICT micro-edge pipeline enforcing sweep → displacement → FVG sequence.

    Usage::

        pipe = ICTPipeline()
        result = pipe.evaluate(
            symbol="GBPUSD", direction="LONG",
            df=df_1h, timestamp=ts,
            account=MicroRiskParams(account_size=10_000),
        )
        if isinstance(result, ICTSignal) and result.passed:
            # Enter limit order at result.entry_level
            pass
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        cfg          = self._load_config(config_path)
        scoring_cfg  = cfg.get("scoring", {})
        self._min_score              = scoring_cfg.get("min_score_to_trade", _DEFAULT_MIN_SCORE)
        self._weights                = scoring_cfg.get("weights", _DEFAULT_WEIGHTS)
        self._grade_a_plus_threshold = scoring_cfg.get("grade_a_plus_threshold", 9.5)
        self._grade_b_threshold      = scoring_cfg.get("grade_b_threshold", 5.0)

        self.session_clf = SessionClassifier(config_path)
        self.sweep_det   = SweepDetector(config_path)
        self.fvg_det     = FVGDetector(config_path)
        self.risk_engine = MicroRiskEngine(config_path)

    # ── Public API ─────────────────────────────────────────────────────────── #

    def evaluate(
        self,
        symbol:    str,
        direction: str,
        df:        pd.DataFrame,
        timestamp: datetime,
        account:   MicroRiskParams,
        atr:       Optional[float] = None,
    ) -> "ICTSignal | ICTVeto":

        if direction not in ("LONG", "SHORT"):
            return ICTVeto(symbol=symbol, direction=direction, timestamp=timestamp,
                           score=0.0, grade=ICTGrade.C,
                           reason=f"Invalid direction '{direction}'")

        df    = self._normalise(df)
        price = float(df["Close"].iloc[-1])

        if atr is None:
            atr = compute_atr(df)

        # ── Robustness: ATR spike veto ────────────────────────────────────
        if len(df) >= 20:
            atr_recent = compute_atr(df.iloc[-20:])
            if atr_recent > 0 and atr > atr_recent * _ATR_SPIKE_MULT:
                return ICTVeto(symbol=symbol, direction=direction, timestamp=timestamp,
                               score=0.0, grade=ICTGrade.VETOED,
                               reason=f"ATR spike: {atr:.5f} > {_ATR_SPIKE_MULT}× avg {atr_recent:.5f}")

        # ── Volatility-adaptive threshold ─────────────────────────────────
        vol_ratio = atr / price if price > 0 else 0.0
        if vol_ratio > _HIGH_VOL_ATR_RATIO:
            threshold = self._min_score - 0.5
            vol_tag   = "HIGH_VOL"
        elif vol_ratio < _LOW_VOL_ATR_RATIO:
            threshold = self._min_score + 0.5
            vol_tag   = "LOW_VOL"
        else:
            threshold = self._min_score
            vol_tag   = "NORMAL"

        scores:        Dict[str, float] = {}
        confirmations: List[str]        = [f"Vol: {vol_tag} (ATR/price={vol_ratio:.4f})"]
        missing:       List[str]        = []

        # ══════════════════════════════════════════════════════════════════
        # STAGE 1 — Kill Zone
        # ══════════════════════════════════════════════════════════════════
        session = self.session_clf.classify(timestamp)
        if session.should_trade:
            scores["kill_zone"] = self._weights.get("kill_zone", 2.0)
            confirmations.append(f"✓ Kill Zone: {session.kill_zone_name}")
        else:
            scores["kill_zone"] = 0.0
            reason = ("NY Lunch" if session.in_ny_lunch
                      else f"{session.kill_zone_name or 'Off-Hours'} (not HP)")
            missing.append(f"✗ Kill Zone ({reason})")

        # ══════════════════════════════════════════════════════════════════
        # STAGE 2 — Liquidity sweep with CLOSE confirmation
        #
        # A wick that does not close back inside the range is a breakout,
        # not a sweep.  Entering on unconfirmed sweeps is the #1 cause of
        # stop-outs in ICT-style trading.
        # ══════════════════════════════════════════════════════════════════
        expected_dir = "BULLISH_SWEEP" if direction == "LONG" else "BEARISH_SWEEP"
        all_sweeps   = self.sweep_det.detect(df)

        # Hard gate: reversal_confirmed must be True
        confirmed_sweeps = [
            s for s in all_sweeps
            if s.direction == expected_dir and s.reversal_confirmed
        ]
        recent_sweep = confirmed_sweeps[0] if confirmed_sweeps else None

        if recent_sweep is not None:
            wick_quality = min(recent_sweep.wick_atr_ratio / 2.0, 1.0)
            scores["sweep"] = self._weights.get("sweep", 2.5) * (0.7 + 0.3 * wick_quality)
            confirmations.append(
                f"✓ Sweep: {recent_sweep.direction} @ {recent_sweep.swept_level:.5f} "
                f"(wick {recent_sweep.wick_atr_ratio:.1f}×ATR, close confirmed)"
            )
        else:
            scores["sweep"] = 0.0
            unconfirmed = [s for s in all_sweeps if s.direction == expected_dir]
            if unconfirmed:
                missing.append(
                    f"✗ Sweep wick detected but close NOT reversed "
                    f"(level {unconfirmed[0].swept_level:.5f}) — likely breakout, not sweep"
                )
            else:
                missing.append(f"✗ No {expected_dir} detected in window")

        # ══════════════════════════════════════════════════════════════════
        # STAGE 2.5 — Displacement
        #
        # After a genuine sweep, institutional flow creates a strong
        # impulse candle in the reversal direction.  This is the
        # Market Structure Shift (MSS) — the moment smart money commits.
        # Without it, the sweep may be a fakeout before continuation.
        #
        # Requirement: at least one candle AFTER the sweep whose BODY
        # (not wick) is ≥ 55% of ATR in the reversal direction.
        # ══════════════════════════════════════════════════════════════════
        disp_score, disp_label = self._displacement_score(df, recent_sweep, direction, atr)
        scores["displacement"] = disp_score * self._weights.get("displacement", 2.0)
        if disp_score > 0:
            confirmations.append(f"✓ Displacement: {disp_label}")
        else:
            missing.append(f"✗ No displacement after sweep — {disp_label}")

        # ══════════════════════════════════════════════════════════════════
        # STAGE 3 — Post-sweep FVG + retracement
        #
        # The FVG we want was CREATED BY the displacement candle.
        # Any FVG that predates the sweep is from a different context.
        #
        # Entry model: LIMIT ORDER at FVG midpoint, not market order.
        # Price retracing to the FVG midpoint = price returning to fill
        # the institutional orders left behind in the displacement.
        # ══════════════════════════════════════════════════════════════════
        bull_fvg, bear_fvg, bull_ob, bear_ob = self.fvg_det.nearest_actionable(df)
        tap_fvg = bull_fvg if direction == "LONG" else bear_fvg
        tap_ob  = bull_ob  if direction == "LONG" else bear_ob

        fvg_score   = 0.0
        entry_level: Optional[float] = None

        if tap_fvg is not None and recent_sweep is not None:
            if tap_fvg.formed_at >= recent_sweep.formed_at:
                # Post-sweep FVG — check if price is retesting it
                if tap_fvg.price_tapping(price, proximity_fraction=0.7):
                    fvg_score   = self._weights.get("fvg_tap", 2.0)
                    entry_level = tap_fvg.midpoint
                    confirmations.append(
                        f"✓ FVG retest: {tap_fvg.kind} "
                        f"[{tap_fvg.bottom:.5f}–{tap_fvg.top:.5f}] "
                        f"entry @ {entry_level:.5f}"
                    )
                else:
                    # Post-sweep FVG exists but price hasn't pulled back yet
                    fvg_score   = self._weights.get("fvg_tap", 2.0) * 0.25
                    entry_level = tap_fvg.midpoint
                    missing.append(
                        f"✗ FVG not yet reached (current {price:.5f}, "
                        f"FVG @ {tap_fvg.midpoint:.5f}) — set limit order"
                    )
            else:
                # FVG predates sweep — wrong context
                missing.append(
                    f"✗ FVG predates sweep (FVG @ {tap_fvg.formed_at}, "
                    f"sweep @ {recent_sweep.formed_at}) — stale imbalance"
                )
        elif tap_ob is not None and recent_sweep is not None:
            # Order block as fallback — partial credit
            fvg_score   = self._weights.get("fvg_tap", 2.0) * 0.35
            entry_level = tap_ob.midpoint
            missing.append(f"✗ OB only (no post-sweep FVG) @ {tap_ob.midpoint:.5f}")
        elif recent_sweep is None:
            missing.append("✗ No FVG assessed (no confirmed sweep)")
        else:
            missing.append("✗ No FVG or OB in window")

        scores["fvg_tap"] = fvg_score

        # ══════════════════════════════════════════════════════════════════
        # STAGE 4 — Market structure
        # ══════════════════════════════════════════════════════════════════
        ms_score, ms_label = self._market_structure_score(df, direction)
        scores["market_structure"] = ms_score * self._weights.get("market_structure", 1.5)
        if ms_score > 0:
            confirmations.append(f"✓ Market structure: {ms_label}")
        else:
            missing.append(f"✗ Market structure: {ms_label}")

        # ══════════════════════════════════════════════════════════════════
        # STAGE 5 — Premium / Discount alignment
        # ══════════════════════════════════════════════════════════════════
        pd_score, pd_label = self._pd_alignment_score(df, price, direction)
        scores["pd_alignment"] = pd_score * self._weights.get("pd_alignment", 1.0)
        if pd_score > 0:
            confirmations.append(f"✓ PD: {pd_label}")
        else:
            missing.append(f"✗ PD: {pd_label}")

        total_score = sum(scores.values())
        grade       = self._grade(total_score, threshold=threshold)

        # ══════════════════════════════════════════════════════════════════
        # STAGE 6 — Risk engine gate
        # Entry price = FVG midpoint (limit), not current market price.
        # ══════════════════════════════════════════════════════════════════
        if grade in (ICTGrade.A_PLUS, ICTGrade.A):
            ep = entry_level if entry_level is not None else price

            # Use structural stop (swept level) when a sweep is confirmed.
            # This is 3-5× tighter than ATR stop and correctly models ICT
            # invalidation: if price closes back through the swept level,
            # the setup is gone.
            if recent_sweep is not None:
                stop = self.risk_engine.structural_stop(
                    swept_level=recent_sweep.swept_level,
                    direction=direction,
                    atr=atr,
                )
            else:
                stop = self.risk_engine.suggest_stop(ep, direction, atr)

            sz   = self.risk_engine.size(
                direction=direction, entry=ep, stop_loss=stop, atr=atr, params=account
            )
            if isinstance(sz, RiskVeto):
                return ICTVeto(symbol=symbol, direction=direction, timestamp=timestamp,
                               score=total_score, grade=ICTGrade.VETOED,
                               reason=f"Risk gate: {sz.reason} — {sz.detail}",
                               component_scores=scores,
                               confirmations=confirmations, missing=missing)
            return ICTSignal(
                symbol=symbol, direction=direction, timestamp=timestamp,
                score=total_score, grade=grade, sizing=sz,
                session_status=session,
                sweep=recent_sweep, nearest_fvg=tap_fvg, nearest_ob=tap_ob,
                entry_level=ep,
                component_scores=scores,
                confirmations=confirmations, missing=missing,
            )

        return ICTVeto(
            symbol=symbol, direction=direction, timestamp=timestamp,
            score=total_score, grade=grade,
            reason=f"Score {total_score:.1f} < threshold {threshold:.1f} (grade {grade})",
            component_scores=scores, confirmations=confirmations, missing=missing,
        )

    # ── Stage 2.5: Displacement helper ────────────────────────────────────── #

    @staticmethod
    def _displacement_score(
        df:           pd.DataFrame,
        sweep:        Optional[SweepResult],
        direction:    str,
        atr:          float,
    ) -> tuple[float, str]:
        """
        Look for a strong body candle in the reversal direction after the sweep.

        Returns (score 0–1, description).
        Body must be ≥ _DISPLACEMENT_BODY_ATR × ATR.
        Looks at up to _DISPLACEMENT_LOOKBACK bars after the sweep candle.
        """
        if sweep is None:
            return 0.0, "no sweep to displace from"
        if atr <= 0:
            return 0.0, "ATR zero"

        # Find the bar index of the sweep
        try:
            sweep_loc = df.index.searchsorted(sweep.formed_at)
        except Exception:
            return 0.0, "sweep timestamp not in index"

        if sweep_loc >= len(df) - 1:
            return 0.0, "sweep is last bar — no post-sweep data yet"

        best_ratio = 0.0
        best_label = ""
        end = min(sweep_loc + _DISPLACEMENT_LOOKBACK + 1, len(df))

        for i in range(sweep_loc + 1, end):
            bar    = df.iloc[i]
            o, c   = float(bar["Open"]), float(bar["Close"])
            body   = abs(c - o)
            ratio  = body / atr

            if direction == "LONG" and c > o and ratio > best_ratio:
                best_ratio = ratio
                best_label = f"bullish body {ratio:.2f}×ATR @ bar {i - sweep_loc} after sweep"

            elif direction == "SHORT" and c < o and ratio > best_ratio:
                best_ratio = ratio
                best_label = f"bearish body {ratio:.2f}×ATR @ bar {i - sweep_loc} after sweep"

        if best_ratio >= _DISPLACEMENT_BODY_ATR:
            # Scale score: 0.6 at min threshold, 1.0 at 2× threshold
            score = min(0.6 + 0.4 * (best_ratio - _DISPLACEMENT_BODY_ATR) / _DISPLACEMENT_BODY_ATR, 1.0)
            return round(score, 3), best_label

        if best_label:
            return 0.0, f"body {best_ratio:.2f}×ATR < required {_DISPLACEMENT_BODY_ATR}×ATR"
        return 0.0, f"no {direction.lower()} candle found in {_DISPLACEMENT_LOOKBACK} bars after sweep"

    # ── Stage 4: Market structure ──────────────────────────────────────────── #

    @staticmethod
    def _market_structure_score(df: pd.DataFrame, direction: str) -> tuple[float, str]:
        if len(df) < 15:
            return 0.0, "Not enough bars"
        p        = 5
        highs    = df["High"]
        lows     = df["Low"]
        roll_max = highs.rolling(2 * p + 1, center=True).max()
        roll_min = lows.rolling(2 * p + 1, center=True).min()
        sh_vals  = highs[highs == roll_max].dropna().values
        sl_vals  = lows[lows == roll_min].dropna().values
        if len(sh_vals) < 2 or len(sl_vals) < 2:
            return 0.0, "Insufficient swing data"
        hh = sh_vals[-1] > sh_vals[-2]
        hl = sl_vals[-1] > sl_vals[-2]
        lh = sh_vals[-1] < sh_vals[-2]
        ll = sl_vals[-1] < sl_vals[-2]
        if direction == "LONG":
            if hh and hl:   return 1.0, "BOS Bullish (HH+HL)"
            if hl and lh:   return 0.8, "CHoCH Bullish (HL holds)"
            return 0.0,     "Bearish structure — no LONG alignment"
        else:
            if lh and ll:   return 1.0, "BOS Bearish (LH+LL)"
            if lh and hl:   return 0.8, "CHoCH Bearish (LH confirmed)"
            return 0.0,     "Bullish structure — no SHORT alignment"

    # ── Stage 5: PD array ─────────────────────────────────────────────────── #

    @staticmethod
    def _pd_alignment_score(df: pd.DataFrame, price: float, direction: str) -> tuple[float, str]:
        high = float(df["High"].tail(20).max())
        low  = float(df["Low"].tail(20).min())
        if high <= low:
            return 0.0, "Invalid range"
        eq  = (high + low) / 2
        pct = (price - low) / (high - low)
        if direction == "LONG":
            if price < eq:
                depth = (eq - price) / (eq - low)
                return min(0.5 + 0.5 * depth, 1.0), f"Discount {pct:.0%} of range"
            return 0.0, f"Premium {pct:.0%} — wrong side for LONG"
        else:
            if price > eq:
                depth = (price - eq) / (high - eq)
                return min(0.5 + 0.5 * depth, 1.0), f"Premium {pct:.0%} of range"
            return 0.0, f"Discount {pct:.0%} — wrong side for SHORT"

    # ── Grading ────────────────────────────────────────────────────────────── #

    def _grade(self, score: float, threshold: Optional[float] = None) -> ICTGrade:
        t = threshold if threshold is not None else self._min_score
        if score >= self._grade_a_plus_threshold: return ICTGrade.A_PLUS
        if score >= t:                            return ICTGrade.A
        if score >= self._grade_b_threshold:      return ICTGrade.B
        return ICTGrade.C

    # ── Static helpers ─────────────────────────────────────────────────────── #

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        rename = {c: c.capitalize()
                  for c in df.columns if c.lower() in ("open","high","low","close","volume")}
        return df.rename(columns=rename)[["Open","High","Low","Close"]].dropna()

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
