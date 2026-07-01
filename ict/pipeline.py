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
import os
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
_DEFAULT_MIN_SCORE = 7.5
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "kill_zone":        2.0,
    "sweep":            4.0,   # HYP-034 2026-05-23: ms zeroed, freed pts → sweep+1.5
    "displacement":     1.0,
    "fvg_tap":          3.0,   # HYP-034 2026-05-23: freed pts → fvg_tap+1.0
    # HYP-034 confirmed anti-edge: ms>=1.5 → commitment failure 87.5% accuracy.
    # BOS/CHOCH is lagging — by the time structure shifts, displacement is done.
    # Zeroed 2026-05-23. Score still computed and logged for monitoring.
    "market_structure": 0.0,
    # HYP-024 confirmed anti-edge: pd_alignment>0 → 20% WR, pd_alignment=0 → 35% WR.
    # Zeroed 2026-05-19. Score still computed and logged for monitoring.
    "pd_alignment":     0.0,
}

# Volatility-adaptive threshold adjustments
_HIGH_VOL_ATR_RATIO = 0.008
_LOW_VOL_ATR_RATIO  = 0.002
_ATR_SPIKE_MULT     = 3.0

# Displacement quality
_DISPLACEMENT_BODY_ATR = 0.55
_DISPLACEMENT_LOOKBACK = 5

# ── Volume proxy (forex has no tick data — use bar range relative to ATR) ──── #
# Bar range (H-L) / ATR correlates 0.87+ with real institutional participation.
# Wide-range bars = institutional order flow.  Narrow-range bars = retail noise.
_VOL_PROXY_RATIO_MIN  = 1.3   # displacement bar range must be ≥ 1.3×ATR to confirm
_VOL_PROXY_LOOKBACK   = 20    # bars used to compute average range

# ── ADR exhaustion gate ────────────────────────────────────────────────────── #
# ADR = Average Daily Range (20-day).  ICT rule: if the day has already moved
# its full average range, the fuel is gone.  Late setups against exhausted ADR
# are one of the most common traps in retail ICT trading.
_ADR_HARD_VETO  = 0.85   # current range ≥ 85% of ADR → instant veto
_ADR_SOFT_VETO  = 0.70   # current range ≥ 70% of ADR → subtract from score
_ADR_LOOKBACK   = 20     # days for ADR computation


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
    # Retrospective-labeling metadata (no trading effect): the intended entry
    # limit and structural stop the veto prevented, populated only when they
    # were already computed before the gate fired. Early gates (ADR exhaustion,
    # HYP046 displacement) veto before entry exists, so these stay None there.
    entry_level:      Optional[float] = None
    stop:             Optional[float] = None
    # ADR consumed at signal time (0–1). Computed by the ADR-exhaustion gate
    # and carried here so the veto ledger records the structured value instead
    # of having to parse it back out of the reason string.
    adr_pct:          float = 0.0


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
        pipeline_cfg = cfg.get("pipeline", {})
        self._atr_spike_mult = pipeline_cfg.get("atr_spike_veto_multiplier", _ATR_SPIKE_MULT)
        self._adr_hard_veto = pipeline_cfg.get("adr_exhaustion_threshold", _ADR_HARD_VETO)
        self._adr_soft_veto = pipeline_cfg.get("adr_soft_penalty_threshold", _ADR_SOFT_VETO)
        self._displacement_body_atr = pipeline_cfg.get("displacement_atr_multiplier", _DISPLACEMENT_BODY_ATR)
        self._overconfirm_threshold = pipeline_cfg.get("overconfirmation_penalty_threshold", 9.0)
        self._overconfirm_slope = pipeline_cfg.get("overconfirmation_penalty_slope", 0.5)

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
        # ── Sovereign context injected by ict-engine/orchestrator.py ──────
        # Pre-fetched by the orchestrator (the designated cross-layer bridge)
        # and passed in here to keep this module isolated.
        ict_alloc_weight: float = 1.0,
        ict_alloc_veto_reason: str = "",
        bridge_thresholds: Optional[Dict] = None,
        commitment_result: Optional[object] = None,
        ny_am_mode: bool = False,
        weekly_df: Optional[pd.DataFrame] = None,
    ) -> "ICTSignal | ICTVeto":

        if direction not in ("LONG", "SHORT"):
            return ICTVeto(symbol=symbol, direction=direction, timestamp=timestamp,
                           score=0.0, grade=ICTGrade.C,
                           reason=f"Invalid direction '{direction}'")

        # ── Allocation engine gate (Stage 0a) ───────────────────────────
        # ict_alloc_weight pre-fetched by orchestrator from allocation_engine.
        # ict_weight=0.0 → veto (regime hostile to ICT).
        _ict_alloc_weight = ict_alloc_weight
        if _ict_alloc_weight == 0.0:
            return ICTVeto(symbol=symbol, direction=direction, timestamp=timestamp,
                           score=0.0, grade=ICTGrade.VETOED,
                           reason=ict_alloc_veto_reason or "ALLOCATION_ZERO: regime hostile to ICT")

        # ── Cross-system bridge gate (Stage 0b) ──────────────────────────
        # bridge_thresholds pre-fetched by orchestrator from cross_system_bridge.
        # HALT_NEW blocks all new entries when Library convergence is extreme.
        # TIGHTEN adjusts thresholds below (min_score raised to 8.0).
        _bridge_thresholds: Dict = bridge_thresholds or {}
        if not _bridge_thresholds.get("active", True):
            return ICTVeto(symbol=symbol, direction=direction, timestamp=timestamp,
                           score=0.0, grade=ICTGrade.VETOED,
                           reason=f"BRIDGE_HALT_NEW: {_bridge_thresholds.get('reason', '')[:80]}")

        df    = self._normalise(df)
        price = float(df["Close"].iloc[-1])

        if atr is None:
            atr = compute_atr(df)

        # ── Robustness: ATR spike veto ────────────────────────────────────
        if len(df) >= 20:
            atr_recent = compute_atr(df.iloc[-20:])
            if atr_recent > 0 and atr > atr_recent * self._atr_spike_mult:
                return ICTVeto(symbol=symbol, direction=direction, timestamp=timestamp,
                               score=0.0, grade=ICTGrade.VETOED,
                               reason=f"ATR spike: {atr:.5f} > {self._atr_spike_mult}× avg {atr_recent:.5f}")

        # ── ADR exhaustion gate ───────────────────────────────────────────
        # Compute today's range vs 20-day average daily range.
        # If the day has already moved its full average range, there is no
        # room left for price to run to the target — setup is a trap.
        #
        # Session-aware: London/NY overlap and high-impact event windows
        # routinely extend beyond the average daily range. Apply a looser
        # 1.10× threshold during those sessions so genuine extended-range
        # FVGs aren't blocked by a threshold calibrated for quiet periods.
        adr_pct, adr_label = self._adr_exhaustion(df)
        session_pre = self.session_clf.classify(timestamp)
        _is_overlap = getattr(session_pre, "session_name", "") in ("London", "NY_PM")
        adr_hard = self._adr_hard_veto * (1.10 if _is_overlap else 1.0)
        if adr_pct >= adr_hard:
            return ICTVeto(symbol=symbol, direction=direction, timestamp=timestamp,
                           score=0.0, grade=ICTGrade.VETOED,
                           reason=f"ADR exhausted: {adr_pct:.0%} of average daily range consumed",
                           adr_pct=adr_pct)

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

        # Bridge TIGHTEN overrides threshold floor to 8.0
        _bridge_min = _bridge_thresholds.get("min_score", 0.0)
        if _bridge_min > threshold:
            threshold = _bridge_min
            vol_tag   = f"{vol_tag}+BRIDGE_TIGHTEN"

        scores:        Dict[str, float] = {}
        confirmations: List[str]        = [f"Vol: {vol_tag} (ATR/price={vol_ratio:.4f})"]
        missing:       List[str]        = []

        # ADR soft penalty — room still exists but getting tight
        adr_penalty = 0.0
        if adr_pct >= self._adr_soft_veto:
            adr_penalty = 1.0
            missing.append(f"⚠ ADR {adr_pct:.0%} consumed — limited room to target (−1.0 score)")
        else:
            confirmations.append(f"✓ ADR {adr_pct:.0%} consumed — room available")

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
        disp_score, disp_label = self._displacement_score(
            df, recent_sweep, direction, atr, self._displacement_body_atr
        )
        scores["displacement"] = disp_score * self._weights.get("displacement", 2.0)
        if disp_score > 0:
            confirmations.append(f"✓ Displacement: {disp_label}")
        else:
            missing.append(f"✗ No displacement after sweep — {disp_label}")

        # ══════════════════════════════════════════════════════════════════
        # HYP-046 DISPLACEMENT GATE — confirmed 2026-05-26
        # disp<1.5 trades: WR=15%, avgR=-0.167 (19% of London trades)
        # disp>=1.5 trades: WR=35%, avgR=+0.594 (81% of London trades)
        # Net: +0.146R/trade vs keeping all. Hard veto at threshold.
        # Weight=1.0 (_DEFAULT_WEIGHTS) → component max=1.0 → threshold=0.75
        # (same 75th-percentile cut as 1.5 threshold at weight=2.0)
        # ══════════════════════════════════════════════════════════════════
        _disp_component = scores.get("displacement", 0.0)
        if _disp_component < 0.75:
            return ICTVeto(
                symbol=symbol, direction=direction, timestamp=timestamp,
                score=0.0, grade=ICTGrade.VETOED,
                reason=(
                    f"HYP046_DISP_GATE: disp_component={_disp_component:.2f} < 0.75 "
                    f"(low-disp trades: WR=15%, avgR=-0.167; confirmed 2026-05-26)"
                ),
                component_scores=scores, confirmations=confirmations, missing=missing,
            )

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

        raw_score = max(0.0, sum(scores.values()) - adr_penalty)

        # Over-confirmation penalty: A+ setups (score > 9.0) are empirically worse
        # than A setups (7.5–9.0). High score = late entry = move already done.
        # The sweet spot is the slightly uncomfortable setup, not the perfect one.
        # Encode that discovery: cap at 9.5, penalise anything above 9.0.
        if raw_score > self._overconfirm_threshold:
            over_confirm_penalty = (raw_score - self._overconfirm_threshold) * self._overconfirm_slope
            total_score = raw_score - over_confirm_penalty
            missing.append(
                f"⚠ Over-confirmation penalty: score {raw_score:.1f} → {total_score:.1f} "
                f"(A+ setups run late — sweet spot is A grade)"
            )
        else:
            total_score = raw_score

        grade = self._grade(total_score, threshold=threshold)

        # ══════════════════════════════════════════════════════════════════
        # STAGE 5.5 — Reddit sentiment filter (contrarian signal)
        # Reddit retail is directionally wrong often enough to be useful as a
        # FADE signal. When retail is strongly bullish on a pair and we're
        # looking to go LONG, that's a warning, not a confirmation.
        # Logic: if net_score > 3 and direction matches retail → -0.5 penalty
        #        if net_score > 5 and direction matches retail → veto (crowded)
        #        if net_score < -3 and direction opposes retail → +0.3 boost
        # ══════════════════════════════════════════════════════════════════
        total_score = self._apply_reddit_filter(
            symbol=symbol, direction=direction,
            score=total_score, confirmations=confirmations, missing=missing,
        )
        # Re-grade after sentiment adjustment
        grade = self._grade(total_score, threshold=threshold)

        # ══════════════════════════════════════════════════════════════════
        # STAGE 5.6 — Weekly trend alignment (2026-05-30)
        # Weekly EMA20/EMA50 cross: EMA20>EMA50 = bullish, EMA20<EMA50 = bearish.
        # Shorts against a bullish weekly trend and longs against a bearish
        # weekly trend have statistically negative R — block them.
        # Gate skipped (not vetoed) when weekly_df is None or has <20 bars.
        # weekly_df passed by caller — needs 2y of 1W bars for EMA convergence.
        # ══════════════════════════════════════════════════════════════════
        if weekly_df is not None and len(weekly_df) >= 20:
            _wema20 = float(weekly_df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
            _wema50 = float(weekly_df["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
            _weekly_bullish = _wema20 > _wema50
            _weekly_bearish = _wema20 < _wema50
            if direction == "SHORT" and _weekly_bullish:
                return ICTVeto(
                    symbol=symbol, direction=direction, timestamp=timestamp,
                    score=total_score, grade=ICTGrade.VETOED,
                    reason=(
                        f"WEEKLY_TREND_CONFLICT: shorting against weekly uptrend "
                        f"(EMA20={_wema20:.5f} > EMA50={_wema50:.5f})"
                    ),
                    component_scores=scores, confirmations=confirmations, missing=missing,
                    entry_level=entry_level,
                )
            elif direction == "LONG" and _weekly_bearish:
                return ICTVeto(
                    symbol=symbol, direction=direction, timestamp=timestamp,
                    score=total_score, grade=ICTGrade.VETOED,
                    reason=(
                        f"WEEKLY_TREND_CONFLICT: longing against weekly downtrend "
                        f"(EMA20={_wema20:.5f} < EMA50={_wema50:.5f})"
                    ),
                    component_scores=scores, confirmations=confirmations, missing=missing,
                    entry_level=entry_level,
                )
            elif _weekly_bullish or _weekly_bearish:
                _trend_dir = "bullish" if _weekly_bullish else "bearish"
                confirmations.append(
                    f"✓ Weekly trend aligned — {_trend_dir} "
                    f"(EMA20={_wema20:.5f} EMA50={_wema50:.5f})"
                )

        # ══════════════════════════════════════════════════════════════════
        # HYP-047 SCORE CEILING — confirmed 2026-05-26
        # Score [7-8): 52% WR +1.210R | [8-8.5): 36% WR +0.790R
        # Score [8.5-9): WR decay begins | Score 9+: negative R
        # Hard veto at 8.5 — above this threshold WR decays monotonically.
        # ══════════════════════════════════════════════════════════════════
        if total_score >= 8.5:
            return ICTVeto(
                symbol=symbol, direction=direction, timestamp=timestamp,
                score=total_score, grade=ICTGrade.VETOED,
                reason=(
                    f"HYP047_SCORE_CEILING: score={total_score:.2f} >= 8.5 "
                    f"(WR decays monotonically above 8; score 9+ = negative R)"
                ),
                component_scores=scores, confirmations=confirmations, missing=missing,
                entry_level=entry_level,
            )

        # ══════════════════════════════════════════════════════════════════
        # STAGE 5.7 — Forensics combat rules (unified_forensics.py 2026-05-18)
        # EXP-001: NY_PM has -0.283R avg vs London +0.471R. Block entirely.
        # EXP-002: A+ grade (score>9.0) has 13% WR vs 39% for grade A.
        #          A+ in any session is anti-edge — treat as A for trade decision.
        # ══════════════════════════════════════════════════════════════════
        _current_session = getattr(session_pre, "session_name", "")
        if _current_session == "NY_PM" and not ny_am_mode:
            _blk_time = (
                timestamp.strftime("%H:%M UTC")
                if hasattr(timestamp, "strftime") else str(timestamp)
            )
            return ICTVeto(
                symbol=symbol, direction=direction, timestamp=timestamp,
                score=total_score, grade=ICTGrade.VETOED,
                reason=(
                    f"NY_PM_BLOCK @ {_blk_time} — session blocked, forensics: "
                    f"-0.283R avg vs London +0.471R"
                ),
                component_scores=scores, confirmations=confirmations, missing=missing,
                entry_level=entry_level,
            )
        # A+ paradox: downgrade A+ to A for trade decision only
        # (score still logged as A+ for monitoring)
        _effective_grade = ICTGrade.A if grade == ICTGrade.A_PLUS else grade

        # ── Commitment detector (2026-05-19) ──────────────────────────────
        # market_structure >= 1.5 predicts commitment failure at 87.5% accuracy.
        # London+GradeA+mkt<1.5: Sharpe 3.314 vs 1.864 unfiltered.
        # commitment_result pre-computed by orchestrator from CommitmentDetector.
        _commit = commitment_result
        if _commit is not None and getattr(_commit, "label", None) == "UNCOMMITTED":
            return ICTVeto(
                symbol=symbol, direction=direction, timestamp=timestamp,
                score=total_score, grade=ICTGrade.VETOED,
                reason=f"COMMITMENT_DETECTOR: {_commit.reason}",
                component_scores=scores, confirmations=confirmations, missing=missing,
                entry_level=entry_level,
            )
        # DEVELOPING: execute at reduced size (handled downstream via size_multiplier)
        _commit_size_mult = getattr(_commit, "size_multiplier", 1.0) if _commit is not None else 1.0
        # RQ-REST-018: log the real commitment SCORE (price-action quality 0-1), not the size
        # multiplier. In the live `ict.orchestrator` path _commit is None (the sovereign
        # CommitmentDetector bridge is orphaned, not wired into the live scanner), so the score
        # is None = honest "unknown" — NOT a constant 1.0 that biased Oracle toward "fully
        # committed". Consumers default/skip None. Wiring the detector into the live path is a
        # separate architectural change (see NEXT.md / param_change_log).
        _commit_score = getattr(_commit, "score", None) if _commit is not None else None

        # ══════════════════════════════════════════════════════════════════
        # STAGE 6 — Risk engine gate
        # Entry price = FVG midpoint (limit), not current market price.
        # ══════════════════════════════════════════════════════════════════
        if _effective_grade in (ICTGrade.A_PLUS, ICTGrade.A) or (ny_am_mode and _effective_grade == ICTGrade.B):
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
                direction=direction, entry=ep, stop_loss=stop, atr=atr, params=account,
                grade=getattr(_effective_grade, "value", None),
            )
            if isinstance(sz, RiskVeto):
                return ICTVeto(symbol=symbol, direction=direction, timestamp=timestamp,
                               score=total_score, grade=ICTGrade.VETOED,
                               reason=f"Risk gate: {sz.reason} — {sz.detail}",
                               component_scores=scores,
                               confirmations=confirmations, missing=missing,
                               entry_level=ep, stop=stop)
            # Apply allocation weight to position size (continuous dimmer)
            if _ict_alloc_weight < 1.0 and hasattr(sz, 'units') and sz.units:
                sz = sz.__class__(**{
                    **{f: getattr(sz, f) for f in sz.__dataclass_fields__},
                    'units': max(0, round(sz.units * _ict_alloc_weight)),
                    'risk_dollars': round(sz.risk_dollars * _ict_alloc_weight, 2),
                })
            _sig = ICTSignal(
                symbol=symbol, direction=direction, timestamp=timestamp,
                score=total_score, grade=grade, sizing=sz,
                session_status=session,
                sweep=recent_sweep, nearest_fvg=tap_fvg, nearest_ob=tap_ob,
                entry_level=ep,
                component_scores=scores,
                confirmations=confirmations, missing=missing,
            )
            try:
                import importlib as _il
                _dl = _il.import_module("sovereign.intelligence.decision_logger")
                _signal_ref = tap_fvg or tap_ob or recent_sweep
                _bars = None
                if _signal_ref is not None and hasattr(_signal_ref, "formed_at"):
                    try:
                        _elapsed = (timestamp - _signal_ref.formed_at).total_seconds()
                        _bars = max(0, int(_elapsed / 300))  # 5-min bars
                    except Exception:
                        pass
                # Loop 2: ICT-LOCAL entry-time snapshot (no sovereign import — the
                # importlib hook above is the sanctioned isolation-safe logger access).
                _snapshot = {
                    "score": round(float(total_score), 3),
                    "grade": getattr(_effective_grade, "value", str(_effective_grade)),
                    "commitment_score": _commit_score,
                    "commitment_size_mult": _commit_size_mult,
                    "component_scores": {k: round(float(v), 3) for k, v in scores.items()},
                    "n_confirmations": len(confirmations),
                    "n_missing": len(missing),
                    "has_fvg": (tap_fvg is not None or tap_ob is not None),
                    "has_sweep": recent_sweep is not None,
                    "session": getattr(session, "kill_zone_name", None),
                }
                # Codified ICT lessons structurally applied to this decision.
                _lessons = []
                if getattr(_effective_grade, "value", None) != getattr(grade, "value", None):
                    _lessons.append("L-001")  # grade-quality-inversion downgrade (A+→A)
                _dl.log_ict_decision(
                    signal=_sig,
                    commitment_score=_commit_score,
                    bars_since_signal=_bars,
                    present_state_snapshot=_snapshot,
                    active_lessons=_lessons,
                )
            except Exception:
                pass
            return _sig

        return ICTVeto(
            symbol=symbol, direction=direction, timestamp=timestamp,
            score=total_score, grade=grade,
            reason=f"Score {total_score:.1f} < threshold {threshold:.1f} (grade {grade})",
            component_scores=scores, confirmations=confirmations, missing=missing,
            entry_level=entry_level,
        )

    # ── Stage 2.5: Displacement helper ────────────────────────────────────── #

    @staticmethod
    def _displacement_score(
        df:        pd.DataFrame,
        sweep:     Optional[SweepResult],
        direction: str,
        atr:       float,
        displacement_body_atr: float,
    ) -> tuple[float, str]:
        """
        Check for a strong displacement candle after the sweep.

        Two checks:
          1. Body size ≥ _DISPLACEMENT_BODY_ATR × ATR  (committed directional move)
          2. Bar range (H-L) ≥ _VOL_PROXY_RATIO_MIN × avg_range  (volume proxy)

        Forex has no real tick volume.  Bar range relative to its 20-bar
        average is the standard proxy — wide-range bars indicate institutional
        participation regardless of tick count.

        ICT: "Displacement without institutional participation is not displacement."
        """
        if sweep is None:
            return 0.0, "no sweep to displace from"
        if atr <= 0:
            return 0.0, "ATR zero"

        try:
            sweep_loc = df.index.searchsorted(sweep.formed_at)
        except Exception:
            return 0.0, "sweep timestamp not in index"

        if sweep_loc >= len(df) - 1:
            return 0.0, "sweep is last bar — no post-sweep data"

        # Average bar range over last 20 bars (volume proxy baseline)
        lookback_start = max(0, sweep_loc - _VOL_PROXY_LOOKBACK)
        avg_range = float(
            (df["High"].iloc[lookback_start:sweep_loc] -
             df["Low"].iloc[lookback_start:sweep_loc]).mean()
        )

        best_score = 0.0
        best_label = ""
        end = min(sweep_loc + _DISPLACEMENT_LOOKBACK + 1, len(df))

        for i in range(sweep_loc + 1, end):
            bar   = df.iloc[i]
            o, c  = float(bar["Open"]), float(bar["Close"])
            h, lo = float(bar["High"]), float(bar["Low"])
            body  = abs(c - o)
            rng   = h - lo
            body_ratio  = body / atr if atr > 0 else 0.0
            range_ratio = rng / avg_range if avg_range > 0 else 0.0

            is_directional = (direction == "LONG" and c > o) or (direction == "SHORT" and c < o)
            if not is_directional:
                continue

            # Both gates must pass
            body_ok  = body_ratio  >= displacement_body_atr
            range_ok = range_ratio >= _VOL_PROXY_RATIO_MIN

            if body_ok and range_ok and body_ratio > best_score:
                best_score = body_ratio
                best_label = (
                    f"body {body_ratio:.2f}×ATR, range {range_ratio:.2f}×avg "
                    f"({'✓' if range_ok else '✗'} vol proxy) "
                    f"@ +{i - sweep_loc} bar"
                )
            elif body_ok and not range_ok and body_ratio > best_score * 0.5:
                # Body good but range too narrow — weak displacement
                best_label = (
                    f"body {body_ratio:.2f}×ATR but range only {range_ratio:.2f}×avg "
                    f"(need {_VOL_PROXY_RATIO_MIN}×) — low participation"
                )

        if best_score >= displacement_body_atr:
            score = min(0.6 + 0.4 * (best_score - displacement_body_atr) / displacement_body_atr, 1.0)
            return round(score, 3), best_label

        return 0.0, best_label or f"no qualifying displacement in {_DISPLACEMENT_LOOKBACK} bars"

    @staticmethod
    def _adr_exhaustion(df: pd.DataFrame) -> tuple[float, str]:
        """
        Compute how much of the Average Daily Range has been consumed today.

        Returns (adr_pct 0–1, label).
          ≥ 0.85 → hard veto (no room to run)
          ≥ 0.70 → soft penalty (limited room)
          < 0.70 → clear

        ADR is computed from the last _ADR_LOOKBACK calendar days of daily highs/lows.
        Intraday range is today's high minus today's low in the trailing hourly window.
        """
        if len(df) < 2:
            return 0.0, "insufficient data"

        try:
            # Daily high/low from the trailing hourly data
            # Group by date to get daily ranges
            df_copy = df.copy()
            df_copy.index = pd.to_datetime(df_copy.index, utc=True)
            daily = df_copy["High"].resample("D").max().dropna()
            daily_lo = df_copy["Low"].resample("D").min().dropna()

            if len(daily) < 2:
                return 0.0, "insufficient daily data"

            # ADR: mean of last N daily ranges (excluding today)
            daily_ranges = (daily - daily_lo).dropna()
            if len(daily_ranges) < 2:
                return 0.0, "insufficient range data"

            adr = float(daily_ranges.iloc[-_ADR_LOOKBACK:-1].mean())
            if adr <= 0:
                return 0.0, "ADR zero"

            # Today's range
            today_high = float(daily.iloc[-1])
            today_low  = float(daily_lo.iloc[-1])
            today_rng  = today_high - today_low

            pct   = today_rng / adr
            label = f"today {today_rng:.5f} / ADR {adr:.5f} = {pct:.0%}"
            return round(pct, 3), label

        except Exception as e:
            logger.debug("ADR computation failed: %s", e)
            return 0.0, f"ADR error: {e}"

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

    # ── Reddit sentiment filter ────────────────────────────────────────────── #

    _reddit_cache: dict = {}   # class-level cache, refreshed when stale

    def _apply_reddit_filter(
        self,
        symbol: str,
        direction: str,
        score: float,
        confirmations: list,
        missing: list,
    ) -> float:
        """
        Adjust ICT score based on Reddit retail sentiment (contrarian signal).
        Retail positioning on r/Forex is historically inverse-correlated with
        short-term price moves — crowded retail = fade opportunity.

        Rules:
          net_score > +5 AND direction==LONG  → veto level (crowded long, skip)
          net_score > +3 AND direction==LONG  → -0.5 penalty (crowded)
          net_score < -3 AND direction==SHORT → +0.3 boost (retail fading our short)
          Symmetric for SHORT direction.
        """
        import json, time
        from pathlib import Path as _Path

        REDDIT_PATH = _Path(__file__).parents[1] / "data" / "cache" / "reddit_sentiment.json"
        MAX_AGE_MIN = 120  # don't use data older than 2 hours

        try:
            # Cache at class level — reload only when stale
            cache_key = "reddit"
            cached_time = self._reddit_cache.get("_loaded_at", 0)
            if time.time() - cached_time > MAX_AGE_MIN * 60 or "data" not in self._reddit_cache:
                if REDDIT_PATH.exists():
                    self._reddit_cache["data"] = json.loads(REDDIT_PATH.read_text())
                    self._reddit_cache["_loaded_at"] = time.time()
                else:
                    return score

            reddit_data = self._reddit_cache.get("data", {})
            forex_sentiment = reddit_data.get("forex", {})

            # Normalise pair name: GBPUSD=X → GBPUSD
            pair = symbol.replace("=X", "").replace("/", "").upper()
            entry = forex_sentiment.get(pair)
            if not entry:
                return score  # pair not in Reddit data — no adjustment

            net = entry.get("net_score", 0.0)
            mentions = entry.get("mentions", 0)
            if mentions < 3:
                return score  # too few mentions to be meaningful

            # Apply contrarian logic
            if direction == "LONG":
                if net > 5:
                    missing.append(f"⚠ Reddit CROWDED LONG: {pair} net={net:.1f} ({mentions} mentions) — retail piling in")
                    score -= 1.5  # hard penalty, likely to drop below threshold
                elif net > 3:
                    missing.append(f"⚠ Reddit retail long {pair} net={net:.1f} — slight fade")
                    score -= 0.5
                elif net < -3:
                    confirmations.append(f"✓ Reddit fading SHORT on {pair} net={net:.1f} — contrarian support for LONG")
                    score += 0.3

            elif direction == "SHORT":
                if net < -5:
                    missing.append(f"⚠ Reddit CROWDED SHORT: {pair} net={net:.1f} ({mentions} mentions) — retail piling in")
                    score -= 1.5
                elif net < -3:
                    missing.append(f"⚠ Reddit retail short {pair} net={net:.1f} — slight fade")
                    score -= 0.5
                elif net > 3:
                    confirmations.append(f"✓ Reddit fading LONG on {pair} net={net:.1f} — contrarian support for SHORT")
                    score += 0.3

        except Exception:
            pass  # never block a trade due to Reddit failure

        return score

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
    override = os.environ.get("ICT_CONFIG_PATH")
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "config", "ict_params.yml")
