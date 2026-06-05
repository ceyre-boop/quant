"""
DecisionChain — sovereign/oracle/decision_chain.py

Single gateway for every trade. Five questions in order.
No shortcuts, no bypasses. Logs every decision to data/decisions/decision_chain.jsonl.

Q1: Should I trade today?       (DailyReadiness)
Q2: What is the present state?  (PresentState — stub until Component 2 is built)
Q3: What could happen?          (OutcomeEnumerator — stub until Component 3)
Q4: How much to risk?           (DynamicSizer — stub until Component 4)
Q5: PropRiskManager final check (already live)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

ROOT      = Path(__file__).parents[2]
CHAIN_LOG = ROOT / "data" / "decisions" / "decision_chain.jsonl"


@dataclass
class DecisionRecord:
    pair:          str
    timestamp:     str
    direction:     str = ""
    verdict:       str = "PENDING"
    q1_status:     str = ""
    q1_reason:     str = ""
    q2_constraint: float = 0.0
    q2_dimensions: int   = 0
    q3_ev:         float = 0.0
    q3_confidence: str   = ""
    q4_risk:       float = 0.0
    q4_reasoning:  str   = ""
    q5_allowed:    bool  = True
    q5_reason:     str   = ""
    execution:     dict  = field(default_factory=dict)


class DecisionChain:
    """
    Evaluates a candidate trade through all 5 oracle questions.
    Returns a result dict with `status` key.

    Stubs for Q2/Q3/Q4 allow Phase 1 to be live immediately;
    plug in real components as they are built.
    """

    def evaluate(
        self,
        pair:      str,
        direction: str,
        df:        pd.DataFrame,
        bridge=None,
        dry_run:   bool = False,
    ) -> dict:
        ts  = datetime.now(timezone.utc).isoformat()
        rec = DecisionRecord(pair=pair, timestamp=ts, direction=direction)

        # ── Kill switch: master freeze blocks ALL trade evaluation (any caller) ──
        from sovereign.utils.kill_switch import trading_frozen
        frz = trading_frozen()
        if frz:
            rec.verdict   = "DENIED_FROZEN"
            rec.q1_status = "FROZEN"
            rec.q1_reason = f"SYSTEM FROZEN ({frz.get('mode')}): {frz.get('reason', '')}"
            self._log(rec)
            return {"status": "DENIED", "reason": f"SYSTEM_FROZEN: {frz.get('reason', '')}"}

        # ── Q1: Should I trade today? ─────────────────────────────────────
        from sovereign.oracle.daily_readiness import DailyReadiness
        readiness = DailyReadiness(bridge).assess()
        rec.q1_status = readiness.status
        rec.q1_reason = readiness.reason
        if readiness.status == "SIT":
            rec.verdict = "DENIED_READINESS"
            self._log(rec)
            return {"status": "DENIED", "reason": readiness.reason}

        # ── Q2: What is the present state? ───────────────────────────────
        # Stub: accept all — plug in PresentState when Component 2 is built
        present = _PresentStateStub()
        rec.q2_constraint  = present.constraint_score
        rec.q2_dimensions  = present.dimensions_aligned
        if present.constraint_score < 0.33:
            rec.verdict = "DENIED_CONSTRAINT"
            self._log(rec)
            return {"status": "DENIED", "reason": f"CONSTRAINT_LOW: {present.dimensions_aligned}/6"}

        # ── Q3: What could happen? ────────────────────────────────────────
        # Stub: neutral EV — plug in OutcomeEnumerator when Component 3 is built
        outcomes = _outcomes_stub()
        rec.q3_ev         = outcomes["expected_value_r"]
        rec.q3_confidence = outcomes["confidence"]
        if not outcomes["trade_worthy"]:
            rec.verdict = "DENIED_EV"
            self._log(rec)
            return {"status": "DENIED", "reason": f"EV_LOW: {outcomes['expected_value_r']:.3f}R"}

        # ── Q4: How much to risk? — Dynamic Risk Engine (SOLE sizing authority) ──
        from sovereign.risk import engine_adapter
        _entry = float(df["Close"].iloc[-1])
        _stop = (float(df["Low"].tail(5).min()) if direction == "LONG"
                 else float(df["High"].tail(5).max()))
        _grade = getattr(rec, "grade", None) or "B"   # ungraded signals get the conservative B base
        _decision = engine_adapter.size(pair, direction, _entry, _stop, grade=_grade)
        risk_pct = _decision.final_risk_pct
        if readiness.status == "REDUCE":
            risk_pct = round(risk_pct * 0.5, 5)     # readiness only ever reduces further
        sizing = {"risk_pct": risk_pct, "reasoning": _decision.reasoning}
        rec.q4_risk      = sizing["risk_pct"]
        rec.q4_reasoning = sizing["reasoning"]

        # ── Q5: PropRiskManager final check ──────────────────────────────
        from sovereign.risk.prop_risk_manager import PropRiskManager
        if bridge is None:
            from sovereign.execution.oanda_bridge import OandaBridge
            bridge = OandaBridge()
        risk_check = PropRiskManager(bridge).check_trade_allowed(pair, direction, sizing["risk_pct"])
        rec.q5_allowed = risk_check.allowed
        rec.q5_reason  = risk_check.reason
        if not risk_check.allowed:
            rec.verdict = "DENIED_RISK"
            self._log(rec)
            return {"status": "DENIED", "reason": risk_check.reason}

        # ── All 5 passed ──────────────────────────────────────────────────
        rec.verdict = "APPROVED"

        if dry_run:
            self._log(rec)
            return {
                "status":    "DRY_RUN",
                "risk_pct":  sizing["risk_pct"],
                "reasoning": sizing["reasoning"],
                "q1":        readiness.status,
            }

        # Execute
        oanda_pair = _to_oanda_pair(pair)
        entry = float(df["Close"].iloc[-1])
        stop  = float(df["Low"].tail(5).min())  if direction == "LONG" \
                else float(df["High"].tail(5).max())
        risk_dist = abs(entry - stop)
        if risk_dist == 0:
            rec.verdict   = "VETOED"
            rec.execution = {"status": "VETOED", "reason": "ZERO_RISK_DISTANCE"}
            self._log(rec)
            return rec.execution

        tp    = (entry + 2 * risk_dist) if direction == "LONG" else (entry - 2 * risk_dist)
        units = bridge.compute_units(oanda_pair, entry, stop, sizing["risk_pct"])
        if units == 0:
            rec.verdict   = "VETOED"
            rec.execution = {"status": "VETOED", "reason": "ZERO_UNITS"}
            self._log(rec)
            return rec.execution

        result = bridge.place_trade(oanda_pair, direction, units, stop, tp)
        rec.execution = result
        self._log(rec)
        return result

    # ── logging ──────────────────────────────────────────────────────────────

    def _log(self, rec: DecisionRecord) -> None:
        CHAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp":     rec.timestamp,
            "pair":          rec.pair,
            "direction":     rec.direction,
            "verdict":       rec.verdict,
            "q1_status":     rec.q1_status,
            "q1_reason":     rec.q1_reason,
            "q2_constraint": rec.q2_constraint,
            "q2_dimensions": rec.q2_dimensions,
            "q3_ev":         rec.q3_ev,
            "q3_confidence": rec.q3_confidence,
            "q4_risk":       rec.q4_risk,
            "q4_reasoning":  rec.q4_reasoning,
            "q5_allowed":    rec.q5_allowed,
            "q5_reason":     rec.q5_reason,
            "execution":     rec.execution,
        }
        with CHAIN_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info("[DecisionChain] %s %s → %s | Q1=%s Q5=%s",
                    rec.pair, rec.direction, rec.verdict, rec.q1_status, rec.q5_reason or "OK")


# ── Stubs (replaced when real components are built) ──────────────────────────

class _PresentStateStub:
    """Placeholder until PresentState (Component 2) is built."""
    constraint_score  = 1.0   # always pass — no real assessment yet
    dimensions_aligned = 6
    has_fvg           = False
    matching_green    = False
    green_hit_rate    = None
    session           = "UNKNOWN"


def _outcomes_stub() -> dict:
    """Placeholder until OutcomeEnumerator (Component 3) is built."""
    return {
        "expected_value_r": 0.30,
        "trade_worthy":     True,
        "confidence":       "STUB",
    }


def _sizing_stub(readiness) -> dict:
    """Placeholder until DynamicSizer (Component 4) is built."""
    base = 0.0075
    risk = base * (0.5 if readiness.status == "REDUCE" else 1.0)
    return {
        "risk_pct":  risk,
        "reasoning": f"stub: base={base} readiness={readiness.status}",
    }


def _to_oanda_pair(pair: str) -> str:
    """Convert any pair format to OANDA 'XXX_YYY'."""
    clean = pair.replace("=X", "").replace("_", "")
    if len(clean) == 6:
        return clean[:3] + "_" + clean[3:]
    return clean
