#!/usr/bin/env python3
"""Briefing context — the data contract + continuous sizing multiplier for the L1 synthesizer.

This is the wiring layer the "AlphaZero and Stockfish Report" (research/ALPHAZERO_STOCKFISH_REPORT.md)
asks for on the AlphaZero half: turn the morning briefing synthesizer's self-scored call into
something the rest of the system can READ and (softly) act on.

Three responsibilities, all freeze-safe and provenance-honest:

  1. load_current_briefing()   — read data/agent/daily_briefing.json (the agent-facing contract
                                  morning_market_briefing.build() writes each run).
  2. compute_multiplier(...)   — turn confidence + regime + (optional) direction-vs-carry alignment
                                  into a CONTINUOUS sizing multiplier. NEVER a veto: the return is a
                                  strictly-positive float in a conservative band, and it degrades to
                                  exactly 1.0 (no effect) whenever the call is a deterministic fallback,
                                  unverified-absent, or self-contradictory.
  3. briefing_stamp()          — a compact, verified=False context blob to attach to every decision-log
                                  entry so a later pass can match the briefing call to the trade outcome.

PROVENANCE DISCIPLINE (non-negotiable): the briefing is Oracle's analytical journal, NOT a validated
edge (provenance.verified is False by design). Its confidence is a self-scored call whose calibration
is unproven until the scorecard accumulates a real sample. So the multiplier band is deliberately
narrow (±20% at the extreme) and the design is fail-to-neutral: uncertainty shrinks the effect toward
1.0, it never manufactures conviction.

INSTRUMENT-MISMATCH HONESTY: the synthesizer's directional_bias is an ES/NQ (equities) regime read;
the carry engine trades G10 forex. They are not the same instrument. So the multiplier's PRIMARY
input is confidence + regime QUALITY (a rotation warning lowers trust in ANY signal), and the
direction-vs-carry alignment nudge is applied ONLY when a caller supplies a same-context directional
read. The report's "direction aligns with the carry signal" is honoured as a small optional nudge,
not the backbone of the multiplier.

ISOLATION: pure-Python, reads one JSON file. Imports nothing from ict/ or the execution path. Safe to
import from anywhere on the sovereign side.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRIEFING_JSON = ROOT / "data" / "agent" / "daily_briefing.json"

# --- multiplier band (conservative by design — this is context, not a proven edge) ---
MULT_FLOOR = 0.80        # hard floor > 0 → can NEVER veto a trade
MULT_CEIL = 1.20         # hard ceiling → high confidence can nudge up, never blow out size
_CONF_SPAN = 0.15        # confidence maps to ±15% around neutral before other adjustments
_ROTATION_HAIRCUT = 0.90  # ROTATION_WARN → treat every signal as lower quality
_ALIGN_BONUS = 1.05      # same-context directional agreement with carry → small size-up
_ALIGN_PENALTY = 0.95    # same-context directional disagreement → small size-down


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_current_briefing(path: Path | None = None) -> dict | None:
    """Return the current agent-facing briefing dict, or None if absent/unreadable.

    The path is resolved from the module global at CALL time (not bound as a default) so it
    stays overridable in tests and consistent if BRIEFING_JSON is reassigned.
    """
    p = path if path is not None else BRIEFING_JSON
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def compute_multiplier(
    confidence: int | float | None,
    directional_bias: str | None,
    *,
    synthesis_source: str | None = None,
    regime_call: str | None = None,
    carry_direction: str | None = None,
) -> dict:
    """Continuous sizing multiplier from the briefing call. NEVER returns <= 0.

    Fail-to-neutral rules (any → multiplier 1.0, effect_applied False):
      - deterministic fallback (no real synthesis this run)
      - confidence missing
      - directional_bias missing/NEUTRAL AND no regime signal to act on

    carry_direction (optional): a same-context directional read ("LONG"/"SHORT") the CALLER
    supplies (e.g. the sign of a carry setup it is about to size). When present, agreement with
    the briefing bias applies a small bonus, disagreement a small penalty. Absent → no nudge.

    Returns a dict: {multiplier, components, effect_applied, verified, computed_at, inputs}.
    """
    src = (synthesis_source or "").lower()
    components: dict[str, float] = {}
    reason = None

    # Fail-to-neutral gates.
    if "deterministic" in src or "fallback" in src:
        reason = "deterministic_fallback — no real synthesis this run; multiplier neutralised"
    elif confidence is None:
        reason = "confidence absent — multiplier neutralised"

    if reason is not None:
        return {
            "multiplier": 1.0, "components": {}, "effect_applied": False,
            "verified": False, "reason": reason, "computed_at": _now(),
            "inputs": {"confidence": confidence, "directional_bias": directional_bias,
                       "synthesis_source": synthesis_source, "regime_call": regime_call,
                       "carry_direction": carry_direction},
        }

    try:
        conf = float(confidence)
    except Exception:
        conf = 50.0
    conf = max(0.0, min(100.0, conf))

    # 1) confidence → ±_CONF_SPAN around neutral (conf=50 is neutral / no information).
    m_conf = 1.0 + ((conf - 50.0) / 50.0) * _CONF_SPAN
    components["confidence"] = round(m_conf, 4)
    mult = m_conf

    # 2) regime QUALITY haircut — a rotation warning lowers trust in every signal.
    if (regime_call or "").upper() in ("ROTATION_WARN", "ROTATION_DIVERGENCE"):
        components["regime_rotation_haircut"] = _ROTATION_HAIRCUT
        mult *= _ROTATION_HAIRCUT

    # 3) OPTIONAL same-context direction-vs-carry alignment nudge.
    bias = (directional_bias or "").upper()
    cd = (carry_direction or "").upper()
    if cd in ("LONG", "SHORT") and bias in ("LONG", "SHORT"):
        if bias == cd:
            components["carry_alignment"] = _ALIGN_BONUS
            mult *= _ALIGN_BONUS
        else:
            components["carry_alignment"] = _ALIGN_PENALTY
            mult *= _ALIGN_PENALTY

    mult = max(MULT_FLOOR, min(MULT_CEIL, mult))
    return {
        "multiplier": round(mult, 4),
        "components": components,
        "effect_applied": abs(mult - 1.0) > 1e-9,
        "verified": False,   # briefing is never a verified edge
        "reason": "continuous multiplier — context only, not a veto",
        "computed_at": _now(),
        "inputs": {"confidence": conf, "directional_bias": bias,
                   "synthesis_source": synthesis_source, "regime_call": regime_call,
                   "carry_direction": carry_direction},
    }


def multiplier_from_briefing(briefing: dict | None = None, *, carry_direction: str | None = None) -> dict:
    """Convenience: compute the multiplier straight from a briefing dict (or the current file)."""
    b = briefing if briefing is not None else load_current_briefing()
    if not b:
        return {"multiplier": 1.0, "components": {}, "effect_applied": False, "verified": False,
                "reason": "no briefing available — multiplier neutralised", "computed_at": _now(),
                "inputs": {}}
    return compute_multiplier(
        b.get("confidence"), b.get("directional_bias"),
        synthesis_source=b.get("synthesis_source"),
        regime_call=b.get("regime_call") or b.get("meta_regime"),
        carry_direction=carry_direction,
    )


def briefing_stamp(briefing: dict | None = None) -> dict | None:
    """Compact, verified=False context blob to attach to a decision-log entry.

    Lets a later pass match a trade's outcome back to the briefing call that preceded it
    (the AlphaZero learning loop's per-trade attribution). Returns None if no briefing exists,
    so callers can attach conditionally without guarding.
    """
    b = briefing if briefing is not None else load_current_briefing()
    if not b:
        return None
    return {
        "date": b.get("date"),
        "generated_at": b.get("generated_at"),
        "directional_bias": b.get("directional_bias"),
        "confidence": b.get("confidence"),
        "regime_call": b.get("regime_call") or b.get("meta_regime"),
        "key_level": b.get("key_level"),
        "synthesis_source": b.get("synthesis_source"),
        "verified": False,
        "role": "context_only",
    }


# ─── STAGED — DO NOT WIRE until 2026-07-28 + explicit Colin ledger stamp ──────────────────────────
# The carry SIZING ENGINE hookup crosses the execution-path freeze (CLAUDE.md standing constraint,
# freeze runs until 2026-07-28). This function is the ONE-LINE switch to flip after the unlock.
# It is intentionally NOT called anywhere. Wiring it in requires editing carry_engine, which is
# frozen. See NEXT.md → "Staged for 2026-07-28 unlock".
def staged_carry_size_multiplier(carry_direction: str | None = None) -> float:  # pragma: no cover
    """STAGED (unwired). After 2026-07-28 unlock, the carry sizing engine calls this to fold the
    briefing multiplier into position size, e.g.:

        from sovereign.briefing.briefing_context import staged_carry_size_multiplier
        size *= staged_carry_size_multiplier(carry_direction=sig.direction)

    Until the unlock this is dead code by design — importing it has no effect on any live path.
    """
    return multiplier_from_briefing(carry_direction=carry_direction)["multiplier"]
