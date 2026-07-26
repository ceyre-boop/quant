"""
Firebase State Writer — Alta Investments
integration/firebase_state_writer.py

Shared best-effort RTDB write helpers for system components.

ALL writes are best-effort: wrapped in try/except so a Firebase outage
or misconfigured credential NEVER breaks the core trading logic.

Usage (from any component):
    from integration.firebase_state_writer import (
        broadcast_oracle_reflection,
        broadcast_petroulas_decision,
        broadcast_carry_signals,
        broadcast_mesolimbic_session,
    )

Each function fails silently (logs a warning) if Firebase is unavailable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

DB_URL = "https://clawd-trading-7b8de-default-rtdb.firebaseio.com/"


def _rtdb_ref(path: str):
    """Return a firebase_admin.db reference, or None if unavailable."""
    try:
        from firebase_admin import db
        return db.reference(path)
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Oracle
# =============================================================================

def broadcast_oracle_reflection(output: dict) -> None:
    """
    Broadcast reflect_cycle output to:
      /oracle/latest_reflection
      /oracle/health

    Called from sovereign/oracle/reflect_cycle.py after writing the local
    JSON file.  Never raises — failure is logged and swallowed.

    Args:
        output: The full dict returned by run_reflect().
    """
    try:
        reflection = output.get("reflection", {})
        cand = reflection.get("candidate_lesson") if isinstance(reflection, dict) else None
        health_note = reflection.get("system_health_note") if isinstance(reflection, dict) else None
        retirement = reflection.get("retirement_flag") if isinstance(reflection, dict) else None

        ref = _rtdb_ref("oracle/latest_reflection")
        if ref:
            ref.set({
                "date": output.get("date"),
                "generated_at": output.get("generated_at"),
                "estimated_cost_usd": output.get("estimated_cost_usd"),
                "harvest_days_read": output.get("harvest_days_read"),
                "candidate_lesson": cand,
                "system_health_note": health_note,
                "retirement_flag": retirement,
                "updated_at": _now_iso(),
            })

        # Update health node
        health_ref = _rtdb_ref("oracle/health")
        if health_ref:
            health_ref.update({
                "last_reflect_date": output.get("date"),
                "last_reflect_ok": True,
                "degraded": False,
                "degraded_reason": None,
                "updated_at": _now_iso(),
            })

        # Update lessons summary if active lessons can be counted
        try:
            import json
            from pathlib import Path
            ROOT = Path(__file__).resolve().parents[1]
            wisdom = ROOT / "I_am_a_good_trader.md"
            if wisdom.exists():
                text = wisdom.read_text()
                active = text.count("## Lesson")
                lessons_ref = _rtdb_ref("oracle/lessons_summary")
                if lessons_ref:
                    lessons_ref.update({
                        "active_lessons": active,
                        "last_sync": _now_iso(),
                        "updated_at": _now_iso(),
                    })
        except Exception:
            pass

        logger.info("[FirebaseStateWriter] oracle/latest_reflection broadcast OK")

    except Exception as exc:
        logger.warning(f"[FirebaseStateWriter] oracle broadcast failed (non-fatal): {exc}")


# =============================================================================
# Petroulas Gate
# =============================================================================

def broadcast_petroulas_decision(decision) -> None:
    """
    Broadcast a PetroulsasDecision to /petroulas/decisions.

    Called from imbalance_engine/petroulas_gate.py PetroulsasGate.evaluate()
    after each gate run.  Never raises.

    Args:
        decision: PetroulsasDecision dataclass instance.
    """
    try:
        kimi = decision.kimi_score
        kimi_data = None
        if kimi is not None:
            kimi_data = {
                "magnitude": kimi.magnitude,
                "conviction": kimi.conviction,
                "petroulas_worthy": kimi.petroulas_worthy,
                "consensus_blindspot": kimi.consensus_blindspot,
                "time_horizon_days": kimi.time_horizon_days,
            }

        payload = {
            "symbol": decision.symbol,
            "approved": decision.approved,
            "position_size_pct": decision.position_size_pct,
            "normal_size_pct": decision.normal_size_pct,
            "fault_quality": decision.fault_quality,
            "xgb_confidence": decision.xgb_confidence,
            "stress_score": decision.stress_score,
            "kimi_score": kimi_data,
            "reason": decision.reason,
            "thesis_id": decision.thesis_id,
            "timestamp": decision.timestamp,
            "updated_at": _now_iso(),
        }

        # Write to last_decision
        ref = _rtdb_ref("petroulas/decisions/last_decision")
        if ref:
            ref.set(payload)

        # Increment counter
        counters_ref = _rtdb_ref("petroulas/decisions")
        if counters_ref:
            existing = counters_ref.get() or {}
            key = "approved_count" if decision.approved else "rejected_count"
            counters_ref.update({
                key: (existing.get(key, 0) or 0) + 1,
                "updated_at": _now_iso(),
            })

        # If approved, record under active_theses keyed by thesis_id
        if decision.approved and kimi is not None:
            thesis_ref = _rtdb_ref(f"petroulas/active_theses/{decision.thesis_id}")
            if thesis_ref:
                thesis_ref.set({
                    **payload,
                    "falsification_test": kimi.falsification_test if kimi else None,
                    "arithmetic_proof": kimi.arithmetic_proof if kimi else None,
                })

        logger.info(
            f"[FirebaseStateWriter] petroulas/decisions broadcast OK "
            f"({decision.symbol} approved={decision.approved})"
        )

    except Exception as exc:
        logger.warning(f"[FirebaseStateWriter] petroulas broadcast failed (non-fatal): {exc}")


# =============================================================================
# Carry Engine
# =============================================================================

def broadcast_carry_signals(signals: list) -> None:
    """
    Broadcast carry engine scan results to:
      /carry/rate_differentials/{pair}
      /carry/positions/{pair}

    Called from sovereign/forex/carry_engine.py CarryEngine.log_signals()
    after each daily scan.  Never raises.

    Args:
        signals: List of CarrySignal dataclass instances.
    """
    try:
        for sig in signals:
            pair = sig.ticker  # e.g. "EURUSD=X" → strip suffix if needed
            clean = pair.replace("=X", "").replace("=", "")

            # Rate differentials
            diff_ref = _rtdb_ref(f"carry/rate_differentials/{clean}")
            if diff_ref:
                diff_ref.update({
                    "differential_bps": sig.carry_spread_bps,
                    "carry_direction": sig.direction,
                    "last_updated": _now_iso(),
                })

            # Position state
            pos_ref = _rtdb_ref(f"carry/positions/{clean}")
            if pos_ref:
                pos_ref.update({
                    "active": sig.direction != "FLAT",
                    "direction": sig.direction,
                    "size_lots": sig.units if sig.direction != "FLAT" else None,
                    "updated_at": _now_iso(),
                })

            # Also update session positions
            session_ref = _rtdb_ref(f"session/positions/{clean}")
            if session_ref:
                session_ref.update({
                    "direction": sig.direction,
                    "updated_at": _now_iso(),
                })

        # Update carry/meta with last scan time
        meta_ref = _rtdb_ref("carry/meta")
        if meta_ref:
            meta_ref.update({"last_scan": _now_iso()})

        logger.info(
            f"[FirebaseStateWriter] carry signals broadcast OK "
            f"({len(signals)} pairs)"
        )

    except Exception as exc:
        logger.warning(f"[FirebaseStateWriter] carry broadcast failed (non-fatal): {exc}")


# =============================================================================
# Mesolimbic / Cursus Honorum
# =============================================================================

def broadcast_mesolimbic_session(
    player: str,
    score: int,
    total: int,
    elo_before: int,
    elo_after: int,
    category_results: dict,
    session_id: Optional[str] = None,
) -> None:
    """
    Broadcast a Cursus Honorum game session result to:
      /mesolimbic/elo/{player}
      /mesolimbic/category_scores/{player}
      /mesolimbic/sessions

    Called from games/cursus_honorum/session.py after each game session.
    Never raises.

    Args:
        player: "colin" or "alphazero"
        score: Correct answers
        total: Total questions
        elo_before: Elo before this session
        elo_after: Elo after this session
        category_results: {category: {"correct": int, "total": int}}
        session_id: Optional session identifier (defaults to timestamp)
    """
    try:
        sid = session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ts = _now_iso()

        # Update player Elo
        elo_ref = _rtdb_ref(f"mesolimbic/elo/{player}")
        if elo_ref:
            existing = elo_ref.get() or {}
            elo_ref.update({
                "current": elo_after,
                "peak": max(elo_after, existing.get("peak", 0) or 0),
                "games_played": (existing.get("games_played", 0) or 0) + 1,
                "last_game": ts,
            })

        # Update category scores
        for cat, results in category_results.items():
            pct = results["correct"] / results["total"] if results.get("total") else 0.0
            cat_ref = _rtdb_ref(f"mesolimbic/category_scores/{player}/{cat}")
            if cat_ref:
                cat_ref.update({
                    "score": round(pct, 3),
                    "correct": results["correct"],
                    "total": results["total"],
                    "last_tested": ts,
                })

        # Log session
        session_ref = _rtdb_ref(f"mesolimbic/sessions/history/{sid}")
        if session_ref:
            session_ref.set({
                "player": player,
                "score": score,
                "total": total,
                "pct": round(score / total, 3) if total else 0.0,
                "elo_before": elo_before,
                "elo_after": elo_after,
                "elo_delta": elo_after - elo_before,
                "category_results": category_results,
                "timestamp": ts,
            })

        # Update sessions meta
        sessions_ref = _rtdb_ref("mesolimbic/sessions")
        if sessions_ref:
            existing = sessions_ref.get() or {}
            sessions_ref.update({
                "total_sessions": (existing.get("total_sessions", 0) or 0) + 1,
                "last_session": ts,
                "updated_at": ts,
            })

        logger.info(
            f"[FirebaseStateWriter] mesolimbic session broadcast OK "
            f"({player} {score}/{total} Elo {elo_before}→{elo_after})"
        )

    except Exception as exc:
        logger.warning(f"[FirebaseStateWriter] mesolimbic broadcast failed (non-fatal): {exc}")
