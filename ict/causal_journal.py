"""
ict/causal_journal.py
=====================
Layer-8 ICT causal-chain journal — the update_outcome() loop for ICT setups.

Every evaluated ICT setup writes ONE record here with its full causal chain:
which level fired, what the regime/bias were, whether bias was aligned, the size
multiplier, the ICT grade + component scores, the computed R:R, and the ACTION the
pipeline took (ENTERED / DISCARDED / VETOED / BELOW_MIN_RR / NO_OPPOSING_LEVEL) with
a discard_reason. When the trade closes, update_outcome() back-fills outcome +
outcome_r — CLAUDE.md NON-NEGOTIABLE #2 (the Oracle cannot learn without it).

    Journal file: data/agent/ict_causal_chain.jsonl  (one JSON object per line)

This makes "was it the level or the bias?" answerable, because DISCARDED and VETOED
setups are logged too — not just entries.

ISOLATION (CLAUDE.md NON-NEGOTIABLE #1)
---------------------------------------
Imports ONLY the standard library. Imports NOTHING from sovereign/. It MAY read the
neutral alta_platform contract files (regime/bias JSON) as *data* — never their code.
It reads the pipeline's result objects by duck-typing (getattr), so it never needs to
import the ict result types either. Safe to import from ict/.

Everything fails SOFT: a journal write must never block or crash a trade decision.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
JOURNAL_PATH = REPO / "data" / "agent" / "ict_causal_chain.jsonl"
REGIME_CONTRACT = REPO / "data" / "agent" / "system_regime_state.json"
BIAS_DIR = REPO / "data" / "bias"

# Valid actions (schema-level enum). Kept as constants — descriptors, not thresholds.
ENTERED = "ENTERED"
DISCARDED = "DISCARDED"
VETOED = "VETOED"
BELOW_MIN_RR = "BELOW_MIN_RR"
NO_OPPOSING_LEVEL = "NO_OPPOSING_LEVEL"


# ── helpers ─────────────────────────────────────────────────────────────────── #

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_symbol(sym: str) -> str:
    return (sym or "").upper().replace("=X", "").replace("_", "").replace("/", "").replace("-", "")


def _iso(ts: Any) -> str:
    if hasattr(ts, "isoformat"):
        try:
            return ts.isoformat()
        except Exception:
            return str(ts)
    return str(ts)


def _date_part(ts: Any) -> str:
    s = _iso(ts)
    return s[:10] if len(s) >= 10 else s


def make_setup_id(symbol: str, direction: str, entry_level: Optional[float], timestamp: Any) -> str:
    """Deterministic setup id, reproducible from either the evaluation site or the
    paper-trader close site WITHOUT threading state between them.

    Keyed on (symbol, direction, rounded entry, calendar date). Under the paper
    protocol (one open trade per pair, max 3 concurrent, closed same session) this
    is effectively unique among concurrently-open setups, so update_outcome can match
    the still-OPEN ENTERED record by id. No numeric threshold is hardcoded — the
    rounding just mirrors the paper trader's own round(entry, 5).
    """
    sym = _norm_symbol(symbol)
    d = (direction or "").upper()
    ent = f"{float(entry_level):.5f}" if entry_level is not None else "NA"
    return f"{sym}_{d}_{ent}_{_date_part(timestamp)}"


def _read_regime_state() -> Optional[str]:
    """ict_equities regime verdict from the neutral contract, or None if unavailable.
    Reads JSON as data — no import of the regime module (isolation-safe)."""
    try:
        if not REGIME_CONTRACT.exists():
            return None
        d = json.loads(REGIME_CONTRACT.read_text(encoding="utf-8"))
        sec = ((d.get("strategies") or {}).get("ict_equities") or {})
        v = sec.get("verdict") or sec.get("status")
        if v:
            return str(v)
        # ict_equities section is an empty stub today → fall back to top-level status.
        return str(d.get("status")) if d.get("status") else None
    except Exception:
        return None


def _read_bias_state(timestamp: Any) -> Optional[str]:
    """Latest daily directional bias (LONG/SHORT/NEUTRAL) from data/bias/bias_*.json.
    Prefers the file matching the setup's date; else the newest. None if absent."""
    try:
        if not BIAS_DIR.exists():
            return None
        day = _date_part(timestamp)
        exact = BIAS_DIR / f"bias_{day}.json"
        path = exact if exact.exists() else None
        if path is None:
            files = sorted(BIAS_DIR.glob("bias_*.json"))
            if not files:
                return None
            path = files[-1]
        d = json.loads(path.read_text(encoding="utf-8"))
        direction = d.get("direction")
        return str(direction).upper() if direction else None
    except Exception:
        return None


def _bias_aligned(direction: str, bias_state: Optional[str]) -> Optional[bool]:
    """True/False if bias is directional and comparable; None if bias is NEUTRAL or
    unknown (honest — never coerce an unknown into a False)."""
    if not bias_state or bias_state in ("NEUTRAL", "FLAT", "NONE"):
        return None
    d = (direction or "").upper()
    if d not in ("LONG", "SHORT"):
        return None
    return (d == "LONG" and bias_state == "LONG") or (d == "SHORT" and bias_state == "SHORT")


def _classify_action(result: Any) -> tuple[str, Optional[str]]:
    """Map a pipeline result (ICTSignal | ICTVeto) to (action, discard_reason).

    ICTSignal (has .sizing) → ENTERED. Otherwise inspect the veto reason string.
    Duck-typed so this module imports no ict result types (isolation stays trivial).
    """
    if getattr(result, "sizing", None) is not None:
        return ENTERED, None
    reason = str(getattr(result, "reason", "") or "")
    low = reason.lower()
    if "risk gate" in low and ("r:r" in low or "rr" in low or "reward" in low or "min_rr" in low):
        return BELOW_MIN_RR, reason
    if (
        "no confirmed sweep" in low
        or "no fvg or ob" in low
        or "no opposing" in low
        or "no sweep" in low
        or "detected in window" in low
    ):
        return NO_OPPOSING_LEVEL, reason
    if "score" in low and "threshold" in low:
        return DISCARDED, reason
    return VETOED, reason


def _extract_level(result: Any) -> dict[str, Any]:
    """Pull level identity/quality from whichever structural object is present:
    nearest FVG (preferred) → order block → sweep. All optional."""
    out = {"level_id": None, "level_type": None, "level_tf": None, "level_quality_score": None}

    fvg = getattr(result, "nearest_fvg", None)
    ob = getattr(result, "nearest_ob", None)
    sweep = getattr(result, "sweep", None)

    if fvg is not None:
        out["level_type"] = "FVG"
        out["level_id"] = f"FVG_{getattr(fvg, 'kind', '')}_{_iso(getattr(fvg, 'formed_at', ''))}"
        out["level_quality_score"] = _safe_float(getattr(fvg, "size_atr_ratio", None))
    elif ob is not None:
        out["level_type"] = "OB"
        out["level_id"] = f"OB_{getattr(ob, 'kind', '')}_{_iso(getattr(ob, 'formed_at', ''))}"
        out["level_quality_score"] = _safe_float(getattr(ob, "impulse_atr_ratio", None))
    elif sweep is not None:
        out["level_type"] = "SWEEP"
        out["level_id"] = f"SWEEP_{getattr(sweep, 'direction', '')}_{_iso(getattr(sweep, 'formed_at', ''))}"
        out["level_quality_score"] = _safe_float(getattr(sweep, "wick_atr_ratio", None))
    return out


def _safe_float(v: Any) -> Optional[float]:
    try:
        return round(float(v), 5)
    except (TypeError, ValueError):
        return None


def _compute_rr(entry: Optional[float], stop: Optional[float], target: Optional[float]) -> Optional[float]:
    """Reward-to-risk from entry/stop/target when all three are present."""
    try:
        e, s, t = float(entry), float(stop), float(target)
        risk = abs(e - s)
        if risk <= 0:
            return None
        return round(abs(t - e) / risk, 3)
    except (TypeError, ValueError):
        return None


# ── public: append one evaluated-setup record ──────────────────────────────── #

def journal_result(
    result: Any,
    symbol: str,
    direction: str,
    timestamp: Any,
    size_multiplier: float = 1.0,
) -> Optional[str]:
    """Append ONE causal-chain record for an evaluated setup and return its setup_id.

    Called once per pipeline.evaluate() — for entries AND rejects alike. Never raises;
    a failed write logs a warning and returns None so the trade decision is unaffected.
    """
    try:
        entry_level = getattr(result, "entry_level", None)
        setup_id = make_setup_id(symbol, direction, entry_level, timestamp)
        action, discard_reason = _classify_action(result)

        grade = getattr(result, "grade", None)
        grade = getattr(grade, "value", grade)

        regime_state = _read_regime_state()
        bias_state = _read_bias_state(timestamp)

        level = _extract_level(result)

        # R:R — from sizing (entry vs stop vs tp1) if an entry, else from any
        # entry/stop the veto carried (retrospective).
        sizing = getattr(result, "sizing", None)
        stop = getattr(sizing, "stop_loss", None) if sizing is not None else getattr(result, "stop", None)
        tp1 = getattr(sizing, "tp1", None) if sizing is not None else None
        rr = _compute_rr(entry_level, stop, tp1)

        comp = getattr(result, "component_scores", None) or {}
        comp = {k: _safe_float(v) for k, v in dict(comp).items()}

        record = {
            "setup_id": setup_id,
            "symbol": symbol,
            "timestamp": _iso(timestamp),
            "logged_at": _now_iso(),
            "level_id": level["level_id"],
            "level_type": level["level_type"],
            "level_tf": level["level_tf"],           # None until the caller tracks TF
            "level_quality_score": level["level_quality_score"],
            "regime_state": regime_state,
            "bias_state": bias_state,
            "bias_aligned": _bias_aligned(direction, bias_state),
            "size_multiplier": _safe_float(size_multiplier),
            "ict_grade": str(grade) if grade is not None else None,
            "component_scores": comp,
            "r_r_computed": rr,
            "action": action,
            "discard_reason": discard_reason,
            "direction": direction,
            "score": _safe_float(getattr(result, "score", None)),
            # Outcome fields — filled by update_outcome() on close.
            "outcome": None,
            "outcome_r": None,
            "exit_timestamp": None,
        }
        _append(record)
        return setup_id
    except Exception as exc:  # never block a trade
        logger.warning("causal_journal.journal_result failed (trade continues): %s", exc)
        return None


def _append(record: dict[str, Any]) -> None:
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# ── public: close the loop ──────────────────────────────────────────────────── #

def update_outcome(
    setup_id: str,
    outcome: str,
    outcome_r: Optional[float] = None,
    exit_timestamp: Optional[str] = None,
) -> bool:
    """Back-fill outcome + outcome_r on the matching OPEN, ENTERED record.

    CLAUDE.md NON-NEGOTIABLE #2: every ICT close MUST call this. Matches the most
    recent record whose setup_id equals `setup_id`, action == ENTERED, and outcome is
    still None. Rewrites the JSONL in place. Fails LOUD (warns + returns False) when
    nothing matches — a close that matches nothing means the loop is not closing.
    Never fabricates a record.
    """
    try:
        if not JOURNAL_PATH.exists():
            logger.warning("update_outcome: journal missing at %s (setup_id=%s)", JOURNAL_PATH, setup_id)
            return False

        rows: list[dict[str, Any]] = []
        for line in JOURNAL_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append({"_raw": line})

        # Match the LAST open ENTERED record with this setup_id (FIFO close of the
        # newest matching entry; there is at most one open per pair by protocol).
        target_idx = None
        for i in range(len(rows) - 1, -1, -1):
            r = rows[i]
            if (
                r.get("setup_id") == setup_id
                and r.get("action") == ENTERED
                and r.get("outcome") is None
            ):
                target_idx = i
                break

        if target_idx is None:
            logger.warning(
                "update_outcome: NO open ENTERED record for setup_id=%s (outcome=%s) — "
                "loop did not close; nothing fabricated",
                setup_id, outcome,
            )
            return False

        rows[target_idx]["outcome"] = outcome
        rows[target_idx]["outcome_r"] = _safe_float(outcome_r) if outcome_r is not None else None
        rows[target_idx]["exit_timestamp"] = exit_timestamp or _now_iso()

        tmp = JOURNAL_PATH.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
        tmp.replace(JOURNAL_PATH)
        return True
    except Exception as exc:
        logger.warning("update_outcome failed for setup_id=%s: %s", setup_id, exc)
        return False
