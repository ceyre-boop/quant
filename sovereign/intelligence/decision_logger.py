"""
Decision Logger — captures the full reasoning context at the moment of entry.

Every approved trade (ICT or Forex) writes one record here.  Six months later,
Oracle can read the reasoning behind each trade — not just the outcome —
and identify structural causes even at small sample sizes.

Log file: data/decision_logs/decisions_YYYY_MM.jsonl  (one file per month)
Schema: see DecisionRecord dataclass below.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sovereign.utils.timestamps import canonical_timestamp, normalize_timestamp, timestamps_match

LOG_DIR = Path("data/decision_logs")

log = logging.getLogger(__name__)


def _norm_pair(p: str) -> str:
    """Normalize a pair to a venue-agnostic key for outcome matching.

    Forex decisions log the yfinance ticker ('GBPUSD=X'); OANDA fills use
    'GBP_USD'; ICT logs plain 'GBPUSD'. All three must compare equal or the
    win/loss loop silently never closes (CLAUDE.md NON-NEGOTIABLE #2).
    """
    return (p or "").upper().replace("=X", "").replace("_", "").replace("/", "").replace("-", "")


def _now_iso() -> str:
    return canonical_timestamp()


def _log_path() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    month = datetime.now(timezone.utc).strftime("%Y_%m")
    return LOG_DIR / f"decisions_{month}.jsonl"


# ─── Schema ──────────────────────────────────────────────────────────────────

@dataclass
class DecisionRecord:
    # Identity
    entry_timestamp:        str
    system:                 str           # ICT | FOREX
    pair:                   str

    # Direction & levels
    direction:              str           # LONG | SHORT
    entry_level:            Optional[float]
    stop_loss:              Optional[float]
    tp1:                    Optional[float]
    tp2:                    Optional[float]

    # Signal context
    signal_layers_active:   list[str]     # e.g. ["SWEEP", "FVG", "KILL_ZONE"]
    grade:                  Optional[str]  # A+ | A | B | C (ICT) or None (Forex)
    session:                Optional[str]  # LONDON | NY_AM | NY_PM
    score:                  Optional[float]

    # Market context at entry
    vix_at_entry:           Optional[float]
    rate_differential_zscore: Optional[float]
    cot_percentile:         Optional[float]
    library_match:          Optional[str]  # "PATTERN at 0.XX" or None
    commitment_score:       Optional[float]
    bars_since_signal:      Optional[int]
    adr_pct_used:           Optional[float]  # how much of ADR was consumed at entry

    # Sizing rationale
    risk_pct:               Optional[float]
    risk_dollars:           Optional[float]
    why_this_size:          str            # human-readable sizing chain

    # Trade thesis
    why_this_trade:         str            # human-readable entry reasoning

    # Component scores for later analysis
    component_scores:       dict[str, float] = field(default_factory=dict)
    confirmations:          list[str]        = field(default_factory=list)
    missing:                list[str]        = field(default_factory=list)

    # Entry-time context (Loop 2 — lets Oracle reason over WHY, not just THAT)
    present_state_snapshot: dict[str, Any]   = field(default_factory=dict)
    active_lessons:         list[str]        = field(default_factory=list)

    # Outcome (filled in later by forensic engine on close)
    outcome:                Optional[str]    = None   # WIN | LOSS | OPEN
    r_realized:             Optional[float]  = None
    exit_timestamp:         Optional[str]    = None


import logging as _logging
_dlog = _logging.getLogger("decision_logger")


def _append(record: DecisionRecord) -> None:
    path = _log_path()
    with open(path, "a") as f:
        f.write(json.dumps(asdict(record), default=str) + "\n")


def _safe_append(record: DecisionRecord) -> None:
    """Append record; log warning on failure — never raise. Trades must not be blocked."""
    try:
        _append(record)
    except Exception as exc:
        _dlog.warning("Decision log write failed (trade continues): %s", exc)


# ─── ICT builder ─────────────────────────────────────────────────────────────

def log_ict_decision(
    signal: Any,                          # ICTSignal
    vix_at_entry: Optional[float] = None,
    library_match: Optional[str] = None,
    commitment_score: Optional[float] = None,
    bars_since_signal: Optional[int] = None,
    adr_pct_used: Optional[float] = None,
    cot_percentile: Optional[float] = None,
    rate_diff_z: Optional[float] = None,
    present_state_snapshot: Optional[dict] = None,
    active_lessons: Optional[list[str]] = None,
) -> DecisionRecord:
    """
    Build and persist a decision record from an approved ICTSignal.
    Call this immediately before returning/yielding the signal for execution.

    present_state_snapshot / active_lessons (Loop 2): entry-time context so the
    Oracle can reason over WHY a trade won/lost, and EdgeMonitor can attribute
    outcomes to lessons. ICT callers pass ICT-LOCAL context only (no sovereign
    import — isolation rule #1).
    """
    sz = signal.sizing

    # Active signal layers — only components that actually scored
    active_layers = [
        k.upper() for k, v in signal.component_scores.items() if v and v > 0
    ]

    # why_this_trade — synthesize from confirmations + session + grade
    confirmations = signal.confirmations or []
    session_str = ""
    if signal.session_status:
        sn = getattr(signal.session_status, "kill_zone_name", None) or \
             getattr(signal.session_status, "session_name", None)
        if sn:
            session_str = f"{sn} session"
    grade_str = signal.grade.value if hasattr(signal.grade, "value") else str(signal.grade)
    parts = [f"Grade {grade_str}"]
    if session_str:
        parts.append(session_str)
    parts.extend(confirmations[:4])  # top 4 confirmations
    if library_match:
        parts.append(f"Library: {library_match}")
    why_trade = " + ".join(parts)

    # why_this_size — expose the sizing chain
    risk_pct = getattr(sz, "risk_pct", None)
    risk_dollars = getattr(sz, "risk_dollars", None)
    base_risk = f"{risk_pct:.2%} account" if risk_pct else "unknown"
    size_parts = [f"Base risk {base_risk}"]
    if commitment_score is not None and commitment_score != 1.0:
        size_parts.append(f"commitment {commitment_score:.2f}×")
    why_size = " × ".join(size_parts) + f" = ${risk_dollars:.2f}" if risk_dollars else " × ".join(size_parts)

    # TP levels from sizing
    tp1 = getattr(sz, "tp1", None)
    tp2 = getattr(sz, "tp2", None)

    record = DecisionRecord(
        entry_timestamp=signal.timestamp.isoformat() if hasattr(signal.timestamp, "isoformat") else str(signal.timestamp),
        system="ICT",
        pair=signal.symbol,
        direction=signal.direction,
        entry_level=signal.entry_level,
        stop_loss=getattr(sz, "stop_loss", None),
        tp1=tp1,
        tp2=tp2,
        signal_layers_active=active_layers,
        grade=grade_str,
        session=session_str or None,
        score=signal.score,
        vix_at_entry=vix_at_entry,
        rate_differential_zscore=rate_diff_z,
        cot_percentile=cot_percentile,
        library_match=library_match,
        commitment_score=commitment_score,
        bars_since_signal=bars_since_signal,
        adr_pct_used=adr_pct_used,
        risk_pct=risk_pct,
        risk_dollars=risk_dollars,
        why_this_size=why_size,
        why_this_trade=why_trade,
        component_scores=dict(signal.component_scores),
        confirmations=list(signal.confirmations),
        missing=list(signal.missing),
        present_state_snapshot=present_state_snapshot or {},
        active_lessons=list(active_lessons or []),
    )

    _safe_append(record)
    return record


# ─── Forex builder ────────────────────────────────────────────────────────────

def log_forex_decision(
    pair: str,
    direction: str,
    entry_level: float,
    stop_loss: float,
    hold_days: int,
    risk_pct: float,
    signal_layers: list[str],
    rate_diff_z: Optional[float] = None,
    vix_at_entry: Optional[float] = None,
    cot_percentile: Optional[float] = None,
    library_match: Optional[str] = None,
    commitment_score: Optional[float] = None,
    freshness_mult: Optional[float] = None,
    kelly_fraction: Optional[float] = None,
    size_mult: Optional[float] = None,
    bars_since_signal: Optional[int] = None,
    extra: Optional[dict] = None,
    present_state_snapshot: Optional[dict] = None,
    active_lessons: Optional[list[str]] = None,
) -> DecisionRecord:
    """
    Build and persist a decision record for an approved forex macro signal.
    Call from the live scan or paper trading execution path.

    present_state_snapshot / active_lessons (Loop 2): entry-time context. The
    sovereign/forex path MAY pass the full PresentState snapshot here.
    """
    # why_this_trade
    parts = []
    if rate_diff_z is not None:
        sign = "+" if rate_diff_z > 0 else ""
        parts.append(f"rate diff z={sign}{rate_diff_z:.2f}")
    parts.extend(signal_layers[:4])
    if library_match:
        parts.append(f"Library: {library_match}")
    if vix_at_entry is not None:
        parts.append(f"VIX {vix_at_entry:.1f}")
    why_trade = " + ".join(parts) if parts else "macro signal"

    # why_this_size — expose the multiplier chain
    size_parts = []
    if kelly_fraction is not None:
        size_parts.append(f"Kelly {kelly_fraction:.2f}×")
    if commitment_score is not None:
        size_parts.append(f"commitment {commitment_score:.2f}×")
    if freshness_mult is not None:
        size_parts.append(f"freshness {freshness_mult:.2f}×")
    if size_mult is not None:
        size_parts.append(f"VIX-slope {size_mult:.2f}×")
    final_risk = f"= {risk_pct:.2%} risk"
    why_size = " × ".join(size_parts) + " " + final_risk if size_parts else final_risk

    record = DecisionRecord(
        entry_timestamp=_now_iso(),
        system="FOREX",
        pair=pair,
        direction=direction,
        entry_level=entry_level,
        stop_loss=stop_loss,
        tp1=None,
        tp2=None,
        signal_layers_active=signal_layers,
        grade=None,
        session=None,
        score=None,
        vix_at_entry=vix_at_entry,
        rate_differential_zscore=rate_diff_z,
        cot_percentile=cot_percentile,
        library_match=library_match,
        commitment_score=commitment_score,
        bars_since_signal=bars_since_signal,
        adr_pct_used=None,
        risk_pct=risk_pct,
        risk_dollars=None,
        why_this_size=why_size,
        why_this_trade=why_trade,
        component_scores=extra or {},
        confirmations=signal_layers,
        missing=[],
        present_state_snapshot=present_state_snapshot or {},
        active_lessons=list(active_lessons or []),
    )

    _safe_append(record)
    return record


# ─── Outcome updater (called by forensic engine on trade close) ───────────────

def _outcome_entry_match(stored_ts: str, incoming_ts: str) -> bool:
    """Match a closing trade's entry timestamp to a stored decision's entry timestamp.

    Tiered, because a FOREX decision is logged at SIGNAL time (`_now_iso()`), but the
    outcome backfill keys on the OANDA FILL time (the trade's `openTime`), which
    routinely lands in a LATER clock-hour than the signal (e.g. signal 12:43 → fill
    13:34). Strict date+hour matching (`timestamps_match`, `[:13]`) silently dropped
    those — the closed loop never closed for forex, breaking NON-NEGOTIABLE #2 while
    the unit tests (which reuse identical/same-hour timestamps) stayed green.

    Tier 1: existing strict behaviour (exact / date+hour / forensic date-only) — unchanged.
    Tier 2: same-UTC-date fallback. Safe because forex fires ≤~1 trade per pair per day
    and the caller already constrains the match to a same-pair record that is still OPEN
    (outcome is None); the scan picks the oldest open record, so concurrent same-day
    signals close FIFO. Does not loosen the ICT/forensic paths (they resolve in Tier 1).
    """
    if timestamps_match(stored_ts, incoming_ts):
        return True
    s = normalize_timestamp(str(stored_ts).strip())
    i = normalize_timestamp(str(incoming_ts).strip())
    return bool(s and i and s[:10] == i[:10])


def update_outcome(
    pair: str,
    entry_timestamp: str,
    outcome: str,
    r_realized: float,
    exit_timestamp: Optional[str] = None,
    system: Optional[str] = None,
) -> bool:
    """
    Back-fill outcome into the matching decision record.
    Rewrites the monthly log file — only call this on trade close, not frequently.
    Returns True if a matching record was found and updated.

    Matching strategy: exact timestamp first, then fuzzy date-prefix match
    (handles forensic engine passing truncated dates like "2026-05-26 03:45").
    Only updates records still OPEN (outcome is None).
    """
    # Derive month key from the timestamp so we open the right file
    from sovereign.utils.timestamps import normalize_timestamp
    normalized = normalize_timestamp(entry_timestamp)
    try:
        month = datetime.fromisoformat(normalized).strftime("%Y_%m")
    except Exception:
        month = datetime.now(timezone.utc).strftime("%Y_%m")

    log_path = LOG_DIR / f"decisions_{month}.jsonl"
    if not log_path.exists():
        return False

    records = []
    found = False
    skipped: list[str] = []  # same-pair OPEN records we could not match (for fail-loud logging)
    target_pair = _norm_pair(pair)
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if not found and _norm_pair(obj.get("pair")) == target_pair and obj.get("outcome") is None:
                if system and obj.get("system") != system:
                    skipped.append(f"system={obj.get('system')}!={system}")
                    records.append(obj)
                    continue
                stored_ts = obj.get("entry_timestamp", "")
                if _outcome_entry_match(stored_ts, entry_timestamp):
                    obj["outcome"] = outcome
                    obj["r_realized"] = r_realized
                    obj["exit_timestamp"] = exit_timestamp or _now_iso()
                    found = True
                else:
                    skipped.append(f"ts={stored_ts}!~{entry_timestamp}")
            records.append(obj)
        except Exception:
            records.append({"_raw": line})

    if found:
        log_path.write_text("\n".join(json.dumps(r, default=str) for r in records) + "\n")
    else:
        # Fail loud: a closed trade that matches nothing means the loop is not closing.
        log.warning(
            "update_outcome: NO match for %s @ %s (system=%s) — %s",
            pair, entry_timestamp, system,
            ("; ".join(skipped) if skipped else "no OPEN same-pair record in this month's log"),
        )

    return found
