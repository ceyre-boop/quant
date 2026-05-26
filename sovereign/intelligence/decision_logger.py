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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

LOG_DIR = Path("data/decision_logs")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    # Outcome (filled in later by forensic engine on close)
    outcome:                Optional[str]    = None   # WIN | LOSS | OPEN
    r_realized:             Optional[float]  = None
    exit_timestamp:         Optional[str]    = None


def _append(record: DecisionRecord) -> None:
    path = _log_path()
    with open(path, "a") as f:
        f.write(json.dumps(asdict(record), default=str) + "\n")


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
) -> DecisionRecord:
    """
    Build and persist a decision record from an approved ICTSignal.
    Call this immediately before returning/yielding the signal for execution.
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
    )

    _append(record)
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
) -> DecisionRecord:
    """
    Build and persist a decision record for an approved forex macro signal.
    Call from the live scan or paper trading execution path.
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
    )

    _append(record)
    return record


# ─── Outcome updater (called by forensic engine on trade close) ───────────────

def update_outcome(
    pair: str,
    entry_timestamp: str,
    outcome: str,
    r_realized: float,
    exit_timestamp: Optional[str] = None,
) -> bool:
    """
    Back-fill outcome into the matching decision record.
    Rewrites the monthly log file — only call this on trade close, not frequently.
    Returns True if a matching record was found and updated.
    """
    # Find the file that would contain this entry_timestamp
    try:
        ts = datetime.fromisoformat(entry_timestamp.replace("Z", "+00:00"))
        month = ts.strftime("%Y_%m")
    except Exception:
        month = datetime.now(timezone.utc).strftime("%Y_%m")

    log_path = LOG_DIR / f"decisions_{month}.jsonl"
    if not log_path.exists():
        return False

    records = []
    found = False
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if obj.get("pair") == pair and obj.get("entry_timestamp") == entry_timestamp:
                obj["outcome"] = outcome
                obj["r_realized"] = r_realized
                obj["exit_timestamp"] = exit_timestamp or _now_iso()
                found = True
            records.append(obj)
        except Exception:
            records.append({"_raw": line})

    if found:
        log_path.write_text("\n".join(json.dumps(r, default=str) for r in records) + "\n")

    return found
