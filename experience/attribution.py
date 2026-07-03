"""Mechanical attribution classifier (W2) — implements ATTRIBUTION_RUBRIC.md v1 exactly.

The rubric is the law: this module refuses to run if the rubric file's hash isn't in
its pinned list (anti-gaming: you cannot change the law and keep classifying under the
old name). Missing inputs → AMBIGUOUS, never a guess.

Artifacts: data/experience/attributions_YYYY_MM.jsonl (idempotent on decision_id).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
RUBRIC = ROOT / "experience" / "ATTRIBUTION_RUBRIC.md"
OUT_DIR = ROOT / "data" / "experience"
PINNED_RUBRIC_HASHES = {
    "7e646084468f7e71",  # v1 (first 16 hex of sha256, committed 2026-07-02)
}
SLIPPAGE_OVERLAY_R = 0.15

Class = Literal["thesis_confirmed", "thesis_invalidated", "luck_good", "luck_bad", "AMBIGUOUS"]


def rubric_hash() -> str:
    return hashlib.sha256(RUBRIC.read_bytes()).hexdigest()[:16]


def assert_rubric_pinned() -> str:
    h = rubric_hash()
    if h not in PINNED_RUBRIC_HASHES:
        raise SystemExit(f"ATTRIBUTION_RUBRIC.md hash {h} is not pinned — the law changed; "
                         f"bump the rubric version AND this pin together (rubric §8).")
    return h


@dataclass(frozen=True)
class ClosedDecision:
    decision_id: str
    engine: Literal["carry", "exit_shadow", "predictive"]
    thesis_kind: Literal["structural_carry", "hypothesis"]
    hyp_id: str | None = None
    predicate_eval_at_exit: dict | None = None      # {name: bool} — hypothesis engines
    rate_diff_sign_entry: int | None = None         # carry proxy inputs
    rate_diff_sign_exit: int | None = None
    vix_gate_entry: str | None = None
    vix_gate_exit: str | None = None
    exit_reason: str = "UNKNOWN"                    # TIME|TRAILING|STOP|REVERSAL|CB_REFRESH|UNKNOWN
    realized_r: float | None = None
    fill_slippage_r: float | None = None


@dataclass(frozen=True)
class Attribution:
    decision_id: str
    cls: Class
    overlays: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    rationale: str = ""
    rubric_sha: str = ""


def _thesis_alive(d: ClosedDecision) -> bool | None:
    """True/False, or None when unevaluable (rubric §2)."""
    if d.thesis_kind == "hypothesis":
        if not d.predicate_eval_at_exit:
            return None
        return all(bool(v) for v in d.predicate_eval_at_exit.values())
    # structural carry proxy: rate-diff sign AND vix-gate state unchanged
    if None in (d.rate_diff_sign_entry, d.rate_diff_sign_exit, d.vix_gate_entry, d.vix_gate_exit):
        return None
    return (d.rate_diff_sign_entry == d.rate_diff_sign_exit
            and d.vix_gate_entry == d.vix_gate_exit)


def classify(d: ClosedDecision) -> Attribution:
    sha = assert_rubric_pinned()
    overlays: list[str] = []
    ev: dict = {"exit_reason": d.exit_reason, "realized_r": d.realized_r}

    if d.realized_r is None or d.exit_reason == "UNKNOWN":                    # §3.1
        return Attribution(d.decision_id, "AMBIGUOUS", overlays, ev,
                           "missing realized R or unknown exit reason", sha)
    if d.fill_slippage_r is not None and abs(d.fill_slippage_r) > SLIPPAGE_OVERLAY_R:  # §3.2
        overlays.append("execution_variance")
        ev["fill_slippage_r"] = d.fill_slippage_r
    if d.exit_reason == "CB_REFRESH":                                          # §3.6
        overlays.append("policy_exit")

    alive = _thesis_alive(d)                                                   # §3.3
    ev["thesis_alive"] = alive
    if alive is None:
        return Attribution(d.decision_id, "AMBIGUOUS", overlays, ev,
                           "thesis-alive test unevaluable (missing inputs)", sha)

    if d.realized_r > 0:                                                       # §3.4
        if alive:
            return Attribution(d.decision_id, "thesis_confirmed", overlays, ev,
                               "win with thesis intact", sha)
        return Attribution(d.decision_id, "luck_good", overlays, ev,
                           "won while the thesis was dead — flagged, not celebrated", sha)
    if not alive:                                                              # §3.5
        return Attribution(d.decision_id, "thesis_invalidated", overlays, ev,
                           "loss with thesis dead at/before exit", sha)
    return Attribution(d.decision_id, "luck_bad", overlays, ev,
                       "loss with thesis intact — variance around a live thesis", sha)


def write_attributions(atts: list[Attribution], month: str) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"attributions_{month.replace('-', '_')}.jsonl"
    existing = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                existing.add(json.loads(line).get("decision_id"))
    n = 0
    with path.open("a") as fh:
        for a in atts:
            if a.decision_id in existing:
                continue
            row = asdict(a)
            row["ts"] = datetime.now(timezone.utc).isoformat()
            fh.write(json.dumps(row, default=str) + "\n")
            existing.add(a.decision_id)
            n += 1
    return n


def read_attributions() -> list[dict]:
    rows = []
    for f in sorted(OUT_DIR.glob("attributions_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows
