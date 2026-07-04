"""experience/citations.py — L3: citation records for precedents surfaced in a weekly review.

A citation is a falsifiable prediction, not decoration: "this week resembles {precedent}, so
expect {analogy_prediction} within {outcome_days}d" is a claim with a due date
(scoring_due). A future job can grade it against what actually happened — that closes the
loop the same way experience/attribution.py closes it for individual trades, but one level up
(regime analogy instead of a single decision).

Storage: data/experience/citations.jsonl (append-only, tracked). Idempotent by citation_id.
Mirrors experience/journal.py's JOURNAL_DIR / experience/library_annex.py's ANNEX_PATH
monkeypatch convention via the module-level CITATIONS_PATH constant.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITATIONS_PATH = ROOT / "data" / "experience" / "citations.jsonl"

METHOD = "structured_v1"       # stamped on every citation — see experience/precedents.py docstring


def _as_date(d) -> date:
    return d if isinstance(d, date) else date.fromisoformat(str(d)[:10])


def make_citation(week: dict, precedent: dict, decision_ids: list[str], basis: dict | None = None) -> dict:
    """Build one citation record.

    week: {"tag": "2026-W27", "end": date|"YYYY-MM-DD", "review_path": str} — as available
        inside experience.weekly_review.build_review (week_bounds()'s tag/end, and the
        review markdown path being written).
    precedent: one dict from experience.precedents.find_precedents.
    decision_ids: journal decision_ids the citation is attached to (the week's acted rows).
    basis: optional extra similarity_basis/context fields — board_extremes, pairs,
        analogy_prediction, rubric_sha. Never required; every key has a safe default.
    """
    basis = basis or {}
    end = _as_date(week["end"])
    outcome_days = int(precedent.get("outcome_days") or 0)
    scoring_due = end + timedelta(days=min(outcome_days, 90))

    return {
        "citation_id": f"cite:{week['tag']}:{precedent['entry_id']}",
        "week": week["tag"],
        "review_path": str(week.get("review_path", "")),
        "entry_id": precedent["entry_id"],
        "entry_source": precedent["source"],
        "label": precedent["label"],
        "event_date": precedent["event_date"],
        "why": precedent["why"],
        "what_followed": precedent["what_followed"],
        "outcome_days": outcome_days,
        "severity": precedent["severity"],
        "similarity_basis": {
            "method": METHOD,
            "matched_tags": precedent.get("matched_tags", []),
            "board_extremes": basis.get("board_extremes", []),
            "score": precedent.get("score", 0),
        },
        "decision_ids": list(decision_ids or []),
        "pairs": basis.get("pairs", []),
        "analogy_prediction": basis.get(
            "analogy_prediction",
            f"If history rhymes with {precedent['label']} ({precedent['event_date']}), expect "
            f"dynamics similar to \"{precedent['what_followed']}\" to play out over "
            f"~{outcome_days}d.",
        ),
        "scoring_due": str(scoring_due),
        "scored": None,
        "rubric_sha": basis.get("rubric_sha", ""),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def read_citations() -> list[dict]:
    if not CITATIONS_PATH.exists():
        return []
    return [json.loads(l) for l in CITATIONS_PATH.read_text().splitlines() if l.strip()]


def append_citations(citations: list[dict]) -> int:
    """Idempotent append: citations whose citation_id already exists on disk are skipped."""
    if not citations:
        return 0
    CITATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {c.get("citation_id") for c in read_citations()}
    written = 0
    with CITATIONS_PATH.open("a") as fh:
        for c in citations:
            if c["citation_id"] in existing:
                continue
            fh.write(json.dumps(c, default=str) + "\n")
            existing.add(c["citation_id"])
            written += 1
    return written
