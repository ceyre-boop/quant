"""experience/library_annex.py — L1: the sibling annex for LIVED entries (VOLUME_XI_LIVED).

The canonical Alexandrian Library (`sovereign/risk/alexandrian_library.py::ALL_ENTRIES`,
`models/alexandrian_library.json`) is a fixed historical registry, LIVE-read by
`sovereign/orchestrator.py:122` to feed the live `size_modifier` — this module NEVER imports
the writer path of that class and NEVER touches its JSON file. This repo's own lived
experience (weekly reviews, attributed decisions, sealed hypotheses) needs somewhere to land
without risking that live surface, so it appends here instead: same LibraryEntry-compatible
shape (entry_id/volume/label/date/description/outcome/outcome_days/severity/tags) plus
provenance (source_kind/source_ref/added_at), all implicitly filed under one new volume,
VOLUME_XI_LIVED.

Storage: data/experience/library_annex.jsonl (append-only, tracked). Idempotent by entry_id —
re-running any L1 converter (experience/library_ingest.py) never duplicates a row. Mirrors
experience/journal.py's JOURNAL_DIR monkeypatch convention via the module-level ANNEX_PATH
constant.

entry_id namespaces (assigned by the converters in library_ingest.py, not here):
  annex:review:{tag}        — from a weekly review markdown file
  annex:attr:{decision_id}  — from an attribution row
  annex:seal:{hyp_id}       — from a ledger "INTERIM SEAL" annotation
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNEX_PATH = ROOT / "data" / "experience" / "library_annex.jsonl"

VOLUME_XI_LIVED = "VOLUME_XI_LIVED"


@dataclass(frozen=True)
class LivedEntry:
    """One lived-experience entry — LibraryEntry-shaped, plus provenance.

    Field order/names deliberately mirror sovereign.risk.alexandrian_library.LibraryEntry
    (entry_id, volume, label, date, description, outcome, outcome_days, severity, tags) so
    experience/precedents.py can treat canonical and annex entries uniformly at retrieval
    time; source_kind/source_ref/added_at are the annex-only provenance fields.
    """
    entry_id:      str
    volume:        str          # almost always VOLUME_XI_LIVED
    label:         str
    date:          str          # representative date, YYYY-MM-DD
    description:   str
    outcome:       str
    outcome_days:  int
    severity:      int          # 0=benign, 1=moderate, 2=severe, -1=positive (canonical scale)
    tags:          list
    source_kind:   str          # review | attribution | ledger_seal
    source_ref:    str          # provenance pointer: review path, decision_id, or hyp_id
    added_at:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def read_entries() -> list[LivedEntry]:
    """All annex entries, in file order. [] if the annex doesn't exist yet."""
    if not ANNEX_PATH.exists():
        return []
    out = []
    for line in ANNEX_PATH.read_text().splitlines():
        if line.strip():
            out.append(LivedEntry(**json.loads(line)))
    return out


def append_entries(entries: list[LivedEntry]) -> int:
    """Idempotent append: entries whose entry_id already exists on disk are skipped.

    Returns the number of NEW rows written (0 on a pure re-run — the ingest CLI relies on
    this to be safe to re-invoke without duplicating history).
    """
    if not entries:
        return 0
    ANNEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {e.entry_id for e in read_entries()}
    written = 0
    with ANNEX_PATH.open("a") as fh:
        for e in entries:
            if e.entry_id in existing:
                continue
            fh.write(json.dumps(asdict(e), default=str) + "\n")
            existing.add(e.entry_id)
            written += 1
    return written
