#!/usr/bin/env python3
"""experience/library_ingest.py — L1: convert lived history into annex entries.

Three read-only source kinds, three converters, one idempotent sink
(experience.library_annex.append_entries). NEVER imports sovereign.risk.alexandrian_library's
writer path and NEVER writes models/alexandrian_library.json — that file is LIVE-read by
sovereign/orchestrator.py:122 and this ticket's byte-equality test enforces it stays untouched
before/after any ingest run.

entry_id namespaces (must match experience/library_annex.py's docstring):
  annex:review:{tag}        — one entry per weekly-review file WITH non-trivial surprises
  annex:attr:{decision_id}  — one entry per closed, attributed decision
  annex:seal:{hyp_id}       — one entry per ledger annotation whose note startswith
                              "INTERIM SEAL" (see scripts/research/run_positioning_family*.py)

CLI: `python -m experience.library_ingest --backfill [--dry-run]` — an OPERATOR step, run
from the main checkout after this ticket merges. This module's own code never invokes itself;
tests drive the converters and backfill() directly against fixtures.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience import attribution as att          # noqa: E402
from experience import journal                      # noqa: E402
from experience.library_annex import LivedEntry, VOLUME_XI_LIVED, append_entries  # noqa: E402

# Monkeypatchable like every other path constant in this package (journal.JOURNAL_DIR,
# library_annex.ANNEX_PATH, citations.CITATIONS_PATH) — tests never touch the real paths.
REVIEW_DIR = ROOT / "review"
LEDGER_PATH = ROOT / "data" / "agent" / "hypothesis_ledger.json"

_SURPRISES_RE = re.compile(r"## Surprises\n(.*?)(?:\n## |\Z)", re.DOTALL)
_HEADER_RE = re.compile(r"\((\d{4}-\d{2}-\d{2}) .* (\d{4}-\d{2}-\d{2})\)")
_CLASSES_RE = re.compile(r"Classes:\s*(\{.*\})")

# Attribution class -> annex severity (canonical scale: 0=benign,1=moderate,2=severe,-1=positive).
# ONE explicit, reviewable dict — see experience/precedents.py's BOARD_EXTREME_TAGS for the
# same "put the judgment call in a named dict, not scattered logic" convention.
_ATTRIBUTION_SEVERITY = {
    "thesis_confirmed":   -1,   # win, thesis intact — the system working as designed
    "luck_good":           1,   # win despite a dead thesis — flagged, not celebrated
    "thesis_invalidated":  0,   # loss, thesis properly invalidated — benign/expected
    "luck_bad":            1,   # loss despite a live thesis — normal variance, worth remembering
    "AMBIGUOUS":           0,
}


def entries_from_review(md_path: Path) -> list[LivedEntry]:
    """One LivedEntry per weekly-review file, IF it recorded a non-trivial surprise.

    A review whose Surprises section is empty/"none" carries no signal worth citing later —
    skipped (returns []) rather than padding the annex with content-free rows. Never raises:
    a malformed/unreadable review file yields [] (ingest must survive a hand-edited review).
    """
    md_path = Path(md_path)
    try:
        text = md_path.read_text()
    except OSError:
        return []

    tag = md_path.stem                                              # "2026-W27"
    m = _HEADER_RE.search(text.splitlines()[0] if text else "")
    end = m.group(2) if m else tag

    sm = _SURPRISES_RE.search(text)
    bullets = [ln[2:].strip() for ln in (sm.group(1).splitlines() if sm else "") if ln.startswith("- ")]
    bullets = [b for b in bullets if b and b.lower() != "none"]
    if not bullets:
        return []

    cm = _CLASSES_RE.search(text)
    tags = {"weekly_review"}
    if cm:
        tags |= {w.lower() for w in re.findall(r"'(\w+)'", cm.group(1))}

    return [LivedEntry(
        entry_id=f"annex:review:{tag}",
        volume=VOLUME_XI_LIVED,
        label=f"WEEKLY_REVIEW_{tag}",
        date=end,
        description=f"Weekly self-review {tag}: " + "; ".join(bullets[:3]),
        outcome="; ".join(bullets),
        outcome_days=7,
        severity=0,
        tags=sorted(tags),
        source_kind="review",
        source_ref=str(md_path),
    )]


def entries_from_attributions(atts: list[dict], journal_by_id: dict[str, dict]) -> list[LivedEntry]:
    """One LivedEntry per attribution row (1:1) — every closed, classified decision is a
    lived data point, AMBIGUOUS included (the annex is history, not a highlight reel).
    Missing journal context degrades to safe defaults rather than skipping the row.
    """
    out = []
    for a in atts:
        did = a.get("decision_id", "")
        j = journal_by_id.get(did, {})
        cls = a.get("cls", "AMBIGUOUS")
        out.append(LivedEntry(
            entry_id=f"annex:attr:{did}",
            volume=VOLUME_XI_LIVED,
            label=f"{cls}_{j.get('pair', 'UNKNOWN')}",
            date=str(j.get("decision_ts", a.get("ts", "")))[:10],
            description=a.get("rationale", ""),
            outcome=f"{cls}: realized_r={((a.get('evidence') or {}).get('realized_r'))}",
            outcome_days=0,
            severity=_ATTRIBUTION_SEVERITY.get(cls, 0),
            tags=sorted({cls, j.get("engine", ""), *(a.get("overlays") or [])} - {""}),
            source_kind="attribution",
            source_ref=did,
        ))
    return out


def entries_from_ledger_seals(path: Path) -> list[LivedEntry]:
    """One LivedEntry per ledger annotation whose note startswith "INTERIM SEAL" — these are
    pre-registered-but-unresolved hypotheses (family BH pending); worth citing as "here's a
    live question, not yet a verdict," distinct from a CONFIRMED/REJECTED_OOS result.
    Never raises: missing/malformed ledger yields [].
    """
    path = Path(path)
    try:
        entries = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    out = []
    for entry in entries:
        hyp_id = entry.get("id", "")
        for note in entry.get("annotations", []):
            text = note.get("note", "")
            if not text.startswith("INTERIM SEAL"):
                continue
            out.append(LivedEntry(
                entry_id=f"annex:seal:{hyp_id}",
                volume=VOLUME_XI_LIVED,
                label=entry.get("name", hyp_id),
                date=str(note.get("date", ""))[:10],
                description=entry.get("mechanism") or entry.get("methodology_note", ""),
                outcome=text[:300],
                outcome_days=0,
                severity=0,                       # no verdict yet — neither positive nor negative
                tags=sorted({t for t in [entry.get("family", ""), entry.get("prior_expectation", "")] if t}),
                source_kind="ledger_seal",
                source_ref=hyp_id,
            ))
            break  # one entry per hyp_id even if multiple INTERIM SEAL annotations accrue
    return out


def backfill(dry_run: bool = False) -> dict:
    """Run all three converters over the current REVIEW_DIR/journal+attributions/LEDGER_PATH
    and append the union to the annex (idempotent — safe to re-run).

    Returns counts per source_kind plus "appended" (rows actually new to the annex). dry_run
    computes everything but skips the append_entries call, so nothing is written.
    """
    review_entries = []
    if REVIEW_DIR.exists():
        for p in sorted(REVIEW_DIR.glob("*.md")):
            review_entries.extend(entries_from_review(p))

    atts = att.read_attributions()
    journal_by_id = {r["decision_id"]: r for r in journal.read_all()}
    attribution_entries = entries_from_attributions(atts, journal_by_id)

    seal_entries = entries_from_ledger_seals(LEDGER_PATH) if LEDGER_PATH.exists() else []

    all_entries = review_entries + attribution_entries + seal_entries
    appended = 0 if dry_run else append_entries(all_entries)

    return {
        "review": len(review_entries),
        "attribution": len(attribution_entries),
        "ledger_seal": len(seal_entries),
        "candidates": len(all_entries),
        "appended": appended,
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.backfill:
        result = backfill(dry_run=a.dry_run)
        print(f"[library_ingest] backfill: {result}")
    else:
        ap.print_help()
