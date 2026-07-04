# Plan — TICK-004: Library ascension slice (L1 + L2a + L2b-stub + L3)

Design verified against code 2026-07-03 (Plan-agent pass; approved in
`Plans/context-day-2-imperative-stonebraker.md` Appendix). ADDITIVE ONLY.

## Binding facts
- `models/alexandrian_library.json` is LIVE-read by `sovereign/orchestrator.py:122`
  (feeds live `size_modifier`). It is NEVER written by this work — byte-equality test
  enforces. Lived entries go to the sibling annex `data/experience/library_annex.jsonl`
  (`VOLUME_XI_LIVED` namespace).
- No honest board→23-dim feature mapping exists (features come from raw price arrays
  via `market_memory.extract_features`, 200+ SPY bars). v1 retrieval = structured
  (tags/severity/date/mechanism), `method:"structured_v1"` recorded in every citation.
  Real feature path = LIB-FEAT-1 (separate ticket, deferred: network dep in launchd).
- `experience/weekly_review.py` imported only by tests → additive edits safe.
- Do not touch any importer of `sovereign.risk.alexandrian_library`:
  `sovereign/orchestrator.py:122` (LIVE), `sovereign/intelligence/cross_system_bridge.py:463`,
  `ict/library_bridge.py:48`, `scripts/build_alexandrian_library.py`,
  `scripts/build_market_memory.py`, `scripts/rq_009_library_ablation.py`.

## Files (new unless EDIT)
- `experience/library_annex.py` — `LivedEntry` dataclass + append/read of the annex
  jsonl; dedupe by entry_id; `ANNEX_PATH` module constant (monkeypatch pattern, like
  `journal.JOURNAL_DIR`).
- `experience/precedents.py` — structured retrieval over `ALL_ENTRIES` + annex;
  board-extremes→mechanism-keyword mapping as ONE explicit reviewable dict; board via
  `sovereign.sentiment.store.connect(read_only=True)`; degrades to tag-only if DB
  locked/absent.
- `experience/citations.py` — L3 citation records → `data/experience/citations.jsonl`;
  dedupe by citation_id.
- `experience/library_ingest.py` — L1 converters (review-md / attribution-row /
  ledger-INTERIM-SEAL → LivedEntry) + idempotent `--backfill` CLI
  (`python -m experience.library_ingest --backfill [--dry-run]`).
- `experience/precedent_service.py` — L2b stub: `query(board_state_row, top_k=3) -> []`
  unless `experience.precedents.decision_time_enabled` is true; inert (nothing imports it).
- EDIT `experience/weekly_review.py` — guarded "## Precedents (Alexandrian Library)"
  section + citation writes inside `build_review`; ENTIRE block in one try/except so
  the review never dies on library errors; dry_run writes no citations.
- EDIT `config/parameters.yml` — additive:
  `experience: {precedents: {review_enabled: true, decision_time_enabled: false, top_k: 3}}`
- Tests (new files): `tests/test_library_annex.py`, `tests/test_precedents.py`,
  `tests/test_library_ingest.py`, `tests/test_citations.py`,
  `tests/test_weekly_review_precedents.py` — mirror `tests/test_experience.py`
  monkeypatch conventions.

## Signatures
`append_entries(list[LivedEntry])->int` · `read_entries()->list[LivedEntry]` ·
`week_board_extremes(start,end)->list[dict]` · `week_context_tags(rows,atts,extremes)->set[str]`
· `find_precedents(context_tags, top_k=3)->list[dict]` returning {entry_id, source
(canonical|annex), label, event_date, why, what_followed, outcome_days, severity,
matched_tags, score} · `make_citation(week, precedent, decision_ids, basis)->dict` ·
`append_citations(list)->int` · `entries_from_review(md_path)` ·
`entries_from_attributions(atts, journal_by_id)` · `entries_from_ledger_seals(path)`
(annotation startswith "INTERIM SEAL") · `backfill()->dict` (counts per source_kind).

LivedEntry fields: entry_id, volume, label, date, description, outcome, outcome_days,
severity, tags, source_kind, source_ref, added_at. entry_id namespaces:
`annex:review:{tag}` / `annex:attr:{decision_id}` / `annex:seal:{hyp_id}`.

Citation record: citation_id `cite:{week}:{entry_id}`, week, review_path, entry_id,
entry_source, label, event_date, why, what_followed, outcome_days, severity,
similarity_basis {method:"structured_v1", matched_tags, board_extremes, score},
decision_ids, pairs, analogy_prediction, scoring_due (week_end + min(outcome_days,90)),
scored:null, rubric_sha, ts.

## Tests must include
Idempotent re-append no-op · annex+canonical retrieved together · DB-absent
degradation · `models/alexandrian_library.json` BYTE-EQUALITY before/after ingest ·
flag-off → no section + no citations · precedents module raising → review still
writes md/ledger · dry_run → no citation file.

## Build order (single builder session)
annex → precedents → citations → ingest+CLI → parameters.yml + weekly_review section
→ (operator step, NOT the builder: `python -m experience.library_ingest --backfill`)
→ full suite: baseline 1039 passed / 40 known-failed + new tests green.

## Acceptance
All new tests green; suite baseline holds; canonical json byte-identical; review
dry-run over W27 produces a Precedents section with ≥1 citation; citations.jsonl rows
carry scoring_due; nothing imports precedent_service.
