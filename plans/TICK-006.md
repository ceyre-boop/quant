# Plan — TICK-006: Review forensics feeds

Approach: six independent, guarded, additive reads in `experience/weekly_review.py`,
each contributing one section/line to the review body. Every feed wrapped in its own
try/except (a missing/corrupt source degrades to an explicit "feed unavailable: X"
line, never an exception). Sequenced AFTER TICK-005 merges (same file).

## Feeds (source → review output)
1. `data/oracle/reflections/{recent}.json` → "System health (Oracle)" line from
   `system_health_note` — MARKED `[source quarantined: RED-1 open — context only]`
   until TICK-011 lands. Never enters proposals.
2. `data/agent/hypothesis_ledger.json` → "This week's research" — Counter over
   entries/annotations dated in-window (INTERIM SEAL / BLOCKED / verdict counts).
   Closes the propose-but-never-hear loop; also dedup input: never re-propose an id
   with a terminal verdict.
3. `data/ledger/veto_ledger_{YYYY_MM}.jsonl` (+ ict variant) → acted:abstained:vetoed
   ratio + top veto stages.
4. `audit/reports/{date}.json` (latest ≤ week end) → one parity line: L1 determinism,
   L2 match rate, continuity violations.
5. `data/oracle/lesson_velocity.json` → current lesson + formation stage (one line).
6. `data/oracle/market_briefings/latest.json` → FRED macro_economic summary block
   ONLY (regime_read is ES/NQ-specific — discarded; provenance.verified=false noted).

## Files
- EDIT `experience/weekly_review.py` — one new function per feed
  (`_feed_oracle_health(week)` etc.), one "## Forensics" assembly block; all additive.
- NEW `tests/test_weekly_review_forensics.py` — per-feed: present/absent/corrupt ×
  review-still-generates; fixture files under tmp_path; monkeypatch path constants
  (add module-level constants for each source path, mirroring JOURNAL_DIR pattern).

## Risks
- Same-file conflict with TICK-005's Precedents section → build strictly after merge.
- Oracle text leaking into machine pathways → feeds render as REVIEW PROSE only;
  the proposals writer is untouched except the dedup check (2).

## Verification
`python3 -m pytest tests/test_weekly_review_forensics.py tests/test_experience.py -q`
green; W27 dry-run shows all six lines (or explicit unavailable-markers); full suite
baseline holds (1038 passed / 40 known-failed / 1+ skipped post-oanda-gate).
