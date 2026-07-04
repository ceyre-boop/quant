# R2 — Library + vault retargeting audit (2026-07-03, read-only)

## MAJOR CORRECTION to the morning picture
The Alexandrian Library is NOT unconsumed — it is LIVE in the ICT path:
- `ict/orchestrator.py:277` calls `query_library()` EVERY scan via `ict/library_bridge.py`
  (lazy function-level import, line 48) — threat_score/regime/size_modifier gate ICT
  entries and sizing TODAY.
- `sovereign/orchestrator.py:2093` calls `learn()` on crash detection → `_save()` →
  **models/alexandrian_library.json is live-WRITTEN, event-driven**, not just live-read.
  (Reinforces TICK-004's annex rule: our ingest never touches the canonical json.)
- Why the isolation law tolerates it: the import is inside the function body; the AST
  check walks module-level imports only. Verdict: SANCTIONED-BRIDGE (report-only;
  the NN#1 bridge-story wording is already on Colin's §S2 list).
The true gap: the Library serves live ICT gating but is ABSENT from the memory/review
loop — exactly what TICK-004 wires (unchanged).

## Verdicts
- [ict/library_bridge.py] → feeds ICT entry gating + sizing TODAY (live, hot path) ·
  could feed capital-rationing/threat visibility · gap: threat progression surfaced
  nowhere · cheapest: include threat_score in the existing dashboard export (NOT
  Firebase — that page is dead per 06-30 audit; scout's firebase.put suggestion
  rejected) · LEAVE the bridge itself; display-wire is a parity ticket.
- [cross_system_bridge.py:463] → callers are archive/agent_scheduler + ict-engine/
  orchestrator — neither scheduled/live · ATTIC-CANDIDATE (batches with the §S2
  equity-engine/ict-engine ruling).
- [vault-system-graph ~526 notes] → read by humans only; last regen Jul 3 00:38; no
  post-learn rebuild hook · could answer module-risk/dependency audit queries · gap:
  no schedule, no code consumer · cheapest: post-learn or nightly re-gen hook (3
  lines) · RETARGET (low rank — human-serving).
- [00-BRAIN] → active human memory organ (NEXT/DECISIONS current) · LEAVE.
- [.smart-env RAG ~671 embeddings] → zero consumers, plugin-managed · could serve
  semantic search over research notes · ATTIC-CANDIDATE unless a consumer is ever
  specced (not v1 anything).
- [alexandrian auto_learn pathway] → LIVE (event-driven crash learning) · gap: no
  downstream notification (vault regen, review awareness of new entries) · TICK-004's
  annex+review wiring gives the review side; vault regen is the 3-line hook above ·
  LEAVE core, RETARGET notifications.

## Headlines
1. The Library already fights in one theater (ICT live gating) — the rewiring is
   giving it the SECOND theater (memory/review loop), which is TICK-004, unchanged.
2. models/alexandrian_library.json is live-written by learn() — treat as an execution-
   path data file forever (freeze-listed today).
3. Vault graph + RAG embeddings are human-side or orphaned — cheap regen hook worth
   a low-rank ticket; embeddings stay attic-candidate until a real consumer exists.
