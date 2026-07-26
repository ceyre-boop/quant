---
name: session-close
description: Run the Alta end-of-session closing protocol (git hygiene, safety checks, gate/shadow verification, NEXT.md handoff). Use when the user says close the session, session close, end of session, wrap up, run the closing protocol, or clocking out.
---

# Session Close

Source of truth: **`SESSION_CLOSING_PROTOCOL.md`** at repo root. Do not duplicate its
commands here — read it fresh each run, since it gets reconciled against the repo over time.

## What to do

1. Read `SESSION_CLOSING_PROTOCOL.md` top to bottom.
2. Execute every level in order — J (Janitor) → T (Technician) → A (Analyst) → M (Manager) → C (CEO).
   For each item: run its command (or perform its check), then judge PASS/FAIL against
   that item's stated PASS definition — not against your own intuition of what "looks fine."
3. On FAIL:
   - **Levels 1–2 (J, T):** the session is not done. Take the FAIL action in the protocol
     (fix it, or if it's a hard safety stop like J-6/M-2/M-3, halt and surface it —
     do not proceed to close).
   - **Levels 3–5 (A, M, C):** ticket it and note it in NEXT.md; don't block the close on these.
   - A-5 (Obsidian sync) may be validly SKIPPED if the session was purely operational —
     say so explicitly rather than silently omitting it.
4. Write the Level 3/5 narrative items directly into `NEXT.md` (A-1, C-1, C-2, C-3) —
   these are prose, not commands; there's nothing to execute, only to write honestly.
5. When every item is GREEN or explicitly logged as skipped with a reason, append the
   **CLOSING STAMP** block (format at the bottom of `SESSION_CLOSING_PROTOCOL.md`) to
   `NEXT.md`, with the real tallies from this run — never copy the example numbers verbatim.
6. Report back to the user: a compact per-level PASS/FAIL/SKIP summary, anything that
   blocked the close, and the final stamp you appended.

## Notes

- Several protocol commands were reconciled against this specific repo on 2026-07-25
  (see inline "Verified/Reconciled" notes in the protocol file itself, e.g. T-3's PASS
  count and M-3's `trade_mode` value). If a command still fails to run for reasons other
  than an actual FAIL condition (missing file, import error), that's a protocol bug —
  fix the protocol file, note the correction inline the way existing ones are noted, and
  proceed; don't let a broken command block an otherwise-clean close.
- Never touch frozen execution-path files while running this (`forex_exit_manager`,
  `decide_exit`, `exit_machine`) — this skill only reads/verifies (M-1), it never edits them.
- Git hygiene items (J-1/J-2/J-3) may hit heavy concurrent dirty state from other sessions/
  worktrees — never `git add -A`. Stage only files you yourself changed this session.
