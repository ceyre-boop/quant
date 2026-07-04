# Plan — TICK-008: Health resurrection (diagnose-first)

## Phase A — diagnosis (read-only, one pass)
Per stale job (health.responder Jun-14, futures.bias Jun-14, hypothesis.generator
Jun-15, research.factory Jun-15, oracle.reflect Jun-28, stray_tripwire WatchPaths
Jun-7): launchctl print state, last err-log tail, plist StartInterval vs log reality,
manual dry invocation where side-effect-free. Name the root cause per job in
`audit/health_diagnosis_2026-07.md` BEFORE any fix. (Context: generator /
session_close / cache.refresh were file-repaired from .corrupt-20260701 on Jul 1 —
repair restored code, evidently not schedules/inputs.)

## Phase B — staleness deadlines (additive)
- NEW `sovereign/health/staleness.py`: table of {job → artifact path → max age}
  (deadlines from each job's own schedule ×2); `check() -> [{job, artifact, age,
  deadline, status}]`.
- Surface: morning-brief section + `data/agent/health.json` merge (existing
  dashboard health panel reads it) — additive keys only.
- The invariant-guard job (TICK-004, other session) and plist_watchdog integrate
  here later; do not duplicate their checks.

## Phase C — plist log-redirects (Colin-reviewed batch)
Prepare unified diff for bench/evening_prep/render_keepalive plists (StandardOutPath/
StandardErrorPath → ~/quant/logs/) + the reload commands. DELIVERED AS A DIFF —
applied by Colin (live-organ config; same class as TICK-013).

## Verification
Phase A doc names ≥1 concrete cause per job (not "restarted, seems fine").
Phase B: `python3 -m sovereign.health.staleness` lists all 20+ jobs with correct
ages; unit test with fixture mtimes; suite baseline holds.
