# Claim Verification Spec — the falsification contract for audit findings

A Layer-4 companion to `audit/invariants_spec.md`. Where the invariant spec asserts that the
*system* is not silently succeeding on bad inputs, this spec asserts that a *claim about the
system* has been mechanically tested before anyone acts on it.

Read by `audit/claim_check.py` (read-only; imports nothing from the execution path, not even
for vocabulary). The machine-readable fence below is the source of every rule — it is
sha256-hashed and version-checked at load, exactly like the invariant and divergence specs.

## Why this exists

On 2026-07-20 two autonomous audit passes produced **six false claims**, each recommending an
action that would have caused damage. Every one was falsified in under a minute once someone
looked. The corrections are `9bc2849` and `d12fe17`.

| claim | reality |
|---|---|
| "CS229 stack — all 11 modules, not imported by anything live. Suggest deprecating." | 11/11 imported by `sovereign/orchestrator.py`; deprecating breaks the orchestrator |
| "`pca_compressor.py` — equity pipeline depends on it, file doesn't exist" | deliberately removed with the ML line; no live importer |
| "NY AM Scanner — silent crash since May, 0-byte logs" | ran that morning; the 0-byte file is the *launchd* log, empty by design |
| "`alexandrian_library.learn()` never fires" | wired at `orchestrator.py:2093` behind an 8% drawdown gate |
| "`RISK_FRAMEWORK.md` (ratified 2026-07-20)" | the file did not exist when cited |
| "`stanford_cs229/`" | no such directory |

Documentation did not prevent this. `AGENT_DIRECTIVE.md` already carried eight standing rules
and `NEXT.md:1467` already warned against drive-by fixes during audits. Both were in force.
Hence a checker that runs.

## The claim classes

- **C1 — `EXISTS`.** "File X does not exist" / "X is missing" / "described but not built".
  Resolves the path. On a miss, searches deletion history so **deliberately removed** is
  distinguished from **never built** — the difference between "the doc is stale" and "rebuild
  it". A claim of absence is `REFUTED` when the path resolves.

- **C2 — `IMPORTED`.** "X is not imported by any live path" / "inert" / "safe to deprecate".
  Import edges are found by **AST parse, never substring search**. Substring matching produced
  two false positives this month by matching a module name inside its own docstring. Live
  roots are the entry scripts of installed plists, plus their transitive imports. A claim of
  deadness is `REFUTED` when any importer is found, and the importing file and line are
  reported so the refutation is itself checkable.

- **C3 — `LOGPATH`.** "Job silently crashing" / "0-byte logs" / "dark since <date>".
  Resolves **both** the plist's `StandardOutPath`/`StandardErrorPath` **and** every log path
  the target script writes to itself. An empty launchd log beside a large script log is a
  healthy job, not a crash. Also reports any in-script time guard, because an early `exit 0`
  outside a trading window is the specific thing that made a working scanner look dead.

- **C4 — `CITED`.** "per DOC.md" / "ratified in X" / "specified by Y". Verifies the cited
  document exists and contains the cited string. Catches a constant being justified by a
  document that has not been written.

## Verdicts

`CONFIRMED` · `REFUTED` · `UNVERIFIABLE`

`UNVERIFIABLE` is a **first-class verdict, not a soft pass**. A claim the tool cannot
mechanically test must be labelled so and never silently counted as confirmed. Most prose
claims will land here; that is honest, and better than a tool implying coverage it lacks.

Exit code is non-zero when any claim is `REFUTED`, so the checker can gate a routine.

## Known limits — stated so the tool is not over-trusted

- **Dynamic imports are invisible to an AST walk.** `ict/pipeline.py` reaches sovereign
  through an `importlib` hook; that edge will not appear in the graph. C2 therefore reports
  `UNVERIFIABLE` rather than `CONFIRMED` when it finds no importer but detects dynamic-import
  machinery in the live set. An over-trusted checker is the same failure one level up.
- **The tool never acts.** No auto-fixing, no ticket filing, no LLM adjudication. It
  falsifies; a human decides.

```yaml claims-spec
spec_version: 1

# Roots treated as "live": installed plists point at these entry scripts.
live_root_globs:
  - scripts/*.plist
  - execute_daily.py
  - sovereign/orchestrator.py

# Directories excluded from every scan. Worktrees are parallel-agent copies and
# counting them inflates every inventory number.
exclude_dirs:
  - .git
  - __pycache__
  - .claude/worktrees
  - node_modules
  - archive

# Package roots searched when resolving a module name to a file.
package_roots:
  - sovereign
  - execution
  - ict
  - backtester
  - audit
  - scripts

# Markers that identify a deliberate removal rather than an absence.
removal_evidence_paths:
  - SYSTEM_STATUS.md
  - trial/subtraction_verdicts.md
  - NEXT.md

# C3: a launchd log this size or smaller is not by itself evidence of failure.
empty_log_bytes: 0

# C3: shell guards that cause a healthy early exit.
time_guard_patterns:
  - "UTC_TIME"
  - "exit 0"

# C3: a non-empty log older than this proves the job RAN once, not that it is
# still alive. Beyond it, "silently dead" cannot be refuted on log contents alone —
# the verdict downgrades to UNVERIFIABLE rather than a false REFUTED.
log_recency_days: 4
```
