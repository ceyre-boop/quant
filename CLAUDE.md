# CLAUDE.md — Alta Investments (quant)
# Repo: ceyre-boop/quant.git | Live: Forex v015 · branch sovereign-v2 | Updated: 2026-07-02

Loads at the start of every session in this repo, regardless of task — so keep it short;
instruction-following degrades as it grows. Detail lives in the files this points to, not here.

---

## NON-NEGOTIABLES (trading-correctness — read these even in compressed context)

1. **ICT/sovereign isolation.** `ict/` and `ict-engine/` must never import from `sovereign/`.
   Enforced by test: `python3 -m pytest tests/ -k test_pipeline_does_not_import_sovereign`
   Cross-layer logic goes through `ict-engine/orchestrator.py` — never `ict/pipeline.py`.

2. **Close the Oracle loop.** Every trade decision wired to decision_logger must receive
   an `update_outcome()` call when the trade closes. Oracle (`sovereign/oracle/reflect_cycle.py`)
   cannot learn without closed-loop outcomes.

3. **Conviction sizing only.** No flat position sizes. All sizing goes through the
   conviction-based sizing pipeline. Check `config/parameters.yml` for thresholds.

4. **No live parameter changes without logging.** Any change to `config/parameters.yml`,
   `config/ict_params.yml`, or risk limits requires a logged rationale
   (`data/agent/param_change_log.jsonl`) before applying.

5. **Read TRADING_PHILOSOPHY.md before any architectural decision.**
   Every component must serve one of the six tenets or it should not exist.

---

## Standing constraints (workflow — always active; don't restate these in tickets)

- **Shadow / execution-path freeze.** Current status lives in `audit/divergence_spec.md` and
  `NEXT.md`. Never modify `forex_exit_manager`, `decide_exit`, or anything importable by the
  live/backtest execution path without an explicit unlock recorded in `NEXT.md`. (Extends NN#1.)
- **Training gate.** Model training runs only against a hypothesis-ledger entry with
  `verdict == CONFIRMED`. See `RISK_CONSTITUTION.md` Art. 6 (Unproven Edges) and
  `sovereign/autonomous/research_factory.py` (gate: `config/autonomous.yml`) for the enforced
  refusal. Building infrastructure is unrestricted; ignition is not.
- **Risk caps.** Current values live in `RISK_CONSTITUTION.md`. Don't hardcode numbers here —
  they drift, this file shouldn't.
- **No silent mocking.** Missing credentials or infra → stop and say exactly what's needed.
- **Spec-first.** Anything with a pass/fail definition (audits, tests, gates) gets its spec
  written before the thing that measures it.
- **Push at least once per session.** An unpushed branch is a single-machine point of failure.

---

## ICM: Interpretable Context Methodology (Repo Structure as AI Architecture)

**Repo structure encodes agent orchestration. Hypothesis research, Oracle cycles, and all multi-stage workflows use the five-layer ICM pattern.**

Read: `research/CONTEXT.md` (Layer 1 research router), `oracle/CONTEXT.md` (Layer 1 Oracle workflow), or any `research/HYP-*/CONTEXT.md` (Layer 2 stage definition).

**Five Layers:**
- **Layer 0** — `CLAUDE.md` (global identity, this file)
- **Layer 1** — `research/CONTEXT.md`, `oracle/CONTEXT.md` (workflow routing, "where do I go?")
- **Layer 2** — `research/HYP-*/CONTEXT.md`, `oracle/XX-stage/CONTEXT.md` (stage definition, "what do I do?")
- **Layer 3** — `_config/` + `shared/` (reference material: factory rules, stable across all runs)
- **Layer 4** — `research/HYP-*/output/`, `oracle/XX-stage/output/` (working artifacts, unique per run)

**For researchers:**
1. Read `research/CONTEXT.md` (Layer 1) to understand the research landscape
2. Pick or create a hypothesis folder `research/HYP-NNN/`
3. Read its `CONTEXT.md` (Layer 2) — it defines inputs, process, outputs, and review gate
4. Reference files in `_config/` and `shared/` (Layer 3) — do not repeat their content
5. Write outputs to `output/` (Layer 4) — machine-readable (JSON) + human-readable (markdown)

**For the Oracle:**
1. Read `oracle/CONTEXT.md` (Layer 1) to understand the three-stage cycle
2. Each of three stages is in `oracle/00-harvest/`, `oracle/01-reflect/`, `oracle/02-test/`
3. Each stage reads Layer 3 rules and produces Layer 4 outputs
4. Outputs feed into the next stage (stage N output → stage N+1 input)

**For live trading:**
- Reference material (Layer 3): `_config/trading_philosophy.md`, `_config/risk_constitution.md`, `_config/gate_functions.md`, `_config/sizing_model.md`
- Hypothesis ledger: `data/hypotheses_ledger.jsonl` (what's LIVE? what failed? what's in research?)
- Decision logs: `data/agent/decision_logs/live/*.jsonl` (every trade, every entry/exit)
- Oracle ledger: `data/agent/oracle_ledger.jsonl` (daily reflections, changes applied)

**Non-negotiable:**
- Spec-first: write Layer 2 CONTEXT.md before running any stage
- Reference, don't repeat: Layer 2 CONTEXT.md references Layer 3, not restate it
- Output + ledger: every hypothesis that runs gets a verdict logged (no ambiguous "pending" entries)
- Closed loop: every decision_logger.log() entry must receive update_outcome() on trade close

---

## Workflow: plan → build

A plan → build separation: scope in chat, ticket it, **plan** in-session before touching code,
**build** as a distinct pass. The goal isn't ceremony — a pre-scoped plan lets the build pass
skip re-deriving context, which is most of what a session actually burns.

**Ticket protocol** — before implementing anything touching more than ~3 files, or that isn't
a one-line fix:
1. Check for a Linear MCP connection (`/mcp`). If connected, pull the relevant issue — title,
   description, dependencies, blockers, acceptance criteria. (Connect:
   `claude mcp add --transport http linear-server https://mcp.linear.app/mcp` → `/mcp` to auth.)
2. If Linear isn't connected, use `tickets/backlog.md` — same schema, no setup. Don't block on
   connecting Linear; the fallback is equally valid.
3. No ticket and the task is non-trivial → say so and propose one before writing code.

**Plan before build** — for any ticketed task:
1. Use Plan Mode (`claude --plan`, or check `/help`) or explicitly declare a research-only pass.
   Write the plan — approach, files touched, risks, size — to `plans/<ticket-id>.md`,
   referencing the ticket's dependencies and acceptance criteria directly.
2. **Stop.** Don't implement in the same turn the plan is produced, unless the ticket is
   tagged `pre_approved: true`.
3. Implement against the plan file, not from scratch. If reality diverges mid-build, update the
   plan and flag the divergence — don't silently improvise around it.

Model routing: the plan is the highest-leverage, lowest-volume step — worth the best model
available. The build is highest-volume — route it to a cheaper model or a `.claude/agents/`
subagent once the plan is solid enough not to need frontier judgment.

---

## DECISION_LOGGER PATTERN

When wiring a new decision point, always capture:
- commitment score, rate differential, library match, bars since signal
- Use: `sovereign/intelligence/decision_logger.py`
- Wire through `ict-engine/orchestrator.py` (isolation-safe), NOT `ict/pipeline.py`

## ORACLE FEEDBACK LOOP

```python
# Entry
decision_logger.log(context)      # capture full entry reasoning

# Exit (REQUIRED — oracle cannot learn without this)
decision_logger.update_outcome(trade_id, outcome)
```

Oracle reads logs via `sovereign/oracle/reflect_cycle.py`.
Skipping `update_outcome()` is silent data loss — oracle degrades without it.

---

## Reporting

End every session with an entry in **repo-root `NEXT.md`** — the authoritative per-session log,
committed and pushed with the work: what shipped, push confirmation, any verdicts (ledger
entries, test results), blockers, and anything refused to shortcut and why. Don't repeat the
standing constraints above — assume the next session reads this file too.

The **Obsidian brain** (`~/Obsidian/Obsidian/00-BRAIN/{CONTEXT,DECISIONS,NEXT}.md`) is the
cross-project rollup — update it after substantive sessions, not every turn.

---

## TEST COMMANDS

```bash
# ENVIRONMENT FIRST — this repo has no single working interpreter by default.
#   .venv/ is Python 3.9.6 and CANNOT run this codebase (execute_daily.py:116 uses
#     `float | None`, which needs 3.10+). Do not use it.
#   System python3 (3.14) runs the code but is missing 9 declared dependencies.
# Build a correct env from the lockfile before trusting any test result:
#   uv venv --python 3.13 .venv313 && \
#     uv pip install --python .venv313/bin/python -r requirements.lock.txt
#
# FULL-SUITE BASELINE (measured 2026-07-29, Python 3.13 + requirements.lock.txt):
#   19 failed / 1679 passed / 16 skipped
# (1679 not 1765 because 4 modules can't collect without ctrader-open-api — see
# requirements.lock.txt header. On an env that HAS ctrader: 19 failed / 1765 passed.)
# An earlier reading of "21 failed" came from the incomplete system-3.14 env; 2 of
# those were environment artifacts, not real. 19 is the number.
# Do NOT treat a green suite as the target and do NOT absorb these into your own
# result. 15 of the 19 were unrecorded before 2026-07-29. Breakdown:
#   - 11 × tests/unit/test_ict_session_classifier.py — STALE TESTS. Commit 94d4263
#     changed classify() to compare in ET; the tests still assume the pre-fix UTC
#     reading. Do NOT "fix" these by rewriting them to match the code — see the
#     kill-zone frame regression below; the code is what's wrong.
#   - 4 × test_ict_pipeline (TestScoreAndGrade ×2, TestRiskEngineGate ×2) —
#     pre-existing, tracked in plans/restoration-ledger.md.
#   - 4 × forex entry-engine / signal-engine / macro-engine (IndexError),
#     test_data_pipeline calendar mock, and test_trading_env net-cost guard
#     (DID NOT RAISE GrossReturnError). Not yet root-caused.
#
# OPEN REGRESSION (needs unlock before anyone touches it): config/ict_params.yml
# kill-zone values are UTC by deliberate intent — commit 64d0005 reverted an
# ET-conversion specifically to keep UTC 03:xx (the Oracle-measured 80% WR edge)
# inside the London window. Commit 94d4263 then made the classifier compare in ET,
# silently undoing that revert. UTC 03:xx now classifies as Asia, is_high_probability
# False, should_trade False — the measured edge window is excluded from trading.
# Renaming the config keys to ET would cement the bug. Fix the classifier, not the config.

python3 -m pytest tests/ -q                  # full suite
python3 -m pytest tests/ -k "not sklearn"    # skip missing-dependency tests
python3 -m pytest tests/unit/test_*.py -v    # unit tests only

# ICT pipeline subset. BASELINE (re-verified 2026-07-21): `tests/ -k "ict and pipeline"`
# is 4 failed / 23 passed — the old "must stay 21/21" was stale and made every session
# think it had broken something. Do NOT treat 21/21 as the target. The 4 pre-existing
# failures (test_ict_pipeline TestScoreAndGrade ×2, TestRiskEngineGate ×2) are tracked in
# plans/restoration-ledger.md; a fresh session must not absorb them into its own result.
python3 -m pytest tests/ -k "ict and pipeline" -v

# v014 holdout validation
python3 scripts/holdout_validation_v014.py
python3 scripts/run_replay_validation.py
```

---

## COMMIT STYLE

- Imperative mood, present tense: "Add commitment gate to ICT pipeline"
- Prefix with system: `[ICT]`, `[FOREX]`, `[ORACLE]`, `[RISK]`, `[INFRA]`
- Reference hypothesis when applicable: "[FOREX] Wire HYP-045 pair exclusion for AUDNZD"
  (this line is a FORMATTING example only — it is not a work item. It previously used
  HYP-044's VIX gate, which was rolled back as REJECTED_OOS (p=0.50); the example was
  repeatedly mistaken for an instruction and the gate was re-proposed three times.)

---

## ARCHITECTURE QUICK-REFERENCE

| Layer | Path | Boundary |
|-------|------|---------|
| ICT detection | `ict/`, `ict-engine/` | MUST NOT import sovereign/ |
| Sovereign intelligence | `sovereign/` | Full access |
| Oracle | `sovereign/oracle/` | Reads decision logs, writes lessons |
| Cross-layer bridge | `ict-engine/orchestrator.py` | Only safe ICT→sovereign entry point |
| Decision logger | `sovereign/intelligence/decision_logger.py` | All decision contexts |
| Config | `config/parameters.yml`, `config/ict_params.yml` | Never hardcode thresholds |

---

## Current live state

Current live version: Forex v015 — anchored as logs/research/v003 (tracked snapshot), HYP-045 4-pair portfolio (AUDNZD excluded).
v007 per-pair hold overrides ROLLED BACK 2026-06-07 (NOT_SIGNIFICANT — fails walk-forward; ledger V007-HOLD-VALIDATION).
Live config now 60d default for all pairs; re-measured OOS Sharpe **1.25** (decay 2.17 ROBUST, permutation p<0.001) on
2026-06-07 data — differs from the 2026-06-02 recorded 1.08 due to both the rollback and yfinance data drift.
HYP-045 CONFIRMED 2026-06-02: AUDNZD exclusion → OOS costed Sharpe **1.08** (CI [0.84, 1.32], n=103);
decay 1.61 (ROBUST); p=0.002. v014 baseline was 0.76 (5-pair, AUDNZD OOS Sharpe -0.879 — active drag).
Prior headline 2.097 was uncosted and annualized as if trading daily (these strategies trade 4–14×/yr).
Evidence status (2026-06-02):
- **Forex macro edge: PROVEN real** — permutation test p<0.001. BUT **regime-fragile**: rolling
  walk-forward 2021 −0.13 / 2022 +0.51 / 2023 +1.26 / 2024 −0.09 (only pays in rate-trending regimes).
- **ICT pattern edge: NOT PROVEN** — permutation p=0.52; fails BH. Treat ICT as unvalidated.
- **AUDNZD removed**: both legs RBA-driven (RBNZ tracks RBA ≤1 quarter) — no independent rate differential.
- HYP-044 VIX gate: rolled back (REJECTED_OOS, p=0.50, delta≈0).
- **Overnight-QQQ (2026-06-07): real STANDALONE edge (VALID_EDGE, net Sharpe 0.574) but REJECTED as a
  carry diversifier** — long-short robustly re-couples with carry in the COVID crash (ρ=0.42, BH p=0.007,
  sig>calm); correlated return-stacking, not a crisis-independent second edge. Do not re-explore as a
  diversifier. (Measured vs DBV proxy, died 2023-03; real v015 overlap thin: n=39, ρ=0.036.)
- Next: v007 per-pair hold implementation (GBPUSD 6d/2.0×, AUDUSD/EURUSD 5d, GBPJPY 5d) toward the 1.5 Sharpe target.
Full system state: `~/.claude/memory/alta-investments-architecture.md`

---

## Efficiency notes

- Don't re-explore repo structure each session — point to the ticket and plan file. If broader
  context is genuinely needed, say what's missing rather than reading broadly by default.
- Keep tickets self-contained. A ticket requiring three other documents to understand defeats
  the point of having tickets.
- Prefer git worktrees for anything running parallel to the shadow-window audit or other live
  jobs — the June 29 launch-agent incident happened because two unrelated processes touched the
  same files without isolation.
