# Alta Investments — quant

Systematic trading research and execution. Live line: **Forex v015** on branch `sovereign-v2`.

Operating rules for contributors (human or agent) live in `CLAUDE.md` — read that before
changing anything. This file is orientation only.

---

## Evidence status — read this before trusting any number here

This repo deliberately records what is *not* proven alongside what is.

| Line | Status |
|---|---|
| **Forex macro carry (v015)** | **Real edge** — permutation p<0.001. But **regime-fragile**: rolling walk-forward 2021 −0.13 / 2022 +0.51 / 2023 +1.26 / 2024 −0.09. It pays in rate-trending regimes and not otherwise. |
| **ICT pattern edge** | **Not proven** — permutation p=0.52, fails Benjamini-Hochberg. Treat as unvalidated. |
| **Overnight-QQQ** | Real standalone edge, **rejected** as a carry diversifier (re-couples in crisis, ρ=0.42). Do not re-explore as one. |

Headline Sharpe figures are easy to misread. The v015 OOS costed Sharpe of ~1.25 is **not** a
daily-trading number — these strategies trade 4–14×/year. An earlier 2.097 headline was
uncosted and annualised as if trading daily. Sizing, not signal, is the lever.

Current state and hypothesis verdicts: `NEXT.md`, `data/agent/hypothesis_ledger.json`.

---

## Environment — do this first

**There is no working interpreter by default.** `.venv/` is Python 3.9.6 and cannot run this
codebase (`execute_daily.py` uses `float | None`, which needs 3.10+). The system `python3` may
be missing declared dependencies. Build from the lockfile:

```bash
uv venv --python 3.13 .venv313
uv pip install --python .venv313/bin/python -r requirements.lock.txt
```

`requirements.txt` is the loose spec; `requirements.lock.txt` is 107 resolved pins and is what
you should install. Regenerate with:

```bash
uv pip compile requirements.txt --python-version 3.13 --output-file requirements.lock.txt
```

Two known constraints, both documented in the lockfile header:

- **`ctrader-open-api` is excluded** — it pins `protobuf==3.20.1`, unresolvable against
  `firebase-admin`. Install it in a separate environment via `requirements-ctrader.txt` if you
  need the cTrader bridge.
- **The lock pins pandas 3.x.** Recorded historical results were produced under pandas 2.x. The
  lock buys determinism going forward; it does not reproduce historical numbers (TICK-098).

---

## Tests

```bash
.venv313/bin/python -m pytest tests/ -q
```

**Baseline: 19 failed / 1679 passed / 16 skipped.** A green suite is not the target — do not
absorb these into your own result, and do not "fix" them by rewriting assertions. Failures are
catalogued with root causes in `CLAUDE.md` under TEST COMMANDS. The largest cluster (11, in
`test_ict_session_classifier.py`) tracks a real open regression, not stale tests — see TICK-093.

One test is a hard invariant and must never be softened:

```bash
pytest tests/ -k test_pipeline_does_not_import_sovereign
```

---

## Architecture

| Layer | Path | Boundary |
|---|---|---|
| ICT detection | `ict/` | **MUST NOT import `sovereign/`** |
| Sovereign intelligence | `sovereign/` | Full access |
| Oracle (cognition) | `sovereign/oracle/` | Reads decision logs, writes lessons |
| Decision logger | `sovereign/intelligence/decision_logger.py` | All decision contexts |
| Config | `config/parameters.yml`, `config/ict_params.yml` | Never hardcode thresholds |

ICT/sovereign isolation is enforced by test and is non-negotiable. Cross-layer logic goes
through the orchestrator, never through `ict/pipeline.py` directly.

**The Oracle loop only learns if it is closed.** Every decision logged at entry must receive an
`update_outcome()` call when the trade closes. Skipping it is silent data loss.

```python
decision_logger.log(context)                        # entry
decision_logger.update_outcome(trade_id, outcome)   # exit — REQUIRED
```

---

## Scheduling

Jobs run under **macOS launchd** — not cron, not Windows Task Scheduler. Plists live in
`scripts/com.alta.*.plist`; roughly 50 are loaded. A job that commits but never pushes fails
silently unless it exits non-zero — see TICK-097 for that exact pattern and its fix.

---

## Working here

- **Spec first.** Anything with a pass/fail definition gets its spec written before the thing
  that measures it.
- **No silent mocking.** Missing credentials or infrastructure → stop and say what is needed.
- **Execution-path freeze.** Do not modify `forex_exit_manager`, `decide_exit`, or anything
  importable by the live/backtest path without an unlock recorded in `NEXT.md`.
- **Training gate.** Model training runs only against a hypothesis-ledger entry with
  `verdict == CONFIRMED`. Building infrastructure is unrestricted; ignition is not.
- **Log parameter changes.** Any change to `config/parameters.yml`, `config/ict_params.yml`, or
  risk limits requires a logged rationale in `data/agent/param_change_log.jsonl` first.
- **Push every session.** An unpushed branch is a single-machine point of failure.

Tickets are in `tickets/backlog.md`. Session history and verdicts are in `NEXT.md` — append,
don't rewrite.

---

## Key documents

| Document | Purpose |
|---|---|
| `CLAUDE.md` | Operating rules, test baseline, non-negotiables |
| `NEXT.md` | Per-session log — what shipped, verdicts, blockers |
| `TRADING_PHILOSOPHY.md` | Six tenets; every component must serve one |
| `RISK_CONSTITUTION.md` | Risk caps and unproven-edge policy |
| `tickets/backlog.md` | Open work |

---

## License

Proprietary — all rights reserved.
