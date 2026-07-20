# AGENT_DIRECTIVE.md — Alta Investments Autonomous Trading Agent

**Standing order for every autonomous Claude session in ~/quant.**
**Read this file first. Do exactly what it says. Deviate from nothing.**

> **Every command below has been verified against the actual script interface**
> (`--help` / source read on 2026-07-19). If a command here does not match what a
> script accepts, that is an incident — see STANDING RULE 9. Do not "fix" the
> command by guessing; stop and file the incident.

---

## MISSION

Pass a prop firm challenge (FunderPro $200k, spec: HYP-103) using two confirmed
gapper-equity edges. Every market day is a learning event — log it, measure it,
commit it. Never modify live trading parameters, filters, or thresholds outside
a pre-registered hypothesis with a sealed holdout confirmation. The edges are
frozen. The machine runs them.

**Active edges:**
- **HYP-093 "Undertow"** — Fade short: smallcap gap-up ≥40%, entry 10:45 ET bar,
  stop 25% adverse (post-entry high basis), exit EOD. VALID_BUT_BELOW_FLOOR on
  backtests; HYP-103 is the live operating spec.
- **HYP-107 "Divining Rod"** — Runner long: gap-up ≥30%, volume and momentum
  filters, 09:30 entry. REAL_BUT_MARGINAL_EXECUTION_UNRESOLVED; still in
  shadow measurement phase.

**Governing documents (read before any architectural decision):**
- `TRADING_PHILOSOPHY.md` — six tenets; nothing runs that violates them
- `RISK_CONSTITUTION.md` — hard caps; do not override
- `audit/divergence_spec.md` and `NEXT.md` — shadow/execution freeze status

---

## ROUTINE — 08:00 ET (fired by com.alta.morning_agent, 07:55 launchd)

Goal: build the complete information layer, generate Oracle bias, scan gapper
candidates, and update the prop sim dashboard before 09:30.

**Working directory for every command in this routine:** `cd ~/quant` first, so
that `python3 -m execution.*` resolves and `python3 -c "from execution... "`
imports work. All of these modules use launchd-safe `.env` parsing; no shell
environment is assumed.

### Step 1 — Information layer pull
```bash
cd ~/quant
python3 -m execution.context
```
Real interface: `execution.context` accepts `--day DATE`, `--out DIR`, `--print`.
Bare invocation targets today and writes the consolidated context to
`data/context/`. Add `--print` if you want the health table on stdout. Every
source gets a FRESH/STALE/SILENT_NULL/UNAVAILABLE status. If any critical source
is UNAVAILABLE or ERROR, log the failure explicitly in NEXT.md — do not proceed
silently.

### Step 2 — Oracle bias
```bash
python3 -m execution.bias
```
Real interface: `--day DATE`, `--out DIR`, `--score DIRECTION`, `--record`. Bare
invocation reads today's context output and writes the session bias. The bias
gates NOTHING by design (ARCHITECTURE.md L1/L2 wall). Record it; do not let it
modify signals.

### Step 3 — Gapper candidate scan
```bash
python3 -m execution.signals
```
Real interface: `--day DATE`, `--replay`, `--no-news`, `--out DIR`.
**There is no `--date` flag and `--day` only accepts an ISO date
(`YYYY-MM-DD`) — passing `today` crashes.** Bare invocation targets today.
To score a past session from the archive: `--day 2026-07-15 --replay`.
Runs both HYP-107 and HYP-093 filters against the pre-market movers list and
writes the GO/NO-GO list to `data/decisions/signals_{date}.json`. Every rejection
reason is logged (NO_GO rows carry reasons — do not suppress them).

### Step 4 — Update prop sim dashboard
```bash
python3 scripts/sync_dashboard_data.py
```
**This script takes NO arguments** — it has no argument parser. Do not pass
`--eod` or any other flag; extra args are silently ignored, which means a flag
you expect to change behaviour will not. It refreshes
`data/agent/dashboard_state.json` and `logs/prop_challenge_sim.json` with today's
candidate count, running fill P&L from the shadow logs, and challenge status.
If it exits non-zero, do NOT retry with different flags — apply STANDING RULE 9.

### Step 5 — Summarize and commit
Append a morning-pass entry to NEXT.md:
```
## {date} · MORNING PASS
Context: {FRESH/STALE counts} | Bias: {bias direction} | Candidates GO: {n} NO-GO: {n}
Dashboard: challenge_day={n}, running_pnl={x}%
```
Then commit and push:
```bash
git add data/context/ data/agent/dashboard_state.json data/decisions/ NEXT.md
git commit -m "[AGENT] Morning pass {date}: {n} GO candidates"
git push
```

---

## ROUTINE — 09:30 ET (part of the 07:55 morning pass; no separate job)

The GO/NO-GO output written by Step 3 (`data/decisions/signals_{date}.json`) **is**
the logged trade intent for the session. There is nothing further to run at the
open.

**What Claude must do at this point:**

1. Print the ranked GO list to stdout (launchd captures this to
   `morning_agent.log`). `execution.signals` already prints the ranked GO list
   and the first NO_GO reasons when Step 3 runs — re-run it if you need the list
   again: `python3 -m execution.signals`.
2. For each GO candidate, note symbol, filter (HYP-093 or HYP-107), entry params.

**Do NOT run `execution.harness` here.** The harness is measurement-only and
requires one of `--live | --replay DATE | --backfill FILE` — there is no
`--log-intent` flag. Bid/ask fill capture for the session is performed by the
separate `com.alta.execution_harness` launchd job at 16:05 ET, which runs:
```bash
python3 -m execution.harness --live --signals data/decisions
```
(`--signals DIR` makes only GO signals fill.) The morning agent does not invoke
the harness.

**Standing constraint:** if any step reports a frozen-config hash mismatch (the
HYP-093 / HYP-107 filter configs in `execution/config.py` are frozen), abort
immediately. Write the failure to NEXT.md and push. Do not trade on a
drift-detected day.

---

## ROUTINE — 16:05 ET (fired by com.alta.eod_agent, 16:00 launchd)

Goal: collect the session's fills, run EOD reconciliation (which also writes the
Obsidian session note), update the dashboard, commit.

**Note:** `com.alta.execution_harness` (Python, launchd 16:05 ET) runs the
bid/ask capture automatically. This EOD Claude agent fires at 16:00 and
coordinates the post-harness reconciliation and narrative.

### Step 1 — Wait for harness output
Check that `data/execution/fill_log.jsonl` has a row for today's date. If the
harness hasn't run yet (it fires at 16:05; this agent fires at 16:00), wait up
to 3 minutes and re-check before proceeding. This is a file check — no script.

### Step 2 — Run EOD reconciliation (writes the Obsidian note)
```bash
python3 -m execution.eod
```
Real interface: `--day DATE`, `--dry-run`, `--fills FILE`, `--signals DIR`,
`--score-bias DIRECTION`. Bare invocation reconciles today. **This step already
writes `Trading/Ops/System-EOD-{date}.md` into the Obsidian vault** —
`execution.eod` calls `obsidian.write_eod_note()` internally
(`execution/eod.py:273`). There is no separate Obsidian step and no
`execution.obsidian` CLI (it is a pure library). Use `--dry-run` to render the
note without touching the vault. If the vault is not mounted, `eod` raises
`VaultUnavailable` loudly — that is an incident (STANDING RULE 9), not a silent
skip.

### Step 3 — Update dashboard
```bash
python3 scripts/sync_dashboard_data.py
```
No arguments (see morning Step 4). Refreshes challenge P&L, fill count, and
running Sharpe vs the HYP-103 spec from the shadow logs.

### Step 4 — EOD NEXT.md entry + commit
```
## {date} · EOD PASS
Fills: {n} (GO: {n_go}, NO-GO: {n_nogo}) | Session P&L: {x}%
Challenge status: day {n}, running DD {x}%, target {x}% remaining
Harness delta (vs backtest): {x}pp
```
```bash
git add data/execution/fill_log.jsonl data/agent/dashboard_state.json NEXT.md
git commit -m "[AGENT] EOD pass {date}: {n} fills, session {x}%"
git push
```

---

## ROUTINE — 21:00 ET (fired by com.alta.research_agent, Sun–Thu)

Goal: synthesize the recent market session's data, run exploratory
micro-backtests from the queue, update the weekly pattern file, and flag any
candidate that looks worth pre-registering — for OPERATOR review, never
autonomous promotion.

### Step 1 — Pull recent movers data
`execution.alpaca` is a pure library with no CLI. Import `movers()` directly —
it returns the current top-gainers screener snapshot (a single snapshot; the
library has **no multi-session lookback**, so do not claim `--lookback 5`):
```bash
cd ~/quant
python3 -c "
import json, pathlib
from execution.alpaca import movers
g = movers(top=50)
p = pathlib.Path('data/research/gapper/movers_recent.json')
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(g, indent=2))
print(f'wrote {p} ({len(g)} gainers)')
"
```
If Alpaca returns a 403, that is an entitlement error (`AlpacaEntitlementError`),
raised loudly — treat it as an incident (STANDING RULE 9), do not retry blindly.

### Step 2 — Drain the research queue (bounded)
```bash
python3 scripts/run_research_queue.py --max 5
```
Real interface: `--max MAX` (default 25), `--dry-run`. **There is no
`--quick-scan` flag.** `--max 5` keeps the nightly pass light and bounded; use
`--dry-run` to print the picks without executing anything. Findings are written
by the queue tasks; review `data/agent/findings.jsonl`.

### Step 3 — Update weekly pattern file
Append to `research/weekly_pattern_update.md`:
```markdown
## {date} — nightly pattern update
- Sessions reviewed: {n}
- New patterns flagged: {descriptions or "none"}
- Candidates queued for operator review: {ids or "none"}
- Notes: {any anomalies, regime shifts, unusual volume}
```

### Step 4 — Flag surviving candidates (operator-gated; NO autonomous prereg)
Pre-registration is an operator decision (STANDING RULE 4). **Do not run
`scripts/vrp_sign_prereg.py --draft-from …` — that flag does not exist**; the
script only supports `--write` / `--check` and is specific to the VRP-sign
prereg, not a generic findings drafter. If a pattern from Step 2 clears a
first-pass directional permutation test (p < 0.10, n ≥ 20), do NOT register it
autonomously. Instead, record it as a candidate in NEXT.md and
`data/agent/findings.jsonl` and stop:
```
## {date} · RESEARCH — candidate flagged for operator
{pattern}: first-pass p={x}, n={n}. Awaiting operator sign-off before prereg.
```
To validate an existing VRP-sign prereg file only: `python3 scripts/vrp_sign_prereg.py --check`.

### Step 5 — Commit
```bash
git add research/weekly_pattern_update.md data/research/ data/agent/findings.jsonl NEXT.md
git commit -m "[AGENT] Nightly research {date}"
git push
```

---

## STANDING RULES (active every routine, no exceptions)

1. **Never retune parameters.** The HYP-093 and HYP-107 filter configs in
   `execution/config.py` are frozen. Any drift from the frozen-config hash check
   is an incident, not a tuning opportunity. Log and abort.

2. **Never touch the sealed holdouts.** Files under `data/research/gapper/holdout/`
   and `data/research/yield_frontier/gauntlet/` are read-protected. Do not
   open, read, or pass them to any analysis routine.

3. **Always commit and push after each routine.** An unpushed branch is a
   single-machine point of failure (CLAUDE.md standing constraint). If git push
   fails, log the failure in NEXT.md and retry once. If it fails twice, write
   the error to `logs/push_failures.log` and stop — do not continue accumulating
   uncommitted work.

4. **No parameter changes without a logged hypothesis.** Any proposed threshold
   change requires an entry in `data/agent/param_change_log.jsonl` with a rationale
   BEFORE it is applied. If you find yourself wanting to change a number, write
   the hypothesis instead. Pre-registration of new edges is operator-gated.

5. **Log every decision, including abstentions.** A day with zero GO candidates
   is a valid and important data point. Log it as such. Do not re-run the scanner
   hoping for a different result.

6. **Degraded sources are NOT silent nulls.** If a data source returns STALE,
   SILENT_NULL, or UNAVAILABLE, log it explicitly. Never propagate a silent null
   as data downstream.

7. **Oracle feedback loop is mandatory.** Any trade decision logged to
   `sovereign/intelligence/decision_logger.py` must receive an `update_outcome()`
   call at close. Skipping it degrades the Oracle silently.

8. **The shadow/execution-path freeze is in effect.** Do not modify
   `forex_exit_manager`, `decide_exit`, or anything importable by the live/backtest
   execution path without an explicit unlock recorded in `NEXT.md`. Equities
   execution path (`execution/harness.py`, `execution/signals.py`) falls under the
   same rule — changes require a plan file and NEXT.md unlock.

9. **A command that fails is an incident, not a puzzle to solve.** If a command
   exits non-zero or fails to parse its arguments, STOP and write an incident note
   to `logs/incidents/`. Do not attempt to repair or rewrite the command. Do not
   proceed to the next step. Do not commit. The incident note should record: the
   exact command run, the exit code, the stderr/stdout tail, and the routine and
   step it failed at. The next session will triage it.

---

## ESCALATION

If any step produces an unhandled exception or an unrecognized state, do the
minimum: write a clear error description to `logs/incidents/` (STANDING RULE 9)
and to NEXT.md, commit the NEXT.md note, push it, and exit. Do not attempt
heroic recovery. The next session will have the error in context.

If the frozen-config hash verification fails: STOP. Commit the failure
note. Do not run any signal or fill logic.

If the prop challenge account hits -8% drawdown (80% of the -10% limit): STOP
all signal logging. Write `data/agent/CHALLENGE_NEAR_LIMIT.md` with the current
equity state and wait for operator decision.
