# AGENT_DIRECTIVE.md — Alta Investments Autonomous Trading Agent

**Standing order for every autonomous Claude session in ~/quant.**
**Read this file first. Do exactly what it says. Deviate from nothing.**

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

### Step 1 — Information layer pull
```bash
cd ~/quant
python3 -m execution.context
```
This writes the consolidated morning context to `data/context/`. Every source
gets a FRESH/STALE/SILENT_NULL/UNAVAILABLE status. If any critical source is
UNAVAILABLE or ERROR, log the failure explicitly in NEXT.md — do not proceed
silently.

### Step 2 — Oracle bias
```bash
python3 -m execution.bias
```
Reads the context output and generates the session bias. The bias gates NOTHING
by design (ARCHITECTURE.md L1/L2 wall). Record it; do not let it modify signals.

### Step 3 — Gapper candidate scan
```bash
python3 -m execution.signals --date today
```
Runs both HYP-107 and HYP-093 filters against the pre-market movers list.
Produces the GO/NO-GO candidate list. Every rejection reason is logged (the
scanner emits NO_GO rows with reasons — do not suppress them).

### Step 4 — Update prop sim dashboard
```bash
python3 scripts/sync_dashboard_data.py
```
Refreshes `data/agent/dashboard_state.json` and `logs/prop_challenge_sim.json`
with today's candidate count, running fill P&L from the shadow logs, and
challenge status. If sync_dashboard_data.py errors, read `data/agent/dashboard_state.json`
directly, update `last_updated` and `next_trade_window`, and write it back.

### Step 5 — Summarize and commit
Append a morning-pass entry to NEXT.md:
```
## {date} · MORNING PASS
Context: {FRESH/STALE counts} | Bias: {bias direction} | Candidates GO: {n} NO-GO: {n}
Dashboard: challenge_day={n}, running_pnl={x}%
```
Then commit and push:
```bash
git add data/context/ data/agent/dashboard_state.json NEXT.md
git commit -m "[AGENT] Morning pass {date}: {n} GO candidates"
git push
```

---

## ROUTINE — 09:30 ET (included in the 07:55 morning pass)

The signals module (`execution/signals.py`) scores candidates produced by the
morning scan using the frozen config (`execution/config.py FROZEN`). The GO/NO-GO
output is written to `data/decisions/signals_{date}.json`.

**What Claude must do at this step (within the morning run, after the market opens
if signals is re-run at 09:30, or as part of the 07:55 pre-open pass):**

1. Print the ranked GO list to stdout (launchd captures this to morning_agent.log).
2. Log intent to the harness input file so the EOD fill-capture knows what to price:
   ```bash
   python3 -m execution.harness --log-intent
   ```
   This records signal timestamps without capturing fills (fills require T+16min).
3. For each GO candidate: note symbol, filter (HYP-093 or HYP-107), entry params.

**Standing constraint:** if `execution/config.py verify_frozen_hash()` fails, abort
immediately. Write the failure to NEXT.md and push. Do not attempt to trade on a
drift-detected day.

---

## ROUTINE — 16:05 ET (fired by com.alta.eod_agent, 16:00 launchd)

Goal: collect the session's fills, run EOD reconciliation, update the dashboard,
write the Obsidian session note.

**Note:** `com.alta.execution_harness` (Python, launchd 16:05 ET) runs the
bid/ask capture automatically. The EOD Claude agent coordinates the broader EOD,
which includes post-harness reconciliation and narrative writing.

### Step 1 — Wait for harness output
Check that `data/execution/fill_log.jsonl` has a row for today's date. If the
harness hasn't run yet (it fires at 16:05; this agent fires at 16:00), wait up
to 3 minutes and re-check before proceeding.

### Step 2 — Run EOD reconciliation
```bash
python3 -m execution.eod
```
This produces `Trading/Ops/System-EOD-{date}.md` in the Obsidian vault.

### Step 3 — Update dashboard
```bash
python3 scripts/sync_dashboard_data.py --eod
```
Refreshes the challenge P&L, fill count, running Sharpe vs HYP-103 spec.

### Step 4 — Obsidian note
```bash
python3 -m execution.obsidian --session-close
```
Writes the session note to `Trading/Oracle-Log/{date}.md`.

### Step 5 — EOD NEXT.md entry + commit
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

Goal: synthesize the recent market session's data, run exploratory micro-backtests,
update the weekly pattern file, and pre-register any hypothesis that survives a
quick scan.

### Step 1 — Pull recent movers data
```bash
python3 -m execution.alpaca --pull-movers --lookback 5
```
Writes the last 5 sessions' gap universe to `data/research/gapper/movers_recent.json`.

### Step 2 — Run micro-backtests
For any candidate pattern spotted in recent data that is NOT already in the
hypothesis ledger, run a quick screen:
```bash
python3 scripts/run_research_queue.py --quick-scan
```
Log findings to `data/agent/findings.jsonl`.

### Step 3 — Update weekly pattern file
Append to `research/weekly_pattern_update.md`:
```markdown
## {date} — nightly pattern update
- Sessions reviewed: {n}
- New patterns flagged: {descriptions or "none"}
- Hypotheses queued: {hyp ids or "none"}
- Notes: {any anomalies, regime shifts, unusual volume}
```

### Step 4 — Pre-register surviving hypotheses
If any pattern from Step 2 clears a first-pass directional permutation test
(p < 0.10, minimum n ≥ 20), pre-register it:
```bash
python3 scripts/vrp_sign_prereg.py --draft-from findings.jsonl
```
This writes a draft to `data/research/preregister/` for operator review.
Do NOT promote to the hypothesis ledger without operator sign-off.

### Step 5 — Commit
```bash
git add research/weekly_pattern_update.md data/research/ NEXT.md
git commit -m "[AGENT] Nightly research {date}"
git push
```

---

## STANDING RULES (active every routine, no exceptions)

1. **Never retune parameters.** The HYP-093 and HYP-107 filter configs in
   `execution/config.py` are frozen. Any drift from `verify_frozen_hash()` is
   an incident, not a tuning opportunity. Log and abort.

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
   the hypothesis instead.

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
   execution path (`execution/harness.py`, `execution/scan.py`) falls under the
   same rule — changes require a plan file and NEXT.md unlock.

---

## ESCALATION

If any step produces an unhandled exception or an unrecognized state, do the
minimum: write a clear error description to NEXT.md, commit it, push it, and
exit. Do not attempt heroic recovery. The next session will have the error in
context and can address it properly.

If the `execution/config.py` hash verification fails: STOP. Commit the failure
note. Do not run any signal or fill logic.

If the prop challenge account hits -8% drawdown (80% of the -10% limit): STOP
all signal logging. Write `data/agent/CHALLENGE_NEAR_LIMIT.md` with the current
equity state and wait for operator decision.
