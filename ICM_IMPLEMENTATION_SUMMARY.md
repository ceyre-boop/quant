# ICM Implementation Summary

**Date:** 2026-08-12  
**Reference:** arXiv:2603.16021v2 — "Interpretable Context Methodology: Folder Structure as Agent Architecture" (Van Clief & McDermott)  
**Status:** ✅ COMPLETE (prototype structure built, first hypothesis (HYP-093) scaffolded, Oracle cycle formalized)

---

## What Was Built

### Layer 0 (Global Identity)
- ✅ Updated `CLAUDE.md` with ICM section explaining five-layer structure

### Layer 1 (Workflow Routers)
- ✅ `research/CONTEXT.md` — Routes research work to individual hypothesis folders; lists active/live/graveyard hypotheses
- ✅ `oracle/CONTEXT.md` — Explains three-stage daily reflection cycle (harvest → reflect → test)

### Layer 2 (Stage Definitions)
- ✅ `research/HYP-093/CONTEXT.md` — First hypothesis, fully defined (mechanism_validation stage)
- ✅ `oracle/00-harvest/CONTEXT.md` — Harvest logs, market state, regime classification
- ✅ `oracle/01-reflect/CONTEXT.md` — Analyze metrics, identify anomalies, propose lessons
- ✅ `oracle/02-test/CONTEXT.md` — Holdout validation, verdicts, recommendations

### Layer 3 (Reference Material / Factory)
**Directory:** `_config/` + `shared/`

Reference files (stable, configured once, reused across all runs):
- ✅ `_config/LAYER3_README.md` — Guide to the factory concept
- ✅ `_config/trading_philosophy.md` — Copied from root (six tenets)
- ✅ `_config/risk_constitution.md` — Copied from root (Articles 1–6, ratified)
- ✅ `_config/gate_functions.md` — NEW: Sharpe gate (0.30 DSR), permutation test (p<0.05), OOS degradation (<20%)
- ✅ `_config/sizing_model.md` — NEW: Conviction-based sizing, rate differential, library match, carry heat cap
- ✅ `shared/hypothesis_ledger_schema.md` — NEW: Ledger entry structure (what to record per hypothesis)
- ✅ `shared/decision_logger_schema.md` — NEW: Entry/exit logging schema (Oracle reads this)

### Layer 4 (Working Artifacts / Product)
**Folders created, awaiting first runs:**
- ✅ `research/HYP-093/output/` — Will contain: mechanism_validation.md, backtest_spec.md, analysis.json, human_review.md
- ✅ `oracle/00-harvest/output/` — Will contain: decision_logs_summary.json, market_state.json, hypothesis_status.json
- ✅ `oracle/01-reflect/output/` — Will contain: reflection_metrics.json, anomalies.json, lessons.json
- ✅ `oracle/02-test/output/` — Will contain: test_results.json, test_verdict.json

---

## Key Design Decisions

### 1. **Specification-First**
Each Layer 2 CONTEXT.md is **both** AI instruction **and** human documentation. It defines:
- **Inputs** (what data do we read? from which layers?)
- **Process** (what steps do we execute?)
- **Outputs** (what deliverables do we produce?)
- **Review gate** (what makes this stage PASS or FAIL?)

This ensures no ambiguity. The specification is the contract.

### 2. **Reference vs. Working Artifacts**
- **Layer 3** (reference) is stable: trading_philosophy, risk_constitution, gate criteria
- **Layer 4** (artifacts) is mutable per-run: backtest results, analysis output, verdicts

Each hypothesis stage separates these completely. No contamination of rules by outputs.

### 3. **Token Efficiency**
ICM delivers focused context windows (2,000–8,000 tokens per stage) vs. monolithic approaches (30,000–50,000 tokens with irrelevant material). The paper shows this improves LLM performance on sequential workflows.

### 4. **Handoff Protocol**
When work passes to another person/session:
1. Read Layer 1 (e.g., `research/CONTEXT.md`) — understand the landscape
2. Read Layer 2 (e.g., `research/HYP-093/CONTEXT.md`) — understand this specific task
3. Read Layer 3 (e.g., `_config/gate_functions.md`) — learn the rules once
4. Open Layer 4 (`output/`) — see the actual deliverables
5. No context outside these four layers is necessary

---

## How This Applies to Quant Repo

### Research Workflow (Pre-ICM → Post-ICM)

**Before ICM:**
- Hypotheses scattered across markdown files (HYP-071_VALIDATION_REPORT.md, HYP-108_method_falsification.md)
- No clear spec of what "validate" means
- Unclear what next stage expects as input
- Backtest code + rules mixed together
- Hard to hand off or parallelize

**After ICM:**
- Each hypothesis gets a folder: `research/HYP-NNN/CONTEXT.md` + outputs/
- Layer 2 CONTEXT.md defines the full stage (inputs → process → outputs → review gate)
- Layer 3 rules are external (gate_functions, sizing_model)
- Layer 4 outputs are structured (JSON for machines, markdown for humans)
- Easy to hand off, audit, or parallelize multiple hypotheses

### Oracle Cycle (Pre-ICM → Post-ICM)

**Before ICM:**
- Reflection scattered across launchd + Python scripts
- Unclear when each stage completes
- Unclear what metrics matter for decision-making
- Changes applied live without holdout validation
- Hard to trace why a change was made or rejected

**After ICM:**
- Oracle is three explicit stages: harvest → reflect → test
- Each stage has clear inputs (Layer 4 from previous stage), process, outputs
- Lessons are proposed (reflect stage) then validated (test stage) before live
- Every change is logged with rationale
- Easy to rewind, audit, or modify the cycle

### Live Trading (Enhanced by ICM)

**Decision logging:**
- Each entry logs commitment_score, rate_diff, library_match, conviction
- Schema defined in `shared/decision_logger_schema.md`
- Oracle reads these logs and learns

**Hypothesis ledger:**
- Central record of all hypotheses: LIVE, IN_RESEARCH, GRAVEYARD
- Schema in `shared/hypothesis_ledger_schema.md`
- Queries: "Which hypotheses are live? Which failed recently? Which are in mechanism validation?"

**Risk constitution:**
- Moved to Layer 3 (`_config/risk_constitution.md`) — sacred, immutable within a cycle, changes are deliberate
- All sizing code references this, not hardcoded values

---

## First Run: HYP-093 (Gapper Fade)

**Status:** Ready to execute  
**Stage:** mechanism_validation  
**CONTEXT.md:** Specifies exactly what "validate the gapper fade mechanism" means

**To run:**
1. Read `research/HYP-093/CONTEXT.md` top-to-bottom
2. Execute the process (load data, descriptive analysis, statistical test, data quality check)
3. Write outputs to `output/`:
   - `mechanism_validation.md` (human-readable report)
   - `data_quality_check.json` (machine-readable gate)
   - `backtest_spec.md` (frozen contract for next stage)
   - `analysis.json` (ledger metadata)
4. Human review: read `mechanism_validation.md`, approve or reject
5. If approve: advance to next stage (hypothesis_testing)
6. If reject: close hypothesis (update ledger)

**Expected duration:** 2–3 days (mechanism validation only; full backtest comes next stage)

---

## Oracle Cycle: First Run

**Status:** Ready to automate  
**Frequency:** Daily at 02:30 UTC (after US market close)  
**Duration:** ~15 minutes (harvest: 5 min, reflect: 5 min, test: 5 min)

**To run manually (for testing):**
```bash
# Stage 00: Harvest
python sovereign/autonomous/oracle_cycle.py --stage 00-harvest

# Stage 01: Reflect (reads output of harvest)
python sovereign/autonomous/oracle_cycle.py --stage 01-reflect

# Stage 02: Test (reads output of reflect)
python sovereign/autonomous/oracle_cycle.py --stage 02-test
```

**Outputs appear in:**
- `oracle/00-harvest/output/` — decision_logs_summary.json, market_state.json
- `oracle/01-reflect/output/` — lessons.json, anomalies.json
- `oracle/02-test/output/` — test_verdict.json
- `data/agent/oracle_ledger.jsonl` — appended with daily verdict

---

## Non-Negotiable Rules (Enforced by ICM)

1. **Spec-first** — Layer 2 CONTEXT.md must be written before execution
2. **Reference, don't repeat** — Layer 2 references Layer 3, doesn't restate it
3. **Output + ledger** — Every stage produces deliverables + ledger entry
4. **Closed loop** — Every decision_logger.log() must get update_outcome()
5. **No silent mocking** — Missing data/infra = stop and report, never hand-wave

These are not guidelines; they're structural consequences of the ICM design.

---

## Deliverables Summary

| Layer | What | Path | Status |
|-------|------|------|--------|
| **0** | Global identity | CLAUDE.md (updated) | ✅ Done |
| **1** | Research router | research/CONTEXT.md | ✅ Done |
| **1** | Oracle router | oracle/CONTEXT.md | ✅ Done |
| **2** | HYP-093 stage | research/HYP-093/CONTEXT.md | ✅ Done |
| **2** | Oracle harvest | oracle/00-harvest/CONTEXT.md | ✅ Done |
| **2** | Oracle reflect | oracle/01-reflect/CONTEXT.md | ✅ Done |
| **2** | Oracle test | oracle/02-test/CONTEXT.md | ✅ Done |
| **3** | Reference factory | `_config/` + `shared/` | ✅ Done (8 files) |
| **4** | HYP-093 output | research/HYP-093/output/ (ready) | ✅ Scaffolded |
| **4** | Oracle outputs | oracle/*/output/ (ready) | ✅ Scaffolded |

---

## Next Steps

### Immediate (This week)
1. **Run HYP-093 mechanism validation** (2–3 days)
   - Execute the process in `research/HYP-093/CONTEXT.md`
   - Produce outputs in `output/`
   - Human review + advance or reject

2. **Test Oracle cycle manually** (1 day)
   - Dry-run all three stages on recent trade data
   - Verify outputs are correct format
   - Validate verdicts make sense

### Short-term (Next 2 weeks)
3. **Automate Oracle via launchd** (1 day)
   - Wire up `com.alta.oracle_cycle` launchd job
   - Run at 02:30 UTC daily
   - Monitor first week of outputs

4. **Migrate other active hypotheses to ICM** (3 days)
   - HYP-071 (Staged Exit Rule) → `research/HYP-071/CONTEXT.md`
   - HYP-096 (Short Call Verticals) → `research/HYP-096/CONTEXT.md`
   - Each gets Layer 2 spec + Layer 4 output folders

5. **Update hypothesis ledger** (1 day)
   - Populate `data/hypotheses_ledger.jsonl` with all current + historical hypotheses
   - Establish single source of truth for hypothesis status

### Medium-term (Next month)
6. **Formalize Layer 3 amendments** (ongoing)
   - Any change to gate_functions, sizing_model, etc. gets dated amendment block
   - Ratification trail in CLAUDE.md

7. **Build queries** (1 day)
   - "Which hypotheses are live?" — query hypotheses_ledger.jsonl
   - "Show me yesterday's trades" — query decision_logs/
   - "Did any hypothesis degrade this week?" — scan oracle_ledger.jsonl

---

## Validation Checklist

**Before declaring ICM "live":**

- [ ] HYP-093 mechanism validation completes and produces all four outputs
- [ ] Human can read mechanism_validation.md and understand the entire analysis
- [ ] Backtest spec is unambiguous (next stage knows exactly what to do)
- [ ] Oracle harvest/reflect/test runs once end-to-end
- [ ] Each stage produces expected outputs
- [ ] Verdicts make sense (no spurious APPLY_LIVE or REJECT recommendations)
- [ ] CLAUDE.md ICM section is clear (new session reads it, understands the structure)
- [ ] No ambiguity in handoff protocol (anyone can take over HYP-093 at any stage)

---

## References

**Paper:** Van Clief & McDermott (2026) "Interpretable Context Methodology: Folder Structure as Agent Architecture"  
**arXiv:** 2603.16021v2  
**Key insight:** Folder hierarchy + markdown files replace framework-level orchestration. Simpler, more auditable, easier to modify at any stage.

**Implementation notes:**
- Five-layer pattern maps directly to quant repo structure (CLAUDE.md ← Layer 0, NEXT.md ← Layer 1, etc.)
- Token efficiency: 2,000–8,000 per stage vs. 30,000–50,000 monolithic
- Human oversight enabled by explicit stage contracts (CONTEXT.md files)
- Versioning is automatic (git tracks all folders and outputs)

---

## Commit Message

```
[ICM] Implement Interpretable Context Methodology for research & Oracle workflows

- Layer 0: Update CLAUDE.md with ICM section
- Layer 1: Add research/CONTEXT.md router + oracle/CONTEXT.md workflow
- Layer 2: Create HYP-093 stage definition + three Oracle stages (harvest, reflect, test)
- Layer 3: Build factory (_config/ + shared/): gate functions, sizing model, ledger schemas
- Layer 4: Scaffold output folders for HYP-093 + Oracle stages (awaiting first runs)

Reference: arXiv 2603.16021v2 (Van Clief & McDermott)
Enables: Structured hypothesis testing, daily Oracle reflection, multi-person handoffs
Impact: Token efficiency (8k/stage vs 50k monolithic), human-readable audits, parallel work

Next: Execute HYP-093 mechanism validation, test Oracle cycle, migrate HYP-071/HYP-096
```

---

## Questions or Issues?

- **"How do I add a new hypothesis?"** → Read `research/CONTEXT.md` (Layer 1)
- **"What goes in HYP-* outputs?"** → Read `research/HYP-093/CONTEXT.md` (Layer 2)
- **"What are the gate criteria?"** → Read `_config/gate_functions.md` (Layer 3)
- **"How does the Oracle work?"** → Read `oracle/CONTEXT.md` (Layer 1)
- **"Where's the ledger?"** → `data/hypotheses_ledger.jsonl` (Layer 4 factory, appended by stages)

All information is filesystem-navigable. No hidden state. No implicit context.
