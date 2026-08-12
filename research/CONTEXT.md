# Research Workflow (Layer 1: CONTEXT.md)

**This file routes research work through hypothesis discovery, validation, and deployment.**

*Last updated: 2026-08-12*

---

## Purpose

The research/ directory contains all hypothesis testing. Each hypothesis gets its own folder with:
- A CONTEXT.md file defining the hypothesis and its validation stages
- A references/ folder with Layer 3 rules relevant to this hypothesis
- An output/ folder with results, verdict, and ledger entry

This CONTEXT.md (Layer 1) routes work to the right hypothesis folder.

---

## Workflow Map

```
Research Task
  ↓
1. Pick or create hypothesis (HYP-NNN)
  ↓
2. Navigate to research/HYP-NNN/
  ↓
3. Read research/HYP-NNN/CONTEXT.md (Layer 2 stage definition)
  ↓
4. Execute the stage (process, outputs)
  ↓
5. Human review at gate (PASS/FAIL)
  ↓
6. [If PASS] Advance to next stage (update CONTEXT.md)
  ↓
7. [If FAIL] Close hypothesis (update hypothesis_ledger.jsonl)
  ↓
8. Commit results + update root NEXT.md
```

---

## Current Hypotheses

### Active (IN_RESEARCH)

- **HYP-093** — Yield Frontier (Gapper Fade)
  - Status: mechanism_validation stage
  - Folder: `research/HYP-093/`
  - Assigned: colin
  - Next: Validate that gappers fade >0.3% median by 10:30 ET

- **HYP-071** — Staged Exit Rule (L2 Exit Value Function)
  - Status: hypothesis_testing stage (provisional pass on gross-R)
  - Folder: `research/HYP-071/`
  - Blocker: Step 2 recompute needed; awaiting Colin seal
  - Next: Finalize net-return backtest

### LIVE

- **HYP-045** — Rate Differential Edge (Forex Carry)
  - Status: live_monitor
  - System: Forex v015 (4-pair portfolio: GBPUSD, EURUSD, AUDUSD, GBPJPY)
  - Evidence: OOS Sharpe 1.25, permutation p<0.001, CONFIRMED
  - Ledger: `data/hypotheses_ledger.jsonl` line ~1400

### GRAVEYARD (Do NOT revisit without new theory)

- **HYP-089** — 12-month TSMOM
  - Verdict: NOT_SIGNIFICANT (Sharpe 0.277 < 0.30 gate)
  - Closed: 2026-07-12

- **HYP-090** — MODERN (daily adaptive params)
  - Verdict: NOT_SIGNIFICANT (random placebo better)
  - Closed: 2026-07-11

- **HYP-085** — Political Alpha (news-driven)
  - Verdict: NOT_SIGNIFICANT (p=0.3637)
  - Closed: 2026-07-08

(See `data/hypotheses_ledger.jsonl` for full graveyard.)

---

## Creating a New Hypothesis

1. **Assign an ID** — `HYP-NNN` where NNN is the next unused number
2. **Create folder** — `mkdir research/HYP-NNN/`
3. **Write hypothesis statement** — 2–3 sentences in `research/HYP-NNN/CONTEXT.md`
4. **Add to Layer 1** — Update this file's "Active" section
5. **Log in ledger** — Append to `data/hypotheses_ledger.jsonl`:
   ```json
   {
     "hypothesis_id": "HYP-NNN",
     "title": "...",
     "status": "IN_RESEARCH",
     "stage": "mechanism_validation",
     "created_date": "YYYY-MM-DD",
     "context_folder": "research/HYP-NNN/"
   }
   ```
6. **Commit** — Push the new hypothesis folder structure

---

## Next Steps (What to work on next)

**By priority:**

1. **HYP-093 mechanism validation** (blocking HYP-096)
   - Read `research/HYP-093/CONTEXT.md`
   - Validate gapper fade mechanism: median fade >0.3%, p<0.05
   - ETA: 2026-08-15

2. **HYP-071 Step 2 recompute** (blocking deployment)
   - Recompute gross-R net-return for exit value function
   - Awaiting Colin seal on net-R interpretation
   - ETA: 2026-08-13

3. **Oracle reflection cycle** (ongoing)
   - Daily 02:30 ET: read decision_logs, reflect, write lessons
   - Check: are all trade closures getting update_outcome() calls?
   - Monitor: hypothesis_id-specific Sharpe degradation (early warning)

---

## Reference Material (Layer 3)

All hypotheses read from:
- `_config/trading_philosophy.md` — Six tenets (statistical utility, regime appropriateness, system health, etc.)
- `_config/risk_constitution.md` — Capital preservation (Article 1–6)
- `_config/gate_functions.md` — Mechanical gates (Sharpe, permutation, OOS degradation)
- `_config/sizing_model.md` — Conviction-based sizing, carry heat caps
- `shared/hypothesis_ledger_schema.md` — How to record verdicts
- `shared/decision_logger_schema.md` — How to log every trade entry/exit

Do NOT repeat these in individual hypothesis CONTEXT.md files. Reference them.

---

## Handoff Protocol

**When handing off a hypothesis to another person or session:**

1. Read this file (Layer 1) to understand the landscape
2. Navigate to the hypothesis folder (e.g., `research/HYP-093/`)
3. Read its CONTEXT.md (Layer 2) top-to-bottom
4. Look at Layer 3 references it cites
5. Open the output/ folder and inspect intermediate results
6. No context outside these files is necessary

**If a hypothesis is blocked:**
- Blocker is stated in "Current Hypotheses" section above
- Go to `research/HYP-NNN/output/blockers.txt` for details
- Example: HYP-071 blocked on "Colin decision on gross vs net interpretation"

---

## Ledger Integration

The master ledger lives in `data/hypotheses_ledger.jsonl`. Update it when:
- Hypothesis status changes (IN_RESEARCH → LIVE or GRAVEYARD)
- Evidence arrives (backtest_results, verdict)
- Stage advances (mechanism_validation → hypothesis_testing)

Query the ledger:
```bash
# All LIVE hypotheses
cat data/hypotheses_ledger.jsonl | jq 'select(.status == "LIVE")'

# Recent failures
cat data/hypotheses_ledger.jsonl | jq 'select(.modified_date > "2026-08-05" and .status == "GRAVEYARD")'
```

---

## Non-Negotiable Rules

1. **Spec-first.** Write the stage CONTEXT.md BEFORE running the backtest. The spec is the contract.
2. **Output + Ledger.** Every hypothesis that runs gets a verdict logged. No ambiguous "pending" entries.
3. **Close the Oracle loop.** Every live entry gets decision_logger.log(). Every exit gets update_outcome().
4. **No silent mocking.** If data is missing or infrastructure fails: log it, stop, report. Don't hand-wave.
5. **Reference, don't repeat.** Individual hypothesis CONTEXT.md files reference Layer 3 rules, not restate them.

---

## Amendment History

- **2026-08-12** — Layer 1 formalized for ICM integration. Workflow map added.
- **2026-07-13** — Initial version (research directory organized).
