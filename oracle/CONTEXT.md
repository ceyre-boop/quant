# Oracle Cycle (Layer 1: CONTEXT.md)

**The daily reflection workflow that learns from trades and improves sizing.**

*Last updated: 2026-08-12*

---

## Purpose

Every morning at 02:30 UTC, the Oracle reads yesterday's trades and market data, reflects on what worked and what didn't, proposes improvements, and tests them on holdout data before live implementation.

**Cycle time:** ~15 minutes  
**Frequency:** Daily (Mon–Fri, after US market close)  
**Automated:** Yes, via launchd + Python orchestrator

---

## The Three Stages

```
[Day N: 21:00 UTC, market closes]
         ↓
[Day N: 02:30 UTC, Oracle wakes up]
         ↓
Stage 00: Harvest
  • Read decision logs from Day N
  • Read market state, macro events
  • Extract trade summary + regime classification
  • Output: decision_logs_summary.json, market_state.json
         ↓
Stage 01: Reflect  [2:35 UTC]
  • Analyze: did commitment_score predict outcome?
  • Analyze: did regime fit?
  • Identify anomalies (surprising wins/losses)
  • Propose changes (e.g., "increase sizing for rate_diff > 140")
  • Output: lessons.json with proposed changes
         ↓
Stage 02: Test  [2:45 UTC]
  • For each proposed change, run holdout backtest
  • Check: does change improve Sharpe or hurt it?
  • Verify: does change respect Article 1 & 2 caps?
  • Output: verdicts (APPLY_LIVE, TRIAL, or REJECT)
         ↓
[If verdicts approved: changes go live at next market open]
[If verdicts rejected: discard, wait for next day's data]
```

---

## Key Principle

**The Oracle learns incrementally, never catastrophically.**

- No single day's reflection causes a large change
- All proposed changes are tested on holdout data first
- If a change doesn't help, it's rejected (no harm)
- If a change helps, it's trialed on 5–10 live trades before permanent
- All decisions are reversible within 24 hours

---

## Stage Structure (ICM Pattern)

Each of the three stages is self-contained:

### Stage 00: Harvest
- **CONTEXT.md** — Defines what "harvest" means (read logs, read market state, extract summaries)
- **references/** — Layer 3 (decision_logger schema, macro catalog)
- **output/** — Layer 4 (decision_logs_summary.json, market_state.json)

### Stage 01: Reflect
- **CONTEXT.md** — Defines what "reflect" means (correlate metrics, check regimes, identify anomalies)
- **references/** — Layer 3 (trading philosophy, decision schema)
- **output/** — Layer 4 (reflection_metrics.json, anomalies.json, lessons.json)

### Stage 02: Test
- **CONTEXT.md** — Defines what "test" means (holdout backtest, gate checking, verdict generation)
- **references/** — Layer 3 (gate functions, sizing model, risk constitution)
- **output/** — Layer 4 (test_results.json, test_verdict.json)

Each stage reads from Layer 3 reference material and produces Layer 4 outputs that feed into the next stage.

---

## Handoff Between Stages

**Stage 00 → Stage 01:**
- Harvest output (decision logs, market state) goes to reflect input

**Stage 01 → Stage 02:**
- Reflection output (proposed lessons) goes to test input

**Stage 02 → Live:**
- Test verdicts (APPLY_LIVE) go into effect

---

## Inputs (Layer 3 Reference Material)

All three stages read from:
- `../_config/trading_philosophy.md` — Decision-making framework
- `../_config/risk_constitution.md` — Capital preservation rules (sacred)
- `../_config/gate_functions.md` — Sharpe gate, permutation test
- `../shared/decision_logger_schema.md` — Trade log schema
- `../_config/market_regimes.md` — Regime classification rules

Do not repeat these in individual stage CONTEXT.md files. Reference them.

---

## Outputs (What gets built each day)

Each day produces:
1. `00-harvest/output/decision_logs_summary.json` — Trade metrics by hypothesis
2. `01-reflect/output/lessons.json` — Proposed changes with rationale
3. `02-test/output/test_verdict.json` — Go/no-go on each proposal
4. `oracle_ledger.jsonl` (appended) — Daily entry recording all verdicts

---

## Oracle Ledger

The oracle keeps a permanent record of every day's reflection:

```json
{
  "date": "2026-08-12",
  "cycle_id": "oracle-2026-08-12",
  "stage_00_harvest": {
    "trades_completed": 4,
    "total_pnl_usd": 1847.30,
    "hypothesis_breakdown": { ... }
  },
  "stage_01_reflect": {
    "lessons_proposed": 3,
    "anomalies_found": 1
  },
  "stage_02_test": {
    "verdicts_approved": 2,
    "verdicts_rejected": 1
  },
  "changes_applied_today": [
    {
      "hypothesis_id": "HYP-045",
      "change": "rate_diff > 140: sizing 0.75 → 0.85",
      "applied_at": "2026-08-13T12:00:00Z"
    }
  ]
}
```

Location: `data/agent/oracle_ledger.jsonl` (append-only)

---

## Automation

**Orchestrator:** `sovereign/autonomous/oracle_cycle.py`  
**Trigger:** launchd (`com.alta.oracle_cycle`, daily 02:30 UTC)  
**Execution:**
1. Run stage 00 (harvest)
2. Wait for completion
3. Run stage 01 (reflect)
4. Wait for completion
5. Run stage 02 (test)
6. If all passed: append to oracle_ledger.jsonl
7. If any stage fails: send alert email, stop (no automatic retry until next day)

---

## Manual Overrides

**If Colin wants to override the Oracle:**
1. Edit `data/system/KILL_SWITCH.json` to freeze the cycle
2. Manually edit stage outputs (e.g., approve a rejected verdict)
3. Commit the change with explanation
4. At next market open, the change goes live

**Example:**
```json
{
  "frozen": true,
  "reason": "Testing new rate_diff threshold manually. Do not run oracle cycle until 2026-08-15.",
  "frozen_by": "Colin",
  "frozen_at": "2026-08-12 15:30 ET"
}
```

---

## Non-Negotiable Rules

1. **No live change without holdout test** (stage 02 must approve)
2. **No change to core rules** (trading philosophy, risk constitution) without explicit ratification
3. **All proposed changes are logged** (even rejected ones go in the ledger)
4. **Cycle runs every day** (missing a day is a gap; data loss)
5. **No manual trades during 02:30–03:00 UTC** (Oracle run time; could create conflicts)

---

## Amendment History

- **2026-08-12** — Formalized for ICM. Three-stage structure, daily reflection loop documented.
- **2026-07-01** — Initial oracle cycle shipped (pre-ICM structure).

---

## Next Steps

1. **Run Stage 00 tomorrow** (2026-08-13 02:30 UTC) — first harvest of the new structure
2. **Monitor outputs** (check oracle/ folder for daily results)
3. **Review Stage 02 verdicts** (should see APPLY_LIVE or TRIAL recommendations within 48 hours)

---

## Getting Help

- **Stage 00 failing?** → Check decision_logger logs; may need backfill
- **Stage 01 giving wrong lessons?** → Probably n too small (need 10+ trades for statistical confidence)
- **Stage 02 rejecting good ideas?** → Holdout test may be too strict; review gate criteria
- **Cycle timing off?** → Check launchd plist timing and log paths
