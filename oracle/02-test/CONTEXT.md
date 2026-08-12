# Oracle Stage 02: Test Lessons on Holdout Data

**Layer 2: Validate proposed changes before live implementation**

*Last updated: 2026-08-12*

---

## Purpose

The reflection stage proposed changes (e.g., "increase sizing for rate_diff > 140 bps"). Before applying them live, test on holdout data to ensure they don't hurt.

**Runs at:** 02:45 UTC (after reflection completes)

---

## Inputs (Layer 3 + Layer 4)

### Layer 3 (Reference)
- `_config/gate_functions.md` — Sharpe gate (must pass before live implementation)
- `_config/sizing_model.md` — Sizing rules (any change must stay within caps)

### Layer 4 (Working artifacts)
- `../01-reflect/output/lessons.json` — Proposed changes from yesterday
- `data/hypotheses_ledger.jsonl` — Historical trades for holdout testing
- `data/backtests/holdout/` — Reserved unseen data (never trained on)

---

## Process

### Step 1: For Each Lesson, Design a Test (Automated + AI, ~5 min)

**Question:** If we apply this proposed change, will it hurt or help on holdout data?

**Example:** Lesson says "increase rate_diff > 140 sizing from 0.75 to 0.85"

**Test design:**
1. Filter holdout trades where: hypothesis == HYP-045 AND rate_diff > 140 bps
2. Split: 50% with old sizing (0.75), 50% with new sizing (0.85)
3. Compute Sharpe for each group
4. Did new sizing outperform? Or was it random?

**Output:** `test_design.json`
```json
{
  "date": "2026-08-12",
  "lesson_id": "lesson-001",
  "hypothesis_id": "HYP-045",
  "proposed_change": "Increase rate_diff > 140 sizing: 0.75 → 0.85",
  "holdout_test_design": {
    "filter": "hypothesis=HYP-045 AND rate_diff_bps > 140",
    "holdout_trades_available": 34,
    "test_method": "A/B split (50/50 old/new sizing)",
    "comparison_metric": "Sharpe ratio",
    "gate": "Sharpe with new sizing must be >= old sizing - 5% (no more than 5% degradation)"
  }
}
```

### Step 2: Run Holdout Test (Automated, ~10 min)

**Actions:**
1. For each proposed lesson:
   - Filter holdout data to matching trades
   - Re-run backtest with proposed change applied
   - Compare Sharpe (new) vs Sharpe (old)
   - Check: does new sizing fit within Article 1 & 2 caps?
2. Compute p-value (is the improvement real, or random?)
3. Flag any that degrade performance

**Output:** `test_results.json`
```json
{
  "date": "2026-08-12",
  "test_results": [
    {
      "lesson_id": "lesson-001",
      "hypothesis_id": "HYP-045",
      "proposed_change": "rate_diff > 140: sizing 0.75 → 0.85",
      "holdout_sharpe_old": 1.12,
      "holdout_sharpe_new": 1.19,
      "sharpe_improvement_pct": 6.3,
      "p_value": 0.18,
      "verdict": "MARGINAL_IMPROVEMENT",
      "interpretation": "New sizing improves Sharpe by 6%, but p=0.18 (not significant). Recommend: trial on next 10 live trades.",
      "risk_check": {
        "article_1_cap_respected": true,
        "article_2_cap_respected": true,
        "max_position_size_new": 627.50,
        "cap_allowed": 750
      }
    }
  ]
}
```

### Step 3: Generate Ledger Entry & Recommendations (Automated, ~2 min)

**Actions:**
1. For each tested lesson:
   - If holdout Sharpe ≥ gate threshold: recommend APPLY_LIVE
   - If holdout Sharpe < gate threshold: recommend REJECT or TRIAL
2. Write recommendation with rationale
3. Record in Oracle ledger

**Output:** `test_verdict.json`
```json
{
  "date": "2026-08-12",
  "recommendations": [
    {
      "lesson_id": "lesson-001",
      "hypothesis_id": "HYP-045",
      "proposed_change": "rate_diff > 140: sizing 0.75 → 0.85",
      "verdict": "APPLY_LIVE_TRIAL",
      "rationale": "Holdout test shows 6% Sharpe improvement. Not significant (p=0.18), but no degradation risk. Safe to trial on next 10 live trades.",
      "next_action": "Monitor HYP-045 rate_diff > 140 entries for next 5 days. If Sharpe stays >1.0, make change permanent."
    }
  ]
}
```

---

## Outputs

Write to `output/`:

1. **`test_design.json`** — Test methodology for each proposed change
2. **`test_results.json`** — Holdout Sharpe (old vs new), p-value, verdict
3. **`test_verdict.json`** — Final recommendations (APPLY_LIVE, TRIAL, or REJECT)
4. **`oracle_ledger_entry.json`** — Timestamp, all verdicts, summary

---

## Success Criteria

✅ **All proposed lessons have been tested** (no untested changes)  
✅ **No holdout test shows degradation** (no recommended changes hurt performance)  
✅ **Risk checks pass** (Article 1 & 2 caps still respected under new sizing)  
✅ **Verdicts are clear** (APPLY_LIVE, TRIAL, or REJECT — no ambiguous cases)

---

## Failure Modes

❌ **Holdout test shows degradation** (e.g., Sharpe drops 10%)  
   → Verdict = REJECT (do not apply)  
   → Log in Oracle ledger (lesson failed validation)  
   → Recommendation: investigate why reflection was wrong (data artifact? regime shift?)

❌ **Risk check fails** (new sizing exceeds caps)  
   → Verdict = REJECT  
   → Lesson itself was violating risk constitution; reframe

---

## Next Stage

If verdict = APPLY_LIVE: change goes into effect at next market open.

If verdict = TRIAL: monitor (next 5 days) before permanent implementation.

If verdict = REJECT: proposed change is discarded; reflect on why lesson was incorrect.

---

## Non-Negotiable

- **No change is live without holdout test.** Reflection is blind without validation.
- **Risk caps are sacred.** Any proposed change that touches Article 1, 2, or 3 must pass risk checks first.
- **P-values matter.** Do not apply a change based on 2 trades, even if holdout looks good.
