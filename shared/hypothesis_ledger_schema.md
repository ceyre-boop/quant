# Hypothesis Ledger Schema

**The ledger is the formal record of every hypothesis tested. Its structure is machine-readable.**

*Last updated: 2026-08-12*

---

## Purpose

Every hypothesis in research enters the ledger when testing begins. The ledger records:
1. What was tested (hypothesis_id, description)
2. When it was tested (date range)
3. What the evidence says (verdict, Sharpe, p-value, OOS degradation)
4. What stage it reached (mechanism → hypothesis_test → deployment_readiness → LIVE or GRAVEYARD)
5. Linked folder (CONTEXT.md and outputs)

The ledger is permanent. Entries are never deleted. Graveyarded hypotheses remain searchable.

---

## Schema (Machine-Readable)

Each hypothesis is one JSON object in `data/hypotheses_ledger.jsonl`:

```json
{
  "hypothesis_id": "HYP-093",
  "title": "Yield Frontier — Gapper Fade Structural Reversion",
  "description": "Intraday gap closure on short-term maturity options generates positive P&L by selling IV into gap-risk premium.",
  
  "created_date": "2026-07-13",
  "modified_date": "2026-08-12",
  
  "status": "IN_RESEARCH",
  "stage": "mechanism_validation",
  "expected_next_stage": "hypothesis_testing",
  
  "evidence": {
    "verdict": null,
    "mechanism_confirmed": false,
    "data_quality_gate": "PENDING",
    "mechanism_p_value": null,
    "mechanism_median_fade_pct": null
  },
  
  "backtest_results": {
    "in_sample_sharpe": null,
    "out_of_sample_sharpe": null,
    "permutation_p_value": null,
    "oos_degradation_pct": null,
    "gates_passed": []
  },
  
  "related_hypotheses": ["HYP-096", "HYP-085"],
  "related_systems": ["yield_frontier"],
  "context_folder": "research/HYP-093/",
  "decision_log_filter": "gapper_fade",
  
  "notes": "Awaiting mechanism validation. Pre-spec written. Gapper universe sourced.",
  "assigned_to": "colin"
}
```

---

## Field Reference

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `hypothesis_id` | String | ✅ | Format: HYP-NNN (e.g., HYP-093). Globally unique. |
| `title` | String | ✅ | One-line hypothesis name. |
| `description` | String | ✅ | 2–3 sentence hypothesis statement. |
| `created_date` | ISO8601 | ✅ | When hypothesis was first logged. |
| `modified_date` | ISO8601 | ✅ | Last update date. |
| `status` | Enum | ✅ | IN_RESEARCH, LIVE, GRAVEYARD |
| `stage` | String | ✅ | mechanism_validation, hypothesis_testing, deployment_readiness, live_monitor |
| `expected_next_stage` | String | ⚠️ | What stage comes next if this one PASSes. |
| `evidence.verdict` | Enum | ❌ | PASS, FAIL, or null if in progress. |
| `evidence.mechanism_confirmed` | Boolean | ❌ | Is the underlying mechanism real (p<0.05)? |
| `evidence.mechanism_p_value` | Float | ❌ | P-value from mechanism validation. |
| `backtest_results.in_sample_sharpe` | Float | ❌ | In-sample Sharpe from backtest. |
| `backtest_results.out_of_sample_sharpe` | Float | ❌ | OOS Sharpe from walk-forward. |
| `backtest_results.permutation_p_value` | Float | ❌ | P-value from permutation test. |
| `backtest_results.oos_degradation_pct` | Float | ❌ | (IS - OOS) / IS as %. |
| `backtest_results.gates_passed` | Array | ❌ | List of gate names that passed (e.g., ["sharpe_30", "permutation_p05"]). |
| `related_hypotheses` | Array | ⚠️ | Linked HYP-* IDs (sister/competing hypotheses). |
| `related_systems` | Array | ⚠️ | Which live systems would use this (forex, ict, yield, etc.). |
| `context_folder` | Path | ✅ | Relative path to research/HYP-*/CONTEXT.md and outputs. |
| `decision_log_filter` | String | ❌ | Filter key to isolate live trades for this hypothesis (for Oracle). |
| `notes` | String | ⚠️ | Human-readable status notes (blockers, decisions, rationale). |
| `assigned_to` | String | ⚠️ | Who is currently working on this. |

---

## Lifecycle

### Stage 1: mechanism_validation
**Check:** Is the underlying edge real (not just random noise)?

```json
{
  "status": "IN_RESEARCH",
  "stage": "mechanism_validation",
  "evidence": {
    "verdict": "PASS",
    "mechanism_confirmed": true,
    "mechanism_p_value": 0.012,
    "mechanism_median_fade_pct": 0.8
  }
}
```

**Advance to:** hypothesis_testing
**Retreat to:** GRAVEYARD (if mechanism p-value ≥ 0.05)

---

### Stage 2: hypothesis_testing
**Check:** Does the strategy beat random entry/exit? Sharpe ≥ 0.30? OOS Sharpe within 20% of IS?

```json
{
  "status": "IN_RESEARCH",
  "stage": "hypothesis_testing",
  "backtest_results": {
    "in_sample_sharpe": 0.95,
    "out_of_sample_sharpe": 0.78,
    "permutation_p_value": 0.018,
    "oos_degradation_pct": 18,
    "gates_passed": ["sharpe_30", "permutation_p05", "oos_degradation_20"]
  }
}
```

**Advance to:** deployment_readiness (if all gates pass)
**Retreat to:** GRAVEYARD (if any gate fails)

---

### Stage 3: deployment_readiness
**Check:** Is sizing model calibrated? Are live trade rules exact and unambiguous? Is an Oracle filter ready?

```json
{
  "status": "IN_RESEARCH",
  "stage": "deployment_readiness",
  "notes": "Sizing calibrated (RCK conviction model). Live trade rules written and reviewed. Oracle filter ready."
}
```

**Advance to:** LIVE (human sign-off)
**Retreat to:** hypothesis_testing (if issues found)

---

### Stage 4: LIVE
**Check:** Running live. Daily monitoring via Oracle reflection cycle.

```json
{
  "status": "LIVE",
  "stage": "live_monitor",
  "notes": "Live since 2026-07-13. 47 trades YTD. Sharpe 1.15 on 60d window."
}
```

---

### GRAVEYARD
**Check:** Hypothesis failed a gate. Formally recorded. Never re-test without new evidence.

```json
{
  "status": "GRAVEYARD",
  "stage": "archived",
  "evidence": {
    "verdict": "FAIL",
    "failure_reason": "Mechanism p-value 0.52 (no edge)"
  },
  "notes": "Tested 2026-08-10. Gapper fade showed no predictive power (median close within noise). Do not re-explore without new theory."
}
```

---

## Amendment History

- **2026-08-12** — Schema formalized for ICM integration. Decision_log_filter added (for Oracle queries).
- **2026-07-13** — Initial version (ledger created for hypothesis tracking).

---

## File Location

```
data/hypotheses_ledger.jsonl
```

Each line is one hypothesis JSON object (no line breaks within objects). Append-only.

**Example query (shell):**
```bash
# Find all LIVE hypotheses
cat data/hypotheses_ledger.jsonl | jq 'select(.status == "LIVE")'

# Find all hypotheses in yield_frontier system
cat data/hypotheses_ledger.jsonl | jq 'select(.related_systems[] | select(. == "yield_frontier"))'

# Find failures from last 30 days
cat data/hypotheses_ledger.jsonl | jq "select(.modified_date > \"2026-07-13\" and .status == \"GRAVEYARD\")"
```
