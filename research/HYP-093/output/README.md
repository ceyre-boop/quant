# HYP-093 Output (Layer 4)

**Layer 4 holds working artifacts — outputs unique to this run.**

These files are generated during hypothesis testing and consumed by the next stage (or by the human reviewer).

## Expected Deliverables (From CONTEXT.md)

At the end of the mechanism_validation stage, this folder should contain:

### 1. `mechanism_validation.md`
Human-readable analysis report.
- Plots of gapper fade vs non-gappers
- Quantile analysis (do larger gappers fade more?)
- Volatility profile (do 5 DTE IVs collapse by close?)
- Statistical test result (sign test p-value)
- Conclusion: mechanism is REAL / WEAK / ARTIFACT

### 2. `data_quality_check.json`
Machine-readable data quality gate result.
```json
{
  "polygon_latency_max_seconds": 2.1,
  "missing_5min_bars": 14,
  "ssr_halted_events": 847,
  "status": "PASS"
}
```

### 3. `backtest_spec.md`
Frozen specification for the next stage (hypothesis_testing).
- Exact entry rules, exit rules, sizing model
- Hashed (sha256) for integrity tracking
- DO NOT modify after human review; advancing stages must follow this spec exactly

### 4. `analysis.json`
Metadata for ledger handoff.
```json
{
  "hypothesis_id": "HYP-093",
  "stage": "mechanism_validation",
  "mechanism_verdict": "REAL",
  "mechanism_p_value": 0.012,
  "next_stage": "hypothesis_testing",
  "ready_to_backtest": true
}
```

### 5. `human_review.md`
Human reviewer's approval or rejection.
```markdown
Reviewer: Colin  
Date: 2026-08-12  
Verdict: APPROVE  
Notes: Median fade is real; spot-check confirmed.
Signature: Colin Eyre, 2026-08-12 15:30 ET
```

---

## Handoff Protocol

**When advancing to the next stage:**

1. Human reviews this folder's contents
2. Reads `mechanism_validation.md` end-to-end
3. Spot-checks key results (plots, p-value, confidence intervals)
4. Signs off in `human_review.md`
5. Commits all files + updates `../../data/hypotheses_ledger.jsonl`

**Next stage reads:**
- `backtest_spec.md` (what to test)
- `analysis.json` (stage completion metadata)

---

## Non-Negotiable Rules

1. **Layer 4 is mutable per-run** — each stage produces its own outputs
2. **Layer 4 is consumed by the next stage** — outputs of stage N become inputs to stage N+1
3. **Layer 4 is auditable** — every file must be reproducible from the Layer 2 CONTEXT.md process
4. **Layer 4 is versioned** — git tracks all output changes (no deletions, only appends/corrections)

---

## Status

**Stage:** mechanism_validation  
**Start Date:** 2026-08-12  
**Expected Complete:** 2026-08-15  
**Deliverables:** Pending (stage not yet run)
