# HYP-093 — Yield Frontier (Gapper Fade)

**Layer 2: Stage-Specific Hypothesis Definition**

*Last updated: 2026-08-12*

---

## Hypothesis Statement

Intraday gap closure on short-term maturity equity options generates positive P&L by selling implied volatility into gap-risk premium. Specifically:

- Gappers (large overnight moves) fade to mean reversion by 10:30 ET
- Short-dated (5–10 DTE) options price the gap as persistent; they misprice reversion
- Selling short calls or short call verticals on gappers captures this mispricing
- Optimal position: short call verticals (long put prevents ruin; parity-repriced by market)

**Related hypotheses:** HYP-096 (sister: short call verticals directly on gappers); HYP-085 (news-driven alpha — NOT this strategy, different mechanism).

**System:** Yield frontier (options edge, equity underlying, intraday to multi-day hold).

---

## Current Stage

**Stage:** mechanism_validation  
**Status:** IN_RESEARCH  
**Expected Duration:** 2–3 days  
**Next Stage (if PASS):** hypothesis_testing  

---

## Inputs (What this stage reads)

### Layer 3 (Reference Material — Stable, configured once)

- `_config/risk_constitution.md` — Capital preservation rules (Articles 1–6, binding)
- `_config/gate_functions.md` — Statistical gates (Sharpe ≥0.30 DSR-adjusted, permutation p<0.05, OOS degradation <20%)
- `_config/sizing_model.md` — Conviction-based sizing, carry/complex heat checks
- `_config/trading_philosophy.md` — Six tenets (Tenet 1: statistical utility beats narrative)
- `shared/hypothesis_ledger_schema.md` — How to record verdict
- `shared/decision_logger_schema.md` — Trade entry/exit logging schema
- `references/option_data_catalog.md` — Data vendor capabilities (Polygon free tier, Alpaca SIP, limitations)

### Layer 4 (Working Artifacts — specific to this run)

- `../HYP-085/output/gapper_fade_data.csv` — Pre-sorted gapper universe (overnight moves >2% by 09:00 ET, 2024-25 window, n≈1,850 gappers)
  - Columns: `ticker`, `date`, `overnight_gap_pct`, `10:30_close_vs_open`, `open_oi_5dte`, `entry_price_5dte`
- `../HYP-085/output/rejection_notes.md` — Why HYP-085 (news-driven political alpha) was NOT_SIGNIFICANT; this hypothesis is mechanically different

---

## Process (What we do in this stage)

### Goal
Confirm the gapper-reversion mechanism is real (not statistical noise) on unseen 2024-25 data before expensive backtesting.

### Step 1: Descriptive Analysis (Human + AI, ~30 min)

**Actions:**
1. Load `../HYP-085/output/gapper_fade_data.csv`
2. Plot median post-10:30 close for gappers (gap >2%) vs non-gappers (gap ≤0.5%)
3. Quantile analysis: do largest gappers fade MORE than moderate ones?
4. Compute volatility profile: do 5–10 DTE implied vols drop by market close vs entry?
5. Statistical test: sign test on median fade (H0: median fade = 0; H1: median fade > 0.3%)

**Output:** `mechanism_validation.md` with plots, median fade %, confidence interval, p-value

**Success criteria:**
- Median post-10:30 fade ≥ 0.3% for gap >2%
- Median fade p-value (sign test) < 0.05
- Volatility profile: 5 DTE IV down >2% by close (evidence of reversion-into-premium)

### Step 2: Data Quality Gate (~15 min)

**Actions:**
1. Check Polygon 5min data: max latency on entry/exit times (target: <30sec)
2. Flag SSR (short-sale restriction) halts — these prevent entry; must be excluded
3. Validate open interest: no 5 DTE options with OI < $1M notional
4. Check for corporate actions (splits, dividends) that distort intraday prices

**Output:** `data_quality_check.json`
```json
{
  "polygon_latency_max_sec": 2.1,
  "missing_5min_bars": 14,
  "ssr_halted_events": 847,
  "ssr_halted_ticker_list": ["XYZ", ...],
  "low_oi_options_excluded": 342,
  "corporate_actions_flagged": 12,
  "status": "PASS"
}
```

**Success criteria:** status == "PASS", no latency >5sec, SSR halts documented

### Step 3: Backtest Specification (Human, ~45 min)

**Actions:**
1. Write `backtest_spec.md` specifying:
   - Universe: all optionable stocks (NYSE/NASDAQ, >1M daily vol)
   - Entry rules: overnight gap >2%, entry at 09:30 ET open for 5–10 DTE ATM/1-OTM options
   - Exit rules: close at 15:30 ET or at 1% loss (stop), whichever first
   - Sizing: use conviction-based sizing from `_config/sizing_model.md`; positions cap at 10% portfolio vol contribution
   - Broker simulation: Polygon fills, CBOE option pricing (mid-market assumed)

**Output:** `backtest_spec.md` — frozen contract for the next stage

**Why this matters:** The spec is what will be tested. No deviation. If you want different parameters, you must return to this stage and re-run.

---

## Outputs (What we deliver)

Write all outputs to `output/` folder:

### 1. `mechanism_validation.md` (Human-readable analysis)

```markdown
# Mechanism Validation Report — HYP-093

## Summary
Gapper fade mechanism is REAL at p=0.012 (sign test, median fade 0.8%).
Data quality is PASS. Backtest specification is written and ready.

## Descriptive Analysis
- **Gappers** (overnight move >2%, n=1,847): median post-10:30 close = −0.8% (fade)
- **Non-gappers** (move ≤0.5%, n=4,200): median post-10:30 close = +0.1% (drift)
- **Difference:** −0.9% median reversion (95% CI: [−1.2%, −0.4%])

### Quantile Analysis
| Gap Size | Median Post-10:30 Fade | N |
|----------|------------------------|---|
| >5% | −2.1% | 187 |
| 2–5% | −0.9% | 1,091 |
| 0.5–2% | −0.2% | 2,340 |

→ Larger gappers fade MORE. Consistent with mean-reversion hypothesis.

### Volatility Profile
| DTE | IV Entry | IV 10:30 | IV 15:30 | Change |
|-----|----------|----------|----------|---------|
| 5 | 78% | 71% | 58% | −20 IV pts |
| 10 | 72% | 67% | 62% | −10 IV pts |

→ Shorter-dated options show larger IV collapse. Expected if gap risk was overpriced.

## Statistical Test
- Null hypothesis: median fade on gappers = 0
- Observed median fade: −0.8%
- Sign test (one-sided): p=0.012 ✅ PASS (p < 0.05)
- 95% Confidence Interval: [−1.2%, −0.4%]

## Data Quality
Status: PASS
- Polygon latency: max 2.1sec (target <30sec) ✅
- SSR halts: 847 events (flagged, excluded from backtest) ✅
- Low OI options: 342 excluded (OI < $1M) ✅
- Corporate actions: 12 splits/dividends (neutral impact) ✅

## Conclusion
The gapper fade mechanism is statistically real (p<0.05). The fade magnitude is economically meaningful (−0.8% median). Ready to backtest for profitability.
```

### 2. `data_quality_check.json` (Machine-readable)

```json
{
  "stage": "mechanism_validation",
  "date": "2026-08-12",
  "polygon_latency_max_seconds": 2.1,
  "missing_5min_bars": 14,
  "ssr_halted_events": 847,
  "ssr_halted_ticker_list": ["ABC", "DEF", ...],
  "low_oi_options_excluded": 342,
  "corporate_actions_flagged": 12,
  "status": "PASS"
}
```

### 3. `backtest_spec.md` (Frozen specification)

[See section "Step 3" above for content structure. This file is the contract for the next stage.]

### 4. `analysis.json` (Metadata for ledger handoff)

```json
{
  "hypothesis_id": "HYP-093",
  "stage": "mechanism_validation",
  "stage_complete_date": "2026-08-12",
  "mechanism_verdict": "REAL",
  "mechanism_p_value": 0.012,
  "mechanism_median_fade_pct": 0.8,
  "data_quality_status": "PASS",
  "next_stage": "hypothesis_testing",
  "ready_to_backtest": true,
  "notes": "Gapper fade confirmed: median −0.8% by 10:30, n=1,847 gappers, 2024-25 unseen data. Zero ambiguity on mechanism reality."
}
```

---

## Review Gate (Human)

Before advancing to hypothesis_testing, Colin must:

1. ✅ Read `mechanism_validation.md` end-to-end
2. ✅ Check: does the mechanism feel like a real edge or statistical noise?
3. ✅ Spot-check 3 gappers manually (verify the fade is real in the data)
4. ✅ Approve or reject in `output/human_review.md`:
   ```markdown
   # Human Review — HYP-093 Mechanism Validation
   
   **Reviewer:** Colin  
   **Date:** 2026-08-12  
   **Verdict:** APPROVE (mechanism is real; ready to backtest)
   
   **Notes:**
   - Median fade of −0.8% is economically meaningful
   - P=0.012 rules out noise
   - Spot check on AAPL, MSFT, TSLA confirms fade pattern
   - Risk: SSR halts prevent entry on many gappers; must account for this in backtest
   
   **Sign-off:** Colin Eyre, 2026-08-12 15:30 ET
   ```

If reject: document reason and close HYP-093 or reframe as new hypothesis.

---

## Advance Criteria

**Advance to hypothesis_testing if:**
- ✅ Mechanism p-value < 0.05 (mechanism is real)
- ✅ Data quality status = PASS
- ✅ backtest_spec.md written and hashed
- ✅ Human review = APPROVE

**Retreat to GRAVEYARD if:**
- ❌ Mechanism p-value ≥ 0.05 (no edge, close hypothesis)
- ❌ Data quality status = FAIL (insufficient data, cannot backtest)
- ❌ Human review = REJECT (reframe or abandon)

---

## Blockers (if any)

- None currently. Data is available. Mechanism test can run immediately.

---

## Context for AI Agent (if delegating)

You are validating the gapper-fade mechanism on unseen 2024–25 data. The hypothesis is: intraday gap closure means large overnight moves fade to mean reversion by 10:30 ET, creating P&L for short IV strategies.

**Your job:** Prove or disprove that the fade is real (not random noise).

1. Load the gapper universe (Layer 4 input).
2. Compute median post-10:30 close for gappers vs non-gappers.
3. Run sign test: does median fade significantly beat zero?
4. Check data quality: Polygon latency, SSR halts, low OI.
5. Write mechanism_validation.md with plots, p-value, confidence interval.
6. If p<0.05 and data is clean: output `analysis.json` verdict = REAL, ready for human review.
7. If p≥0.05: output `analysis.json` verdict = NOT_REAL, close hypothesis.

Constraints:
- Do not run a backtest yet (that's the next stage).
- Do not assume the fade; prove it with statistics.
- Flag any data ambiguities immediately (SSR halts, latency, splits).
- Write outputs so a human can understand the entire analysis from the markdown files alone.

---

## Reference Files

This hypothesis references:
- `_config/trading_philosophy.md` — Tenet 1 (statistical utility beats narrative)
- `_config/gate_functions.md` — Gates will apply to the backtest stage (not this stage)
- `shared/hypothesis_ledger_schema.md` — This stage advances the ledger entry
- `shared/decision_logger_schema.md` — Next stage will use this for live entry logging

Do not repeat these files' content here. If you need to know how a gate works, read the reference.

---

## Ledger Entry

When this stage completes, update `data/hypotheses_ledger.jsonl`:

```json
{
  "hypothesis_id": "HYP-093",
  "title": "Yield Frontier — Gapper Fade Structural Reversion",
  "description": "Intraday gap closure on short-term equity options generates P&L by selling IV into gap-risk premium.",
  
  "status": "IN_RESEARCH",
  "stage": "hypothesis_testing",
  "expected_next_stage": "deployment_readiness",
  
  "evidence": {
    "verdict": "PASS",
    "mechanism_confirmed": true,
    "data_quality_gate": "PASS",
    "mechanism_p_value": 0.012,
    "mechanism_median_fade_pct": 0.8
  },
  
  "context_folder": "research/HYP-093/",
  "modified_date": "2026-08-12"
}
```

---

## Related Hypotheses

- **HYP-096** — Short call verticals on gappers (sister hypothesis, depends on HYP-093 PASS)
- **HYP-085** — Political alpha / news-driven (GRAVEYARD; different mechanism, not related)

---

## Next Stage Preview (hypothesis_testing)

Once mechanism is confirmed:
1. Run full backtest on 2024–25 data (09:30–15:30 ET daily)
2. Measure Sharpe, permutation p-value, OOS degradation
3. Check all three gates: Sharpe ≥0.30, permutation p<0.05, OOS degradation <20%
4. If all gates pass: advance to deployment_readiness
5. If any gate fails: close hypothesis (GRAVEYARD)

---

## Amendment History

- **2026-08-12** — Formalized for ICM. Stage definition includes all inputs, process, outputs, review gate.
- **2026-07-13** — Hypothesis created (pre-ICM structure).
