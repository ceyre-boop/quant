# Oracle Stage 01: Reflection & Learning

**Layer 2: Infer patterns from yesterday's trade outcomes**

*Last updated: 2026-08-12*

---

## Purpose

Given today's trades and market state, extract lessons. Which decision metrics (commitment score, rate differential, library match) predicted winners vs losers? What should we adjust?

**Runs at:** 02:35 UTC (after harvest completes)

---

## Inputs (Layer 3 + Layer 4)

### Layer 3 (Reference)
- `../../shared/decision_logger_schema.md` — Trade field definitions
- `_config/trading_philosophy.md` — The six tenets (reasoning framework)

### Layer 4 (Working artifacts from previous stage)
- `../00-harvest/output/decision_logs_summary.json` — Hypothesis-specific trade metrics
- `../00-harvest/output/market_state.json` — Yesterday's regime and macro state
- `../00-harvest/output/hypothesis_status.json` — Which hypotheses were active

---

## Process

### Step 1: Analyze Decision Metrics (Automated + AI, ~5 min)

**Question:** Did decision_metrics (commitment_score, rate_diff, library_match) predict trade outcome?

**Actions:**
1. For each live hypothesis, extract trades from decision logs
2. Compute correlation: commitment_score vs pnl_pct (Pearson r)
3. Compute correlation: rate_differential vs pnl_pct (for forex only)
4. Compute correlation: library_match vs pnl_pct
5. Are winning trades characterized by higher conviction? Or did high conviction fail today?

**Output:** `reflection_metrics.json`
```json
{
  "date": "2026-08-12",
  "hypothesis_analysis": {
    "HYP-045": {
      "trades": 2,
      "commitment_score_vs_pnl": {
        "correlation": 0.68,
        "p_value": 0.31,
        "interpretation": "weak signal; n=2 insufficient"
      },
      "rate_differential_vs_pnl": {
        "correlation": 0.92,
        "p_value": 0.08,
        "interpretation": "strong; wider spreads predicted larger wins"
      },
      "lesson": "Rate differential remains the strongest predictor. Commitment score is noisy (n too small)."
    }
  }
}
```

### Step 2: Check Regime Appropriateness (Automated, ~2 min)

**Question:** Did yesterday's regime fit the system, or did we trade out-of-regime?

**Actions:**
1. Read yesterday's regime from harvest output
2. For each live hypothesis, check: does this hypothesis have a regime preference?
3. Did we trade in a preferred regime, or out of it?
4. If out-of-regime: that trade is expected to have lower Sharpe

**Example:**
- HYP-045 (carry) prefers rate_trending regime
- Yesterday: market was rate_trending (regime_confidence 0.79)
- Expected: carries should have worked well
- Actual: both HYP-045 trades won ✅
- Lesson: "Regime was favorable. Carry edge working as expected."

**Output:** `regime_appropriateness.json`
```json
{
  "date": "2026-08-12",
  "yesterday_regime": "rate_trending",
  "hypothesis_regime_fit": {
    "HYP-045": {
      "preferred_regime": "rate_trending",
      "fit": "ALIGNED",
      "expected_performance": "strong",
      "actual_performance": "strong",
      "lesson": "Carry edge confirmed in aligned regime"
    }
  }
}
```

### Step 3: Identify Anomalies (AI reasoning, ~3 min)

**Question:** Did any trade violate expectations? Any surprising wins or losses?

**Actions:**
1. For each trade: was outcome consistent with pre-entry commitment_score?
   - High commitment trade that lost? → Investigate signal degradation
   - Low commitment trade that won? → Lucky, or signal not captured?
2. Were any losses larger than protective stops should allow? → Risk management breach
3. Did any wins exceed target exits? → Exited too early?

**Output:** `anomalies.json`
```json
{
  "date": "2026-08-12",
  "anomalies": [
    {
      "trade_id": "HYP-093-2026-08-12-1001",
      "anomaly_type": "SURPRISING_WIN",
      "commitment_score": 0.45,
      "outcome_pnl_pct": 2.8,
      "interpretation": "Low commitment entry hit target. Suggests entry signal criteria may be too strict; consider loosening.",
      "lesson": "Commitment 0.4–0.5 is not useless; may merit larger position size."
    }
  ]
}
```

---

## Outputs

Write to `output/`:

1. **`reflection_metrics.json`** — Correlation analysis (commitment, rate_diff, library_match vs outcome)
2. **`regime_appropriateness.json`** — Did yesterday's regime fit the systems?
3. **`anomalies.json`** — Surprising outcomes or risk management deviations
4. **`lessons.json`** — Actionable recommendations for next stage

---

## Lessons Format

Each lesson is a hypothesis-specific insight:

```json
{
  "date": "2026-08-12",
  "lessons": [
    {
      "hypothesis_id": "HYP-045",
      "lesson": "Rate differential > 140 bps predicts larger wins (r=0.92). Consider sizing up when differential is >140.",
      "confidence": "medium (n=2 trades; need n>10 for high confidence)",
      "proposed_action": "Increase conviction multiplier for rate_diff>140 from 0.75 to 0.85 (trial: 10 trades)"
    },
    {
      "hypothesis_id": "HYP-093",
      "lesson": "Commitment score 0.4–0.5 is not a reject filter; even low-conviction gappers faded yesterday.",
      "confidence": "low (n=2 trades, but contradicts prior assumption)",
      "proposed_action": "Monitor: trial sizing on commitment 0.4–0.5 entries (next 5 opportunities)"
    }
  ]
}
```

---

## Success Criteria

✅ **Correlations computed** (no missing data)  
✅ **Regime appropriateness assessed** (hypothesis preferences clear)  
✅ **Anomalies identified** (any surprising trades flagged)  
✅ **Lessons extracted** (at least one actionable insight per hypothesis)

---

## Next Stage

`02-test/CONTEXT.md` takes these lessons and validates them on unseen hold-out data before live implementation.

---

## Non-Negotiable

- **Only compute on closed trades** (both entry and exit logged). Do not guess at outcomes.
- **Do not re-size live positions yet.** Lessons are proposed, not enacted. Testing comes next.
- **N matters.** Do not claim high confidence on 2 trades. Flag insufficient data.
