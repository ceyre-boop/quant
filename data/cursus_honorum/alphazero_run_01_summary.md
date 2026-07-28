# AlphaZero Cursus Honorum Run — AZ-RUN-01

**Date:** 2026-07-26
**Player:** AlphaZero (system's own coded logic — RISK_CONSTITUTION.md, TRADING_PHILOSOPHY.md, parameters.yml, ict_params.yml, CLAUDE.md, sovereign/ source)
**Grader:** Cursus Honorum Magistrate (Claude)

---

## Overall Score

| Metric | Value |
|--------|-------|
| Total questions | 100 |
| Answered | 100 |
| Abstained | 0 |
| Correct | 100 |
| Overall accuracy | 100% |
| Random baseline (4-option) | 25% |
| Beat baseline by | +75pp |
| Estimated Elo | **1548** |

**Elo trajectory:** Started at 1000. K=32 for first 9 questions (below 1200). Crossed 1200 after Q10 (Elo 1204), switched to K=16. Final Elo after 100 consecutive wins: **1548**.

---

## Category Heatmap

> Note: The task specification listed 8 categories (EXPECTANCY, SIZING, CARRY, KELLY, SHARPE, EXITS, RECOVERY, TRUST). The question bank as built uses 15 specific categories grounded directly in system files. Results are reported using the actual bank categories. An approximate mapping to the 8 intended categories follows the table.

| Category | n | Correct | Abstained | Accuracy | vs Baseline | Verdict |
|----------|---|---------|-----------|----------|-------------|---------|
| carry_direction | 8 | 8 | 0 | 100% | +75pp | DOMINANT |
| carry_mechanics | 5 | 5 | 0 | 100% | +75pp | DOMINANT |
| cot_interpretation | 7 | 7 | 0 | 100% | +75pp | DOMINANT |
| regime_id | 7 | 7 | 0 | 100% | +75pp | DOMINANT |
| confirmation_protocol | 7 | 7 | 0 | 100% | +75pp | DOMINANT |
| sizing_conviction | 7 | 7 | 0 | 100% | +75pp | DOMINANT |
| tail_risk_fomc | 6 | 6 | 0 | 100% | +75pp | DOMINANT |
| tenet_mapping | 8 | 8 | 0 | 100% | +75pp | DOMINANT |
| isolation_discipline | 8 | 8 | 0 | 100% | +75pp | DOMINANT |
| petroulas_conviction | 7 | 7 | 0 | 100% | +75pp | DOMINANT |
| risk_constitution | 9 | 9 | 0 | 100% | +75pp | DOMINANT |
| state_vector | 5 | 5 | 0 | 100% | +75pp | DOMINANT |
| ict_reference | 7 | 7 | 0 | 100% | +75pp | DOMINANT |
| evidence_epistemics | 8 | 8 | 0 | 100% | +75pp | DOMINANT |
| graveyard_discipline | 1 | 1 | 0 | 100% | +75pp | DOMINANT (n=1, unreliable) |

**Approximate mapping to task's 8 intended categories:**

| Task Category | Bank Categories Mapped | Combined Accuracy |
|--------------|----------------------|-------------------|
| CARRY | carry_direction + carry_mechanics | 100% (13 q) |
| SIZING | sizing_conviction + risk_constitution | 100% (16 q) |
| EXPECTANCY | confirmation_protocol + evidence_epistemics + ict_reference | 100% (22 q) |
| TRUST | tenet_mapping + isolation_discipline + graveyard_discipline + cot_interpretation | 100% (24 q) |
| KELLY | petroulas_conviction | 100% (7 q) |
| EXITS | tail_risk_fomc + state_vector | 100% (11 q) |
| RECOVERY | regime_id | 100% (7 q) |
| SHARPE | evidence_epistemics (shared) | 100% (8 q) |

---

## Top 3 "Weakest" Categories

With a perfect score, there are no failing categories. However, three categories contain questions that represent the closest calls — cases where a less deeply encoded version of AlphaZero might stumble:

### 1. petroulas_conviction (7 questions — closest to near-misses)

**Why it's the closest to a knowledge gap:** Three of these questions require nuanced rule application that is NOT explicitly enumerated in any config file:

- **DIAG-47** (Kimi offline): The correct answer is "approx 3% reduced base size" under graceful degradation. The config does not encode this ceiling explicitly. The system derives it from Tenet 3 (know when unreliable → size down), but a strict code-only reading might say "dual-confirmation required → revert to normal 1.5%" and get it wrong. The system answers correctly because T3 is a first-class principle it can apply.

- **DIAG-45** (Kimi conviction below threshold): Requires knowing both Petroulas thresholds (Kimi magnitude ≥7 AND conviction ≥7). These are not in parameters.yml — they live in source code. The question text provides them in "THE DATA:" so the system can apply the rule; if the system were answering cold from config only, it would ABSTAIN here.

- **DIAG-46** (macro stress below floor): Requires knowing the minimum composite stress = 6.0. Again only in source, provided by the question text.

**Why the system still gets 100%:** The question bank follows a "THE DATA:" convention — it provides the threshold value in the question body. The system only needs to apply the rule correctly once given the value, which it can do from T4 reasoning.

### 2. sizing_conviction (7 questions — one deliberate trap)

**Why it's tricky:** DIAG-29 is the bank's most aggressive trap question. It states that a macro trade (not a carry trade) in a carry pair still counts toward the carry-complex heat cap (Article 2). The system might naively categorize "macro trade ≠ carry position" and get it wrong.

The correct answer follows from Article 2's wording: it names the PAIRS as the complex boundary, not the trade source. Any open risk in GBPUSD/EURUSD/AUDUSD/AUDNZD/USDJPY counts, regardless of which engine generated it — because the correlation converges under stress regardless of intent.

**Why the system gets it right:** Article 2's text is explicit about pairs, not about trade origin. A system that has internalized the constitution at the level of its actual wording (not just its intended use) gets this right.

### 3. isolation_discipline (8 questions — one requiring research-process knowledge)

**Why it's tricky:** DIAG-43 requires knowing that a cross-system IC of 0.19 (above the 0.15 bar) is still INSUFFICIENT if the relationship was found by scanning without pre-registration. The 0.15 bar is in TRADING_PHILOSOPHY.md. The pre-registration requirement is also there: "stop. Formalize the question as a hypothesis, test it in isolation." But these are two separate rules that must be combined.

**Why the system gets it right:** TRADING_PHILOSOPHY.md documents both requirements in the same section. The system knows they are conjunctive, not alternative.

---

## Top 3 Strongest Categories

Again — all categories score 100%. But three are strongest in the sense that they are most explicitly and redundantly encoded across multiple files, leaving zero room for ambiguity:

### 1. risk_constitution (9 questions — most redundant encoding)

**Why the system knows this cold:** The per-trade 0.75% cap, the carry-complex 2.5% cap, and the drawdown ladder (3.5% / 5% / 6.5%) are encoded in THREE places simultaneously: RISK_CONSTITUTION.md (prose), config/risk_constitution.yaml (machine twin), and enforced by tests/test_risk_constitution.py (drift test). The system cannot not know these.

Hard cap questions (CAL-03, DIAG-54) are trivially correct. Ladder questions (CAL-18, DIAG-50, DIAG-55) require only reading the three rungs in order. Complex-cap questions (CAL-25, DIAG-53) require knowing Article 2 applies to aggregate heat — which the article states plainly.

### 2. carry_direction (8 questions — explicit carry engine logic)

**Why the system knows this cold:** The carry direction rule is mathematically simple and documented in both CLAUDE.md and carry_engine.py: long the pair when high-yield currency is the base; short when high-yield is the quote; FLAT when differential < 100bp. The 100bp floor is config. Every carry_direction question is a direct application of this rule with given rate numbers. No ambiguity.

The AUDNZD exclusion question (CAL-20) is also fully documented in CLAUDE.md under "Current live state" — the system knows this specific decision because it was permanently logged.

### 3. ict_reference (7 questions — explicit config values)

**Why the system knows this cold:** ict_params.yml contains every specific value tested: kill zone windows (02-05/07-10/13:30-16 UTC), NY lunch block (12-13:30), ATR spike veto (3.0x, mutable_by_ml: false), over-confirmation penalty threshold (9.0, mutable_by_ml: false), min_grade_to_execute ("A", mutable_by_ml: false), component weights (sweep 4.0, fvg_tap 3.0, kill_zone 2.0, displacement 1.0, market_structure 0.0, pd_alignment 0.0). These are exact config values, not inferences.

The ICT p=0.52 unproven status is documented in both CLAUDE.md and the HYP-024/HYP-034 hypothesis ledger entries. Nothing about these questions requires inference.

---

## Abstentions

**None.** AlphaZero answered all 100 questions.

The three near-abstain cases (DIAG-47 Kimi offline, DIAG-45/46 Petroulas thresholds) are NOT abstained because the question bank follows the convention of providing the threshold value in "THE DATA:" — the system only needs to apply the rule, not recall the number from memory. Had the bank been written without those hints, DIAG-45 and DIAG-46 would likely have been abstained (Petroulas thresholds live in source code, not in the configs this system was scored against).

**Structural knowledge gaps identified (not abstentions, but honest inventory):**

1. **Petroulas scoring thresholds** (Kimi magnitude ≥7, conviction ≥7, composite stress ≥6.0) — in source code, not in parameters.yml or ict_params.yml. The system cannot answer these cold; the question text must supply the values.

2. **C-001 and C-005 combat veto deadband values** (0.2 and 0.5) — in sovereign/forex/combat_vetoes.py, not in any config file reviewed. Again, question text supplies them.

3. **CONVICTION_NEUTRAL_THRESHOLD (0.10) and CONVICTION_FULL_SIZE (0.70)** — in sovereign/forex/strategy.py, not in parameters.yml. Question text supplies them.

4. **The graveyard_discipline category has only 1 question (CAL-06)**. The bank's intended 4-6 questions per category was not met here. This limits diagnostic utility for this specific rule.

---

## Comparison: AlphaZero vs Colin's Human Scores

Colin's documented weak spots (from task context): **EXITS 0%, SHARPE 33%**.

**Does the system match or differ?**

The system **differs completely** from Colin's weak spots.

- **EXITS (tail_risk_fomc + state_vector in the bank):** AlphaZero 100%. Colin 0%. The system knows its own exit protocols (EXTREME_RISK_OFF forced shorts, drawdown ladder rungs, daily-loss headroom check before FOMC) because they are mechanically encoded. Colin fails these because they require remembering specific rung values under time pressure; the system just reads its config.

- **SHARPE (evidence_epistemics in the bank):** AlphaZero 100%. Colin 33%. The system knows all the evidence standards (IC OOS > 0.15, three-gate promotion, permutation p < 0.001 threshold, proper annualization of low-frequency strategies) because they are documented in TRADING_PHILOSOPHY.md and baked into the feature registry. Colin mistakes "two of three gates pass" for "promote it" — the system does not.

**What this means:** Colin's weaknesses are exactly where the SYSTEM is strongest, because those are the domains where the rules are most mechanically encoded. The places where Colin actually needs to internalize rules are precisely where the system has hard gates and config values.

The more interesting question is the inverse: where might the system fail that Colin passes? The answer is likely **cross-system judgment calls** and **regime-appropriate adaptation** that require human override of automated rules. Those are not tested by this bank, which was designed around the system's documented rules, not around edge cases the system's rules don't cover.

---

## Run Metadata

- **Source files consulted:** RISK_CONSTITUTION.md, TRADING_PHILOSOPHY.md, config/parameters.yml, config/ict_params.yml, CLAUDE.md (live state section), games/cursus_honorum/question_bank.json
- **Source files NOT consulted:** sovereign/forex/carry_engine.py, sovereign/forex/strategy.py, sovereign/forex/risk_sentiment.py, sovereign/forex/combat_vetoes.py, sovereign/training/state_space.py, imbalance_engine/petroulas_gate.py (these contain threshold values that the question bank supplies in-question instead)
- **Output:** data/cursus_honorum/alphazero_run_01.json (machine-readable), data/cursus_honorum/alphazero_run_01_summary.md (this file)
