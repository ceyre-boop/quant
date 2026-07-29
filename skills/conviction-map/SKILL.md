---
name: conviction-map
description: Manually runs the three Petrules Gate organs for any name or event, returns conviction tier + sizing guidance
---

## TRIGGER PHRASES
"conviction map on X", "run petrules on X", "what's consensus vs what I think", "conviction check", "petrules check"

## WHAT YOU ARE DOING
You are running the three organs of the Petrules Gate — a structured conviction engine for Alta Investments. The output is a widget with three panels (one per organ), a conviction tier badge, and a variant perception section. This is not a trade recommendation — it is a conviction measurement. Sizing follows from the tier, subject to the conviction-based sizing pipeline in `config/parameters.yml`.

---

## PHASE 1: GET COLIN'S HYPOTHESIS

Before running the organs, you need to know what Colin thinks. Extract from his message:
- The instrument
- His directional thesis (long/short)
- The key insight or divergence he sees (if stated)
- The expected timeframe

If the hypothesis is not stated, ask: "What's your thesis — what do you think will happen and why?" Do not run the conviction map on "I want to buy X" without a thesis. The map measures divergence from consensus — no thesis = no divergence to measure.

---

## PHASE 2: ORGAN 1 — CONSENSUS BASELINE

**Goal**: What does the market currently expect?

Research and assemble:

1. **Analyst estimates** (equities): Mean EPS estimate, mean revenue estimate, mean price target, range of targets, number of analysts. Source: search "[ticker] consensus estimates analyst" on finviz, Seeking Alpha, or Yahoo Finance.

2. **Options-implied move** (if applicable): Search "[ticker] options implied move earnings". Express as ±X%.

3. **COT positioning** (futures/forex): Search "[instrument] COT speculative net positioning". State current net long/short and change from prior week. Is positioning at an extreme (>80th or <20th percentile vs. 3-year range)?

4. **Narrative scan**: Read the top 5 headlines for this instrument in the last 14 days. What is the consensus story — what does "everyone know" about this name right now? Summarize in 2 sentences.

5. **Positioning signal**: Is the crowd leaning long, short, or neutral? State clearly.

Compile into:
- CONSENSUS DIRECTION: BULLISH / BEARISH / NEUTRAL
- CONSENSUS STRENGTH: STRONG / MODERATE / WEAK
- KEY CONSENSUS BELIEF: one sentence stating what the market believes

---

## PHASE 3: ORGAN 2 — DIVERGENCE DETECTOR

**Goal**: How far does Colin's thesis diverge from consensus? Are there historical analogs?

Steps:

1. **State the divergence explicitly**: Write "Consensus believes [X]. Colin's thesis is [Y]. The specific divergence is [Z]." Be concrete. If there is no divergence (Colin agrees with consensus), note that — the conviction tier will be capped at Tier 2.

2. **Historical analog search**: Search for situations where this same divergence setup occurred. Look for:
   - Same instrument, similar macro environment
   - Similar consensus belief + contrarian thesis
   - Search: "[instrument] [divergence keyword] historical analog" or "when did [instrument] diverge from [consensus narrative]"
   - Also check `data/research/hypothesis_ledger.jsonl` if accessible — any confirmed edges that match this setup structure

3. **Analog outcomes**: For each analog found (up to 3), state:
   - Period (year/quarter)
   - What happened (direction, magnitude, timeframe)
   - Whether the contrarian thesis was correct or consensus won
   - What the key differentiator was

4. **Divergence score**: Based on the above, assign:
   - STRONG DIVERGENCE: thesis directly contradicts consensus, supported by 2+ analogs
   - MODERATE DIVERGENCE: thesis differs from consensus but not extreme, 1 analog or partial support
   - WEAK DIVERGENCE: thesis slightly differs, or analogs are mixed
   - NO DIVERGENCE: thesis aligns with consensus — note this, tier is capped at Tier 2

---

## PHASE 4: ORGAN 3 — CONVICTION SCORE

**Goal**: Synthesize the evidence into a conviction tier.

### Tier Definitions

| Tier | Score Range | Meaning | Sizing |
|------|-------------|---------|--------|
| TIER 1 | 0.0–0.4 | Edge unclear, divergence not supported by data | Do not trade or minimum exploratory size |
| TIER 2 | 0.4–0.7 | Divergence supported but not compelling, or no divergence | 0.5–1.0× normal unit |
| TIER 3 | 0.7–0.85 | Strong divergence signal, analogs supportive, regime aligned | 1.0–1.5× normal unit |
| TIER 4 | 0.85+ | Rare screamer: strong divergence, multiple analogs, regime perfect, disclosed flow confirms | Up to f_max — never override f_max |

### Scoring Rubric (add up the points, divide by 7 for 0–1 score)

| Factor | Points |
|--------|--------|
| Divergence from consensus exists and is meaningful | 0 or 1 |
| Divergence is supported by 2+ historical analogs | 0 or 1 |
| Regime is aligned with the thesis (carry regime, market regime) | 0 or 1 |
| Disclosed flow (Form 4, COT, unusual options) confirms thesis direction | 0 or 1 |
| Calendar risk is LOW (no CB/earnings/macro event within 14 days) | 0 or 1 |
| Setup has a clean technical level and positive R:R (≥2:1) | 0 or 1 |
| ICT or pattern confirmation present (NOTE: ICT edge unvalidated p=0.52 — count only 0.5 if this is the only technical signal) | 0 or 0.5 |

State the raw score and the tier assignment.

### Disclosed Flow Summary

Compile what public disclosures show:
- Form 4 filings (insider buys/sells in last 30 days)
- 13D/G filings (activist or large-stake disclosure)
- Unusual options activity (large put/call imbalances, unusual OI changes)
- COT change direction for futures/forex

Label each: CONFIRMS THESIS / CONTRADICTS THESIS / NEUTRAL / DATA UNAVAILABLE

---

## PHASE 5: VARIANT PERCEPTION SECTION

**What would change this conviction HIGHER?**
List 3 specific conditions. Example: "BOE raises rates unexpectedly → WIDENING regime, Tier 3 → Tier 4"

**What would change this conviction LOWER?**
List 3 specific conditions. Example: "Fed signals pause → NARROWING regime, Tier 2 → Tier 1"

These are the live monitoring conditions. If any occur before entry, re-run this skill.

---

## PHASE 6: RENDER THE WIDGET

Call `mcp__visualize__show_widget` with an HTML widget containing:

**Panel 1 — Consensus Baseline**
- Consensus direction badge (bullish/bearish/neutral, color-coded)
- Key consensus belief (one sentence)
- Positioning extreme flag if applicable
- Options-implied move if available

**Panel 2 — Divergence Detector**
- The divergence statement (Consensus vs. Thesis)
- Analog table: Period | Outcome | Contrarian Won? (3 rows max)
- Divergence strength badge

**Panel 3 — Conviction Score**
- Scoring rubric table with checkmarks/X for each factor
- Raw score (e.g., 4.5/7)
- Conviction tier badge — large, prominent
- Sizing guidance for this tier

**Disclosed Flow Summary**
- Compact table: Source | Signal | Alignment with Thesis

**Conviction Tier Badge**
- TIER 1: gray
- TIER 2: amber (#fbbf24)
- TIER 3: green (#4ade80)
- TIER 4: cyan/electric (#22d3ee) with glow effect

**Variant Perception Section**
Two columns: "Raises conviction ↑" (green) | "Lowers conviction ↓" (red). 3 bullets each.

**Action Recommendation** (one line):
- Tier 1: "Do not enter — insufficient evidence"
- Tier 2: "Small position acceptable — monitor for upgrade"
- Tier 3: "Full position size justified — enter with defined stop"
- Tier 4: "Maximum conviction — size to f_max per risk constitution"

**Visual spec:**
- Dark background (#0f1117), cards with border (#1e2433)
- Footer: "Alta Investments · Petrules Gate · [instrument] · [date] · Public data only"
- No external JS. Inline CSS only.

---

## CONSTRAINTS

- Never recommend exceeding f_max. f_max is defined in `config/parameters.yml` and `RISK_CONSTITUTION.md`. Do not hardcode it — reference the config.
- ICT pattern signals count as 0.5 points max in the scoring rubric (edge is unvalidated, permutation p=0.52).
- Public data only. Never speculate about undisclosed information.
- If Colin provides no thesis, do not run the organs. Ask for the thesis first.
- Conviction score is advisory. Final sizing decision is Colin's. This tool informs, not decides.
