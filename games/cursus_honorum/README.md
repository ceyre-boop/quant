# Cursus Honorum — Question Bank

A 100-question preset bank for the **Cursus Honorum** learning game: a Roman-themed
trading-education game that teaches the **Sovereign** system and uses Elo as proof of mastery.
Claude (the Magistrate) is the sole author. Every "optimal" answer maps to a real tenet of
`TRADING_PHILOSOPHY.md` and is grounded in the live config/code — a wrong optimal would
train the wrong reflex, so the bank was authored and verified against the real system.

> **Scope boundary (read first).** This deliverable is **the 100 grounded questions + the
> tracking schema only**. The Elo engine, the difficulty-router implementation, the API
> wiring, and the UI are **out of scope** here — Colin listed those separately. Nothing in
> this folder touches or imports any execution-path file; it is pure content + schema.

## Files

| File | What it is |
|------|-----------|
| `question_bank.json` | The 100 questions (`meta` + `questions[]`). |
| `categories.json` | The rank bands, the six tenets, and the diagnostic **category registry** to track. |
| `README.md` | This document. |

## Question record shape

```jsonc
{
  "id": "DIAG-27",                 // CAL-01..CAL-30 (calibration), DIAG-01..DIAG-70 (diagnostic)
  "phase": "diagnostic",           // "calibration" | "diagnostic"
  "rank": "Praetor",               // Quaestor 800-1000 / Aedile 1000-1200 / Praetor 1200-1400 /
                                   //   Consul 1400-1600 / Dictator 1600-1800 / SPQR 1800+
  "category": "sizing_conviction", // one tag from categories.json (the per-area tracking key)
  "elo_weight": 1350,              // difficulty anchor the router uses for placement + updates
  "prompt_roman": "...THE DATA: ...", // 2-3 sentence SPQR-voice scenario + a THE DATA block
  "options": { "A": "...", "B": "...", "C": "...", "D": "..." },
  "correct": "B",                  // the single optimal letter
  "tenet": "T3",                   // which of the six tenets justifies the optimal (see categories.json)
  "stockfish_explanation": "..."   // mentor-voice WHY, in Colin's language (not textbook)
}
```

## The 100: calibration vs diagnostic

**First 30 = CALIBRATION (`CAL-01`..`CAL-30`).** Ordered easy → hard (`elo_weight` is
monotonically non-decreasing across the 30), one clear concept each, spread across **all six
ranks** and a broad topic mix. Their job is to **locate a ballpark Elo** fast — start easy,
widen. Rank spread: Quaestor 6, Aedile 6, Praetor 6, Consul 6, Dictator 4, SPQR 2.

**Next 70 = DIAGNOSTIC (`DIAG-01`..`DIAG-70`).** Deeper, each tagged by `category` so
**strengths/weaknesses are trackable**. Every category is covered multiple times at varying
difficulty so **per-category accuracy becomes measurable** — this is the "where does the
number hold, where does it drop" map. Counts per diagnostic category:

| Category | # | Category | # |
|----------|---|----------|---|
| carry_direction | 6 | tenet_mapping | 5 |
| risk_constitution | 6 | isolation_discipline | 5 |
| evidence_epistemics | 6 | petroulas_conviction | 5 |
| cot_interpretation | 5 | ict_reference | 5 |
| regime_id | 5 | carry_mechanics | 4 |
| confirmation_protocol | 5 | tail_risk_fomc | 4 |
| sizing_conviction | 5 | state_vector | 4 |

(Calibration questions are *also* category-tagged, so `carry_mechanics`, `tail_risk_fomc`,
`state_vector`, and `graveyard_discipline` pick up additional coverage there.)

## How the AlphaZero difficulty router should read this

1. **Calibration first.** Serve `CAL-01`→`CAL-30` in order. Each item's `elo_weight` is the
   difficulty anchor; standard Elo update after each answer lands the player in a **ballpark
   rank band** (`categories.json.ranks`). Because they start easy and widen, a few items
   bracket the player quickly.
2. **Then diagnostic, category-aware.** Once a provisional Elo exists, sample `DIAG-*` items
   whose `elo_weight` sits near the player's current estimate, **stratified by `category`**.
   The router reads `category` to decide *what to probe next*: keep the player near their
   level globally, but deliberately sample **each category** enough times to get a stable
   per-category accuracy.
3. **Track per-category accuracy** as `correct / attempted` per `category` tag (weight by
   `elo_weight` if you want an Elo-per-category instead of a raw hit rate). The output is a
   **profile, not a single number**: e.g. strong `carry_direction`/`regime_id`, weak
   `isolation_discipline`/`cot_interpretation`. Widen sampling where accuracy is HIGH (push
   difficulty up), narrow and re-test where it DROPS (confirm the weakness, then teach to it
   via the `stockfish_explanation`).
4. **Tenet roll-up (optional).** Each item also carries a `tenet`, so the same accuracy
   tracking can roll up to a per-tenet mastery view (which of the six philosophy pillars the
   player has vs hasn't internalized).

## Correctness pass (done)

A full correctness pass was completed on every question:

- **Grounding.** Read before authoring: `TRADING_PHILOSOPHY.md`, `RISK_CONSTITUTION.md`,
  `config/parameters.yml`, `config/ict_params.yml`, and the live code —
  `sovereign/forex/{carry_engine,strategy,risk_sentiment,combat_vetoes}.py`,
  `sovereign/training/state_space.py`, `imbalance_engine/petroulas_gate.py`. Every numeric
  in a `THE DATA` block (rate diffs in bp, VIX/term-structure lines, grade/conviction
  thresholds, breaker rungs, the 8-dim vector, Petroulas gates) matches the real system.
- **Answer verification.** For every question the `correct` option was re-checked to be
  genuinely right *per the cited tenet and the scenario numbers*; ambiguous or wrong optimals
  were fixed. Key internally-consistent facts encoded: long the high-yield **BASE** / 100bp
  floor; forex grade ladder (A+ ≥2.0 **and** conv ≥0.60, A ≥1.5, B ≥0.5, C <0.5); conviction
  neutral floor 0.10 / full 0.70; combat vetoes C-001 (deadband 0.2) & C-005 (<0.5); the
  RISK_CONSTITUTION per-trade **0.75%** cap overriding the grade table, carry-complex **2.5%**
  heat, and the **3.5/5/6.5** breaker ladder below the 8% trailing halt; HYP-045 4-pair
  (AUDNZD excluded — both legs RBA-driven); ICT as an **unproven (p=0.52)** TP/SL reference
  layer, not a predictor; the Petroulas dual-confirmation gate (XGB >0.65 **and** Kimi
  mag ≥7 / conv ≥7 → 3-5%, hard 5% ceiling); COT Tue-measured/Fri-published + JPY inverse.
- **Answer-position balance.** After drafting, the optimal letter was redistributed to an even
  **25 / 25 / 25 / 25** across A/B/C/D (a per-question permutation), so the bank cannot be
  gamed by always picking one letter — the Elo signal stays honest. Letter references inside
  each `stockfish_explanation` were remapped in lock-step with their options.
- **Structural validation.** All 100 records have the required fields; `correct` is always a
  real option key; every `prompt_roman` has a `THE DATA` block; ids are unique; every
  `category` and `tenet` resolves against `categories.json`.
