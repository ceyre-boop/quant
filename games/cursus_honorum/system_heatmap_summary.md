# System Self-Play — Cursus Honorum Diagnostic (2026-07-26)

## What this is

An automated "player" answered all 100 questions in `question_bank.json` by applying the
system's **actual coded/documented decision logic** — not by reading the bank's answer key
first. Full methodology, per-question provenance, and scores: `system_heatmap.json`.
Player logic: `build_heatmap.py` (kept alongside this summary in the scratchpad session that
produced it; the committed artifact is the JSON output, not the script).

Each of the 100 answers was derived from one of these named sources, tagged per-question as
`provenance` in the JSON:

| Provenance | Meaning | Count |
|---|---|---|
| `CODE` | Literal executable constant/threshold read directly from `carry_engine.py`, `risk_sentiment.py`, `combat_vetoes.py`, `strategy.py`, `state_space.py`, `petroulas_gate.py`, or `RISK_CONSTITUTION.md` Articles 1/2/3/6. | 69 |
| `CONFIG_DISABLED` | Same coded arithmetic, but the module is config-disabled in production (`combat_vetoes.py`, `enabled: false` — its own docstring calls it "ANALYSIS INSTRUMENT ONLY... NET -143R"). Flagged, not abstained: the math is real, the live gate is not. | 3 |
| `DOC_TENET` | The six tenets in `TRADING_PHILOSOPHY.md` — real, encoded doctrine, but there is no executable classifier that maps a scenario to a tenet ID. Judgment against documented prose, not a formula. | 14 |
| `LEDGER_FACT` | A specific historical/ledger record (hypothesis verdicts, measured Sharpe, p-values) quoted in `CLAUDE.md` / the hypothesis ledger. Encoded as recorded state, not a formula. | 11 |
| `LLM_BOUNDARY` | The production decision is delegated to an LLM call (Kimi's `petroulas_worthy` judgment of "arithmetic vs narrative"), not to deterministic code. **Marked `abstain=True`** — genuinely not something the coded system decides. | 3 |

## Random-baseline control

Every category score is reported against a 25% uniform-random baseline (4 options/question),
plus one seeded simulation (`seed=20260726`) that landed at 15% — a reminder that a single
random run swings well below the 25% expectation on n=100; the theoretical 25% is the number
to compare against, not the one simulated draw.

## Score

**97/100 correct, 3 abstained (0 wrong).** Every abstention was a `LLM_BOUNDARY` item inside
`petroulas_conviction` (CAL-17, DIAG-45, DIAG-48) — cases where the real system hands the
"is this arithmetic or narrative" call to Kimi, not to code. Scored as incorrect for accuracy
purposes (conservative), which is why `petroulas_conviction` shows 57% instead of 100%.

## Per-category (native 15-category scheme — matches the actual game's heatmap)

| Category | n | System acc. | vs. 25% baseline | Abstain |
|---|---|---|---|---|
| carry_direction | 8 | 100% | +75pp | 0 |
| carry_mechanics | 5 | 100% | +75pp | 0 |
| confirmation_protocol | 7 | 100% | +75pp | 0 |
| cot_interpretation | 7 | 100% | +75pp | 0 |
| evidence_epistemics | 8 | 100% | +75pp | 0 |
| graveyard_discipline | 1 | 100% | +75pp | 0 |
| ict_reference | 7 | 100% | +75pp | 0 |
| isolation_discipline | 8 | 100% | +75pp | 0 |
| **petroulas_conviction** | 7 | **57%** | +32pp | **3** |
| regime_id | 7 | 100% | +75pp | 0 |
| risk_constitution | 9 | 100% | +75pp | 0 |
| sizing_conviction | 7 | 100% | +75pp | 0 |
| state_vector | 5 | 100% | +75pp | 0 |
| tail_risk_fomc | 6 | 100% | +75pp | 0 |
| tenet_mapping | 8 | 100% | +75pp | 0 |

## Colin's 8-bucket scheme — could not be verified in this repo

Colin's own heatmap buckets (EXPECTANCY / SIZING / CARRY / KELLY / SHARPE / EXITS / RECOVERY /
TRUST) do not appear anywhere in `games/cursus_honorum/` or elsewhere in this repo (checked via
grep across the codebase and the game's own `index.html`, which renders its heatmap directly off
the native 15-category scheme). They are presumed to come from a separate, external assessment
Colin took. The table below is therefore a **hand-authored, best-effort projection** of the 15
bank categories onto those 8 labels — not a verified equivalence — with the mapping documented
in `build_heatmap.py`'s `CAT_TO_BUCKET` table (risk_constitution's 9 questions were further split
between KELLY and RECOVERY by keyword match on "ladder/drawdown/breaker/flatten/halve/halt").

| Bucket | n | System acc. | vs. 25% baseline | Abstain | Notes |
|---|---|---|---|---|---|
| EXPECTANCY | 16 | 100% | +75pp | 0 | evidence_epistemics, graveyard_discipline, ict_reference |
| SIZING | 12 | 100% | +75pp | 0 | sizing_conviction, state_vector |
| CARRY | 13 | 100% | +75pp | 0 | carry_direction, carry_mechanics |
| **KELLY** | 13 | **77%** | +52pp | **3** | risk_constitution (cap questions) + petroulas_conviction |
| SHARPE | 13 | 100% | +75pp | 0 | regime_id, tail_risk_fomc |
| **EXITS** | **0** | **undefined** | — | — | **No bank category maps to exit logic at all — zero coverage, not zero accuracy.** |
| RECOVERY | 3 | 100% | +75pp | 0 | risk_constitution (drawdown-ladder questions) |
| TRUST | 30 | 100% | +75pp | 0 | cot_interpretation, confirmation_protocol, tenet_mapping, isolation_discipline |

## Verdict

The system's own weakest point, once the answer key is set aside, is narrow and specific: it is
not "which pattern," "how much risk," or "which regime" — it is **judging whether a stated thesis
is genuine arithmetic or dressed-up narrative for conviction-sized Petroulas trades**. That
judgment is the one place in the entire architecture where a threshold/formula genuinely does not
exist; the code hard-gates on `magnitude>=7 AND conviction>=7`, but *scoring* magnitude and
conviction from a written thesis is explicitly Kimi's (an LLM's) call, not the code's. Every other
category — carry direction, sizing caps, regime classification, veto arithmetic, state-vector
math, isolation boundaries, even doctrine/tenet mapping and historical-evidence recall — resolved
cleanly from a literal read of the source files and documents, with zero mismatches against the
bank's own answer key.

This is a genuinely different shape of weakness than Colin's own human heatmap (EXITS 0%, SHARPE
33% per the prompt brief). The bank contains **no exit-logic questions at all**, so this run
cannot speak to whether the coded system handles exits any better than Colin did — that is a real
gap in the diagnostic tool itself, not evidence the system is strong there. Where the two *can* be
compared, they diverge sharply: SHARPE-adjacent content here (regime_id, tail_risk_fomc) scored
100%, because those categories are almost entirely deterministic threshold comparisons (VIX vs.
20/25/35, contango vs. backwardation) that the code executes exactly — a different task than
Colin's presumed conceptual/quantitative reasoning under time pressure. The one place this run's
weakness *rhymes* with a human failure mode is Petroulas conviction-sizing: a human under-primed
on "is this really arithmetic" would make the same mistake Kimi is trusted to avoid, which is
exactly why that judgment was pulled out of code and put in front of a model in the first place.

The one other honest caveat: `combat_vetoes.py` (3 `confirmation_protocol` questions) is a
config-disabled analysis instrument in production (net -143R if used as a blanket gate per its
own docstring) — the questions test its internal arithmetic correctly, but a system player
checking live config state would note these vetoes currently fire on nothing in production.
