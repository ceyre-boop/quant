# System Self-Play — COLD Run (Hardcoded-Threshold-Blind) — 2026-07-26

## Why this run exists

The prior run (`system_heatmap.json` / `system_heatmap_summary.md`, 97/100) let the player use
any numeric threshold the question text handed over, even when that number lives ONLY in Python
source with no config backing (Petroulas's Kimi/XGBoost gates, combat-veto deadbands,
`CONVICTION_NEUTRAL_THRESHOLD`, VIX regime lines). That wasn't a cold test of the system's
*principled* knowledge — it was closer to "can you recite the number we just gave you." This run
does the honest version: **abstain wherever a question's correct answer genuinely depends on a
threshold that lives only in code, with no config file backing it.**

A separate, independent parallel run exists at `data/cursus_honorum/alphazero_run_01.json` (not
touched, not read into this script) with the same warm-run caveat — preserved as-is per instruction.

## Threshold provenance check (done before scoring, not after)

| Config file | What it externalizes | Verdict |
|---|---|---|
| `config/risk_constitution.yaml` | Art.1 0.75% per-trade, Art.2 2.5% carry-heat, Art.3 3.5/5/6.5 ladder | **Config-backed** — proper pattern |
| `config/parameters.yml` | grade_risk table (A+/A/B/C sizes), `max_daily_loss_pct`, `min_training_samples: 200` | **Config-backed** — proper pattern |
| `config/ict_params.yml` | scoring weights, `atr_spike_veto_multiplier`, `min_grade_to_execute`, kill zones, over-confirmation penalty | **Config-backed** — proper pattern |
| `sovereign/forex/config/combat_vetoes.yaml` | C-001/C-005/C-006/C-003 deadband/weak_rate/atr_floor/momentum_opp | **Config-backed** (module itself `enabled: false` in prod) |
| *(none)* `sovereign/forex/carry_engine.py` | `MIN_CARRY_SPREAD_BPS=100`, `CARRY_RISK_PER_PAIR=0.003`, `ATR_STOP_MULTIPLE=3.0` | **Hardcoded only** — no `import yaml`, no config read |
| *(none)* `sovereign/forex/strategy.py` | `CONVICTION_NEUTRAL_THRESHOLD=0.10`, `CONVICTION_FULL_SIZE=0.70`, grade-boundary numbers (2.0/1.5/0.60) | **Hardcoded only** |
| *(none)* `sovereign/forex/risk_sentiment.py` | `VIX_RISK_OFF_THRESHOLD=25.0`, `VIX_RISK_ON_THRESHOLD=20.0`, `VIX_EXTREME_THRESHOLD=35.0`, ±1.0 term-structure band (inline, unnamed) | **Hardcoded only** |
| *(none)* `imbalance_engine/petroulas_gate.py` | `MIN_COMPOSITE_STRESS=6.0`, `MIN_XGB_CONFIDENCE=0.65`, `MIN_KIMI_MAGNITUDE=7`, `MIN_KIMI_CONVICTION=7`, `NORMAL_SIZE_PCT=1.5`, `PETROULAS_BASE_PCT=3.0`, `PETROULAS_MAX_PCT=5.0` | **Hardcoded only** |
| doc-only | cross-system IC promotion bar `0.15` (comment in `sovereign/forensics/latent_feature_search.py`) | **Hardcoded/documented only**, no yaml |

Full list with per-question usage: `system_heatmap_cold.json.hardcoded_threshold_findings`.

## Score

**79/100 correct, 21 abstained, 0 wrong.** Every answered question is still correct
(`answered_accuracy_excl_abstentions = 1.00`) — the cold player never guessed and never
rubber-stamped; it either derived the answer from a config-backed number / doctrine / structural
fact / ledger record, or it abstained.

## Per-category

| Category | n | Cold acc. | Abstain | What changed vs. warm run |
|---|---|---|---|---|
| carry_direction | 8 | 75% | 2 | DIAG-01, DIAG-05 are genuine 100bp-floor boundary cases |
| carry_mechanics | 5 | 80% | 1 | DIAG-09's 0.3%×4=1.2% arithmetic needs the hardcoded per-pair constant |
| confirmation_protocol | 7 | 100% | 0 | combat-veto deadbands are config-backed |
| cot_interpretation | 7 | 100% | 0 | no hardcoded magic-number dependency |
| evidence_epistemics | 8 | 100% | 0 | ledger facts, not tunable code constants |
| graveyard_discipline | 1 | 100% | 0 | N>=200 is config (`min_training_samples`) |
| ict_reference | 7 | 100% | 0 | all gates config-backed (`ict_params.yml`) |
| isolation_discipline | 8 | 88% | 1 | DIAG-41's 0.11-vs-0.15 IC boundary is the doc-only bar |
| **petroulas_conviction** | 7 | **0%** | **7** | **every gate value (stress/XGB/Kimi/size ceiling) is hardcoded only** |
| **regime_id** | 7 | **29%** | **5** | **VIX 20/25/35 lines and the ±1 term-structure band are hardcoded only** |
| risk_constitution | 9 | 100% | 0 | fully config-backed (`config/risk_constitution.yaml`) |
| sizing_conviction | 7 | 57% | 3 | conviction-neutral floor + grade-boundary derivation are hardcoded only |
| state_vector | 5 | 100% | 0 | structural/definitional, no tunable threshold |
| tail_risk_fomc | 6 | 67% | 2 | 2 of 6 need the hardcoded EXTREME_RISK_OFF line; the rest are principle or config (`max_daily_loss_pct`) |
| tenet_mapping | 8 | 100% | 0 | doctrine mapping, no numeric threshold involved |

## Colin's 8-bucket projection (same caveat as the warm run: scheme not found in this repo)

| Bucket | n | Cold acc. | Abstain |
|---|---|---|---|
| EXPECTANCY | 16 | 100% | 0 |
| SIZING | 12 | 75% | 3 |
| CARRY | 13 | 77% | 3 |
| **KELLY** | 13 | **46%** | **7** |
| **SHARPE** | 13 | **46%** | **7** |
| **EXITS** | **0** | undefined | — (no bank coverage, unchanged from warm run) |
| RECOVERY | 3 | 100% | 0 |
| TRUST | 30 | 97% | 1 |

## Verdict

Cold and blind, the system's real gaps sharpen into exactly two modules: **`imbalance_engine/petroulas_gate.py`** (0% — every gate that decides a 3-5% conviction-sized position lives as a bare class attribute with zero config backing) and **`sovereign/forex/risk_sentiment.py`** (29% — the VIX 20/25/35 regime lines and the term-structure band that decide RISK_ON/OFF/EXTREME/carry-unwind overrides are equally bare). Both control real position-sizing and regime-override decisions, and neither can be audited, tuned, or reasoned about from outside the Python source — a parameter change to either requires an engineer to edit and redeploy code rather than a reviewable config diff. That is precisely the failure mode CLAUDE.md's "never hardcode thresholds, use `config/parameters.yml`" rule exists to prevent, and this diagnostic found it living undisturbed in two of the highest-stakes decision points in the whole system (position size and regime override), plus smaller footholds in `strategy.py`'s grade/conviction boundaries and `carry_engine.py`'s floor/sizing constants.

By contrast, `RISK_CONSTITUTION.md`'s caps and `ict_params.yml`'s scoring gates are the system doing this correctly — both fully externalized to a reviewable YAML twin, both scored 100% cold. That contrast is the actionable finding: the pattern to follow already exists in the codebase twice; it just wasn't applied to carry, regime, sizing, or Petroulas.

This is a materially different — and more useful — signal than the warm run's near-perfect
score. 97/100 said "the system's documented logic is internally consistent." 79/100 (with 0
wrong) says "when the system's own hardcoded numbers aren't handed to it, the specific place its
knowledge lives ungoverned is Petroulas conviction-sizing and the VIX/regime engine." Neither run
touches Colin's human EXITS/SHARPE gap directly — the bank still has no exit-logic category — but
the cold SHARPE-bucket score (46%, driven by regime_id's hardcoded VIX lines) is the first result
in either run that lands anywhere near Colin's own SHARPE 33%, and for a completely different
reason: Colin's gap was presumably conceptual, this run's is that the classification boundary
itself is not derivable from any config a reviewer could point to.
