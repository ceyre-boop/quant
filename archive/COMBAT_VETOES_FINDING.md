# Combat Vetoes — Validation Finding (Phase 0 STOPPED)

**Date:** 2026-06-03 · **Verdict:** DO NOT wire the combat vetoes as a live selection gate.

## What Phase 0 proposed
Operationalize the forensic "combat rules" (`data/research/combat_rules.json`): force-skip setups
matching four conditions before the risk engine sizes them, on the claim that they recover **−273R**
(79.7%) of all historical losses — with **MACRO_AGAINST (C-001) = −158R, "46% of everything ever
given back."**

## What the honest replay found
`scripts/validate_combat_vetoes.py` reproduced the forensic's −273.23R **exactly** — confirming the
number is real *as the forensic defines it*. But the forensic defines it wrong: it sums only the
**losing** trades of each condition and never counts the **winners the same condition produced.**

Measuring **net expectancy** (winners + losers) on the same 863-trade forensic set:

| Condition | Skips | Losers avoided | Winners forgone | **Net** |
|---|---|---|---|---|
| C-001 MACRO_AGAINST | 386 | −158.4R (189) | **+243.6R (197)** | **+85.3R** |
| C-003 COUNTER_MOMENTUM | 353 | — | — | **+117.9R** |
| C-005 RATE_SIGNAL_WEAK | 195 | — | — | **+33.0R** |
| C-006 VOLATILITY_FLOOR | 158 | — | — | **+0.4R** |
| **All four (OR)** | **711 / 863 (82%)** | **+280.6R recovered** | **+424.0R forgone** | **−143.5R** |

- MACRO_AGAINST trades average **+0.221R/trade — *above* the whole-set average of +0.182R/trade.**
  Trading "against the rate differential" is **not** an anti-edge; it's slightly above average.
- Blanket-skipping all four conditions would skip **82% of every trade** and turn the system **−143R
  worse**, forgoing +424R of winners to avoid +280R of losses.

## Root cause
**Survivorship / selection-on-outcome bias.** The forensic conditioned on the outcome (losses) and
the setup, then attributed the loss to the setup — without checking that the setup also produced
(bigger) winners. "X% of losses came from condition C" says nothing about whether vetoing C helps; you
must measure C's **net expectancy**, which the forensic never did.

## Secondary issue
The forensic set is **7 pairs** (incl. AUDNZD, USDCAD, GBPJPY) — **not** the live v015 4-pair config.
Its conclusions wouldn't transfer to production even if the methodology were sound.

## What ships
- `sovereign/forex/combat_vetoes.py` — the condition module, retained as an **analysis instrument**,
  `enabled: false` so nothing sizes around the live system on it.
- `scripts/validate_combat_vetoes.py` — the replay that produced this finding (the corrected lens:
  net expectancy per condition).
- **No live wiring.** `forex_live_scan.py` and `macro_engine.py` are untouched. Monday's scan runs
  the proven edge + the validated risk engine, with no unvalidated selection gate degrading it.

## The honest next step (a real Phase 0, if pursued)
A proper selection study: per-condition **net expectancy on the v015 4-pair trades, out-of-sample**,
looking for a genuinely negative-expectancy *subset* (not a loss-only tail). Only conditions that are
net-negative OOS — and survive the same permutation/holdout discipline as every other edge — earn a
veto. The combat rules as written do not clear that bar.
