# HYP-071 Step 2 — Tabular Exit Value Function · Validation Report
**Computed:** 2026-07-23 · **Status:** PROVISIONAL — read with Colin against the locked §7 gate
**Harness:** `scripts/research/hyp_071_exit_value_function.py` · **Results:** `data/research/HYP-071_tabular_exit_value_results.json`
**Prereg (locked):** `data/research/preregister/HYP-071_tabular_exit_value.yaml` (hash `3d500bda…`, verified)
**Addendum (locked):** `data/research/preregister/HYP-071_interpretation_notes.yaml` (hash `c1fab80…`, verified)

---

## The one-line result

The value table was computed and it **PASSED the locked §7 gate — provisionally**: it reconciles
exactly, the sensible structure it finds is CPCV sign-stable, and that structure agrees across the
2023-24 OOS and 2025-26 forward windows. **This contradicts the pre-registered expectation of
NOT_SIGNIFICANT** (the "4th confirmation of the data-ceiling thesis"). That makes the result
interesting — and makes the caveats below load-bearing. **Do not treat this as CONFIRMED. It is a
provisional PASS on GROSS returns that has a specific, plausible way of being an artifact.**

---

## What ran (the locked non-negotiable sequence, all cleared)

| Gate | Result |
|---|---|
| Prereg v2 hash | OK (`3d500bda…`) |
| Addendum hash | OK (`c1fab80…`) |
| Reconcile gate (decade portfolio Sharpe = 0.6886 ± 0.01) | **0.6886** — exact |
| Re-trace parity vs canonical ledger | **459/459 (100%)**, 0 dropped |
| Canonical decade ledger restored after run | yes |

Config: block length L=5, 10,000 resampled continuations/cell, 60-day ATR percentile window,
λ=0.5 downside penalty. 108-cell board (ATR tercile × excursion × hold-fraction × RSI-extreme × carry
alignment); 54 cells are carry-not-aligned → N/A by construction (REVERSAL), leaving the evaluated half.

## The numbers

| Metric | Value | Locked threshold | Verdict |
|---|---|---|---|
| Decade members / forward members | 2111 / 246 | — | forward window is THIN |
| EXIT_NOW cells (decade) | 45 | — | — |
| CPCV-stable, economically-sensible EXIT_NOW divergences | **10** | ≥1 | pass |
| …of those, forward-consistent (agree 2023-24 ↔ 2025-26) | **9** | ≥1 | pass |
| Forward agreement fraction | 0.870 (n=23) | qualitative | supports |
| Separability (hiking ↔ cutting) agreement | 0.862 (n=29) | qualitative | supports |
| Regime-window (60d ↔ 252d) agreement | 0.854 (n=48) | ≥0.90 for "robust" | **FALSE — below bar** |

## The structure it found (the 9 forward-consistent EXIT_NOW divergences)

Every one of these cells has `static_action = HOLD_AND_TRAIL` today; the table says **EXIT_NOW**.
All are **carry-aligned, non-RSI-extreme** cells. The pattern is coherent and economically sensible:

| cell | ATR | excursion | hold | n | margin (V_hold−V_exit) | CPCV stable |
|---|---|---|---|---|---|---|
| 72 | high | underwater | early | 70 | −0.214 | yes |
| 84 | high | modest | early | 154 | −0.239 | yes |
| 76 | high | underwater | mid | 80 | −0.134 | yes |
| 88 | high | modest | mid | 103 | −0.177 | yes |
| 92 | high | modest | late | 58 | −0.104 | yes |
| 44 | mid | underwater | late | 32 | −0.078 | yes |
| 56 | mid | modest | late | 60 | −0.084 | yes |
| 8  | low | underwater | late | 39 | −0.054 | yes |
| 20 | low | modest | late | 75 | −0.069 | yes |

Read plainly: **high-ATR positions (any hold stage) and late-hold lower-ATR positions that are
underwater-to-modest should exit rather than hold-and-trail.** This is exactly the economic prior in
the vision doc ("when volatility spikes… the right move is probably to take it") — which is both
reassuring (mechanism-plausible) and a reason for suspicion (a prior that confirms itself is the
easiest kind to fool yourself with).

## Why this is PROVISIONAL, not CONFIRMED — the load-bearing caveats

1. **GROSS returns (the big one).** R is gross by locked design. The harness's own caveat:
   *"swap/carry would shift marginal EXIT_NOW cells toward EXIT."* But note the direction carefully —
   these 9 divergences are all **carry-aligned** cells. Positive carry income is a reason to **HOLD
   longer**, so folding in correctly-modelled carry would push marginal cells **back toward HOLD**,
   i.e. it could **erode** this PASS. And carry financing in this repo is known to be mis-modelled
   (~10× too small, sign flip on EUR-short — memory `project_swap_calibration`, TICK-024). Until the
   table is recomputed with corrected net financing, the count of surviving EXIT_NOW divergences is
   **not trustworthy**. This is the single most important reason not to act on it.

2. **Regime-window robustness FAILED its own bar.** 60d-vs-252d agreement is 0.854 < the 0.90 the
   summary flags as "robust". The optimal policy is somewhat sensitive to the ATR-percentile window
   choice — a sign the structure is not rock-solid.

3. **Thin forward window.** 246 members across 2025-26. Forward agreement of 0.870 on n=23 common
   cells is encouraging but not decisive; a handful of cells flipping would change the story.

4. **Interpretation caveat (locked addendum).** Vol innovations are ~white. Any structure here is
   **excursion geometry conditioned on entry regime, NOT intra-trade vol momentum** — the table is
   not discovering a predictable vol path; it is describing the shape of P&L excursions given the
   regime you entered in. That is a weaker, though still usable, claim.

## Verdict

**PROVISIONAL PASS by the letter of the locked §7 gate**, with the honest reading that it is most
likely a **gross-returns artifact** that the corrected-carry recompute (caveat #1) may erase. It does
**not** cleanly confirm the data-ceiling thesis, but neither does it earn a CONFIRMED stamp. The
correct next research step (freeze-independent, runnable now) is to **recompute the identical table
on NET returns** using the corrected financing model from TICK-024, and see how many of the 9
survive. If they survive net-of-carry AND the regime-window agreement clears 0.90, this becomes a
real finding worth a ledger CONFIRMED. If they collapse, the data-ceiling thesis stands and the
unlock is new data.

**No ledger verdict is sealed by this session** — this is flagged PROVISIONAL and left for Colin to
adjudicate, per the research method (prereg → untouched test → *human seals the survivor*).

## Freeze status — APPLYING this is FROZEN

Computing and validating the table is allowed research and is done. **Applying it to the live exit
machine is FROZEN** until 2026-07-28 **and** an explicit Colin ledger stamp. The specific rule-cell
changes it implies are staged, unapplied, in `research/HYP-071_STAGED_EXIT_RULE_PROPOSAL.md`. Nothing
in `forex_exit_manager`, `decide_exit`, `exit_machine`, or any config was touched.
