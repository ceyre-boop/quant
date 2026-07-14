# FVG × Fractal-Corridor Study — Charter (TICK-035 / HYP-098)

**Question.** Does intraday Fair-Value-Gap structure on NQ, conditioned on the Fractal
Corridors indicator (Colin's Pine v6, repo root), contain a tradable edge that beats the
shop's best sealed results — constitutional yield > 0.023%/day (HYP-093 at sealed sizing),
with holdout Sharpe reported against carry's 1.25 for context?

**Method.** THE RESEARCH METHOD, exactly (see OPTIMIZATION_PROGRAM.md): mine dirty inside
the fence → extract shared structure → prereg + hash with both priors → test on untouched
holdout → seal. Route B framing: intraday is NEW data; the sealed daily nulls (HYP-082
fractal, HYP-083 daily-FVG, ICT forex gates p=0.52, HYP-090 adaptivity) are disclosed
priors, not re-litigated.

**Windows.** Mining 2018-01→2024-06 (loader hard-truncates; fence-tested). Holdout
2024-07→2026-06 — one prior daily-resolution test on this span (HYP-095) disclosed in
family accounting.

**Causality rules (the two known traps, closed by construction).**
1. FVG detection and ATR are computed strictly from bars ≤ t (rolling trailing ATR,
   shifted; no whole-frame statistics — the `ict/fvg_detector.py` leak is not inherited).
2. A pivot of depth d exists for trading purposes only from bar (pivot_index + d) — the
   Pine `pvts()` confirmation delay, ported explicitly. Corridor lines use only the two
   most recent CONFIRMED pivots per side per depth, extrapolated to the current bar.
Look-ahead canary test required green before any mining number is read.

**Mining grid (counted in mined_n.json, stamped MINING):** entry families {formation-retest,
displacement-continuation, mitigation-fade} × corridor condition {none, inside-d2,
beyond-band-with, beyond-band-against, slope-agree} × killzone {LONDON, NY_OPEN, NY_PM, ALL,
per ict/session_classifier.py verbatim} × management {stop 0.5/1.0×gap; target 1R/2R/session-flat}
× FVG min size {0.5, 1.5 ATR} on 5-min bars. MNQ costs (0.87 pts/RT) on every fill.

**Discipline.** Every artifact stamped; every cell counted; bar never lowers; a
well-powered null is a deliverable; new families in later rounds are new structure, not
re-tuned parameters; execution-path freeze and Article 6 absolute; isolation whitelist
enforced by AST test; outputs only under data/research/fvg_corridor/.
