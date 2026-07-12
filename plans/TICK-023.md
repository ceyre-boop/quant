# TICK-023 — HYP-090 "MODERN": pre-registered adaptive walk-forward study

**Approved:** 2026-07-11 (plan-mode approval, Claude Code / Molly session).
**Master plan (full detail):** `Plans/glistening-juggling-clover.md` (same dir, case-insensitive APFS).
**Prereg (THE LAW once P0 lands):** `data/research/preregister/HYP-090_modern_adaptive_params.json`.

One-paragraph summary: Colin's recurring adaptive-parameters idea ("MODERN": daily trailing-window
parameter sweeps + regime map), tested end-to-end once at maximal scope (full surface incl. pair
selection) to seal the family with a receipt. Precompute-then-replay over 5,775 variants; arms
A0 static / A1 recent-winner / A2 regime-matched / A3 placebo floor; gauntlet = block-bootstrap
+ BH m=6 + DSR at n_trials=5,775 + A3 envelope + per-year non-degrade + switching-cost criterion;
abort on reconcile drift (0.6886±0.01, never re-tuned). Registered prior: NOT_ROBUST. Prior kills
disclosed: HYP-065/066/067, 180-config exit sweep, regime router. Read-only study — outputs only
`data/research/modern/`; never touches live params (`monthly_reopt.py` is the anti-pattern).

Build order: P0 prereg+ticket → P1 reconcile-gate→freeze→signals → P2 1,540 kernel runs + M2M
decomposition → P3 selection engines + look-ahead tests → P4 replay+gauntlet+verdict+report+push.
One [RESEARCH] commit per phase. 12 tests. End-to-end runtime < 1 hour, single core.
