# Prop-Funnel EV Simulator — Summary Report (TICK-022)

Seed 7 · env {'numpy': '2.4.4', 'platform': 'macOS-26.3.1-arm64-arm-64bit-Mach-O', 'python': '3.14.4'} · parity: ALL EXACT

## The two questions this tool was built to answer

**1. "Can a strategy pass a sim prop test 100 times?"** — Only with a TRUE Sharpe well above anything this firm has proven. See `charts/frontier_pass_*.png`: P(pass 100/100) requires per-attempt pass ≥ 99.99%%… realistically Sharpe ≳ 2 at low vol — and at low vol the same configuration cannot produce meaningful monthly income. The verdict table's `P(pass 100/100)` column shows every real strategy's number.

**2. "Can it make $10k/month consistently for years?"** — See `P($10k every mo ×12)` in the table and `charts/tension_*.png`: the pass contour and the income contour do not overlap at survivable ruin risk until TRUE Sharpe ≳ 2-3, or until capital is much larger (income scales with account size; a $10k month on $100k = 10%/mo).

> **Assumes i.i.d. attempts/months — FALSE under regime shift. p^100 in particular treats 100 attempts as independent draws of the same edge; a regime that kills the edge kills all remaining attempts at once.**

## Honest input inventory

- CARRY (PROVEN, regime-fragile): decade Sharpe 0.69, OOS-window 1.25, fresh 2025-26 ≈ flat. The forward-band SCENARIO rows {0, 0.69, 1.25} bracket what carry might be going forward.
- ICT (UNPROVEN p=0.52): backtest pools only. The live sample is its own finding: **27 closed outcomes, 3W/24L** vs backtest WR ~63.6%% (selection-biased, LOW_N).
- FUTURES ORB (UNVALIDATED): n=2 replay trades → INSUFFICIENT_DATA row; Phase R (operator-gated) can regenerate a larger replay pool.
- SYNTHETIC frontier: requirements map, existence not claimed.

## Program EV reality

Best PROVEN-strategy cell: **carry_oos × MFF_100K** → program EV ≈ $2138.0/mo at P(funded) 0.6802 (pricing UNVERIFIED_PRICING). Set against the caveat that fresh-window carry measured ≈ flat — the SCENARIO S0 row is the pessimistic bracket.

## Open items for Colin

1. **Pricing**: every fee/payout number is UNVERIFIED_PRICING — verify against live firm pricing before acting on EV rankings.
2. **Return-scale convention**: carry rows use the monte_carlo_prop convention (R = pnl_pct/risk_pct). The equity-curve display convention is ~100x smaller in dollars.
3. **rules_engine.py divergence (documented, not fixed)**: its `dd_trail_stops_at_starting` caps the floor at initial−dd from day one, making its 'trailing' effectively static. Parity presets mirror it; TOPSTEP/APEX rows use real trailing semantics.
4. **Intraday-trailing bracket**: APEX-style rows use the pessimistic κ-stressed bound (no MFE data on daily pools).
5. **Zero-edge rows show positive EV/mo — read carefully**: the funded account is effectively a call option (payouts keep positive months, the firm eats drawdowns), and the reset-to-initial payout policy plus unverified pricing make that option value model-OPTIMISTIC. Real firms' consistency rules, payout minimums and pricing exist precisely to close this; do NOT read S0-positive-EV as free money.

## Sizing policy results

- carry_oos × MFF_100K [PROVEN_REGIME_FRAGILE]: best challenge 3.0x / funded 2.0x → $4207.0/mo (see charts/sizing_*.png). objective = program EV/month; the best cell typically sizes the CHALLENGE hot and the FUNDED account cooler — verify constraint columns before acting
- carry_oos × FTMO_100K_SWING [PROVEN_REGIME_FRAGILE]: best challenge 3.0x / funded 2.0x → $3833.0/mo (see charts/sizing_*.png). objective = program EV/month; the best cell typically sizes the CHALLENGE hot and the FUNDED account cooler — verify constraint columns before acting
- ict_window_B × MFF_100K [UNPROVEN]: best challenge 3.0x / funded 1.5x → $4112.0/mo (see charts/sizing_*.png). objective = program EV/month; the best cell typically sizes the CHALLENGE hot and the FUNDED account cooler — verify constraint columns before acting

## Artifacts

- `verdict_table.md` / `.csv` — every strategy × firm, sorted by program EV/mo
- `charts/frontier_pass_*.png`, `charts/frontier_income_*.png` — requirements maps
- `charts/tension_*.png` — the pass-vs-income-vs-ruin contours
- `charts/sizing_*.png` — policy grids · `charts/days_to_pass.png` — time cost
- `results.json` — full machine-readable output · `parity/parity_report.json`