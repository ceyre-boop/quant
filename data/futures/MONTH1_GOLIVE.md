# MES/MNQ — Month-1 Go-Live Discipline

> Created 2026-06-07. The gate between "backtest works" and "live money survives" is execution
> integrity, psychological drift, and undiscovered costs — not the strategy. Month 1 is paper-only;
> its job is to measure paper-vs-live divergence and stress the infrastructure. Live capital is a
> month-2 decision, gated by the criteria below. **No exceptions.**

## Go / No-Go gates for Month-2 (live capital) — all six must be GREEN
- [ ] Paper-vs-backtest **fill divergence < 20%** of expected edge/trade, over **30+ trades**
- [ ] Oracle **calibration accuracy > 55%** directional at the 1-day horizon
- [ ] **Prop-firm trailing-drawdown sim: PASSED** (real paper equity curve vs Apex/TopStep/MFFU rules)
- [ ] **IBBarFeed live for 10+ consecutive sessions** with no silent failures
- [ ] **150-trade falsification gate** reached, or on a clear trajectory
- [ ] **No unresolved infrastructure surprises** from Week-2 stress testing

If all six green at month end → fund. If two or more red → run month-2 on paper and fix. No exceptions.

## 4-Phase month framework
- **Week 1 — Baseline calibration.** Don't touch the strategy. Log every trade: backtest expected
  fill vs actual IB paper fill; expected vs realized slippage; Oracle morning call vs what actually
  set up. If fill divergence > 20% of expected edge/trade → STOP, the cost model is wrong.
- **Week 2 — Stress the infrastructure.** Deliberately break it on paper: IB Gateway disconnect
  mid-session (reconnect? ghost position?); early-close holiday schedule (does the ORB timer fire on
  bad data?); bar-feed gaps (fail loud, or silently corrupt VWAP?). Discover these on paper.
- **Week 3 — Oracle calibration.** Run `scripts/futures_calibration.py` every morning, not just at
  month end. If macro directional bias is right < 55% at 1-day, the ORB direction filter is noise.
- **Week 4 — Prop-firm sim.** Run the exact Apex/TopStep/MFFU rules against the ACTUAL paper equity
  curve (not simulated). If you'd have hit trailing DD, not ready. If passed, real data point.

## The 6 things that kill live traders in month 1
1. **"It was working" bias.** Flip a bias ONLY when its predefined falsifier is hit — never on PnL.
2. **Cost-drag compounding.** Know the minimum trades/day where the strategy is still net-positive
   after costs. Run the validate script at 25% / 50% / 75% of current entry frequency.
3. **Positional memory.** At what single-trade $ loss does your behavior change? That's your real
   position-size ceiling — not the Kelly fraction.
4. **Single points of failure.** IB → yfinance fallback is fine for replay. For live: what if
   yfinance is also down? What if the Oracle crashes at 9:28 before ORB? Document failure modes +
   manual overrides before you need them.
5. **Session-discipline decay.** Log a one-sentence rationale BEFORE every entry. Accountability
   paper trading alone never creates.
6. **The 150-trade falsification gate.** Respect it. A 10-trade hot streak is not a reason to fund.

## Falsifier protocol (IMPLEMENTED + verified 2026-06-07)
Every session's directional bias must carry a numeric `key_levels.invalidation` in
`data/futures/oracle_mornings.jsonl`. `sovereign/futures/scalp_strategy.kill_level` takes the
**soonest** of that oracle falsifier and the rules level (overnight high for SHORT / low for LONG);
`bias_invalidated` flips the bias to NEUTRAL once price crosses it. SHORT dies ABOVE, LONG dies BELOW.

- **Today (2026-06-07, MES SHORT):** oracle falsifier **7460** ("reclaim and hold above 7460
  invalidates bearish continuation") vs overnight-high backstop 7578.75 → **kill = 7460 (binding)**.
  Verified: intact at 7400.5, flips NEUTRAL above 7460. **Accepted as-is.** ADR already 105% used —
  bounces likely sharp; entry timing matters more than direction.

## Open finding — conviction is dual-sourced (month-1 reconciliation item)
Sizing reads `data/futures/bias_log.jsonl` (rules engine) → today **conviction 1**, VIX-capped
("VIX 21.5 — use conviction 1 max"). `oracle_mornings.jsonl` (LLM) says **conviction 2** but is read
ONLY for the falsifier price, never sizing. The conservative VIX-capped conviction governs size
(correct) and `below_proven_bar` (conviction < 2) fires correctly. Benign now, but two bias
generators producing divergent reads is a source-of-truth smell — reconcile in month-1.

## Month-1 infrastructure build backlog (NOT yet done — build before/during month 1)
1. **PAI notify gateway** for macro triggers — phone alert when a trigger fires headless. Critical
   before live; build first (else you babysit a screen or miss entries).
2. **IBBarFeed swap** in the live monitor — stop running fills on yfinance bars (data ≠ fills is a
   live-money integrity issue). Then ≥ 1 full week paper on the real IB feed before any live capital.
3. **Paper-vs-backtest divergence metric/dashboard** — one number per session (expected edge vs
   realized edge, %), watched weekly. This is the metric that predicts prop-eval survival.
4. **Prop-firm rule sim harness** — Apex/TopStep/MFFU rules vs the actual paper equity curve.
5. **Conviction single-source-of-truth** — reconcile bias_log (rules) vs oracle_mornings (LLM).

## Immediate next action each session (the discipline that separates systematic from gambling)
Before every replay/session: confirm the day's bias carries a numeric falsifier in
`oracle_mornings.jsonl`, and that `kill_level` reflects it. Today: done (7460).
