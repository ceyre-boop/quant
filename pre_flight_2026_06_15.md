# Pre-Flight Verification — 2026-06-15

Read-only audit + authorized dry-run / kill-switch test, run Sunday before the next
market open. Honest readiness assessment. **No live config was modified.** System ends
thawed and unfrozen.

---

## GO / NO-GO ASSESSMENT

### 🟢 SYSTEM IS GO — WITH CAVEATS

**0 RED. 3 YELLOW.** Nothing blocks the open. The caveats below are either expected
weekend behavior or items to confirm, not faults. Read the **Known Risks** section — they
are real and live.

---

## GREEN ITEMS

- **Scheduled loops** — all five plists exist, `state = active`, last exit code `0` where run.
  `com.alta.health.responder` is firing on its 30-min cadence (16 clean passes, e.g. 01:28 →
  01:58 → 02:28 → 02:58 UTC). `com.alta.research.factory` and `com.alta.hypothesis.generator`
  are loaded and ran correctly on manual invocation; neither has hit its first *scheduled*
  fire yet (4h interval / nightly 03:00 PT) — expected, not a fault.
- **Conviction gate** — `sovereign/forex/strategy.py:12` = `0.10`, matching the 2026-06-05
  authorization (the later of two log entries that day re-authorized 0.10 with the
  BELOW_PROVEN_BAR tagging condition).
- **Per-trade risk ceiling** — `risk_config.yaml` `ceiling: 0.010` (1% hard clamp). Prop layer
  `enabled: true`, `max_drawdown_pct: 0.08`.
- **Kill switch** — `data/system/KILL_SWITCH` absent (unfrozen). Full freeze→FROZEN→thaw→RUNNING
  cycle **verified live** (Phase 5): frozen scan reported `SYSTEM FROZEN (soft)` and placed
  nothing; `alta status` returned `🟢 RUNNING` after thaw.
- **OANDA practice** — `.env` has `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`, `OANDA_LIVE=0`,
  `OANDA_BASE_URL`. Live API reachable; **NAV $109,819.26** (practice, account
  101-001-…-001). Credentials valid (confirmed by the successful Phase-6 call).
- **Open positions reconciled** — 2 open trades, both authorized against `oanda_fills.jsonl`:
  | Trade | Pair | Dir | Units | Entry | Stop | uPL | Held |
  |-------|------|-----|-------|-------|------|-----|------|
  | #83 | GBP_USD | LONG | 10000 | 1.33631 | 1.32894 | +$76.30 | ~4d |
  | #29 | USD_JPY | LONG | 10000 | 159.491 | 158.907 | +$37.60 | ~14d |
  No orphan/unauthorized positions.
- **Forex scan dry-run** — completed clean, 4 pairs evaluated (USDJPY/GBPUSD/AUDUSD/EURUSD),
  0 signals (valid NO_SIGNALS on weekend/stale Friday-close data). Sub-0.10 convictions
  (AUDUSD 0.080, EURUSD 0.063) correctly fell below the gate.
- **oracle_daily_summary.json** — ~20h old (<24h). Refreshes on tonight's Oracle cycle.

## YELLOW ITEMS

1. **Portfolio cap is 6%, audit expected 8%.** `risk_config.yaml` has `max_portfolio_heat: 0.06`
   (correlated-risk cap) *and* prop `max_drawdown_pct: 0.08`. These are different mechanisms,
   not a single 8% portfolio cap. *Why yellow:* expectation vs config mismatch. *Action:* confirm
   6% heat is intentional (it likely is — tighter than prop drawdown by design). *Time:* 5 min review.
2. **`com.alta.futures.bias` ES=F fetch fails on the weekend.** The futures (ES/NQ Track-2) bias
   job logs `RuntimeError: No price data returned for ES=F` — yfinance returns nothing for the
   continuous future when markets are closed. *Why yellow:* unverified whether the weekday
   pre-market (06:00 PT Mon) run gets data; this is the futures sandbox, not the forex Monday
   path. *Action:* watch Monday's 06:00 PT run; if it fails, the futures monitor stays NEUTRAL
   (no scalps) but **forex is unaffected**. *Time:* observe Monday, ~0 effort.
3. **Dispatch queue holds 2 stale-loop items — both weekend false positives.** `forex_scan`
   (58h) and `morning_briefing` (57h) flagged DATA_STALE. Both plists are **weekday-only**
   (Weekday 1–5); last ran Fri 06-12; next run is the upcoming weekday session. The Health
   Responder dispatched them correctly given its input, but `loop_health` thresholds (30h/26h)
   don't account for weekends. *Why yellow:* noisy queue, not a real outage. *Action:* **none
   before open** — both self-resolve on the next weekday fire. *Recommendation (not applied):*
   make `loop_health` weekend-aware for weekday-only loops. *Time:* ~20 min, post-open.

## RED ITEMS

None.

---

## Honest Known Risks (live, acknowledged — not "100% ready")

- **The forex edge is regime-fragile.** Validated real (permutation p<0.001) but it only pays in
  rate-trending regimes (rolling walk-forward: 2021 −0.13 / 2022 +0.51 / 2023 +1.26 / 2024 −0.09).
  Flat regimes lose. This is a known live risk, not a bug.
- **Recent performance is poor — the system is already de-sizing.** The dry-run reported
  `Readiness: REDUCE — Recent WR 0%: losing-streak protocol`. The risk engine is correctly in
  reduce-size mode. Expect smaller positions until win rate recovers.
- **The 0.10 conviction gate admits BELOW_PROVEN_BAR trades.** Authorized for data collection;
  sub-0.35 trades are tagged experimental so the Oracle can test that band separately. These are
  not proven-edge trades — they are instrumented bets.
- **Single human operator, no redundancy.** No second person and no automated failover covers a
  bad fill, a stuck loop, or a broker outage during the session. The autonomous layer dispatches
  and proposes; it does not act on the trading path.

---

## Phase-by-phase verdicts

| Phase | Scope | Verdict |
|-------|-------|---------|
| 1 | Five scheduled loops | 🟢 (futures.bias 🟡) |
| 2 | Critical config | 🟢 (portfolio cap 🟡) |
| 3 | Data freshness | 🟢 (bias_log weekend gap, account_summary by-API) |
| 4 | Forex scan dry-run | 🟢 0 signals, clean |
| 5 | Kill-switch test | 🟢 full cycle verified |
| 6 | Open positions | 🟢 2 trades, reconciled |
| 7 | Dispatch queue | 🟡 2 weekend false positives, no action |

*System state at end of audit: thawed, unfrozen, RUNNING. No live config changed.*
