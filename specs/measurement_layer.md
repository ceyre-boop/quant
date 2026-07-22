# The Conscience — Measurement Layer Spec
## Alta Investments · specs/ · 2026-07-22
### The second organ of the nervous system: is the process being followed, and is the edge still true?

**Status:** SPEC (pre-code) · **Companion to:** `specs/system_regime_contract.md`
**Design source:** Part 2 of the trader-process analysis (the "great trader" measurement layer)
**Isolation:** built in `alta_platform/` — imports neither `ict/` nor `sovereign/`; reads their
journals/ledgers as data, writes one verdict file. Same wall discipline as the regime organ.
**Freeze-safe:** new files only; touches no execution path.

---

## Why this exists

The regime organ answers "is the market favorable for this edge right now." The conscience
answers the other half every serious trader needs: **is the process actually being followed,
and is the edge still true on live data?** Without it, a strategy can be favorable-by-regime
and quietly broken-by-drift, and nobody would know. This is the measurement layer that turns
"I think it works" into "the data says it still works — or it doesn't, so halt."

It runs on slow clocks (not every tick): a health pass each session + rollups weekly/monthly.
Its output is a single contract `data/agent/system_health_verdict.json` that every strategy
reads alongside the regime contract. **A strategy trades only when regime says FAVORABLE and
conscience says HEALTHY.** Two gates, both live.

---

## The verdict contract — `data/agent/system_health_verdict.json`

```json
{
  "generated_at": "...", "status": "OK | STALE | DEGRADED",
  "strategies": {
    "undertow_gapper": {
      "kill_switch": "TRADE | REDUCE | HALT",
      "reason": "...",
      "edge_health": {
        "live_expectancy_R": null, "backtest_expectancy_R": 0.23,
        "divergence_flag": false, "n_live": 3, "n_needed": 250,
        "status": "INSUFFICIENT_DATA"
      },
      "process_adherence": { "decisions_matched_rules_pct": null, "status": "..." },
      "forecast_vs_execution": { "read_accuracy": null, "execution_quality": null }
    },
    "carry": { ... }, "ict_equities": { ... }
  },
  "portfolio": { "consecutive_breaker_hits": 0, "data_integrity": "OK|FAIL", "kill_switch": "TRADE" }
}
```

## The five measurements (Part 2, made real)

1. **Kill switch (Part 2 #9) — the headline.** Per strategy + portfolio: TRADE / REDUCE / HALT.
   HALT conditions (pre-registered, config-driven, not editable mid-drawdown):
   - consecutive drawdown-breaker hits ≥ N, OR
   - data-integrity failure (stale regime/mirror, missing feed), OR
   - realized stats fall OUTSIDE the backtest confidence band, OR
   - live-vs-backtest execution divergence past threshold.
   REDUCE = a softer band (approaching a limit). This is the one that "prevents a blown
   account," so it ships first and it fails safe (missing data → HALT, never TRADE).

2. **Edge health (Part 2 #3, #6).** Live/shadow expectancy per setup class vs the backtest
   expectancy, with a divergence flag and an honest `n_live` vs `n_needed`. Below n_needed →
   `INSUFFICIENT_DATA` (never a confident "healthy" from 3 trades).

3. **Process adherence (Part 2 #7).** From the causal-chain journal: what fraction of live
   decisions matched the pre-registered rules. Process score, separate from P&L. This is the
   "review decisions, not outcomes" discipline made countable.

4. **Forecast vs execution (Part 2 #10).** Track "was the read right" separately from "was the
   fill good." Two numbers, never conflated — the whole point of the bias-blame critique.

5. **Cost/capacity realism (Part 2 #4).** Live realized cost per trade vs modeled; flag when
   net drifts from gross. (Reads fills; UNAVAILABLE until live fills exist.)

## Reader

`alta_platform/health_client.py`: `get_health(strategy) -> HealthRead` with `.kill_switch`,
`.edge_divergence`, `.stale`. Safe-by-default: missing/stale → HALT. A strategy's sizing does:
`if get_health(s).kill_switch == "HALT" or get_regime(s).verdict == "STAND_ASIDE": skip()`.

## Fail-loud, never fake (same as the regime organ)

Every measurement carries a status. INSUFFICIENT_DATA is a first-class value — the shadow has
3 signals, so edge_health is INSUFFICIENT_DATA, and the kill switch treats an unproven edge as
REDUCE (small), not TRADE (full). The conscience never green-lights on absent evidence.

## Build order

1. `alta_platform/measurement.py` + `health_client.py` + isolation test. Kill-switch skeleton
   that reads the regime contract's data-integrity + the drawdown breaker + consecutive-breaker
   count, writes `system_health_verdict.json`. Fails safe (no data → HALT/REDUCE).
2. Edge-health section: read the gapper shadow ledger + gauntlet backtest expectancy, compute
   live-vs-backtest with honest n. (Undertow first — it has a shadow.)
3. Process-adherence + forecast-vs-execution: read the causal-chain journal (needs the ICT
   setup ledger from Layer 8; UNAVAILABLE until it fills).
4. `scripts/build_system_health.py` + `scripts/com.alta.system_health_verdict.plist` (30-min,
   not installed — hand Colin the load command).
5. Wire ICT to read `get_health("ict_equities")` at the same gate it reads regime.

## Definition of done (this organ)

- `alta_platform/measurement.py` + `health_client.py`, isolation test green both directions.
- `system_health_verdict.json` written on a schedule, fail-safe.
- Undertow edge-health reads INSUFFICIENT_DATA honestly (3 shadow signals < 250 needed).
- Kill switch defaults REDUCE/HALT on missing data, never TRADE.
- ICT reads health at its sizing gate. NEXT.md updated.

---

*Alta Investments · specs/measurement_layer.md · v1.0*
*"The regime organ asks if the market is right. The conscience asks if you are. Trade only when both say yes."*
