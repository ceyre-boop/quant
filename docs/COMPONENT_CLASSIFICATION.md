# Alta — Component Classification

> Companion to [`ARCHITECTURE.md`](./ARCHITECTURE.md). Every significant component classified against
> the two-layer frame:
>
> - **L1 — Predictive** *(the AlphaZero analog)* · decides *what to bet and which way*
> - **L2 — Evaluative** *(the Stockfish analog)* · decides *what to do with a position we hold* (deterministic, no AI). Two jobs: **dispose** (entry-veto + sizing, live) and **run-to-exit** (position management, backtest-only).
> - **Infra** · serves both, predicts nothing
> - **Research** · not live; answering a measurable question
> - **Dead** · graveyard / orphaned
>
> *State* = working / partial / static / display-only / broken. *Must change* = what aligns it with
> the wall. Audited read-only against `sovereign-v2`, 2026-06-24. No code was changed.

---

## Layer 1 — PREDICTIVE *(AlphaZero analog)*

| Component | File · symbol | State | What must change |
|---|---|---|---|
| Macro/carry signal engine | `sovereign/forex/macro_engine.py::MacroEngine.score_pair` | **working** — the validated edge (perm p<0.001) | Nothing structural. Keep it humble (NEUTRAL below floor). Stale docstring says 0.35; live floor is 0.10. |
| Forex signal array + regime gate | `sovereign/forex/signal_engine.py::build_signal_frame` / `build_signal_arrays` | **working** | Emits `{signal, hold_days, size_mult}` + Bull+VIX gate. Clean. Keep stop/target geometry out of L1 reasoning. |
| Conviction floor (Buffett filter) | `sovereign/forex/strategy.py::CONVICTION_NEUTRAL_THRESHOLD = 0.10` | **working**, logged (authorized 2026-06-05) | None. This is the model for the param-change discipline. Fix the 0.35 comment drift in `macro_engine.py:11`. |
| Entry gating | `sovereign/forex/entry_engine.py` (uses the floor at :329) | **working** | None. Applies the L1 floor; does not manage exits. |
| AI directional bias (synthesis) | `sovereign/briefing/synthesize.py::synthesize` | **working but wired NOWHERE** | Give it a job: feed it into the agreement gate (ARCHITECTURE §7 step 2). Until then it gates nothing — `provenance.verified=false`. |
| Morning briefing assembler | `scripts/morning_market_briefing.py` | **display-only** — *"never ingested as a trading input"* (:86) | Becomes a real L1 input only via the agreement-gated setup pipeline. Today: journal, not gate. |
| Big-move classifier | `sovereign/.../big_move.py` (`alta_*`/display) | **display-only, unvalidated** | Stays display until it clears promotion gates. Must not gate trades. (DISPLAY-ONLY per project memory.) |
| Futures pre-market bias | `futures_bias.py` / ES-NQ bias inputs | **display-only / killed inputs** | The 5-input bias engine was killed (p=0.57). Don't revive inputs; infra reusable. |
| Futures decision (entry) | `sovereign/futures/decision_engine.py::evaluate_entry` | **partial — BLEEDS L1+L2** | Returns setup+direction (L1) **and** stop/target/contracts (L2) in one `EntryDecision`. Anti-pattern §6.3. Split prediction from evaluation so each is testable alone. |

---

## Layer 2 — EVALUATIVE *(Stockfish analog)*

> **Two jobs, opposite completeness.** The **dispose** half (entry-veto + sizing — `risk_engine.decide`
> + the gates + `position_sizer`) is **live and apex-quality**. The **run-to-exit** half
> (position management — `_simulate_forex_core`) is **shelved in the backtester**. The veto is what apex
> desks have; only the management half is missing live.

| Component | File · symbol | State | What must change |
|---|---|---|---|
| **Entry veto — "Stockfish disposes"** | `sovereign/risk/layers/gates.py::run_gates` | **working, live** | None. 6 hard gates → `final_risk=0` (daily-loss, max-dd buffer, prop guard, health-not-ok [Tenet 3], threat-critical, MC-breach). The deterministic entry veto. Leave it. |
| Risk engine (sole sizing authority) | `sovereign/risk/risk_engine.py::decide` / `RiskEngine` | **working, live** | None. "Sole sizing authority; every op can only REDUCE risk; never executes." The live **dispose** engine (Citadel propose/veto). |
| Deterministic exit machine (**run-to-exit**) | `sovereign/forex/fast_backtester.py::_simulate_forex_core` | **sophisticated — BACKTEST-ONLY** | 6 exits (stop / trailing-ATR / Donchian / reversal / time / cb_refresh) + pyramiding. **Port to a live position-manager** (§7 step 1) to complete the Stockfish layer. The #1 gap, highest-leverage build. |
| Backtest wrapper | `sovereign/forex/fast_backtester.py::simulate_forex_trades_arrays` | **working** (simulation) | Reference implementation for the live port. Keep as the validation oracle (live manager must replay-match it). |
| Position sizer | `sovereign/forex/position_sizer.py::PositionSizer.size` | **working** | `units = risk_usd / stop_dist`. The sizing/leverage review (`risk_adjusted_pnl_pct`) folds in here — §7 step 4, *after* the exit engine is live. Leverage decision, logged; not a free Sharpe win. |
| Risk layer — base size | `sovereign/risk/layers/base_size.py` | working | None. |
| Risk layer — drawdown | `sovereign/risk/layers/drawdown.py` | working | None. |
| Risk layer — gates | `sovereign/risk/layers/gates.py` | working | None — the veto layer detailed in the "Entry veto" row above. |
| Risk layer — Kelly | `sovereign/risk/layers/kelly.py` | working | None. |
| Risk layer — portfolio | `sovereign/risk/layers/portfolio.py` | working | None. |
| Risk layer — prop | `sovereign/risk/layers/prop.py` | working | None. |
| Risk layer — regime | `sovereign/risk/layers/regime.py` | working | None. |
| Risk layer — volatility | `sovereign/risk/layers/volatility.py` | working | None. |
| Live execution / exits (OANDA) | `sovereign/execution/oanda_bridge.py::place_trade` | **static — THE GAP** | Sets `stopLossOnFill`+`takeProfitOnFill` (GTC) at fill, then nothing. No trade-modify endpoint exists. Add an amend/trail/partial path driven by the live position-manager (§7 step 1 & 3). |
| Live scan loop | `ict/orchestrator.py::scan_once` / `watch` | **opens trades; NO re-eval** | `watch()` re-scans for *new* entries only. Add the open-trade re-evaluation call into the loop once the position-manager exists. |
| Paper execution | `ict/paper_trader.py` | working (practice) | Current live forex runs paper/practice. Position-manager must work in both paper and live OANDA. |
| Other venue bridges | `sovereign/execution/{tradovate,ctrader}_bridge.py`, `venue_router.py` | working (futures/CFD) | Tradovate has `close_partial` — reference for the forex partial-exit build (§7 step 3). |
| Partial exit / scale-down | *(does not exist for forex)* | **MISSING** | Build into the live manager after step 1. Bank at +1R, trail the rest. Pure evaluation. |

---

## Infrastructure (serves both layers; predicts nothing)

| Component | File · symbol | State | Notes |
|---|---|---|---|
| Decision logger | `sovereign/intelligence/decision_logger.py` | working | Captures entry reasoning; **must call `update_outcome()` at close** (project rule #2 — Oracle can't learn without it). |
| Kill switch | `alta freeze/thaw/status` → `data/system/KILL_SWITCH` | working | Soft-freezes trading path + hard-blocks approve_edge; monitoring/cognition stay alive. |
| Proof engine | `scripts/prove.py` | working | The honesty oracle — full-decade √n Sharpe 0.69, OOS 1.25, decay/permutation. |
| Param-change discipline | `data/agent/param_change_log.jsonl` | working | No live param changes without a logged rationale first (project rule #4). |
| Hypothesis ledger | `data/agent/hypothesis_ledger.json` | working | Permanent record; the graveyard. |
| Pre-registration | `data/research/preregister/*`, `_methodology_ok` | working | Freezes specs before looking — the HYP-044 antidote. |
| Canonical runner | `scripts/run_hypothesis.py` | working | Costed IS/OOS + permutation + BH + decay gate. The gauntlet. |
| Oracle (cognition) | `sovereign/oracle/{oracle_cycle,reflect_cycle}.py` | working | Reads logs → 1 lesson/day. **Not a trading input** — cognition only. |
| Data fetchers | `scripts/fetch_fred_economic.py`, `fetch_macro_cache.py`, `harvest_daily_panel.py`, `sovereign/forex/data_fetcher.py` | working | Read-only data assembly. Panel feeds the factory, never the trading path directly. |
| Cross-system bridge | `sovereign/intelligence/cross_system_bridge.py` | working | The only sanctioned inter-system channel (macro threat). Feeds L2 threat-state, not L1. |
| Capital allocator | `sovereign/intelligence/capital_allocator.py` | working | Regime/health-based sizing multipliers. L2-adjacent (sizing throttle). |
| Test suite | `tests/` (ICT 21/21, isolation test) | working | Enforces `ict/ ↛ sovereign/` isolation and pipeline invariants. |

---

## Research (not live; each answers a measurable question)

| Component | File · symbol | Verdict | Notes |
|---|---|---|---|
| VRP / iron-condor | `sovereign/research/vrp/*`, `scripts/validate_vrp.py` | **DATA_INSUFFICIENT** | Stage-2 NO_TRADES (1-SD condor needs ~$135k/contract on $100k). Blocked on ThetaData options sub. |
| Regime-router screen | `scripts/regime_screen.py`, `data/research/preregister/regime_router_screen.json` | **NOT_SUPPORTED** | Bull+VIX cross-pair routing — no clean signal (commit 908b46f). |
| HYP-027 re-validation | discovery regime track + `regime_screen.py` | **inert/decayed** | Two independent tests agree the deployed gate is inert (delta_p=0.318). |
| Discovery pipeline | `sovereign/discovery/*`, `data/discovery/*` | **0 VALID_EDGE of 28** | Carry edge is irreducible; daily-OHLC mining exhausted → second edge needs NEW DATA, not cleverer mining. |
| Research panel | `data/research/panel/*` | working (collecting) | The data asset for future hypotheses. Routes through the factory, never bypasses it. |

---

## Dead (graveyard / orphaned)

| Component | Status | Notes |
|---|---|---|
| HYP-044 (VIX-13 tightening) | **REJECTED_OOS** | In-sample +0.242 → OOS 0.000. Do not revisit without new evidence (graveyard rule). |
| ES/NQ 5-input bias engine | **KILLED** (p=0.57) | Inputs sub-base-rate. Infra reusable; inputs dead. Holdout 2024-25 untouched. |
| HYP-007 per-pair hold overrides | **ROLLED BACK** (NOT_SIGNIFICANT) | Fails walk-forward; ledger V007-HOLD-VALIDATION. |
| Overnight-QQQ diversifier | **REJECTED** as carry diversifier | Real standalone edge but re-couples with carry in COVID crash. Don't re-explore as a diversifier. |
| Archived schedulers | orphaned | `agent_scheduler.py` and similar — archived, not in the live loop. |

### Not dead — clarification
`ict-engine/` is **not** dead. It is the isolation-safe cross-layer bridge
(`ict-engine/orchestrator.py`) — the only sanctioned ICT→sovereign entry point, imported by the
paper/live runners. Don't graveyard it. (Note: a prior memory flagged `ict-engine/` as orphaned vs
the launchd-loaded `ict/` package — the *scanner daemon* uses `ict/`, but `ict-engine/orchestrator.py`
remains the canonical cross-layer bridge per project rule #1. Verify the live import path before
touching either.)

---

## Cross-references

- Architecture & the wall: [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- The constitution (six tenets, time-horizon doctrine): `../TRADING_PHILOSOPHY.md`
- Project invariants (isolation, Oracle loop, conviction sizing, param logging): `../CLAUDE.md`
- Honest performance numbers: `scripts/prove.py` output (0.69 full-decade / 1.25 OOS, regime-fragile)
