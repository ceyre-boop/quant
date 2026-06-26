# Alta — Two-Layer Architecture

> **Status:** Design doctrine, grounded in a read-only audit of the live `sovereign-v2` tree
> (2026-06-24). This describes how the system *is wired today* and the wall it *should* hold, so
> the two are never confused. Companion: [`COMPONENT_CLASSIFICATION.md`](./COMPONENT_CLASSIFICATION.md)
> classifies every component against this frame.
>
> **This is doctrine, not a build order in progress.** Nothing here has been built or changed; the
> build sequence (§7) is the proposal. Read in ~15 minutes.

---

## 1. The frame — why two layers

Trading is two different problems wearing one coat. We keep losing money and clarity by treating
them as one. The split below is the fix.

### The market is not chess

Chess is *closed-form*: complete information, fixed rules, deterministic transitions, a finite
(if huge) game tree. A strong evaluation function plus search wins, because the position *is* the
truth and the truth is fully visible.

Markets are the opposite on every axis:

| Property | Chess | Market |
|---|---|---|
| Information | Complete | Asymmetric, partial, lagged |
| Rules | Fixed | Reflexive — your action moves the board |
| Transitions | Deterministic | Stochastic |
| Adversary | One, visible | Many, hidden, adaptive |
| Truth | The board | Unknowable until after the fact |

The consequence is hard and non-negotiable: **prediction is bounded near a coin flip.** The best
quant fund in history (Medallion) wins on something like a 51% hit rate. Our own honest numbers say
the same thing — the forex carry edge is real (permutation p<0.001) but its full-decade √n Sharpe is
**0.69**, and it only pays in rate-trending regimes (2022 +0.51 / 2023 +1.26 vs 2021 −0.13 /
2024 −0.09). *Predicting direction is a low-ceiling game and always will be.* Stacking more
predictive cleverness past the ceiling doesn't raise it; it overfits. Our own discovery test proved
this: 28 candidate edges mined off daily OHLC → **zero** survived the gauntlet. The carry edge is
the only real one in that data, and it is irreducible.

### But evaluation *is* chess-shaped

Once you are *in* a position, the problem changes character completely. "Given that I hold this
trade, at this price, with this much open risk, this drawdown, this volatility — what do I do now?"
is a **closed-form** problem:

- The inputs are bounded and observable (price, equity, drawdown, ATR, time-in-trade, threat state).
- The rules can be fixed in advance.
- The transitions are deterministic — *we* decide the exit, the size, the stop.
- There is a *correct* answer given the rules, and it is computable.

This is the distinction Jane Street draws as **"have a view on the price, not on the market"**, and
the one Druckenmiller lived by: *being right matters far less than how much you make when right and
how little you lose when wrong.* That second half — capture and protection — is pure evaluation, and
it is solvable.

### The split, and why it is multiplicative

> **Layer 1 — PREDICTIVE.** Decide *what to bet on and which way.* Edges + AI bias → setups.
> Inherently humble; ceiling ~55%. *This is the chess-isn't-real layer.*
>
> **Layer 2 — EVALUATIVE.** Decide *what to do about a position we already hold.* A deterministic
> exit / size / risk state machine. **No AI, no prediction, post-entry.** *This is the chess-shaped
> layer.*

Profit is not the sum of the two layers — it is the **product**:

```
Sharpe ≈ f(prediction quality) × g(evaluation quality)
```

A 52% predictive edge with disciplined evaluation (cut losers fast, let winners run, size to
conviction) is a profitable fund. The *same* 52% edge with sloppy evaluation (symmetric static
stops, flat size, no trailing) is a break-even-minus-costs grinder. **Most of the realizable Sharpe
lives in Layer 2, where the problem is actually solvable** — and that is exactly the layer we have
under-built. We poured years into shaving the predictive ceiling and left the evaluative engine
running in simulation only (see §3).

### Reference architecture: AlphaZero proposes, Stockfish disposes

This split is not novel — it is a translation of a documented, tested pattern from another
adversarial computational domain: **chess engines.** The two layers map onto the two engine
paradigms exactly.

| | **AlphaZero** (≈ Layer 1) | **Stockfish** (≈ Layer 2) |
|---|---|---|
| Output | Probabilistic — move distribution + confidence | Deterministic — one correct action |
| Method | Learned pattern recognition / intuition | Rule-based calculation to depth |
| Improves via | Training on outcomes | Logged code/eval changes only |
| Stance | "This favors LONG ~62%" | "Given this state, exit now" |
| Ambiguity | Handles it | Refuses it |
| Native to | The open problem | The bounded problem |
| Reads | Context | State |
| Verb | Forecasts | Executes |

The precedent is concrete, not a metaphor reaching for authority: the strongest modern engines
**hybridize** the two. Stockfish itself adopted a learned evaluation (NNUE) on top of deterministic
alpha-beta search in 2020 and got stronger; Leela (the open-source AlphaZero descendant) sits at the
learned-intuition pole. Learned proposal + deterministic calculation, fused, beats either pole alone.
We are translating that proven structure to trading, not inventing one.

**The handoff (this is the operating contract).** AlphaZero suggests *candidate* setups with
conviction → Stockfish evaluates each candidate against the current account / risk / regime / health
state → **vetoes or accepts** → runs every accepted trade to a deterministic exit. *AlphaZero
proposes; Stockfish disposes.* Crucially, **Stockfish's veto is on entries, not only exits** — a
high-conviction setup is still rejected if the account is in drawdown, an exposure cap is hit, or the
system is unhealthy. This is exactly Citadel's pod structure (analysts propose trades; the risk desk
holds veto authority), and it is already how our risk engine is wired (§3a).

**This is the line between apex and retail.** Renaissance, Jane Street, Druckenmiller, Citadel all run
*both* layers — pattern recognition under uncertainty **and** mechanical execution discipline. Retail
runs only the AlphaZero: hunches, reads, some of them genuinely good — with no Stockfish behind them,
so positions get managed on feel. AlphaZero alone produces lucky years and catastrophic years.
Stockfish alone (pure systematic, no predictive intelligence) produces mediocre returns. **The fusion
is the alpha**, and it is an architectural property, not an algorithmic one — which is why this
document is about *separation*, not a better signal.

### Relationship to the existing Time-Horizon Doctrine

This split is **orthogonal** to the existing Time-Horizon Separation Doctrine
(`TRADING_PHILOSOPHY.md`). That doctrine is a *horizontal* wall between **systems** (ICT intraday /
Forex multi-day / Equity) — no feature sharing across horizons. This document adds a *vertical* wall
*inside each system* between **predict** and **evaluate**. Both walls hold at once: ICT's predictor
must not borrow Forex's predictor (horizontal), and ICT's predictor must not reach into ICT's exit
machine (vertical). The two are complementary constraints, not competitors.

---

## 2. Layer 1 — PREDICTIVE *(the AlphaZero analog)*

**Purpose.** Answer one question: *should we have a position, and which direction?* Nothing about
how to manage it. This is the AlphaZero of the system: it recognizes patterns under uncertainty and
outputs a **confidence-weighted** read ("this pair favors LONG, conviction 0.62") — never a claim of
certainty. Its job is to be right *often enough* to feed the evaluator good candidates, not to be
right always. It cannot be deterministic and it accepts the ~55% ceiling as a fact of the domain.

**Inputs (state of the world before entry).**
- Validated macro/carry features — real-rate differential momentum, IRP z-score, cycle divergence,
  PPP z, Hurst regime (`sovereign/forex/macro_engine.py`).
- Regime context — Bull+VIX gate (HYP-027), VIX slope (`sovereign/forex/signal_engine.py`).
- AI bias — an Opus-synthesized directional read over news + lead-lag + volume profile
  (`sovereign/briefing/synthesize.py`).

**Outputs (a setup, or nothing).**
- Forex live: `MacroEngine.score_pair()` → `ForexSignal{direction ∈ {LONG, SHORT, NEUTRAL},
  conviction ∈ [0,1], hold_days, primary_driver}`.
- Forex backtest array form: `signal_engine.build_signal_frame()` → DataFrame
  `{signal ∈ {−1,0,+1}, hold_days, size_mult}`.
- AI bias: `synthesize()` → `{directional_bias, confidence 0–100, regime_call, key_level,
  narrative}`.

**Constraints.**
- Must be **humble**. Below the conviction floor it must emit NEUTRAL (no-trade), not a guess. The
  Buffett filter does exactly this: `conviction < CONVICTION_NEUTRAL_THRESHOLD (0.10, authorized
  2026-06-05) → NEUTRAL` (`sovereign/forex/strategy.py`, `macro_engine.py:192`).
- Every predictive feature must clear the promotion gates (IC>0.15 OOS, positive walk-forward
  marginal, no holdout degradation) before it can move the output. Unproven signals may be *recorded*
  but must not *gate trades*.
- It produces a setup. **It never specifies how the trade is managed** — no stop logic, no sizing,
  no exit rules cross this boundary outward (§4).

**Success criterion.** Directional hit rate *and* setup selectivity, measured out-of-sample. ~55% is
a ceiling, not a failure line. A Layer-1 that says "no trade" most days is working as designed.

**Current state.** *Partially built.*
- The validated carry predictor is **live and real** (`macro_engine.py`, `signal_engine.py`) —
  permutation p<0.001, the only edge that survived discovery.
- The AI bias predictor **exists but is wired to nothing.** `morning_market_briefing.py` is
  explicitly `provenance.verified=false` and the briefing is *"never ingested as a trading input"*
  (`scripts/morning_market_briefing.py:13,86`). It is a journal, not a gate.
- **The #1 Layer-1 gap:** there is *no* setup pipeline that emits a trade only when the validated
  edge **and** the AI bias **agree**. The two predictors run in separate rooms. (§7, build step 2.)

---

## 3. Layer 2 — EVALUATIVE *(the Stockfish analog)*

**Purpose.** Given a candidate (at entry) and a position (after entry), decide — deterministically —
the correct action: accept or veto the entry, size it, then exit / trail / scale / hold. Same state
in, same action out, every time. No intuition, no forecast, no feel.

**Layer 2 has two jobs, and they are in opposite states of completeness — this is the key finding:**
- **(a) Entry-veto + sizing — "Stockfish disposes."** Evaluate each candidate against account / risk /
  regime / health state; reject it or size it mechanically. **BUILT, LIVE, apex-quality** (§3a).
- **(b) Run-to-exit — manage the open position deterministically.** **SHELVED IN SIMULATION** — the
  full machine runs only in the backtester; live is a static bracket + poll (§3b). *This is the #1
  gap.*

The veto half is already what apex desks have. Only the management half is missing in production.

**Inputs (state only — never a forecast).**
- Price, OHLC, ATR / realized vol.
- Open-trade state: entry price, direction, bars-in-trade, best/worst price, unrealized PnL.
- Account state: equity, drawdown, open risk, consecutive losses.
- Threat state from the cross-system bridge (macro halt/tighten).

**Outputs.**
- Exit decisions: stop hit / trailing-stop hit / Donchian break / time exit / scale.
- Size at entry: `units = risk_usd / stop_distance` (`sovereign/forex/position_sizer.py`).
- A risk verdict: final position size after the 8-layer engine
  (`sovereign/risk/risk_engine.py::decide()` + `sovereign/risk/layers/*`).

**Constraints.**
- **No AI. No prediction. No directional opinion.** It does not ask "where is price going?" It asks
  "given the rules and the current state, what is the correct action?" — and computes it.
- **Deterministic and replayable.** Same state in → same action out, every time. This is what makes
  it testable, auditable, and trustworthy at 2am.
- It consumes the *fact* of a Layer-1 setup (we are long EURUSD because L1 said so) but consumes
  **none of L1's reasoning.** Conviction scores, AI confidence, narrative — none of it crosses
  inward (§4).

**Success criterion.** Given a fixed stream of Layer-1 setups, Layer-2 maximizes the captured Sharpe
— larger wins, smaller losses, drawdown control. Measured by replaying the *same* signals through
different evaluation rules and comparing.

### 3a. Entry-veto + sizing — "Stockfish disposes" — BUILT, LIVE, apex-quality

This half is already what a top desk has, and it needs no rework. `sovereign/risk/risk_engine.py::decide()`
is, by its own docstring, *"the SOLE sizing authority"*; *"every operation can only REDUCE risk"*;
*"the engine NEVER executes — it sizes and constrains only."* It runs a deterministic cascade —
`desired = base × vol × dd × regime; capped = min(desired, kelly, portfolio, prop); final = 0 if any
hard gate fires` — fail-loud, and audited to a decisions log.

The **veto** lives in Layer-0 `sovereign/risk/layers/gates.py::run_gates`, which forces `final_risk = 0,
size = 0` on six deterministic, state-only conditions: **daily-loss limit**, **max-drawdown buffer**,
**internal prop guard**, **health-not-ok** (this *is* Tenet 3 — "systems must know when they are
unreliable" — the engine vetoes itself when its heartbeat is down), **macro threat-critical** (from the
cross-system bridge), and **Monte-Carlo breach probability**. A high-conviction Layer-1 candidate is
still rejected if any fires. Sizing of accepted trades is mechanical: `position_sizer.size` →
`units = risk_usd / stop_distance`. **This is the Citadel propose/veto structure, already in the
codebase, deterministic and correct.** Leave it alone.

### 3b. Run-to-exit — SHELVED IN SIMULATION — the #1 gap

The other half — managing the open position after entry — is the apex engine that exists *only in the
backtester.* `sovereign/forex/fast_backtester.py::_simulate_forex_core()` is a sophisticated 6-state
deterministic machine: **stop**, **trailing-ATR**, **Donchian break**, **reversal**, **time**,
**cb_refresh**, plus **pyramiding** (scale-up into winners). It is everything the management half
should be — and it never runs in production.

**The live path has none of it.** `oanda_bridge.place_trade()` sets a *static* `stopLossOnFill` +
`takeProfitOnFill` (GTC) at the moment of fill, and that is the end of it. `ict/orchestrator.py::scan_once()`
opens the trade; `watch()` just re-runs `scan_once` looking for *new* entries. **There is no open-trade
re-evaluation loop, and no trade-modify endpoint anywhere in the live code** (grep-confirmed: no
trailing-stop amend, no partial close, no cb_refresh live). A live trade rides a fixed bracket until one
side is touched.

**The #1 Layer-2 gap, stated plainly:** the Stockfish *veto* is apex and live; the Stockfish
*run-to-exit* is shelved in simulation. We dispose correctly at the door and then manage on
autopilot with a static stop and a poll. Porting `_simulate_forex_core` to a live position-manager
*completes the Stockfish layer* and is the single highest-leverage build in the system (§7, build
step 1).

---

## 4. The hard wall

The wall runs between "what to bet" (L1) and "what to do about the bet we hold" (L2). It exists
because contamination in either direction destroys the property that makes each layer work.

### The handoff across the wall
The contract is one-directional and narrow: **suggestion → evaluation/veto → action → deterministic
exit.** AlphaZero (L1) hands across only a *candidate* — direction, the entry trigger, and the risk
geometry (stop distance) it implies. Stockfish (L2) hands back only a *verdict* — accept-and-size, or
veto. **No reasoning, probability, or narrative crosses inward; no forecast crosses that can override
the veto.** Stockfish vetoes *entries* (§3a), not only exits — conviction can propose a trade, but it
can never reopen a gate the risk engine closed.

### What Layer 1 may NOT do
- Set or move a stop, a target, or a trailing rule.
- Decide position size.
- Re-evaluate an open trade.
- Reach into the exit machine to "rescue" a losing position because the model still likes it.

### What Layer 2 may NOT do
- Form a directional opinion or predict.
- Read a conviction score, AI confidence, or narrative to decide an exit.
- Size a position on a forecast (size comes from *risk distance*, not from how sure we are — sizing
  to conviction is a deliberate, **bounded** L1→L2 input, see below, not a free-form forecast feed).

### The data that crosses, and the direction it crosses
Only the **fact** of a setup crosses L1 → L2: *direction, the entry trigger, and the risk geometry
(stop distance) the setup implies.* That is the bracket Layer 2 starts from. **No reasoning, no
probability, no narrative crosses.** Nothing crosses L2 → L1 at all — the evaluator never tells the
predictor what to think.

### The ONE sanctioned crossing (named, deterministic, logged)
There is exactly one place where a Layer-1 signal legitimately reaches into Layer-2: the forex exit
machine uses `signal_today` to trigger a **reversal** exit (the live signal flipped against the open
trade) and a **cb_refresh** continuation (the signal re-confirms the position)
(`fast_backtester.py:106–109`). This *is* a prediction input timing an exit — but it is permitted
because it is **deterministic** (a hard rule, `signal_today != direction → exit`; no model
discretion at exit time) and **logged** (the exit reason is recorded). It is the single allowed
breach, and it must stay deterministic. Any *new* desire to let prediction touch an exit must be
justified against this one precedent and logged the same way — or it is an anti-pattern (§6).

### Sizing-to-conviction is an allowed, bounded L1→L2 input
`size_mult` (`signal_engine.py`) and conviction-scaled holds let a *stronger* setup take a *larger*
position. This is allowed because it is **bounded and pre-registered** — conviction modulates a
multiplier within fixed limits; it does not hand Layer 2 a free-form forecast to act on. The line:
conviction may *scale* a size within bounds; it may never *originate* an exit decision (except the
one sanctioned crossing above).

### Parameter-change flow (applies to both layers)
No live parameter — a stop multiplier, a conviction floor, a risk limit — changes without a logged
rationale in `data/agent/param_change_log.jsonl` *before* it goes live (project rule #4). The 0.10
conviction floor is the model citizen here: lowered from 0.35, then retroactively authorized and
annotated 2026-06-05. The wall is enforced not just in code structure but in this change discipline.

### Isolation invariant (unchanged)
The existing `ict/ ↛ sovereign/` import isolation still holds and is orthogonal to this wall. The
two-layer wall lives *inside* each system; the isolation wall lives *between* the ICT pipeline and
the sovereign intelligence layer. Cross-layer logic routes through `ict-engine/orchestrator.py`,
never `ict/pipeline.py`.

---

## 5. Classification of existing components (summary)

Full per-file table in [`COMPONENT_CLASSIFICATION.md`](./COMPONENT_CLASSIFICATION.md). The shape:

**Layer 1 (predict).** `macro_engine.py`, `signal_engine.py` — *working, validated, the real edge.*
`briefing/synthesize.py` + `morning_market_briefing.py` — *working but display-only (the gap).*
`big_move.py`, `futures_bias.py` — *unvalidated, display-only.* `futures/decision_engine.py` —
*partial, bleeds into L2.*

**Layer 2 (evaluate).** `fast_backtester._simulate_forex_core` — *sophisticated, backtest-only;
must be ported to live (the gap).* `position_sizer.py` — *working; the sizing review folds in here.*
`risk_engine.py` + `risk/layers/*` (8 layers) — *working.* `oanda_bridge.py` live exits — *static
stop/TP + poll; the gap.*

**Infrastructure (serves both, predicts nothing).** `decision_logger.py`, kill switch
(`alta freeze`), `prove.py`, `param_change_log`, the hypothesis ledger, pre-registration, data
fetchers, the test suite.

**Research (not live; answering a measurable question).** VRP (Stage-2 NO_TRADES), regime-router
screen (NOT_SUPPORTED), HYP-027 re-validation, the discovery pipeline.

**Dead (graveyard / orphaned).** Killed hypotheses (HYP-044 VIX-13 tightening, REJECTED_OOS),
archived schedulers. *Note: `ict-engine/` is **not** dead — it is the isolation-safe bridge.*

### The two gaps, restated
1. **L2 gap (bigger):** the deterministic exit machine is not live. Production = static bracket +
   poll. **Highest-leverage fix in the system.**
2. **L1 gap:** no agreement-gated setup pipeline. The validated edge and the AI bias never have to
   agree before a trade fires; the AI bias gates nothing.

---

## 6. Anti-patterns (the failure modes this architecture forbids)

1. **AI sets or moves an exit.** Letting a model "decide" in real time whether to hold a loser is how
   discretionary blowups happen. Exits are deterministic rules over state. *The one exception is the
   logged, deterministic `signal_today` reversal/cb_refresh crossing — and it stays deterministic.*

2. **The exit machine is asked to predict.** If Layer 2 starts reading conviction or forecasts to
   choose an exit, it is no longer the chess-shaped solvable problem — it inherits Layer 1's
   coin-flip ceiling. Keep it on state only.

3. **One object carries both layers (the futures bleed).**
   `futures/decision_engine.evaluate_entry()` returns `EntryDecision{setup_type, direction` (L1) `+
   stop, target, contracts, confidence` (L2)`}` in a single object. This is the contamination
   pattern named: prediction and evaluation fused at birth, so neither can be tested or replaced in
   isolation. Forex keeps them apart; futures does not. Don't spread this pattern; refactor it.

4. **Sizing on a forecast.** Size comes from *risk distance* (`units = risk_usd / stop_dist`), bounded
   conviction-scaling aside. Sizing *up* because the model feels confident — beyond the pre-registered
   `size_mult` band — is letting a coin-flip predictor set leverage. That is how a 51% edge goes
   bankrupt on a losing streak.

5. **Improving the system by adding predictive cleverness past the ceiling.** Discovery already
   proved 28 candidates add nothing. The lever is Layer 2 (capture/protection), not a 29th signal.

6. **AlphaZero overrides Stockfish's veto.** Forcing a trade the risk engine halted — or upsizing
   past a ceiling — because the model "feels confident." This is the retail failure mode in its
   purest form: the predictor reopening a gate the evaluator closed. The deterministic veto (§3a)
   exists precisely so conviction *cannot* do this. Same in reverse: never widen a stop live to give a
   losing thesis "room to work." Propose freely; never override the dispose.

---

## 7. Build sequence (the proposal — nothing built yet)

Ordered by leverage. Each step is a separate, gated build with its own approval; this is the map,
not a license to start.

**Step 1 — Complete the Stockfish run-to-exit: port the exit machine to live (L2). [Highest leverage.]**
The Stockfish *veto* half (§3a) is already apex and live; this step finishes the Stockfish layer by
making its *management* half live too. Lift `_simulate_forex_core`'s exit rules — trailing-ATR,
Donchian, reversal, time, cb_refresh — into a live **position-manager** that re-evaluates each open
OANDA trade on every tick/bar and amends the order (trailing stop, scale, partial), replacing today's
static-stop-and-poll. *Why first:* the wall is already clean here, the logic already exists and is
tested in simulation, and most of the realizable Sharpe lives in capture/protection. This converts a
built-but-shelved asset into live edge. **Validation:** replay the live manager against the backtester
on identical signals → captured Sharpe must match simulation within tolerance before it goes live;
shadow before flipping.

**Step 2 — Build the agreement-gated setup pipeline (L1).**
A daily pipeline that emits a setup **only when the validated forex signal and the AI bias agree** on
direction; otherwise a logged no-trade. This finally gives the AI bias a job (a veto/confirm gate)
without letting it set exits or size. **Validation:** measure hit-rate and selectivity of
agreement-gated vs ungated setups out-of-sample — it must add selectivity, not just remove trades
(the HYP-044 trap). If it doesn't beat the ungated base, it stays a journal.

**Step 3 — Add PARTIAL_EXIT / SCALE_DOWN to the live manager (L2).**
Neither the live path *nor* the backtester currently takes partials. Banking a portion at +1R and
trailing the rest is textbook capture improvement and pure evaluation. Build it into the Step-1
manager once that is live and validated.

**Step 4 — Fold the sizing/leverage decision into L2.**
The `position_sizer` review (the `risk_adjusted_pnl_pct = pnl_pct × risk_pct` question) is a Layer-2
concern — it is about *how much*, not *which way*. It is **not** a free Sharpe win (it is
leverage-invariant on Sharpe; it scales the displayed dollar curve), so it is a deliberate risk
decision, logged via `param_change_log`, made *after* the exit engine is live so we are sizing a real
evaluation engine and not a static bracket. **Deferred until Step 1 lands.**

**Rationale for the order:** L2-first because the wall is already clean there, the highest-leverage
gap is there, and the logic already exists (lowest build risk, highest payoff). L1-agreement second
because it is genuinely new predictive plumbing and lower-ceiling by nature. Partials and sizing fold
into the live L2 engine once it exists. We do **not** chase a 29th predictive signal — discovery
proved that well is dry.

---

## Appendix — wall-location quick reference (firsthand, 2026-06-24)

| Concern | Layer | Analog | File · symbol | State |
|---|---|---|---|---|
| Carry/macro direction | L1 | AlphaZero | `sovereign/forex/macro_engine.py::score_pair` | working (validated edge) |
| Signal array + regime gate + size_mult | L1 | AlphaZero | `sovereign/forex/signal_engine.py::build_signal_frame` | working |
| AI directional bias | L1 | AlphaZero | `sovereign/briefing/synthesize.py::synthesize` | working, **wired nowhere** |
| Conviction floor (Buffett filter) | L1 | AlphaZero (humility gate) | `sovereign/forex/strategy.py` `=0.10` | working, logged 2026-06-05 |
| Entry veto (6 hard gates) | L2 | Stockfish — **dispose** | `sovereign/risk/layers/gates.py::run_gates` | **working, live** |
| Risk verdict / sole sizing authority | L2 | Stockfish — **dispose** | `sovereign/risk/risk_engine.py::decide` + `risk/layers/*` | **working, live** |
| Position size | L2 | Stockfish — dispose | `sovereign/forex/position_sizer.py::size` | working |
| Deterministic exit machine | L2 | Stockfish — **run-to-exit** | `sovereign/forex/fast_backtester.py::_simulate_forex_core` | sophisticated, **backtest-only (gap)** |
| Live execution / exits | L2 | Stockfish — run-to-exit | `sovereign/execution/oanda_bridge.py::place_trade` | **static stop/TP + poll (gap)** |
| Live scan loop | L2/infra | — | `ict/orchestrator.py::scan_once`, `watch` | opens trades; **no re-eval loop** |
| Sanctioned L1→L2 crossing | wall | propose→dispose | `_simulate_forex_core` `signal_today` reversal/cb_refresh | deterministic + logged (allowed) |
| Layer bleed (anti-pattern) | — | both fused | `sovereign/futures/decision_engine.py::evaluate_entry` | L1+L2 in one object |
