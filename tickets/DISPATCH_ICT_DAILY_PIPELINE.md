# DISPATCH — ICT Daily Pipeline Build
## Alta Investments · Work Order for Claude Code
### Priority: High · Touches: `ict/`, `ict-engine/`, `execute_daily.py`

---

## ⚠️ SHELVED / RE-SCOPED — 2026-07-24

**This spec is MIS-SCOPED and will not be built as written.** Colin has clarified ICT's
intended role in the system, and it contradicts the premise of the eight-layer predictive
pipeline below.

ICT (Inner Circle Trader concepts, learned via the TJR bootcamp) was the first framework
built into this system — the "first eyes" for reading the market. Its purpose is NOT
prediction and NOT edge generation. It is a **reference layer for setting take-profit and
stop-loss, and occasionally entries.** ICT levels are useful because so many other traders
watch them — self-fulfilling reference points that make discretionary TP/SL/entry choices
easier — not because the structure predicts direction. This matches the repo's own
evidence: the ICT pattern edge is NOT PROVEN (permutation p=0.52, fails BH; see
`CLAUDE.md` current-live-state and `plans/restoration-ledger.md`). The system has since
moved toward more complex theory (carry macro, self-play, Petrules); ICT is deliberately
deprioritized as a signal source.

Framing ICT as an 8-layer event-driven predictive/signal pipeline (Regime Map → Level
Engine → Bias Engine → Proximity Controller → Entry Engine → Position Manager → Risk →
Journal, below) was the wrong scope for what ICT is for in this system. **Do not build
`scripts/ict_daily_pipeline.py`.** The absence of that script is intentional steady
state, not an unfinished gap — see `tickets/DISPATCH_DAILY_INTELLIGENCE_PIPELINE.md`
Phase 2e, which already skips gracefully when the file doesn't exist.

The existing ICT code that provides reference levels (`ict/pipeline.py`,
`ict/fvg_detector.py`, `ict/sweep_detector.py`, `ict/session_classifier.py`,
`ict/daily_bias.py`, `ict/memory_engine.py`) stays as-is and untouched by this
re-scoping — this note only retires the *build spec* below, not the shipped code.

The original spec is preserved below for history — do not resurrect it without a fresh
Colin-approved dispatch that scopes ICT correctly as TP/SL/entry reference, not signal
generation.

---

## What You Are Building

A complete, production-quality daily execution pipeline for the ICT/level-centric
trading system. Every morning the pipeline wakes up, runs eight layers in sequence,
and either places a trade or logs exactly why it didn't. No guesses, no stubs, no
hardcoded placeholders reaching live sizing.

The architecture is described below, layer by layer. Build all of it. Nothing less.

---

## What Exists Now (Read This Before Touching Anything)

`execute_daily.py` currently runs three phases:

**Phase 1 — Pre-market checklist.** Checks Alpaca connection, model files, risk config,
ledger writability. Non-blocking — it logs FAIL but continues. This behavior is intentional
and must be preserved.

**Phase 2 — Kill zone execution.** Calls `SovereignOrchestrator` on trinity assets
(META, PFE, UNH). Builds a `SovereignFeatureRecord` with hardcoded placeholders for most
features (Hurst=0.55, RSI=50, etc.). Computes real ATR via `ForexSignalEngine._compute_atr_pct`
(the ATR fix landed 2026-07-20 — do not touch it). Refuses to size off a None ATR. Routes
through `SovereignOrchestrator.run_session()`.

**Phase 3 — Post-session report.** Reads ledger, logs summary. Currently minimal.

**`ict/pipeline.py`** exists and implements the ICT causal chain: session classifier →
sweep detector → FVG detector → market structure → PD alignment → risk gate → `ICTSignal`.
Scores are calibrated (market_structure and pd_alignment zeroed after confirmed anti-edge
results — do not re-enable them). Pipeline is isolated: zero imports from `sovereign/`.

**`ict/daily_bias.py`** exists. Produces `PairBias` per pair (LONG/SHORT/NEUTRAL,
confidence 0–1, blackout bool). Reads Library context and ForexFactory calendar.

**`ict/memory_engine.py`** exists. Clusters historical scan results, returns `MemoryMatch`
with expected outcome and historical win rate. Requires ≥20 closed trades to activate.

**`ict/sweep_detector.py`, `ict/fvg_detector.py`, `ict/session_classifier.py`** — all exist.

**The carry forex strategy (`sovereign/forex/`)** is separate, frozen, and untouched by this
build. This build is ICT/equities pipeline only.

---

## The Eight Layers — Build Exactly This

### Layer 0 — Canonical Price Store

Right now `execute_daily.py` calls `feed.get_latest_bar(symbol)` for a single bar.
That is not enough.

Build a `PriceStore` class (`ict/price_store.py`) that:

- Fetches OHLCV for a configurable lookback (default: 200 daily bars, 100 4H bars,
  200 15m bars) from the Alpaca feed at session start.
- Stores all timeframes derived from the same raw tick/bar source. A 4H bar is
  resampled from the 15m feed — not fetched independently — so they cannot disagree.
- Exposes `get_bars(symbol, timeframe)` returning a clean pandas DataFrame with
  lowercase OHLCV columns and a UTC datetime index.
- Logs a hard error and skips the symbol if fewer than `min_bars` rows are available.
  No silent short-window calculations.

Wire `PriceStore` into `execute_daily.py` so it runs once at session start and
every downstream component reads from it.

### Layer 1 — Regime Map

Build a `RegimeMap` class (`ict/regime_map.py`) that runs on a slow clock (called
once at session start, not per-bar):

- **Trend/range/transition classification** per symbol: use ADX from the daily bars.
  ADX > 25 = trending, ADX < 20 = ranging, in between = transitioning. Source these
  thresholds from `config/ict_params.yml` — not hardcoded.
- **Volatility percentile**: compute ATR14 percentile against the trailing 252-day ATR
  distribution. Output: LOW / NORMAL / HIGH / EXTREME (quartile-based).
- **Correlation cluster state**: for instruments that share a currency (yen pairs,
  dollar pairs), compute pairwise 20-day rolling correlation. If correlation > 0.80
  across two instruments, flag them as the same cluster today. Output is a dict mapping
  each symbol to a cluster label.

`RegimeMap` output is a `RegimeState` dataclass per symbol:
```python
@dataclass
class RegimeState:
    symbol: str
    trend_label: str          # TRENDING | RANGING | TRANSITIONING
    vol_percentile: str       # LOW | NORMAL | HIGH | EXTREME
    correlation_cluster: str  # cluster_id or 'INDEPENDENT'
    adx: float
    atr14: float
    atr_percentile: float     # 0–1
```

Downstream use: the regime label determines which timeframe pairing is active (see
Layer 2), how wide stops need to be, and whether historical level stats should be
trusted (HIGH/EXTREME vol = tighten sizing, not tighten stops).

### Layer 2 — Level Engine

The existing `ict/fvg_detector.py` and `ict/sweep_detector.py` detect features.
They do not produce scored, stateful level objects. Build `ict/level_engine.py` to
close that gap.

A **level** is not a line. It is an object with state. Build a `Level` dataclass:

```python
@dataclass
class Level:
    symbol: str
    level_tf: str               # '1D' | '4H' | '15m' — timeframe that produced it
    level_type: str             # 'FVG' | 'ORDER_BLOCK' | 'SWEEP_POINT' | 'SR'
    price_low: float
    price_high: float
    midpoint: float
    formed_at: datetime
    state: str                  # PENDING | APPROACHED | ACTIVE | CONSUMED | INVALIDATED
    touch_count: int
    rejection_magnitude: float  # average rejection % on historical touches
    quality_score: float        # 0–10
    last_touched: Optional[datetime]
```

`LevelEngine.scan(symbol, price_store, regime_state)` produces a list of `Level`
objects for the current session. Scoring:
- Untested levels score higher than touched levels.
- Older levels decay (time-since-formation penalty, configurable in `ict_params.yml`).
- Rejection magnitude at past touches is the strongest quality signal.
- Higher timeframe levels score higher than lower timeframe levels.

Timeframe pairing rule (hard, not configurable at runtime):
- A level formed on 1D or 4H is only valid for entries on 15m.
- A level formed on 1H is only valid for entries on 5m.
- No other pairings. If the pairing ratio is outside the configured band, the level
  is marked INVALIDATED before it reaches the entry engine.

Level lifecycle: a level's state is updated each session run. A level that price has
traded through (not just touched) is CONSUMED. A level that has not been retested in
60 sessions is INVALIDATED. These are not deleted — they are archived. The journal
(Layer 8) needs them.

### Layer 3 — Bias Engine

`ict/daily_bias.py` already exists and produces directional scores. The gap is that
its output is currently treated as a veto: if bias is against the trade, the trade dies.

**Change the contract.** Bias is a confluence weight, not a gate.

Modify the pipeline to consume `PairBias.confidence` as a multiplier on position size:
- Aligned bias (trade direction matches bias): `size_multiplier = 1.0 + (confidence × 0.5)`
  → max 1.5× size at full confidence.
- Opposed bias: `size_multiplier = 1.0 - (confidence × 0.5)` → min 0.5× size.
- Neutral bias: `size_multiplier = 1.0`.
- Blackout: `size_multiplier = 0.0` → trade blocked entirely. This is the only case
  where bias kills a trade. All other cases are sizing modulation only.

Document this contract change with a comment in `pipeline.py` explaining why: a level
with a confirmed reaction signal at a high-quality scored level is tradeable even
against bias — bias tells us to size down, not to sit out. The level is the primary
signal. Direction is a confluence weight.

Wire the new multiplier into `MicroRiskEngine` — the sizing pipeline must receive it
before computing final position size.

### Layer 4 — Proximity and Patience Controller

Build `ict/proximity_controller.py`. This is the component that enforces patience
architecturally rather than relying on discipline.

`ProximityController.arm(levels, current_price)` takes the scored level list and
returns only the levels within the approach zone (configurable: default 1.0× ATR
from midpoint). All other levels are PENDING — the entry engine never sees them.

The controller runs on every bar during the session. It promotes levels from PENDING
to APPROACHED when price enters the zone, and fires an event to the entry engine.
Chasing is impossible because the entry engine is not called until the controller fires.

The controller must also enforce **event proximity suppression**: if a Level 3+ economic
event (as tagged by the calendar feed in `daily_bias.py`) is fewer than 15 minutes
away, no level is promoted regardless of price proximity. Add this as a configurable
parameter (`pre_event_blackout_minutes` in `ict_params.yml`).

Log every arm/disarm decision with the level ID, distance, and reason.

### Layer 5 — Entry Engine

`ict/pipeline.py` already implements the core ICT causal chain. The gap is that it
runs as a one-shot scan, not as an event-driven listener that wakes on proximity events.

Refactor `ICTPipeline` to expose:

```python
def on_level_approached(self, level: Level, bars_15m: pd.DataFrame) -> Optional[ICTSignal]:
    """
    Called by ProximityController when a level is approached.
    Runs the full pipeline on the paired lower timeframe.
    Returns ICTSignal or None.
    """
```

Stop placement is structural, not fixed-pip. Stop goes beyond the level boundary
(FVG low/high or sweep point) by 1.0× ATR. Never a fixed-pip stop. Source
the ATR from `PriceStore`, not recomputed inline.

Target is the next opposing level from `LevelEngine`, not an arbitrary R multiple.
If no opposing level exists within a configurable range, the setup is discarded.

Compute R:R from structural stop and structural target. Below the minimum threshold
(source from `ict_params.yml`, default: 1.5R), discard the setup and log why.

The existing grade system (A+/A/B/C/VETOED) is preserved. Only A+ and A grades
proceed to position sizing. B and C are logged for the journal (Layer 8).

### Layer 6 — Position Manager

This is the most underdeveloped part of the current system. Build `ict/position_manager.py`.

The position manager runs on a slow clock after entry — checking every N minutes for
exit conditions. It does not poll on every bar. Frequency: configurable, default 5m.

Events that trigger position manager decisions:

- **Minor level approach**: if price approaches a minor opposing level before target,
  evaluate partial exit (scale out 50% of position, move stop to break-even).
- **Break-even trigger**: once price moves 1R in the direction of the trade, move stop
  to break-even. Hard rule, no exceptions.
- **Trail trigger**: once price moves 2R, begin trailing the stop at the most recent
  swing point on the 15m timeframe.
- **Re-entry handling**: if a level gives a reaction that stalls at break-even and then
  price revisits the level with another rejection signature, log it as a re-entry
  candidate. Do not auto-enter the re-entry — flag it for the next proximity check.
- **EOD close**: if the position is open at a configurable pre-close time (default:
  15 minutes before market close), close it at market regardless of P&L. No overnight
  holds on ICT setups unless explicitly configured per-symbol.

All position manager decisions are logged with the triggering condition, price at decision,
and action taken.

Wire `update_outcome()` on `decision_logger` at every close event. The Oracle cannot
learn without this. Missing `update_outcome()` calls are silent data loss. This is
non-negotiable — it is CLAUDE.md Non-Negotiable #2.

### Layer 7 — Risk and Execution

The existing `MicroRiskEngine` handles per-trade sizing. Extend it with two new checks:

**Correlated exposure cap**: before sizing any new trade, check if any open position
shares a correlation cluster with the proposed symbol (from `RegimeMap`). If the
combined notional exposure of the cluster exceeds the per-cluster cap (source from
`RISK_CONSTITUTION.md` / `config/parameters.yml`), reduce the new position to fit
within the cap. Log the reduction.

**Daily drawdown breaker**: track intraday P&L across all ICT positions. If intraday
loss exceeds the daily drawdown limit from `RISK_CONSTITUTION.md`, set a session-level
flag that causes `ProximityController.arm()` to return an empty list for the rest of
the session. No more trades that day. Log the trigger and the P&L at the time.

Do not hardcode any risk numbers. Every cap comes from config.

### Layer 8 — Journal and Feedback Loop

Extend the trade ledger to capture the full causal chain for every setup evaluated,
not just every trade taken. Schema:

```python
{
  "setup_id": str,
  "symbol": str,
  "timestamp": str,
  "level_id": str,
  "level_type": str,
  "level_tf": str,
  "level_quality_score": float,
  "regime_state": RegimeState.as_dict(),
  "bias_state": PairBias.as_dict(),
  "bias_aligned": bool,
  "size_multiplier": float,
  "ict_grade": str,
  "component_scores": dict,
  "r_r_computed": float,
  "action": str,   # ENTERED | DISCARDED | VETOED | BELOW_MIN_RR | NO_OPPOSING_LEVEL
  "discard_reason": Optional[str],
  "outcome": Optional[str],  # filled by update_outcome()
  "outcome_r": Optional[float],
}
```

Write every evaluated setup to `data/agent/ict_setup_ledger.jsonl` — not just entries.

This makes the central diagnostic possible: was it the level or was it the bias?
You can now answer that with data. Every session's post-report should compute:
- Setups by grade vs. setups that traded
- Win rate by level_type
- Win rate by regime_state.trend_label
- Win rate aligned vs. opposed bias

These four numbers are the weekly review inputs.

---

## Daily Execution Flow (The New `execute_daily.py`)

After this build, `run_full_session()` runs in this order:

```
1. PriceStore.load()                     — fetch all timeframes, one source of truth
2. RegimeMap.scan()                      — regime label per symbol
3. LevelEngine.scan()                    — scored level objects with state
4. DailyBiasEngine.get_biases()          — directional scores + blackouts
5. ProximityController.arm()             — watch for price entering approach zones

   [session loop — runs until close or daily drawdown breaker fires]
   5a. On approach event → Entry Engine  — ICT causal chain + R:R gate
   5b. On entry → PositionManager starts — trail/partial/break-even on timer
   5c. On close → decision_logger.update_outcome()

6. post_session_report()                 — grade distribution, win rate splits, P&L
```

The monthly re-optimisation block that exists at the top of `main()` is preserved
and untouched.

---

## Hard Constraints (These Are Not Negotiable)

These come from `CLAUDE.md` non-negotiables. Build to them, not around them.

1. **ICT isolation is absolute.** `ict/` and `ict-engine/` must never import from
   `sovereign/`. New files in `ict/` follow the same rule. Cross-layer logic goes
   through `ict-engine/orchestrator.py` only. The existing test
   `test_pipeline_does_not_import_sovereign` must pass after every commit.

2. **`update_outcome()` must fire on every trade close.** Wire it in `PositionManager`
   on every exit path: target hit, stop hit, trailing stop, break-even, EOD close.
   Missing calls are silent oracle degradation. If you cannot wire it, stop and say so.

3. **No hardcoded risk numbers.** Every cap, every threshold, every timeout reads from
   `config/parameters.yml` or `config/ict_params.yml`. If a value needs to be added to
   one of those files, add it and log the addition to `data/agent/param_change_log.jsonl`
   with rationale. That log entry is required before the parameter is used anywhere.

4. **Shadow mode first.** The execution path freeze is active. Any new component that
   touches `forex_exit_manager`, `decide_exit`, or anything importable by the live
   execution path requires an explicit unlock recorded in `NEXT.md` before you touch it.
   ICT components are new, so this primarily means: do not import into or be imported
   by the existing forex sovereign stack without an explicit unlock.

5. **No silent failures.** If a feed is unavailable, a model is missing, or a required
   input returns None — log the specific failure and skip the symbol. Never substitute
   a placeholder that reaches sizing. The ATR fix (2026-07-20) is the precedent.

6. **Spec the Level object lifecycle before building the state machine.** Write
   `specs/level_lifecycle.md` describing all states and transitions before any code
   touches Level state. The spec ships in the same commit as the code. This is the
   spec-first rule from CLAUDE.md applied to the only genuinely novel state machine
   in this build.

---

## What NOT to Touch

- `sovereign/forex/` — carry strategy, frozen
- `sovereign/oracle/reflect_cycle.py` — except to call `update_outcome()` correctly
- `ict/pipeline.py` scoring weights — current weights are calibrated. Do not adjust
  `market_structure` or `pd_alignment` weights. They were zeroed for cause (HYP-024,
  HYP-034). Log entries if you think they're wrong; do not change them.
- `config/parameters.yml` risk caps — do not change values, only add new keys if needed
- `execute_daily.py` ATR logic — the 2026-07-20 fix is correct as-is

---

## Commit Protocol

One commit per layer that fully passes tests. Do not bundle all eight layers into
one commit. The test suite gate:

```bash
python3 -m pytest tests/ -k "ict and pipeline" -v
```

Baseline: 4 failed / 23 passed (pre-existing failures documented in
`plans/restoration-ledger.md`). Your commits must not add new failures.

Every commit: `[ICT]` prefix, imperative mood, reference the layer number.
Example: `[ICT] Add PriceStore canonical multi-timeframe feed (Layer 0)`

Push after each layer. An unpushed branch is a single-machine point of failure.

---

## Definition of Done

- All eight layers implemented and wired into `execute_daily.py`
- `python3 -m pytest tests/ -k "ict and pipeline"` passes at or above baseline
- `test_pipeline_does_not_import_sovereign` passes
- `specs/level_lifecycle.md` exists and matches the implementation
- Every trade close calls `decision_logger.update_outcome()`
- No hardcoded risk numbers in any new file
- `data/agent/ict_setup_ledger.jsonl` receives entries on paper runs
- `NEXT.md` updated with: what shipped, push confirmation, any new pre-registered
  hypotheses generated by this build (level quality scoring is a candidate)

---

*Alta Investments · Dispatch Work Order · 2026-07-22*
*"The level is the signal. Direction is a weight. Patience is a component."*
