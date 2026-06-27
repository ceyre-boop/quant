#!/usr/bin/env python3
"""Stage the hand-authored concept/architecture hubs for the Alta-System knowledge graph.

Writes ~20 interpretive notes into <vault>/Trading/System/Concepts/. These are the synthesized
'system thesis' layer the deterministic generator (build_obsidian_graph.py) cannot extract — they
reference, never duplicate, docs/ARCHITECTURE.md, TRADING_PHILOSOPHY.md, and the generated notes.
Filenames match the [[Concept]] wikilinks emitted by the generator's 00-System-Index.

Run AFTER the generator's full run (so manifest-pruning never touches these). Idempotent overwrite.
"""
from pathlib import Path

VAULT = Path("/Users/taboost/Obsidian/Obsidian")
OUT = VAULT / "Trading" / "System" / "Concepts"
DATE = "2026-06-26"

def fm(title: str, subsystem: str = "concept") -> str:
    return (f"---\ndate: {DATE}\ntitle: \"{title}\"\ntype: system-doc\nproject: Alta\n"
            f"subsystem: {subsystem}\n---\n")

LINKS = "> Links: [[00-System-Index]] · [[Alta-MOC]] · [[Discovery-Ledger]] · [[Oracle-Context]]\n"

NOTES: dict[str, str] = {}

NOTES["Two-Layer-Wall"] = f"""{fm("Two-Layer Wall — predict vs evaluate")}{LINKS}
# The Two-Layer Wall

A **vertical** wall inside each trading system separating **Layer 1 — PREDICTIVE** (decide *what to
bet and which way*) from **Layer 2 — EVALUATIVE** (decide *what to do with a position we hold*,
deterministically, no AI). It is orthogonal to the existing horizontal [[ICT-Sovereign-Isolation]]
wall (which separates *systems* by time-horizon). Both hold at once.

**Why:** profit ≈ f(prediction) × g(evaluation) — multiplicative. Prediction is bounded near a coin
flip ([[Carry-Edge]], [[Regime-Fragility]]); most realizable Sharpe lives in L2, where the problem is
chess-shaped and solvable. See [[AlphaZero-Stockfish]].

**What crosses (L1 → L2):** only the *candidate* — direction, entry trigger, risk geometry (stop
distance). **No reasoning, probability, or narrative crosses.** Nothing crosses L2 → L1.

**The ONE sanctioned crossing:** the forex exit machine reads `signal_today` to trigger
reversal/cb_refresh exits ([[Six-Exit-States]]) — a prediction input timing an exit, *permitted*
because it is deterministic + logged.

**Status (firsthand audit):** wall **intact in forex**; one bleed (anti-pattern) in
`futures/decision_engine.evaluate_entry` ([[sovereign.futures.decision_engine]]) which fuses L1+L2 in
one object.

Full doctrine: `docs/ARCHITECTURE.md`. Component-by-component: `docs/COMPONENT_CLASSIFICATION.md`.
"""

NOTES["AlphaZero-Stockfish"] = f"""{fm("AlphaZero proposes, Stockfish disposes")}{LINKS}
# AlphaZero / Stockfish — the reference architecture

The [[Two-Layer-Wall]] is a translation of a documented, tested pattern from chess engines.

- **Layer 1 = AlphaZero** — probabilistic, learned pattern recognition, confidence-weighted ("favors
  LONG ~0.62"), ~55% ceiling. Native to the *open* problem.
- **Layer 2 = Stockfish** — deterministic, rule-based calculation, one correct action, refuses
  ambiguity. Native to the *bounded* problem.

Precedent (stated accurately): modern engines **hybridize** — Stockfish adopted a learned eval (NNUE)
on top of deterministic search in 2020; Leela sits at the learned pole. Learned proposal +
deterministic calculation, fused, beats either pole alone.

**Handoff:** *AlphaZero proposes; Stockfish disposes.* Candidate setup w/ conviction → evaluate
against account/risk/regime/health state → **veto or accept** → run to a deterministic exit. Stockfish
vetoes **entries**, not only exits (Citadel pod structure: analysts propose, risk has veto).

**L2 has two jobs, opposite completeness:**
- **Dispose (entry-veto + sizing) — BUILT, LIVE, apex** → [[Eight-Risk-Layers]],
  [[sovereign.risk.risk_engine]], [[Conviction-Sizing]].
- **Run-to-exit — SHELVED in the backtester** → [[Six-Exit-States]] (the #1 gap).

Apex desks run both; retail runs only the AlphaZero. The fusion is the alpha — an *architectural*
property, not an algorithmic one.
"""

NOTES["Conviction-Sizing"] = f"""{fm("Conviction sizing — no flat positions")}{LINKS}
# Conviction Sizing

**NON-NEGOTIABLE:** no flat position sizes. All sizing flows through the conviction-based pipeline.

- L1 emits a **conviction** ∈ [0,1] ([[sovereign.forex.macro_engine]]) and a bounded **size_mult**
  ([[sovereign.forex.signal_engine]]). Below the floor (`CONVICTION_NEUTRAL_THRESHOLD = 0.10`,
  authorized 2026-06-05) → NEUTRAL / no trade (the Buffett filter).
- L2 turns that into units: `units = risk_usd / stop_distance` ([[sovereign.forex.position_sizer]]),
  then the [[Eight-Risk-Layers]] engine ([[sovereign.risk.risk_engine]]) applies the final verdict.

**Wall rule:** conviction may *scale* a size within pre-registered bounds; it may **never** originate
an exit (that's [[Six-Exit-States]]). Sizing on a free-form forecast is a named anti-pattern.

**Open lever:** the `risk_adjusted_pnl_pct = pnl_pct × risk_pct` question is a *leverage* decision
(leverage-invariant on Sharpe), logged via param_change_log, deferred until the live exit engine
lands. Thresholds: [[parameters.risk]], [[risk_config.base]], [[risk_config.kelly]].
"""

NOTES["The-Gauntlet"] = f"""{fm("The Gauntlet — promotion standard")}{LINKS}
# The Gauntlet

Nothing reaches LIVE without clearing the full statistical gauntlet — the antidote to the factor zoo
and to [[Tenet-1-Statistical-Utility]] violations.

- **Permutation test** (≥1000, research factory ≥10000) — beat the shuffled null.
- **Benjamini-Hochberg** — survive multiple-comparison FDR correction.
- **Walk-forward** — positive marginal contribution per fold, no holdout degradation.
- **Decay ratio ≥ 0.50** — OOS keeps ≥half the IS edge.
- **Both-sides** — report performance in BOTH regime states; never condition on outcome.

Promotion bars (`TRADING_PHILOSOPHY.md`): IC > 0.15 OOS · positive walk-forward marginal · no 6-month
holdout degradation. Enforced mechanically: `_methodology_ok` in
[[sovereign.autonomous.research_factory]]; canonical runner `scripts/run_hypothesis.py`. See
[[Pre-Registration]]. Outcomes live in the [[Hypothesis-Ledger]].
"""

NOTES["Pre-Registration"] = f"""{fm("Pre-Registration — freeze before looking")}{LINKS}
# Pre-Registration

Freeze the spec — regime definition, threshold grid, pairs, IS/OOS splits, the falsification test,
the *expected* verdict — to disk **before** running anything. A hash-locked JSON under
`data/research/preregister/`. The generator/runner asserts the live spec matches (tripwire) and
refuses a tuned spec.

**The cautionary tale — HYP-044** ([[HYP-044]]): an in-sample VIX-13 tightening (+0.242) that
collapsed OOS (0.000). Pre-registration + [[The-Gauntlet]] exist so the next HYP-044 is caught before
it goes live. See also [[Discovery-Meta-Finding]] and the methodology config [[autonomous.methodology]].
"""

NOTES["Regime-Fragility"] = f"""{fm("Regime fragility — the honest number")}{LINKS}
# Regime Fragility

The carry edge is **real but regime-fragile**. The proof engine (`scripts/prove.py`,
[[Reporting-MOC]] `equity_curve.v1`) draws the honest curve:

- Full-decade (2015–2024) √n Sharpe **0.69** (WEAK — below the 0.8 viable bar).
- OOS (2023–2024) Sharpe **1.25** — a favorable window, not the steady state.
- Rolling walk-forward: 2021 −0.13 / 2022 +0.51 / 2023 +1.26 / 2024 −0.09 — **pays only in
  rate-trending regimes.**

Two truths the curve exposed: (1) the headline "1.25" was a window; (2) Sharpe 1.25 = only +0.4%/2yr
in dollars → **sizing is the lever, not a new edge** ([[Conviction-Sizing]]). The lift path is L2
capture/protection ([[AlphaZero-Stockfish]]), not a 29th signal ([[Discovery-Meta-Finding]]). Source
edge: [[Carry-Edge]].
"""

NOTES["Carry-Edge"] = f"""{fm("Carry edge — the one real edge")}{LINKS}
# The Carry Edge

The single validated edge in the system: the macro **rate-differential / carry risk premium** on
daily FX. Permutation p<0.001 (real, not luck). Lives in [[sovereign.forex.macro_engine]] (real-rate
differential momentum, IRP z, cycle divergence, PPP z, Hurst) → [[sovereign.forex.signal_engine]].

**It is irreducible:** [[Discovery-Meta-Finding]] tested 28 candidate edges off daily OHLC → 0
survived; nothing improves it, gates only subtract. It is also **[[Regime-Fragility|regime-fragile]]**
— full-decade Sharpe 0.69, pays only when rates trend.

v015 lives as a 4-pair portfolio (AUDNZD excluded — both legs RBA-driven, no independent differential).
Related hypotheses: [[HYP-027]] (USDJPY regime gate, now inert), [[HYP-028]], [[HYP-034]].
A second, uncorrelated edge requires **new data**, not cleverer mining.
"""

NOTES["Oracle-Closed-Loop"] = f"""{fm("Oracle closed loop — how the system learns")}{LINKS}
# The Oracle Closed Loop

The cognition layer: 1 Opus call/day → HARVEST → REFLECT → TEST → CODIFY → 1 lesson.
[[sovereign.oracle.oracle_cycle]], [[sovereign.oracle.reflect_cycle]].

**NON-NEGOTIABLE #2:** every decision wired to the decision logger MUST receive an `update_outcome()`
when the trade closes — Oracle cannot learn without closed-loop outcomes.
[[sovereign.intelligence.decision_logger]].

**It was silently broken — twice.** (1) 2026-06-23: `system="ICT"` for FOREX fills + `GBPUSD=X` pair
mismatch dropped all outcomes. (2) 2026-06-26 (commit e68380a): decisions stamped at *signal* time but
backfill keyed on *fill* time, matched at hour-precision — any signal→fill crossing a clock hour
silently dropped. Both fixed; "Oracle learns live" was false until then. The loop feeds context from
the daily FRED macro pull / the research panel and writes lessons tracked by the lesson-velocity tracker.
"""

NOTES["Kill-Switch"] = f"""{fm("Kill switch — freeze the trading path")}{LINKS}
# Kill Switch

`alta freeze` / `thaw` / `status` → `data/system/KILL_SWITCH`. A **soft** freeze of the trading path
plus a **hard** block on `approve_edge`. Monitoring and cognition ([[Oracle-Closed-Loop]]) stay alive
so the system keeps watching and learning while flat. Committed e689bba / fad525f.

Distinct from the [[Eight-Risk-Layers]] in-flight halt (`run_gates` → size 0): the kill switch is the
operator-level master stop; the risk gates are the per-decision automatic veto
([[Tenet-3-Know-When-Unreliable]]).
"""

NOTES["ICT-Sovereign-Isolation"] = f"""{fm("ICT ↔ Sovereign isolation invariant")}{LINKS}
# ICT ↔ Sovereign Isolation

**NON-NEGOTIABLE:** `ict/` and `ict-engine/` must **never** import from `sovereign/`. Enforced by
`tests/ -k test_pipeline_does_not_import_sovereign`. This is the **horizontal** time-horizon wall
(ICT intraday / Forex multi-day / Equity) from `TRADING_PHILOSOPHY.md` — no feature sharing across
horizons (contamination produces correlation artifacts that look like edge).

Cross-layer logic routes through **`ict-engine/orchestrator.py`** ([[ict-engine.orchestrator]]) — the
only safe ICT→sovereign entry point — never `ict/pipeline.py`. Orthogonal to the vertical
[[Two-Layer-Wall]]; both hold at once. (Note: the live scanner daemon loads the `ict/` package;
`ict-engine/` is the bridge, not orphaned.)
"""

NOTES["Six-Exit-States"] = f"""{fm("Six exit states — the run-to-exit machine")}{LINKS}
# The Six Exit States

The deterministic run-to-exit machine `_simulate_forex_core` ([[sovereign.forex.fast_backtester]]) —
the apex of [[AlphaZero-Stockfish|Stockfish's]] management half. Six exits + pyramiding (scale-up):

1. **stop** — fixed ATR stop hit
2. **trailing-ATR** — `trailing_atr_mult` ratchet
3. **donchian** — break of the rolling low (strict_mode only — OFF in the canonical proven backtest)
4. **reversal** — `signal_today` flips against the position (a sanctioned [[Two-Layer-Wall]] crossing)
5. **time** — max hold reached
6. **cb_refresh** — signal re-confirms; continuation

**THE #1 GAP:** this runs **only in the backtester**. Live = `oanda_bridge.place_trade`
([[sovereign.execution.oanda_bridge]]) sets a static stop/TP + poll; no trade-modify endpoint, no
re-eval loop. No PARTIAL_EXIT/SCALE_DOWN anywhere. Build 1 = port this to a live position-manager.
"""

NOTES["Eight-Risk-Layers"] = f"""{fm("Eight risk layers — the dispose engine")}{LINKS}
# The Eight Risk Layers (the live entry-veto)

[[sovereign.risk.risk_engine]] `decide()` — "the SOLE sizing authority; every op can only REDUCE
risk; never executes." This is [[AlphaZero-Stockfish|Stockfish's]] dispose half — **live and apex.**

Cascade: `desired = base × vol × dd × regime; capped = min(desired, kelly, portfolio, prop);
final = 0 if any hard gate fires`. Layers (`sovereign/risk/layers/`):

- **base_size** ([[sovereign.risk.layers.base_size]]) — grade-scaled base risk → [[risk_config.base]]
- **volatility** · **drawdown** · **regime** — compounding modulators (≤1.0) → [[risk_config.volatility]], [[risk_config.drawdown]], [[risk_config.regime]]
- **kelly** · **portfolio** · **prop** — binding ceilings via `min()` → [[risk_config.kelly]], [[risk_config.portfolio]], [[risk_config.prop]]
- **gates** ([[sovereign.risk.layers.gates]]) — **6 hard vetoes** → size 0: daily-loss, max-dd buffer,
  internal prop guard, **health-not-ok** ([[Tenet-3-Know-When-Unreliable]]), macro threat-critical,
  MC-breach. → [[risk_config.gates]]

The deterministic [[Two-Layer-Wall|entry veto]]: a high-conviction L1 candidate is still rejected if a
gate fires. Operator master stop = [[Kill-Switch]].
"""

NOTES["Data-Flow-Pipeline"] = f"""{fm("Data-flow pipeline — provider to lesson")}{LINKS}
# Data-Flow Pipeline

End-to-end path (detail: `docs/DATA_FLOW.md`):

**Data providers** ([[Data-MOC]] feeds: yfinance / OANDA / Polygon / Databento) → **L1 signal**
([[sovereign.forex.macro_engine]] → [[sovereign.forex.signal_engine]]) → **decision logged**
([[sovereign.intelligence.decision_logger]]) → **L2 risk verdict** ([[Eight-Risk-Layers]]) →
**execution** ([[sovereign.execution.oanda_bridge]], static exits today — [[Six-Exit-States]]) →
**ledger** (`data/ledger/oanda_fills.jsonl`) → **outcome backfilled** (`update_outcome`) →
**[[Oracle-Closed-Loop]]** → 1 lesson/day.

Context-only side feeds (never trading inputs): the daily FRED macro pull, the daily research panel, the morning
+ EOD briefs ([[Briefing-MOC]], `provenance.verified=false`). The proof engine
([[Regime-Fragility]]) reads the ledger to draw the honest curve.
"""

NOTES["Discovery-Meta-Finding"] = f"""{fm("Discovery meta-finding — the well is dry")}{LINKS}
# Discovery Meta-Finding

The edge-discovery pipeline ([[Discovery-MOC]], `scripts/discover.py`) ran **28 candidates**
(18 price-pattern + 10 regime filters) through [[The-Gauntlet]]. **Result: 0 VALID_EDGE.**

What it TRULY means:
1. The **[[Carry-Edge]] is the only real edge in this data, and it is irreducible** — nothing improves
   it; adding gates HURTS (removes trades without lifting Sharpe). The machine, asked "best way to
   trade," answered: *the plain base signal you already deploy.* Empirical proof of
   [[Tenet-1-Statistical-Utility]].
2. **Feature importance is flat** → a macro-driven, non-pattern-mineable surface. Daily FX OHLC has no
   microstructure alpha.
3. **HYP-027 fails re-validation here too** ([[HYP-027]]) — corroborated by the regime screen.
4. **A second edge needs NEW DATA, not cleverer mining** — intraday microstructure, options (VRP),
   order-flow. The discovery test empirically justifies the data-acquisition strategy.

Full synthesis: [[Discovery-Ledger]].
"""

# ── The six tenets (TRADING_PHILOSOPHY.md) ──────────────────────────────────────────────────────
NOTES["Tenet-1-Statistical-Utility"] = f"""{fm("Tenet 1 — Statistical utility beats narrative")}{LINKS}
# Tenet 1 — Statistical Utility Beats Narrative Coherence

A feature earns its place by improving measurable out-of-sample expectancy, not by making intuitive
sense. **The pd_alignment lesson (HYP-024):** removing a logically coherent feature lifted win rate
20%→35% — the system improved by *removing* intelligence. Enforced by the promotion gates
([[The-Gauntlet]]) and proven again by the [[Discovery-Meta-Finding]]. Constitution:
`TRADING_PHILOSOPHY.md`.
"""

NOTES["Tenet-2-Regime-Appropriateness"] = f"""{fm("Tenet 2 — Regime appropriateness")}{LINKS}
# Tenet 2 — Regime Appropriateness Beats Strategy Quality

The question is not "is this strategy good?" but "is it appropriate *right now*?" Systems thrive and
decay in regime-specific ways — see [[Regime-Fragility]] (carry pays only in rate-trending regimes).
The capital allocator + regime performance tracker ([[Intelligence-MOC]]) route sizing by
regime mechanically. Risk modulation: [[risk_config.regime]]. Constitution: `TRADING_PHILOSOPHY.md`.
"""

NOTES["Tenet-3-Know-When-Unreliable"] = f"""{fm("Tenet 3 — Know when unreliable")}{LINKS}
# Tenet 3 — Systems Must Know When They Are Unreliable

A mature system can say "I am currently unreliable" — and freeze itself. This is the most important
safety feature in the stack. Made concrete: the **health-not-ok** hard gate in the
[[Eight-Risk-Layers]] ([[sovereign.risk.layers.gates]]) vetoes to size 0 when the health heartbeat is
down; system health lives in [[Intelligence-MOC]]. Operator-level: [[Kill-Switch]].
Constitution: `TRADING_PHILOSOPHY.md`.
"""

NOTES["Tenet-4-Premature-Complexity"] = f"""{fm("Tenet 4 — Premature complexity kills")}{LINKS}
# Tenet 4 — Premature Complexity Kills More Systems Than Lack of Edge

Most systems die *after* edge is first confirmed — the creator stacks signals before understanding
where edge originates. **Guard:** every new component must answer a measurable question before it is
built; if it can't with existing data, **collect the data first** ([[Discovery-Meta-Finding]],
[[Carry-Edge]] → second edge needs new data). The research panel + VRP track embody "collect first."
Constitution: `TRADING_PHILOSOPHY.md`.
"""

NOTES["Tenet-5-Orchestration"] = f"""{fm("Tenet 5 — Orchestration is the durable edge")}{LINKS}
# Tenet 5 — Orchestration Is The Durable Edge

Individual strategy edges decay; the ability to route capital to the right system at the right time
does not. The cross-system bridge (macro threat gating), capital allocator (sizing throttle), and
regime performance tracker ([[Intelligence-MOC]]) form the orchestration layer — it should grow
faster than any single strategy. The [[Autonomous-MOC]] closure layer extends
this. Constitution: `TRADING_PHILOSOPHY.md`.
"""

NOTES["Tenet-6-Research-Debt"] = f"""{fm("Tenet 6 — Research debt is existential")}{LINKS}
# Tenet 6 — Research Debt Is Existential Risk

Failed experiments must be formally recorded — research that isn't logged does not exist, and insights
not logged are rediscovered at cost. The [[Hypothesis-Ledger]] (50 entries) + the permanent feature
graveyard (`lab/feature_registry.py`) enforce this; [[Pre-Registration]] makes every test honest
before the fact. The [[Discovery-Ledger]] is the living research synthesis. Constitution:
`TRADING_PHILOSOPHY.md`.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, body in NOTES.items():
        (OUT / f"{name}.md").write_text(body)
    print(f"Wrote {len(NOTES)} concept hubs to {OUT}")
    for n in sorted(NOTES):
        print(f"  {n}.md")


if __name__ == "__main__":
    main()
