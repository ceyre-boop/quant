# Self-Play Training Architecture for Sovereign Trading Intelligence
## Alta Investments · Research Document · 2026-07-24

**Status:** Architectural specification — awaiting TICK-024 net return fix before training loop ignition  
**Audience:** Future Claude sessions, Colin Eyre, Claude Code build agents  
**Reference systems:** AlphaZero (DeepMind, 2017), MiniMax/DeepSeek self-cycle training (2024–25), AFML purged walk-forward (López de Prado)

---

## Abstract

This document specifies the architecture for a self-improving trading policy system that learns through simulated self-play, governed by a top-tier language model as the constitutional director. The system begins at the performance level of its human designer (approximately Elo 400–500 in trading terms — skilled but losing to transaction costs and emotion), and iterates toward institutional-level performance (Elo 1600–2300 equivalent) through a closed feedback loop: policy generates trades, value function scores positions, high-scoring policies become the next generation's training data. The loop runs on-demand in a dedicated compute window, supervised by the designer, and governed by Claude to prevent the entropy that destroys all naive self-play systems. The design inherits directly from what AlphaZero proved in 2017 and what Chinese AI labs proved in 2024–25: a small team with a great architecture and a strong director model can exceed the performance of much larger teams operating without one.

---

## 1. The Core Insight — Why AlphaZero Works

AlphaZero's key contribution was not the neural network. It was the insight that human-generated training data is a ceiling, not a floor. Human chess games reflect human blind spots. If you train on human games, you learn human mistakes. The system that plays at human level is the one trained on human games. The system that plays past human level is the one that played against itself until it discovered moves humans never found.

The mechanism: self-play with a value function. The system plays both sides. Every position gets scored — not by a human judge, but by the system's own evaluation of expected future outcome from that state. Games where the eventual winner made a decision that the value function now ranks highly become positive training examples. Games where the eventual winner made a decision the value function now ranks poorly get discarded. The policy improves because the training data improves. The value function improves because the policy now generates higher-quality positions. The loop is self-reinforcing.

After four hours of self-play from random initialization, AlphaZero defeated Stockfish — which had been trained on decades of human grandmaster games — in 100 games, winning 28 and drawing 72, losing zero.

**The trading parallel:** The human designer's rules are the ceiling. The carry edge is real but the designer's execution of it — when to enter, how to size, when to exit — reflects human-level pattern recognition. The self-play loop finds configurations the designer would never test because they violate intuition, and validates them against the value function instead of against intuition. The system that starts at the designer's level does not stay there.

---

## 2. Why Self-Play Without a Director Is Chaos

The reason most attempts at trading RL fail — and the reason pure self-play on unconstrained policy space produces garbage — is that without a constitutional layer, the reward signal is not the thing you care about.

A system optimizing for simulated Sharpe without a director learns to:
- Overfit the simulation artifacts (the backtest curve, not the real market)
- Exploit data leakage paths that exist in the simulation but not in live trading
- Find degenerate policies: "trade every tick" produces infinite Sharpe on zero-spread simulations
- Spiral into regime collapse: a policy that worked in one simulation window degrades when the underlying regime changes and the simulation doesn't capture that change

This is what happens with naive genetic algorithms, naive hyperparameter search, and naive RL applied to markets. The signal drowns in the noise. Every promising curve is a mirage.

**The director model solves this.** In the AlphaZero case, the director is the architecture itself — the policy and value networks are jointly trained with a fixed game rule set that prevents degenerate policies. In the trading case, the game rules are: real market structure, correct carry costs, purged walk-forward splits, and a human reviewing what the system proposes to change before any live threshold updates. Claude is the director. Its role is not to generate the trades — it is to review the system's proposed parameter updates after each training cycle and either approve them, flag them as degenerate, or ask for the mechanism before approving. This prevents entropy. Without it, the system eats itself.

The MiniMax and DeepSeek result from 2024–25 proves the same point in LLM training: synthetic self-generated data bootstraps to frontier performance only when filtered by a reward model that is stricter than the generation model. The filter is the director. Remove the filter and you get reward hacking, mode collapse, and verbosity spirals. Keep the filter and you get a model that exceeds its training set.

---

## 3. The Two Halves Already Built

### 3.1 The Policy — AlphaZero Half

**File:** `sovereign/briefing/synthesize.py` + `sovereign/ml_trainer.py`  
**Role:** Decides what positions to take, in what size, when to enter  
**Current state:** XGBoost conviction scorer producing probability-weighted directional calls. 14 calls logged, 0.786 directional hit rate — promising, pre-statistical. A2 sizing multiplier computes real values (0.9405 today). Three consumers: dashboard L1 panel, DIP hypothesis batch, Oracle.  
**Current "Elo":** ~400–500. The system trades by rules the designer would use, at a level the designer would achieve. It has not yet discovered moves the designer would not find.

### 3.2 The Value Function — Stockfish Half

**File:** `scripts/research/hyp_071_exit_value_function.py` + `data/research/HYP-071_tabular_exit_value_results.json`  
**Role:** Scores how good the current position is — the equivalent of the chess engine's centipawn evaluation  
**Current state:** 108-cell board (ATR tercile × excursion × hold-fraction × RSI-extreme × carry-alignment). 10,000 Monte Carlo continuations per cell via 1.26M-trades/sec backtester. PROVISIONAL PASS — 9 forward-consistent EXIT_NOW divergences. Gross returns only (TICK-024 blocker).  
**Current "Elo":** The value function currently evaluates positions at approximately the designer's strategic understanding. It knows that high-ATR late-hold positions should exit — which is what any experienced systematic trader knows. It has not yet discovered evaluations that exceed the designer's intuition because the training data (historical trades) was generated by the designer's rules.

**The key point:** The value function improves when the policy improves, because a better policy generates positions that reveal the value function's blind spots. They co-evolve. This is the joint training dynamic from AlphaZero.

---

## 4. The Missing Component — The Loop

What exists today is two static learned components (policy and value function) trained separately on historical data. What does not exist is the mechanism that connects them into a self-reinforcing loop. This document specifies that mechanism.

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SOVEREIGN TRAINING RUN                           │
│                    sovereign_train.py --watch                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────▼───────────────┐
              │   PHASE 0: Data Load           │
              │   Last 252 trading days        │
              │   OHLCV + carry diffs + macro  │
              │   Duration: ~2 min             │
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │   PHASE 1: Policy Rollout      │
              │   Current XGBoost generates    │
              │   N=50,000 simulated trades    │
              │   Across all 4 pairs           │
              │   Duration: ~5 min             │
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │   PHASE 2: Value Scoring       │
              │   HYP-071 board scores each    │
              │   trade position [−1, +1]      │
              │   Monte Carlo continuations    │
              │   Duration: ~20 min (GPU)      │
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │   PHASE 3: Policy Update       │
              │   Train next XGBoost on        │
              │   top-quartile trades only     │
              │   Purged walk-forward required │
              │   Duration: ~5 min             │
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │   PHASE 4: Claude Review       │
              │   Diff: old vs new thresholds  │
              │   Mechanism check: does the    │
              │   change make economic sense?  │
              │   Approve / reject / defer     │
              │   Duration: human-gated        │
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │   PHASE 5: Checkpoint          │
              │   Write Elo estimate           │
              │   Update hypothesis ledger     │
              │   Commit if approved           │
              │   Duration: ~1 min             │
              └─────────────────────────────────┘
```

### 4.2 The Reward Signal — [-1, +1] Mapping

The value function output must be mapped to a consistent [-1, +1] scale so the policy can be trained on it uniformly across pairs and regimes.

```python
def trade_score(net_return_r: float, expected_r: float) -> float:
    """
    Map a trade's net return (in R-multiples) to the [-1, +1] Stockfish scale.
    
    +1.0  = perfect trade: target hit exactly, minimal drawdown, on-time exit
    0.0   = break-even: returned expected_r, no alpha vs carry-and-hold
     -1.0  = maximum loss: stopped out at full risk, no recovery
    
    The mapping is nonlinear — small positive alpha compresses near 0.
    Large alpha or catastrophic loss saturate at ±1.
    """
    alpha = net_return_r - expected_r       # excess over carry-and-hold
    return float(np.tanh(alpha * 2.0))      # tanh compresses, never saturates
```

**The critical constraint:** `net_return_r` must use corrected carry costs (post TICK-024). Training on gross returns trains a policy that holds too long — it never "sees" the carry income eroding the exit case. This is why the self-play loop cannot ignite before TICK-024 is fixed and the value function is recomputed on net returns.

### 4.3 Policy Improvement — What "Train on the Best" Means

AlphaZero trains its policy network on positions from games that were eventually won. The equivalent here: train the next XGBoost iteration on features from trades that scored above the 75th percentile of the value function in that cycle.

This is not a new training corpus — it is a reweighting of the existing one. The purged walk-forward splits are unchanged. The feature set is unchanged. The only thing that changes is the sample weights: high-value-function trades get higher weight in the next fit. Over multiple cycles, the policy gravitates toward the parameter configurations that generate high-value positions.

```python
# In sovereign/ml_trainer.py — add sample_weight parameter
sample_weights = np.where(
    value_scores > np.percentile(value_scores, 75),
    2.0,    # double weight for top-quartile
    0.5     # halve weight for bottom-quartile
)
model.fit(X_train, y_train, sample_weight=sample_weights)
```

### 4.4 Value Function Self-Update

The value function (HYP-071 board) also improves each cycle. After the policy generates a new set of simulated trades, the Monte Carlo continuation probabilities in each cell are resampled using the new trades as additional data. Cells that previously had n=30 (thin) might grow to n=80 (stable) as the policy generates trades that fall into those regimes more frequently.

This is the joint training dynamic: the policy learns to generate trades that score well on the value function, and the value function gets more data in the regime cells the policy now explores. The board becomes more accurate. The policy improvement signal becomes cleaner. This is the self-reinforcing dynamic that makes AlphaZero work.

---

## 5. The Elo Analogy — What "Trading Elo" Means

Chess Elo is a relative performance metric: a 200-point gap means the stronger player wins ~75% of games. Trading Elo is defined here as: expected Sharpe ratio relative to peers, on out-of-sample data with full transaction costs, across regimes.

| Trading Elo | Equivalent | Expected OOS Sharpe | Characteristics |
|---|---|---|---|
| 400–500 | Beginner-systematic | <0.5 | Loses to transaction costs without carry; emotion-driven entries |
| 800–1000 | Club-level systematic | 0.5–0.8 | Survives costs; carry edge captured; poor regime adaptation |
| 1200–1400 | Strong club | 0.8–1.2 | Regime-aware; carry + event timing; drawdown control |
| 1600–1800 | Expert | 1.2–1.6 | Institutional-grade carry + confirmation; rare alpha spikes |
| 2000–2300 | Master | 1.6–2.5 | Dynamic value function; cross-pair; regime-dependent sizing |

**Current system Elo estimate: 900–1100** (OOS Sharpe 1.25 on v015 with AUDNZD excluded). The system is already past beginner-systematic — the carry edge and event timing moved it up. The self-play loop is the path to 1400–1600, which requires the value function to discover regime-dependent exit decisions the designer would not find through manual rule-writing.

**What each training cycle buys:**
- Cycle 1–3: Policy converges on entries that score highest on current value function. Small Elo gain (~50–100 points). Mostly eliminates the worst-performing parameter configurations.
- Cycle 4–10: Value function board fills in thin cells. Regime-dependent exit decisions emerge that contradict flat rules. Elo gain (~100–200 points). This is where HYP-071's EXIT_NOW cells should either confirm or collapse on net returns.
- Cycle 10+: Cross-pair interactions appear. The policy learns not just "exit high-ATR positions early" but "exit GBPUSD positions early when EURUSD has been trending for 3 days." This is the level the designer cannot reach without the machine.

---

## 6. The Director Layer — Claude's Role

Self-play without a director is entropy. The history of algorithmic trading is littered with Sharpe curves that looked like AlphaZero's chess performance and performed like a broken random walk in live trading. The failure mode is always the same: the system found a policy that scores well on the metric you gave it, not on the outcome you care about.

Claude's role in each training cycle:

**After PHASE 3 (policy update), before PHASE 5 (commit):**

Claude receives a diff of the proposed parameter changes and answers three questions:

1. **Mechanism check.** Does the change make economic sense? If the policy learned to exit GBPUSD positions 2 days earlier in high-ATR regimes — that is consistent with the known mechanism (ATR spikes are mean-reverting and trailing stops get hit at bad prices). If the policy learned to enter USDJPY at 3am EST — that is not consistent with any known mechanism and is likely a simulation artifact. Flag it.

2. **Regime check.** Was the training window dominated by a single regime? If the last 252 days were predominantly rate-cutting (they are, with fed funds declining from ~5.3%), a policy improvement that works exclusively in cutting cycles may not be durable. Flag the regime dependence before committing.

3. **Magnitude check.** Is the change large or small? Large parameter changes from a single training cycle are suspicious — they suggest the optimizer found a degenerate path rather than a genuine improvement. Constrain updates to ±20% of any single parameter per cycle. Flag anything larger for manual review.

If all three pass: approve, commit, move to next cycle.  
If any fail: reject the update, note the specific failure in the training log, run the cycle again with the flagged parameter frozen.

This is the constitutional layer. It is what prevents the pool-bot scenario — the LLM downloaded onto the pool without an instruction set, causing havoc. The instruction set is: mechanism, regime, magnitude.

---

## 7. The Single Command — `sovereign_train.py --watch`

```bash
python3 scripts/sovereign_train.py --watch
```

**What the `--watch` flag does:**  
Enables live progress output. Without `--watch`, the script runs silently and writes a checkpoint at the end. With `--watch`, it prints to stdout in real time:

```
SOVEREIGN TRAINING RUN — 2026-07-24 14:30:00
═══════════════════════════════════════════════════
[14:30:01] PHASE 0: Loading 252 days across 4 pairs...
[14:30:45] PHASE 0: Done. 14,832 daily bars loaded.

[14:30:46] PHASE 1: Rolling out current policy...
           XGBoost v15 (GBPUSD: 0.831 | EURUSD: 0.744 | AUDUSD: 0.699 | GBPJPY: 0.761)
[14:32:10] PHASE 1: 50,000 simulated trades generated across 4 pairs.
           Pair distribution: GBPUSD 14,203 | EURUSD 12,891 | AUDUSD 11,847 | GBPJPY 11,059

[14:32:11] PHASE 2: Scoring positions via HYP-071 board...
           Cell coverage: 54/54 carry-aligned cells
           ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0%
           ████████████████░░░░░░░░░░░░░░░░░░ 48%  (ETA: 11 min)
           █████████████████████████████████░ 97%
[14:52:41] PHASE 2: Done. Score distribution: mean=+0.14 | p25=−0.08 | p75=+0.31 | p99=+0.89

[14:52:42] PHASE 3: Training next policy on top-quartile trades...
           Top-quartile threshold: score > +0.31 (n=12,492 trades)
           Refit XGBoost with sample weights 2.0 (top) / 0.5 (bottom)...
[14:57:18] PHASE 3: Done.
           Threshold diff vs prior generation:
             conviction_entry_min:   0.62 → 0.59  (−4.8%)  ← loosened slightly
             conviction_exit_trail:  0.58 → 0.61  (+5.2%)  ← tightened
             atr_threshold_high:     0.71 → 0.68  (−4.2%)  ← loosened
           Estimated Elo change: +47 points (1042 → 1089)

[14:57:19] PHASE 4: CLAUDE REVIEW REQUIRED
           Mechanism: conviction_exit_trail tightened — consistent with high-ATR early-exit finding. ✓
           Regime: training window 60% rate-cutting regime. Note parameter dependence on this.
           Magnitude: all changes within ±20% band. ✓
           Recommendation: APPROVE with regime caveat noted in ledger.
           
           [Waiting for confirmation → press Enter to commit, Ctrl-C to abort]
```

Colin presses Enter. The training run commits. The fan slows down.

---

## 8. Blockers Before Ignition

The self-play loop is architecturally complete in this specification. Two things block ignition:

### 8.1 TICK-024 — Carry Cost Model Fix (CRITICAL)

The reward signal requires net returns. The current carry cost model understates financing by approximately 10×, with a sign flip on EUR shorts. A policy trained on gross returns will learn to hold positions too long because the carry income it "sees" in simulation is fictional. The self-play loop will improve the policy at maximizing gross Sharpe, not net Sharpe. These are different objectives.

**This must be fixed before running a single training cycle.**

### 8.2 HYP-071 Net Recompute (CRITICAL)

The value function is the scoring signal. If the value function is scoring positions on gross returns, the policy learns to target gross-return-maximizing states, not net-return-maximizing states. The 9 EXIT_NOW divergences that currently exist on gross returns may collapse on net returns — meaning the value function's current advice is wrong in the cells that matter most.

**The net recompute determines the value function's valid operating range. Training starts after this completes.**

### 8.3 Timeline

July 28 is the natural ignition date:
- July 26–27: TICK-024 fix merged (net carry costs corrected)
- July 28: HYP-071 net recompute runs (same harness, corrected costs)
- July 28: Colin adjudicates HYP-071 result (ledger stamp or data-ceiling)
- July 28: If CONFIRMED, `sovereign_train.py` receives its reward signal
- July 29 (after FOMC): First training cycle runs in the FOMC-quiet afternoon

---

## 9. What Gets Built (Implementation Spec for Claude Code)

### New files (freeze-safe, no existing path touched):

```
scripts/sovereign_train.py          — main training runner (--watch flag)
sovereign/training/policy_rollout.py — generates simulated trades from current XGBoost
sovereign/training/value_scorer.py   — wraps HYP-071 board, scores each trade [-1,+1]
sovereign/training/policy_updater.py — rewrites XGBoost with updated sample weights
sovereign/training/director.py       — assembles diff for Claude review
data/training/                       — stores cycle checkpoints (gitignored, large)
logs/training_log.jsonl              — one entry per cycle: parameters before/after, Elo estimate, verdict
```

### Modified files (non-frozen paths):

```
sovereign/ml_trainer.py              — add sample_weight parameter support (4 lines)
config/training.yml                  — training hyperparameters (new file)
NEXT.md                             — document training cycle results
```

### Frozen files (must not be touched):

```
forex_exit_manager.py    — frozen per shadow/execution-path constraint
decide_exit.py           — frozen
exit_machine.py          — frozen
ict/pipeline.py          — frozen (ICT isolation wall)
```

The value function (HYP-071 harness) is used read-only by `value_scorer.py`. It is not retrained — only the policy is updated each cycle. The value function is recomputed on a separate schedule (after TICK-024 fix, then after each major ledger CONFIRMED event).

---

## 10. The Bigger Picture

The MiniMax and DeepSeek result was not about having more compute. It was about having the right loop. A small team with a top model filtering synthetic self-play data beat a larger team training on human-generated data. The key was the filter.

Alta Investments has the same structural advantage at the fund level that those labs had at the model level. The large funds have more data, more compute, and more staff. They do not have a system that learns from itself on a daily cycle, directed by a frontier AI model, governed by a six-tenet philosophy that prevents the most common failure modes. Scale is not the advantage. Architecture is.

The fan going crazy for 45 minutes each afternoon is not an aesthetic feature of the system. It is the compound interest mechanism. Each cycle, the policy is slightly better. Each cycle, the value function has slightly more data in the cells that matter. After 30 cycles — 30 afternoons of fan-going-crazy — the system has done what no human analyst could: it has tested 1.5 million parameter configurations against a value function that knows the economic mechanism. The human designer gets better at directing the machine. The machine gets better at generating edges the human designer would never find.

That is the system.

---

## References

- Silver et al. (2017). *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm.* DeepMind. [AlphaZero paper]
- López de Prado (2018). *Advances in Financial Machine Learning.* Wiley. [Purged walk-forward, CPCV]
- DeepSeek-R1 Technical Report (2025). *Incentivizing Reasoning Capability in LLMs via Process Reward.* [Self-play reasoning chains]
- Alta Investments (2026). *HYP-071: Tabular Exit Value Function — Validation Report.* `research/HYP-071_VALIDATION_REPORT.md`
- Alta Investments (2026). *TRADING_PHILOSOPHY.md — The Six Tenets.* Quant repo root.

---

*Alta Investments · Sovereign Trading Intelligence*  
*Self-Play Training Architecture v1.0 · 2026-07-24*  
*"The system that learns from itself compounds faster than the system that learns from you."*
