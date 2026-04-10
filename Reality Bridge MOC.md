# The Reality Bridge — Backtest to Live Capital 🏛️

#validation #live-trading #hardening #reality-check #capital-deployment #institutional

> **The Moment:** The backtest shows 79.49% win rate, 65% XGBoost accuracy, 42% growth edge, 11,090 labeled signals. The question is no longer "does the edge exist in data?" The question is "does the edge survive contact with reality?" These are different questions. This document bridges them.

> **The Hard Truth:** Every great backtest has died in live trading. Not because the edge wasn't real — but because the bridge between simulation and reality was never properly built. You are not going to be one of those stories. You are going to build the bridge first.

---

## 🗺️ Map of Contents

- [[#The Six Ways Backtests Lie]]
- [[#Phase 0 — Pre-Deployment Audit]]
- [[#Phase 1 — Paper Trading (The Real Test)]]
- [[#Phase 2 — Micro Live Deployment]]
- [[#Phase 3 — Controlled Scaling]]
- [[#Phase 4 — Institutional Deployment]]
- [[#The Statistical Validation Framework]]
- [[#Live Performance Tracking]]
- [[#The Degradation Detection System]]
- [[#Execution Reality vs Backtest Assumptions]]
- [[#The Psychological Reality Layer]]
- [[#The Kill Switch Protocol]]
- [[#The Milestone Roadmap]]

---

## The Six Ways Backtests Lie

> Before a single dollar goes live, understand exactly how your backtest could be showing you something that is not real. This is not pessimism. This is the most important analysis you will ever do.

### Lie 1 — Lookahead Bias (The Most Common Killer)

```
WHAT IT IS:
  The model used information that would not have been available at decision time.
  This is often invisible and subtle.

HOW IT HAPPENS IN YOUR SYSTEM:
  - Feature calculated using data from T+1 but signal generated at T
  - Earnings data backfilled with restated numbers (companies restate earnings)
  - Index membership: S&P 500 stocks today were not all in S&P 500 in 2020
  - Adjusted close prices: splits/dividends adjusted backward change past prices
  - Implied volatility data aligned to wrong timestamp

HOW TO TEST FOR IT:
  Take your best 10 trades from the backtest.
  Open a chart of each one at the exact entry date/time.
  Ask: "At THIS exact moment, what data did the model actually have access to?"
  Verify every feature was computable from data available BEFORE that timestamp.
  If even ONE feature was "future" data → your win rate is inflated.

SPECIFIC CHECK FOR YOUR 43 FEATURES:
  [ ] Session timing features: computable in real-time ✓
  [ ] Liquidity sweep detection: requires price to have ALREADY swept → computable
  [ ] Displacement features: requires completed candle → check if using close vs open
  [ ] Volatility regime: ATR calculated from which candles exactly?
  [ ] Premium/discount alignment: FVG identified from which candle close?
  
  THE DANGER ZONE: Any feature that uses the CURRENT bar's close
  to generate a signal that executes on the CURRENT bar's close.
  In reality: you see the close AFTER it closes. You enter on the NEXT open.
  If backtest enters AT the close that generated the signal → lookahead bias.
```

### Lie 2 — Survivorship Bias

```
WHAT IT IS:
  Your 57-asset universe only contains assets that survived and are liquid today.
  Assets that went bankrupt, delisted, or became illiquid are missing.
  The missing assets would have hurt your win rate.

HOW BAD IS IT FOR YOUR SYSTEM:
  For ETFs (SPY, GLD, etc.): Low risk. ETFs rarely delist.
  For individual stocks (AAPL, TSLA, PFE): Medium risk.
  For leveraged ETFs (UVXY, VIXY): HIGH RISK.
    - Leveraged ETFs undergo reverse splits constantly
    - Their price history looks different pre/post split
    - Adjusted prices create fake "signals" that never existed

HOW TO TEST:
  [ ] Check each asset's price history for reverse splits in your test period
  [ ] Verify UVXY / VIXY adjusted data matches what was actually trading
  [ ] Check if any assets were added to your universe BECAUSE they performed well
      in the period you tested (selection bias)
```

### Lie 3 — Execution Assumption Errors (The Silent Killer)

```
WHAT IT IS:
  The backtest assumes fills that are impossible or improbable in reality.

THE MOST DANGEROUS ASSUMPTIONS IN YOUR BACKTEST:
  
  A) Fill at the exact signal price:
     In reality: You see the candle close → place order → fill on NEXT bar
     Your backtest might be filling at the close that generated the signal
     Fix: All entries should be at next bar OPEN minimum
     Impact on win rate: -5% to -15% depending on how often price gaps against you

  B) No slippage:
     In reality: Market orders move price against you
     UVXY/VIXY in particular: bid-ask spread can be $0.05-$0.15 wide
     Silver (SLV): spread is manageable but not zero
     Fix: Add 0.05-0.10% slippage per trade minimum; more for low-liquidity names

  C) No market impact:
     As you scale capital, YOUR orders move the market
     At $10K per trade: negligible
     At $100K per trade: meaningful for mid-cap and ETFs
     At $1M per trade: you ARE the market for some of your names

  D) Continuous liquidity:
     Backtest assumes you can always get filled
     In reality: At earnings releases, FOMC, news events → spreads widen 5-10x
     Gap risk: Price can move 3-5% before market open with NO way to enter/exit

REALISTIC EXECUTION ADJUSTMENT:
  Apply these to your backtest before trusting any metric:
  
  Entry: Assume fill at NEXT BAR OPEN (not signal bar close)
  Slippage: Add 0.1% round-trip for liquid names; 0.25% for UVXY/VIXY
  Commission: $0 for most retail brokers but model $0.005/share institutional
  Spread: Model half the average bid-ask spread as additional cost
  
  Re-run your backtest with these assumptions.
  Your 79.49% win rate may drop to 71-74%.
  That is STILL an exceptional edge. But know the real number.
```

### Lie 4 — Regime Overfitting

```
WHAT IT IS:
  Your 8-month training period may have been a single regime.
  A model trained on one regime fails in a different regime.

YOUR SPECIFIC RISK:
  8 months = August 2025 to April 2026
  What was the macro regime during this period?
  
  Ask: Was this period predominantly:
    - Bull market with low volatility?
    - Bear market with high volatility?
    - Sideways / range-bound?
    - Rate-rising environment?
    - Rate-falling environment?

  IF your 8 months were primarily one regime type:
    Your model learned that regime's patterns.
    It has NEVER seen the other regimes.
    When regime shifts: model accuracy drops sharply.

HOW TO TEST:
  [ ] Identify which macro regime dominated your training period
  [ ] Find 3-4 months in recent history with DIFFERENT regime
  [ ] Run your model on those months WITHOUT retraining
  [ ] If accuracy drops below 55% → regime-dependent edge
  [ ] If accuracy holds above 60% → regime-robust edge (much rarer; much better)

THE 4 MONTHS YOU'RE HARVESTING NOW (Dec 2025 → Apr 2026):
  This is EXACTLY why completing the harvest is critical.
  Q1 volatility, earnings cycles, Fed pivots = different sub-regimes.
  A model that trained on those months will be dramatically more robust.
```

### Lie 5 — Multiple Testing / Data Mining Bias

```
WHAT IT IS:
  If you tested hundreds of parameter combinations to find your 43 features
  and their thresholds, some of those "discoveries" are statistical accidents.
  With enough tests, random noise looks like signal.

YOUR SPECIFIC RISK:
  43 features × multiple threshold options each = hundreds of parameter choices
  If these were optimized on the same data they're evaluated on:
    The true out-of-sample accuracy is likely 3-7% lower than reported.

THE FIX:
  Your 65% accuracy on the retrained model is IN-SAMPLE accuracy.
  The critical number is OOS (out-of-sample) accuracy.
  
  You need: A dataset the model has NEVER seen during training.
  Use the 4 months being harvested now (Dec 2025 → Apr 2026) as pure OOS test.
  Run the trained brain on those months BEFORE telling the model the answers.
  
  IF OOS accuracy ≥ 60%: Real edge. Deploy with confidence.
  IF OOS accuracy 55-59%: Real but smaller edge. Reduce position sizes.
  IF OOS accuracy < 55%: Overfit. Retrain with regularization. Do not deploy.
```

### Lie 6 — The Psychology Gap (The Final Boss)

```
WHAT IT IS:
  Even if every other assumption is perfect,
  you are a human being who will experience:
  - A 10-trade losing streak (statistically inevitable with any strategy)
  - A 5% drawdown on real money (feels very different from paper)
  - The temptation to override the model during its losing periods
  - The temptation to size up after a winning streak
  - News events that feel like they "should" override the signal

THE MATH OF THE LOSING STREAK:
  At 79.49% win rate, the probability of losing N trades in a row:
  
  Lose 3 in a row:  (0.2051)³ = 0.86%  → happens roughly every 116 trades
  Lose 4 in a row:  (0.2051)⁴ = 0.18%  → happens roughly every 565 trades
  Lose 5 in a row:  (0.2051)⁵ = 0.036% → happens roughly every 2,777 trades
  
  With 11,090 signals → you WILL see 3-loss streaks ~96 times in this dataset.
  You WILL see 4-loss streaks ~20 times.
  
  WHEN (not if) you hit a 4-loss streak with real money:
  The model is not broken.
  The edge is not gone.
  The math is working exactly as expected.
  You must not override it.
  
  This is why paper trading with emotional tracking is mandatory.
  You need to know in advance how YOU respond to losing streaks.
```

---

## Phase 0 — Pre-Deployment Audit

> Complete every item before any live capital. This is your preflight checklist. No exceptions.

### The 48-Hour Audit

```
EXECUTION AUDIT:
[ ] Verify all signal entries use NEXT BAR OPEN, not signal bar close
[ ] Re-run backtest with 0.1% slippage assumption on liquid names
[ ] Re-run backtest with 0.25% slippage on UVXY/VIXY/low-liquidity names
[ ] Confirm commission structure with your broker ($0 retail or model institutional)
[ ] Document average bid-ask spread for each of your 57 assets
[ ] Check each asset for reverse splits or corporate actions in test period

DATA AUDIT:
[ ] Pull raw (unadjusted) price data for UVXY and VIXY — verify patterns hold
[ ] Verify earnings data timestamps: are you using announce date or next trading day?
[ ] Confirm FRED macro data is available at the correct lag (FRED often has 1-day delay)
[ ] Check FVG / OB identification: confirmed on CLOSED candle or open candle?
[ ] Run 10 random backtest signals manually: verify each feature was actually computable

REGIME AUDIT:
[ ] Document the exact macro regime during your 8-month training period
[ ] Find a different-regime period in prior years and run model on it
[ ] Run XGBoost feature importance — which 10 features matter most?
[ ] For each top-10 feature: does it still exist in live market conditions?

MODEL AUDIT:
[ ] Confirm XGBoost is NOT peeking at future data during feature calculation
[ ] Verify train/test split is TIME-BASED (not random) — no future data in training
[ ] Check that the 65% accuracy is on a held-out test set, not training data
[ ] Compute confidence calibration: when model says 80% confidence, is it right 80%?

OUTPUT:
Audit report saved to: /vault/validation/pre_deployment_audit.md
Every item marked PASS, FAIL, or NEEDS INVESTIGATION
No live capital until all FAIL items resolved
```

---

## Phase 1 — Paper Trading (The Real Test)

> Paper trading is not practice. It is the most important phase of validation. Treat it as if real money is at risk. Track every decision. Track every emotion.

### The Paper Trading Protocol

```
DURATION: 3 months minimum / 200 signals minimum (whichever is LONGER)
WHY: You need enough trades for statistical significance AND enough time
     to experience at least one regime shift and one losing streak.

EXECUTION RULES (simulate reality exactly):
  Enter at: Next bar open after signal (NOT at signal price)
  Slippage: Manually deduct 0.1% per trade from equity curve
  Position size: Exactly as the live system would size (A+/A/B/C grades)
  Stop loss: Hard stop, same as live rules — no "it would have recovered"
  Exits: At exact ICT targets, same as live system

RECORD FOR EVERY TRADE:
  - Asset, direction, grade (A+/A/B/C)
  - Signal timestamp and features that triggered it
  - Entry price (next open), stop, targets
  - Exit price, exit reason (target/stop/ICT signal)
  - Slippage actually observed (check real bid-ask at entry time)
  - Emotional state at entry (1-10 confidence; any hesitation?)
  - Did you want to override the model? (YES/NO — be honest)
  - Outcome: Win/Loss/Breakeven, R-multiple

WHAT YOU ARE TESTING:
  Primary: Does the edge hold in live-time conditions vs backtest?
  Secondary: Does execution assumption match reality?
  Tertiary: Can YOU execute this system without psychological interference?
```

### Paper Trading Success Criteria

```python
# These metrics must be met before ANY real capital deployment

MINIMUM THRESHOLDS (must clear ALL of these):

win_rate_paper         >= 0.72    # Allow 7.5% degradation from 79.49%
sharpe_ratio_paper     >= 1.5     # Annualized
max_drawdown_paper     <= 0.12    # 12% max drawdown
avg_r_multiple_paper   >= 1.8     # Average win in R terms
model_accuracy_paper   >= 0.60    # Live XGBoost accuracy (vs 65% backtest)
n_trades_minimum       >= 200     # Statistical minimum
regime_coverage        >= 2       # Must see at least 2 different regimes

# Degradation tolerance:
# Paper win rate should be within 10% of backtest win rate
# If degradation > 10%: investigate before deployment
# If degradation > 15%: halt deployment; audit execution assumptions

degradation_threshold = 0.10
if (backtest_win_rate - paper_win_rate) > degradation_threshold:
    print("HALT: Execution degradation exceeds tolerance")
    print("Investigate: slippage? lookahead? regime shift?")
```

### The Paper Trade Journal Structure

```markdown
## Paper Trade #[N] — [DATE] [ASSET] [LONG/SHORT]

### Signal
- Grade: A+ / A / B / C
- XGBoost Confidence: [X]%
- ICT Setup: [Liquidity Sweep / OB Entry / FVG / MSS]
- Session: London / NY Open / PM Session
- Regime: VIX=[X], Trend=[Bull/Bear/Neutral]

### Entry
- Signal Time: [TIME]
- Signal Price: [PRICE] (what backtest would have used)
- Actual Entry: Next open = [PRICE] (what you actually get)
- Slippage: [$ difference]
- Position Size: [X shares] at [Y% risk]

### Management
- Stop: [PRICE] — [$ risk]
- T1: [PRICE] — [R:R ratio]
- T2: [PRICE] — [R:R ratio]
- T3: [PRICE] — [R:R ratio]

### Exit
- Exit Price: [PRICE]
- Exit Reason: [T1/T2/T3/Stop/Time/Signal invalidation]
- Result: [+X.XX R] or [-1.0 R]

### Reality Check
- Did slippage match assumption? [YES/NO — actual was X%]
- Did model confidence correlate with outcome? [YES/NO]
- Emotional override temptation: [YES/NO — describe]
- Any lookahead issues discovered? [YES/NO]

### Running Totals
- Trade N of [goal]
- Running Win Rate: [X]%
- Running Sharpe: [X]
- Equity Curve Delta vs Backtest: [+/-X%]
```

---

## Phase 2 — Micro Live Deployment

> Real money. Small size. The psychological validation layer.

### Entry Criteria to Phase 2

```
GATE: All of these must be true before any live money:

[ ] Paper trading complete (200+ trades, 3+ months)
[ ] Paper win rate ≥ 72%
[ ] Paper Sharpe ≥ 1.5
[ ] Pre-deployment audit: all critical items PASS
[ ] OOS accuracy of XGBoost on harvested 4-month data ≥ 60%
[ ] You have experienced at least ONE 3-loss streak in paper trading
    and did not override the system during it
[ ] Broker live connection tested: orders flow, fills confirmed, data feed verified
[ ] Kill switch protocol documented and tested
[ ] Maximum daily loss limit configured in broker system
```

### Phase 2 Capital Structure

```
ACCOUNT STRUCTURE FOR PHASE 2:

Dedicated account: SEPARATE from any other savings/investments
  Reason: You must be able to lose 100% of this account without
  life consequences. If losing this account would cause hardship,
  the account is too large.

Starting capital: $5,000 - $15,000 (your choice based on circumstances)
  This is not about returns. It is about testing execution with real stakes.

Position sizing:
  A+ signal: Risk 0.5% per trade (NOT the 150% base size from Phase 4)
  A signal:  Risk 0.25% per trade
  B signal:  Risk 0.1% per trade
  C signal:  Skip entirely in Phase 2

Maximum daily loss: 2% of Phase 2 account
  IF hit: Stop trading for the day. No exceptions.

Maximum weekly loss: 5% of Phase 2 account
  IF hit: Stop trading for the week. Audit what happened.

Phase 2 Duration: 3 months / 300 live trades minimum

WHAT YOU ARE PROVING IN PHASE 2:
  1. Execution works in live market (fills, slippage, system connectivity)
  2. YOUR PSYCHOLOGY holds under real P&L
  3. Edge survives real-money conditions
  4. No hidden bugs in the execution pipeline
```

### Phase 2 Success Criteria

```python
# After 300 live trades (3+ months):

MINIMUM TO ADVANCE TO PHASE 3:

win_rate_live_p2       >= 0.70    # Allow 9.5% degradation (execution + psychology)
sharpe_ratio_p2        >= 1.2
max_drawdown_p2        <= 0.15    # 15% max allowed before investigation
n_trades_p2            >= 300
psychology_score       >= 0.85    # (trades executed per signal) / (signals generated)
                                  # Did you take the trades the system told you to?
                                  # 0.85+ = you followed the system 85% of the time
no_override_losses     = True     # Trades you manually overrode must be tracked
                                  # Overrides that lost money = expensive lessons
                                  # Overrides that won money = dangerous false confidence

CRITICAL COMPARISON:
paper_to_live_degradation = paper_win_rate - live_win_rate
IF degradation > 0.05 (5%):
    investigate:
        - Slippage worse than paper assumption?
        - Psychological overrides hurting performance?
        - Market microstructure different from paper simulation?
        - System bugs only visible under real execution?
```

---

## Phase 3 — Controlled Scaling

> The edge is validated. Capital deployment scales. Psychology is tested at increasing stakes.

### The Scaling Ladder

```
PRINCIPLE: Double capital only when CURRENT scale is proven stable.
Never skip a rung. The psychology changes at every level.

RUNG 1 — PROOF OF CONCEPT: $5K–$15K (Phase 2)
  Duration: 3 months minimum
  Purpose: Execution validation, psychology baseline
  Success: 300 trades, ≥ 70% win rate, ≤ 15% drawdown

RUNG 2 — VALIDATION: $25K–$50K
  Duration: 3 months minimum  
  Purpose: Confirm edge holds at higher stake psychological pressure
  New Challenge: PDT rule ($25K min for day trades — but you're swing trading)
  Position size: A+ = 1% risk; A = 0.75%; B = 0.5%
  Success: 200 trades, ≥ 70% win rate, psychology score ≥ 0.85

RUNG 3 — GROWTH DEPLOYMENT: $50K–$150K
  Duration: 6 months
  Purpose: Scale into A+ signals at meaningful capital
  Introduce: Grade-based scaling (A+ at 150% base from your V2 model)
  New Challenge: Market impact beginning on smaller names
  Monitor: Execution quality metrics — is slippage increasing with size?
  Success: 400 trades, ≥ 69% win rate, ≤ 18% max drawdown

RUNG 4 — INSTITUTIONAL: $150K–$500K
  Duration: Ongoing
  Purpose: This is the desk operating at institutional capacity
  Full V2 scaling model deployed
  All 57 assets running
  Sector/regime filters fully operational
  Kimi + XGBoost dual confirmation for size-up signals
  
RUNG 5 — SCALING CAPITAL: $500K+
  At this level:
  - Some of your 57 assets have market impact concerns
  - Universe expansion needed (more assets, same quality filter)
  - Execution algorithms needed (VWAP, TWAP for large orders)
  - Prime brokerage relationships become relevant

NEVER ADVANCE A RUNG BECAUSE OF IMPATIENCE.
ONLY ADVANCE BECAUSE CURRENT RUNG METRICS ARE MET.
```

### The Drawdown Response Protocol

```
At every scale, these are your responses to drawdowns:

DRAWDOWN 0-5%: Normal. No action. Continue executing.
  This is within expected variance for any day.

DRAWDOWN 5-10%: Yellow Alert.
  [ ] Review last 20 trades: is win rate still tracking?
  [ ] Check if macro regime has shifted (VIX spike? yield curve move?)
  [ ] No reduction in position size yet
  [ ] Increase logging and monitoring

DRAWDOWN 10-15%: Orange Alert.
  [ ] Reduce ALL position sizes by 50% immediately
  [ ] No new Tier C or Tier B signals — A and A+ only
  [ ] Full audit of last 30 trades: any pattern in losses?
  [ ] Check if a new regime is breaking the model
  [ ] Do NOT stop trading — reduced size keeps you in the game

DRAWDOWN 15-20%: Red Alert.
  [ ] STOP all live trading
  [ ] Full model audit: is the edge still there?
  [ ] Run current market data through paper trading for 2 weeks
  [ ] Only resume if paper trading confirms edge is intact
  [ ] Resume at 25% of normal position size
  [ ] Build back gradually

DRAWDOWN > 20%: Kill Switch.
  [ ] Hard stop: all positions closed
  [ ] Full investigation: see Kill Switch Protocol
  [ ] This should NEVER happen with proper Rung management
      (reducing size at 10% prevents getting to 20%)
```

---

## Phase 4 — Institutional Deployment

> When Phase 3 is stable across 500+ live trades and multiple regime transitions.

### Full V2 Scaling Model Deployment

```
At this stage, the system runs as designed:

SIGNAL GRADES (from your V2 model):
  A+ (Score ≥ 8.5): 150% base position size
    Conditions: XGBoost ≥ 75% confidence + ICT A+ setup + Kimi confirmation
    Expected frequency: 5-10% of signals

  A (Score 7.0-8.4): 100% base position size
    Conditions: XGBoost ≥ 65% confidence + solid ICT setup
    Expected frequency: 20-30% of signals

  B (Score 5.5-6.9): 50% base position size
    Conditions: XGBoost ≥ 55% + reasonable setup
    Expected frequency: 40-50% of signals

  C (Score < 5.5): 25% base position size
    Use sparingly. Monitor win rate for C signals specifically.
    IF C signal win rate < 55%: eliminate C trades entirely.

BASE POSITION SIZE (at $150K account, 1% base risk = $1,500 per trade):
  A+ signal: $2,250 risk per trade
  A signal:  $1,500 risk per trade
  B signal:    $750 risk per trade
  C signal:    $375 risk per trade (rarely used)

CONCURRENT POSITIONS:
  Maximum 5 open positions simultaneously
  Maximum 2 A+ positions simultaneously
  Maximum total account risk at any time: 8%
```

---

## The Statistical Validation Framework

> The mathematical proof that the edge is real at every stage.

### The Sequential Probability Ratio Test (SPRT)

```python
import numpy as np
from scipy import stats

def sprt_edge_test(trade_results, h0_win_rate=0.50, h1_win_rate=0.70,
                   alpha=0.05, beta=0.20):
    """
    Sequential test: after each trade, determine if edge is proven
    or disproven with statistical confidence.
    
    h0: win rate = 0.50 (no edge; coin flip)
    h1: win rate = 0.70 (your expected live edge after degradation)
    alpha: false positive rate (5%)
    beta: false negative rate (20%)
    """
    A = (1 - beta) / alpha            # Upper boundary: edge proven
    B = beta / (1 - alpha)            # Lower boundary: no edge proven

    log_likelihood = 0
    decisions = []

    for i, win in enumerate(trade_results):
        # Update likelihood ratio
        if win:
            log_likelihood += np.log(h1_win_rate / h0_win_rate)
        else:
            log_likelihood += np.log((1-h1_win_rate) / (1-h0_win_rate))

        ratio = np.exp(log_likelihood)

        if ratio >= A:
            decision = 'EDGE_PROVEN'
        elif ratio <= B:
            decision = 'NO_EDGE_PROVEN'
        else:
            decision = 'CONTINUE_TESTING'

        decisions.append({
            'trade_n': i+1,
            'cumulative_wins': sum(trade_results[:i+1]),
            'win_rate': sum(trade_results[:i+1]) / (i+1),
            'likelihood_ratio': ratio,
            'decision': decision
        })

        if decision != 'CONTINUE_TESTING':
            print(f"Trade {i+1}: {decision} (LR={ratio:.3f})")
            break

    return decisions

# Expected result with true 72% win rate:
# Edge proven in approximately 50-80 trades
# This gives you early confirmation without waiting for 300 trades
```

### Running Bayesian Win Rate Estimate

```python
from scipy.stats import beta as beta_dist
import numpy as np

class LiveEdgeTracker:
    """
    Track edge in real-time using Bayesian updating.
    After every trade, update your estimate of the true win rate.
    """

    def __init__(self, prior_alpha=1, prior_beta=1):
        # Start with uninformative prior (flat)
        # OR use backtest as prior: alpha=7949, beta=2051 (from 79.49% win rate)
        # Strong prior = more trades needed to shift estimate
        # Weak prior = new data shifts estimate quickly
        # RECOMMENDATION: Use weak prior for live validation phase
        self.alpha = prior_alpha  # wins + 1
        self.beta  = prior_beta   # losses + 1
        self.trades = []

    def update(self, outcome: bool, grade: str, asset: str):
        """Update after each trade."""
        self.alpha += int(outcome)
        self.beta  += int(not outcome)
        self.trades.append({'outcome': outcome, 'grade': grade, 'asset': asset})

    def current_estimate(self):
        """Current posterior estimate of win rate."""
        mean = self.alpha / (self.alpha + self.beta)
        ci_95 = beta_dist.interval(0.95, self.alpha, self.beta)
        ci_99 = beta_dist.interval(0.99, self.alpha, self.beta)

        return {
            'mean_win_rate':    mean,
            'ci_95':            ci_95,
            'ci_99':            ci_99,
            'n_trades':         len(self.trades),
            'edge_above_50pct': ci_95[0] > 0.50,   # Are we certain edge > 50%?
            'edge_above_60pct': ci_95[0] > 0.60,   # Are we certain edge > 60%?
            'edge_above_65pct': ci_95[0] > 0.65,   # Are we certain edge > 65%?
        }

    def grade_breakdown(self):
        """Win rate by signal grade — critical for scaling validation."""
        for grade in ['A+', 'A', 'B', 'C']:
            grade_trades = [t for t in self.trades if t['grade'] == grade]
            if len(grade_trades) >= 20:  # minimum sample
                win_rate = sum(t['outcome'] for t in grade_trades) / len(grade_trades)
                print(f"Grade {grade}: {win_rate:.1%} win rate ({len(grade_trades)} trades)")

# Use this on EVERY live trade from Day 1
tracker = LiveEdgeTracker()
```

### The Weekly Statistical Summary Template

```markdown
## Live Performance Summary — Week [N] | [DATE RANGE]

### Trade Statistics
- Trades This Week: [N]
- Cumulative Trades: [N]
- Weekly Win Rate: [X]%
- Cumulative Win Rate: [X]% (CI: [X]% - [X]%)
- Bayesian Win Rate Estimate: [X]% (95% CI: [X]% - [X]%)
- Edge Confirmation Status: PROVEN / TESTING / DISPROVEN

### By Grade
| Grade | Trades | Win Rate | Avg R | Expected Win Rate |
|---|---|---|---|---|
| A+ | N | X% | X.X | 85%+ |
| A | N | X% | X.X | 75%+ |
| B | N | X% | X.X | 65%+ |
| C | N | X% | X.X | 55%+ |

### Execution Quality
- Avg Slippage Per Trade: [X]% (target < 0.12%)
- Fill Rate: [X]% (orders filled vs signals generated)
- Avg Entry vs Signal Price: +[X]% (how much worse than backtest?)
- Execution Degradation vs Backtest: [X]%

### Model Performance
- XGBoost Live Accuracy: [X]% (backtest: 65%)
- Regime: [BULL/BEAR/NEUTRAL] — VIX: [X]
- Top 3 Assets This Week: [list]
- Bottom 3 Assets (any pattern?): [list]

### Equity Curve
- Week Return: [X]%
- Cumulative Return: [X]%
- Max Drawdown This Week: [X]%
- Running Max Drawdown: [X]%
- Sharpe (annualized): [X]

### Psychology Score
- Signals Generated: [N]
- Trades Taken: [N]
- Compliance Rate: [X]% (taken/generated)
- Override Attempts: [N] (wins: N, losses: N)
- Note any emotional interference: [text]

### Backtest vs Live Comparison
| Metric | Backtest | Paper | Live Phase 2 | Live Phase 3 |
|---|---|---|---|---|
| Win Rate | 79.49% | [X]% | [X]% | [X]% |
| Sharpe | [X] | [X] | [X] | [X] |
| Max DD | [X]% | [X]% | [X]% | [X]% |
| Avg R | [X] | [X] | [X] | [X] |

### Decision
[ ] Continue at current scale
[ ] Advance to next rung (criteria met)
[ ] Reduce size (drawdown alert)
[ ] Halt and audit (kill switch criteria met)
```

---

## Live Performance Tracking

### The Equity Curve Dashboard

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LiveEquityCurve:
    """Real-time equity curve with backtest comparison."""

    def __init__(self, starting_capital, backtest_daily_returns):
        self.capital = starting_capital
        self.equity_curve = [starting_capital]
        self.backtest = backtest_daily_returns
        self.trade_log = []

    def add_trade(self, r_multiple, capital_risked_pct):
        """Add a completed trade result."""
        pnl_pct = r_multiple * capital_risked_pct
        self.capital *= (1 + pnl_pct)
        self.equity_curve.append(self.capital)
        self.trade_log.append(r_multiple)

    def performance_metrics(self):
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        cumulative = (self.capital - self.equity_curve[0]) / self.equity_curve[0]
        peak = max(self.equity_curve)
        drawdown = (self.capital - peak) / peak

        return {
            'total_return':    cumulative,
            'sharpe':          returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown':    drawdown,
            'current_capital': self.capital,
            'n_trades':        len(self.trade_log),
            'win_rate':        sum(1 for r in self.trade_log if r > 0) / len(self.trade_log)
        }

    def vs_backtest_score(self):
        """How closely is live tracking the backtest prediction?"""
        # Live performance / Backtest prediction
        # 0.90+ = excellent tracking (within 10%)
        # 0.80-0.90 = acceptable (execution friction)
        # < 0.80 = investigation needed
        live_return = self.performance_metrics()['total_return']
        expected_return = (1 + self.backtest).prod() - 1
        return live_return / expected_return if expected_return != 0 else 0
```

### The Degradation Detection System

```python
def detect_performance_degradation(live_trades, backtest_stats, window=50):
    """
    Rolling window comparison of live vs backtest performance.
    Alert if degradation exceeds thresholds.
    """
    if len(live_trades) < window:
        return 'INSUFFICIENT_DATA'

    recent = live_trades[-window:]
    recent_win_rate = sum(1 for t in recent if t['r_multiple'] > 0) / window
    recent_avg_r    = sum(t['r_multiple'] for t in recent) / window

    degradation_wr = backtest_stats['win_rate'] - recent_win_rate
    degradation_r  = backtest_stats['avg_r'] - recent_avg_r

    # Regime shift detection
    vix_now  = get_current_vix()
    vix_training = backtest_stats['avg_training_vix']
    regime_shift = abs(vix_now - vix_training) > 10  # VIX shifted >10 points

    alerts = []
    if degradation_wr > 0.10:
        alerts.append(f"WIN RATE DEGRADATION: {degradation_wr:.1%} below backtest")
    if degradation_r > 0.5:
        alerts.append(f"R-MULTIPLE DEGRADATION: {degradation_r:.2f}R below backtest")
    if regime_shift:
        alerts.append(f"REGIME SHIFT DETECTED: VIX={vix_now} vs training={vix_training}")

    severity = 'NORMAL' if not alerts else \
               'CAUTION' if len(alerts) == 1 else \
               'WARNING' if len(alerts) == 2 else 'CRITICAL'

    return {
        'severity':         severity,
        'alerts':           alerts,
        'live_win_rate':    recent_win_rate,
        'backtest_win_rate': backtest_stats['win_rate'],
        'degradation':      degradation_wr,
        'regime_shift':     regime_shift,
        'action': {
            'NORMAL':   'Continue at full size',
            'CAUTION':  'Monitor closely; no size change',
            'WARNING':  'Reduce to 50% position size; audit',
            'CRITICAL': 'Stop trading; full investigation'
        }[severity]
    }
```

---

## The Psychological Reality Layer

> This is not soft. Psychology is a quantitative variable. Measure it. Manage it.

### The Pre-Trade State Check

```
Run this check before EVERY trading session:

PHYSICAL:
  [ ] Sleep: 7+ hours last night? (IF NO: reduce to B-grade trades only)
  [ ] No significant substances in last 12 hours
  [ ] Not physically ill or in pain

EMOTIONAL:
  [ ] Emotional state: Neutral (7-10) or better on scale of 1-10
  [ ] No major life stress event in last 48 hours that isn't processed
  [ ] Not angry about a previous losing trade

FINANCIAL:
  [ ] Current drawdown: within acceptable range (<5%)?
  [ ] No financial pressure (rent, bills) directly tied to this account
  [ ] Last week's performance: can I accept it without need to "make it back"?

IF ANY BOX UNCHECKED:
  Trade at minimum size (C-grade rules) OR don't trade at all.
  The system needs your full, unimpaired execution.
  A bad execution day undoes multiple good system days.
```

### Measuring Override Temptation (Quantitatively)

```python
class PsychologyTracker:
    """
    Track every moment you wanted to override the system.
    This is as important as trade results.
    """

    def __init__(self):
        self.override_log = []
        self.execution_log = []

    def log_signal(self, signal_id, grade, direction, did_take: bool, reason_skip: str = None):
        """Log every signal — taken or skipped."""
        self.execution_log.append({
            'signal_id':   signal_id,
            'grade':       grade,
            'direction':   direction,
            'took_trade':  did_take,
            'skip_reason': reason_skip  # 'fear', 'news', 'gut', 'drawdown', etc.
        })

    def log_override_urge(self, trade_id, urge_type: str, acted_on: bool, outcome: str):
        """
        Log every time you felt the urge to override the system.
        urge_type: 'early_exit', 'move_stop', 'skip_entry', 'size_up', 'size_down'
        acted_on: did you actually override?
        outcome: 'beneficial', 'harmful', 'neutral' (measure result)
        """
        self.override_log.append({
            'trade_id':   trade_id,
            'urge_type':  urge_type,
            'acted_on':   acted_on,
            'outcome':    outcome
        })

    def psychology_report(self):
        n_signals = len(self.execution_log)
        n_taken   = sum(1 for e in self.execution_log if e['took_trade'])
        compliance = n_taken / n_signals if n_signals > 0 else 0

        n_overrides = sum(1 for o in self.override_log if o['acted_on'])
        harmful_overrides = sum(1 for o in self.override_log
                                 if o['acted_on'] and o['outcome'] == 'harmful')

        return {
            'compliance_rate':      compliance,        # Target: > 85%
            'override_rate':        n_overrides / n_signals,
            'harmful_override_rate': harmful_overrides / max(n_overrides, 1),
            'top_skip_reasons':     pd.Series([e['skip_reason'] for e in self.execution_log
                                               if not e['took_trade']]).value_counts().to_dict()
        }

# The compliance_rate tells you more about your edge than the win rate does.
# A 79% backtest win rate executed at 70% compliance = a 55% win rate in practice.
# The system is only as good as your ability to run it.
```

---

## The Kill Switch Protocol

> Defined in advance. Non-negotiable. The circuit breakers that protect capital from catastrophic loss.

### Automatic Kill Switches (No Human Decision Required)

```
LEVEL 1 — DAILY STOP (automated):
  Trigger: Daily loss exceeds 2% of account
  Action: All new orders blocked for remainder of day
  Override: Not possible. Hard coded in broker API.

LEVEL 2 — WEEKLY STOP (automated):
  Trigger: Weekly loss exceeds 5% of account
  Action: All new orders blocked until Monday 9:30 AM ET
  Override: Not possible.

LEVEL 3 — DRAWDOWN STOP (automated):
  Trigger: Account drawdown from high-water mark exceeds 15%
  Action: All positions reduced by 50%; no new trades until human review
  Override: Requires written documentation of cause and plan.

LEVEL 4 — MODEL FAILURE STOP (automated):
  Trigger: Rolling 30-trade win rate falls below 55%
  Action: All new signals suspended pending audit
  Override: Requires model audit, regime analysis, minimum 2-week paper trading confirmation.
```

### Manual Kill Switch Conditions

```
TRIGGER ANY OF THESE → STOP ALL TRADING IMMEDIATELY:

[ ] System anomaly: Model generating signals outside historical confidence ranges
[ ] Data feed failure: Market data delayed or corrupted
[ ] Execution anomaly: Orders not filling as expected; slippage > 3x normal
[ ] Macro event: Unforeseen systemic event (circuit breaker, exchange halt, 9/11-type event)
[ ] Personal crisis: Any personal situation impairs judgment
[ ] Broker issue: Account access problems, margin call, settlement issue

KILL SWITCH PROCEDURE:
  Step 1: Close all open positions at market (accept whatever fills)
  Step 2: Cancel all pending orders
  Step 3: Document: timestamp, account value, reason for kill switch
  Step 4: Do not trade for minimum 48 hours
  Step 5: Full investigation before resumption (see audit protocol)
  Step 6: Resume at 25% position size; build back incrementally
```

---

## The Milestone Roadmap

> The exact sequence from today to institutional operation. Dates are illustrative — advance only when criteria are met.

```
TODAY: Pre-Deployment Audit
  → 48-hour audit of all six lie-types
  → Re-run backtest with realistic execution assumptions
  → OOS test on harvested 4-month data
  → Document real adjusted win rate (may be 72-75%; still excellent)

MONTH 1-3: Paper Trading
  → 200+ signals executed as paper trades
  → Full psychology tracking
  → Bayesian win rate estimate forming
  → Target: 72%+ paper win rate, SPRT edge proven

MONTH 4-6: Phase 2 Micro Live ($5K-$15K)
  → 300 live trades at micro size
  → Execution quality metrics collected
  → Psychology score measured
  → Target: 70%+ live win rate, Sharpe ≥ 1.2, ≤ 15% drawdown

MONTH 7-9: Phase 3 Rung 2 ($25K-$50K)
  → 200 additional live trades at moderate size
  → At least 1 regime transition observed and navigated
  → Grade-based scaling introduced
  → Target: 69%+ win rate, confirm A+ grade outperforms

MONTH 10-18: Phase 3 Rung 3 ($50K-$150K)
  → Full V2 scaling model deployed
  → All 57 assets running with volatility gate active
  → Sector/regime filters validated
  → Kimi + XGBoost dual confirmation for max size signals
  → Target: 68%+ win rate, Sharpe ≥ 1.5, compounding visible

MONTH 18-36: Phase 4 Institutional ($150K-$500K)
  → Universe expansion as capital grows
  → Research program feeding new validated signals into XGBoost
  → Tier 3 experiments being validated and promoted
  → The desk is a compounding machine

THE NUMBER THAT MATTERS MOST:
  Not the win rate.
  Not the Sharpe.
  Not the backtest return.

  The number that matters is:

  LIVE TRADE #1,000.

  At 1,000 live trades, the law of large numbers has done its work.
  At 1,000 live trades, you know your real win rate with a 99% confidence interval
  that is ±3% wide.
  At 1,000 live trades, you are no longer validating.
  You are operating.

  Every trade from now until Trade #1,000 is scientific data collection.
  Treat it accordingly.
```

---

## Related Notes

- [[Edge Research Program MOC]]
- [[Imbalance Engine MOC]]
- [[ICT Swing Trade Decision Engine]]
- [[Intelligence System MOC]]
- [[XGBoost Bias Engine]]
- [[Kimi Fault Detector]]
- [[Trade Journal Template]]
- [[Psychology Tracker]]
- [[Weekly Statistical Summary]]
- [[Kill Switch Log]]
- [[Equity Curve Dashboard]]
- [[Pre-Deployment Audit Report]]

---

*The backtest proved the edge exists in data. Paper trading proves the edge survives real time. Phase 2 proves the edge survives real money. Phase 3 proves the edge survives real scale. Each phase is a different question with a different answer. You have answered the first question. Answer them all in order. The 1,000th live trade is not the finish line — it is the beginning.*
