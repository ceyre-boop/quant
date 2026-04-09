# The Imbalance Engine — Mathematical Market Fault Detection 🔭

#imbalance #prediction #mathematical-edge #dual-model #conviction #structural-fault #weeks-months-ahead

> **Core Philosophy:** Markets are not random. They are thermodynamic systems under pressure. Imbalances build — in positioning, in valuation, in credit, in volatility surfaces, in cross-asset flows — long before price moves. Mathematics detects the pressure. The XGBoost layer reads the daily temperature. The Kimi layer reads the fault lines. When both point at the same crack in the same wall — you bet like Petroulas.

---

## 🗺️ Map of Contents

- [[#The Two-Model Conviction Architecture]]
- [[#Layer A — XGBoost Continuous Intelligence]]
- [[#Layer B — Kimi Fault Detector (The Petroulas Protocol)]]
- [[#The Dual Confirmation Signal]]
- [[#Mathematical Frameworks for Early Detection]]
- [[#Structural Imbalance Indicators]]
- [[#The Mathematics of Divergence]]
- [[#Regime Detection Before Transition]]
- [[#Positioning Imbalance Mathematics]]
- [[#Volatility Surface Fault Lines]]
- [[#Cross-Asset Stress Propagation]]
- [[#The Conviction Sizing Model]]
- [[#Historical Imbalance Case Studies]]
- [[#Integration with Existing Pipeline]]
- [[#The Full Detection Checklist]]

---

## The Two-Model Conviction Architecture

> Two completely different intelligences looking at the same market from different angles. Disagreement = noise. Agreement on a fault = rare, high-conviction, size up.

```
┌─────────────────────────────────────────────────────────────┐
│                    MARKET DATA UNIVERSE                      │
│  Price | Volume | Filings | Macro | Options | Flows | News  │
└─────────────┬───────────────────────────┬───────────────────┘
              │                           │
              ▼                           ▼
   ┌──────────────────┐        ┌──────────────────────┐
   │  XGBOOST ENGINE  │        │    KIMI FAULT ENGINE  │
   │  (The Quant)     │        │   (The Petroulas)     │
   │                  │        │                       │
   │ - 29 features    │        │ - Structural thesis   │
   │ - 57 assets      │        │ - Narrative analysis  │
   │ - 58.3% accuracy │        │ - Imbalance magnitude │
   │ - Direction bias │        │ - Conviction score    │
   │ - Confidence     │        │ - "Is this a fault?"  │
   │   0.45/0.55      │        │   YES / NO / MAYBE    │
   └────────┬─────────┘        └──────────┬────────────┘
            │                             │
            ▼                             ▼
   ┌────────────────────────────────────────────────┐
   │            DUAL CONFIRMATION GATE              │
   │                                                │
   │  BOTH agree on direction + fault exists?       │
   │                                                │
   │  IF YES → FAULT CONFIRMED                      │
   │    → Size up to 3-5% risk (Petroulas Mode)     │
   │    → Set 3-6 month time horizon                │
   │    → Ignore consensus analyst opinion          │
   │                                                │
   │  IF XGBoost only → Normal swing trade          │
   │    → Standard 1-2% risk (ICT Engine rules)     │
   │                                                │
   │  IF Kimi only → Monitor only                   │
   │    → Paper trade until XGBoost confirms        │
   │    → Thesis may be early                       │
   │                                                │
   │  IF neither → No position                      │
   └────────────────────────────────────────────────┘
```

---

## Layer A — XGBoost Continuous Intelligence

> Your existing system. Its job is to be right 58%+ of the time across all market conditions. Expand it with the following to detect early imbalances specifically.

### Expanding Your 29 Features Toward Imbalance Detection

Add these feature categories to `feature_builder.py` for multi-week-ahead prediction:

#### Macro Divergence Features (Weeks to Months Ahead)
```python
# Features that detect building pressure before price moves

# 1. Yield Curve Slope Rate of Change (2nd derivative)
yield_spread = treasury_10yr - treasury_2yr
yield_slope_velocity = yield_spread.diff(5)   # weekly change in spread
yield_slope_acceleration = yield_slope_velocity.diff(5)  # acceleration
# Accelerating inversion = credit stress building

# 2. Credit Spread Expansion Rate
hy_spread = HYG_yield - treasury_10yr        # high yield vs risk free
ig_spread = LQD_yield - treasury_10yr        # investment grade vs risk free
spread_divergence = hy_spread - ig_spread    # HY widening faster than IG = stress
credit_stress_velocity = hy_spread.diff(20)  # 4-week rate of change

# 3. M2 Money Supply Rate of Change (FRED: M2SL)
m2_growth_yoy = m2.pct_change(252)
m2_acceleration = m2_growth_yoy.diff(20)
# M2 deceleration → liquidity withdrawal → equities lag by 6-9 months

# 4. Real Rates vs Equity Earnings Yield
real_rate = treasury_10yr - inflation_breakeven_10yr  # TIPS spread
earnings_yield = 1 / sp500_pe_ratio
equity_risk_premium = earnings_yield - real_rate
# ERP compressing below 2% = overvalued; rising above 5% = undervalued
# This is your "imbalance meter" for equities vs bonds

# 5. Corporate Earnings Revision Breadth
upgrades = analyst_upgrades_count
downgrades = analyst_downgrades_count
revision_ratio = (upgrades - downgrades) / (upgrades + downgrades)
# Ratio falling below -0.3 = earnings deterioration building

# 6. ISM / PMI Divergence (Manufacturing vs Services)
ism_mfg = fred.get_series('MANEMP')
ism_svc = fred.get_series('NMFCI')
ism_divergence = ism_mfg - ism_svc
# Mfg falling while Services strong = sector rotation signal
# Both falling = broad economic contraction incoming

# 7. TED Spread (Interbank stress)
ted_spread = libor_3m - tbill_3m
ted_velocity = ted_spread.diff(10)
# TED > 0.5% and rising = banks distrusting each other = 2008-style signal

# 8. Dollar Liquidity Stress (Cross-Currency Basis)
# EURUSD 3M basis swap → negative = USD shortage globally
# When negative AND widening → global dollar stress → emerging markets at risk
```

#### Positioning Imbalance Features
```python
# 9. COT Commercials Net Position Z-Score
# Normalize commercial net position vs 3-year history
cot_commercial_zscore = (commercial_net - commercial_net.mean()) / commercial_net.std()
# Z-score < -2 = commercials most bearish in 3 years = major top signal
# Z-score > +2 = commercials most bullish in 3 years = major bottom signal

# 10. Options Skew (Put/Call Vol Ratio)
put_iv_25delta = options_chain_25delta_put_iv
call_iv_25delta = options_chain_25delta_call_iv
skew = put_iv_25delta - call_iv_25delta
skew_zscore = (skew - skew.rolling(60).mean()) / skew.rolling(60).std()
# Extreme negative skew = crash protection being bought = fear building

# 11. Short Interest Aggregate Change
si_aggregate = weighted_avg_short_interest(watchlist)
si_velocity = si_aggregate.diff(20)
# Rising aggregate short interest + falling price = bears in control
# Rising short interest + rising price = short squeeze fuel building

# 12. Margin Debt Rate of Change (FINRA)
margin_debt_mom = margin_debt.pct_change(20)
# Margin debt falling YoY = deleveraging = equities at risk
# This led market by 2-4 months in 2000, 2008, 2022
```

#### Valuation Divergence Features
```python
# 13. Sector P/E Relative to 10yr Average
for sector in sectors:
    pe_zscore = (sector_pe - sector_pe.mean()) / sector_pe.std()
    # Z-score > 2 = historically expensive = reversion risk
    # Z-score < -2 = historically cheap = recovery potential

# 14. Price-to-Sales (more robust than P/E; earnings can be manipulated)
ps_ratio_aggregate = market_cap_aggregate / revenue_aggregate
ps_zscore = (ps_ratio - ps_ratio.rolling(252*5).mean()) / ps_ratio.rolling(252*5).std()

# 15. Buffett Indicator (Total Market Cap / GDP)
total_market_cap = wilshire_5000_total_market_cap   # FRED: WILL5000IND
gdp = fred.get_series('GDP')
buffett_indicator = total_market_cap / gdp
buffett_zscore = (buffett_indicator - buffett_indicator.mean()) / buffett_indicator.std()
# > 2 std above mean = historically followed by poor 10yr returns
```

### XGBoost Time Horizon Extension

```python
# Your current model: short-term bias
# Add these target variables to predict DIFFERENT horizons

targets = {
    'direction_5d':  (returns_5d > 0).astype(int),    # existing
    'direction_20d': (returns_20d > 0).astype(int),   # NEW: 4-week
    'direction_60d': (returns_60d > 0).astype(int),   # NEW: 3-month
    'magnitude_20d': pd.qcut(returns_20d, 5, labels=False),  # NEW: quintile
    'drawdown_20d':  max_drawdown_next_20d > 0.10,    # NEW: risk of 10%+ drop
}

# Train SEPARATE models for each horizon
# The 60d model uses DIFFERENT feature importances than 5d
# Macro features dominate at 60d
# Technical features dominate at 5d

# Stacking: use 60d model output AS A FEATURE in the 5d model
# "The long-range radar is pointing at trouble" informs the short-range radar
```

---

## Layer B — Kimi Fault Detector (The Petroulas Protocol)

> The Kimi model is not trying to be right 58% of the time. It is trying to be right on the 3-5 times per year when the market has a genuine structural fault — and size those bets like Petroulas.

### What Kimi Is Looking For (The Fault Taxonomy)

```
FAULT TYPE 1 — VALUATION DISLOCATION
  Definition: An asset class is priced for a world that no longer exists
  or cannot exist given current macro constraints.

  Examples of this exact fault:
    - Tech stocks in 2000: priced for infinite growth; zero earnings
    - Housing in 2006: priced for infinite appreciation; no income
    - Bonds in 2021: priced for zero rates forever; inflation already beginning
    - Google in Petroulas example: priced for decline when dominance intact

  Kimi Detection Prompt:
    "Analyze [ASSET] vs its fundamental anchor [EARNINGS/FCF/REVENUE].
     At current price, what growth rate is implied for next 10 years?
     Is that growth rate physically achievable given market size, competition,
     and macro constraints? What does consensus assume vs what is mathematically
     possible? Identify the specific assumption that is wrong."

FAULT TYPE 2 — POSITIONING DISLOCATION
  Definition: Everyone is on the same side of the trade.
  When the crowd is maximum consensus → the trade is already priced in
  → any disappointment = massive unwind

  Signals:
    - COT Commercials at extreme (Z > 2 or Z < -2)
    - Fund manager surveys: >80% bullish or bearish on same theme
    - Short interest at historical extremes
    - Hedge fund crowding: 50+ funds holding same top 10 positions (13F data)

  Kimi Detection Prompt:
    "Based on 13F data showing [X] funds concentrated in [SECTOR/STOCK],
     and COT positioning showing commercials at [Z-score] extreme,
     and fund manager survey showing [X%] consensus on [THESIS]:
     What is the unwind scenario? What catalyst would force position reversal?
     How large is the price impact if 30% of longs exit simultaneously?
     Calculate the implied move using: avg daily volume × unwind days needed."

FAULT TYPE 3 — NARRATIVE DISLOCATION
  Definition: The dominant market narrative is factually incorrect,
  and math can prove it, even while everyone believes it.

  Examples:
    - "The Fed will cut rates 6 times in 2024" (priced in Jan 2024) → cut once
    - "AI will replace all software engineers" → labor data shows opposite
    - "China growth will keep commodity demand infinite" → demographic math says no

  Kimi Detection Prompt:
    "The current consensus narrative is: [NARRATIVE].
     This narrative implies: [MATHEMATICAL IMPLICATION].
     Test the following: [SPECIFIC FALSIFIABLE PREDICTION FROM NARRATIVE].
     Using data: [DATA SOURCES], is this prediction consistent with reality?
     If the narrative is wrong, what is the correct narrative, and what
     does it imply for [ASSET] price over 3-6 months?"

FAULT TYPE 4 — LIQUIDITY DISLOCATION
  Definition: An asset is priced assuming liquidity that will not be there
  when sellers need it. The fault is structural, not cyclical.

  Examples:
    - Corporate bond ETFs in 2020: daily liquidity, illiquid underlying
    - Crypto stablecoins with fractional reserves (Terra/LUNA)
    - Private equity "marks" vs actual exit multiples in rising rate environment
    - Real estate REITs in rising rate environment (duration mismatch)

  Kimi Detection Prompt:
    "Analyze the liquidity mismatch in [ASSET]:
     1. What liquidity does the asset promise to investors?
     2. What is the actual liquidity of underlying holdings?
     3. Under what scenario does this mismatch become a crisis?
     4. What is the mathematical probability of that scenario given current
        rates, flows, and market conditions?
     5. What is the first observable signal that the crack is appearing?"

FAULT TYPE 5 — REFLEXIVITY DISLOCATION
  Definition: Soros's reflexivity — market price is affecting fundamentals
  which are affecting price. The feedback loop is self-reinforcing and
  will eventually self-correct violently.

  Examples:
    - Rising stock price → company issues shares → buys more of what's rising
    - Passive investing inflows → top stocks get bigger → more inflows
    - Meme stock short squeeze: price rise → media coverage → retail buys → price rise
    - Real estate: rising prices → wealth effect → more buying → prices rise further

  Signal: The relationship between price and fundamentals has INVERTED.
  Normally: fundamentals drive price.
  In reflexivity: price is driving fundamentals.
  The fault: this cannot persist indefinitely. Mean reversion is violent when it comes.
```

### The Kimi Fault Scoring Prompt (Master Template)

```python
KIMI_FAULT_PROMPT = """
You are a structural fault detector for financial markets. Your role is identical
to a geologist studying fault lines — you are NOT predicting the earthquake date,
you are measuring the stress that has accumulated and determining if it is enough
to produce a major move.

ASSET UNDER ANALYSIS: {asset}
CURRENT PRICE: {price}
CURRENT CONSENSUS VIEW: {consensus}

DATA PROVIDED:
- Valuation metrics: {valuation_data}
- Positioning data (COT, 13F, short interest): {positioning_data}
- Macro context: {macro_data}
- Historical comparable periods: {comparable_periods}

TASK:
1. FAULT IDENTIFICATION: Does a structural imbalance exist? (YES/NO)
   If YES: classify as Type 1-5 (Valuation/Positioning/Narrative/Liquidity/Reflexivity)

2. MAGNITUDE ASSESSMENT: On a scale of 1-10:
   - 1-3: Minor imbalance; normal market noise
   - 4-6: Real imbalance; meaningful correction possible (10-20%)
   - 7-9: Major imbalance; significant move highly probable (20-50%)
   - 10: Systemic fault; tail risk event possible (50%+)

3. TIMING ESTIMATE: When does the imbalance likely resolve?
   - Catalyst-dependent: needs specific event to trigger
   - Self-resolving: mathematics will force resolution within [timeframe]
   - Unknown timing: fault confirmed but trigger unpredictable

4. THE CONSENSUS BLINDSPOT: Why do market analysts NOT see this?
   Be specific. What assumption are they making that is wrong?
   What data are they ignoring?

5. CONVICTION SCORE: 1-10
   Based on: quality of evidence, historical precedent, magnitude, timing visibility

6. FALSIFICATION: What specific observable outcome in the next 30 days
   would PROVE this thesis is wrong? Be mathematically specific.

OUTPUT FORMAT: JSON
{
  "fault_exists": true/false,
  "fault_type": "1-5 or None",
  "magnitude": 1-10,
  "direction": "LONG/SHORT/NEUTRAL",
  "timing_weeks": [min, max],
  "consensus_blindspot": "string",
  "conviction_score": 1-10,
  "falsification_test": "string",
  "petroulas_worthy": true/false  // magnitude >= 7 AND conviction >= 7
}
"""
```

---

## The Dual Confirmation Signal

```python
def dual_confirmation_check(xgb_result, kimi_result):
    """
    The gate that determines whether this is a normal trade
    or a Petroulas-mode conviction bet.
    """

    xgb_direction   = xgb_result['direction']       # LONG / SHORT / NEUTRAL
    xgb_confidence  = xgb_result['confidence']      # 0.0 to 1.0
    kimi_direction  = kimi_result['direction']       # LONG / SHORT / NEUTRAL
    kimi_magnitude  = kimi_result['magnitude']       # 1-10
    kimi_conviction = kimi_result['conviction_score']  # 1-10
    petroulas_flag  = kimi_result['petroulas_worthy']  # bool

    # ── GATE 1: Direction must agree ──────────────────────────
    if xgb_direction != kimi_direction:
        return {
            'signal': 'CONFLICT',
            'action': 'MONITOR_ONLY',
            'risk_pct': 0,
            'note': 'Models disagree. No position. Watch for convergence.'
        }

    # ── GATE 2: Petroulas Mode (Structural Fault Confirmed) ───
    if (petroulas_flag and
        xgb_confidence >= 0.60 and     # XGBoost highly confident
        kimi_magnitude >= 7 and         # Major fault
        kimi_conviction >= 7):          # High conviction thesis

        return {
            'signal': 'FAULT_CONFIRMED',
            'action': 'PETROULAS_MODE',
            'risk_pct': calculate_petroulas_risk(kimi_magnitude, kimi_conviction),
            'horizon_weeks': kimi_result['timing_weeks'],
            'note': f"STRUCTURAL FAULT. Type {kimi_result['fault_type']}. "
                    f"Size up. Ignore consensus. Hold to resolution.",
            'falsification': kimi_result['falsification_test']
        }

    # ── GATE 3: High Agreement (Not Fault-Level, But Elevated) ─
    if xgb_confidence >= 0.55 and kimi_conviction >= 5:
        return {
            'signal': 'HIGH_AGREEMENT',
            'action': 'ELEVATED_SIZE',
            'risk_pct': 2.0,
            'note': 'Strong signal. ICT engine rules apply. 2% risk.'
        }

    # ── GATE 4: Normal Swing Trade ─────────────────────────────
    if xgb_confidence >= 0.55:
        return {
            'signal': 'STANDARD',
            'action': 'ICT_ENGINE',
            'risk_pct': 1.0,
            'note': 'Normal setup. Follow ICT Decision Engine rules.'
        }

    return {
        'signal': 'WEAK',
        'action': 'NO_TRADE',
        'risk_pct': 0,
        'note': 'Insufficient confidence. Wait.'
    }


def calculate_petroulas_risk(magnitude, conviction):
    """
    Scale risk with fault quality.
    Maximum 5% for perfect 10/10/10 fault.
    Minimum 3% for threshold-clearing fault.
    """
    quality_score = (magnitude + conviction) / 20   # 0 to 1
    return 3.0 + (quality_score * 2.0)              # 3% to 5%
```

---

## Mathematical Frameworks for Early Detection

> These are the mathematical tools that see what price charts cannot.

### 1. Principal Component Analysis on Macro Regime

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Build macro feature matrix (monthly observations, years of history)
macro_features = pd.DataFrame({
    'yield_curve':      yield_10y - yield_2y,
    'credit_spread':    hy_spread,
    'dollar_strength':  dxy_mom_3m,
    'vix_level':        vix,
    'oil_mom':          oil.pct_change(60),
    'copper_gold_ratio': copper / gold,
    'real_rates':       tips_10yr,
    'm2_growth':        m2.pct_change(252),
    'ism_composite':    (ism_mfg + ism_svc) / 2,
    'consumer_conf':    umich_sentiment,
})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(macro_features.dropna())
pca = PCA(n_components=3)
components = pca.fit_transform(X_scaled)

# PC1 typically = "risk-on / risk-off" axis
# PC2 typically = "inflation vs deflation" axis
# PC3 typically = "growth vs recession" axis

# ANOMALY DETECTION:
# Current reading outside 2-sigma of historical distribution on any PC
# = regime is in unusual territory = imbalance may be building

from scipy import stats
pc_zscores = stats.zscore(components)
anomaly = (abs(pc_zscores) > 2).any(axis=1)
# Periods of anomaly = elevated probability of major regime shift
```

### 2. The Equity Risk Premium Model (ERP)

> The single most mathematically rigorous measure of whether equities are over or under priced right now. Used by Damodaran, Buffett, and every serious macro fund.

```
ERP = Earnings Yield - Real Risk-Free Rate

Earnings Yield  = (Trailing 12M EPS of S&P 500) / (Current S&P 500 Level)
                = 1 / P/E ratio
Real Risk-Free   = 10yr Treasury Yield - 10yr Breakeven Inflation (TIPS spread)

Current ERP interpretation:
  ERP > 5%  → Equities historically cheap vs bonds → strong expected 10yr return
  ERP 3-5%  → Fair value range → normal expected returns
  ERP 1-3%  → Expensive → poor expected 10yr returns
  ERP < 1%  → Extreme overvaluation → historical precursor to major bear markets
  ERP < 0%  → BONDS OFFER BETTER RISK-ADJUSTED RETURN THAN STOCKS
               This is a structural fault signal
               (This occurred in 2021-2022; S&P then fell 25%)

RATE OF CHANGE matters more than level:
  ERP falling at >0.5% per quarter = compression signal
  3 consecutive quarters of compression = fault building
```

```python
import pandas as pd
from fredapi import Fred

fred = Fred(api_key='YOUR_KEY')

sp500_pe    = fred.get_series('MULTPL/SHILLER_PE_RATIO_MONTH')  # Shiller CAPE
treasury_10 = fred.get_series('DGS10')
tips_10     = fred.get_series('DFII10')   # 10yr TIPS = real rate
breakeven   = treasury_10 - tips_10

earnings_yield = 100 / sp500_pe          # percent
erp = earnings_yield - tips_10

erp_zscore = (erp - erp.rolling(120).mean()) / erp.rolling(120).std()
# Z-score < -1.5 = historically poor forward returns
# Z-score > +1.5 = historically excellent entry points
```

### 3. The Shiller CAPE + Mean Reversion Mathematics

```python
# Cyclically Adjusted P/E Ratio (10yr inflation-adjusted earnings average)
# The most predictive valuation metric for 10yr forward returns
# R² of ~0.6 for 10yr return prediction (strong for finance)

cape = sp500_pe_shiller   # from FRED: MULTPL/SHILLER_PE_RATIO_MONTH

# Implied 10yr annual return from CAPE (empirical regression):
# E[10yr return] ≈ 1/CAPE + inflation_expectation - premium_for_risk
implied_10yr_return = (1/cape) * 100  # rough approximation

# Z-score vs 130 years of history:
cape_zscore = (cape - cape.mean()) / cape.std()
# CAPE Z > 1.5 → historically followed by negative 10yr real returns
# CAPE Z < -1.5 → historically best 10yr entry points

# VELOCITY: How fast is CAPE changing?
cape_velocity = cape.diff(12)  # annual change in CAPE
# CAPE rising fast = multiple expansion = momentum
# CAPE falling fast = multiple compression = value emerging

# ALERT LEVEL:
if cape_zscore > 2.0 and cape_velocity > 5:
    print("ALERT: Extreme valuation + accelerating expansion = FAULT TYPE 1")
```

### 4. Dynamic Factor Divergence (What XGBoost Should Flag)

```python
# Factor momentum divergence — when style factors DISAGREE, regime change is near

# Calculate rolling 20-day momentum of key factors
factors = {
    'momentum': SMB_return,      # 12-1 month momentum factor
    'value':    HML_return,      # high vs low book-to-market
    'quality':  RMW_return,      # profitable vs unprofitable
    'low_vol':  BAB_return,      # betting against beta
    'size':     SMB_return,      # small vs large cap
}

factor_df = pd.DataFrame(factors)
factor_corr = factor_df.rolling(60).corr()

# Factor correlation SPIKING (all factors moving together) = crisis signal
avg_pairwise_corr = factor_corr.groupby(level=0).mean().mean(axis=1)
corr_zscore = (avg_pairwise_corr - avg_pairwise_corr.mean()) / avg_pairwise_corr.std()

# When factor correlations spike above 2 std → correlation crisis
# All diversification fails → cascading liquidation risk

# Alternatively: MOMENTUM vs VALUE diverging widely
mom_vs_val_spread = SMB_12m_return - HML_12m_return
spread_zscore = (mom_vs_val_spread - mom_vs_val_spread.rolling(252).mean()) / \
                 mom_vs_val_spread.rolling(252).std()
# Extreme positive spread (momentum crushing value) = late cycle signal
# Reversal of this spread is violent and fast
```

---

## Structural Imbalance Indicators

> These are the specific, mathematically computable signals that detect structural faults. Each one has a threshold for "fault territory."

### The Imbalance Dashboard (Track These Weekly)

| Indicator | Normal Range | Fault Territory | Direction Signal | Source |
|---|---|---|---|---|
| ERP (Equity Risk Premium) | 3–5% | <1% = BEARISH | Falling = markets overvalued | FRED + Shiller |
| Shiller CAPE Z-Score | -1 to +1 | >2.0 = BEARISH | Rising fast = bubble phase | FRED |
| COT Commercial Z-Score | -1 to +1 | <-2 or >+2 | Extreme = major turn | CFTC |
| Credit Spread Velocity | -5 to +5 bps/wk | >10 bps/wk rising = BEARISH | Rising = risk off | FRED |
| TED Spread | 0.1–0.3% | >0.5% = STRESS | Rising = bank stress | FRED |
| M2 YoY Growth | 4–8% | <0% = BEARISH | Falling = liquidity drain | FRED |
| Yield Curve (10y-2y) | 0–200 bps | <0 = WARNING | Inverting = recession in 6-18mo | FRED |
| Margin Debt YoY | -10% to +20% | <-20% = CRASH RISK | Falling = deleveraging | FINRA |
| Factor Correlation Spike | <0.3 | >0.6 = CRISIS | Spiking = correlation crisis | Calculated |
| Put/Call Vol Skew Z | -1 to +1 | <-2 = CRASH FEAR | Extreme neg = panic buying puts | CBOE |
| Buffett Indicator | 80–120% | >150% = BEARISH | >180% = extreme overvaluation | Wilshire/FRED |
| ISM Manufacturing | 48–58 | <45 = CONTRACTION | Consecutive months <50 = recession signal | ISM |

```python
# Weekly imbalance dashboard updater
def compute_imbalance_dashboard():
    dashboard = {}

    # ERP
    erp = compute_erp()
    dashboard['erp'] = {
        'value': erp.iloc[-1],
        'fault': erp.iloc[-1] < 1.0,
        'zscore': erp_zscore.iloc[-1]
    }

    # CAPE
    cape = fred.get_series('MULTPL/SHILLER_PE_RATIO_MONTH')
    cape_z = (cape - cape.mean()) / cape.std()
    dashboard['cape'] = {
        'value': cape.iloc[-1],
        'fault': cape_z.iloc[-1] > 2.0,
        'zscore': cape_z.iloc[-1]
    }

    # Count active faults
    active_faults = sum(1 for k, v in dashboard.items() if v['fault'])
    dashboard['fault_count'] = active_faults
    dashboard['petroulas_alert'] = active_faults >= 3  # 3+ simultaneous faults = major signal

    return dashboard
```

---

## The Mathematics of Divergence

> Divergence is when two things that should track each other stop tracking. The gap is the stored energy. When it closes, price moves violently.

### Type 1 — Price vs Fundamentals Divergence

```
Price diverges from earnings/revenue/cash flow = valuation gap
This gap closes one of two ways:
  a) Fundamentals grow to match price (price was right; analysts were wrong)
  b) Price falls to match fundamentals (price was wrong; math was right)

Mathematics: implied growth rate at current price
  PEG ratio: P/E / Earnings Growth Rate
  PEG > 2 = paying too much for growth; growth must double to justify price
  PEG < 1 = growth not priced in; potentially undervalued (Petroulas Google logic)

For tech/growth stocks:
  DCF with current P/S: what revenue growth rate justifies this valuation?
  Rule: IF required growth rate > 50% per year for 5 years → physically improbable
        → Price MUST fall OR multiple must compress
        → Short-side fault
```

### Type 2 — Cross-Asset Divergence (Your Multi-Asset Engine)

```
Copper vs Equities:
  Copper leads equities by 4-6 weeks (Dr. Copper = global growth gauge)
  IF copper falling for 3 months AND equities still at highs:
    → DIVERGENCE = equities must follow copper down
    → This is a TRADEABLE structural fault
    → Copper's fundamentals are more physical/real than stock narratives

High Yield Bonds vs Equities:
  HYG (high yield ETF) leads S&P by 2-4 weeks at major turns
  IF HYG making lower highs while S&P making higher highs:
    → Credit is deteriorating but equity investors are ignoring it
    → Credit is smarter money; it will be right
    → This divergence ALWAYS closes; always in the direction credit predicts

Semiconductor vs Tech:
  Semiconductors are the raw material of tech earnings
  IF SOX (Philly Semiconductor Index) declining AND QQQ still rising:
    → Tech earnings WILL be revised down; semis see revenue earlier
    → Gap will close; QQQ will follow SOX

Treasury Bonds vs Rate-Sensitive Equities:
  Real estate, utilities, preferred shares all trade like bonds
  IF 10yr yield rising fast AND rate-sensitive equities not falling yet:
    → Math says they MUST fall; income competitors are now more attractive
    → Time to market: 4-8 weeks for rates to be fully priced into equities
```

```python
def detect_cross_asset_divergence(asset_a, asset_b, lookback=60, threshold=2.0):
    """
    Detect when two historically correlated assets diverge significantly.
    Returns divergence z-score; above threshold = fault building.
    """
    # Calculate rolling correlation
    corr = asset_a.rolling(lookback).corr(asset_b)

    # Z-score the spread between their normalized returns
    ret_a = asset_a.pct_change(lookback)
    ret_b = asset_b.pct_change(lookback)

    spread = ret_a - ret_b
    spread_zscore = (spread - spread.rolling(252).mean()) / spread.rolling(252).std()

    # Divergence = historically correlated assets moving in opposite directions
    current_corr = corr.iloc[-1]
    current_spread_z = spread_zscore.iloc[-1]

    if current_corr > 0.5 and abs(current_spread_z) > threshold:
        direction = 'A_OUTPERFORMING' if current_spread_z > 0 else 'B_OUTPERFORMING'
        return {
            'divergence': True,
            'spread_zscore': current_spread_z,
            'historical_correlation': current_corr,
            'direction': direction,
            'implication': f"Gap likely closes; {direction} expected to reverse"
        }
    return {'divergence': False}
```

---

## Regime Detection Before Transition

> The most valuable prediction is not "which direction" but "are we about to change regimes." Regime changes cause the largest moves.

### Hidden Markov Model for Regime Detection

```python
from hmmlearn.hmm import GaussianHMM
import numpy as np

# Fit HMM to market returns to identify hidden regimes
returns = sp500_returns.values.reshape(-1, 1)

model = GaussianHMM(
    n_components=3,   # 3 regimes: bull / neutral / bear/crisis
    covariance_type='full',
    n_iter=100
)
model.fit(returns)

# Decode current regime
hidden_states = model.predict(returns)
current_regime = hidden_states[-1]

# Transition probability matrix
# P(next_regime | current_regime) = model.transmat_
# IF P(transition to crisis | currently bull) has jumped from 5% to 20%:
#   → System is detecting increased probability of regime change
#   → This is a weeks-ahead signal; price has not moved yet

# Regime characteristics
for i in range(3):
    regime_returns = returns[hidden_states == i]
    print(f"Regime {i}: mean={regime_returns.mean():.4f}, "
          f"vol={regime_returns.std():.4f}, count={len(regime_returns)}")

# Practical use:
# Regime 0: High return, low vol = Bull
# Regime 1: Low return, medium vol = Neutral/Transition
# Regime 2: Negative return, high vol = Bear/Crisis

# FAULT SIGNAL: Currently in Regime 0 BUT:
#   Transition probability to Regime 2 has increased by >10% in last 4 weeks
#   → Regime shift probability is building even though price still rising
```

### The Yield Curve as a Probabilistic Recession Model

```python
# Federal Reserve of New York official model
# Probability of recession in 12 months based on 10yr-3mo spread

import numpy as np
from scipy.stats import norm

def ny_fed_recession_probability(spread_10y_3m):
    """
    NY Fed's probit model.
    Spread = 10yr yield minus 3-month yield (in percent)
    Returns probability of recession in next 12 months.
    """
    # Probit coefficients from NY Fed paper (Estrella & Mishkin 1996)
    alpha = -0.6103
    beta  = -0.5582
    z = alpha + beta * spread_10y_3m
    return norm.cdf(z)

# Historical reliability:
# Probability > 30% has preceded every recession since 1960
# False positive rate: very low (~1 per decade)
# Lead time: 6-18 months before recession begins

# TRADING APPLICATION:
# Recession probability rising through 20% → reduce equity exposure
# Rising through 30% → significant short bias (Petroulas level signal)
# Falling from above 30% → recovery trade beginning (Petroulas long signal)
```

---

## Positioning Imbalance Mathematics

> When everyone is positioned the same way, the position IS the fault. The unwind IS the move.

### COT-Based Trading Model

```python
import pandas as pd
import requests

def fetch_cot_data():
    """Fetch CFTC COT data — released every Friday."""
    # https://www.cftc.gov/dea/futures/deahistfo.htm
    url = "https://www.cftc.gov/files/dea/history/fut_fin_txt_2024.zip"
    # Parse CSV; key columns: market_name, commercial_long, commercial_short
    pass

def compute_cot_signal(commercial_long, commercial_short, lookback=156):
    """
    Commercial net positioning z-score.
    Lookback = 3 years of weekly data = 156 observations.
    """
    net_position = commercial_long - commercial_short
    rolling_mean = net_position.rolling(lookback).mean()
    rolling_std  = net_position.rolling(lookback).std()
    z_score = (net_position - rolling_mean) / rolling_std

    # Signal interpretation:
    # Z > +1.5 → Commercials historically bullish → BUY SIGNAL
    # Z < -1.5 → Commercials historically bearish → SELL SIGNAL
    # Z > +2.0 → STRONG BUY (once per year frequency)
    # Z < -2.0 → STRONG SELL (once per year frequency)

    if z_score.iloc[-1] > 2.0:   return ('STRONG_BUY',  z_score.iloc[-1])
    if z_score.iloc[-1] > 1.5:   return ('BUY',         z_score.iloc[-1])
    if z_score.iloc[-1] < -2.0:  return ('STRONG_SELL', z_score.iloc[-1])
    if z_score.iloc[-1] < -1.5:  return ('SELL',        z_score.iloc[-1])
    return ('NEUTRAL', z_score.iloc[-1])
```

### Hedge Fund Crowding Model (from 13F Data)

```python
def compute_crowding_score(positions_df, top_n=50):
    """
    Measure how crowded a stock is among top hedge funds.
    High crowding = position is consensus = vulnerable to unwind.
    """
    # Count how many of top 50 funds hold this stock
    holder_count = positions_df.groupby('ticker')['fund_name'].nunique()

    # Concentration = avg % of fund portfolio
    avg_concentration = positions_df.groupby('ticker')['pct_of_portfolio'].mean()

    # Recent additions = momentum of fund interest
    qoq_change = positions_df.groupby('ticker')['shares'].pct_change()

    crowding_score = (
        0.4 * rank_normalize(holder_count) +
        0.3 * rank_normalize(avg_concentration) +
        0.3 * rank_normalize(qoq_change)
    )

    # Crowding > 0.8 = extremely crowded = Fault Type 2 territory
    # Crowding < 0.2 = undiscovered = potential before 13F buying wave begins
    return crowding_score

# DUAL USE:
# HIGH CROWDING + KIMI detects narrative breaking:
#   → Short setup: everyone owns it, no one left to buy, catalyst breaks narrative
# LOW CROWDING + KIMI detects underpriced asset:
#   → Long setup: funds will discover this over next 2-3 quarters
#   → This is Petroulas-mode: you're in before the institution wave
```

---

## Volatility Surface Fault Lines

> The options market prices risk. When the volatility surface becomes distorted, it is pricing a fault that the underlying price has not yet acknowledged.

### VIX Term Structure as Fear Detector

```python
# VIX = 30-day implied vol
# VIX3M = 93-day implied vol
# Contango (VIX < VIX3M) = normal; calm market
# Backwardation (VIX > VIX3M) = near-term fear > long-term fear = crisis

vix_ratio = vix / vix_3m

# Signal thresholds:
# ratio < 0.85: Extreme complacency (contango steep) → volatility likely to spike
# ratio 0.85-1.0: Normal
# ratio 1.0-1.15: Mild stress
# ratio > 1.15: Backwardation = active fear = hedging happening
# ratio > 1.30: Extreme backwardation = crisis conditions

# FAULT SIGNAL:
# VIX ratio has been in 0.85 range for >60 days
# AND equity prices are at new highs
# = Market is pricing no risk while being at all-time highs
# = Volatility is EXTREMELY cheap to buy
# = Structural setup for long volatility / protective puts

# The math: expected volatility return
# Long VIX when ratio < 0.85 has historically returned +40% over next 3 months
# on average — with very high variance
```

### Skew as a Positioning Detector

```python
# 25-delta skew = IV(25d put) - IV(25d call)
# Normal: skew is slightly negative (puts more expensive = crash protection premium)
# Extreme negative skew: markets are paying HIGH premium for crash protection
#   = someone with large positions is VERY worried
#   = they know something, or they're very large

# Skew Z-score:
skew_zscore = (current_skew - skew.rolling(252).mean()) / skew.rolling(252).std()

# Trading:
# Skew Z < -2 = extreme crash fear = contrarian BUY signal (usually)
#   Exception: if fundamentals support the fear → crash protection is CHEAP
#   This is the Petroulas detection: options market pricing a fault 
#   before price moves, but at extreme skew, the hedge is expensive
#   meaning smart money is paying up for protection → something is wrong

# Skew Z > +1.5 = complacency (no one buying puts) = SELL signal
#   Markets at highs + nobody hedging = setup for sharp drop
```

---

## Cross-Asset Stress Propagation

> When a fault erupts, where does the stress travel first? This model tells you which assets to short first and which to long as the contagion spreads.

### The Propagation Sequence (Historical Pattern)

```
FAULT ERUPTION SEQUENCE (typically plays out over 2-8 weeks):

WEEK 1 — First Cracks:
  Credit spreads widen (high yield first, then investment grade)
  Small-cap equities weaken (Russell 2000 leads large cap down)
  Emerging market currencies weaken
  VIX rises from below 15 → above 20

WEEK 2-3 — Contagion:
  Large-cap equities begin falling
  Safe havens bid up (Treasury bonds rally, gold rallies, JPY strengthens)
  Commodity currencies weaken (AUD, CAD)
  Cyclical sectors lead decline (Financials, Materials, Energy)

WEEK 3-5 — Cascade:
  Momentum/growth stocks accelerate lower
  Dollar strengthens (USD safe haven bid)
  Investment grade credit starts to widen
  Volatility products spike

WEEK 5-8 — Capitulation or Stabilization:
  VIX peaks (often 30-50 in moderate events, 80+ in crises)
  Put/call ratio spikes
  Margin calls force selling of anything liquid
  IF bottoming: Credit leads equity recovery

TRADE SEQUENCE GIVEN FAULT DETECTION:
  Day 0 (fault confirmed): Short HYG or buy HYG puts
  Day 3-7: Short Russell 2000 (IWM)
  Day 7-14: Short S&P sector ETF of affected sector
  Day 14-21: Short S&P 500 (SPY) if propagation confirmed
  Day 21+: Cover shorts; buy deep value/defensive as capitulation approaches
```

---

## The Conviction Sizing Model

> When the dual confirmation fires — the Petroulas mode — sizing must reflect the quality of the thesis. Here is the complete mathematical model.

### Conviction-Scaled Kelly Fraction

```python
def petroulas_position_size(
    account_balance,
    fault_magnitude,     # 7-10 (only called in Petroulas mode)
    conviction_score,    # 7-10
    xgb_confidence,      # 0.60-1.0
    win_rate_history,    # empirical win rate of Petroulas trades (build this over time)
    avg_win_r,           # average R-multiple when Petroulas mode is right
    avg_loss_r=1.0       # you lose 1R when wrong
):
    """
    Full Kelly fraction, then apply safety discount.
    """
    # Blended confidence: combine all three signals
    p_win = (
        0.40 * (conviction_score / 10) +
        0.35 * xgb_confidence +
        0.25 * win_rate_history
    )
    p_loss = 1 - p_win

    # Kelly fraction
    b = avg_win_r  # net profit per unit risked
    kelly = (b * p_win - p_loss) / b

    # Quality-adjusted fraction (higher magnitude = more confidence in sizing)
    quality = (fault_magnitude + conviction_score) / 20  # 0 to 1
    adjusted_kelly = kelly * quality

    # Cap at 5% for safety (fractional Kelly)
    # Use 50% of full adjusted Kelly (half-Kelly principle)
    safe_fraction = min(adjusted_kelly * 0.5, 0.05)

    dollar_risk = account_balance * safe_fraction
    return dollar_risk, safe_fraction * 100

# Example: 10/10 fault, XGBoost 70% confidence, historical 60% win rate on Petroulas trades
# avg_win = 5R (these are big moves when right)
# p_win = 0.4*(10/10) + 0.35*0.70 + 0.25*0.60 = 0.40 + 0.245 + 0.15 = 0.795
# kelly = (5 * 0.795 - 0.205) / 5 = (3.975 - 0.205) / 5 = 0.754
# quality = (10+10)/20 = 1.0
# adjusted = 0.754 * 1.0 = 0.754
# safe = min(0.754 * 0.5, 0.05) = min(0.377, 0.05) = 0.05 → 5% of account
```

### The Patience Rule (Petroulas Mode Only)

```
Normal swing trade: 2-10 days
Petroulas mode: 3-18 weeks

RULES that DIFFER from normal ICT Engine in Petroulas mode:

1. TIME STOP is extended to 12 weeks (not 5 days)
   The thesis is structural; it takes time to play out

2. DRAWDOWN TOLERANCE is wider: allow 1.5× ATR drawdown (not 1× ATR)
   Structural faults cause choppy path to resolution

3. ADD TO POSITION if fault STRENGTHENS (new confirming data):
   IF 4 weeks in AND a new piece of evidence confirms the thesis:
     → Add up to 50% more to position
     → Requires: no stop hit, thesis still intact, XGBoost still aligned

4. DO NOT EXIT on consensus analyst upgrades/downgrades:
   If 15 analysts downgrade a stock you're LONG because they don't see the fault:
   → THIS IS BULLISH. They were already not positioned.
   → More potential buyers when they reverse.

5. FALSIFICATION DISCIPLINE:
   At entry, you defined a falsification test (from Kimi prompt).
   CHECK IT WEEKLY.
   IF falsification test triggers → EXIT IMMEDIATELY regardless of P&L.
   The thesis is wrong. The fault does not exist as described.
   Take the loss. Small losses on wrong Petroulas theses protect capital for correct ones.
```

---

## Historical Imbalance Case Studies

> Every major move was predictable with the math. These are your training examples.

### Case 1 — The 2021-2022 Rate Shock (Fault Type 1 + 3)

```
FAULT: Bonds and growth stocks priced for zero rates forever.
MATH SIGNAL:
  - ERP fell below 1% by Q3 2021 (first time since 2007)
  - CAPE Z-score reached 2.3 (only higher in 1999-2000 bubble)
  - Real rates at -1.1% (historically negative → equities always overcorrected)
  - M2 growth at +27% YoY (highest since WWII) → inflation inevitable
  - COT data: Large speculators record short in TBond futures (wrong positioning)

CONSENSUS VIEW IN 2021: "Inflation is transitory. Fed won't hike."
MATHEMATICAL REALITY: With M2 at +27%, you needed only a freshman economics
  equation: MV = PQ. M doubled. Q hadn't grown 27%. P (inflation) MUST rise.
  This was not an opinion. It was arithmetic.

RESULT: Bonds fell 30%, NASDAQ fell 35%, growth stocks fell 50-80%.
LEAD TIME: Math was conclusive by June 2021. Move began November 2021. 5 months.
```

### Case 2 — The 2023 Regional Bank Crisis (Fault Type 4)

```
FAULT: Regional banks held long-duration bonds purchased at 0% rates.
MATH SIGNAL:
  - Duration mismatch: 10yr bond holdings, 1yr deposits (liquidity mismatch)
  - Mark-to-market losses on held-to-maturity portfolios: calculable from public filings
  - FDIC H.8 data showed unrealized losses building throughout 2022
  - SVB 10-K (filed Feb 2023): $91B in long-duration securities, tangible equity of $16B
    → Simple math: if rates +200bp → mark-to-market loss exceeds equity → insolvent

CONSENSUS VIEW: "Banks are well capitalized. Stress tests show no issues."
MATHEMATICAL REALITY: The stress tests didn't model 500bp rate hike.
  Duration × Rate Change × Portfolio Size = Loss
  10yr duration × 5% × $91B = $45.5B loss on $16B equity = insolvent

RESULT: SVB failed March 2023. KRE (Regional Bank ETF) fell 45%.
LEAD TIME: Math was visible from 10-K filing. Fault existed 12 months before crisis.
```

### Case 3 — The Petroulas Pattern (Fault Type 3, Bullish)

```
FAULT: Market pricing an asset as a declining business when mathematics
  shows structural moats are deepening.

GOOGLE (ALPHABET) SPECIFIC:
  - Search monopoly: 91% global search market share
  - YouTube: 2nd largest search engine on earth
  - Cloud: Growing 28% YoY
  - AI integration: Gemini embedded in all products
  - PE at historic lows despite record FCF margins

MATH SIGNAL:
  - FCF yield > 5% at a time when 10yr was 4.5%
    → Growth stock paying more than equivalent-risk fixed income
    → Either bonds are wrong OR Google is wrong
    → Google's business was growing; not declining
  - Insider buying (Form 4): Sundar Pichai personally bought at these levels
  - 13F: Multiple top funds adding aggressively while retail sold

CONSENSUS VIEW: "AI will destroy Google search. OpenAI will win."
MATHEMATICAL REALITY: Google processed 8.5 billion searches/day.
  Even if ChatGPT replaced 10% of searches → Google still processes 7.65B/day.
  At $0.003 revenue per search → still dominant FCF generator.
  And Google HAD the AI models (Gemini, DeepMind).
  The narrative was wrong; the math was clear.

RESULT: GOOGL +87% in 18 months from that thesis.
```

---

## Integration with Existing Pipeline

### Where These Models Plug Into Your System

```python
# orchestrator/backtest_lifecycle.py — add these hooks:

class ImbalanceLayer:
    """
    Runs weekly alongside the bias engine.
    Outputs fault signals to Obsidian vault.
    """

    def __init__(self):
        self.dashboard = ImbalanceDashboard()
        self.divergence = CrossAssetDivergence()
        self.regime_hmm = RegimeHMM()
        self.cot_model  = COTSignalModel()
        self.kimi_client = KimiClient()

    def weekly_scan(self) -> dict:
        """Run every Sunday after market close."""

        dashboard = self.dashboard.compute()
        divergences = self.divergence.scan_all_pairs()
        regime_prob = self.regime_hmm.transition_probabilities()
        cot_signals = self.cot_model.compute_all_commodities()

        # High-priority alerts
        alerts = []
        if dashboard['fault_count'] >= 3:
            alerts.append(('MACRO_FAULT', dashboard))
        for d in divergences:
            if d['spread_zscore'] > 2.5:
                alerts.append(('DIVERGENCE_FAULT', d))
        if regime_prob['to_crisis'] > 0.15:
            alerts.append(('REGIME_TRANSITION', regime_prob))

        # For each alert: escalate to Kimi for fault thesis
        kimi_results = []
        for alert_type, alert_data in alerts:
            kimi_result = self.kimi_client.analyze_fault(
                alert_type, alert_data, KIMI_FAULT_PROMPT
            )
            kimi_results.append(kimi_result)

        # Write to Obsidian vault
        self.write_fault_report(alerts, kimi_results)

        return {
            'alerts': alerts,
            'kimi_faults': kimi_results,
            'petroulas_signals': [
                r for r in kimi_results if r['petroulas_worthy']
            ]
        }

    def write_fault_report(self, alerts, kimi_results):
        """Auto-generate Obsidian daily brief section."""
        petroulas_count = sum(1 for r in kimi_results if r['petroulas_worthy'])
        report = f"""
## Imbalance Engine Report — {datetime.now().strftime('%Y-%m-%d')}

### Active Faults: {len(alerts)}
### Petroulas Signals: {petroulas_count}

{'🚨 DUAL CONFIRMATION ACTIVE' if petroulas_count > 0 else '✅ No Petroulas signal this week'}

"""
        for r in kimi_results:
            if r['petroulas_worthy']:
                report += f"""
#### ⚡ FAULT CONFIRMED: {r['asset']}
- Type: {r['fault_type']} — Magnitude: {r['magnitude']}/10
- Direction: {r['direction']}
- Conviction: {r['conviction_score']}/10
- Horizon: {r['timing_weeks']} weeks
- Consensus Blindspot: {r['consensus_blindspot']}
- Falsification Test: {r['falsification_test']}
- Risk: {calculate_petroulas_risk(r['magnitude'], r['conviction_score'])}%
"""

        with open('vault/analysis/imbalance_report.md', 'w') as f:
            f.write(report)
```

---

## The Full Detection Checklist

> Run this weekly. When score ≥ 7, feed to Kimi for fault thesis generation.

```
QUANTITATIVE FAULT DETECTION SCAN
Week of: [DATE]

VALUATION:
  [ ] ERP below 2% ..................... (score +2 if yes)
  [ ] CAPE Z-score above 1.5 ........... (score +1 if yes)
  [ ] Buffett Indicator above 160% ...... (score +1 if yes)

POSITIONING:
  [ ] COT commercial Z-score extreme .... (score +2 if |Z| > 2)
  [ ] Hedge fund crowding > 80th pctile . (score +1 if yes)
  [ ] Short interest at 3yr extreme ..... (score +1 if yes)

MACRO STRESS:
  [ ] Yield curve inverted > 3 months ... (score +2 if yes)
  [ ] Credit spreads widening > 10 wks .. (score +2 if yes)
  [ ] M2 growth below 2% YoY ............ (score +1 if yes)
  [ ] TED spread above 0.4% ............. (score +1 if yes)

CROSS-ASSET DIVERGENCE:
  [ ] Copper/Equity divergence Z > 2 .... (score +2 if yes)
  [ ] HYG/SPY divergence Z > 2 .......... (score +2 if yes)
  [ ] VIX term structure backwardation ... (score +1 if yes)

REGIME:
  [ ] HMM transition prob to crisis >15% . (score +2 if yes)
  [ ] NY Fed recession prob above 25% ... (score +2 if yes)
  [ ] Factor correlation spike Z > 1.5 .. (score +1 if yes)

TOTAL SCORE: ___ / 24

THRESHOLD:
  Score 0-4:   No fault. Normal market. ICT engine only.
  Score 5-8:   Elevated risk. Reduce size 25%. Monitor weekly.
  Score 9-14:  Fault building. Feed to Kimi. Paper trade thesis.
  Score 15-20: Major fault likely. Kimi thesis required. XGBoost confirm. Petroulas mode.
  Score 21+:   Systemic fault possible. Maximum position. Kimi fault prompt mandatory.
```

---

## Related Notes

- [[ICT Swing Trade Decision Engine]]
- [[Intelligence System MOC]]
- [[Quantitative Finance MOC]]
- [[Swing Trading MOC]]
- [[Son of Anton Architecture]]
- [[XGBoost Bias Engine]]
- [[Kimi Integration]]
- [[COT Analysis Weekly]]
- [[ERP Dashboard]]
- [[Imbalance Report Auto-Generated]]
- [[Petroulas Trade Log]]
- [[Fault Falsification Tracker]]

---

*The crowd is almost always right about direction and almost always wrong about magnitude and timing. The fault detector finds where the crowd is not just wrong — but mathematically impossible. That is where you size up.*
