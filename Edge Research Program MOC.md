# The Edge Research Program 🔬

#research #scientific-method #edge #imbalance #experimental #classical #iteration

> **The Mission:** Find a mathematical and scientific edge in detecting market imbalances better than anyone alive. Start with what is proven to work across decades. Layer experimental methods on top. Treat every failure as data. Iterate until the edge is undeniable. There is no deadline — only direction.

---

## 🗺️ Map of Contents

- [[#The Research Philosophy]]
- [[#Tier 1 — The Classical Bedrock (Proven Over Decades)]]
- [[#Tier 2 — The Robust Extensions (Proven But Underused)]]
- [[#Tier 3 — The Experimental Frontier (Novel — Unknown If It Works)]]
- [[#The Scientific Method Applied to Trading Research]]
- [[#The Hypothesis Registry]]
- [[#The Experiment Log Structure]]
- [[#Failure Analysis Protocol]]
- [[#The Iteration Engine]]
- [[#Edge Validation Criteria]]
- [[#The Research Roadmap]]
- [[#Integration Checkpoints]]

---

## The Research Philosophy

```
THREE TIERS. THREE PURPOSES.

TIER 1 — CLASSICAL BEDROCK
  What: Methods that have produced documented edge across 50-100+ years
  Why: These are your anchor. When Tier 2 and 3 fail, this keeps you alive.
  Rule: Never abandon Tier 1. It is not exciting. That is the point.
  Examples: Yield curve, COT positioning, earnings momentum, value mean reversion

TIER 2 — ROBUST EXTENSIONS
  What: Methods proven to work but underused by the majority of practitioners
  Why: The edge here is real but smaller; requires precision to extract
  Rule: Validate on your own data before deploying capital. Never assume.
  Examples: Options flow as leading indicator, cross-asset divergence, insider clustering

TIER 3 — EXPERIMENTAL FRONTIER
  What: Novel hypotheses — untested or minimally tested combinations
  Why: If ONE of these works, it becomes a durable, non-crowded edge
  Rule: Scientific method only. No hope. No narrative. Only data.
  Capital allocation: Never more than 10% of research time AND 0% of live capital
  until validated through Tier 1 confirmation + out-of-sample test

THE RELATIONSHIP:
  Tier 1 keeps you profitable while Tier 2 and 3 are being developed.
  Tier 2 amplifies Tier 1 signals with higher precision.
  Tier 3 is the lab. Most experiments fail. That is expected and fine.
  One successful Tier 3 experiment, fully validated, becomes Tier 2.
  A Tier 2 method proven across multiple market regimes becomes Tier 1.
  This is how the edge compounds over years.
```

---

## Tier 1 — The Classical Bedrock (Proven Over Decades)

> These methods have been documented by serious researchers across 50-100+ years of market history. They work. They are boring. They are the foundation.

### T1-A: The Yield Curve (Recession Predictor — 7/7 Since 1965)

**What it is:** The spread between 10-year and 2-year Treasury yields.
**Track record:** Predicted every US recession since 1960 with zero false positives at the right threshold.
**Lead time:** 6–18 months before recession begins; 3–12 months before equity peak.

```python
# Implementation — dead simple, extraordinarily powerful
from fredapi import Fred
fred = Fred(api_key='YOUR_KEY')

t10 = fred.get_series('DGS10')   # 10yr Treasury
t2  = fred.get_series('DGS2')    # 2yr Treasury
spread = t10 - t2

# Signal rules (proven, not experimental):
# Spread < 0 (inverted) → watch for recession in 6-18 months
# Spread < 0 for 3+ consecutive months → high confidence signal
# Spread re-steepening FROM negative → recession is starting NOW (not a relief signal)
#   The steepening-from-inversion is when you short equities hardest

# NY Fed probability model (Estrella & Mishkin, 1996 — 30 years proven):
from scipy.stats import norm
def recession_prob(spread_10y_3m):
    return norm.cdf(-0.6103 - 0.5582 * spread_10y_3m) * 100

# Rule: prob > 30% = meaningful short bias on equities
# Rule: prob > 50% = Petroulas-level signal
```

**Why it works:** Banks borrow short-term and lend long-term. Inverted curve = negative margin = banks stop lending = credit contraction = recession. This is structural, not cyclical. It will keep working.

**Current status in your system:** ✅ Running in `frameworks.py`. Add to XGBoost feature set immediately.

---

### T1-B: COT Commercial Positioning (Smart Money Tracker — 60+ Years)

**What it is:** CFTC Commitment of Traders — commercial hedger net positions.
**Track record:** Commercials (producers/consumers of the commodity) are consistently correct at extremes. Z-score > ±2 on 3-year rolling basis has marked major turns in commodities, currencies, and bonds for decades.
**Lead time:** 4–12 weeks.

```python
# The two-line implementation of a 60-year edge
def cot_signal(net_commercial, lookback=156):
    z = (net_commercial - net_commercial.rolling(lookback).mean()) \
         / net_commercial.rolling(lookback).std()
    return z.iloc[-1]

# Z > +1.5 → long signal
# Z > +2.0 → strong long signal (Petroulas consideration)
# Z < -1.5 → short signal
# Z < -2.0 → strong short signal
# Frequency of Z > 2: ~2-4 times per year per market
```

**Why it works:** Commercials don't speculate. They use futures to lock in prices for actual business operations. When they're extremely long, it means they see value at current prices in their actual business. They are right far more than they are wrong at extremes.

**Current status:** ✅ COT framework in `falsification.py`. Need to connect to live CFTC feed.

---

### T1-C: Earnings Momentum + Post-Earnings Drift (PEAD — 50+ Years)

**What it is:** Stocks that beat earnings estimates continue drifting in the direction of the surprise for 30–90 days.
**Track record:** First documented by Ball & Brown (1968). Replicated in virtually every market studied. One of the most robust anomalies in academic finance.
**Lead time:** The drift begins immediately post-earnings. Entry is available 1–3 days after the announcement.

```python
# PEAD implementation
def earnings_surprise_score(actual_eps, consensus_eps, stock_price):
    # Standardized Unexpected Earnings (SUE)
    surprise_pct = (actual_eps - consensus_eps) / abs(consensus_eps)
    # Normalize by historical surprise std dev for this stock
    sue = surprise_pct / eps_surprise_std_history

    # SUE > +1: positive drift expected 30-60 days
    # SUE > +2: strong positive drift
    # SUE < -1: negative drift expected
    # SUE < -2: strong negative drift
    return sue

# Amplifiers (multiply confidence):
# + Revenue also beat → drift stronger
# + Guidance raised → drift extends 60-90 days
# + High short interest → potential short squeeze on top of drift
# + Stock in technical uptrend → trend + fundamental momentum
```

**Why it still works:** Institutional investors cannot fully reposition in a single day on a large holding. They accumulate/distribute over weeks. Analyst upgrades follow slowly. This creates a persistent, tradeable drift.

---

### T1-D: Value Mean Reversion (100+ Years — Graham, Buffett, Fama-French)

**What it is:** Cheap stocks (low P/B, P/E, P/S relative to history) outperform expensive stocks over 3–5 year horizons.
**Track record:** Fama-French HML factor. Documented across every major market globally. Works in US, Europe, Asia, Emerging Markets.
**Lead time:** 6 months to 3 years. Too slow for swing trading but critical for Petroulas-mode theses.

```python
# Value signal for Petroulas-mode (structural undervaluation)
def value_score(ticker_data, universe_data):
    # Cross-sectional percentile rank within universe
    pb_rank  = percentile_rank(ticker_data['pb_ratio'],  universe_data['pb_ratio'])
    pe_rank  = percentile_rank(ticker_data['pe_ratio'],  universe_data['pe_ratio'])
    ps_rank  = percentile_rank(ticker_data['ps_ratio'],  universe_data['ps_ratio'])
    fcf_rank = percentile_rank(ticker_data['fcf_yield'], universe_data['fcf_yield'])

    # Composite value score (0-100 percentile)
    value = (pb_rank + pe_rank + ps_rank + fcf_rank) / 4

    # Below 20th percentile = historically cheap = long candidate
    # Above 80th percentile = historically expensive = short candidate
    return 100 - value  # flip so higher = more undervalued
```

**Why it works:** Human psychology consistently overpays for growth and underpays for boring businesses. Mean reversion is mathematical inevitability when driven by fundamentals.

---

### T1-E: Price Momentum (50+ Years — Jegadeesh & Titman 1993)

**What it is:** Stocks that have performed well over the past 12 months (excluding the most recent month) continue to outperform for the next 3-12 months.
**Track record:** Documented in over 40 countries. Strongest in individual stocks. Works across asset classes.
**Lead time:** Signal is based on past 12 months; predicts next 3-12 months.

```python
def momentum_score(returns_series):
    # Standard momentum: 12-month return, skip most recent month
    mom_12_1 = returns_series.shift(21).rolling(252).sum()  # 12mo skip 1mo

    # Cross-sectional rank
    return mom_12_1.rank(pct=True)  # 0-1; higher = stronger momentum

# Momentum crash risk: momentum REVERSES in market crashes
# CRITICAL: When VIX > 30 OR yield curve re-steepening from inversion:
#           TURN OFF momentum strategy; go to value/defensive
```

**Why it works:** Underreaction to information. Institutional investors take months to fully price in good/bad news. The drift IS the underreaction being corrected.

---

### T1-F: Insider Buying as Leading Indicator (70+ Years)

**What it is:** Corporate insiders buying their own stock in the open market (Form 4 — buy-side only, not option exercise) predicts positive returns 3–12 months ahead.
**Track record:** Documented by Jaffe (1974), Seyhun (1986), and continuously replicated. Cluster buying (multiple insiders simultaneously) is the strongest version.

```python
def insider_signal(form4_data, ticker, lookback_days=90):
    recent = form4_data[
        (form4_data['ticker'] == ticker) &
        (form4_data['transaction_type'] == 'P') &  # Purchase only
        (form4_data['days_ago'] <= lookback_days)
    ]

    # Score components
    n_unique_insiders = recent['insider_name'].nunique()
    total_dollar_value = recent['transaction_value'].sum()
    is_ceo_cfo = recent['title'].str.contains('CEO|CFO').any()
    is_open_market = (recent['acquisition_code'] == 'P').all()  # Not option exercise

    score = 0
    if n_unique_insiders >= 3:     score += 3   # Cluster buying = strongest signal
    if total_dollar_value > 1e6:   score += 2   # Significant dollar amount
    if is_ceo_cfo:                 score += 2   # C-suite conviction
    if is_open_market:             score += 1   # Real money, not compensation

    # Score 6-8: Petroulas-worthy insider signal
    # Score 3-5: Meaningful but not extreme
    # Score 0-2: Weak; note but don't act
    return score
```

---

## Tier 2 — The Robust Extensions (Proven But Underused)

> These work. The evidence is solid. They are underused because they require more infrastructure to extract. That infrastructure is what you are building.

### T2-A: Options Market as Crystal Ball

The options market prices probability distributions. When sophisticated players (institutions, hedge funds) have strong directional conviction, it shows up in options flow BEFORE price moves.

```python
# 1. Put/Call Open Interest Ratio — institutional positioning proxy
pc_oi_ratio = put_open_interest / call_open_interest
# Extreme high (>1.5): Hedges being placed → downside expected
# Extreme low (<0.5): Calls being bought → upside expected

# 2. Unusual Options Activity — the strongest signal
# Large block trades far OTM that expire in 30-90 days
# Someone is betting on a specific move within a specific timeframe
# Filter: premium > $500K, volume > 10x average, OTM by 5%+

# 3. Volatility Skew (most underused)
# 25-delta put IV minus 25-delta call IV
# Steep negative skew = crash protection being aggressively purchased
# = someone large is very worried = fade the rally

# 4. Term Structure Anomaly
# If 30-day IV < 60-day IV < 90-day IV (normal contango):
#   market sees no near-term risk; complacency signal → short vol is crowded
# If 30-day IV > 90-day IV (backwardation):
#   near-term fear exceeds long-term fear → acute event expected
```

### T2-B: Credit Market Leads Equity (2–6 Week Lead)

```python
# High yield credit spreads lead equity markets at major turns
# HYG is the ETF proxy; actual spread = HYG_yield - treasury_10yr

import yfinance as yf
hyg = yf.download('HYG', start='2010-01-01')['Adj Close']
spy = yf.download('SPY', start='2010-01-01')['Adj Close']

# Cross-correlation to find optimal lead
from scipy.signal import correlate
import numpy as np

hyg_ret = hyg.pct_change().dropna()
spy_ret = spy.pct_change().dropna()

# Find lag where HYG best predicts SPY
xcorr = correlate(spy_ret, hyg_ret, mode='full')
lags = np.arange(-len(spy_ret)+1, len(spy_ret))
optimal_lag = lags[xcorr.argmax()]
# Historically: HYG leads SPY by 10-15 trading days at turns

# Implementation: if HYG making lower highs while SPY at highs → short SPY
# If HYG recovering while SPY still falling → long SPY
```

### T2-C: Sector Rotation Clock (Empirical Business Cycle)

```python
# Sam Stovall's GICS Sector Rotation Model (from Standard & Poor's)
# Different sectors lead/lag the business cycle by predictable amounts

SECTOR_CYCLE_MAP = {
    # Sector: (leads_economy_by_months, performs_best_in_phase)
    'XLK': ('Tech',        +6,  'recovery'),
    'XLY': ('Discretionary', +4, 'early_expansion'),
    'XLI': ('Industrials', +2,  'mid_expansion'),
    'XLB': ('Materials',   +1,  'late_expansion'),
    'XLE': ('Energy',       0,  'late_expansion'),
    'XLF': ('Financials',  -2,  'early_contraction'),
    'XLP': ('Staples',     -4,  'contraction'),
    'XLV': ('Healthcare',  -4,  'contraction'),
    'XLU': ('Utilities',   -6,  'recession'),
    'GLD': ('Gold',        -6,  'recession/stagflation'),
}

# Identify current cycle phase from macro indicators
# Then: overweight sectors that are about to enter their strong phase
# This is a 3-6 month forward-looking signal

def identify_cycle_phase(yield_curve, ism, unemployment_trend, credit_spreads):
    if yield_curve > 0 and ism > 52 and unemployment_trend < 0:
        return 'mid_expansion'
    elif yield_curve > 0 and ism < 50 and unemployment_trend > 0:
        return 'late_expansion'
    elif yield_curve < 0 and ism < 48:
        return 'early_contraction'
    elif yield_curve < -0.5 and unemployment_trend > 0.5:
        return 'recession'
    elif yield_curve.diff(4) > 0 and ism.diff(4) > 0:
        return 'recovery'
    return 'transition'
```

### T2-D: Smart Money Flow Index (Composite Institutional Signal)

```python
# Combine multiple institutional signals into single composite
def smart_money_composite(ticker, date):
    scores = {}

    # 13F net buying (last quarter)
    scores['institutional_13f'] = institutional_net_buying_score(ticker)

    # Form 4 insider signal
    scores['insider']           = insider_signal(form4_data, ticker)

    # COT positioning (for ETFs and futures)
    scores['cot']               = cot_signal_for_asset(ticker)

    # Options flow unusual activity
    scores['options_flow']      = unusual_options_score(ticker)

    # Short interest change
    scores['short_interest']    = short_interest_signal(ticker)

    # Dark pool volume ratio
    scores['dark_pool']         = dark_pool_buy_ratio(ticker)

    # Weighted composite
    weights = {'institutional_13f': 0.25, 'insider': 0.25, 'cot': 0.15,
               'options_flow': 0.20, 'short_interest': 0.10, 'dark_pool': 0.05}

    composite = sum(scores[k] * weights[k] for k in weights)
    return composite  # -1 to +1; above 0.5 = strong institutional conviction
```

---

## Tier 3 — The Experimental Frontier

> These are hypotheses. Some are informed by theory. Some are informed by observation. None are proven. All are testable. This is the lab.

---

### X-01: Satellite-Derived Economic Activity as Leading Indicator

**Hypothesis:** Night-light intensity from satellite imagery correlates with economic activity 1–2 quarters ahead of GDP reports. Anomalies in specific regions predict sector returns.

**Theory:** Economic activity emits light. More factories running at night = more production. Satellite data is real-time; GDP is backward-looking and delayed.

**Testable prediction:**
```
H0: Night-light growth in industrial zones does NOT predict manufacturing stock returns
H1: Night-light growth anomaly (Z > 1.5 vs prior year) in US Rust Belt industrial
    counties predicts XLI (Industrials ETF) positive 3-month returns with
    accuracy > 55% vs base rate of 52%
```

**Data source:** NASA Black Marble (free), NOAA VIIRS (free)
**Test methodology:** Download county-level night-light data monthly. Compute YoY change. Test cross-sectional correlation with subsequent 3-month sector returns.
**Effort to test:** 2-3 weeks of data engineering. Medium.
**Prior art:** Some hedge funds (Two Sigma, Renaissance) reportedly use satellite data. But night-light → SECTOR ROTATION specifically is underexplored.

---

### X-02: Language Model Semantic Drift as Regime Signal

**Hypothesis:** The semantic distance between consecutive Fed communications (FOMC minutes, speeches) is measurable and predicts policy surprises 4–8 weeks before they occur.

**Theory:** When the Fed is about to change direction, their language changes first — gradually. Embedding these documents in a semantic vector space and measuring drift velocity should detect policy pivot before markets price it.

**Testable prediction:**
```
H0: Semantic drift between consecutive FOMC minutes does NOT predict
    10yr Treasury yield change over next 30 days
H1: Semantic drift > 0.15 cosine distance between consecutive FOMC documents
    predicts 10yr yield moving in direction consistent with language shift
    with accuracy > 58% vs base rate
```

**Implementation:**
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')  # free, runs locally on Ollama

def fomc_drift_signal(minutes_text_t, minutes_text_t_minus_1):
    # Embed both documents
    emb_current  = model.encode([minutes_text_t])
    emb_previous = model.encode([minutes_text_t_minus_1])

    # Cosine distance (0=identical, 2=opposite)
    distance = 1 - cosine_similarity(emb_current, emb_previous)[0][0]

    # Direction of drift: which themes increased/decreased?
    hawkish_words = ['inflation', 'tighten', 'restrictive', 'elevated', 'above target']
    dovish_words  = ['growth', 'employment', 'accommodative', 'easing', 'below target']

    hawk_score_now  = sum(minutes_text_t.lower().count(w) for w in hawkish_words)
    hawk_score_prev = sum(minutes_text_t_minus_1.lower().count(w) for w in hawkish_words)
    hawk_drift = hawk_score_now - hawk_score_prev

    return {
        'semantic_distance': distance,       # magnitude of change
        'hawk_drift': hawk_drift,            # direction of change
        'signal': 'HAWKISH' if hawk_drift > 3 else 'DOVISH' if hawk_drift < -3 else 'NEUTRAL'
    }
```

**This runs on your Ollama instance.** No API cost. Pure local inference.

---

### X-03: Cross-Asset Momentum Correlation Inversion

**Hypothesis:** When the historical correlation between two assets inverts (was positive, becomes negative, or vice versa), it reliably predicts a large directional move in the LESS liquid asset within 20–40 trading days.

**Theory:** Correlations between assets are maintained by arbitrageurs and institutional hedgers. When those correlations break down, it means the institutional plumbing has changed — someone large is repositioning. The less liquid asset lags. You front-run the catch-up.

**Testable prediction:**
```
H0: Correlation inversion between Asset_A and Asset_B does NOT predict
    directional move in less-liquid Asset_B
H1: When 20-day correlation between SPY and XLF flips from positive to negative
    AND persists for 3+ days, XLF produces a directional move >3% within
    30 days in the direction opposite to its recent trend,
    with accuracy > 60%
```

**Test pairs (start with these, expand):**
- SPY ↔ XLF (Financials): rates story
- Oil ↔ Airlines: cost story
- USD ↔ Emerging Markets: dollar story
- Copper ↔ AUD: China story
- VIX ↔ Growth Stocks: risk story

```python
def correlation_inversion_scan(returns_df, lookback_short=20, lookback_long=60):
    results = []
    for col_a in returns_df.columns:
        for col_b in returns_df.columns:
            if col_a >= col_b: continue

            corr_short = returns_df[col_a].rolling(lookback_short).corr(returns_df[col_b])
            corr_long  = returns_df[col_a].rolling(lookback_long).corr(returns_df[col_b])

            # Inversion = short-term correlation has flipped sign vs long-term
            inversion = (corr_short.iloc[-1] * corr_long.iloc[-1]) < 0
            magnitude = abs(corr_short.iloc[-1] - corr_long.iloc[-1])

            if inversion and magnitude > 0.4:
                results.append({
                    'pair': (col_a, col_b),
                    'short_corr': corr_short.iloc[-1],
                    'long_corr':  corr_long.iloc[-1],
                    'inversion_magnitude': magnitude
                })
    return sorted(results, key=lambda x: -x['inversion_magnitude'])
```

---

### X-04: Earnings Call Prosody Analysis (Non-Verbal Signal)

**Hypothesis:** The acoustic properties of an earnings call — speaking rate, pitch variance, pause frequency of the CEO/CFO — contain information about confidence/deception that predicts subsequent stock performance beyond text sentiment alone.

**Theory:** Deceptive or anxious speech has measurable acoustic signatures (slower rate, more pauses, higher pitch variance). Humans are poor at detecting these consciously. Machines are not.

**Testable prediction:**
```
H0: Acoustic features of earnings calls add zero predictive value beyond
    text-based NLP sentiment
H1: CEO speaking rate variance (std dev of WPM across call) combined with
    pause frequency in response to analyst questions predicts 30-day
    stock return direction with accuracy > 56% incremental to text NLP
```

**Implementation:**
```python
# Step 1: Download earnings call audio (Seeking Alpha, EarningsWhispers)
# Step 2: Transcribe with timestamps using Whisper (free, runs on Ollama)
# Step 3: Extract acoustic features

import whisper  # OpenAI Whisper — runs locally
import librosa  # Audio analysis
import numpy as np

def analyze_call_prosody(audio_file_path):
    # Load and analyze audio
    y, sr = librosa.load(audio_file_path)

    # Pitch analysis
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean   = pitches[magnitudes > magnitudes.mean()].mean()
    pitch_var    = pitches[magnitudes > magnitudes.mean()].std()

    # Energy (volume) — drops during uncertain speech
    rms_energy     = librosa.feature.rms(y=y)[0]
    energy_variance = rms_energy.std()

    # Speaking rate via Whisper word timestamps
    model = whisper.load_model('base')
    result = model.transcribe(audio_file_path, word_timestamps=True)
    word_times = [w['end'] - w['start'] for seg in result['segments']
                  for w in seg.get('words', [])]
    speaking_rate = len(word_times) / (sum(word_times) + 1e-9)

    # Pause detection
    pauses = [t for t in word_times if t > 0.8]  # pauses > 800ms
    pause_rate = len(pauses) / len(word_times)

    return {
        'pitch_variance':  pitch_var,
        'energy_variance': energy_variance,
        'speaking_rate':   speaking_rate,
        'pause_rate':      pause_rate,
        'confidence_proxy': speaking_rate / (pause_rate + 0.01)  # higher = more confident
    }
```

**This is genuinely novel.** Text NLP of earnings calls exists. Acoustic analysis + combining with price prediction at this granularity is not published in literature to meaningful degree.

---

### X-05: Order Flow Imbalance Autocorrelation Decay Rate

**Hypothesis:** The rate at which intraday order flow imbalance (OFI) autocorrelation decays across different stocks is a fingerprint of institutional presence. Stocks where OFI autocorrelation decays slowly (persistent imbalance) are experiencing sustained institutional accumulation/distribution, predictable at the swing trade horizon.

**Theory:** Random retail order flow has rapidly decaying autocorrelation. Institutional programs (which execute over days/weeks) create slow-decaying autocorrelation in OFI. Detecting this early gives you the direction of the smart money program BEFORE it completes.

**Testable prediction:**
```
H0: OFI autocorrelation decay rate does NOT predict 5-day forward return
H1: Stocks in the bottom quartile of OFI autocorrelation decay rate
    (most persistent order flow imbalance) produce positive 5-day forward
    returns with accuracy > 57% when OFI is positive,
    negative returns with accuracy > 57% when OFI is negative
```

```python
import statsmodels.api as sm

def ofi_decay_analysis(tick_data, lookback_days=5):
    """
    Compute order flow imbalance autocorrelation decay rate.
    tick_data: DataFrame with columns ['buy_volume', 'sell_volume'] at 5-min intervals
    """
    ofi = tick_data['buy_volume'] - tick_data['sell_volume']
    ofi_normalized = ofi / (tick_data['buy_volume'] + tick_data['sell_volume'] + 1)

    # Autocorrelation at lags 1-20
    acf_values = sm.tsa.acf(ofi_normalized, nlags=20)

    # Fit exponential decay: ACF(lag) = A * exp(-lambda * lag)
    from scipy.optimize import curve_fit
    def exp_decay(lag, A, lam): return A * np.exp(-lam * lag)

    lags = np.arange(1, 21)
    try:
        popt, _ = curve_fit(exp_decay, lags, acf_values[1:], p0=[1.0, 0.5])
        decay_rate = popt[1]  # lambda — lower = slower decay = more persistent
    except:
        decay_rate = 1.0  # default fast decay

    # Sign of persistent OFI
    ofi_direction = 'BUY' if ofi.mean() > 0 else 'SELL'

    return {
        'decay_rate': decay_rate,
        'persistent': decay_rate < 0.3,      # slow decay = institutional
        'ofi_direction': ofi_direction,
        'signal': ofi_direction if decay_rate < 0.3 else 'NOISE'
    }
```

**Note:** Requires intraday tick data. Polygon.io free tier provides this. This IS something large quantitative shops have studied internally, but the specific combination of decay rate + swing trade horizon + integration with ICT structure is novel.

---

### X-06: Differential Equation Model of Institutional Accumulation

**Hypothesis:** Institutional accumulation of a stock follows a logistic growth curve (the same ODE used for population growth). Fitting this model to volume and price data allows prediction of the inflection point (the "breakout") weeks before it occurs.

**Theory:** An institution buying a large position cannot buy at market price without moving it. They accumulate slowly. Volume increases follow a logistic pattern — slow start, acceleration, saturation (when they finish). The inflection point of the logistic curve is the breakout.

**The Math:**
```
Accumulation Model (Logistic ODE):
  dA/dt = r·A·(1 - A/K)

Where:
  A = accumulated institutional position (estimated from volume anomaly)
  r = accumulation rate (estimated from rolling volume trend)
  K = maximum position (estimated from float and institutional capacity)
  t = time

Solution: A(t) = K / (1 + ((K - A₀)/A₀)·e^{-rt})

Inflection point (breakout prediction):
  t_inflection = (1/r) · ln((K - A₀)/A₀)
  
At the inflection point:
  - Accumulation is at A = K/2 (halfway done)
  - Rate of accumulation is MAXIMUM
  - Institutions transition from quiet buying to accepting market impact
  - Volume spikes
  - Price breaks out
  - This IS the ICT "institutional order flow" that the ICT Decision Engine reacts to
  - Difference: you are predicting it BEFORE the breakout, not after
```

```python
from scipy.optimize import curve_fit
from scipy.stats import zscore
import numpy as np

def fit_logistic_accumulation(volume_series, price_series, lookback=60):
    """
    Fit logistic model to volume anomaly to predict institutional accumulation phase.
    """
    # Volume anomaly = excess volume above 20-day average
    vol_ma = volume_series.rolling(20).mean()
    vol_anomaly = (volume_series - vol_ma) / vol_ma
    vol_anomaly_cumsum = vol_anomaly.clip(lower=0).cumsum()  # only count buying anomaly

    if len(vol_anomaly_cumsum) < lookback:
        return None

    t = np.arange(lookback)
    A = vol_anomaly_cumsum.iloc[-lookback:].values
    A = A - A.min()  # normalize to start at 0

    # Fit logistic curve
    def logistic(t, K, r, t0):
        return K / (1 + np.exp(-r * (t - t0)))

    try:
        popt, pcov = curve_fit(logistic, t, A, p0=[A.max(), 0.1, lookback/2],
                                bounds=([0, 0, 0], [A.max()*2, 1, lookback]))
        K, r, t0 = popt

        current_t = lookback - 1
        current_A = logistic(current_t, K, r, t0)
        pct_complete = current_A / K  # what % of accumulation is done

        days_to_inflection = t0 - current_t  # negative = already past inflection

        return {
            'pct_accumulation_complete': pct_complete,
            'days_to_inflection': days_to_inflection,
            'accumulation_rate': r,
            'signal': (
                'PRE_BREAKOUT'    if 0 < days_to_inflection < 10 else
                'AT_INFLECTION'   if -2 < days_to_inflection <= 0 else
                'POST_INFLECTION' if days_to_inflection <= -2 else
                'EARLY'
            )
        }
    except:
        return None
```

**Why this is interesting:** ICT identifies order blocks and FVGs as evidence of institutional activity AFTER the move. This model attempts to detect the ACCUMULATION PHASE before the move using the same mathematics that describes any growth-to-saturation process in nature. It connects your ICT framework to a first-principles physical model.

---

### X-07: Entropy-Based Market Complexity as Regime Predictor

**Hypothesis:** The Shannon entropy of the daily return distribution across a stock universe measures market "complexity." When entropy drops sharply (returns become too uniform — everyone winning or everyone losing), a regime change is imminent.

**Theory:** High entropy = diverse outcomes = healthy market. Low entropy = correlation crisis forming = drawdown risk. This is information theory applied to cross-sectional return distributions.

**The Math:**
```
Shannon Entropy of Cross-Sectional Returns:
  H(t) = -Σ p(r_i) · log₂(p(r_i))

Where:
  p(r_i) = fraction of stocks in return bucket i (e.g., decile)

  Maximum entropy (10 deciles, uniform) = log₂(10) = 3.32 bits
  Minimum entropy (all stocks same return) → 0 bits

Signal:
  H falling below 2.0 bits: returns are concentrating → correlation crisis building
  H below 1.5 bits: something is very wrong → defensive positioning
  H above 3.0 bits: maximum diversity → trend-following works best here
```

```python
from scipy.stats import entropy
import numpy as np
import pandas as pd

def cross_sectional_entropy(returns_matrix):
    """
    Compute Shannon entropy of cross-sectional return distribution.
    returns_matrix: DataFrame, rows=dates, cols=stocks, values=daily returns
    """
    entropies = []

    for date in returns_matrix.index:
        daily_rets = returns_matrix.loc[date].dropna()

        # Bin returns into 10 deciles
        bins = np.percentile(daily_rets, np.arange(0, 110, 10))
        hist, _ = np.histogram(daily_rets, bins=bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]  # remove zeros for log

        H = entropy(probs, base=2)
        entropies.append({'date': date, 'entropy': H})

    return pd.DataFrame(entropies).set_index('date')

# Rolling entropy with Z-score
# entropy_z < -2: Extremely low diversity = correlation crisis = sell
# entropy_z > +1.5: Very high diversity = good environment = buy trend/momentum
```

---

## The Scientific Method Applied to Trading Research

> Every Tier 3 hypothesis is a scientific experiment. The rules of science apply. No exceptions. Narrative is not evidence. Evidence is evidence.

### The Hypothesis Lifecycle

```
STAGE 1 — FORMATION
  Source: Observation, theory, anomaly in existing data
  Output: Written hypothesis in falsifiable form
    "Variable X predicts outcome Y with accuracy > Z% over horizon H
     in condition C, measured on dataset D"
  Requirements:
    - Specific threshold (not "it seems to work")
    - Defined horizon
    - Defined market condition (works in all regimes? bull only?)
    - Named dataset (no moving the goalposts)

STAGE 2 — IN-SAMPLE DEVELOPMENT
  Data: First 70% of available history
  Goal: Find the optimal parameterization
  Rules:
    - ONE hypothesis per experiment
    - Document every parameter tried
    - Record ALL results (not just the good ones — critical)
    - Target: Does any version of this hypothesis beat the base rate?
  Output: Best parameter set, in-sample accuracy, t-statistic

STAGE 3 — OUT-OF-SAMPLE TEST
  Data: Remaining 30% that was NEVER TOUCHED during Stage 2
  This is the moment of truth — no more adjustments
  Rules:
    - Run ONCE. Not iteratively.
    - Accept the result regardless of outcome
    - A failed OOS test is not a failure; it is information
  Output: OOS accuracy vs in-sample accuracy (degradation tells you overfitting degree)
  Pass threshold: OOS accuracy > base rate + 2% AND t-stat > 2.0

STAGE 4 — PAPER TRADING (Live but No Capital)
  Duration: Minimum 3 months; ideally 6 months
  Goal: Does it work in live conditions with real execution constraints?
  Tests for:
    - Lookahead bias you didn't notice in backtest
    - Data quality issues
    - Execution reality (can you actually get fills at model prices?)
  Pass threshold: Paper results within 20% of backtest performance

STAGE 5 — LIVE MICRO-DEPLOYMENT
  Capital: 0.1% of account maximum per signal
  Duration: 50 trades minimum
  Goal: Does the edge hold with real money and real psychology?
  Pass threshold: Sharpe > 0.5, positive EV per trade
  Promotion to Tier 2: 100+ trades, Sharpe > 0.8, passing multiple regime test

STAGE 6 — PROMOTION OR RETIREMENT
  IF passes all stages → Tier 2 method → scale capital allocation
  IF fails at any stage → DOCUMENT WHY → retire to "lessons learned"
  No zombie experiments — either working or retired, never "still trying"
```

---

## The Hypothesis Registry

> Every hypothesis ever tested. Every result recorded. This is the scientific record.

### Registry Template (One Entry Per Hypothesis)

```markdown
## HYP-[NUMBER]: [SHORT NAME]

**Filed:** [DATE]
**Tier:** 3 (experimental)
**Status:** FORMING / TESTING_IN_SAMPLE / TESTING_OOS / PAPER_TRADING / LIVE / RETIRED

### Hypothesis Statement
"[VARIABLE X] predicts [OUTCOME Y] with accuracy > [Z]% over [H] horizon
 in [CONDITION C], measured on [DATASET D]."

### Theoretical Basis
[Why should this work? What is the causal mechanism?]

### Falsification Test
[Specific, observable outcome that proves the hypothesis WRONG]

### Data Required
- Source: [where to get it]
- Preprocessing: [what transformations]
- Availability: [free/paid, real-time/delayed]

### In-Sample Results
- Dataset: [dates, universe]
- Accuracy: [X]% vs base rate [Y]%
- t-statistic: [Z]
- Best parameters: [list]
- Date completed: [DATE]

### Out-of-Sample Results
- Accuracy: [X]% vs in-sample [Y]%
- Degradation: [Z]% — [acceptable/excessive]
- t-statistic: [N]
- PASS / FAIL
- Date completed: [DATE]

### Paper Trading Results
- Duration: [N months]
- Trades: [N]
- Sharpe: [X]
- PASS / FAIL

### Decision
[ ] PROMOTE to Tier 2 → deploy live capital
[ ] RETIRE → document lessons learned
[ ] CONTINUE paper trading → need more data

### Lessons Learned
[What did this teach us regardless of outcome?]
```

### Active Hypothesis Registry

| ID | Name | Filed | Status | Current Stage | Priority |
|---|---|---|---|---|---|
| HYP-001 | Satellite Night-Light Economic Activity | — | FORMING | Stage 1 | Medium |
| HYP-002 | FOMC Semantic Drift | — | FORMING | Stage 1 | High |
| HYP-003 | Correlation Inversion Pairs | — | FORMING | Stage 1 | High |
| HYP-004 | Earnings Call Prosody | — | FORMING | Stage 1 | Medium |
| HYP-005 | OFI Autocorrelation Decay | — | FORMING | Stage 1 | High |
| HYP-006 | Logistic Accumulation Model | — | FORMING | Stage 1 | Very High |
| HYP-007 | Cross-Sectional Entropy | — | FORMING | Stage 1 | Medium |

---

## The Experiment Log Structure

> Every experiment generates a log. Logs are never deleted. They compound into institutional knowledge.

```
/vault/research/experiments/
  HYP-001/
    hypothesis.md        ← the formal hypothesis
    data_notes.md        ← data quality observations
    code/
      fetch_data.py
      run_experiment.py
      analyze_results.py
    results/
      in_sample_results.csv
      oos_results.csv
      paper_trade_log.csv
    analysis/
      in_sample_analysis.md
      oos_analysis.md
      decision.md         ← promote or retire; why
    lessons_learned.md    ← written regardless of outcome
```

### The Result Recording Standard

```python
# Every experiment must record these metrics — no exceptions

REQUIRED_METRICS = {
    # Performance
    'accuracy':           float,   # directional accuracy
    'base_rate':          float,   # what random/naive gives you
    'accuracy_above_base': float,  # the actual edge
    't_statistic':        float,   # is it statistically significant
    'p_value':            float,   # probability of false positive

    # Risk
    'sharpe_ratio':       float,
    'max_drawdown':       float,
    'win_rate':           float,
    'avg_win_r':          float,
    'avg_loss_r':         float,
    'expected_value':     float,   # win_rate * avg_win - loss_rate * avg_loss

    # Robustness
    'n_trades':           int,     # sample size
    'n_years':            float,   # time covered
    'n_regimes_tested':   int,     # bull, bear, sideways at minimum
    'oos_degradation_pct': float,  # (in_sample - oos) / in_sample

    # Metadata
    'hypothesis_id':      str,
    'data_start':         date,
    'data_end':           date,
    'parameters':         dict,
    'run_date':           datetime
}
```

---

## Failure Analysis Protocol

> A failed experiment is worth more than a successful one if you understand WHY it failed.

### The Post-Mortem Framework

```
When a hypothesis fails (OOS accuracy < threshold OR p-value > 0.05):

QUESTION 1 — Was the failure due to OVERFITTING?
  Signs: In-sample accuracy >> OOS accuracy (>10% degradation)
  Cause: Too many parameters optimized on too little data
  Lesson: Simplify the model; fewer parameters; more data required

QUESTION 2 — Was the failure due to REGIME DEPENDENCY?
  Signs: Works in bull markets, fails in bear markets (or vice versa)
  Cause: The underlying mechanism only exists in certain conditions
  Lesson: Add regime filter; it may still be valuable as a conditional signal

QUESTION 3 — Was the failure due to DATA QUALITY?
  Signs: Unexpected performance pattern; results don't make intuitive sense
  Cause: Survivorship bias, lookahead bias, point-in-time data issues
  Lesson: Fix the data pipeline; re-test with clean data

QUESTION 4 — Was the failure due to TRANSACTION COSTS?
  Signs: Gross performance good; net performance poor
  Cause: High turnover strategy requires unrealistically tight spreads
  Lesson: Either reduce turnover or require larger gross edge to clear costs

QUESTION 5 — Was the hypothesis fundamentally WRONG?
  Signs: Poor performance even in-sample with clean data
  Cause: The causal mechanism doesn't exist or is too weak to trade
  Lesson: The mechanism assumption was wrong; update worldview accordingly

DOCUMENT ALL FIVE ANSWERS. Do not just mark "FAILED" and move on.
The failure analysis IS the research value.
```

### The Failure-to-Insight Pipeline

```
Failed Hypothesis → Post-Mortem → Categorize Failure Type
     │
     ├── Overfitting → Simplified version → Re-test
     ├── Regime-dependent → Add regime filter → New hypothesis
     ├── Data issue → Clean data → Re-test
     ├── Transaction cost → Reduce turnover → New hypothesis
     └── Fundamentally wrong → Update model of world → New direction

Every failed experiment generates AT LEAST one new hypothesis.
Failures are not dead ends. They are forks.
```

---

## The Iteration Engine

> Speed of learning, not speed of trading, is the competitive advantage.

### Learning Rate Optimization

```
GOAL: Get from hypothesis to validated edge as fast as possible
WITHOUT cutting corners that destroy validity

SPEED LEVERS:

1. SYNTHETIC DATA for rapid in-sample testing
   Use Monte Carlo + known statistical properties to generate
   1000 years of synthetic market data in seconds
   Test hypothesis on synthetic data BEFORE touching real data
   If it doesn't work on synthetic data with known properties → abandon fast

2. PARALLEL EXPERIMENTATION
   Run multiple hypotheses through Stage 2 simultaneously
   AI (your Ollama instance) runs overnight backtests
   You review results each morning
   1 experiment/week → 5 experiments/week with AI assistance

3. PROGRESSIVE COMPLEXITY
   Start with SIMPLEST version of each hypothesis
   IF simple version shows edge → add complexity to improve
   IF simple version fails → complex version almost certainly fails
   Simple = fast to test; complex = slow to test
   Never start complex

4. THE WEEKLY REVIEW CADENCE
   Monday: Review all active experiment results from prior week
   Tuesday: Write new hypotheses based on Monday observations
   Wednesday-Friday: Data engineering + code for new experiments
   Saturday-Sunday: Long runs; deep analysis

5. THE 80/20 RESEARCH RULE
   80% of research time → Tier 1 improvements (highest return on time)
   15% of research time → Tier 2 development (validated extensions)
   5% of research time → Tier 3 experiments (high risk, high reward)
   DO NOT invert this. Experimental work is exciting but Tier 1 pays the bills.
```

### The Compound Research Effect

```
Year 1: Tier 1 operational. Tier 2 being validated. 2 Tier 3 experiments running.
         → Edge: Tier 1 provides moderate, reliable returns

Year 2: Tier 2 methods integrated. 1-2 Tier 3 methods validated and promoted.
         → Edge: Tier 1 + Tier 2 = noticeably better performance
         → XGBoost retrained with new features from validated methods

Year 3: Another 2-3 Tier 3 methods validated. Prior Tier 3 successes now Tier 1.
         → Edge: Compounding. System significantly better than Year 1.
         → The failed experiments have improved the model of markets even more.

The research program compounds exactly like a financial investment.
Early years: slow visible progress, important foundational work.
Later years: accelerating returns as each validated method builds on prior.
```

---

## Edge Validation Criteria

> These are the minimum bars. If a method doesn't clear all of them, it does not get capital. No exceptions, no matter how good the backtest looks.

### The Five Gates

```
GATE 1 — STATISTICAL SIGNIFICANCE
  t-statistic of mean return > 2.0
  p-value < 0.05
  Sample size > 200 independent trades
  IF fails: More data needed, or edge is too small to detect

GATE 2 — ECONOMIC SIGNIFICANCE
  Expected value per trade > 0.3R
  (After realistic transaction costs)
  IF fails: Edge exists but not enough to overcome friction

GATE 3 — ROBUSTNESS
  Works in at least 2 of 3 market regimes (bull, bear, sideways)
  OOS degradation < 30% of in-sample performance
  Works on at least 3 different asset classes (not just SPY)
  IF fails: Overfitted to specific conditions; may still be valuable as conditional signal

GATE 4 — SURVIVABILITY
  Max drawdown in backtest < 25%
  No single year with return worse than -15%
  No 3-month period worse than -12%
  IF fails: Edge may be real but too volatile; use smaller Kelly fraction

GATE 5 — IMPLEMENTATION
  Can execute with realistic fill assumptions (limit orders, not market orders)
  Can monitor without 24/7 screen time (alerts-based)
  Does not require tick-level data to execute (swing trade horizon)
  IF fails: Theoretically valid but operationally impossible
```

---

## The Research Roadmap

> Sequenced priorities. Do these in order. Do not skip ahead.

### Phase 1 — Solidify Tier 1 (Months 1-2)

```
[ ] Connect COT live feed to imbalance_engine — automate weekly update
[ ] Validate PEAD model on your universe (57 assets) — confirm 50%+ accuracy
[ ] Build insider signal scraper from OpenInsider.com — automate Form 4 pull
[ ] Add ERP, CAPE, recession prob as live features to XGBoost
[ ] Retrain XGBoost with macro features — measure accuracy improvement
[ ] Deploy sector rotation clock — which phase are we in right now?

Target: XGBoost accuracy improves from 58.3% → 61%+ with macro features
```

### Phase 2 — Build Tier 2 Infrastructure (Months 2-4)

```
[ ] Options flow scraper (Unusual Whales free tier) — unusual activity alerts
[ ] HYG/SPY divergence signal — live alert when 2-week divergence detected
[ ] Smart Money Composite score — combine all Tier 2 signals per ticker
[ ] Credit spread monitoring (FRED H.15 series, daily pull)
[ ] Integrate Smart Money Composite as XGBoost feature layer

Target: Petroulas gate accuracy improves — fewer false positives, clearer signals
```

### Phase 3 — Launch Tier 3 Experiments (Months 3-6, parallel with Phase 2)

```
Priority order (highest ROI on research time first):

[ ] HYP-006: Logistic Accumulation Model
    Why first: Uses only OHLCV data you already have; no new data source needed
    Time to test: 3 weeks
    Potential: Direct integration with ICT setup detection

[ ] HYP-002: FOMC Semantic Drift
    Why second: FOMC minutes are free text; Ollama runs locally; easy data
    Time to test: 2 weeks
    Potential: Rate decision predictor = bond + equity signal

[ ] HYP-003: Correlation Inversion Pairs
    Why third: OHLCV data only; tests on existing 57-asset universe
    Time to test: 2 weeks
    Potential: Cross-asset early warning system

[ ] HYP-005: OFI Autocorrelation Decay
    Why fourth: Requires intraday data (Polygon free tier); slightly more work
    Time to test: 4 weeks
    Potential: Strongest institutional footprint detector if validated

[ ] HYP-007: Cross-Sectional Entropy
    Why fifth: Simple to compute; tests on existing universe
    Time to test: 1 week
    Potential: Regime filter that improves ALL other signals

[ ] HYP-004: Earnings Call Prosody
    Why last: Requires audio files; most engineering overhead
    Time to test: 6 weeks
    Potential: Genuinely novel; could be substantial edge if validated
```

### Phase 4 — Integration and Scaling (Months 6-12)

```
[ ] Integrate all validated Tier 3 methods as XGBoost features
[ ] Full retrain with complete feature set — 2yr+ training data
[ ] Calibrate Petroulas gate with empirical accuracy data from real trades
[ ] Build performance attribution — which features contribute most to accuracy
[ ] Begin Tier 3 second generation — new experiments informed by Phase 3 lessons
[ ] Document the complete model in research papers (even private ones — forces rigor)
```

---

## Integration Checkpoints

> How the research program feeds back into the live trading system.

```python
# Monthly integration checklist

def monthly_integration_review():
    """
    Run first Sunday of every month.
    Determines which research advances get promoted to live system.
    """

    # 1. Any Tier 3 hypotheses cleared Stage 3 (OOS test)?
    #    IF YES → begin paper trading immediately
    oos_graduates = hypothesis_registry.filter(status='OOS_PASSED')

    # 2. Any paper trading experiments cleared Stage 4?
    #    IF YES → deploy at 0.1% capital
    paper_graduates = hypothesis_registry.filter(status='PAPER_PASSED')

    # 3. Any live micro-deployments reached 50 trades?
    #    IF YES → full statistical review; promote or retire
    live_reviews = hypothesis_registry.filter(
        status='LIVE', n_trades_gte=50
    )

    # 4. Add validated features to XGBoost and retrain
    new_features = [h.feature_name for h in paper_graduates]
    if new_features:
        retrain_xgboost(add_features=new_features, data_years=2)
        log_accuracy_improvement()

    # 5. Update Kimi fault prompt with newly validated signals
    if paper_graduates:
        update_kimi_prompt(new_signals=paper_graduates)

    # 6. Write monthly research report to vault
    write_to_vault('research/monthly_review.md', {
        'new_features_added': new_features,
        'xgboost_accuracy_delta': accuracy_improvement,
        'active_experiments': len(hypothesis_registry.active()),
        'retired_experiments': len(hypothesis_registry.retired()),
        'lessons_this_month': compile_lessons()
    })
```

---

## Related Notes

- [[Imbalance Engine MOC]]
- [[ICT Swing Trade Decision Engine]]
- [[Intelligence System MOC]]
- [[Quantitative Finance MOC]]
- [[XGBoost Bias Engine]]
- [[Kimi Fault Detector]]
- [[Hypothesis Registry]]
- [[Experiment Logs]]
- [[Failure Analysis Archive]]
- [[Son of Anton Architecture]]
- [[Research Monthly Reviews]]
- [[Feature Importance Tracker]]

---

*Science is not about being right. It is about being less wrong over time. Every experiment, win or lose, reduces the error bars on your model of the world. The edge does not come from a single brilliant insight. It comes from a thousand careful experiments and the discipline to keep running them.*
