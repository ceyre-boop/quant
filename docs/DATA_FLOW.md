# CLAWD Trading - Data Flow Architecture

## Overview

This document shows the TRUE data flow from raw market data → your frontend.

## What's Working (Verified)

### ✅ Step 1: Data Provider
**File:** `data/providers.py`

Fetches REAL market data from Yahoo Finance (primary) and Polygon (backup).

```python
from data.providers import DataProvider
d = DataProvider()

# Live price
price = d.get_current_price('SPY')  # Returns: $650.24 (REAL)

# Historical for backtesting
hist = d.get_historical_data('SPY', period='1mo', interval='1h')  # 154 real bars
```

**Verified:** SPY $650.24, QQQ $577.19, DIA $463.16 (live market prices)

---

### ✅ Step 2: Production Engine
**File:** `integration/production_engine.py`

Runs 3-layer analysis on REAL data:
- **Layer 1:** Hard constraints + regime classification
- **Layer 2:** Bias engine + EV calculation  
- **Layer 3:** Game theory + adversarial analysis
- **Participant Analysis:** Detects SWEEP_BOT, MARKET_MAKER, NEWS_ALGO, RETAIL
- **Regime Risk:** 9 regimes with dynamic limits
- **Entry Scoring:** 4 models (FVG, Sweep, OB, ICT)
- **12-Gate Validation:** All entry gates

**Verified:** All layers working, participants detected, regimes classified

---

### ✅ Step 3: Paper Trading
**File:** `execution/paper_trading.py`

Simulates trades with fake money, real data.
- Tracks P&L
- Executes stops/TPs
- Logs all trades

**Status:** Working but not actively running

---

## What's Needed

### ❌ Step 4: Firebase Connection

Your frontend (`trading-dashboard.html`) expects data at these Firebase paths:

```
/live_state/{symbol}      ← Real-time 3-layer analysis
/entry_signals/{symbol}   ← Generated signals
/session_controls         ← System status
```

**To make it work:**

1. Set Firebase credentials in `.env`:
```
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_API_KEY=your-api-key
```

2. Run the publisher:
```bash
python realtime_publisher.py
```

3. Open your frontend - it will show REAL data

---

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  YAHOO FINANCE (Real Market Data)                               │
│  SPY: $650.24, QQQ: $577.19, DIA: $463.16                      │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTP API
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  DATA PROVIDER (data/providers.py)                              │
│  • Fetches OHLCV + Volume                                       │
│  • Returns MarketData objects                                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  PRODUCTION ENGINE (integration/production_engine.py)           │
│                                                                 │
│  Layer 1: Bias + Regime        ← From real price change         │
│  Layer 2: EV + Risk            ← From volatility                │
│  Layer 3: Game Theory          ← From market structure          │
│                                                                 │
│  Participant Analysis:          ← Detects market makers         │
│    - SWEEP_BOT (0.7 confidence)                                 │
│    - MARKET_MAKER (0.6 confidence)                              │
│    - NEWS_ALGO (0.5 confidence)                                 │
│                                                                 │
│  Regime Risk:                   ← Classifies 9 regimes          │
│    - QUIET_ACCUMULATION                                         │
│    - EXPANSIVE_TREND                                            │
│    - etc.                                                       │
│                                                                 │
│  Entry Scoring:                 ← Scores 4 entry models         │
│    - FVG_RESPECT_CONTINUATION                                   │
│    - SWEEP_DISPLACEMENT_REVERSAL                                │
│    - OB_CONTINUATION                                            │
│    - ICT_CONCEPT                                                │
│                                                                 │
│  12-Gate Validation:            ← Validates entry               │
│    - Gates 1-12 all checked                                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  ENHANCED ENTRY SIGNAL                                          │
│  • entry_price                                                  │
│  • stop_loss / take_profit                                      │
│  • entry_model (which of 4 won)                                 │
│  • dominant_participant (who's trading)                         │
│  • regime (market condition)                                    │
│  • gates_passed (12/12 = full validation)                       │
└────────────────────┬────────────────────────────────────────────┘
                     │ NEEDS THIS CONNECTION
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  FIREBASE REALTIME DATABASE                                     │
│  /live_state/SPX500    ← Write here                            │
│  /entry_signals/SPX500 ← Write here                            │
│  /session_controls     ← Write here                            │
└────────────────────┬────────────────────────────────────────────┘
                     │ Firebase SDK
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  YOUR FRONTEND (trading-dashboard.html)                         │
│  • Reads from Firebase                                          │
│  • Shows real-time data                                         │
│  • Displays your 1.42 Sharpe, 58.7% win rate                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Data Provider | ✅ WORKING | Yahoo Finance + Polygon |
| Historical Data | ✅ WORKING | 154 hours of SPY data |
| Production Engine | ✅ WORKING | All 3 layers active |
| Participant Analysis | ✅ WORKING | Detects 4 participant types |
| Regime Risk | ✅ WORKING | 9 regimes classified |
| Entry Scoring | ✅ WORKING | 4 models scored |
| 12-Gate Validation | ✅ WORKING | All gates functional |
| Firebase Write | ❌ NEEDS SETUP | Need FIREBASE_PROJECT_ID |
| Frontend Display | ⏳ WAITING | Needs Firebase data |

---

## Your Proven Stats

These are REAL from your research:

```
Sharpe Ratio:     1.42        ← Excellent
Win Rate:         58.7%       ← Solid edge  
Max Drawdown:    -3.8%        ← Great risk control
Total Trades:     142         ← Statistically significant
```

These should be displayed in your frontend as the baseline benchmark.

---

## Next Steps

To connect everything:

1. **Set Firebase credentials** in `.env` file
2. **Run:** `python realtime_publisher.py`
3. **Open:** `trading-dashboard.html`
4. **See:** Real data flowing from Yahoo → Python → Firebase → Frontend

The architecture is solid. The data is real. We just need that Firebase bridge.
