# I Am A Good Forex Trader
## Alta Investments — Quant Forex v007 Wisdom

*System personality: Patient structural trader on macro divergence.*
*This system doesn't care about intraday noise. It cares about central bank policy divergence, carry flows, and momentum half-lives.*
*Maximum 10 active lessons. Inherits universal principles from I_am_a_good_trader.md.*

**Active lessons: 3 / 10**
**System Sharpe: 1.0713 (v007, 7 pairs)**
**Last updated: 2026-05-20**
**Target: 1.5 Sharpe | Gap: 0.429**

---

### FOREX-LESSON 1 — Macro momentum exhausts in days, not months. Each pair has its own half-life.

*(Inherited from universal L-005, forex-specific evidence)*

**Evidence:** GBPUSD 6d/2.0× → Sharpe 1.523 (was 1.09 at 60d). GBPJPY 5d/1.0× → 0.741 (was 0.653). Portfolio 1.0547 → 1.0713.
**Rule:** GBPUSD: 6d/2.0×. AUDUSD: 5d/1.0×. EURUSD: 5d/1.25×. AUDNZD: 7d/1.25×. GBPJPY: 5d/1.0×. USDCAD/USDJPY: 60d/1.25× (no short-hold benefit confirmed).
**Health:** 🟢 ACTIVE

---

### FOREX-LESSON 2 — Counter-momentum entries outperform aligned entries 3:1. True for both equity and forex.

*(Inherited from universal L-006, cross-system validation)*

**Evidence:** Counter (<-0.2% 5d momentum): +0.331R avg. Aligned (>+0.2%): +0.107R avg. Same 52% WR. 3× R differential. Equity validated first (UNH -0.9% → +11.53%). Forex applying same principle.
**Rule:** `5d_momentum < -0.002 → size ×1.25. 5d_momentum > +0.002 → size ×0.75.`
**Code location:** `sovereign/forex/signal_engine.py` `_compute_size_multipliers()`
**Health:** 🟢 ACTIVE

---

### FOREX-LESSON 3 — VIX term structure is a carry regime thermometer.

*(Inherited from universal L-007)*

**Evidence:** VIX slope [0,1): 75% WR, +0.697R. IC=-0.095, p=0.008.
**Rule:** `VIX_slope > 3.0 → size ×0.85. VIX_slope < 0.0 → size ×0.90.`
**Code location:** `sovereign/forex/signal_engine.py` `_compute_size_multipliers()`
**Health:** 🟢 ACTIVE

---

## Research Queue (Forex-specific)

**The 59 COMMITMENT_FAILURE mystery:**
59 losses are macro-aligned but fail. No single latent feature found (IC max 0.095).
Next experiment (lookahead bias allowed):
For those 59 losses + matched 59 winners, add features:
- Did price sweep a key level in prior 24h?
- Was there a CB event or news release in the 24h window?
- Did the correlated pair (e.g., EURUSD for GBPUSD) move in same direction first?
- What was intraday ATR vs daily ATR? (compressed intraday = no commitment)
Target: one feature shows > 65% separation. That's the latent factor.

**USDCAD retirement decision:**
USDCAD Sharpe: 0.326. Consistently weakest pair.
Oracle task: in 30 days, re-validate USDCAD. If Sharpe < 0.30 on recent 3 months,
recommend retirement. Removing it raises portfolio avg Sharpe mechanically.
