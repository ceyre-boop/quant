# I Am A Good Equity Trader
## Alta Investments — Sovereign Equity Wisdom

*System personality: Mean reversion engine in a strong fundamental universe.*
*Buys what's temporarily weak. Sizes up on pullbacks. Exits on momentum exhaustion.*
*Maximum 10 active lessons. Inherits universal principles from I_am_a_good_trader.md.*

**Active lessons: 2 / 10**
**Last updated: 2026-05-20**
**Universe: META, UNH, AMD, BAC, JPM, SPG, PFE**
**May 6-20 simulation: +2.35% (sized) vs SPY +1.38% — first clean baseline**

---

### EQUITY-LESSON 1 — Counter-momentum sizing: the system's confirmed personality.

*(Inherited from universal L-006, equity discovered first)*

**Discovered:** 2026-05-19 (latent feature search)
**Live validation:** 2026-05-20 (May 6-20 simulation — UNH case)
**Evidence:** UNH -0.9% 5d momentum → 1.25× size → +11.53% in 5 days. AMD +25% already → 0.75× size → +4.29% (system correctly de-emphasized). Simulation avg: +2.35% vs SPY +1.38%.
**Rule:** `5d_momentum < -0.002 → size ×1.25. 5d_momentum > +0.002 → size ×0.75.`
**Impact:** UNH case alone justified the rule — pulling back in a strong fundamental universe = entering at better price, not entering a trend reversal.
**Code location:** `sovereign/forex/signal_engine.py` → Equity orchestrator (wiring pending)
**Health:** 🟢 ACTIVE

---

### EQUITY-LESSON 2 — System reliability is the weakest link. Silent failures cost more than bad trades.

**Discovered:** 2026-05-20 (audit — system dead 18 days on one-line bug)
**Validated:** 2026-05-20 (pegasus_params NameError blocked every ticker silently since May 2)
**Evidence:** pegasus_params missing from SovereignRiskEngine.compute() signature. Every ticker errored before any signal computed. Scanner appeared healthy (exit code 0). 0 trades from May 2-20.
**Rule:** Health check must verify signal generation, not just process execution. 9:30 AM daily gate: if no signal attempt logged in past session → URGENT alert.
**Impact:** +2.35% missed over 14 days. Cost: 1 Python parameter mismatch.
**Code location:** `sovereign/risk/kelly_engine.py` — fixed. Health monitor: PENDING (build sovereign/equity/health_monitor.py).
**Health:** 🟢 ACTIVE (fix applied)

---

## Research Queue (Equity-specific)

**Feature importance study (lookahead bias allowed — finding X):**
Load 1 year of META, UNH, AMD, BAC, JPM daily data.
For every day compute: 5d_momentum, 10d_momentum, 20d_momentum,
RSI_14, distance_from_50SMA, distance_from_200SMA, earnings_proximity.
Target: actual_next_10d_return.
Run feature importance. Find which combination of "weakness signals" most
reliably predicts a bounce. Then remove lookahead and validate properly.

**News intelligence (Phase 1 — data collection in progress):**
Started: 2026-05-20. Daily Reddit sentiment snapshots archiving to data/news/.
IC study requires 30+ daily snapshots (done: 1/30).
Run IC study when snapshots reach 30. If IC > 0.10: build live news pipeline.

**Health monitor (build next session):**
sovereign/equity/health_monitor.py
9:30 AM daily: verify scanner alive, ledger writing, Alpaca connected.
Immediate alert if any fails.
"Best system dead for 2 weeks" cannot happen again.
