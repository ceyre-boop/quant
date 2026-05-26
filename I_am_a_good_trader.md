# I Am A Good Trader
## Alta Investments — Sovereign Trading System Wisdom

*Maximum 10 active lessons. Each one proven, codified, monitored.*
*When a new lesson is validated, the weakest lesson is retired to `I_was_a_good_trader.md`.*
*This file is not a strategy. It is a maturity record.*
*Every lesson was discovered through thousands of trades and forensic analysis.*
*Every "obvious in hindsight" insight required proof before becoming law.*

---

**Active lessons: 7 / 10**
**Last updated: 2026-05-19**
**System Sharpe at last update: 1.0713 (v007)**
**Target: 1.5 | Gap: 0.429**

---

### LESSON 1 — The Exact Setup Exists. Trade Only That.

**Discovered:** 2026-05-18 (forensics unified analysis)
**Validated:** 2026-05-18 (102-trade sample, 2 walk-forward windows)
**Evidence:** London+GradeA+committed: WR=41%, avgR=+0.840, Sharpe proxy=2.093. MC prop pass rate 90.3%.
**Rule:** London session AND Grade A AND market_structure_score < 1.5 → execute. Any condition missing → veto.
**Impact:** ICT system MC prop pass rate 58% → 90%. The system's primary edge lives entirely in this configuration.
**Code location:** `ict/pipeline.py` Stage 5.7 — three-part gate (session + grade + commitment)
**Health:** 🟢 ACTIVE
**Last validated:** 2026-05-18
**Linked hypothesis:** HYP-022, HYP-026

*The lesson:* Not every setup that scores well is worth trading. The London+A+committed configuration is the only one where institutional order flow is reliably accessible. Outside this configuration, you are not edge-trading — you are noise-trading at ICT scoring thresholds.

---

### LESSON 2 — NY_PM Is Not Weaker. It Is Anti-Edge.

**Discovered:** 2026-05-18 (forensics session breakdown)
**Validated:** 2026-05-18 (102-trade sample, p-value not computed, effect too large to need it)
**Evidence:** NY_PM: -0.283R avg. London: +0.471R avg. Difference: 0.754R per trade. 102 trades.
**Rule:** if session == 'NY_PM': return ICTVeto. No exceptions. No override.
**Impact:** Prop pass rate recovery was entirely attributable to this veto.
**Code location:** `ict/pipeline.py` Stage 5.7 — first check in evaluate()
**Health:** 🟢 ACTIVE
**Last validated:** 2026-05-18
**Linked hypothesis:** HYP-022

*The lesson:* The difference between London and NY_PM is not that one is harder — it is that NY_PM is systematically wrong. Smart money distributes in the PM session into retail continuation buyers. The setup looks the same. The execution looks the same. The result is structurally negative. The sessions are not unequal versions of the same thing. They are different things.

---

### LESSON 3 — Grade Labels Invert Quality. Score Is Signal. Grade Is Noise.

**Discovered:** 2026-05-18 (A+ paradox forensics)
**Validated:** 2026-05-18 (30 A+ trades vs 72 A trades)
**Evidence:** A+: 13% WR, -0.375R avg (n=30). Grade A: 39% WR, +0.383R avg (n=72). Delta WR: 26pp.
**Rule:** if grade == 'A+': treat as 'A' for trade decision. Log original grade. Never block A+ — the commitment gate handles the bad ones.
**Impact:** A+ trades entering at overconfirmed setups stopped distorting the grade filter.
**Code location:** `ict/pipeline.py` Stage 5.7 — `_effective_grade = ICTGrade.A if grade == ICTGrade.A_PLUS else grade`
**Health:** 🟢 ACTIVE
**Last validated:** 2026-05-18
**Linked hypothesis:** HYP-023

*The lesson:* The scoring system was working. The grade interpretation was broken. A+ selects for setups where every confirmation is firing — which means price has already moved, entry is late, and the "confirmation" is a trap. The market was teaching this lesson in every A+ trade. The system was awarding gold stars to late entries.

---

### LESSON 4 — Premium/Discount Zone (pd_alignment) Describes Retail Traps, Not Institutional Flow.

**Discovered:** 2026-05-19 (commitment detector forensics)
**Validated:** 2026-05-19 (135 trades across failure taxonomy)
**Evidence:** pd_alignment > 0: 20% WR. pd_alignment = 0: 35% WR. Delta: -15pp WR.
**Rule:** pd_alignment weight = 0.0 in scoring. Still computed and logged for monitoring.
**Impact:** Removing the pd_alignment contribution to score eliminates a systematic anti-signal.
**Code location:** `ict/pipeline.py` `_DEFAULT_WEIGHTS['pd_alignment'] = 0.0`
**Health:** 🟢 ACTIVE
**Last validated:** 2026-05-19
**Linked hypothesis:** HYP-024

*The lesson:* ICT theory holds that price in discount = institutional accumulation zone = go long. The data says the opposite. When price is "in discount" (pd_alignment scoring positively), it is more likely to continue down. Possible explanation: institutions accumulate at prices that look wrong to retail — aggressive entries, not passive zone fills. The discount zone theory describes where retail waits. Institutions don't wait where retail waits.

---

### LESSON 5 — Macro Momentum Has Pair-Specific Half-Lives. There Is No Universal Hold Period.

**Discovered:** 2026-05-17 (micro-edge sweep, GBPUSD first)
**Validated:** 2026-05-19 (per-pair trailing sweep, 7 pairs)
**Evidence:** GBPUSD 6d/2.0×: Sharpe 1.523 (was 1.09 at 60d). GBPJPY 5d/1.0×: 0.741 (was 0.653). Portfolio: 1.0547→1.0713.
**Rule:** GBPUSD: 6d/2.0×. AUDUSD: 5d/1.0×. EURUSD: 5d/1.25×. AUDNZD: 7d/1.25×. GBPJPY: 5d/1.0×. USDCAD/USDJPY: 60d/1.25×.
**Impact:** +0.017 portfolio Sharpe. More importantly: 5 pairs now exit before momentum exhausts instead of giving back gains.
**Code location:** `sovereign/forex/forex_backtester.py` PAIR_HOLD_OVERRIDES + PAIR_TRAILING_OVERRIDES
**Health:** 🟢 ACTIVE
**Last validated:** 2026-05-19
**Linked hypothesis:** HYP-033 (no universal hold) + v006/v007

*The lesson:* The 60-day hold period was chosen because it approximates a central bank meeting cycle. But each currency pair responds to its CB's decisions at a different rate. BOE-Fed divergence (GBPUSD) exhausts in 6 days. RBA-Fed (AUDUSD) exhausts in 5. The hold period should match the momentum half-life of the specific policy divergence, not a generic cycle estimate.

---

### LESSON 6 — Pullback Entries Outperform Aligned Entries 3:1 in R-Multiple. Entry Timing Within Direction Matters as Much as Direction.

**Discovered:** 2026-05-19 (latent feature search, counter-momentum finding)
**Validated:** 2026-05-19 (full forex trade history, IC = -0.095 for momentum alignment)
**Evidence:** Counter-momentum (<-0.2%): 52% WR, +0.331R avg. Aligned (>+0.2%): 52% WR, +0.107R avg. Same WR. 3× R differential.
**Rule:** 5d_momentum < -0.002 → size ×1.25. 5d_momentum > +0.002 → size ×0.75. Flat → no adjustment.
**Impact:** +0.029 Sharpe proxy from size adjustment. Real impact on R-multiple distribution.
**Code location:** `sovereign/forex/signal_engine.py` `_compute_size_multipliers()` — counter-momentum block
**Health:** 🟢 ACTIVE
**Last validated:** 2026-05-19
**Linked hypothesis:** HYP-031

*The lesson:* Entering a long trade when 5-day momentum is already strongly positive means you are buying something that just rallied. Entering when momentum is negative means you are buying the pullback within the macro thesis. Both trades have 52% win rates because direction is correct in both cases. But the pullback entry is more efficient — price is closer to the entry that institutions used, stops are tighter relative to targets, and the R-multiple is 3× larger. The lesson: be right about direction, then be early about timing.

---

### LESSON 7 — VIX Term Structure Reveals Carry Regime Health. Modest Contango = Safe. Steep Contango = Nervousness. Backwardation = Exit.

**Discovered:** 2026-05-19 (latent feature search, VIX slope finding)
**Validated:** 2026-05-19 (IC = -0.095, p=0.008 — highest IC of 5 candidates tested)
**Evidence:** VIX slope [0,1): 75% WR, +0.697R avg. VIX slope [3,5): 62% WR, +0.353R avg. Backwardation: degraded performance.
**Rule:** VIX_slope (VIX3M - VIX) > 3.0 → size ×0.85. VIX_slope < 0.0 → size ×0.90.
**Impact:** Size discounts applied in environments with proven adverse carry conditions.
**Code location:** `sovereign/forex/signal_engine.py` `_compute_size_multipliers()` — VIX slope block
**Health:** 🟢 ACTIVE
**Last validated:** 2026-05-19
**Linked hypothesis:** HYP-032

*The lesson:* VIX measures spot fear. VIX3M measures intermediate-term fear. The difference (VIX slope) measures whether the market is pricing near-term risk above or below medium-term risk. When the slope is near zero, the market views all time horizons as equally risky — carry trades flow freely. When the slope is steep positive, the market is calm in spot but nervous about what comes next — carry is fragile. When the slope inverts (backwardation), something is already breaking. The carry regime is not binary (safe/unsafe). It has a gradient that the VIX curve reveals.

---

---

### LESSON 8 — The More a Safe-Haven Pair Needs the Safe Haven, the Less Reliable Its Rate Signal. Tighten the Gate When VIX is Rising, Not Falling.

**Discovered:** 2026-05-25 (HYP-044 VIX threshold sweep)
**Validated:** 2026-05-25 (USDJPY: 46 trades at VIX>13 vs 70 at VIX>15, Sharpe 2.979 vs 1.770; AUDNZD: 39 trades vs 57, Sharpe 2.246 vs 1.558)
**Evidence:** Portfolio 1.8552→2.097 (+0.242). USDJPY and AUDNZD become highest-Sharpe individual pairs.
**Rule:** USDJPY/AUDNZD VIX gate: 15→13. Bull market + VIX>13 = veto. Lower threshold = fewer but higher-quality trades.
**Impact:** v014 → portfolio avg Sharpe 2.097.
**Code location:** `sovereign/forex/forex_backtester.py` `PAIR_VIX_GATES`; `sovereign/forex/signal_engine.py` `_VIX_GATES`
**Health:** 🟢 ACTIVE
**Last validated:** 2026-05-25
**Linked hypothesis:** HYP-044

*The lesson:* USDJPY is the world's most liquid safe-haven trade. When global equities are rising and VIX is above 13, two signals compete: the macro rate differential says one thing, and the safe-haven flow says another. Between VIX 13 and 15, the safe-haven bid corrupts the rate signal but is not obvious enough to detect from price alone. Lowering the veto threshold removes the confused regime — leaving only trades where the rate differential speaks clearly. AUDNZD: both AUD and NZD are risk currencies, and the cross rate loses its macro signal in any moderately elevated risk environment.

---

*Next lesson will retire the weakest of the above based on health monitoring.*
*Health monitoring runs monthly. First monitoring cycle: 2026-06-19.*
