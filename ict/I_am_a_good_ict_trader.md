# I Am A Good ICT Trader
## Alta Investments — ICT London+GradeA Wisdom

*System personality: Precision surgery in high-participation windows.*
*This system only operates in good lighting. London session. Grade A setups. Market committed.*
*Maximum 10 active lessons. Inherits universal principles from I_am_a_good_trader.md.*

**Active lessons: 3 / 10**
**System Sharpe: Sharpe proxy 2.093 (London+GradeA+committed)**
**Last updated: 2026-05-20**
**pd_alignment fix impact: WR 31.4% → 41.9%, avgR +0.160 → +0.686 (2026-05-20)**

---

### ICT-LESSON 1 — Grade A setups AFTER pd_alignment is excluded are the system's real edge.

**Discovered:** 2026-05-19 (HYP-024 forensics)
**Validated:** 2026-05-20 (fresh backtest with pd_alignment=0.0)
**Evidence:** Before: WR=31.4%, avgR=+0.160. After: WR=41.9%, avgR=+0.686. Delta WR: +10.5pp. Delta avgR: +0.526.
**Rule:** `pd_alignment weight = 0.0`. Premium/discount zone describes retail positioning traps, not where institutions actually enter.
**Impact:** The scoring system now only rewards: kill zone, sweep confirmation, displacement quality, FVG quality, market structure. These five components describe actual institutional behavior.
**Code location:** `ict/pipeline.py` `_DEFAULT_WEIGHTS['pd_alignment'] = 0.0`
**Health:** 🟢 ACTIVE
**Last validated:** 2026-05-20

*The mechanism:* ICT theory says price in premium = short, price in discount = long. Institutions don't follow their own textbook at retail timing. They accumulate on aggressive entries (outside the zone) and distribute on aggressive exits. The pd_alignment component was penalizing the entries that actually work. Removing it is not removing a component — it's removing a trap.

---

### ICT-LESSON 2 — NY_PM is not a weaker London. It's a different system with opposite edge.

*(Inherited from universal L-002 — repeated here for system-specific context)*

**Evidence:** London: +0.471R avg. NY_PM: -0.283R avg. 102-trade sample. Effect size: 0.754R/trade.
**Rule:** London session only. NY_PM unconditional veto. No exceptions.
**Code location:** `ict/pipeline.py` Stage 5.7 — first gate in evaluate()
**Health:** 🟢 ACTIVE

---

### ICT-LESSON 3 — A+ paradox: the scoring system was selecting for overconfirmed setups.

*(Inherited from universal L-003 — repeated here for system-specific context)*

**Evidence post pd_alignment fix:** A+: 2 trades, WR=50%, avgR=+1.750 (previously: 30 trades, 13% WR). The fix dramatically reduced A+ fires — only genuinely exceptional setups now hit A+.
**Rule:** `if grade == 'A+': treat as 'A' for trade decision`. Grade override prevents A+ from bypassing A-level thresholds.
**Code location:** `ict/pipeline.py` Stage 5.7 — `_effective_grade` override
**Health:** 🟢 ACTIVE

---

## Research Queue (ICT-specific)

**Next experiment (lookahead bias allowed — finding X):**
Load 43 post-fix backtest trades. Within Grade A, add features:
- Time within London session (8am vs 10am vs 11am UTC)
- ATR vs 20-day ATR at entry (expanded vs compressed)
- CB event proximity (days until next BOE/ECB meeting)
- GBPUSD/EURUSD correlation on entry day
- Did price sweep a key level in prior 3 bars? (manipulation complete?)

Target: find the sub-Grade-A condition with WR > 50%.
That becomes Grade A+ (data-driven, not theory-driven).

**Next confirmed fix queued:**
HYP-024 application: weight already zeroed. Run full backtest validation.
Status: ✅ DONE — pd_alignment=0.0 in production, WR confirmed +10.5pp.
