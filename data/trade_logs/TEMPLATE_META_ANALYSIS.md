# META-ANALYSIS — 5-7 Trade Review
*Run after every 5-7 closed trade logs. This feeds the ML fitting cycle.*

## CYCLE INFO

| Field | Value |
|-------|-------|
| Cycle # | |
| Trades reviewed | ___ through ___ |
| Date range | |
| Total trades | |
| Analyst | Colin |

---

## SECTION 1: RULE COMPLIANCE AUDIT

| Trade | C-001 | C-005 | C-003 | C-006 | Calendar | Confirm-1 | Confirm-2 | Stop Placed | Exit Logged |
|-------|-------|-------|-------|-------|----------|-----------|-----------|-------------|-------------|
| 1 | | | | | | | | | |
| 2 | | | | | | | | | |
| 3 | | | | | | | | | |
| 4 | | | | | | | | | |
| 5 | | | | | | | | | |
| 6 | | | | | | | | | |
| 7 | | | | | | | | | |

✓ = followed | ✗ = broken | — = N/A

**Summary:**
- Overall compliance rate: ___% (✓ count / total possible)
- Fully compliant trades: n=___
- Trades with ≥1 rule broken: n=___
- Most frequently broken rule: ___

---

## SECTION 2: PERFORMANCE BY COMPLIANCE

| Group | n | Win Rate | Mean R | Best R | Worst R |
|-------|---|----------|--------|--------|---------|
| Fully compliant | | | | | |
| Rule broken | | | | | |
| **Difference** | | | | | |

**Interpretation:**
___

---

## SECTION 3: EXIT QUALITY

| Exit Type | n | Mean R | Notes |
|-----------|---|--------|-------|
| Time exit (day 60) | | | |
| Stop exit | | | |
| CB-invalidation exit | | | |
| Other | | | |

**Exit violations this cycle:**
- Stops moved wider after entry: YES / NO — n=___
- Holds shortened without gate firing: YES / NO — n=___
- Size rules violated: YES / NO — n=___

---

## SECTION 4: GATE VETO LEDGER

*(Trades the gates correctly prevented — these are WINS)*

| Date | Pair | Direction | Gate Fired | What Would Have Happened |
|------|------|-----------|------------|--------------------------|
| | | | | |
| | | | | |

**Veto count this cycle: ___**
**Estimated R saved by vetoes: ____R**

---

## SECTION 5: THE TRADER VERDICT (for this cycle as a whole)

Answer in one honest sentence each:

**Livermore** (did I sit tight?):
___

**PTJ** (defense first — did I assume every position was wrong daily?):
___

**Seykota** (system ran itself, or I overrode it?):
___

**Lipschutz** (did setups have 3:1+ R:R? did I sit on my hands?):
___

**Kovner** (did I know my loss before my entry? was correlation checked?):
___

**Marks** (was I trading consensus or divergence? what was already priced in?):
___

---

## SECTION 6: ML COURSE CORRECTION

*(One proposed adjustment per cycle — no more)*

**Proposed adjustment:**
What: ___
Why: [cite specific trades from this cycle]
Expected outcome: ___
How to validate: ___

**Preregister before touching config:**
Proposed HYP-ID: ___
Hypothesis ledger entry drafted: YES / NO

**DO NOT touch config/parameters.yml until this is preregistered and back-tested.**

---

## SECTION 7: CARRY STATE

*(The carry base never stops — log its status)*

| Pair | Direction | Hold days | Net R to date | Swap (daily) |
|------|-----------|-----------|---------------|--------------|
| | | | | |
| | | | | |

Carry contributing positively: YES / NO
Carry regime intact: YES / NO (rate differentials still present)

---

## NEXT CYCLE SETUP

- [ ] Oracle update_outcome() called for all closed trades
- [ ] Hypothesis ledger updated with cycle outcomes
- [ ] Any new CONFIRMED edges added to config (only after BH + permutation)
- [ ] NEXT.md updated
- [ ] Git pushed

---
*Alta Investments — Sovereign Trading Intelligence*
*"If most traders would learn to sit on their hands 50 percent of the time, they would make a lot more money." — Lipschutz*
