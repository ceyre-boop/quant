# ALTA METHOD — Live Trading Protocol
## Version 1.0 | Derived strictly from advice.md + confirmed research
*"The discipline being the system." — Seykota*

---

## RULE 0: THE PRIME DIRECTIVE

You are not trying to predict. You are waiting for confirmation that the move is already happening, then entering behind it with a defined loss and a time horizon the Oracle measured.

Every rule below comes from a trader who proved it over decades. None of them are optional. The ones who violated their own rules went broke.

---

## THE METHOD (5 steps, in order)

### STEP 1 — PRE-TRADE FILTER (before touching size)

All five must pass or the trade does not happen. This is Seykota's discipline and Hite's machine combined.

| Gate | Condition | Source |
|------|-----------|--------|
| **RATE DIRECTION** | Real rate differential sign must match trade direction | C-001: avg -0.84R when broken, 189 trades |
| **RATE MAGNITUDE** | \|real_rate_diff\| ≥ 0.5% | C-005: edge disappears below this |
| **MOMENTUM** | 63-day momentum must not oppose direction by >1% | C-003: avg -0.91R when broken, 61 trades |
| **VOLATILITY** | ATR_14d_pct ≥ 0.6% | C-006: compressed = cannot reach target |
| **CALENDAR** | No CB decision within 3 days (entry or exit window) | HYP-061 CONFIRMED, CB-blackout gate |

If any gate fails: **do not enter. Log the veto. Move on.**
"There is a time to go long, a time to go short, and a time to go fishing." — Livermore

---

### STEP 2 — ENTRY CONFIRMATION (two required, not one)

Lipschutz rule: most setups don't pass. That is the design.

**Confirmation 1 — Macro says it should happen:**
- Rate differential ≥ 1.0% favoring direction (not just sign — magnitude)
- Confirmation comes from last CB decision (post-CB drift window active, or carry regime established)

**Confirmation 2 — Price says it is already happening:**
- Price has moved in the thesis direction since the catalyst event
- Not a prediction of future movement. Proof of present movement.

Both or nothing. "Being too early is indistinguishable from being wrong." — Alta Tenet 4

---

### STEP 3 — SIZE (before the trade is on, not after)

Kovner's rule: set the stop BEFORE entry. Know the loss before you know if you're right.

**Base size:**
- Risk 0.5% of account per trade (base_risk_pct: 0.005)
- Apply 0.5× Kovner haircut → effective risk = 0.25% per trade
- "Cut intended position size at least in half." — Kovner

**Size UP only when all three are true:**
- \|real_rate_diff\| ≥ 2.5% (B-003 active)
- Momentum confirms (B-001 active)
- spike_prob > 0.85 in the conviction pipeline

**Size DOWN when:**
- 63-day momentum < 0.5% AND ATR < 0.7% → 0.5× (C-002)
- Library sim > 0.90 → Kelly cap 2% (defense mode active, ASIAN_CURRENCY_CONTAGION match)
- More than 2 correlated pairs open → reduce each by 0.5× (Kovner correlation rule)

**Stop placement:**
- ATR × 1.5 from entry (ICT window parameters)
- Hard stop. Not a mental stop. Not a trailing stop at entry. A placed order.
- "The stop is not optional once the trade is on." — PTJ

**Target:**
- Time exit: 60 days default (Oracle-measured drift period)
- No trailing stop on macro carries — this is the Type1 TS finding (HYP-108 research)
- Exit at day 60 at market, or at close of CB decision window, whichever comes first

---

### STEP 4 — HOLD (this is where most people fail)

Livermore's insight: "It never was my thinking that made the big money. It always was my sitting tight."

Rules during the hold:
1. **Do not shorten the hold because the trade "looks stuck."** Stuck is not a signal.
2. **Do not move the stop wider after entry.** The stop was placed with full information. Post-entry information is noise relative to the structural edge.
3. **Do not add to a losing position.** Adding to a winner is allowed if macro confirmation strengthens (scaled-in only, not impulse).
4. **Log every day the trade is open.** Not every hour. Once per day: note price, note if any gate has changed status, note emotional state (PTJ: "assume every position is wrong").

---

### STEP 5 — EXIT AND LOG

Exit conditions (first to trigger wins):
- Day 60: exit at market (time exit)
- Stop hit: exit immediately, no adjustment
- CB decision fires and thesis is invalidated: exit next session open
- CB-blackout window opens (3 days pre-CB): close or hedge, per HYP-061

**Exit log is mandatory.** Without update_outcome(), the Oracle cannot learn. Skipping this is silent data loss. The machine degrades.

---

## WHAT YOU ARE LOGGING (Trade Log Template)

Every trade generates one log. Here is the schema — fill it fully, not partially.

### PRE-TRADE LOG (at entry, before size is placed)

```
DATE: 
PAIR:
DIRECTION: LONG / SHORT
THESIS: [one sentence — what do you believe will happen and why]

GATE CHECK:
  C-001 rate direction:    PASS / FAIL | rate_diff = ____%
  C-005 rate magnitude:    PASS / FAIL | |rate_diff| = ____%  
  C-003 momentum:          PASS / FAIL | 63d momentum = ____%
  C-006 volatility:        PASS / FAIL | ATR_14d = ____%
  CALENDAR (CB blackout):  PASS / FAIL | next CB = ____

CONFIRMATION 1 (macro): [describe the rate divergence evidence]
CONFIRMATION 2 (price):  [describe the price confirmation — where did price already move]

SIZE:
  Account: $____
  Base risk: 0.25% = $____
  Size modifier: BASE / UP / DOWN | reason: ____
  Units: ____
  Stop: ____ (ATR × 1.5 = ____)
  Target: Day ____ (date: ____)

WHAT WOULD MAKE ME WRONG:
  [one condition that would invalidate the thesis — be specific]

PRE-TRADE EMOTIONAL STATE: [calm / anxious / excited / FOMO — honest]
```

### DAILY HOLD LOG (one per day, brief)

```
DATE:
DAY # of trade:
PRICE: entry ____ | current ____ | stop ____ | distance to stop ____
GATES STILL PASSING: YES / NO — if NO, which failed and why
THESIS INTACT: YES / NO
NOTABLE MARKET EVENT TODAY: [or "nothing material"]
EMOTIONAL STATE: [one word]
ACTION TAKEN: HOLD / ADJUSTED STOP (log reason) / SCALED IN (log reason)
```

### EXIT LOG (at close)

```
DATE CLOSED:
HOLD DAYS:
EXIT REASON: TIME / STOP / CB-INVALIDATION / OTHER
EXIT PRICE:
PnL%:
R-MULTIPLE: ____R (actual / initial risk)
SWAP COST: ____% (real OANDA rate × hold days)
NET R (post swap): ____R

GATE COMPLIANCE: All gates followed throughout? YES / NO
  If NO: which rule was broken? When? What was the outcome?

THESIS OUTCOME: CORRECT / INCORRECT / MIXED
  What actually happened vs what you predicted:

WHAT WOULD HAVE IMPROVED THIS TRADE:

WHAT SEYKOTA/LIVERMORE/PTJ/LIPSCHUTZ WOULD SAY ABOUT HOW YOU TRADED THIS:

ORACLE UPDATE: update_outcome() called? YES / NO
```

---

## META-ANALYSIS (after 5–7 completed trade logs)

This is your course-correction cycle. Run it after every 5–7 closed trades. It feeds the ML fitting cycle.

### SECTION 1: RULE COMPLIANCE AUDIT

For each of the 5–7 trades:
```
Trade | C-001 | C-003 | C-002 | C-006 | C-005 | Calendar | Both Confirms | Stop Placed | Exit Logged
  1   |       |       |       |       |       |          |               |             |
  2   |       |       |       |       |       |          |               |             |
...
```
Pass = ✓, Fail = ✗, N/A = —

Calculate:
- Rule compliance rate: ___% (# ✓ / total possible)
- Trades with full compliance: n=___
- Trades with at least one rule broken: n=___

### SECTION 2: PERFORMANCE SPLIT BY COMPLIANCE

```
FULLY COMPLIANT TRADES:
  n = ___
  Win rate: ___%
  Mean R: ____
  Best R: ____ | Worst R: ____

TRADES WITH RULE BROKEN:
  n = ___
  Win rate: ___%
  Mean R: ____
  Which rule was broken most often: ____
```

This is the critical table. If compliant trades outperform broken-rule trades, the rules work. If not, a rule needs examination.

### SECTION 3: EXIT QUALITY REVIEW

```
Time exits: n=___ | mean R: ____
Stop exits: n=___ | mean R: ____
CB-invalidation exits: n=___ | mean R: ____

Were any stops moved after entry? YES / NO
Were any holds shortened without a gate firing? YES / NO
Were any size rules violated? YES / NO
```

### SECTION 4: GATE FAILURE ANALYSIS

Which gates fired (caused a SKIP) vs which were bypassed?

```
Gate       | Times fired | Times bypassed (entry taken anyway) | Cost of bypass (R)
C-001      |             |                                      |
C-003      |             |                                      |
C-005      |             |                                      |
C-006      |             |                                      |
CALENDAR   |             |                                      |
CONFIRM-2  |             |                                      |
```

### SECTION 5: THE TRADER VERDICT

Answer each question in one sentence:

1. **Livermore:** Did you sit tight when the trade was intact? Where did you fidget?
2. **PTJ:** Did you assume every position was wrong daily? Did you have evidence your stop was right?
3. **Seykota:** Did the system run itself, or did you override it? Where?
4. **Lipschutz:** Did the setup have 3:1+ R:R before entry? Did you sit on your hands enough?
5. **Kovner:** Did you know your loss before you knew your entry? Was correlation checked?
6. **Marks:** Did you assess what was already priced in? Were you trading consensus or divergence?

### SECTION 6: ML COURSE CORRECTION (feeds the Oracle)

Based on the above, write **one parameter to test next cycle.** Not five — one.

```
PROPOSED ADJUSTMENT:
  What: [e.g., "raise minimum |rate_diff| threshold from 0.5% to 0.75%"]
  Why: [data from this review — cite the trades]
  Hypothesis: [what outcome would change and by how much]
  Test: [how to validate before applying to live system]
  Preregister as: HYP-___ before touching config
```

---

## THE VETO LEDGER (ongoing)

Every time the gates CORRECTLY prevent a bad trade, log it here.

```
DATE | PAIR | DIRECTION | GATE THAT FIRED | WHAT WOULD HAVE HAPPENED
```

"The best sessions have always been the ones where Sovereign said NO more than it said YES." — Alta advice.md

The veto ledger is not a failure record. It is a wins record.

---

## NON-NEGOTIABLES (from the traders, directly)

1. **Stop is placed before entry. Never adjusted wider after entry.** (PTJ + Kovner)
2. **No averaging down.** (Livermore — this is how he went bankrupt, four times)
3. **No trades without both confirmations.** (Alta Confirmation Protocol)
4. **No trades during CB-blackout window.** (HYP-061 CONFIRMED)
5. **Exit log is mandatory.** (Oracle cannot learn without it)
6. **Veto logged as a win.** (Ellis — avoiding errors is the game)
7. **After 3 consecutive losses: size to 0.5×.** (Hite volatility-suspend principle)
8. **After 5 consecutive losses: stop trading, review the meta-analysis.** (PTJ defense first)

---

*Alta Investments — Sovereign Trading Intelligence*
*Method v1.0 | Derived from: advice.md + confirmed backtests + combat_rules.json*
*"The market didn't beat me. I beat myself." — Livermore*
*The rules exist so that line is never true.*
