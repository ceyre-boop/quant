# ICT Swing Trade Decision Engine ⚙️

#ICT #swing-trading #entry-exit #risk-management #decision-engine #playbook

> **Purpose:** A complete if-then decision engine for professional swing trading using ICT concepts. Every scenario has a rule. No discretion without a framework. No exceptions without a reason.

---

## 🗺️ Map of Contents

- [[#The Non-Negotiable Pre-Conditions]]
- [[#Phase 1 — Market Structure Analysis (Top-Down)]]
- [[#Phase 2 — Kill Zone & Session Timing]]
- [[#Phase 3 — The Setup Checklist (Entry Qualification)]]
- [[#Phase 4 — Refined Entry Logic (ICT Precision)]]
- [[#Phase 5 — Stop Loss Placement]]
- [[#Phase 6 — Position Sizing]]
- [[#Phase 7 — Target Logic & Partial Exits]]
- [[#Phase 8 — Trade Management Tree]]
- [[#Phase 9 — Exit Decision Tree]]
- [[#Phase 10 — Scenario Playbook (All Situations)]]
- [[#Phase 11 — Post-Entry Rules]]
- [[#Phase 12 — The Hard Rules (Never Break)]]
- [[#Quick-Reference Decision Flowchart]]

---

## The Non-Negotiable Pre-Conditions

> Before any analysis begins — run this gate check. If ANY answer is NO → **do not trade today.**

```
☐ Am I emotionally neutral? (not angry, euphoric, anxious, tired, distracted)
☐ Did I sleep adequately?
☐ Is my screen and platform working correctly?
☐ Do I know today's high-impact news events? (FOMC, CPI, NFP, earnings)
☐ Have I reviewed the higher time frame bias this week?
☐ Is my max daily loss limit still intact? (not already hit)
☐ Do I have a clear, pre-planned watchlist?
```

**IF** any box is unchecked → **WAIT. No trade is better than a bad trade.**

**IF** a high-impact news event is within 2 hours → **DO NOT enter new positions.** Wait for the impulse candle to close, then re-evaluate.

**IF** it is the first 15 minutes of the New York open (9:30–9:45 ET) → **OBSERVE ONLY.** ICT calls this the "Judas Swing" window — fake moves are common.

---

## Phase 1 — Market Structure Analysis (Top-Down)

> Always work from Weekly → Daily → 4H → 1H. The higher frame is LAW. Lower frames are permission slips.

### Step 1A — Weekly Chart (Bias Frame)

**IF** Weekly candle is printing Higher Highs + Higher Lows (HH/HL) →
- **Bias = BULLISH**
- Only look for LONG setups on daily/4H
- Short trades require extreme confluence + counter-trend awareness

**IF** Weekly is printing Lower Highs + Lower Lows (LH/LL) →
- **Bias = BEARISH**
- Only look for SHORT setups
- Long trades are counter-trend; skip unless at major Weekly demand

**IF** Weekly structure is unclear / inside bar / consolidation →
- **Bias = NEUTRAL**
- Reduce position size by 50%
- Only trade at extreme range edges with max confluence
- When in doubt, sit out

**Weekly Key Levels to Mark:**
- Previous Week High (PWH) → magnet / resistance
- Previous Week Low (PWL) → magnet / support
- Weekly Fair Value Gaps (FVG) → price seeks to fill
- Weekly Order Blocks (OB) → institutional entry zones
- Weekly Breaker Blocks → former OB that price broke through; now opposite

---

### Step 1B — Daily Chart (Setup Frame)

**IF** Daily trend aligns with Weekly bias →
- Full size trades permitted
- Look for Daily structure mitigation zones

**IF** Daily trend is opposite to Weekly bias →
- Reduce size to 50% max
- Only trade if at major Weekly level
- Treat as higher-risk counter-trend

**Daily Key Levels to Mark:**
- Previous Day High (PDH) → above = bullish continuation magnet
- Previous Day Low (PDL) → below = bearish continuation magnet
- Daily Fair Value Gaps (FVG) → gaps between candle wicks; price fills them
- Daily Order Blocks (OB) → last down candle before impulsive up move (bull OB) / last up candle before impulsive down move (bear OB)
- Daily Breaker Blocks → OB that has been violated; acts as opposite zone
- Daily Liquidity Pools → equal highs (BSL - Buy Side Liquidity) or equal lows (SSL - Sell Side Liquidity)
- NWOG / NDOG → New Week/Day Opening Gaps; price is drawn to fill

**ICT Liquidity Logic:**
```
IF price shows equal highs on daily chart →
  MARK as Buy-Side Liquidity (BSL) pool
  Smart money WILL hunt this level to fill orders
  THEN price reverses (stops are above those highs = fuel for a reversal)

IF price shows equal lows on daily chart →
  MARK as Sell-Side Liquidity (SSL) pool
  Smart money WILL hunt this level
  THEN price reverses (stops are below those lows = fuel for reversal)
```

**NEVER chase price into a liquidity pool.** Wait for the hunt + rejection.

---

### Step 1C — 4H Chart (Confirmation Frame)

**IF** 4H Market Structure Break (MSB) confirms Daily bias →
- Setup is high probability
- Full size

**IF** 4H MSB is against Daily bias →
- This is a retracement only
- Do not enter trend trades
- Wait for MSB in direction of Daily

**4H Key Concepts:**
- **Change of Character (CHOCH):** First break of structure against prevailing trend → signals possible reversal
- **Break of Structure (BOS):** Continuation of trend; add to conviction
- **4H FVG:** Smaller inefficiency; will be filled before major moves
- **4H OB:** Precise entry zone for swing trades

**Decision:**
```
IF Weekly = Bullish AND Daily = Bullish AND 4H = Bullish BOS → HIGH CONVICTION LONG
IF Weekly = Bullish AND Daily = Bullish AND 4H = CHOCH bearish → WAIT (pullback forming)
IF Weekly = Bullish AND Daily = CHOCH bearish → CAUTION (potential reversal) — reduce size
IF all three aligned BEARISH → HIGH CONVICTION SHORT
IF mixed signals on 2+ frames → SKIP TRADE
```

---

## Phase 2 — Kill Zone & Session Timing

> ICT teaches that smart money moves price during specific windows. Trading outside these windows is fighting the algorithm.

### Valid Trading Windows (ET)

| Session | Window | Best For |
|---|---|---|
| London Open Kill Zone | 2:00 AM – 5:00 AM ET | Forex + indices overnight reversals |
| New York Open Kill Zone | 7:00 AM – 10:00 AM ET | Best for swing entries; highest volume |
| New York Lunch | 12:00 PM – 1:00 PM ET | AVOID — low volume, choppy, fake moves |
| New York PM Session | 1:30 PM – 4:00 PM ET | Second best; continuation or reversal |
| Asia Session | 8:00 PM – 12:00 AM ET | Low volatility; range setting for next day |

**IF** you missed the Kill Zone entry → **DO NOT CHASE.**
**Wait for the next Kill Zone.** Chasing = buying tops / selling bottoms.

**IF** it is NY Lunch (12:00–1:30 PM) and you have no open trade → **do not open new positions.**
**IF** you have an open trade during NY Lunch → manage it per Phase 8 rules; do not add.

### Optimal Trade Entry (OTE) Time Windows
- Best swing entries historically form **between 7:00–10:00 AM ET**
- If setup forms outside this window — require **one extra confluence** before entry

---

## Phase 3 — The Setup Checklist (Entry Qualification)

> A trade requires a MINIMUM of 4 of the following 6 criteria. Elite setups have all 6.

```
[ ] 1. Higher Time Frame Bias aligned (Weekly + Daily pointing same direction)
[ ] 2. Institutional Order Flow (IOF) — price trading FROM an OB or FVG zone
[ ] 3. Liquidity was hunted first (price swept a high/low, then reversed)
[ ] 4. Market Structure Shift on entry time frame (1H CHOCH or BOS)
[ ] 5. Fair Value Gap (FVG) present on 15M or 1H that price is entering
[ ] 6. Kill Zone timing active (within valid session window)
```

**IF score = 6/6 → FULL SIZE trade (100% of planned position)**
**IF score = 4–5/6 → REDUCED SIZE (50–75% of planned position)**
**IF score = 3/6 or fewer → NO TRADE. Wait for better setup.**

---

## Phase 4 — Refined Entry Logic (ICT Precision)

### The ICT Setup Cascade

#### 4A — Order Block (OB) Entry Logic

**Bullish OB:**
- Last RED (bearish) candle before a strong impulsive move UP
- Price returns to this candle's body = potential long entry

```
IF price returns to Bullish OB zone:
  IF price shows bullish reaction candle (close above midpoint of OB) → ENTER LONG
  IF price wicks through OB but closes back inside → ENTER LONG (liquidity sweep + reclaim)
  IF price CLOSES below the low of the OB → OB is INVALIDATED → NO TRADE
    → Mark as Breaker Block (now bearish)
    → Wait for price to return to it from below for SHORT
```

**Bearish OB:**
- Last GREEN (bullish) candle before strong impulsive move DOWN
- Price returns to this candle's body = potential short entry

```
IF price returns to Bearish OB zone:
  IF price shows bearish reaction candle (close below midpoint of OB) → ENTER SHORT
  IF price wicks above OB but closes back inside → ENTER SHORT
  IF price CLOSES above the high of the OB → OB is INVALIDATED → NO TRADE
    → Mark as Breaker Block (now bullish)
    → Wait for price to return to it from above for LONG
```

---

#### 4B — Fair Value Gap (FVG) Entry Logic

**FVG Definition:** Gap between candle 1's wick and candle 3's wick (3-candle pattern) — candle 2 is the impulse.
- **Bullish FVG:** Gap below price (imbalance to the upside); price returns to fill it → long entry
- **Bearish FVG:** Gap above price (imbalance to the downside); price returns to fill it → short entry

```
IF price enters a Bullish FVG from above:
  Wait for a 15M or 1H bullish reaction candle within the FVG
  IF reaction candle closes above FVG midpoint → ENTER LONG
    Entry = at open of next candle OR limit at 50% of FVG (OTE)
  IF price fills entire FVG and continues lower → FVG invalidated
    → Do NOT enter
    → Look for next lower FVG or OB

IF price enters a Bearish FVG from below:
  Wait for 15M or 1H bearish reaction candle within FVG
  IF reaction candle closes below FVG midpoint → ENTER SHORT
  IF price fills entire FVG and continues higher → FVG invalidated → No trade
```

**Optimal Trade Entry (OTE) within FVG:**
- ICT's OTE = 62–79% retracement of the impulse leg (Fibonacci-aligned)
- Best entries are at 62%, 70.5%, or 79% retracement into the FVG zone
- `IF price hits 62–79% zone AND shows reaction → HIGHEST PROBABILITY ENTRY`

---

#### 4C — Liquidity Sweep Entry Logic

> The most powerful ICT setup. Smart money hunts stops, THEN reverses.

**Bullish Liquidity Sweep (Spring/Shakeout):**
```
IF price sweeps below a key Low (SSL — Sell Side Liquidity):
  AND price shows a strong bullish candle closing BACK above that low
  AND this occurs in a Kill Zone
  AND Daily/Weekly bias is bullish
  → ENTER LONG immediately on close of that candle
    OR limit order at the low of the sweep candle
  Stop = below the lowest wick of the sweep
  Target = next draw on liquidity (buy side; BSL above)
```

**Bearish Liquidity Sweep (Blow-off):**
```
IF price sweeps above a key High (BSL — Buy Side Liquidity):
  AND price shows a strong bearish candle closing BACK below that high
  AND this occurs in a Kill Zone
  AND Daily/Weekly bias is bearish
  → ENTER SHORT immediately
  Stop = above the highest wick of the sweep
  Target = next draw on liquidity (sell side; SSL below)
```

**IF** the sweep candle is very large (>2× ATR) → WAIT for a 1H pullback to the FVG created by the sweep, THEN enter. Do not chase.

---

#### 4D — Breaker Block Entry Logic

Breaker = former OB that price broke through → now acts as opposite zone.

```
IF a Bullish OB was broken to the downside (price closed below OB low) →
  OB becomes a Bearish Breaker Block
  IF price returns to this zone from below → SHORT entry

IF a Bearish OB was broken to the upside (price closed above OB high) →
  OB becomes a Bullish Breaker Block
  IF price returns to this zone from above → LONG entry
```

**Breaker blocks are HIGH CONVICTION** — they represent trapped traders AND institutional repositioning.

---

#### 4E — Mitigation Block Entry Logic

- Price left a rapid impulse that never retraced → created an imbalance
- Mitigation = price returning to "repair" that imbalance
- Treat exactly like an OB; same entry rules apply

---

#### 4F — Market Structure Shift (MSS) Entry Logic

```
On 15M or 1H chart:

IF in downtrend (LH/LL structure):
  Price makes a new Low
  THEN price creates a Higher High (breaks above previous swing high)
  → This is the CHOCH (Change of Character) = FIRST signal of reversal
  DO NOT enter yet
  Wait for first pullback after CHOCH
  IF pullback forms a 15M Bullish FVG or OB → ENTER LONG
  IF no pullback FVG/OB forms → wait for next 15M CHOCH confirmation

IF in uptrend (HH/HL structure):
  Price makes a new High
  THEN price creates a Lower Low (breaks below previous swing low)
  → CHOCH = first reversal signal
  Wait for first pullback after CHOCH
  IF pullback forms 15M Bearish FVG or OB → ENTER SHORT
```

---

## Phase 5 — Stop Loss Placement

> Stop placement is determined by ICT structure — NEVER by arbitrary percentage or dollar amount.

### Stop Loss Hierarchy (Choose the CLOSEST valid stop to entry)

#### Rule 1 — OB-Based Stop
```
IF entering from Bullish OB:
  Stop = 2–5 ticks BELOW the LOW of the OB candle (not the zone; the actual candle wick)

IF entering from Bearish OB:
  Stop = 2–5 ticks ABOVE the HIGH of the OB candle
```

#### Rule 2 — Liquidity Sweep Stop
```
IF entering after a liquidity sweep (long after SSL sweep):
  Stop = 2–5 ticks below the LOWEST wick of the sweep candle
  (Price should NEVER return there if the setup is valid)

IF entering after BSL sweep (short):
  Stop = 2–5 ticks above the HIGHEST wick of the sweep candle
```

#### Rule 3 — FVG Stop
```
IF entering from within an FVG:
  Stop = below the BOTTOM of the FVG (for longs)
  Stop = above the TOP of the FVG (for shorts)
  (If FVG fills completely, setup is invalid — stop enforces this)
```

#### Rule 4 — ATR Confirmation Check
```
After placing ICT structural stop:
  Calculate: Stop distance in $ / ATR value
  IF stop distance < 0.5× ATR → stop is too tight; likely to be hit by noise
    → Widen to 1× ATR minimum OR skip trade
  IF stop distance > 2.5× ATR → stop is too wide; R:R likely unacceptable
    → Skip trade unless target is proportionally large (4:1+ R:R)
  IF stop distance = 1–2× ATR → IDEAL zone; proceed
```

#### Rule 5 — Hard Maximum Stop (Risk Override)
```
IF structural stop would require risking > 2% of account → DO NOT TAKE THE TRADE
Wait for a tighter setup
A good setup with bad risk = bad trade
```

---

## Phase 6 — Position Sizing

> Position size is DERIVED from stop distance. Risk is FIXED. Size FLOATS.

### Formula
```
Risk Amount = Account Balance × Risk %
Risk % = 1% (standard) | 0.5% (low conviction / counter-trend) | 1.5% (max, 6/6 setup only)

Shares/Contracts = Risk Amount / (Entry Price − Stop Price)
```

### Risk % Decision Tree
```
IF setup score = 6/6 AND aligned with all 3 time frames AND in Kill Zone:
  → Risk 1.5% (maximum)

IF setup score = 5/6 AND 2 of 3 time frames aligned:
  → Risk 1%

IF setup score = 4/6 OR counter-trend trade:
  → Risk 0.5%

IF score < 4/6:
  → DO NOT TRADE

IF already have 1 open trade:
  → Max risk on new trade = 1%

IF already have 2 open trades:
  → No new positions until one closes

IF account is in drawdown > 3%:
  → Max risk per trade = 0.5% until back to flat

IF account is in drawdown > 6%:
  → STOP TRADING. Review journal. Resume next week.
```

---

## Phase 7 — Target Logic & Partial Exits

> ICT teaches: price is DRAWN to liquidity. Targets = the next pool of liquidity above (longs) or below (shorts).

### Primary Target Hierarchy

**LONG Targets (in order of proximity):**
1. Nearest 15M FVG above entry (partial take)
2. Nearest 1H FVG above entry
3. Previous Day High (PDH)
4. Previous Week High (PWH)
5. Nearest Daily OB or Bearish FVG above
6. Buy-Side Liquidity Pool (equal highs above)
7. Weekly BSL / Major swing high

**SHORT Targets (in order of proximity):**
1. Nearest 15M FVG below entry (partial take)
2. Nearest 1H FVG below entry
3. Previous Day Low (PDL)
4. Previous Week Low (PWL)
5. Nearest Daily OB or Bullish FVG below
6. Sell-Side Liquidity Pool (equal lows below)
7. Weekly SSL / Major swing low

### Partial Exit Structure (Standard 3-Part Plan)

```
POSITION SPLIT: Entry = 3 equal parts (1/3 each)

Part 1 — "The Sure Thing" (1/3 of position)
  Exit at: 1st target (nearest liquidity or FVG)
  Minimum R:R: 1.5:1
  Action: Close this portion immediately upon reaching T1
  Then: Move stop on remaining 2/3 to BREAKEVEN

Part 2 — "The Main Move" (1/3 of position)
  Exit at: 2nd target (PDH/PDL or key daily level)
  Minimum R:R: 3:1
  Action: Close upon reaching T2
  Then: Trail stop on final 1/3 to just below T1 level

Part 3 — "The Runner" (1/3 of position)
  Exit at: Maximum target (PWH/PWL or Weekly liquidity)
  Target R:R: 5:1 or more
  Action: Trail aggressively; let price pull it out OR close at Weekly level
  Stop management: Trail below each 4H Higher Low (longs) / above each 4H Lower High (shorts)
```

### IF Target Logic Overrides

```
IF T1 is less than 1.5:1 R:R away:
  → Skip T1 partial; treat T2 as first target
  → Split position into halves instead of thirds

IF T1 is hit but T2 is blocked by a major resistance (Daily OB / Weekly level):
  → Consider closing ENTIRE position at T1
  → Re-enter on a new setup after price digests that level

IF price reaches T1 AND forms a strong reversal candle (bearish engulfing / shooting star for longs):
  → Close ALL remaining position regardless of T2/T3
  → The market is telling you something; listen

IF price stalls for more than 3 full daily candles between entry and T1 with no progress:
  → Close 50% of position (time decay on the idea)
  → Move stop to breakeven on remainder
  → IF price still no movement in 2 more days → CLOSE ALL (time stop)
```

---

## Phase 8 — Trade Management Tree

> Every possible scenario after entry has a pre-defined response. No improvising.

### Scenario A — Trade Immediately Goes In Your Direction

```
IF price moves 1R in your favor within 1–2 candles:
  → Move stop to BREAKEVEN (entry price)
  → This trade is now "free" — worst case is 0 loss
  → Let price run to T1; do not interfere
  → Do NOT add to position here (FOMO adding = destroys R:R)

IF price moves to T1:
  → Close Part 1 (1/3)
  → Move stop to just above entry (lock in tiny profit on remaining)
  → Set alert at T2
  → Walk away; let it work
```

### Scenario B — Trade Goes Sideways (Consolidation)

```
IF price moves <0.5R in either direction for 1–2 days after entry:
  → Do NOT move stop or add
  → This is normal; give the trade time
  → Re-check: Is HTF bias still valid? Is the OB/FVG still intact?
    → IF YES: Hold. The setup is valid; give it 4–5 daily candles max
    → IF HTF structure has changed (new CHOCH against you): Exit immediately at market

IF consolidation continues beyond 5 daily candles without hitting T1:
  → Close 50% of position (time stop partial)
  → Move stop to breakeven on remainder
  → IF 2 more days with no movement → Close everything
```

### Scenario C — Trade Goes Slightly Against You (But Stop Not Hit)

```
IF price retraces 30–50% back toward stop but has NOT hit stop:
  → Do NOT exit early (stop is there for a reason; trust the structure)
  → Do NOT add more (do not average down)
  → Re-examine: Did price close BELOW the OB / FVG that triggered entry?
    → IF NO (just a wick through, closed back inside): HOLD — structure still valid
    → IF YES (candle CLOSED through your structure): EXIT IMMEDIATELY at market
      Do not wait for stop. The setup is invalidated. Take the smaller loss now.

IF price retests your entry level exactly (but stop is below):
  → This is normal; re-test of entry zone is healthy in ICT
  → HOLD unless it closes below the structure
```

### Scenario D — Trade Hits Stop

```
IF stop is hit:
  → The trade is CLOSED. Full stop. No questions.
  → Do NOT immediately re-enter the same setup
  → Do NOT try to "get it back" (revenge trading kills accounts)
  → Journal the trade immediately
  → Wait minimum 30 minutes before evaluating ANY new trade
  → Ask: Was the stop placement correct?
    → IF YES (correct structure, market just ran stops then reversed): This is normal. Loss is part of business.
    → IF NO (stop was too tight / emotional): Note lesson in journal.

IF stop is hit AND price immediately reverses in your original direction:
  → This is a stop hunt (classic ICT)
  → DO NOT re-enter immediately in anger
  → WAIT: If price forms a NEW setup (new FVG + OB + confirmation candle) → re-enter properly
  → Re-entry requires same 4/6 checklist score as original entry
  → This re-entry is a SEPARATE trade with its OWN risk calculation
```

### Scenario E — Trade Reaches T1, Continues to T2

```
IF Part 1 closed at T1 and price continues toward T2:
  → Stop is at breakeven; you have a free trade running
  → Let it work; resist urge to take early profits on Part 2
  → Only intervention: If a DAILY bearish candle (for longs) closes at T2 or beyond → close Part 2
  → Otherwise hold to T2 price target

IF price shoots past T1 without a reaction (large momentum candle):
  → Do NOT close Part 1 on the way up
  → Wait for first pullback candle after the momentum move
  → THEN close Part 1 at market (you're locking in MORE than T1)
```

### Scenario F — Trade Reaches T2, Runner Still Open

```
IF Parts 1 and 2 are closed; Part 3 (runner) still open:
  → Stop is now trailed below last 4H Higher Low (for longs)
  → Each time price makes a new 4H Higher Low → move stop up to just below it
  → Target: T3 (Weekly liquidity / major swing level)
  → Exit trigger options (use first that occurs):
    a) Price reaches T3 target → CLOSE ALL
    b) Price creates 4H CHOCH against you → CLOSE ALL
    c) Price closes below the 4H trailing stop → stopped out with profit
    d) News event imminent (FOMC, NFP) → CLOSE ALL before release
    e) Weekend approaching AND position is open overnight with no clear target hit:
       → IF profit on runner > 2R → HOLD over weekend (worth the gap risk)
       → IF profit on runner < 1R → CLOSE FRIDAY before 3:30 PM ET
```

### Scenario G — Overnight & Weekend Holding Rules

```
IF holding position overnight:
  → Acceptable IF: Stop is at breakeven or better AND HTF bias still valid
  → NOT acceptable IF: Stop is still in original placement AND you are in drawdown on the trade

IF Friday 3:00 PM ET approaches with open position:
  → IF trade is at breakeven or small loss AND T1 not reached → CLOSE (gap risk not worth it)
  → IF trade is profitable AND stop is at breakeven or better:
    → IF major news over weekend expected → CLOSE
    → IF no major catalysts → may hold; reduce to 50% and trail stop
  → IF sitting on 2R+ profit with runner → hold, accept gap risk as cost of large target

NEVER hold a full-size losing position into the weekend.
```

---

## Phase 9 — Exit Decision Tree

> Every exit must fit into one of these categories. If it doesn't, you're improvising.

### Exit Type 1 — Planned Target Exit ✅
- Price reached pre-defined T1, T2, or T3
- Close the corresponding portion
- Update stops on remainder per Phase 8

### Exit Type 2 — Structure Invalidation Exit ⚠️
```
TRIGGER: Candle CLOSES through your setup's defining structure
  → Closes below OB low (for longs)
  → Closes above OB high (for shorts)
  → Closes outside FVG that triggered entry
ACTION: EXIT IMMEDIATELY at market
  Do NOT wait for stop. The setup no longer exists.
  This often saves you 30–50% of the loss vs waiting for stop.
```

### Exit Type 3 — Hard Stop Exit 🛑
- Price hits mechanical stop order
- Trade closes automatically
- No action needed — this is the system working correctly

### Exit Type 4 — Time Stop Exit ⏱️
```
TRIGGER: Trade open for 5+ daily candles with no T1 hit AND no clear momentum
ACTION:
  → Close 50% immediately
  → Move stop to breakeven
  → TRIGGER 2: 2 more days with no progress → close remaining 50%
Rationale: Capital should be working; stale trades = opportunity cost
```

### Exit Type 5 — News/Event Exit 📰
```
TRIGGER: High-impact news event within 1 hour (FOMC, CPI, NFP, earnings)
ACTION:
  → IF trade is profitable: Close 50–100% before the event
  → IF trade is at breakeven: Close 100% before event (not worth the gap)
  → IF trade is losing (but above stop): Close 100% before event
    (news can cause gap through stops = larger loss than intended)
  → Only exception: Runner with stop at breakeven AND 3R+ already banked
    → May hold 25% through news as "free" speculation
```

### Exit Type 6 — Reversal Signal Exit 🔄
```
TRIGGER (for longs):
  Daily bearish engulfing candle forms at or near T2/T3 area
  OR: RSI divergence (price higher high, RSI lower high) at target zone
  OR: 4H CHOCH forms against position
ACTION: Close ALL remaining position immediately
  Do not wait for stop to be hit
  These signals say the move is over; honor them
```

### Exit Type 7 — Emergency Exit (Account Protection) 🚨
```
TRIGGER: Account drawdown hits 6% in a single week
ACTION:
  → Close ALL open positions immediately
  → No new trades for remainder of week
  → Review journal; identify what went wrong
  → Return Monday with 0.5% max risk until back to flat
```

---

## Phase 10 — Scenario Playbook (All Situations)

> Every situation you will encounter, with the exact response.

### SITUATION 1: "I missed the entry — price moved without me"
```
IF price has moved more than 1× ATR past your entry zone WITHOUT returning:
  → DO NOT CHASE
  → Mark the next retracement level (FVG, OB, Fibonacci OTE)
  → Set alerts; wait for price to come back
  → IF no retracement for 3 days → the opportunity has passed; move on
  → There is ALWAYS another setup
```

### SITUATION 2: "Price is near my entry but I'm not sure if it's ready"
```
IF you are unsure whether to enter:
  → Default answer = WAIT
  → Uncertainty is a "no" signal
  → Wait for ONE more confirmation candle that clearly shows intent
  → IF that candle confirms → enter
  → IF it doesn't confirm within the Kill Zone → skip the trade entirely
```

### SITUATION 3: "Price swept the stop and reversed — I was right but got stopped out"
```
→ This is an ICT liquidity sweep on YOUR stops
→ Do NOT re-enter in anger
→ Mark the sweep level
→ Wait for price to form a new structure (15M CHOCH + FVG)
→ If new setup forms → re-enter at 50% size (more cautious after 1 loss)
→ Same setup type: full 4/6 checklist required for re-entry
```

### SITUATION 4: "The market is in a strong trend — should I chase?"
```
IF market has run 3+ days in one direction without a pullback:
  → DO NOT chase trend
  → Strong trends produce the worst entries (buy the top, sell the bottom)
  → Wait for first significant pullback
  → IF pullback hits Daily OB or FVG → evaluate setup via full checklist
  → If that's an A+ setup → THEN enter in the direction of the trend
```

### SITUATION 5: "News caused a gap through my stop"
```
IF news gaps price through your stop level:
  → Accept the loss (may be larger than expected; this is gap risk)
  → DO NOT hold hoping it reverses
  → Close at market immediately on open
  → Review: Should position have been reduced before the event? (probably yes)
  → Lesson: Always check economic calendar; reduce size before known events
```

### SITUATION 6: "I have a big winner — should I hold or take profits?"
```
IF trade is showing 3R+ gain:
  → Parts 1 and 2 should already be closed (per Phase 7)
  → Runner (Part 3) is on; trailing stop manages it
  → DO NOT close the runner early out of fear
  → DO NOT add more at this point (late adding destroys the R:R of the original)
  → Trust the trailing stop; let the market take you out
  → Only manual close: if 4H CHOCH forms or major news approaches
```

### SITUATION 7: "I'm in 3 trades at once and a new perfect setup appears"
```
IF already at maximum concurrent positions (2 trades open):
  → DO NOT enter the new trade no matter how good it looks
  → Attention is diluted with 3+ trades
  → Mark it on chart; track it anyway for learning
  → IF one of your open trades reaches target and closes → evaluate the new setup THEN
```

### SITUATION 8: "I took a loss and feel the urge to trade again immediately"
```
IF you just closed a losing trade:
  → MANDATORY 30-minute waiting period before any new analysis
  → Use this time to journal the loss
  → After 30 min: re-run the Pre-Conditions gate check (Phase 0)
  → IF emotional state is compromised → NO MORE TRADES TODAY
  → Revenge trading is the fastest path to account destruction
```

### SITUATION 9: "My setup is forming but it's not in a Kill Zone"
```
IF all other criteria score 5/6 but timing is wrong (e.g., NY lunch):
  → DO NOT enter
  → Set alert for next Kill Zone
  → IF the setup is still valid at next Kill Zone → enter then
  → IF price moved significantly before Kill Zone → setup may be invalidated; re-evaluate
```

### SITUATION 10: "Price is at a major Weekly level — should I take profits early?"
```
IF runner (Part 3) is approaching a Weekly OB, Weekly FVG, or major swing high/low:
  → YES — close the runner at the Weekly level
  → Weekly levels are where smart money REVERSES price
  → Taking profits at Weekly liquidity targets = professional behavior
  → DO NOT hold through a Weekly level hoping for more
  → Exception: IF price is in a parabolic move with no structure — trail stop aggressively
```

### SITUATION 11: "I want to add to a winning trade"
```
IF price is moving in your direction and you want to add:
  → ONLY acceptable if ALL of these are true:
    a) Stop on original position is at breakeven or in profit
    b) A NEW setup (new OB/FVG) has formed on a pullback — not just momentum
    c) The add does not increase total risk beyond 2% of account
    d) You are still in a Kill Zone or within 1 hour of one
  → The add is treated as a COMPLETELY SEPARATE TRADE with its own stop and target
  → NEVER pyramid into a trade without a new setup; this is hope-based trading
```

### SITUATION 12: "Everything looks perfect but something feels off"
```
IF your checklist says enter but instinct says no:
  → TRUST THE HESITATION
  → Either: identify what's wrong (is there a level you missed?)
  → Or: skip the trade and watch from the sidelines
  → Unexplained discomfort = your subconscious saw something
  → There is no penalty for not trading. There is always another setup.
```

---

## Phase 11 — Post-Entry Rules

### Daily Check-In (Takes 5 minutes)
```
For each open position, ask:
  1. Is the HTF bias still intact? (IF NO → exit)
  2. Has price reached any target? (IF YES → execute Phase 7)
  3. Has price formed any reversal structure? (IF YES → evaluate exit type 6)
  4. Is stop still correctly placed? (Should never need changing; only moving to breakeven/profit)
  5. Is there a news event today that could affect this? (IF YES → see exit type 5 rules)
```

### What You May NOT Do After Entry
```
❌ Move stop further away from entry (widening stops = removing your own protection)
❌ Remove stop entirely "just this once"
❌ Add to a losing position (averaging down)
❌ Change your target to a smaller number because you're nervous
❌ Close a winning trade early because you "feel" it might reverse (if no signal)
❌ Let a 1R winner become a loss (once at 1R, stop MUST move to breakeven)
❌ Hold through a news event with a full-risk stop still in place
```

### What You MUST Do After Entry
```
✅ Set hard stop order immediately (do not rely on mental stops)
✅ Set price alerts at T1, T2, T3 so you don't have to watch
✅ Journal the trade: Entry reason, setup type, time frame alignment, screenshot
✅ Walk away — the trade is managed, not watched
✅ Move stop to breakeven once 1R is reached
✅ Close Part 1 when T1 is hit; no delay, no "maybe it goes further"
```

---

## Phase 12 — The Hard Rules (Never Break)

> These are carved in stone. Any deviation — no matter how justified it feels in the moment — is a system failure.

```
RULE 1:  Never risk more than 2% of account on a single trade. Ever.
RULE 2:  Never move a stop loss against your position.
RULE 3:  Never enter without a pre-defined stop AND target.
RULE 4:  Never trade during NY lunch (12:00–1:30 PM ET) unless managing open trade.
RULE 5:  Never hold a full-size position with original stop through a high-impact news event.
RULE 6:  Never average down on a losing trade.
RULE 7:  Never take a trade with R:R less than 2:1.
RULE 8:  Never enter a setup scoring less than 4/6 on the checklist.
RULE 9:  Stop trading for the day if 3% daily loss is hit. No exceptions.
RULE 10: Stop trading for the week if 6% weekly loss is hit.
RULE 11: Never re-enter a setup immediately after being stopped out.
RULE 12: Always move stop to breakeven once trade reaches 1R profit.
RULE 13: Close Part 1 (1/3 position) at T1 without hesitation.
RULE 14: Never take a counter-trend trade without explicit Weekly-level confluence.
RULE 15: Never trade based on someone else's signal without running your own checklist.
```

---

## Quick-Reference Decision Flowchart

```
START
  │
  ▼
[PRE-CONDITIONS GATE]
  Is emotional state neutral? Check calendar? Platform working?
  │── NO to any → STOP. No trade today.
  │── YES to all → Continue
  │
  ▼
[PHASE 1: HTF BIAS]
  Weekly + Daily + 4H aligned?
  │── 2 or more misaligned → WAIT or 50% size only
  │── All 3 aligned → Continue with full size potential
  │
  ▼
[PHASE 2: KILL ZONE]
  Are we in a valid session window?
  │── NO → Set alert; come back later
  │── YES → Continue
  │
  ▼
[PHASE 3: CHECKLIST]
  Score 4–6 confluences?
  │── < 4 → NO TRADE
  │── 4–5 → Reduced size (50–75%)
  │── 6/6 → Full size (up to 1.5% risk)
  │
  ▼
[PHASE 4: ENTRY TYPE]
  OB entry? FVG entry? Liquidity sweep? Breaker? MSS?
  │── Confirm with 15M candle close as entry trigger
  │── Set limit or enter at market on confirmation candle close
  │
  ▼
[PHASE 5: STOP PLACEMENT]
  Below OB low / above sweep wick / outside FVG
  ATR check: 1–2× ATR distance? → Proceed
  │── < 0.5× ATR → Widen stop
  │── > 2.5× ATR → Skip trade (too wide)
  │
  ▼
[PHASE 6: POSITION SIZE]
  Risk% ÷ (Entry − Stop) = Shares
  │── Max 2% risk regardless of conviction
  │
  ▼
[PHASE 7: TARGETS SET]
  T1 = nearest liquidity (1.5:1 min)
  T2 = daily structure (3:1 min)
  T3 = weekly liquidity (5:1+)
  │
  ▼
[TRADE IS LIVE]
  │
  ├── Goes in your favor immediately → move stop to BE at 1R; close 1/3 at T1
  ├── Goes sideways → hold; time stop at 5 days
  ├── Pulls back but structure intact → hold
  ├── Structure candle closes through OB/FVG → EXIT NOW (invalidation)
  ├── Hits hard stop → take loss; journal; 30-min break
  ├── T1 hit → close 1/3; stop to BE
  ├── T2 hit → close 1/3; trail final 1/3
  └── T3 hit → close all OR trail until 4H CHOCH
  │
  ▼
[POST-TRADE]
  Journal entry → screenshot → note emotion → note what was right/wrong
  Run pre-conditions check before next trade
```

---

## Related Notes

- [[Swing Trading MOC]]
- [[Quantitative Finance MOC]]
- [[ICT Concepts Glossary]]
- [[Market Structure Notes]]
- [[Liquidity & Order Flow]]
- [[Fair Value Gaps Reference]]
- [[Order Block Identification]]
- [[Kill Zone Timer]]
- [[Trading Journal Template]]
- [[Risk Management Framework]]
- [[Position Sizing Calculator]]
- [[Economic Calendar Workflow]]

---

*A plan followed imperfectly beats no plan perfectly. The edge is in the consistency, not the individual trade.*
