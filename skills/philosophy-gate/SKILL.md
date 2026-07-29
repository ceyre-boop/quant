---
name: philosophy-gate
description: Runs any trade idea, system change, or architectural decision through Alta's six trading tenets and returns a tenet-by-tenet alignment scorecard
---

## TRIGGER PHRASES
"philosophy check", "run the gate", "tenet check", "does this violate any tenet", "philosophy gate on X", "tenet alignment"

## WHAT YOU ARE DOING
You are running the Alta Investments Philosophy Gate — a structured alignment check against the six tenets from TRADING_PHILOSOPHY.md and two cross-cutting rules. The output is a widget with a tenet scorecard, cross-cutting rule checks, and a final PROCEED / PAUSE / ABORT recommendation.

This gate applies to: trade ideas, system changes, architectural decisions, parameter modifications, new strategy proposals, and any "should we do this?" question.

---

## PHASE 1: UNDERSTAND WHAT IS BEING GATED

Extract from the user's message:
1. **What is the idea?** (trade, system change, new strategy, parameter change, architectural decision)
2. **The instrument / scope** (if a trade: which asset, direction, thesis; if a system change: which components)
3. **The size / scale** (if a trade: approximate size; if a change: how many files/components affected)
4. **The expected edge** (why should this work?)

If the description is too vague to evaluate (e.g., "philosophy check on buying stocks"), ask: "Which specific trade or decision should I gate?" Do not run the gate on a vague idea — the output will be meaningless.

---

## PHASE 2: READ TRADING_PHILOSOPHY.MD (if accessible)

```bash
cat /sessions/beautiful-funny-hypatia/mnt/quant/TRADING_PHILOSOPHY.md 2>/dev/null | head -200
```

Use the exact tenet language from the file if it exists. If the file is not accessible, use the canonical six tenets below (sourced from CLAUDE.md).

---

## PHASE 3: EVALUATE ALL SIX TENETS

For each tenet, assign a color rating:
- **GREEN**: The idea clearly aligns with this tenet
- **AMBER**: Tension exists — the idea partially satisfies the tenet but with caveats
- **RED**: The idea violates this tenet

Be specific. Do not give GREEN if you're not sure — default to AMBER and explain why.

---

### TENET 1: RISK CONTROL IS THE STRATEGY
*"Avoiding losers is the game. The job is not to find winners — it is to not get destroyed."*

Questions to answer:
- Is there a defined maximum loss case for this trade or change?
- What happens if this is completely wrong? Is the outcome survivable?
- Is there a survival floor — a minimum capital/capacity level below which this cannot be attempted?
- For system changes: could this change cause silent failures that accumulate undetected loss?

GREEN if: loss is capped, defined, and survivable. Stop is placed at a technical level, not arbitrary.
AMBER if: loss is bounded but the bound is vague or untested.
RED if: no defined loss case, or the loss case is "hope it recovers."

---

### TENET 2: INNOVATOR → IMITATOR → LOSER
*"We are imitating proven professional behavior and entering before the crowd completes the move."*

Questions to answer:
- Is this idea imitating something that professionals (institutional flow, smart money, confirmed edges) have already done or set up?
- Are we entering early enough that the crowd has not yet completed the move? Or are we the last buyer at the top?
- Is there any disclosed evidence (Form 4, COT, unusual options, analyst behavior) that professionals are already positioned this way?
- For system changes: is this change imitating a proven design pattern, or is it novel and untested?

GREEN if: clear professional-behavior precedent, disclosed flow confirms, crowd not yet fully positioned.
AMBER if: circumstantial institutional evidence, or crowd positioning is moderate.
RED if: we are buying what everyone already owns, or the idea is purely novel with no precedent.

---

### TENET 3: IT IS NOT THE AVERAGE — IT IS SURVIVING THE BAD DAYS
*"The tail risk is the game. One catastrophic loss erases many average wins."*

Questions to answer:
- Is this sized for the tail? If this trade goes against us by 3× the expected move, what happens?
- What is the maximum plausible adverse scenario, and can the account survive it?
- For forex carry: in a rate reversal or risk-off episode, how bad does the carry position get?
- For system changes: could this change increase drawdown in tail scenarios (market crisis, data failure, broker outage)?

GREEN if: sized conservatively, tail scenario survivable, stop loss in place.
AMBER if: sized moderately but tail scenario is painful, not fatal.
RED if: sized aggressively or tail scenario is account-threatening.

---

### TENET 4: BEING TOO EARLY IS INDISTINGUISHABLE FROM BEING WRONG
*"Confirmation before entry. Predicting is not our game — confirming is."*

Questions to answer:
- Do we have confirmation that the move is already happening, or are we predicting that it will happen?
- What is the specific confirmation signal? (Price broke a level? Rate differential widened? Earnings beat? Institutional buying visible?)
- Is the entry trigger a lagging confirmation or a leading prediction?
- For system changes: has this been tested (backtested, unit tested, validated) or is it theoretical?

GREEN if: two independent confirmations exist — macro says it should happen AND price says it is already happening.
AMBER if: one confirmation exists, second is pending or implied.
RED if: the entry is based on prediction alone with no current-state confirmation.

---

### TENET 5: THE FIXED INCOME WORLD IS WHERE WE HUNT
*"Rate differentials, yield curve shape, and central bank divergence are the primary hunting ground."*

Questions to answer:
- For forex trades: is this rate-differential driven? Is the carry in the right direction? Is the differential WIDENING?
- For equity trades: is there a macro anchor? (earnings growth, sector rotation tied to rate cycle, etc.)
- For new strategies: does the edge connect to rates, credit, or macro — or is it pure price pattern?
- For system changes: does this change improve or degrade our ability to capture rate-driven moves?

GREEN if: directly rate-differential driven, macro anchor clear, regime aligned.
AMBER if: weakly connected to macro, or regime is mixed but not opposed.
RED if: pure price pattern with no macro anchor, or trade is against the rate differential direction.

For carry forex specifically: check the current regime. A trade against NARROWING is RED for this tenet unless conviction is Tier 3+.

---

### TENET 6: THE RACE TO THE BOTTOM IS OUR SIGNAL TO BE CAREFUL
*"When crowding is detected, size is halved automatically. Consensus is our exit signal, not our entry signal."*

Questions to answer:
- Is this trade or idea crowded? (COT positioning at extreme, everyone on same side, high retail sentiment)
- Have multiple mainstream sources picked up on the same setup in the last 7 days?
- Is this the kind of trade that would be discussed positively in a financial media headline today?
- For system changes: are we implementing a popular or trendy approach that everyone is adopting?

GREEN if: trade is contrarian or at least not crowded; COT not at extreme; idea is not mainstream narrative.
AMBER if: moderate crowding; positioning is elevated but not extreme.
RED if: heavily crowded; COT at extreme; trade is current consensus narrative. → SIZE IS HALVED per this tenet.

---

## PHASE 4: CROSS-CUTTING RULE CHECKS

These two rules apply to every decision regardless of tenet scores.

### CONFIRMATION RULE
*"Two confirmations before entry. Macro says it should happen + Price says it is already happening."*

- Confirmation 1 (macro): Is the macro environment saying this trade should work? (Rate differential direction, economic data, sector rotation signal)
- Confirmation 2 (price): Is price already moving in the expected direction? (Trend alignment, break of level, momentum)
- Met: YES / NO / PARTIAL
- If PARTIAL or NO: state what specific confirmation is missing and what would satisfy it.

### SURVIVAL RULE
*"Can this trade be sized to survive being wrong 5 times in a row?"*

- Calculate: if the position loses its maximum defined loss 5 consecutive times, what percentage of capital is consumed?
- Use the conviction-based sizing from `config/parameters.yml` — do not hardcode numbers. If config is not accessible, use the principle: no single trade should risk more than the risk-per-trade defined in RISK_CONSTITUTION.md.
- Met: YES / NO
- If NO: state the minimum position size that would make this rule pass.

---

## PHASE 5: DETERMINE THE RECOMMENDATION

### PROCEED
All six tenets GREEN or AMBER, both cross-cutting rules MET, and no tenet is RED.

### PAUSE
One tenet is RED, or two+ tenets are AMBER, or one cross-cutting rule is NOT MET.
State: what needs to change to flip to PROCEED.

### ABORT
Two or more tenets are RED. This idea should not proceed in its current form.
State: what fundamental redesign would be required.

---

## PHASE 6: RENDER THE WIDGET

Call `mcp__visualize__show_widget` with an HTML widget:

**Header**: "Philosophy Gate — [idea summary] — [date]"
Final recommendation badge (large): PROCEED (green) / PAUSE (amber) / ABORT (red)

**Tenet Scorecard** (6 rows):
| # | Tenet Name | Rating | Key Finding |
|---|------------|--------|-------------|
| 1 | Risk Control Is The Strategy | 🟢/🟡/🔴 | [one line] |
| 2 | Innovator → Imitator → Loser | 🟢/🟡/🔴 | [one line] |
| 3 | Surviving The Bad Days | 🟢/🟡/🔴 | [one line] |
| 4 | Too Early = Wrong | 🟢/🟡/🔴 | [one line] |
| 5 | Fixed Income World | 🟢/🟡/🔴 | [one line] |
| 6 | Race To The Bottom | 🟢/🟡/🔴 | [one line] |

Each row is expandable (CSS details/summary) to show the full evaluation for that tenet.

**Cross-Cutting Rules** (2 rows):
| Rule | Status | Gap |
|------|--------|-----|
| Confirmation Rule | MET / PARTIAL / NOT MET | [what's missing] |
| Survival Rule | MET / NOT MET | [sizing calculation] |

**CROWDING FLAG** (if Tenet 6 is RED):
Amber banner: "⚠️ CROWDING DETECTED — Per Tenet 6, size is halved from normal. Max size = 0.5× normal unit."

**Verdict Card** (PROCEED / PAUSE / ABORT):
- Color: green / amber / red
- Recommendation text: one sentence
- For PAUSE/ABORT: bulleted list of what must change

**Visual spec:**
- Dark background (#0f1117), cards with border (#1e2433)
- Tenet rows: left border color = green/amber/red for the rating
- Details/summary CSS accordion for expanded tenet evaluations
- Footer: "Alta Investments · Philosophy Gate · [date] · Tenets from TRADING_PHILOSOPHY.md"
- Self-contained HTML, no external JS

---

## CONSTRAINTS

- Never give all GREEN ratings without specific evidence for each. If you are not confident about a tenet, default to AMBER and explain why.
- ABORT requires 2+ RED tenets. Do not use ABORT for 1 RED tenet — that is PAUSE.
- The CROWDING flag (Tenet 6 RED) does not cause ABORT alone — but it does mandate halved sizing per the tenet. State this explicitly.
- For ICT-based trade ideas: Tenet 4 (Confirmation) is automatically AMBER because ICT edge is unvalidated (p=0.52). State this in the Tenet 4 evaluation.
- If TRADING_PHILOSOPHY.md is accessible, always read it for the exact tenet language before evaluating. The canonical text takes precedence over this skill's summary.
- For system changes that touch the execution path (forex_exit_manager, decide_exit, anything importable by live/backtest path): always add a Tenet 1 RED if there is no explicit unlock in NEXT.md (shadow/execution-path freeze is active per CLAUDE.md).
