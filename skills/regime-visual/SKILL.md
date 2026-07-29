---
name: regime-visual
description: Color-coded interactive regime timeline widget for any strategy + instrument combination
---

## TRIGGER PHRASES
"regime visual for X", "color-code the history on X", "show me when X paid", "regime map for X", "visual timeline"

## WHAT YOU ARE DOING
You are generating a color-coded interactive HTML regime timeline widget for a strategy + instrument combination. The output is always rendered via `mcp__visualize__show_widget`. The widget shows when the strategy historically paid (green), when it lost (red), when conditions were mixed (amber), and highlights the current state in indigo with a visible border.

---

## PHASE 1: PARSE THE REQUEST

Extract from the user's message:
- **Strategy**: What is the trade idea? (carry, gapper-fade, trend-following, momentum, mean-reversion, etc.) If not stated, infer from the instrument type and context.
- **Instrument / pair**: What asset? (e.g., GBPUSD, AUDUSD, SPY, AAPL, crude oil)
- **Date range**: Default to last 5 years if not specified. Accept "since 2020", "last 3 years", etc.

If the strategy is one of the known Alta systems, apply the known regime definitions:
- **Carry forex** (GBPUSD, EURUSD, AUDUSD, GBPJPY): Regime = rate differential direction. WIDENING → GREEN. NARROWING → RED. FLAT/MIXED → AMBER.
- **Gapper-fade** (equities): Regime = VIX environment + earnings season density. LOW VIX + LOW EARNINGS DENSITY → GREEN. HIGH VIX → RED. MIXED → AMBER.
- **ICT pattern trades**: Regime = NOT PROVEN edge (permutation p=0.52). Widget must include a prominent caveat: "ICT edge unvalidated — regime classification is structural only, not predictive."
- **Unknown strategy**: Ask one clarifying question before proceeding: "What condition makes this strategy work — what needs to be true in the macro/price environment?"

---

## PHASE 2: RESEARCH THE HISTORY

Use `WebSearch` and `mcp__workspace__web_fetch` to gather regime-relevant data for the instrument over the requested period. Prioritize:

1. **Central bank rate decisions** (for forex): Fed, BOE, RBA, BOJ decision dates and direction. Build a timeline of rate differential changes.
2. **Volatility history** (for equities): VIX levels by quarter. Flag quarters where VIX avg > 25 as elevated.
3. **Macro regime periods**: Identify named regimes (e.g., "2022 rate hiking cycle", "2023 disinflation", "2024 soft landing", "2021 COVID recovery"). Label them.
4. **Known strategy performance**: If this is a known Alta edge, reference the walk-forward data already in CLAUDE.md or the hypothesis ledger. The carry walk-forward data is: 2021: -0.13 / 2022: +0.51 / 2023: +1.26 / 2024: -0.09.

Also check `data/research/` for any cached regime or backtest files relevant to this instrument.

---

## PHASE 3: CLASSIFY PERIODS

For each year-quarter in the date range, assign a regime state and color:

| State | Color | Hex | Meaning |
|-------|-------|-----|---------|
| FAVORABLE | Green | #4ade80 | Strategy historically pays in this environment |
| UNFAVORABLE | Red | #f87171 | Strategy historically loses in this environment |
| MIXED | Amber | #fbbf24 | Evidence split; edge reduced |
| CURRENT | Indigo | #818cf8 | Present state — always highlighted with border |
| UNKNOWN | Gray | #64748b | Insufficient data to classify |

Determine current state based on Phase 2 research. The current period always uses CURRENT color regardless of the underlying regime (show the underlying regime as a label inside the current card).

---

## PHASE 4: BUILD THE GREEN-LIGHT CHECKLIST

Before acting on this strategy in this instrument, what must be true? Produce a checklist of 3–6 conditions. For each, determine if it is currently MET, UNMET, or UNKNOWN.

Example for carry GBPUSD:
- [ ] BOE rate > Fed rate (differential WIDENING) — MET / UNMET
- [ ] 20-day trend aligned with carry direction — MET / UNMET
- [ ] No CB decision within 14 days (CB-Blackout Gate: HYP-061 CONFIRMED) — MET / UNMET
- [ ] VIX below 25 (low-vol regime) — MET / UNMET

For gapper-fade:
- [ ] Gap is >2% on earnings or news catalyst — MET / UNMET
- [ ] Pre-market volume is 3× ADV or higher — MET / UNMET
- [ ] No broad market crisis (SPY -3% day) — MET / UNMET

---

## PHASE 5: IDENTIFY UPCOMING FORWARD EVENTS

Search for scheduled events in the next 90 days that could shift the regime:
- Central bank meetings (FOMC, BOE, RBA, BOJ, ECB)
- Major earnings (if equity)
- Macro data releases (CPI, NFP, GDP)
- Geopolitical or regulatory events if salient

Build a table: Date | Event | Potential Regime Impact (SHIFT UP / SHIFT DOWN / NEUTRAL).

---

## PHASE 6: RENDER THE WIDGET

Call `mcp__visualize__show_widget` with an HTML widget containing these 4 elements in order:

**Element 1 — Timeline Bar**
Horizontal bar spanning the full date range. Each period is a colored segment. Segments are labeled with the year or quarter. The CURRENT segment has a white border and "NOW" label. On hover (CSS :hover), each segment shows a tooltip with: period label, regime state, one-line explanation.

**Element 2 — Current State Card**
Prominently displayed card below the timeline. Shows:
- Current regime state in large text (color-coded)
- The underlying conditions driving this state (2–3 bullet points)
- Confidence level: HIGH / MEDIUM / LOW based on data quality

**Element 3 — Forward Events Table**
Table with columns: Date, Event, Regime Impact. Color-code the impact column (green=favorable shift, red=unfavorable shift, amber=neutral).

**Element 4 — Green-Light Checklist**
Final element. Title: "Before Acting — Conditions Required". Each checklist item:
- MET: green checkmark ✓
- UNMET: red X ✗
- UNKNOWN: amber ? 

Below the checklist, a single line: "X of Y conditions met" — and the overall readiness signal: PROCEED / CAUTION / HOLD.

**Visual spec:**
- Dark background (#0f1117)
- Cards with border (#1e2433)
- Legend strip at top-right: colored squares + labels
- Footer: "Alta Investments · Regime Visual · [instrument] · [strategy] · [date]"
- Self-contained HTML, no external libraries
- Timeline must be responsive to widget width

---

## CONSTRAINTS

- Never invent performance data. If historical returns are unknown, say so and label the period UNKNOWN.
- The ICT pattern caveat (unvalidated edge) must appear prominently if the strategy is ICT-based.
- The CB-Blackout Gate (HYP-061, CONFIRMED) always applies to carry forex pairs. Flag any CB decision within 14 days.
- If the user requests a date range with insufficient public data, clearly state the data gap and show UNKNOWN for those periods.
