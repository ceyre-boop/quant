---
name: pre-trade-brief
description: Structured pre-trade research brief with YES/NO/WAIT verdict for any instrument
---

## TRIGGER PHRASES
"brief on X", "should I enter X", "quick check on X", "is this setup clean", "pre-trade check"

## WHAT YOU ARE DOING
You are producing a structured pre-trade research brief for the instrument the user named. The output is a single HTML widget rendered via `mcp__visualize__show_widget`. Total AI work must complete in under 2 minutes. The final output is one of three verdicts: **YES / NO / WAIT**.

- YES = edge present, setup clean, timing good — proceed at conviction-appropriate size
- NO = edge absent, setup broken, or risk too high — do not enter
- WAIT = edge exists but timing is wrong (calendar risk, no confirmation yet, regime misaligned)

---

## PHASE 1: DATA GATHERING (do this before writing anything)

Run these lookups in parallel where possible. Use `mcp__workspace__web_fetch` and `WebSearch`.

1. **Price + regime**: Current price, 20-day and 5-day trend direction, ATR or HV percentile if findable. For forex pairs (GBPUSD, EURUSD, AUDUSD, GBPJPY), also check `data/research/` in the repo for any cached regime state file — read it if it exists.

2. **News**: Search "[instrument] news site:reuters.com OR site:bloomberg.com OR site:ft.com" for the last 7 days. Flag anything that touches earnings, central bank decisions, FDA, M&A, or macro data releases.

3. **Disclosed flow** (public data only — nothing illegal):
   - Equities: Search "[ticker] Form 4 insider" and "[ticker] unusual options" on finviz, unusualwhales, or SEC EDGAR.
   - Futures: Search "[instrument] COT report net positioning" for the latest CFTC release.
   - Forex: Search "[pair] speculative positioning COT" for the latest weekly data.
   - Options: Note implied move if earnings are within 14 days.

4. **Calendar events**: Search "[instrument] earnings OR FDA OR FOMC OR BOE OR RBA OR BOJ date 2026". Flag any event within 72 hours with ⚠️.

5. **Analyst consensus** (equities only): Search "[ticker] analyst price target consensus". Note mean target, range, and any recent upgrades/downgrades in the last 30 days.

---

## PHASE 2: CLASSIFY THE INSTRUMENT

Determine which category applies — this drives what regime signals matter:

- **Carry forex pair** (GBPUSD, EURUSD, AUDUSD, GBPJPY): Regime = rate differential direction (widening = LONG favored, narrowing = SHORT favored, mixed = WAIT). Check if the current regime is CONFIRMED or FRAGILE.
- **Equity — individual stock**: Regime = broad market trend (SPY 20d), sector trend, stock RS.
- **Equity index / ETF**: Regime = macro cycle phase, VIX level vs. 1-year percentile.
- **Futures (commodity, rates)**: Regime = COT net positioning trend + macro backdrop.
- **Options**: Note the strategy type. Flag if IV rank > 50 (sell premium) or < 30 (buy premium).

---

## PHASE 3: BUILD THE 5-SECTION BRIEF

Organize your findings into exactly these 5 sections. Each section is 3–5 lines maximum. Terse. No prose dumps.

### Section 1 — INSTRUMENT + REGIME
- What is it (asset class, sector if equity, rate sensitivity if forex)
- Current price and short-term trend (up/down/flat)
- Regime label: TRENDING / RANGING / TRANSITIONING
- For forex: rate differential direction and conviction level
- Vol context: HV/IV percentile or "elevated / normal / compressed"

### Section 2 — CONSENSUS vs. REALITY
This is the Petrules Gate Organ 1 + Organ 2 run manually.
- **Consensus**: What does the market currently expect? (analyst targets, options-implied move, recent news narrative, COT positioning direction)
- **Disclosed footprints**: What do the insiders/institutions actually appear to be doing? (Form 4 buys/sells, unusual options, COT change direction vs. prior week, large block trades if findable)
- **Gap**: Does the disclosed behavior diverge from the consensus narrative? State explicitly: ALIGNED / DIVERGING / UNCLEAR

### Section 3 — KEY RISKS
List exactly 3 risks. Each is one sentence. Format:
- RISK 1: [name] — [one-sentence description]
- RISK 2: [name] — [one-sentence description]
- RISK 3: [name] — [one-sentence description]
Flag any risk tied to an event within 72 hours with ⚠️.

### Section 4 — SETUP QUALITY
- Is there a clear level (support/resistance/FVG/order block/trend line)? YES / NO
- Timeframe alignment: Are higher TF and entry TF pointing same direction? YES / NO / MIXED
- R:R estimate: If the user stated a stop and target, compute it. If not, estimate based on nearest structure. State as X:1 or "cannot compute — no stop stated"
- Setup cleanliness: CLEAN / MESSY / PENDING (pending = waiting for confirmation candle/break)

### Section 5 — VERDICT
State: **YES**, **NO**, or **WAIT**
Then write one paragraph (4–6 sentences) explaining:
- The primary reason for the verdict
- What the biggest supporting factor is
- What the biggest risk to being wrong is
- For WAIT: what specific condition must change to flip to YES

---

## PHASE 4: RENDER THE WIDGET

Call `mcp__visualize__show_widget` with an HTML widget. Use this layout:

Dark background (#0f1117). Each section is a card with a subtle border (#1e2433).
Header: instrument name, current price, verdict badge (green=YES, red=NO, amber=WAIT).
5 section cards below. Section titles in small caps, muted color (#64748b).
Color coding within each card:
  - Favorable signals: text in #4ade80 (green)
  - Unfavorable signals: text in #f87171 (red)
  - Mixed/watch signals: text in #fbbf24 (amber)
  - Neutral: text in #e2e8f0
Footer: "Alta Investments · Pre-Trade Brief · [date] · Data: public sources only"

The widget must be self-contained HTML — no external JS libraries required. Use inline CSS only. Keep it under 250 lines of HTML.

---

## CARRY PAIR SPECIAL CASE

If the instrument is GBPUSD, EURUSD, AUDUSD, or GBPJPY:

1. Before Phase 1, check `data/research/` for any file matching `*regime*`, `*carry*`, or the pair name. Read it if it exists.
2. In Section 1, include: current regime state (from file or "no cached state — using live data"), the rate differential direction, and whether the regime is WIDENING / NARROWING / MIXED.
3. In the Verdict, explicitly state whether the carry regime supports or opposes the proposed trade direction. A trade against a NARROWING regime requires Tier 3+ conviction to proceed.

---

## CONSTRAINTS

- Never fabricate price, flow, or positioning data. If a data source is unavailable, state "data unavailable" in that line.
- Never recommend a specific position size — that is the conviction-map skill's job.
- Do not speculate about undisclosed information. Public data only.
- If the instrument is completely unknown or unlisted, say so and offer to run the brief with whatever data is findable.
- Calendar events within 72 hours always trigger a WAIT unless the user explicitly states they are aware and have a specific catalyst thesis.
