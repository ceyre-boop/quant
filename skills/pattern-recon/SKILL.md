---
name: pattern-recon
description: Historical analog search for any setup description — returns similarity scores and outcomes from the Alta knowledge base and public sources
---

## TRIGGER PHRASES
"have you seen this before", "pattern recon on X", "what does this setup usually do", "analog search", "run pattern recon"

## WHAT YOU ARE DOING
You are searching for historical analogs to a described setup and returning similarity scores + outcomes. The output is a widget with analog cards sorted by similarity. You search 4 sources in order. You never manufacture confidence — if fewer than 3 analogs are found, you say so explicitly.

---

## PHASE 1: EXTRACT THE SETUP DESCRIPTION

From the user's message, extract:
1. **The instrument**: What asset or pair?
2. **The market condition**: What is the macro/rate/vol environment right now?
3. **The price action / technical setup**: What is the chart doing? (e.g., "consolidating at resistance after a 3-week rally", "gapping up 4% on earnings beat", "failing at prior high in a downtrend")
4. **The thesis**: What does Colin think will happen, and why?
5. **The timeframe**: Swing trade (days/weeks) or position trade (weeks/months)?

If the description is ambiguous, ask one clarifying question before searching. The search is only as good as the setup description.

---

## PHASE 2: SEARCH SOURCE 1 — OBSIDIAN BRAIN

Search the Obsidian vault for related patterns, hypothesis verdicts, and trade debriefs.

```bash
# Search for the instrument
grep -r "<instrument>" ~/Obsidian/Obsidian/Trading/ --include="*.md" -l 2>/dev/null

# Search for key terms from the setup description
grep -r "<key_term_1>\|<key_term_2>" ~/Obsidian/Obsidian/Trading/ --include="*.md" -l 2>/dev/null

# Read the most relevant files found (up to 3)
```

For each matching note, extract:
- Date of the observation
- Setup description
- Outcome (if recorded)
- Hypothesis number (if any)

---

## PHASE 3: SEARCH SOURCE 2 — HYPOTHESIS LEDGER

Check `data/research/hypothesis_ledger.jsonl` if it exists:

```bash
ls /path/to/repo/data/research/hypothesis_ledger.jsonl 2>/dev/null && cat /path/to/repo/data/research/hypothesis_ledger.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    h = json.loads(line.strip())
    print(h.get('id',''), h.get('description',''), h.get('verdict',''), h.get('sharpe',''))
" 2>/dev/null | head -50
```

The repo root is `/sessions/beautiful-funny-hypatia/mnt/quant/` or the path inferred from the session context.

Look for any confirmed (CONFIRMED) or rejected (REJECTED) hypotheses that match the instrument, setup type, or macro environment described.

Key known edges to check against:
- HYP-045: AUDNZD exclusion from carry (CONFIRMED, OOS Sharpe 1.08)
- HYP-061: CB-Blackout Gate — veto 3-14d pre BOE/FED (CONFIRMED)
- Overnight-QQQ: valid standalone edge, REJECTED as carry diversifier (ρ=0.42 in crisis)
- ICT pattern edge: NOT PROVEN (permutation p=0.52)
- Forex carry macro edge: PROVEN but regime-fragile (rolling WF: 2021=-0.13, 2022=+0.51, 2023=+1.26, 2024=-0.09)

---

## PHASE 4: SEARCH SOURCE 3 — LIBRARY

Check `sovereign/intelligence/library/` for pattern matches:

```bash
ls /sessions/beautiful-funny-hypatia/mnt/quant/sovereign/intelligence/library/ 2>/dev/null
```

Read any files that match the instrument or setup type. The Library contains 63 historical patterns. Extract any that structurally resemble the current setup (same asset class, similar macro environment, similar price action description).

---

## PHASE 5: SEARCH SOURCE 4 — ICT MEMORY ENGINE

Check `data/ledger/ict_memory.json` only if the setup is ICT-related (Fair Value Gap, Order Block, displacement, liquidity sweep, etc.):

```bash
cat /sessions/beautiful-funny-hypatia/mnt/quant/data/ledger/ict_memory.json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
# Print pattern summaries
for k, v in list(data.items())[:20]:
    print(k, v)
" 2>/dev/null
```

**Important**: ICT pattern edge is not proven (p=0.52). Any ICT analogs returned must include this caveat.

---

## PHASE 6: PUBLIC HISTORICAL SEARCH

If internal sources return fewer than 3 analogs, supplement with a web search:

Search: "[instrument] [setup description keywords] historical [year range]"
Examples:
- "GBPUSD rally stall at resistance rate differential 2019"
- "gapper fade earnings beat failed 2022 2023"
- "carry trade unwind macro setup 2022"

For each web result, extract:
- Period
- Setup description
- What happened
- Why (the causal mechanism)

Label all web-sourced analogs as "PUBLIC SOURCE — unverified" in the output.

---

## PHASE 7: SCORE SIMILARITY

For each analog found (from any source), assign a similarity score 0.0–1.0 based on feature overlap:

| Feature | Weight |
|---------|--------|
| Same instrument or asset class | 0.25 |
| Same macro/rate environment | 0.25 |
| Same price action pattern | 0.20 |
| Same timeframe | 0.10 |
| Same thesis direction (long/short) | 0.10 |
| Same vol environment | 0.10 |

Sum the weighted scores for each analog. State which features matched and which didn't.

Sort analogs by similarity score descending.

---

## PHASE 8: ANALYZE OUTCOMES

For each analog, determine:
- **Outcome direction**: Did the instrument move in the expected direction? YES / NO / MIXED
- **Outcome magnitude**: How much did it move? (% or pips)
- **Outcome timeframe**: How long did it take?
- **Contrarian or consensus**: Did the contrarian or consensus view win?
- **Key differentiator**: What condition separated the winners from the losers in this cluster of setups?

For the top analog (highest similarity), do a full breakdown. For others, summarize in one line.

---

## PHASE 9: RENDER THE WIDGET

Call `mcp__visualize__show_widget` with an HTML widget:

**Header**: "Pattern Recon — [instrument] — [date]"

**Analog Cards** (sorted by similarity descending):
Each card shows:
- Analog label (period + source)
- Similarity score: progress bar, 0–1 scale, color: green >0.7, amber 0.4–0.7, red <0.4
- Outcome: direction badge (green=expected direction, red=reversal, amber=mixed)
- Outcome magnitude and timeframe (one line)
- Feature match breakdown (5 small badges: matched features green, unmatched gray)

**Top Analog Full Breakdown** (only for the highest-scored analog):
- Full setup description of the analog
- What happened and why
- What was the key differentiator between winners and losers
- Any conditions that were present in the analog but are absent in the current setup (important gaps)

**Bottom Section — Two Columns**:
Left: "What this setup has going for it" — 3 green bullets
Right: "What could make it fail" — 3 red bullets

**ICT Caveat** (if any ICT analogs included):
Amber box at the bottom: "ICT pattern analogs included above. Note: ICT edge is unvalidated (permutation p=0.52, BH fails). These analogs are structural observations, not predictive signals."

**If fewer than 3 analogs found**:
Show a gray "Thin Analog Set" banner: "Only X analog(s) found across all sources. Confidence in pattern recon is LOW. Do not trade on structural similarity alone with fewer than 3 confirmed analogs."

**Visual spec:**
- Dark background (#0f1117), cards with border (#1e2433)
- Similarity score bar: CSS width based on score (e.g., 0.75 → 75% width)
- Footer: "Alta Investments · Pattern Recon · [instrument] · [date] · Sources: Obsidian, Ledger, Library, Web"
- No external JS. Self-contained HTML.

---

## CONSTRAINTS

- Never assign a similarity score above 0.6 to a web-sourced analog unless the match is nearly exact (same instrument, same macro context, same price setup). Public analogs are inherently noisier.
- Never manufacture analogs. If you cannot find enough, say so clearly.
- Do not use the pattern recon output as a standalone trade signal. It is evidence for the conviction map (run conviction-map skill) and the philosophy gate (run philosophy-gate skill).
- Always state the source for each analog: OBSIDIAN / LEDGER / LIBRARY / ICT_MEMORY / PUBLIC.
- If the setup involves a confirmed carry pair (GBPUSD, EURUSD, AUDUSD, GBPJPY), note whether the analog occurred during a WIDENING or NARROWING rate differential regime — this is the primary regime variable for carry.
