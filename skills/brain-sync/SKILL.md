---
name: brain-sync
description: Writes any session output into the Obsidian brain at ~/Obsidian/Obsidian/Trading/System/ in structured, AI-readable format
---

## TRIGGER PHRASES
"sync to obsidian", "add to brain", "update obsidian", "brain sync", "save this to obsidian"

## WHAT YOU ARE DOING
You are writing structured content from this session into the Obsidian knowledge brain at `~/Obsidian/Obsidian/Trading/System/`. The output is a markdown file written to the correct subdirectory, with frontmatter, cross-links, and a confirmation log. Nothing is written until the path is verified.

---

## PHASE 1: VERIFY THE VAULT EXISTS

Run via `mcp__workspace__bash`:
```bash
ls ~/Obsidian/Obsidian/Trading/System/ 2>/dev/null && echo "VAULT_OK" || echo "VAULT_MISSING"
```

If VAULT_MISSING:
- Check alternative paths: `~/Obsidian/Trading/`, `~/Documents/Obsidian/`, `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/`
- Run: `find ~ -name "*.md" -path "*/Trading/System/*" 2>/dev/null | head -5`
- If still not found: tell Colin the vault path is not accessible. Ask for the correct path. Do not write until the path is confirmed.

If VAULT_OK: proceed to Phase 2.

---

## PHASE 2: CLASSIFY THE CONTENT

Determine the content type from what the user provided. Use this classification:

| Type | When to use |
|------|-------------|
| HYPOTHESIS_VERDICT | Any confirmed/rejected/conditional hypothesis result (HYP-NNN) |
| BACKTEST_RESULT | Quantitative backtest or walk-forward output with Sharpe, p-value, drawdown |
| ARCHITECTURAL_DECISION | A change to system design, layer boundaries, or component wiring |
| REGIME_READING | A macro or rate environment classification for a pair or strategy |
| TRADE_DEBRIEF | A completed trade with entry/exit/outcome/lesson |
| RESEARCH_NOTE | Any analysis, market observation, or exploratory finding not yet confirmed |

Ask Colin if the classification is ambiguous. Do not guess and write the wrong type to the wrong subdirectory.

---

## PHASE 3: CHOOSE THE SUBDIRECTORY

Map content type to subdirectory:

| Content Type | Subdirectory |
|-------------|--------------|
| HYPOTHESIS_VERDICT | `~/Obsidian/Obsidian/Trading/System/Hypotheses/` |
| BACKTEST_RESULT | `~/Obsidian/Obsidian/Trading/System/Backtests/` |
| ARCHITECTURAL_DECISION | `~/Obsidian/Obsidian/Trading/System/Architecture/` |
| REGIME_READING | `~/Obsidian/Obsidian/Trading/System/Regimes/` |
| TRADE_DEBRIEF | `~/Obsidian/Obsidian/Trading/System/TradeDebriefs/` |
| RESEARCH_NOTE | `~/Obsidian/Obsidian/Trading/System/Research/` |

If the subdirectory doesn't exist, create it:
```bash
mkdir -p <subdirectory_path>
```

---

## PHASE 4: GENERATE THE FILENAME

Format: `YYYY-MM-DD_<TYPE>_<slug>.md`

Rules:
- Date = today's date in YYYY-MM-DD format. Get it with: `date +%Y-%m-%d`
- TYPE = the content type in lowercase with hyphens (e.g., `hypothesis-verdict`, `backtest-result`)
- Slug = 2–4 word snake_case summary of the content (e.g., `gbpusd_carry_confirmed`, `ict_isolation_fix`, `2022_regime_narrowing`)

Example: `2026-07-22_hypothesis-verdict_hyp045_audnzd_exclusion.md`

---

## PHASE 5: BUILD THE FRONTMATTER

```yaml
---
type: <CONTENT_TYPE>
date: <YYYY-MM-DD>
instrument: <pair or ticker, or "n/a" if not applicable>
hypothesis: <HYP-NNN if applicable, else "n/a">
verdict: <CONFIRMED | REJECTED | CONDITIONAL | PENDING | n/a>
tags: [alta, trading, <type_tag>, <instrument_if_applicable>]
ai_summary: <one sentence: what this note contains, written for AI retrieval>
---
```

The `ai_summary` field must be specific enough to retrieve this note in a semantic search. Bad: "Notes about a backtest." Good: "GBPUSD carry OOS Sharpe 1.25 after v007 hold rollback, measured 2026-06-07, CONFIRMED."

---

## PHASE 6: BUILD THE BODY

Structure the body based on content type. Never write prose dumps — use structured elements only.

### HYPOTHESIS_VERDICT body:
```markdown
## Hypothesis
HYP-NNN: [one-line statement of the hypothesis]

## Verdict
**[CONFIRMED | REJECTED | CONDITIONAL]** — [one-sentence reason]

## Evidence
| Metric | Value |
|--------|-------|
| OOS Sharpe | X.XX (CI [X.XX, X.XX]) |
| p-value | X.XXX |
| n (trades) | NNN |
| Walk-forward | [summary] |

## Conditions (if CONDITIONAL)
- Condition 1
- Condition 2

## Related
- [[HYP-NNN related hypothesis]]
- [[Backtest file if exists]]
```

### BACKTEST_RESULT body:
```markdown
## Strategy
[Name and brief description]

## Parameters
| Param | Value |
|-------|-------|
| [key params from config] | [values] |

## Results
| Metric | In-Sample | Out-of-Sample |
|--------|-----------|---------------|
| Sharpe | X.XX | X.XX |
| Max DD | X% | X% |
| n trades | NNN | NNN |
| p-value | X.XXX | X.XXX |

## Walk-Forward
[Year-by-year Sharpe if available]

## Notes
- [Anything notable about data quality, regime sensitivity, etc.]
```

### ARCHITECTURAL_DECISION body:
```markdown
## Decision
[One sentence: what was decided]

## Context
- Why this decision was needed
- What alternatives were considered

## Implementation
- Files changed:
- Layer boundaries affected:
- Tests that verify this:

## Risk
- What could go wrong with this decision
- How to detect if it breaks
```

### REGIME_READING body:
```markdown
## Instrument / Strategy
[Pair or strategy name]

## Current Regime
**[WIDENING | NARROWING | MIXED | TRENDING | RANGING]** as of [date]

## Evidence
- [Rate differential data]
- [COT positioning]
- [Macro context]

## Forward Look
- Events in next 90 days that could shift regime:
  - [Date]: [Event] → [Expected impact]

## Action Signal
[PROCEED | CAUTION | HOLD] — [one-sentence reason]
```

### TRADE_DEBRIEF body:
```markdown
## Trade Summary
- Instrument: [pair/ticker]
- Direction: [LONG | SHORT]
- Entry: [price] on [date]
- Exit: [price] on [date]
- P&L: [$ or R-multiple]
- Hypothesis: [HYP-NNN if applicable]

## What Worked
- [bullet points]

## What Failed
- [bullet points]

## Lesson
[One specific, actionable lesson for future trades]

## Oracle
- decision_logger entry: [YES | NO | PENDING]
- update_outcome called: [YES | NO — if NO, flag this]
```

### RESEARCH_NOTE body:
```markdown
## Subject
[One line describing the research question]

## Findings
- [bullet points of key findings]

## Data Sources
- [list sources consulted]

## Open Questions
- [bullet points of what's still unknown]

## Next Step
[One specific next action]
```

---

## PHASE 7: FIND CROSS-LINKS

Before writing, search for related notes in the vault:
```bash
grep -r "<instrument OR hypothesis number OR key term>" ~/Obsidian/Obsidian/Trading/ --include="*.md" -l 2>/dev/null | head -10
```

For each related file found, add a `[[wikilink]]` in the appropriate section of the body. Use the filename without extension as the link target.

---

## PHASE 8: WRITE THE FILE

```bash
cat > "<full_path>" << 'EOF'
<frontmatter + body content>
EOF
```

Verify it was written:
```bash
head -5 "<full_path>" && echo "WRITE_OK"
```

---

## PHASE 9: CONFIRM

Report back to Colin:
- Full path of the file written
- Content type classification
- Any cross-links added
- One-line ai_summary that was stored

Do NOT update `MEMORY.md` or any other memory file as part of this skill — that is a separate step Colin controls.

---

## CONSTRAINTS

- Never write to the vault without verifying the path exists first.
- Never overwrite an existing file with the same name — check first with `ls <path>`. If a conflict exists, append a suffix (`_v2`) or ask Colin which version to keep.
- TRADE_DEBRIEF: always check if `update_outcome()` was called for the Oracle loop. If not, flag it prominently in the debrief note and remind Colin that the Oracle cannot learn without closed-loop outcomes.
- Body must use structured elements (tables, bullets, headers). No paragraphs longer than 2 sentences.
- Frontmatter ai_summary must be specific — it is the primary retrieval key for future AI sessions.
