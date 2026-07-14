# Night Shift — 2026-07-14 (operator asleep; explicit approval to spend session budget)

## The plan (Fable's part)

**N1 — Light professional reskin (his #1 ask).** The SPA already runs on CSS variables —
the reskin is a new light palette (white / light-blue #3b82f6 family) + typography/spacing
polish + fixing hardcoded darks (incl. the ICARUS strip gradient), NOT a rewrite. Every ID,
every feature, every line of JS stays — "i like the features we have now." Both index.html
and ict/index.html. Delegated to a worktree agent with strict do-not-break rules.

**N2 — Skills arsenal page (skills.html).** A dashboard page of copy-paste Claude Code
commands: every user-invocable skill/command in his environment with a one-line trader-
friendly description and a copy button, grouped (Trading/Research · Building · Content ·
Ops). Delegated to a second worktree agent with the curated content.

**N3 — Oracle daily digest + progression tracker (mine — needs session context).**
`sovereign/oracle/daily_digest.py`: bounded, cheap, read-only — gathers NEXT.md top
entries, ledger tail, ICARUS shadow state, gate ladder → ONE budget-capped LLM call →
`data/oracle/daily_digest.json` {what_happened, what_it_means, progression: gate ladder
with states, todays_one_thing}. Rendered as a prominent panel under the ICARUS strip.
Curated-context beats a half-built RAG overnight; a true doc-index can come later.
The progression ladder (his words): 30 clean shadow days → 1-day funded-account sim →
broker + pass/fail → earn → compound wisdom.

**N4 — Integration + verify + morning brief.** Merge agent output into BOTH branches,
push, live-verify over the wire, NEXT.md, leave a morning summary.

Plugins (Design/Marketing/Data/Finance/FSI/Bigdata): acknowledged; documented on the
skills page as chat-side tools; no overnight automation built on them (they live in his
chat UI, not this CLI's skill list).

## Budget discipline
Two background agents total (worktree-isolated, self-contained briefs), core work inline.
No swarms. Priority if time runs short: N1 > N2 > N3 > N4 polish.
