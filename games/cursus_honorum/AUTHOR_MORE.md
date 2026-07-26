# Authoring More Questions — No API Key, Claude Max Only

The game is fully offline and reads a static `question_bank.json`. Growing the bank is a
**session-time authoring task** done by Claude (the Magistrate) inside a normal Claude Code /
Claude Max session — never an API call, never a key. This file is the prompt Colin pastes in
when he wants a top-up.

## When to run this

- The heatmap in the game (`localStorage` key `cursus_honorum_save_v1` →
  `categoryStats`) shows weak categories (accuracy below ~50% with a handful of attempts).
- Or just periodically to widen the bank.

## How to get the current weak categories

Open the game in a browser, then in devtools console:

```js
JSON.parse(localStorage.getItem('cursus_honorum_save_v1')).categoryStats
```

Copy that object (or just the categories with low `correct/attempted`) into the prompt below.

## The prompt (paste into a Claude Code session in this repo)

```
Author N new questions for games/cursus_honorum/question_bank.json, weighted toward these
weak categories: <paste categoryStats or list of weak category names here>.

Rules (same schema, same discipline as the existing 100 — read games/cursus_honorum/README.md
and games/cursus_honorum/categories.json first):
1. Every question is "diagnostic" phase, id continuing from the current max DIAG-NN.
2. Ground every prompt_roman + THE DATA block in the real system: read
   TRADING_PHILOSOPHY.md, RISK_CONSTITUTION.md, config/parameters.yml, config/ict_params.yml,
   and the relevant code (sovereign/forex/*, sovereign/training/state_space.py,
   imbalance_engine/petroulas_gate.py) — do not invent numbers.
3. Every `correct` answer must be independently re-verified against the cited `tenet` and the
   scenario's own numbers before it's written down — a wrong optimal trains the wrong reflex.
4. Bias category selection toward the weak list above; still tag each with the correct
   `category` from categories.json (don't force a category that doesn't fit just to hit quota).
5. Keep the answer-letter distribution balanced (~25/25/25/25 A/B/C/D) across the NEW batch
   combined with the existing bank — count current letter distribution first, then pick
   letters for new questions to correct any skew.
6. `stockfish_explanation` in mentor voice (not textbook), explaining WHY the optimal is
   optimal, referencing the tenet.
7. Validate structurally when done: every id unique, every category/tenet resolves against
   categories.json, every correct is a real option key, no duplicate prompts.
8. Append only — never edit or delete existing questions/ids. Bump question_bank.json's
   meta.count (and any meta.generated_at / version field) after appending.
9. This touches games/ only — no execution-path file. Explicit git add of the changed file,
   never -A, given concurrent dirty state on sovereign-v2.

Output: the diff/new records, plus a one-line summary of the new per-category counts.
```

## Why this has no API key anywhere

Claude authoring the questions **is** the Claude Max session already running interactively —
same as this authoring task itself. The game never calls out to anything: it `fetch()`s two
local JSON files and does all Elo/routing math in the browser. Growing the bank is just asking
Claude, in a normal session, to append more grounded records to that JSON file.
