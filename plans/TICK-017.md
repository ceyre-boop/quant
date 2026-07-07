# Plan — TICK-017: citation scorer (A2 — Gγ's measurement instrument)

Design verified by Plan-agent 2026-07-06 (Day-3 appendix D3). Without this the dark
month produces nothing HYP-084 can adjudicate.

## Build
(a) `experience/citations.py` — additive: module constant
`SEVERITY_PREDICTION_MAP = {-1: {"direction":"up","instruments":["V015_EQUITY"]},
0: {"direction":"none"}, 1: {"direction":"vol_up"}, 2: {"direction":"drawdown",
"instruments":["V015_EQUITY"]}}` (documented: −1 = carry-favorable ⇒ equity up /
low-vol persists; "none" ⇒ UNSCOREABLE; vol_up threshold constant lives HERE too:
realized-vol ratio ≥ 1.5 window-vs-prior) + `map_prediction(precedent, pairs)` →
new field `analogy_prediction_v2` {direction, instruments (pairs or V015_EQUITY),
horizon_days=outcome_days} written alongside the UNTOUCHED free-text v1.

(b) NEW `experience/citation_scorer.py`: `SCORES_PATH =
data/experience/citation_scores.jsonl`; `due_citations(as_of)` = scoring_due ≤ as_of
AND scored is None AND citation_id not already in scores file (idempotent);
`resolve(citation)` over [week_end, scoring_due] clamped to last available bar —
up/down via spot_cache Close parquets; vol_up via realized-vol ratio (constant from
the map); drawdown via v015 `daily_portfolio_equity` + `fwd_max_drawdown`
(sovereign/research/positioning/v015_replay.py:68,88); legacy rows lacking v2 get it
derived on-the-fly from severity (read-only), else UNSCOREABLE; `run(as_of)` APPENDS
{citation_id, scored: true|false|"UNSCOREABLE", basis, scored_at} — citations.jsonl
is NEVER rewritten.

(c) `experience/weekly_review.py` — one guarded `_feed_citation_scores()` Counter
line in the Forensics assembly (same try/except idiom as the six existing feeds).

## Tests (tests/test_citation_scorer.py + extend tests/test_weekly_review_forensics.py)
Map totality over {-1,0,1,2} · UNSCOREABLE on missing pairs/parquet/trades (no crash)
· idempotent double-run (one score row per id) · synthetic up-move → true, synthetic
drawdown case · sha256(citations.jsonl) unchanged across run() · monkeypatched
CITATIONS_PATH/SCORES_PATH.

## Acceptance
Suite baseline holds; scorer run over the (currently empty) real citations file is a
clean no-op; UNSCOREABLE is first-class in the forensics line.
