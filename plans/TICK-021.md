# Plan — TICK-021: Political-Alpha V2 event study (HYP-087 Track A + HYP-088 Track B)

Governing spec (LAW): `~/Obsidian/Obsidian/Trading/Research/Political-Alpha-V2-Claude-Code-Spec.md`
Branch: `sovereign-v2` · Module: `research/political_alpha_v2/` (new, isolated from V1)

## Approach

V1 (HYP-085) returned p=0.3637 because it averaged a heterogeneous event population
(tariff announcements + personal attacks) and tested for a mean drift that doesn't exist
at that aggregation. V2 **conditions on event type before testing** and corrects for
multiple comparisons. The deliverable is a `cluster → instrument → direction → timing`
lookup table, not a single p-value.

Two tracks built here (Track C / HYP-086 is separately pre-registered — reference only):
- **Track A (HYP-087):** keyword-cluster the 168 V1 events on `statement_text`, compute
  per cluster×instrument CSAR[0,+72h], within-cluster t-test + Bonferroni.
- **Track B (HYP-088):** congressional BUY clusters → Fisher's exact vs sector-favorable
  policy actions within 90d.

## Isolation (HARD)
`research/political_alpha_v2/` imports nothing from `sovereign/ ict/ ict-engine/ config/
audit/ scripts/`. It may **read** V1 output files (`research/political_alpha/data/
trump_events.jsonl`) as input, but imports no V1 module. Machinery is COPIED into a
self-contained `_lib.py` with provenance comments (same pattern V1 used). AST-enforced by
`tests/test_isolation.py`. No OANDA, no launchd, no live params.

## Files touched (all under research/political_alpha_v2/)
- `config/{cluster_rules,sector_map,policy_events}.json` — LOCKED at Phase 0
- `_lib.py` — self-contained machinery (yfinance, SAR, jsonl IO, env, keyword matcher)
- `build_clusters.py` (P1), `compute_cluster_returns.py` (P2),
  `build_congressional_signal.py` (P3), `run_statistical_tests.py` (P4)
- `tests/test_isolation.py`
- `data/*.jsonl` (outputs), `output/*` (deliverables)
- `data/.gitignore` (raw/ + cache/ ignored; output jsonl committed)

## Phase order (one [RESEARCH] commit each)
- **P0 Config lock** — finalize + commit the 3 config files (+ `_lib.py` + isolation test
  scaffolding so the isolation DoD is green at lock time). **No return/disclosure data may
  be pulled before this commit lands.** Config frozen thereafter.
- **P1 Clustering** — dedup V1 catalog to 168 unique events, classify each into ≤1 cluster
  by priority-ordered keyword rules, 5-trading-day de-dup per cluster×instrument, emit
  `clustered_events.jsonl`, report cluster counts, **lock Bonferroni denom = (clusters with
  ≥1 event) × 9**.
- **P2 SAR matrix** — per event×instrument: yfinance OHLCV → SAR (est window T-252→T-10,
  mean-adjusted, SAR=(R-μ)/σ) → CSAR[0,+72h]=Σ SAR over T+0,T+1,T+2 + pre-window
  CSAR[-48h,0]=Σ SAR over T-2,T-1. `data_ok:false`+`gap_reason` on any gap; never backfill.
- **P3 Congressional** — Quiver `/beta/live/congresstrading` (exponential backoff ≥60s on
  429; house.gov EFTS fallback). Filter BUY≥$15k, map ticker→bucket via sector_map, build
  ≥3-member/30d BUY clusters, match to favorable policy within 90d, Fisher contingency.
- **P4 Tests + output** — exactly the Phase-4 tests (within-cluster t-test, Bonferroni,
  hit-rate, negative-control, CSAR profile plots; Track B Fisher). Emit `cluster_playbook.md`,
  `cluster_sar_plots.png`, `congressional_signal_results.json`, `summary_report.md`.

## Key design decisions (locked into config at P0)
- **Keyword specificity over recall:** energy cluster uses `drill/oil/lng/pipeline`, NOT
  bare `energy` — macro-brag posts that merely mention "energy prices down" must route by
  priority to their primary theme (fed/tariff), not get grabbed by energy. `the fed`/
  `powell`/`interest rate`, not bare `fed`. Priority order = specificity descending.
- **Priority order:** tariff_china → tariff_broad → energy_environment → dollar_fed →
  defense_conflict → deregulation → personal_attack. First match wins; each event ≤1 cluster;
  no-match events are unclustered (excluded, not forced).
- **Bonferroni denom = clusters × 9 (full universe), conservative** — many cluster×instrument
  pairs aren't tested (each themed cluster tests 2-4 instruments) but every raw p is still
  multiplied by the full clusters×9. XLV/some instruments never tested by any cluster →
  inflate the denom conservatively. Documented in summary.
- **personal_attack instruments (spec §7 example shows `[]`, but §4/§7-P4 test #4 requires
  testing it):** resolved toward a functional negative control — assign the active themed
  instruments `[KWEB,SLX,XLE,XLI,DX-Y.NYB,GLD]` so the control actually checks whether
  no-policy attack tweets move the same instruments. Documented in config `note` + summary.
- **policy_events seed:** real dated EOs extracted from the V1 catalog's federal_register
  entries + well-known 2025-26 actions; favorable/adverse per bucket classified at
  compile-time (before any buy-cluster matching), rationale in each `notes`.

## Known failure modes (honest, from spec §10)
- Small cluster n × Bonferroni → low power; a real signal may not reach significance. Report
  honestly, don't loosen the null.
- Keyword misclassification + heavy tariff overlap → priority arbitrariness; disclosed.
- Track B: 45-day disclosure lag; mega-cap tech (AAPL/MSFT/NVDA) has no bucket → excluded →
  likely N<10 BUY clusters → Fisher underpowered. Report, don't suppress.
- Quiver free tier may 429 or require a key on the bulk endpoint → record the gap, don't mock.

## Definition of Done
Spec §9 checklist. A null result is a real result. Any cluster×instrument row that passes is
a *candidate* regime signal, NOT a greenlight — the full discovery gauntlet is a separate gate.
