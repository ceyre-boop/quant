# Plan — TICK-018: geometry_feed extractors (G2 — features for the LOCKED HYP-082/083/084)

Specs are hash-locked FIRST (data/research/preregister/HYP-082/083/084 + GEOMETRY-2026-07
manifest, commit 01cacbd) — this build may not alter any spec'd constant. Design verified
by Plan-agent (Day-3 appendix D2). ALL features trailing-only; warmup rows NULL (never
dropped — ASOF would backfill stale values).

## Files
- NEW `sovereign/sentiment/geometry_feed.py`:
  - `corridor_stats(close, window) -> (r2, dev_sigma)` — trailing linregress of
    log-price over the window ENDING at t; dev = (log p_t − fit_t)/std(residuals).
    Math adapted from research/validate_corridor_feature.py:16-29 (prior/pre-gate —
    math only, thresholds NOT).
  - `detect_fvgs_daily(df, max_age, min_atr_frac) -> (count_20d, unfilled)` —
    REPLICATED 3-bar gap kernel (do NOT import ict — the sentiment wall forbids it
    in both directions AND FVGDetector leaks last-bar ATR); operates on
    df.iloc[:t+1]; trailing ATR ends at t; params min_size_atr_fraction=0.3
    (config/ict_params.yml freeze value, restated in config below).
  - `tri_state_stats(df, window, pctile) -> (state, days_in_consolidation, range_slope)`
    — state = (linregress slope of trailing 20-bar rolling H−L range) < 0 AND current
    range < pctile of trailing 252d ranges; days = consecutive qualifying days.
  - `compute_pair(df, pair, cfg) -> DataFrame`; `update(con=None, pairs=None, start=None) -> dict`
    (feeder idiom: sovereign/sentiment/vix_feed.py; store.upsert keys=(date,pair)).
- EDIT `sovereign/sentiment/store.py`: SCHEMA += `sentiment_geometry_daily`
  (date, pair, corridor_r2, corridor_dev, corridor_window, fvg_count_20d,
  fvg_unfilled, tri_state, days_in_consolidation, range_slope, src_last_bar_date,
  fetched_at, PK(date,pair)).
- EDIT `sovereign/sentiment/board_state.py`: ALTER-ADD the 7 feature columns,
  ASOF LEFT JOIN (s.date >= g.date, pair-matched), REQUIRED_COLUMNS += 7.
- EDIT `scripts/audit_look_ahead.py`: provenance block `src_last_bar_date == date`
  (sentiment_vrp_daily pattern) + empirical-board tuple.
- EDIT `config/parameters.yml` additive block: `sentiment.geometry:` corridor_window
  120, fvg_max_age 20, fvg_min_atr_frac 0.3, tri_window 20, tri_pctile 0.25,
  start "2015-01-01".
- EDIT `experience/precedents.py`: BOARD_EXTREME_TAGS += geometry keys
  (tri_state true → {"consolidation"}; |corridor_dev| ≥ 2 → {"structure_extreme"})
  — required by locked HYP-084.
- EDIT `tests/test_sentiment_board.py:~155`: AST isolation tuple += geometry_feed
  (MANDATORY — the wall must cover the new module).
- NEW `tests/test_sentiment_geometry.py`.

## Data source
Daily OHLC parquets `data/research/positioning_family/spot_cache/{PAIR}_ohlc.parquet`
(Open/High/Low/Close, index Date, →2026-07-03). If a pair's parquet is missing:
degrade loudly (skip pair, count it), never fetch inside the feeder.

## Tests (all offline/synthetic)
(1) truncation-invariance: every feature at t identical on full vs iloc[:t+1] frames;
(2) look-ahead trap: append a 10σ future bar → features at t unchanged;
(3) FVG parity vs ict.fvg_detector.FVGDetector.detect on the SLICED frame (test file
imports both sides — legal outside the wall);
(4) tri_state synthetic contracting/expanding wedges;
(5) board ASOF visibility (row dated d: absent at d−1, present d and d+1) on
in-memory DuckDB;
(6) audit_look_ahead.audit → 0 violations on fixture db;
(7) AST wall covers geometry_feed.

## Constraints
Do NOT run `update()` against the real data/sentiment.db (the orchestrating session
owns real-data runs + the real look-ahead audit). Do NOT touch any prereg file. No
imports from ict/ anywhere under sovereign/sentiment/. Suite baseline must hold.

## Acceptance
New tests green; suite baseline holds; py_compile clean; the parity test pins the
replicated kernel to the ict canon on ≥3 synthetic scenarios (gap up, gap down, no gap).
