"""
sovereign/discovery/ — Edge-Discovery pipeline (research bench).

Discovery GENERATES candidate setups; the system's existing methodology gate
(permutation ≥10k + purged walk-forward + Deflated Sharpe + Benjamini-Hochberg)
DECIDES which are real. Never touches live config or trading — Phase 5 is human review.

Tracks:
  - forex-daily   : yfinance daily, 15yr, 4 pairs (PRIORITY — fully built)
  - nq-intraday   : data/es_nq/*.parquet, 1m/5m (scaffolded; NQ simulator is the wiring point)
  - intraday-fx   : requires vendor data (stub + acquisition runbook)
"""

TRACKS = ("forex-daily", "nq-intraday", "intraday-fx")
