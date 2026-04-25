#!/bin/bash
# Codex parallel task runner.
# Usage: bash scripts/codex_tasks.sh [task_a|task_b|task_c|all]
# Each task runs independently — fire all three in parallel with: bash scripts/codex_tasks.sh all &
#
# Claude Code reviews all output before integration.

set -euo pipefail
REPO=/Users/taboost/quant
TASK="${1:-all}"

run_task_a() {
  echo "[codex] Task A: forex unit tests → starting"
  codex exec \
    -C "$REPO" \
    --full-auto \
    -o "$REPO/logs/codex_task_a_result.md" \
    "Read sovereign/forex/macro_engine.py in full.

Write pytest unit tests for ForexMacroEngine and save them to
tests/unit/test_forex_macro_engine.py.

The file already has these unit test files nearby as style references:
tests/unit/test_backtest.py, tests/unit/test_layer1.py.

Requirements for each test:
- Mock ForexDataFetcher.get_country_macro() to return canned macro dicts
  with keys: rate (float), cpi_yoy (float)
- Mock ForexDataFetcher._fetch_price_series / yfinance.download so no
  network calls are made — return a pd.Series of 120 synthetic prices
- Mock RiskSentimentEngine.override_for_pair() to return None by default
- Mock FairValueModel.score_pair() to return a plausible FairValueSignal
- Mock CycleDetector.score_pair() to return a plausible CycleSignal

Tests to write (cover all logic branches):

1. test_score_pair_long — high positive raw score → direction='LONG', conviction >= 0.35
2. test_score_pair_short — high negative raw score → direction='SHORT'
3. test_score_pair_neutral_low_conviction — raw score near 0 → direction='NEUTRAL'
4. test_score_pair_risk_override — override_for_pair() returns 'SHORT' →
   direction='SHORT' regardless of macro score, conviction boosted by 0.3
5. test_score_pair_unknown_pair — pair not in PAIR_CONFIG → returns None
6. test_score_pair_insufficient_price_history — price series length < 60 → returns None
7. test_score_pair_fv_skip — composite_direction='SKIP' → conviction halved
8. test_compute_hurst_trending — trending series (cumsum of +1s) → hurst > 0.55 → score=0.3
9. test_compute_hurst_mean_reverting — alternating ±1 series → hurst < 0.45 → score=-0.1
10. test_estimate_hold_rate_diff — driver='rate_diff_momentum', high conviction → hold > 0
11. test_scan_all_pairs_returns_top3 — mock score_pair per pair, returns list <= 3

Use pytest fixtures, unittest.mock.patch, and parametrize where it reduces duplication.
No integration tests — all external I/O must be mocked."
  echo "[codex] Task A: done → logs/codex_task_a_result.md"
}

run_task_b() {
  echo "[codex] Task B: cb_calendar.py → starting"
  codex exec \
    -C "$REPO" \
    --full-auto \
    -o "$REPO/logs/codex_task_b_result.md" \
    "Create sovereign/forex/cb_calendar.py.

This module provides central bank meeting calendar data for 2025-2026
and helper functions used by the forex macro engine.

Central banks to include (8 total):
  FED, ECB, BOJ, BOE, SNB, RBA, BOC, RBNZ

Use real, publicly known meeting dates. If a 2026 date is not yet
announced, use the known scheduled window (e.g. 'late January').

Module must export:

1. CB_MEETINGS: dict[str, list[date]]
   Keys are bank names (FED, ECB, etc.), values are sorted lists of
   meeting dates for 2025 and 2026.

2. def get_days_to_next_meeting(bank: str, as_of: date | None = None) -> int
   Returns calendar days until the next scheduled meeting.
   as_of defaults to date.today().
   Returns 999 if no future meeting found.

3. def get_days_since_last_meeting(bank: str, as_of: date | None = None) -> int
   Returns calendar days since the most recent past meeting.
   Returns 999 if no past meeting found.

4. def is_in_blackout_period(bank: str, as_of: date | None = None,
                              blackout_days: int = 10) -> bool
   Returns True if as_of is within blackout_days before a meeting.
   Default blackout window is 10 calendar days.

5. def get_next_meeting(bank: str, as_of: date | None = None) -> date | None
   Returns the next meeting date or None.

Style: follow the conventions in sovereign/forex/pair_universe.py.
Use only stdlib (datetime, typing). No external dependencies.
Add a __all__ list.
Add a simple if __name__ == '__main__' block that prints the next
meeting date for each bank."
  echo "[codex] Task B: done → logs/codex_task_b_result.md"
}

run_task_c() {
  echo "[codex] Task C: forex backtest → starting"
  codex exec \
    -C "$REPO" \
    --full-auto \
    -o "$REPO/logs/codex_task_c_result.md" \
    "Run the forex backtest and save results.

Steps:
1. Run: python3 scripts/run_forex_scan.py --backtest
   Capture all stdout/stderr.

2. Save the raw output to logs/forex_backtest_latest.txt

3. Parse the output and print a summary table sorted by Sharpe ratio
   with columns: PAIR | SHARPE | WIN_RATE | PROFIT_FACTOR | MAX_DD

4. Write a brief (3-5 bullet) interpretation of the results:
   - Which pairs have edge (Sharpe > 1.0)?
   - Any pairs with high win rate but low profit factor (bad RR)?
   - Any pairs to disable based on results?

If the backtest script fails or times out after 120s, report the
error clearly and suggest the most likely fix."
  echo "[codex] Task C: done → logs/codex_task_c_result.md"
}

case "$TASK" in
  task_a) run_task_a ;;
  task_b) run_task_b ;;
  task_c) run_task_c ;;
  all)
    run_task_a &
    PID_A=$!
    run_task_b &
    PID_B=$!
    run_task_c &
    PID_C=$!
    wait $PID_A && echo "[codex] A finished" || echo "[codex] A FAILED"
    wait $PID_B && echo "[codex] B finished" || echo "[codex] B FAILED"
    wait $PID_C && echo "[codex] C finished" || echo "[codex] C FAILED"
    echo "[codex] All tasks complete. Check logs/codex_task_[abc]_result.md"
    ;;
  *) echo "Usage: $0 [task_a|task_b|task_c|all]"; exit 1 ;;
esac
