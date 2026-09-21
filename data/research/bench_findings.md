# Backtest Throughput — Measured Leaderboard

> 2026-09-20T07:00:49Z · arm · 12 cores · numba NOT INSTALLED — @njit kernels run as pure-Python fallback

**Legacy claim (never measured):** 148,193 backtests/sec  
**Measured single-core (90-bar):** 23,017 backtests/sec  
**Measured parallel ceiling:** 104,984 backtests/sec (below the legacy claim)  
**Best bar-evaluations/sec:** 15,792,403

> ⚠ **numba is INACTIVE on Python 3.14.4** — the `@njit` kernels run as a pure-Python fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a numba-compatible Python (≤3.13), not new code.

| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |
|---|---:|---|---:|---:|---:|
| 90bar | 90 | nojit_fallback_1core | 1 | 23,017 | 2,071,507 |
| 90bar | 90 | nojit_fallback_12core | 12 | 104,984 | 9,448,590 |
| 90bar | 90 | pure_python_forex | 1 | 6,232 | 560,927 |
| daily | 2,175 | nojit_fallback_1core | 1 | 970 | 2,109,139 |
| daily | 2,175 | nojit_fallback_12core | 12 | 6,226 | 13,540,639 |
| daily | 2,175 | pure_python_forex | 1 | 234 | 508,221 |
| 5min | 166,941 | nojit_fallback_1core | 1 | 13 | 2,209,467 |
| 5min | 166,941 | nojit_fallback_12core | 12 | 88 | 14,618,498 |
| 5min | 166,941 | pure_python_forex | 1 | 3 | 509,405 |
| 1min | 2,970,637 | nojit_fallback_1core | 1 | 1 | 2,351,181 |
| 1min | 2,970,637 | nojit_fallback_12core | 12 | 5 | 15,792,403 |
| 1min | 2,970,637 | pure_python_forex | 1 | 0 | 601,466 |

_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: heavier data does fewer backtests/sec but ~the same total bar-evaluations._
