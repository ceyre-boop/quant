# Backtest Throughput — Measured Leaderboard

> 2026-08-02T07:00:50Z · arm · 12 cores · numba NOT INSTALLED — @njit kernels run as pure-Python fallback

**Legacy claim (never measured):** 148,193 backtests/sec  
**Measured single-core (90-bar):** 23,004 backtests/sec  
**Measured parallel ceiling:** 73,449 backtests/sec (below the legacy claim)  
**Best bar-evaluations/sec:** 10,888,507

> ⚠ **numba is INACTIVE on Python 3.14.4** — the `@njit` kernels run as a pure-Python fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a numba-compatible Python (≤3.13), not new code.

| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |
|---|---:|---|---:|---:|---:|
| 90bar | 90 | nojit_fallback_1core | 1 | 23,004 | 2,070,393 |
| 90bar | 90 | nojit_fallback_12core | 12 | 73,449 | 6,610,420 |
| 90bar | 90 | pure_python_forex | 1 | 3,916 | 352,428 |
| daily | 2,175 | nojit_fallback_1core | 1 | 957 | 2,081,099 |
| daily | 2,175 | nojit_fallback_12core | 12 | 3,780 | 8,222,077 |
| daily | 2,175 | pure_python_forex | 1 | 176 | 382,573 |
| 5min | 166,941 | nojit_fallback_1core | 1 | 13 | 2,210,920 |
| 5min | 166,941 | nojit_fallback_12core | 12 | 58 | 9,656,393 |
| 5min | 166,941 | pure_python_forex | 1 | 2 | 405,874 |
| 1min | 2,970,637 | nojit_fallback_1core | 1 | 1 | 2,323,617 |
| 1min | 2,970,637 | nojit_fallback_12core | 12 | 4 | 10,888,507 |
| 1min | 2,970,637 | pure_python_forex | 1 | 0 | 562,109 |

_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: heavier data does fewer backtests/sec but ~the same total bar-evaluations._
