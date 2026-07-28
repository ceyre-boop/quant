# Backtest Throughput — Measured Leaderboard

> 2026-07-26T07:01:16Z · arm · 12 cores · numba NOT INSTALLED — @njit kernels run as pure-Python fallback

**Legacy claim (never measured):** 148,193 backtests/sec  
**Measured single-core (90-bar):** 21,430 backtests/sec  
**Measured parallel ceiling:** 67,214 backtests/sec (below the legacy claim)  
**Best bar-evaluations/sec:** 8,125,884

> ⚠ **numba is INACTIVE on Python 3.14.4** — the `@njit` kernels run as a pure-Python fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a numba-compatible Python (≤3.13), not new code.

| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |
|---|---:|---|---:|---:|---:|
| 90bar | 90 | nojit_fallback_1core | 1 | 21,430 | 1,928,743 |
| 90bar | 90 | nojit_fallback_12core | 12 | 67,214 | 6,049,257 |
| 90bar | 90 | pure_python_forex | 1 | 3,478 | 313,068 |
| daily | 2,175 | nojit_fallback_1core | 1 | 742 | 1,614,341 |
| daily | 2,175 | nojit_fallback_12core | 12 | 3,008 | 6,542,734 |
| daily | 2,175 | pure_python_forex | 1 | 124 | 268,871 |
| 5min | 166,941 | nojit_fallback_1core | 1 | 10 | 1,699,827 |
| 5min | 166,941 | nojit_fallback_12core | 12 | 45 | 7,574,842 |
| 5min | 166,941 | pure_python_forex | 1 | 2 | 324,800 |
| 1min | 2,970,637 | nojit_fallback_1core | 1 | 1 | 1,940,836 |
| 1min | 2,970,637 | nojit_fallback_12core | 12 | 3 | 8,125,884 |
| 1min | 2,970,637 | pure_python_forex | 1 | 0 | 456,923 |

_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: heavier data does fewer backtests/sec but ~the same total bar-evaluations._
