# Backtest Throughput — Measured Leaderboard

> 2026-07-19T07:00:47Z · arm · 12 cores · numba NOT INSTALLED — @njit kernels run as pure-Python fallback

**Legacy claim (never measured):** 148,193 backtests/sec  
**Measured single-core (90-bar):** 24,885 backtests/sec  
**Measured parallel ceiling:** 122,186 backtests/sec (below the legacy claim)  
**Best bar-evaluations/sec:** 17,228,945

> ⚠ **numba is INACTIVE on Python 3.14.4** — the `@njit` kernels run as a pure-Python fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a numba-compatible Python (≤3.13), not new code.

| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |
|---|---:|---|---:|---:|---:|
| 90bar | 90 | nojit_fallback_1core | 1 | 24,885 | 2,239,611 |
| 90bar | 90 | nojit_fallback_12core | 12 | 122,186 | 10,996,784 |
| 90bar | 90 | pure_python_forex | 1 | 6,720 | 604,817 |
| daily | 2,175 | nojit_fallback_1core | 1 | 1,037 | 2,256,137 |
| daily | 2,175 | nojit_fallback_12core | 12 | 6,654 | 14,473,537 |
| daily | 2,175 | pure_python_forex | 1 | 241 | 523,874 |
| 5min | 166,941 | nojit_fallback_1core | 1 | 14 | 2,271,851 |
| 5min | 166,941 | nojit_fallback_12core | 12 | 98 | 16,311,713 |
| 5min | 166,941 | pure_python_forex | 1 | 3 | 530,737 |
| 1min | 2,970,637 | nojit_fallback_1core | 1 | 1 | 2,532,593 |
| 1min | 2,970,637 | nojit_fallback_12core | 12 | 6 | 17,228,945 |
| 1min | 2,970,637 | pure_python_forex | 1 | 0 | 621,513 |

_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: heavier data does fewer backtests/sec but ~the same total bar-evaluations._
