# Backtest Throughput — Measured Leaderboard

> 2026-09-27T09:05:23Z · arm · 12 cores · numba NOT INSTALLED — @njit kernels run as pure-Python fallback

**Legacy claim (never measured):** 148,193 backtests/sec  
**Measured single-core (90-bar):** 4,739 backtests/sec  
**Measured parallel ceiling:** 12,571 backtests/sec (below the legacy claim)  
**Best bar-evaluations/sec:** 2,082,762

> ⚠ **numba is INACTIVE on Python 3.14.4** — the `@njit` kernels run as a pure-Python fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a numba-compatible Python (≤3.13), not new code.

| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |
|---|---:|---|---:|---:|---:|
| 90bar | 90 | nojit_fallback_1core | 1 | 4,739 | 426,504 |
| 90bar | 90 | nojit_fallback_12core | 12 | 12,571 | 1,131,353 |
| 90bar | 90 | pure_python_forex | 1 | 1,903 | 171,265 |
| daily | 2,175 | nojit_fallback_1core | 1 | 300 | 651,842 |
| daily | 2,175 | nojit_fallback_12core | 12 | 821 | 1,785,844 |
| daily | 2,175 | pure_python_forex | 1 | 84 | 181,695 |
| 5min | 166,941 | nojit_fallback_1core | 1 | 5 | 847,617 |
| 5min | 166,941 | nojit_fallback_12core | 12 | 12 | 2,082,762 |
| 5min | 166,941 | pure_python_forex | 1 | 0 | 14,351 |
| 1min | 2,970,637 | nojit_fallback_1core | 1 | 0 | 938,913 |
| 1min | 2,970,637 | nojit_fallback_12core | 12 | 0 | 1,480,589 |
| 1min | 2,970,637 | pure_python_forex | 1 | 0 | 73,042 |

_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: heavier data does fewer backtests/sec but ~the same total bar-evaluations._
