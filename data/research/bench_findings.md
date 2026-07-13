# Backtest Throughput — Measured Leaderboard

> 2026-07-12T07:00:50Z · arm · 12 cores · numba NOT INSTALLED — @njit kernels run as pure-Python fallback

**Legacy claim (never measured):** 148,193 backtests/sec  
**Measured single-core (90-bar):** 21,692 backtests/sec  
**Measured parallel ceiling:** 116,650 backtests/sec (below the legacy claim)  
**Best bar-evaluations/sec:** 15,699,027

> ⚠ **numba is INACTIVE on Python 3.14.4** — the `@njit` kernels run as a pure-Python fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a numba-compatible Python (≤3.13), not new code.

| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |
|---|---:|---|---:|---:|---:|
| 90bar | 90 | nojit_fallback_1core | 1 | 21,692 | 1,952,284 |
| 90bar | 90 | nojit_fallback_12core | 12 | 116,650 | 10,498,472 |
| 90bar | 90 | pure_python_forex | 1 | 6,184 | 556,538 |
| daily | 2,175 | nojit_fallback_1core | 1 | 963 | 2,093,809 |
| daily | 2,175 | nojit_fallback_12core | 12 | 6,424 | 13,972,680 |
| daily | 2,175 | pure_python_forex | 1 | 224 | 487,175 |
| 5min | 166,941 | nojit_fallback_1core | 1 | 13 | 2,192,201 |
| 5min | 166,941 | nojit_fallback_12core | 12 | 94 | 15,699,027 |
| 5min | 166,941 | pure_python_forex | 1 | 3 | 493,161 |
| 1min | 2,970,637 | nojit_fallback_1core | 1 | 1 | 2,248,976 |
| 1min | 2,970,637 | nojit_fallback_12core | 12 | 5 | 15,451,818 |
| 1min | 2,970,637 | pure_python_forex | 1 | 0 | 551,250 |

_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: heavier data does fewer backtests/sec but ~the same total bar-evaluations._
