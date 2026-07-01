# Backtest Throughput — Measured Leaderboard

> 2026-06-29T22:02:43Z · Apple M4 Pro · 12 cores · numba NOT INSTALLED — @njit kernels run as pure-Python fallback

**Legacy claim (never measured):** 148,193 backtests/sec  
**Measured single-core (90-bar):** 25,223 backtests/sec  
**Measured parallel ceiling:** 123,168 backtests/sec (below the legacy claim)  
**Best bar-evaluations/sec:** 11,085,141

> ⚠ **numba is INACTIVE on Python 3.14.4** — the `@njit` kernels run as a pure-Python fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a numba-compatible Python (≤3.13), not new code.

| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |
|---|---:|---|---:|---:|---:|
| 90bar | 90 | nojit_fallback_1core | 1 | 25,223 | 2,270,067 |
| 90bar | 90 | nojit_fallback_12core | 12 | 123,168 | 11,085,141 |
| 90bar | 90 | pure_python_forex | 1 | 6,963 | 626,702 |

_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: heavier data does fewer backtests/sec but ~the same total bar-evaluations._
