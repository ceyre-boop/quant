"""Date-block bootstrap: stationary blocks over ORDERED UNIQUE DATES; every row sharing a date
is carried together. This is the fix for the HYP-110 flaw (pooled-event resampling that
treated same-date cross-instrument observations as independent). Politis–Romano indices
follow research/modern/_lib.py::_stationary_block_indices exactly."""
from __future__ import annotations

import numpy as np


def stationary_block_indices(rng: np.random.Generator, n: int, mean_len: int) -> np.ndarray:
    idx = np.empty(n, dtype=np.int64)
    p = 1.0 / mean_len
    pos = 0
    while pos < n:
        start = int(rng.integers(0, n))
        length = min(int(rng.geometric(p)), n - pos)
        idx[pos:pos + length] = (start + np.arange(length)) % n
        pos += length
    return idx


def date_groups(dates) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (unique sorted dates, list of row-index arrays per date, same order)."""
    d = np.asarray(dates)
    uniq = np.unique(d)
    groups = [np.flatnonzero(d == u) for u in uniq]
    return uniq, groups


def date_block_bootstrap(dates, stat, *, L: int = 5, draws: int = 10000, seed: int = 42,
                         rng=None) -> np.ndarray:
    """stat(row_indices: np.ndarray) -> float, evaluated on `draws` resamples where whole
    dates are resampled in stationary blocks and all rows of each drawn date are included."""
    rng = rng or np.random.default_rng(seed)
    uniq, groups = date_groups(dates)
    n = len(uniq)
    out = np.empty(draws)
    for k in range(draws):
        di = stationary_block_indices(rng, n, L)
        rows = np.concatenate([groups[i] for i in di])
        out[k] = stat(rows)
    return out


def ci95(v: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)
