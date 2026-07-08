"""sovereign/discovery/cpcv.py — Combinatorial Purged Cross-Validation (AFML Ch. 12).

López de Prado's CPCV generalises purged k-fold: partition the time-ordered observations
into ``n_groups`` contiguous groups, then for EVERY combination of ``test_groups`` groups
held out as test, the remaining groups form the train fold — PURGED of any train
observation whose label interval [entry, exit] overlaps a test group's interval, with an
optional EMBARGO on the observations immediately following each test group.

Number of train/test splits      = C(n_groups, test_groups)
Number of reconstructable paths   = C(n_groups, test_groups) · test_groups / n_groups

This module is deliberately dependency-light (numpy + stdlib only) so research scripts can
import it without pulling sklearn/xgboost side effects. It generalises the trade-granularity
purge already used inline by ``scripts/research/exit_regime_conditioning._purged_folds``.

Embargo is positional (in entry-time order), so the splitter is dtype-agnostic: ``entry_dt``
and ``exit_dt`` may be ``datetime64`` arrays or any comparable numeric type.
"""
from __future__ import annotations

import itertools
from math import comb
from typing import Iterator

import numpy as np

__all__ = ["combinatorial_purged_splits", "n_cpcv_splits", "n_backtest_paths"]


def n_cpcv_splits(n_groups: int = 6, test_groups: int = 2) -> int:
    """C(n_groups, test_groups) — how many train/test folds the splitter yields."""
    return comb(n_groups, test_groups)


def n_backtest_paths(n_groups: int = 6, test_groups: int = 2) -> int:
    """C(n_groups, test_groups) · test_groups / n_groups — reconstructable backtest paths."""
    return comb(n_groups, test_groups) * test_groups // n_groups


def _overlaps(entry_i, exit_i, t0, t1) -> bool:
    """True if label interval [entry_i, exit_i] overlaps the test interval [t0, t1]."""
    return exit_i >= t0 and entry_i <= t1


def combinatorial_purged_splits(
    entry_dt,
    exit_dt,
    n_groups: int = 6,
    test_groups: int = 2,
    embargo_frac: float = 0.0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) for every combination of ``test_groups`` held-out groups.

    Args:
        entry_dt, exit_dt: per-observation label start/end (datetime64 or numeric), len N.
        n_groups: number of contiguous time-ordered groups to partition into.
        test_groups: groups held out as test per combination (1 ≤ test_groups < n_groups).
        embargo_frac: fraction of N observations to embargo (drop from train) immediately
            after each test group in entry-time order. 0.0 disables the embargo.

    Yields:
        (train_idx, test_idx): int ndarrays of ORIGINAL positions. test_idx is sorted.
        train_idx is purged of test-overlapping intervals and the embargo window.
    """
    entry_dt = np.asarray(entry_dt)
    exit_dt = np.asarray(exit_dt)
    n = len(entry_dt)
    if n != len(exit_dt):
        raise ValueError(f"entry_dt/exit_dt length mismatch: {n} vs {len(exit_dt)}")
    if not (1 <= test_groups < n_groups):
        raise ValueError(f"need 1 <= test_groups ({test_groups}) < n_groups ({n_groups})")
    if n < n_groups:
        raise ValueError(f"need at least n_groups ({n_groups}) observations, got {n}")

    order = np.argsort(entry_dt, kind="stable")          # original idx in entry-time order
    pos_of = np.empty(n, dtype=int)
    pos_of[order] = np.arange(n)                          # original idx -> rank in order
    group_pos = np.array_split(np.arange(n), n_groups)   # ranks per group (contiguous)
    groups_idx = [order[p] for p in group_pos]           # original idx per group
    embargo = int(np.ceil(embargo_frac * n)) if embargo_frac > 0 else 0

    for combo in itertools.combinations(range(n_groups), test_groups):
        combo_set = set(combo)
        test = np.concatenate([groups_idx[g] for g in combo])
        intervals = [(entry_dt[groups_idx[g]].min(), exit_dt[groups_idx[g]].max()) for g in combo]

        embargo_pos: set[int] = set()
        if embargo:
            for g in combo:
                last = group_pos[g][-1]
                embargo_pos.update(range(last + 1, min(last + 1 + embargo, n)))

        train_pool = np.concatenate(
            [groups_idx[g] for g in range(n_groups) if g not in combo_set]
        )
        keep = [
            i for i in train_pool
            if pos_of[i] not in embargo_pos
            and not any(_overlaps(entry_dt[i], exit_dt[i], t0, t1) for t0, t1 in intervals)
        ]
        yield np.asarray(keep, dtype=int), np.sort(test).astype(int)
