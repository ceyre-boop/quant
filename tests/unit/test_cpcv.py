"""tests/unit/test_cpcv.py — Combinatorial Purged Cross-Validation (AFML Ch. 12).

Locks: split count = C(n_groups, test_groups), path count, the AFML purge (no train
observation overlaps a test interval), positional embargo, determinism, and input guards.
"""
from __future__ import annotations

import itertools
from math import comb

import numpy as np
import pytest

from sovereign.discovery.cpcv import (
    combinatorial_purged_splits,
    n_backtest_paths,
    n_cpcv_splits,
)


def _toy(n=60, span_days=1, hold=3):
    """n non-overlapping trades, entry every `span_days`, each lasting `hold` days."""
    entry = np.array([np.datetime64("2015-01-01") + np.timedelta64(i * span_days, "D") for i in range(n)])
    exit_ = entry + np.timedelta64(hold, "D")
    return entry, exit_


def test_split_and_path_counts():
    assert n_cpcv_splits(6, 2) == comb(6, 2) == 15
    assert n_backtest_paths(6, 2) == 15 * 2 // 6 == 5
    entry, exit_ = _toy()
    splits = list(combinatorial_purged_splits(entry, exit_, n_groups=6, test_groups=2))
    assert len(splits) == 15


def test_purge_removes_overlap():
    # overlapping trades (hold longer than spacing) so train/test can collide at boundaries
    entry, exit_ = _toy(n=60, span_days=1, hold=5)
    for train, test in combinatorial_purged_splits(entry, exit_, n_groups=6, test_groups=2, embargo_frac=0.0):
        if len(test) == 0:
            continue
        # AFML purge: no surviving train observation may overlap ANY held-out test observation.
        # (Checked pairwise — non-adjacent test groups legitimately keep the train group between them.)
        for i in train:
            for j in test:
                assert not (exit_[i] >= entry[j] and entry[i] <= exit_[j]), "purge left a train/test overlap"
        # train and test are disjoint
        assert set(train.tolist()).isdisjoint(test.tolist())


def test_every_observation_tested_expected_times():
    entry, exit_ = _toy()
    n_groups, test_groups = 6, 2
    seen = np.zeros(len(entry), dtype=int)
    for _, test in combinatorial_purged_splits(entry, exit_, n_groups, test_groups):
        seen[test] += 1
    # each group appears in C(n_groups-1, test_groups-1) combinations
    assert set(seen.tolist()) == {comb(n_groups - 1, test_groups - 1)}


def test_embargo_drops_following_observations():
    entry, exit_ = _toy(n=60, span_days=1, hold=1)  # non-overlapping → only embargo can purge
    no_emb = list(combinatorial_purged_splits(entry, exit_, 6, 2, embargo_frac=0.0))
    with_emb = list(combinatorial_purged_splits(entry, exit_, 6, 2, embargo_frac=0.1))
    # with a non-overlapping toy, purge alone removes nothing; embargo must shrink some train folds
    total_no = sum(len(tr) for tr, _ in no_emb)
    total_emb = sum(len(tr) for tr, _ in with_emb)
    assert total_emb < total_no


def test_determinism():
    entry, exit_ = _toy()
    a = list(combinatorial_purged_splits(entry, exit_, 6, 2, embargo_frac=0.01))
    b = list(combinatorial_purged_splits(entry, exit_, 6, 2, embargo_frac=0.01))
    assert len(a) == len(b)
    for (tr1, te1), (tr2, te2) in zip(a, b):
        assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2)


def test_numeric_dtype_supported():
    # dtype-agnostic: plain integer "times" work (positional embargo, interval purge)
    entry = np.arange(60)
    exit_ = entry + 2
    splits = list(combinatorial_purged_splits(entry, exit_, 6, 2))
    assert len(splits) == 15


@pytest.mark.parametrize("kw", [
    {"n_groups": 6, "test_groups": 6},   # test_groups must be < n_groups
    {"n_groups": 6, "test_groups": 0},   # test_groups must be >= 1
])
def test_invalid_params_raise(kw):
    entry, exit_ = _toy(n=20)
    with pytest.raises(ValueError):
        list(combinatorial_purged_splits(entry, exit_, **kw))


def test_too_few_observations_raises():
    entry, exit_ = _toy(n=4)
    with pytest.raises(ValueError):
        list(combinatorial_purged_splits(entry, exit_, n_groups=6, test_groups=2))
