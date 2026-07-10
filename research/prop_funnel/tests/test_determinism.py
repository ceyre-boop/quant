"""Determinism (TICK-022): same seed -> identical canonical results."""

import numpy as np

from research.prop_funnel import feeds
from research.prop_funnel._lib import canonical_dumps
from research.prop_funnel.funnel import run_funnel
from research.prop_funnel.rulesets import FirmSpec


def _run(seed: int) -> str:
    spec = FirmSpec.load("FTMO_100K_SWING")
    pool = feeds.synthetic_pool(sharpe_ann=1.0, trades_per_day=1.0)
    row = run_funnel(np.random.default_rng(seed), pool, spec,
                     n_attempts=800, n_funded_sims=800)
    return canonical_dumps(row)


def test_same_seed_identical():
    assert _run(7) == _run(7)


def test_different_seed_differs():
    assert _run(7) != _run(8)


def test_empirical_pool_deterministic():
    spec = FirmSpec.load("MFF_100K")
    pool = feeds.load_carry_oos()
    a = canonical_dumps(run_funnel(np.random.default_rng(3), pool, spec,
                                   n_attempts=500, n_funded_sims=500))
    b = canonical_dumps(run_funnel(np.random.default_rng(3), pool, spec,
                                   n_attempts=500, n_funded_sims=500))
    assert a == b
