"""P3: selection engines on planted toys + universe integrity (HYP-090)."""

import numpy as np
import pytest

from research.modern import selection as sel
from research.modern._lib import daily_sharpe


@pytest.fixture(scope="module")
def uni():
    return sel.VariantUniverse()


def test_universe_shape_and_v015_variant(uni):
    assert uni.V.shape == (5775, uni.D)
    cid, sub = uni.variants[uni.v015_variant]
    assert cid == 384 and sub == (0, 1, 2, 3)
    # v015 variant series == mean of config-384 pair series
    expected = uni.costed[384].mean(axis=0)
    assert np.allclose(uni.V[uni.v015_variant], expected, atol=1e-12)


def test_window_scores_match_direct_sharpe(uni):
    """Prefix-sum rolling Sharpe == direct daily_sharpe on the same slice."""
    scores = uni.window_scores(180)
    rng = np.random.default_rng(5)
    dates = uni.index.values
    for _ in range(20):
        v = int(rng.integers(0, uni.n_variants))
        t = int(rng.integers(400, uni.D))
        start = int(np.searchsorted(dates, dates[t] - np.timedelta64(180, "D"), side="right"))
        window = uni.V[v, start:t + 1]
        expected = daily_sharpe(window) if len(window) >= sel.MIN_WINDOW_OBS else 0.0
        assert scores[v, t] == pytest.approx(expected, abs=1e-9), (v, t)


def test_a1_picks_planted_winner():
    scores = np.zeros((10, 50))
    scores[7, 30:] = 5.0
    days = np.arange(30, 50)
    assert (sel.a1_select(scores, days) == 7).all()


def test_a2_picks_regime_matched_winner():
    """Two regimes; variant 3 wins in regime A, variant 8 in regime B — A2 must
    route by today's regime, not the global recent winner."""
    D = 400
    vectors = np.zeros((D, 5))
    vectors[:, 0] = np.where((np.arange(D) // 50) % 2 == 0, -1.0, 1.0)  # alternating regimes
    scores = np.zeros((10, D))
    scores[3, vectors[:, 0] < 0] = 2.0            # variant 3 shines in regime A days
    scores[8, vectors[:, 0] > 0] = 2.0            # variant 8 in regime B days
    days = np.arange(200, 400)
    got = sel.a2_select(scores, vectors, days, v015_variant=0, k=10)
    regime_today = vectors[days, 0]
    assert (got[regime_today < 0] == 3).all()
    assert (got[regime_today > 0] == 8).all()


def test_a3_deterministic_cross_call():
    days = np.arange(100)
    a = sel.a3_select(5775, days, ("W180", 7))
    b = sel.a3_select(5775, days, ("W180", 7))
    c = sel.a3_select(5775, days, ("W180", 8))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
