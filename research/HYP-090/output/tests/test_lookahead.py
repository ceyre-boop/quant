"""P3: truncation invariance + regime causality (HYP-090 prereg lock).

Selections at day t must be identical when every input after t is deleted and
all derived structures are rebuilt from the truncated arrays."""

import numpy as np
import pytest

from research.modern import regimes, selection as sel


@pytest.fixture(scope="module")
def uni():
    return sel.VariantUniverse()


@pytest.fixture(scope="module")
def zfeat(uni):
    return regimes.build_features(uni.index)


def _truncated_universe(uni, t):
    """Rebuild a universe view truncated at day t (inclusive)."""
    import copy
    u = copy.copy(uni)
    u.costed = uni.costed[:, :, :t + 1]
    u.V = uni.V[:, :t + 1]
    u.D = t + 1
    u._cum1 = uni._cum1[:, :t + 1]
    u._cum2 = uni._cum2[:, :t + 1]
    u.index = uni.index[:t + 1]
    dates = u.index.values
    u._starts = {W: np.searchsorted(dates, dates - np.timedelta64(W, "D"), side="right")
                 for W in sel.WINDOWS}
    return u


def test_truncation_invariance_a1_a2(uni, zfeat):
    rng = np.random.default_rng(9)
    ts = rng.integers(800, uni.D - 5, size=20)
    for W in (90, 365):
        full_scores = uni.window_scores(W)
        vec_full = regimes.regime_vectors(zfeat, W)
        for t in ts:
            t = int(t)
            # full-history selection at t
            a1_full = sel.a1_select(full_scores, np.array([t]))[0]
            a2_full = sel.a2_select(full_scores, vec_full, np.array([t]),
                                    uni.v015_variant)[0]
            # truncated rebuild
            u_tr = _truncated_universe(uni, t)
            tr_scores = u_tr.window_scores(W)
            z_tr = regimes.build_features(uni.index[:t + 1])
            vec_tr = regimes.regime_vectors(z_tr, W)
            a1_tr = sel.a1_select(tr_scores, np.array([t]))[0]
            a2_tr = sel.a2_select(tr_scores, vec_tr, np.array([t]),
                                  uni.v015_variant)[0]
            assert a1_full == a1_tr, f"A1 look-ahead at t={t} W={W}"
            assert a2_full == a2_tr, f"A2 look-ahead at t={t} W={W}"


def test_regime_features_causal(uni, zfeat):
    """Feature values at day t are invariant under future-data deletion."""
    for t in (900, 1500, 2200):
        z_tr = regimes.build_features(uni.index[:t + 1])
        full_row = zfeat.iloc[t].to_numpy()
        tr_row = z_tr.iloc[t].to_numpy()
        both_nan = np.isnan(full_row) & np.isnan(tr_row)
        assert np.allclose(full_row[~both_nan], tr_row[~both_nan], atol=1e-12, equal_nan=True)
