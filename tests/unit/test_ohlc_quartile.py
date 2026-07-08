"""OHLC-quartile encoder + feature-matrix tests.

Two jobs: (1) confirm the quartile encoding is correct and deterministic, and
(2) LOCK the no-lookahead property — appending future bars must NOT change earlier
feature rows. The actual edge (if any) is the validator's job, never a unit assert.
"""
import numpy as np
import pandas as pd

from sovereign.research import ohlc_quartile as oq


def _synthetic_5min(n_days=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03 09:30", periods=n_days * 78, freq="5min", tz="UTC")
    n = len(idx)
    close = 4000 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    o = np.empty(n); o[0] = close[0]; o[1:] = close[:-1]
    wig = np.abs(rng.normal(0, 1.5, n))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, close) + wig,
                         "Low": np.minimum(o, close) - wig, "Close": close,
                         "Volume": rng.integers(1, 100, n)}, index=idx)


# ── (1) encoder correctness ───────────────────────────────────────────────────
def test_encode_quartiles_known_values():
    # quantiles of [1,2,3,4] at 25/50/75/100 = 1.75/2.5/3.25/4.0; /range(4) normalized
    out = oq.encode_bar_quartiles(np.array([1, 2, 3, 4]), ref_low=0.0, ref_range=4.0)
    assert out == [0.4375, 0.625, 0.8125, 1.0]


def test_encode_monotonic_nondecreasing():
    out = oq.encode_bar_quartiles(np.array([10, 5, 8, 3, 9]), ref_low=0.0, ref_range=10.0)
    assert out == sorted(out)  # q25 <= q50 <= q75 <= q100 always


def test_encode_empty_and_zero_range_safe():
    assert oq.encode_bar_quartiles(np.array([]), 0.0, 5.0) == [0, 0, 0, 0]
    assert oq.encode_bar_quartiles(np.array([1, 2, 3]), 0.0, 0.0) == [0, 0, 0, 0]


# ── (2) feature matrix + no-lookahead ─────────────────────────────────────────
def test_feature_matrix_builds_with_expected_columns():
    feat = oq.build_feature_matrix(_synthetic_5min())
    assert feat is not None and len(feat) > 30
    for c in oq.FEATURE_COLS + ["y", "ret"]:
        assert c in feat.columns
    assert set(feat["y"].unique()).issubset({0, 1})
    assert not feat[oq.FEATURE_COLS].isna().any().any()


def test_no_lookahead_earlier_rows_invariant_to_future_bars():
    df = _synthetic_5min(n_days=160)
    cut = len(df) // 2
    feat_partial = oq.build_feature_matrix(df.iloc[:cut])
    feat_full = oq.build_feature_matrix(df)
    assert feat_partial is not None and feat_full is not None
    # Every row present in the partial build must be identical in the full build —
    # i.e. adding future bars cannot retroactively change a past feature row.
    common = feat_partial.index.intersection(feat_full.index)
    assert len(common) > 10
    a = feat_partial.loc[common, oq.FEATURE_COLS]
    b = feat_full.loc[common, oq.FEATURE_COLS]
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)


def test_insufficient_data_returns_none():
    assert oq.build_feature_matrix(_synthetic_5min(n_days=1)) is None
    assert oq.build_feature_matrix(None) is None
