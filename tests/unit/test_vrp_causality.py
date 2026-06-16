"""VRP causality tests — the single most important correctness guarantee.

The Stage-2 harvest return MUST be strictly causal. BTZ's existence statistic legitimately
uses forward realized variance (look-ahead), but that look-ahead must be QUARANTINED to
Stage 1 and never leak into the Stage-2 correlation/Sharpe series. If it leaks, the
orthogonality kill-gate correlates clairvoyant returns and the verdict is worthless.
"""
import numpy as np
import pandas as pd

from sovereign.research.vrp import vrp_calculator as vc


def _synthetic(n=160, seed=0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, n)
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)
    vol = pd.Series(20.0 + rng.normal(0, 1.5, n), index=idx)   # vol index in points
    return close, vol


def test_harvest_return_is_strictly_causal():
    """Perturbing a FUTURE close must not change any earlier harvest return."""
    close, vol = _synthetic()
    h0 = vc.harvest_return_causal(vol, close)

    close2 = close.copy()
    close2.iloc[-1] *= 1.10                      # shock only the final (future) bar
    h1 = vc.harvest_return_causal(vol, close2)

    common = h0.index.intersection(h1.index)[:-1]   # every day except the perturbed one
    assert len(common) > 100
    assert np.allclose(h0.loc[common].to_numpy(), h1.loc[common].to_numpy()), (
        "LOOK-AHEAD LEAK: a future close changed a past harvest return")


def test_harvest_uses_prior_day_implied_vol():
    """harvest(t) must use IV_{t-1}; changing today's vol only must not move today's harvest."""
    close, vol = _synthetic(seed=3)
    h0 = vc.harvest_return_causal(vol, close)

    vol2 = vol.copy()
    vol2.iloc[-1] += 25.0                        # spike only today's vol index
    h1 = vc.harvest_return_causal(vol2, close)

    common = h0.index.intersection(h1.index)
    assert np.allclose(h0.loc[common].to_numpy(), h1.loc[common].to_numpy()), (
        "harvest(t) reacted to IV(t) — it must use IV(t-1), the premium sold at prior close")


def test_btz_gap_is_forward_looking_by_design():
    """Sanity-check the intended asymmetry: the existence gap DOES depend on future returns
    (that is why it is quarantined to Stage 1)."""
    close, vol = _synthetic(seed=7)
    g0 = vc.btz_vrp_gap(vol, close, window=21)

    close2 = close.copy()
    close2.iloc[-1] *= 1.20                       # large future move inflates realized var
    g1 = vc.btz_vrp_gap(vol, close2, window=21)

    common = g0.index.intersection(g1.index)
    assert not np.allclose(g0.loc[common].to_numpy(), g1.loc[common].to_numpy()), (
        "BTZ gap should be forward-looking; if it is not, the existence measure is mis-specified")
