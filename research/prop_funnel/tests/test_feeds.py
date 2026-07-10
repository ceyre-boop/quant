"""Feed adapter tests (TICK-022 P3) — schemas, R-conversion, stamps, fail-loud."""

import numpy as np
import pytest

from research.prop_funnel import feeds
from research.prop_funnel._lib import EvidenceStamp


def test_carry_oos_pool():
    p = feeds.load_carry_oos()
    assert p.n == 110, f"frozen OOS pool drifted: n={p.n}"
    assert p.stamp is EvidenceStamp.PROVEN_REGIME_FRAGILE
    assert p.base_risk_pct == pytest.approx(0.0075)
    assert 0.10 < p.trades_per_day < 0.35            # ~55 trades/yr on a 252-day clock
    assert "REGIME-FRAGILE" in p.caveat
    # R conversion spot-check against the raw file
    import json
    raw = json.loads(feeds.OOS_POOL.read_text())
    pair = sorted(raw)[0]
    t0 = raw[pair][0]
    assert (t0["pnl_pct"] / t0["risk_pct"]) in p.r_values


def test_carry_decade_pool():
    p = feeds.load_carry_decade()
    assert p.n >= 400                                 # 411 at recon; allow small regen drift upward
    assert p.stamp is EvidenceStamp.PROVEN_REGIME_FRAGILE
    assert 8.0 < p.meta["years_span"] < 11.0


def test_carry_scenario_shifts_sharpe():
    base = feeds.load_carry_oos()
    for target in (0.0, 0.69, 1.25):
        sc = feeds.carry_scenario(base, target)
        assert sc.stamp is EvidenceStamp.SCENARIO
        assert sc.sharpe_ann() == pytest.approx(target, abs=1e-9)
        assert sc.n == base.n


def test_ict_windows():
    for window, expected_n in (("london_a", 28), ("london_all", 60), ("window_B", 102)):
        p = feeds.load_ict_window(window)
        assert p.n == expected_n, f"{window} pool drifted: {p.n}"
        assert p.stamp is EvidenceStamp.UNPROVEN
        assert "UNPROVEN" in p.caveat


def test_live_closed_outcomes_exact_filter():
    p = feeds.load_live_closed_outcomes()
    assert p.n == 27, f"decision-log closed-outcome count changed: {p.n} (was 27 at recon)"
    assert p.meta["wins"] == 3 and p.meta["losses"] == 24
    assert p.stamp is EvidenceStamp.LOW_N_SANITY_ONLY
    assert not p.sufficient                           # must never be MC'd
    with pytest.raises(ValueError):
        p.draw(np.random.default_rng(0), 10)


def test_futures_orb_insufficient():
    p = feeds.load_futures_orb()
    assert p.stamp is EvidenceStamp.UNVALIDATED
    assert not p.sufficient                           # n=2 at recon


def test_synthetic_moments_and_determinism():
    pool = feeds.synthetic_pool(sharpe_ann=1.5, trades_per_day=1.0)
    rng = np.random.default_rng(11)
    x = pool.draw(rng, 200_000)
    s_trade = 1.5 / np.sqrt(252.0)
    assert float(x.mean()) == pytest.approx(s_trade, abs=0.01)
    assert float(x.std()) == pytest.approx(1.0, abs=0.02)
    z = (x - x.mean()) / x.std()
    assert float((z**4).mean()) - 3 > 1.0             # fat tails (t4 excess kurtosis)
    # determinism
    y = feeds.synthetic_pool(1.5, 1.0).draw(np.random.default_rng(11), 1000)
    y2 = feeds.synthetic_pool(1.5, 1.0).draw(np.random.default_rng(11), 1000)
    assert np.array_equal(y, y2)
    # sizing map: 1%/day vol at 4 trades/day -> 0.5% risk per trade
    assert feeds.synthetic_risk_pct(0.01, 4.0) == pytest.approx(0.005)


def test_fail_loud_on_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(feeds, "OOS_POOL", tmp_path / "nope.json")
    with pytest.raises(SystemExit):
        feeds.load_carry_oos()
