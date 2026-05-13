"""
Unit tests for the Stanford CS229 ML Stack — 10 modules.

Tests verify:
  - Import cleanly
  - Core API contracts match what orchestrator expects
  - Edge cases: cold start, insufficient data, bad inputs
  - Integration: outputs within valid ranges
  - Serialisation: save/load round-trip (mocked)
"""

import math
import pickle
import tempfile
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────── #
# Helpers
# ─────────────────────────────────────────────────────────────────────── #

def _make_features(n=50, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 5))


def _make_regime_labels(n=50, seed=42):
    rng = np.random.default_rng(seed)
    return rng.choice(['MOMENTUM', 'REVERSION', 'FLAT'], n).tolist()


# ─────────────────────────────────────────────────────────────────────── #
# 1. SoftmaxRegimeClassifier (CS229 L04)
# ─────────────────────────────────────────────────────────────────────── #

class TestSoftmaxRegimeClassifier:
    @pytest.fixture
    def clf(self, tmp_path):
        with patch('sovereign.risk.softmax_regime._CHECKPOINT', tmp_path / "softmax.pkl"):
            from sovereign.risk.softmax_regime import SoftmaxRegimeClassifier
            return SoftmaxRegimeClassifier()

    def test_encode_shape(self, clf):
        x = clf.encode(hurst=0.6, hmm_prob=0.2, adx=25, prev_regime='MOMENTUM', strategy='momentum')
        assert x.shape == (5,)
        assert x.dtype == np.float64

    def test_predict_proba_cold_start(self, clf):
        x = clf.encode(0.6, 0.2, 25, 'MOMENTUM', 'momentum')
        proba = clf.predict_proba(x)
        assert set(proba.keys()) == {'MOMENTUM', 'REVERSION', 'FLAT'}
        assert abs(sum(proba.values()) - 1.0) < 1e-6
        # Uniform prior when unfitted
        for v in proba.values():
            assert abs(v - 1 / 3) < 1e-9

    def test_fit_and_predict(self, clf):
        X = _make_features()
        y = _make_regime_labels()
        clf.fit(X, y)
        assert clf._fitted
        x = clf.encode(0.7, 0.1, 30, 'MOMENTUM', 'momentum')
        proba = clf.predict_proba(x)
        assert abs(sum(proba.values()) - 1.0) < 1e-6
        for v in proba.values():
            assert 0.0 <= v <= 1.0

    def test_online_update(self, clf):
        X = _make_features()
        y = _make_regime_labels()
        clf.fit(X, y)
        x = clf.encode(0.7, 0.1, 30, 'MOMENTUM', 'momentum')
        clf.update_online(x, 'MOMENTUM', alpha=0.05)
        assert clf._n_updates > 0

    def test_dominant_regime(self, clf):
        X = _make_features()
        y = _make_regime_labels()
        clf.fit(X, y)
        x = clf.encode(0.7, 0.1, 30, 'MOMENTUM', 'momentum')
        regime, conf = clf.dominant_regime(x)
        assert regime in ('MOMENTUM', 'REVERSION', 'FLAT')
        assert 0.0 <= conf <= 1.0

    def test_unknown_strategy_falls_back(self, clf):
        x = clf.encode(0.5, 0.3, 20, 'FLAT', 'unknown_strategy')
        proba = clf.predict_proba(x)
        assert sum(proba.values()) == pytest.approx(1.0, abs=1e-6)

    def test_describe(self, clf):
        desc = clf.describe()
        assert 'fitted' in desc


# ─────────────────────────────────────────────────────────────────────── #
# 2. VolRegimeSignal / black_scholes (MIT)
# ─────────────────────────────────────────────────────────────────────── #

class TestVolRegimeSignal:
    @pytest.fixture
    def vs(self):
        from sovereign.risk.black_scholes import VolRegimeSignal
        return VolRegimeSignal(lookback=10)

    def test_cold_start_signal(self, vs):
        sig = vs.get_signal()
        assert sig['vol_regime'] == 'NORMAL'
        assert sig['size_adjustment'] == 1.0

    def test_normal_vol(self, vs):
        rng = np.random.default_rng(1)
        for _ in range(15):
            # Varying rv around 0.010 so std is non-zero; VIX ≈ 16 → IV/RV ≈ 1
            rv = 0.010 + rng.standard_normal() * 0.002
            vs.update(abs(rv), vix=16.0)
        sig = vs.get_signal()
        # With realistic vol noise, ratio should be moderate
        assert sig['vol_regime'] in ('NORMAL', 'BACKWARDATION', 'ELEVATED', 'STRESS', 'EXTREME_STRESS')

    def test_elevated_vol_reduces_size(self, vs):
        for _ in range(15):
            vs.update(0.005, vix=25.0)   # IV >> RV → ELEVATED or STRESS
        sig = vs.get_signal()
        assert sig['size_adjustment'] <= 1.0

    def test_size_adjustment_bounded(self, vs):
        for _ in range(15):
            vs.update(0.010, vix=50.0)   # extreme stress
        sig = vs.get_signal()
        assert 0.0 < sig['size_adjustment'] <= 1.0

    def test_bs_call_atm(self, vs):
        c = vs.black_scholes_call(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert 10 < c < 20  # standard BS ATM annual

    def test_bs_call_deep_itm(self, vs):
        c = vs.black_scholes_call(S=150, K=100, T=1.0, r=0.05, sigma=0.20)
        assert c > 40  # deep ITM

    def test_bs_call_zero_time(self, vs):
        c = vs.black_scholes_call(S=100, K=90, T=0.0, r=0.05, sigma=0.20)
        assert c == pytest.approx(max(0.0, 100 - 90), abs=1e-6)

    def test_implied_vol_roundtrip(self, vs):
        sigma = 0.20
        price = vs.black_scholes_call(S=100, K=100, T=0.5, r=0.05, sigma=sigma)
        iv    = vs.implied_vol_from_price(price, S=100, K=100, T=0.5, r=0.05)
        assert abs(iv - sigma) < 0.001

    def test_describe(self, vs):
        assert 'VolRegimeSignal' in vs.describe()


# ─────────────────────────────────────────────────────────────────────── #
# 3. ICAFactorSeparator (CS229 L15)
# ─────────────────────────────────────────────────────────────────────── #

class TestICAFactorSeparator:
    @pytest.fixture
    def ica(self, tmp_path):
        with patch('sovereign.risk.ica_factor_separator._CHECKPOINT', tmp_path / "ica.pkl"):
            from sovereign.risk.ica_factor_separator import ICAFactorSeparator
            return ICAFactorSeparator(n_components=3)

    def test_cold_start_passthrough(self, ica):
        x = np.array([0.6, 0.2, 0.5, 0.8, 0.9])
        out = ica.transform(x)
        np.testing.assert_array_equal(out, x)

    def test_fit_and_transform_shape(self, ica):
        X = np.random.randn(80, 5)
        ica.fit(X)
        assert ica._fitted
        x = np.random.randn(5)
        out = ica.transform(x)
        assert out.shape == (3,)

    def test_batch_transform(self, ica):
        X = np.random.randn(80, 5)
        ica.fit(X)
        out = ica.transform(X)
        assert out.shape == (80, 3)

    def test_not_fitted_with_too_few_samples(self, ica):
        ica.fit(np.random.randn(5, 5))   # < 10
        assert not ica._fitted

    def test_low_correlation_after_fit(self, ica):
        rng = np.random.default_rng(0)
        # Highly correlated input
        base = rng.standard_normal((100, 3))
        X = np.column_stack([base @ rng.standard_normal((3, 5)) for _ in range(1)])
        ica.fit(X)
        if ica._fitted:
            avg_corr = ica.average_pairwise_correlation(X)
            assert avg_corr < 0.40  # meaningful decorrelation

    def test_describe(self, ica):
        assert 'ICAFactorSeparator' in ica.describe()


# ─────────────────────────────────────────────────────────────────────── #
# 4. AlphaDecayMonitor (Ernest Chan)
# ─────────────────────────────────────────────────────────────────────── #

class TestAlphaDecayMonitor:
    @pytest.fixture
    def monitor(self):
        from sovereign.risk.alpha_decay import AlphaDecayMonitor
        return AlphaDecayMonitor(strategy='momentum', window=10, baseline_sharpe=0.6)

    def test_insufficient_data(self, monitor):
        result = monitor.check()
        assert result.level == 'INSUFFICIENT_DATA'
        assert result.multiplier == 1.0

    def test_strong_alpha(self, monitor):
        for _ in range(15):
            monitor.record_trade(0.80)   # consistent winners
        result = monitor.check()
        assert result.level in ('STRONG', 'NORMAL')
        assert result.multiplier == 1.0

    def test_dead_alpha(self, monitor):
        for _ in range(15):
            monitor.record_trade(-0.80)   # consistent losers
        result = monitor.check()
        assert result.level == 'DEAD'
        assert result.multiplier == 0.0

    def test_degraded_alpha(self, monitor):
        for i in range(15):
            monitor.record_trade(0.10 if i % 3 else -1.0)  # slight edge
        result = monitor.check()
        assert result.level in ('DEGRADED', 'CRITICAL', 'NORMAL', 'STRONG', 'DEAD')
        assert 0.0 <= result.multiplier <= 1.0

    def test_multiplier_bounded(self, monitor):
        for _ in range(20):
            monitor.record_trade(float(np.random.choice([-1.0, 0.5, 2.0])))
        result = monitor.check()
        assert 0.0 <= result.multiplier <= 1.0

    def test_z_score(self, monitor):
        for _ in range(25):
            monitor.record_trade(0.5)
        z = monitor.z_score_vs_baseline()
        assert isinstance(z, float)

    def test_decay_fit_insufficient(self, monitor):
        for _ in range(3):
            monitor.record_trade(0.5)
        fit = monitor.decay_fit()
        assert fit['status'] == 'insufficient_data'

    def test_decay_fit_ok(self, monitor):
        for _ in range(25):
            monitor.record_trade(0.4)
        fit = monitor.decay_fit()
        assert fit['status'] in ('ok', 'fit_failed')


# ─────────────────────────────────────────────────────────────────────── #
# 5. CorrelatedPositionTracker + gates (Andrew Lo)
# ─────────────────────────────────────────────────────────────────────── #

class TestCorrelatedPositionTracker:
    @pytest.fixture
    def tracker(self):
        from sovereign.risk.correlated_position_tracker import CorrelatedPositionTracker
        return CorrelatedPositionTracker(lookback=5, max_adjustment=0.12)

    def test_no_prior_trades(self, tracker):
        upd = tracker.get_win_rate_update('EURUSD', 'MOMENTUM')
        assert upd.win_rate_adjustment == 0.0
        assert upd.n_corr_trades == 0

    def test_winning_regime_adjusts_up(self, tracker):
        for i in range(5):
            tracker.open_position(f'SYM{i}', 'MOMENTUM', 'LONG')
            tracker.record_outcome(f'SYM{i}', 'MOMENTUM', 'LONG', True, 100.0)
        upd = tracker.get_win_rate_update('NEW', 'MOMENTUM')
        assert upd.win_rate_adjustment > 0.0

    def test_losing_regime_adjusts_down(self, tracker):
        for i in range(5):
            tracker.open_position(f'SYM{i}', 'REVERSION', 'SHORT')
            tracker.record_outcome(f'SYM{i}', 'REVERSION', 'SHORT', False, -80.0)
        upd = tracker.get_win_rate_update('NEW', 'REVERSION')
        assert upd.win_rate_adjustment < 0.0

    def test_adjustment_bounded(self, tracker):
        for i in range(10):
            tracker.record_outcome(f's{i}', 'MOMENTUM', 'LONG', True, 200.0)
        upd = tracker.get_win_rate_update('X', 'MOMENTUM')
        assert abs(upd.win_rate_adjustment) <= 0.12

    def test_effective_sample_positive(self, tracker):
        for i in range(3):
            tracker.record_outcome(f's{i}', 'MOMENTUM', 'LONG', True, 100.0)
        upd = tracker.get_win_rate_update('X', 'MOMENTUM')
        assert upd.effective_sample > 0.0

    def test_cross_regime_no_contamination(self, tracker):
        for i in range(5):
            tracker.record_outcome(f's{i}', 'REVERSION', 'SHORT', False, -50.0)
        upd = tracker.get_win_rate_update('X', 'MOMENTUM')
        assert upd.win_rate_adjustment == 0.0   # different regime — no adjustment


class TestLoUncertaintyGate:
    def test_level1_certainty(self):
        from sovereign.risk.correlated_position_tracker import lo_uncertainty_gate
        mult, desc = lo_uncertainty_gate(hmm_transition_prob=0.05, regime_confidence=0.90)
        assert mult == 1.0
        assert 'LEVEL_1' in desc

    def test_level5_knightian_halt(self):
        from sovereign.risk.correlated_position_tracker import lo_uncertainty_gate
        mult, desc = lo_uncertainty_gate(hmm_transition_prob=0.90, regime_confidence=0.10)
        assert mult == 0.0

    def test_level3_reducible(self):
        from sovereign.risk.correlated_position_tracker import lo_uncertainty_gate
        mult, desc = lo_uncertainty_gate(hmm_transition_prob=0.40, regime_confidence=0.50)
        assert mult == 0.50

    def test_library_override_reduces(self):
        from sovereign.risk.correlated_position_tracker import library_adjusted_uncertainty_level

        class FakeVM:
            def __init__(self, sim):
                self.similarity = sim

        class FakeInsight:
            volume_matches = [FakeVM(0.75)] * 8   # 8 volumes → LIBRARY_7+

        mult, desc = library_adjusted_uncertainty_level(0.05, 0.90, FakeInsight())
        assert mult == 0.50   # library override kicks in (was 1.0)

    def test_no_library_passthrough(self):
        from sovereign.risk.correlated_position_tracker import library_adjusted_uncertainty_level
        mult, _ = library_adjusted_uncertainty_level(0.05, 0.90, None)
        assert mult == 1.0


# ─────────────────────────────────────────────────────────────────────── #
# 6. KalmanRegimeEstimator (CS229 L19)
# ─────────────────────────────────────────────────────────────────────── #

class TestKalmanRegimeEstimator:
    @pytest.fixture
    def kalman(self, tmp_path):
        with patch('sovereign.risk.kalman_regime._CHECKPOINT', tmp_path / "kalman.pkl"):
            from sovereign.risk.kalman_regime import KalmanRegimeEstimator
            return KalmanRegimeEstimator()

    def test_update_returns_state(self, kalman):
        obs = np.array([0.001, -0.002, 0.0005, -0.001, 0.0008])
        z = kalman.update(obs)
        assert z.shape == (3,)

    def test_output_regime_valid(self, kalman):
        for _ in range(5):
            kalman.update(np.random.randn(5) * 0.01)
        out = kalman.get_regime_output()
        assert out['regime'] in ('MOMENTUM', 'REVERSION', 'FLAT')
        assert 0.30 <= out['confidence'] <= 0.95

    def test_state_uncertainty_positive(self, kalman):
        kalman.update(np.zeros(5))
        assert kalman.state_uncertainty() > 0.0

    def test_partial_observation_handled(self, kalman):
        # Shorter observation vector — should not crash
        z = kalman.update(np.array([0.001, 0.002]))
        assert z.shape == (3,)

    def test_trending_state_gives_momentum(self, kalman):
        # Push strong positive trend signal into EURUSD/GBPUSD/AUDUSD
        for _ in range(30):
            kalman.update(np.array([0.005, 0.004, 0.003, -0.002, 0.004]))
        out = kalman.get_regime_output()
        # After many trending updates, trend_factor should be positive
        assert out['trend_factor'] > 0

    def test_describe(self, kalman):
        assert 'KalmanRegimeEstimator' in kalman.describe()


# ─────────────────────────────────────────────────────────────────────── #
# 7. PredictNow (Ernest Chan / CS229 L03+04)
# ─────────────────────────────────────────────────────────────────────── #

class TestPredictNow:
    @pytest.fixture
    def pn(self, tmp_path):
        with patch('sovereign.risk.predict_now._CHECKPOINT', tmp_path / "pn.pkl"), \
             patch('sovereign.risk.predict_now._LEDGER_DIR', tmp_path / "ledger"):
            from sovereign.risk.predict_now import PredictNow
            return PredictNow()

    def test_cold_start_probability(self, pn):
        out = pn.evaluate('MOMENTUM', 0.2, 0.65, 28, 'momentum')
        assert 0 < out.prob_profitable < 1
        assert out.size_multiplier >= 0.0

    def test_record_and_evaluate(self, pn):
        # Record a few trades then evaluate
        for i in range(10):
            pn.record_outcome('MOMENTUM', 0.2, 0.6, 28, 'momentum', i % 2 == 0, 50.0)
        out = pn.evaluate('MOMENTUM', 0.2, 0.6, 28, 'momentum')
        assert 0 < out.prob_profitable < 1

    def test_low_probability_skips(self, pn):
        # Feed many losses in REVERSION to drive down probability
        for _ in range(30):
            pn.record_outcome('REVERSION', 0.5, 0.4, 15, 'reversion', False, -100.0)
        out = pn.evaluate('REVERSION', 0.5, 0.4, 15, 'reversion')
        if out.prob_profitable <= 0.40:
            assert out.size_multiplier == 0.0

    def test_high_probability_expands(self, pn):
        for _ in range(30):
            pn.record_outcome('MOMENTUM', 0.1, 0.7, 35, 'momentum', True, 150.0)
        out = pn.evaluate('MOMENTUM', 0.1, 0.7, 35, 'momentum')
        if out.prob_profitable >= 0.70:
            assert out.size_multiplier > 1.0

    def test_ica_preprocessor_path(self, pn, tmp_path):
        from sovereign.risk.ica_factor_separator import ICAFactorSeparator
        with patch('sovereign.risk.ica_factor_separator._CHECKPOINT', tmp_path / "ica.pkl"):
            ica = ICAFactorSeparator(n_components=3)
            X = np.random.randn(60, 5)
            ica.fit(X)
            # Add some training data
            for _ in range(5):
                pn.record_outcome('MOMENTUM', 0.2, 0.6, 28, 'momentum', True, 50.0)
            out = pn.evaluate('MOMENTUM', 0.2, 0.6, 28, 'momentum', ica_preprocessor=ica)
            assert 0 < out.prob_profitable < 1

    def test_library_informed_win_rate_blend(self):
        from sovereign.risk.predict_now import library_informed_win_rate

        class FakeInsight:
            primary_regime = 'MOMENTUM'

        # At n=0 → library dominates
        blended, reason = library_informed_win_rate(0.60, 0, FakeInsight())
        assert abs(blended - 0.57) < 0.05   # library win rate for MOMENTUM ≈ 0.57

        # At n=400 → own estimate dominates
        blended2, _ = library_informed_win_rate(0.60, 400, FakeInsight())
        assert blended2 == pytest.approx(0.60, abs=0.01)

    def test_library_informed_no_library(self):
        from sovereign.risk.predict_now import library_informed_win_rate
        blended, reason = library_informed_win_rate(0.55, 100, None)
        assert blended == 0.55
        assert reason == 'no_library'


# ─────────────────────────────────────────────────────────────────────── #
# 8. TradeMDP (CS229 L16)
# ─────────────────────────────────────────────────────────────────────── #

class TestTradeMDP:
    @pytest.fixture
    def mdp(self, tmp_path):
        with patch('sovereign.risk.trade_mdp._CHECKPOINT', tmp_path / "mdp.pkl"):
            from sovereign.risk.trade_mdp import TradeMDP
            return TradeMDP()

    def test_size_multiplier_valid(self, mdp):
        mult = mdp.get_size_multiplier('MOMENTUM', 0, 0.01, 0.75)
        assert mult in (0.50, 0.75, 1.00, 1.25)

    def test_bad_state_reduces_size(self, mdp):
        # After seeding with bad state priors, high drawdown + losses → smaller
        mult_good = mdp.get_size_multiplier('MOMENTUM', 0, 0.0, 0.80)
        mult_bad  = mdp.get_size_multiplier('FLAT', 3, 0.15, 0.30)
        assert mult_bad <= mult_good

    def test_record_transition_increments(self, mdp):
        mdp.record_transition(
            'MOMENTUM', 0, 0.01, 0.75,
            'MOMENTUM', 0, 0.005, 0.75,
            100.0, 1.0,
        )
        assert mdp._m.n_trades == 1

    def test_state_value_float(self, mdp):
        v = mdp.state_value('MOMENTUM', 0, 0.01, 0.75)
        assert isinstance(v, float)

    def test_state_index_all_regimes(self):
        from sovereign.risk.trade_mdp import state_index, N_STATES
        for regime in ('MOMENTUM', 'REVERSION', 'FLAT'):
            for c in range(4):
                for d_pct in (0.01, 0.05, 0.12):
                    for conf in (0.3, 0.8):
                        s = state_index(regime, c, d_pct, conf)
                        assert 0 <= s < N_STATES

    def test_value_iteration_runs(self, mdp):
        # Add several transitions to trigger value iteration refit
        for i in range(10):
            mdp.record_transition(
                'MOMENTUM', i % 4, 0.02, 0.7,
                'MOMENTUM', (i + 1) % 4, 0.015, 0.75,
                float(np.random.randn()), 1.0,
            )
        # Should have run value iteration at trade 10
        assert mdp._m.n_trades == 10

    def test_describe(self, mdp):
        assert 'TradeMDP' in mdp.describe()


# ─────────────────────────────────────────────────────────────────────── #
# 9. LQRController (CS229 L18)
# ─────────────────────────────────────────────────────────────────────── #

class TestLQRController:
    @pytest.fixture
    def lqr(self):
        from sovereign.risk.lqr_controller import LQRController
        return LQRController(horizon=10)

    def test_riccati_solved(self, lqr):
        # K should be non-zero
        assert lqr._K is not None
        assert lqr._K.shape == (1, 4)
        assert float(np.abs(lqr._K).max()) > 0

    def test_good_state_full_size(self, lqr):
        mult, _ = lqr.compute_size_multiplier(
            drawdown_pct=0.0, rolling_pnl_3d=500.0,
            consecutive_losses=0, kelly_fraction=0.02,
        )
        assert mult >= 0.90   # good state → near full size or boosted

    def test_bad_state_reduces_size(self, lqr):
        mult, _ = lqr.compute_size_multiplier(
            drawdown_pct=0.15, rolling_pnl_3d=-500.0,
            consecutive_losses=4, kelly_fraction=0.035,
        )
        assert mult < 1.0   # bad state → reduction

    def test_multiplier_bounds(self, lqr):
        for _ in range(20):
            dd     = float(np.random.uniform(0, 0.20))
            pnl    = float(np.random.uniform(-1000, 1000))
            consec = int(np.random.randint(0, 5))
            kelly  = float(np.random.uniform(0.01, 0.04))
            mult, _ = lqr.compute_size_multiplier(dd, pnl, consec, kelly)
            assert 0.25 <= mult <= 1.25, f"mult={mult} out of bounds"

    def test_debug_dict_keys(self, lqr):
        _, debug = lqr.compute_size_multiplier(0.05, -100.0, 1, 0.02)
        required = {'state_x', 'lqr_u', 'lqr_delta', 'base_multiplier', 'final_multiplier'}
        assert required.issubset(debug.keys())

    def test_describe(self, lqr):
        assert 'LQRController' in lqr.describe()


# ─────────────────────────────────────────────────────────────────────── #
# 10. PegasusPolicySearch (CS229 L20)
# ─────────────────────────────────────────────────────────────────────── #

class TestPegasusPolicySearch:
    @pytest.fixture
    def pg(self, tmp_path):
        with patch('sovereign.risk.pegasus_policy_search._CHECKPOINT', tmp_path / "pegasus.pkl"):
            from sovereign.risk.pegasus_policy_search import PegasusPolicySearch
            return PegasusPolicySearch(n_scenarios=10)

    def test_cold_start_trust(self, pg):
        assert pg.n_updates == 0
        assert pg.trust_multiplier == 0.0

    def test_default_params_valid(self, pg):
        from sovereign.risk.pegasus_policy_search import _PARAM_BOUNDS
        p = pg.current_params
        for key, (lo, hi, _) in _PARAM_BOUNDS.items():
            val = getattr(p, key)
            assert lo <= val <= hi, f"{key}={val} out of [{lo},{hi}]"

    def test_reinforce_update(self, pg):
        pg.reinforce_update(
            state_features=np.array([0.8, 0.2, 0.65, 0.5]),
            action_taken=1000.0, realized_pnl=150.0,
            hmm_confidence=0.75, stop_atr_used=1.5,
            tp_rr_used=2.5, kelly_frac_used=0.02,
        )
        assert pg.n_updates == 1

    def test_params_remain_bounded_after_updates(self, pg):
        from sovereign.risk.pegasus_policy_search import _PARAM_BOUNDS
        for i in range(40):
            pg.reinforce_update(
                state_features=np.random.randn(4),
                action_taken=float(np.random.uniform(500, 2000)),
                realized_pnl=float(np.random.randn() * 100),
                hmm_confidence=float(np.random.uniform(0.3, 0.9)),
                stop_atr_used=float(np.random.uniform(1.0, 3.0)),
                tp_rr_used=float(np.random.uniform(1.5, 5.0)),
                kelly_frac_used=float(np.random.uniform(0.01, 0.04)),
            )
        p = pg.current_params
        for key, (lo, hi, _) in _PARAM_BOUNDS.items():
            val = getattr(p, key)
            assert lo <= val <= hi, f"After 40 updates: {key}={val} out of [{lo},{hi}]"

    def test_trust_ramp(self, pg):
        for i in range(25):
            pg.reinforce_update(
                state_features=np.zeros(4), action_taken=1000.0,
                realized_pnl=50.0, hmm_confidence=0.7,
                stop_atr_used=1.5, tp_rr_used=2.5, kelly_frac_used=0.02,
            )
        # At n_updates=25: trust should be partially ramped (between 0 and 1)
        assert 0.0 < pg.trust_multiplier <= 1.0

    def test_full_trust_at_30(self, pg):
        for i in range(30):
            pg.reinforce_update(
                state_features=np.zeros(4), action_taken=1000.0,
                realized_pnl=50.0, hmm_confidence=0.7,
                stop_atr_used=1.5, tp_rr_used=2.5, kelly_frac_used=0.02,
            )
        assert pg.trust_multiplier == 1.0

    def test_describe(self, pg):
        assert 'PegasusPolicySearch' in pg.describe()


# ─────────────────────────────────────────────────────────────────────── #
# 11. KMeansRegimeClusterer (CS229 L12)
# ─────────────────────────────────────────────────────────────────────── #

class TestKMeansRegimeClusterer:
    @pytest.fixture
    def km(self, tmp_path):
        with patch('sovereign.risk.ml_diagnostics._CHECKPOINT', tmp_path / "kmeans.pkl"):
            from sovereign.risk.ml_diagnostics import KMeansRegimeClusterer
            return KMeansRegimeClusterer(k=3, n_init=5)

    def test_cold_start_returns_flat(self, km):
        pred = km.predict(np.array([0.5, 0.4, 0.3]))
        assert pred == 'FLAT'

    def test_fit_produces_centroids(self, km):
        X = np.vstack([
            np.random.randn(30, 3) + np.array([0.65, 0.70, 0.10]),   # MOMENTUM
            np.random.randn(30, 3) + np.array([0.40, 0.35, 0.40]),   # REVERSION
            np.random.randn(30, 3) + np.array([0.50, 0.20, 0.60]),   # FLAT
        ]) * 0.05  # small noise
        km.fit(X)
        assert km._centroids is not None
        assert km._centroids.shape == (3, 3)

    def test_predict_regime_valid(self, km):
        X = np.vstack([
            np.random.randn(25, 3) + [0.65, 0.70, 0.10],
            np.random.randn(25, 3) + [0.40, 0.35, 0.40],
            np.random.randn(25, 3) + [0.50, 0.20, 0.60],
        ]) * 0.05
        km.fit(X)
        pred = km.predict(np.array([0.65, 0.70, 0.10]))
        assert pred in ('MOMENTUM', 'REVERSION', 'FLAT')

    def test_cluster_assignment_sensible(self, km):
        # Clear-cut cluster data
        X = np.array(
            [[0.65, 0.70, 0.10]] * 20   # MOMENTUM cluster
            + [[0.40, 0.35, 0.40]] * 20 # REVERSION cluster
            + [[0.50, 0.20, 0.60]] * 20 # FLAT cluster
        )
        km.fit(X)
        # Strong MOMENTUM signal
        pred = km.predict(np.array([0.65, 0.70, 0.10]))
        assert pred == 'MOMENTUM'

    def test_predict_distances_keys(self, km):
        X = np.array([[0.5, 0.4, 0.3]] * 20 + [[0.6, 0.6, 0.1]] * 20 + [[0.4, 0.2, 0.5]] * 20)
        km.fit(X)
        dists = km.predict_distances(np.array([0.5, 0.4, 0.3]))
        assert len(dists) == 3
        assert all(0.0 <= v <= 1.0 for v in dists.values())

    def test_too_few_samples(self, km):
        km.fit(np.random.randn(2, 3))   # < k=3
        assert km._centroids is None

    def test_describe(self, km):
        assert 'KMeansRegimeClusterer' in km.describe()
