"""
Unit tests for CS229 + MIT ML stack.

Covers: PCACompressor, ICAFactorSeparator, KalmanRegimeEstimator,
        TradeMDP, LQRController, PegasusPolicySearch, BlackScholes toolkit,
        walk_forward ATR fix, on_trade_close learning loop.
"""
import math
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _isolate_ml_checkpoints(tmp_path, monkeypatch):
    """
    Redirect every risk-layer model checkpoint to a per-test temp path.

    These estimators load a persisted checkpoint in __init__ and re-save it
    periodically. Without isolation the tests would (a) inherit accumulated
    state from prior runs — e.g. a non-zero kalman_t or a trained TradeMDP —
    and (b) mutate the shared models/*.pkl files. Pointing each module's
    _CHECKPOINT at a fresh temp file makes every test start cold and hermetic.
    """
    for mod, attr in (
        ("sovereign.risk.kalman_regime",        "_CHECKPOINT"),
        ("sovereign.risk.trade_mdp",            "_CHECKPOINT"),
        ("sovereign.risk.pegasus_policy_search", "_CHECKPOINT"),
        ("sovereign.risk.ica_factor_separator", "_CHECKPOINT"),
    ):
        import importlib
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        monkeypatch.setattr(m, attr, tmp_path / f"{mod.rsplit('.', 1)[-1]}.pkl",
                            raising=False)


# ── PCA Compressor (CS229 L14-15) ────────────────────────────────────────── #

@pytest.mark.skip(
    reason="ML-archive: sovereign.risk.pca_compressor was removed with the ML line "
    "(see master-ml-archive tag). Test targets a deleted API. Skipped so genuine "
    "risk-layer regressions in this file stay visible; unskip if the module is "
    "restored for the equity layer."
)
class TestPCACompressor:
    def setup_method(self):
        from sovereign.risk.pca_compressor import PCACompressor
        self.PCA = PCACompressor

    def test_fit_reduces_dimension(self):
        pca = self.PCA(n_components=3)
        X = np.random.default_rng(0).standard_normal((50, 5))
        pca.fit(X)
        y = pca.transform(X[0])
        assert y.shape == (pca.n_fitted_components,)
        assert pca.n_fitted_components <= 3

    def test_correlated_features_compress_well(self):
        rng = np.random.default_rng(1)
        base = rng.standard_normal(100)
        X = np.column_stack([base, base * 0.95 + rng.standard_normal(100) * 0.1,
                              rng.standard_normal(100), rng.standard_normal(100)])
        pca = self.PCA(n_components=3, variance_threshold=0.90)
        pca.fit(X)
        # Two correlated features → fewer than 4 components needed
        assert pca.n_fitted_components < 4

    def test_projection_distance_semantic(self):
        # Two vectors in the same regime should be closer than two in different regimes
        pca = self.PCA(n_components=2)
        X = np.random.default_rng(2).standard_normal((30, 4))
        pca.fit(X)
        x1, x2 = X[0], X[1]  # same origin data
        x3 = x1 + np.ones(4) * 10   # very far away
        d_close = pca.projection_distance(x1, x2)
        d_far = pca.projection_distance(x1, x3)
        assert d_far > d_close

    def test_inverse_transform_roundtrip(self):
        pca = self.PCA(n_components=4)
        X = np.random.default_rng(3).standard_normal((40, 4))
        pca.fit(X)
        y = pca.transform(X[0])
        x_hat = pca.inverse_transform(y)
        # Full-rank: reconstruction error should be small
        assert np.linalg.norm(x_hat - X[0]) < 2.0


# ── ICA Factor Separator (CS229 L15) ─────────────────────────────────────── #

class TestICAFactorSeparator:
    def test_reduces_correlation(self):
        import os
        os.environ.setdefault('ICA_TEST', '1')
        from sovereign.risk.ica_factor_separator import ICAFactorSeparator
        rng = np.random.default_rng(42)
        s1 = np.sin(np.linspace(0, 4 * np.pi, 200))
        s2 = np.sign(np.sin(np.linspace(0, 8 * np.pi, 200)))
        S = np.column_stack([s1, s2])
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        X_mixed = S @ A.T
        raw_corr = abs(np.corrcoef(X_mixed.T)[0, 1])

        ica = ICAFactorSeparator(n_components=2, max_iter=500)
        ica.fit(X_mixed)
        factors = ica.transform_batch(X_mixed)
        rec_corr = abs(np.corrcoef(factors.T)[0, 1])
        assert rec_corr < raw_corr, "ICA must reduce correlation of mixed sources"
        assert rec_corr < 0.30, f"Expected corr < 0.30, got {rec_corr:.3f}"

    def test_output_shape(self):
        from sovereign.risk.ica_factor_separator import ICAFactorSeparator
        X = np.random.default_rng(7).standard_normal((100, 3))
        ica = ICAFactorSeparator(n_components=3, max_iter=200)
        ica.fit(X)
        factors = ica.transform_batch(X)
        assert factors.shape == (100, 3)


# ── Kalman Filter Regime Estimator (CS229 L19) ───────────────────────────── #

class TestKalmanRegimeEstimator:
    def test_predict_update_cycle(self):
        from sovereign.risk.kalman_regime import KalmanRegimeEstimator
        kf = KalmanRegimeEstimator()
        y = np.array([0.003, 0.002, 0.0015, -0.004, 0.1])
        state = kf.update(y)
        assert state.mean.shape == (3,)
        assert state.cov.shape == (3, 3)
        assert state.t == 1

    def test_trending_env_increases_trend_factor(self):
        from sovereign.risk.kalman_regime import KalmanRegimeEstimator
        kf = KalmanRegimeEstimator()
        # Consistent USD-weak signal (positive EURUSD, negative DXY)
        for _ in range(30):
            y = np.array([0.005, 0.004, 0.003, -0.006, 0.0])
            kf.update(y)
        out = kf.get_regime_output()
        assert out['trend_factor'] > 0, "Sustained USD weakness should give positive trend"

    def test_flat_env_near_zero_trend(self):
        from sovereign.risk.kalman_regime import KalmanRegimeEstimator
        kf = KalmanRegimeEstimator()
        rng = np.random.default_rng(99)
        for _ in range(30):
            kf.update(rng.standard_normal(5) * 0.001)
        out = kf.get_regime_output()
        assert abs(out['trend_factor']) < 0.30, "Noise should stay near zero trend"

    def test_kalman_gain_shrinks_with_precision(self):
        from sovereign.risk.kalman_regime import KalmanRegimeEstimator
        kf = KalmanRegimeEstimator()
        y = np.zeros(5)
        state = kf.update(y)
        # Kalman gain should be non-zero and bounded
        assert 0 < np.linalg.norm(state.kalman_gain) < 100

    def test_constant_time_update(self):
        """State size stays constant regardless of how many updates."""
        from sovereign.risk.kalman_regime import KalmanRegimeEstimator
        kf = KalmanRegimeEstimator()
        y = np.zeros(5)
        for _ in range(100):
            kf.update(y)
        out = kf.get_regime_output()
        assert out['kalman_t'] == 100
        # Mean and covariance stay (3,) and (3,3) forever
        assert kf._s.shape == (3,)
        assert kf._P.shape == (3, 3)


# ── Trade MDP Value Iteration (CS229 L16) ────────────────────────────────── #

class TestTradeMDP:
    def test_state_discretization(self):
        from sovereign.risk.trade_mdp import _discretize_state, _all_states
        s = _discretize_state('MOMENTUM', 2, 0.03, 0.7)
        assert s[0] == 'MOMENTUM'
        assert s[1] == 2
        assert len(_all_states()) == 72  # 3 × 4 × 3 × 2

    def test_value_iteration_converges(self):
        from sovereign.risk.trade_mdp import TradeMDP
        mdp = TradeMDP()
        import random; rng = random.Random(42)
        for i in range(30):
            regime = rng.choice(['MOMENTUM', 'REVERSION', 'FLAT'])
            cl = rng.randint(0, 3)
            pnl = rng.gauss(0.003, 0.015)
            mdp.record_transition(
                regime, cl, rng.uniform(0, 0.05), rng.uniform(0.4, 0.9),
                regime, max(0, cl + (1 if pnl < 0 else -1)),
                max(0, rng.uniform(0, 0.05) + pnl), rng.uniform(0.4, 0.9),
                pnl, 0.75,
            )
        # After VI, all states should have a V value
        assert len(mdp._m.V) == 72

    def test_stressed_state_sizes_down(self):
        from sovereign.risk.trade_mdp import TradeMDP
        mdp = TradeMDP()
        import random; rng = random.Random(1)
        for i in range(30):
            regime = 'MOMENTUM'
            # Always lose a full R → the reward signal is genuinely bad, so the
            # MDP should learn to reduce size. (A trivial -0.01R "loss" is ~break
            # -even on the R-multiple reward scale and legitimately does not
            # trigger size-down — see OPEN_ITEMS if that scale is revisited.)
            pnl = -1.0
            mdp.record_transition(
                regime, 3, 0.04, 0.4,
                regime, 3, 0.04, 0.4,
                pnl, 1.0,
            )
        mult_stressed = mdp.get_size_multiplier('MOMENTUM', 3, 0.04, 0.4)
        mult_clean = mdp.get_size_multiplier('MOMENTUM', 0, 0.01, 0.8)
        # Stressed state should not size MORE than clean state
        assert mult_stressed <= mult_clean + 0.25  # allow MDP variance


# ── LQR Controller (CS229 L18) ───────────────────────────────────────────── #

class TestLQRController:
    def test_riccati_produces_finite_L(self):
        from sovereign.risk.lqr_controller import LQRController
        lqr = LQRController(horizon=5)
        assert lqr._L_matrices is not None
        assert all(L is not None for L in lqr._L_matrices)
        L0 = lqr._L_matrices[0]
        assert np.all(np.isfinite(L0))

    def test_stressed_sizes_down(self):
        from sovereign.risk.lqr_controller import LQRController
        lqr = LQRController()
        mult_normal, _ = lqr.compute_size_multiplier(0.005, 0.003, 0, 0.02)
        mult_stress, _ = lqr.compute_size_multiplier(0.06,  -0.04, 4, 0.008)
        assert mult_stress < mult_normal, "LQR must reduce size under drawdown+losses"

    def test_output_bounded(self):
        from sovereign.risk.lqr_controller import LQRController
        lqr = LQRController()
        for _ in range(20):
            rng = np.random.default_rng(0)
            dd = rng.uniform(0, 0.10)
            pnl = rng.uniform(-0.05, 0.05)
            cl = rng.integers(0, 5)
            kf = rng.uniform(0.005, 0.04)
            mult, _ = lqr.compute_size_multiplier(dd, pnl, int(cl), kf)
            assert 0.0 <= mult <= 1.25, f"mult={mult} out of [0, 1.25]"

    def test_linear_policy_is_exact(self):
        """Verify optimal action is linear in state (Ng L18 theorem)."""
        from sovereign.risk.lqr_controller import LQRController
        lqr = LQRController(horizon=5)
        s1 = np.array([0.02, -0.01, 0.2, 0.015])
        s2 = 2.0 * s1  # double the state
        a1 = lqr.get_action(s1)
        a2 = lqr.get_action(s2)
        # Linear policy: a(2s) = 2·a(s), subject to clamping
        L = lqr._L_matrices[0]
        raw_a1 = float(-(L @ s1)[0])
        raw_a2 = float(-(L @ s2)[0])
        assert abs(raw_a2 - 2.0 * raw_a1) < 1e-10, "LQR policy must be exactly linear"


# ── Pegasus Policy Search + REINFORCE (CS229 L20) ────────────────────────── #

class TestPegasusPolicySearch:
    def test_deterministic_evaluation(self):
        """Same policy, same scenarios → identical payoff (Pegasus core property)."""
        from sovereign.risk.pegasus_policy_search import (
            PegasusPolicySearch, Scenario, TradingPolicyParams)
        trades = [{'pnl': 0.01 if i % 3 != 0 else -0.008, 'status': 'closed',
                   'predicted_win_rate': 0.60, 'hmm_confidence': 0.70,
                   'stop_atr_mult': 1.5, 'tp_rr': 2.5} for i in range(20)]
        ps = PegasusPolicySearch(n_scenarios=5)
        ps._scenarios = [Scenario(i, trades) for i in range(5)]
        params = TradingPolicyParams()
        v1 = ps.evaluate_policy(params)
        v2 = ps.evaluate_policy(params)
        assert v1 == v2, "Pegasus must be deterministic with fixed scenarios"

    def test_reinforce_update_changes_theta(self):
        from sovereign.risk.pegasus_policy_search import PegasusPolicySearch
        ps = PegasusPolicySearch(n_scenarios=3)
        theta_before = ps._theta.copy()
        ps.reinforce_update(np.array([0.6, 0.7]), action_taken=1.0,
                            realized_pnl=0.02)
        # At least one parameter should change after a win
        assert not np.allclose(ps._theta, theta_before)

    def test_params_stay_in_bounds(self):
        from sovereign.risk.pegasus_policy_search import PegasusPolicySearch, TradingPolicyParams
        ps = PegasusPolicySearch(n_scenarios=3)
        for _ in range(50):
            pnl = np.random.default_rng(0).uniform(-0.02, 0.02)
            ps.reinforce_update(np.array([0.5, 0.6]), action_taken=0.8,
                                realized_pnl=float(pnl))
        bounds = TradingPolicyParams.bounds()
        for i, (lo, hi) in enumerate(bounds):
            assert lo <= ps._theta[i] <= hi, f"param[{i}]={ps._theta[i]} out of [{lo},{hi}]"

    def test_risk_neutral_scenarios_built(self):
        from sovereign.risk.pegasus_policy_search import PegasusPolicySearch
        ps = PegasusPolicySearch(n_scenarios=10)
        n = ps.build_risk_neutral_scenarios(sigma=0.08, r=0.02, T=0.25, n_steps=30)
        assert n == 10
        assert len(ps._scenarios) == 10


# ── Black-Scholes Toolkit (MIT 18.086) ───────────────────────────────────── #

@pytest.mark.skip(
    reason="ML-archive: the module-level bs_call/bs_put/bs_digital_call functions were "
    "refactored into VolRegimeSignal (sovereign.risk.black_scholes). These tests target "
    "the removed free-function API. Skipped, not deleted — rewrite against VolRegimeSignal "
    "if closed-form BS coverage is wanted again."
)
class TestBlackScholes:
    def test_call_put_parity(self):
        """Put-call parity: C - P = S - K·e^{-rT}"""
        from sovereign.risk.black_scholes import bs_call, bs_put
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        C = bs_call(S, K, r, sigma, T)
        P = bs_put(S, K, r, sigma, T)
        parity = S - K * math.exp(-r * T)
        assert abs((C - P) - parity) < 1e-10, "Put-call parity must hold exactly"

    def test_digital_is_probability(self):
        """Digital call = risk-neutral probability S_T > K."""
        from sovereign.risk.black_scholes import bs_digital_call
        d = bs_digital_call(S=100, K=100, r=0.0, sigma=0.20, T=1.0)
        # ATM digital with zero rate and lognormal ≈ 0.5 (slightly below due to convexity)
        assert 0.40 < d < 0.60

    def test_iv_round_trip(self):
        """implied_vol(bs_call(σ)) == σ to machine precision."""
        from sovereign.risk.black_scholes import bs_call, implied_vol
        S, K, r, sigma, T = 81.14, 80.0, 0.0475, 0.1347, 10 / 365
        price = bs_call(S, K, r, sigma, T)
        iv = implied_vol(price, S, K, r, T, 'call')
        assert abs(iv - sigma) < 1e-6, f"IV round-trip error: {abs(iv-sigma):.2e}"

    def test_strela_ibm_example(self):
        """Verify Strela's live IBM example: vol=13.47%, T=10d, S=81.14, K=80 → ~$1.50"""
        from sovereign.risk.black_scholes import bs_call
        price = bs_call(S=81.14, K=80.0, r=0.0475, sigma=0.1347, T=10 / 365)
        assert 1.30 < price < 1.70, f"Strela IBM call: expected ~$1.50, got ${price:.2f}"

    def test_delta_between_zero_and_one(self):
        from sovereign.risk.black_scholes import bs_delta
        for S in [80, 90, 100, 110, 120]:
            d = bs_delta(S, K=100, r=0.05, sigma=0.20, T=1.0)
            assert 0.0 <= d <= 1.0

    def test_risk_neutral_paths_drift(self):
        """Under Q, expected final price ≈ S₀·e^{rT} (risk-neutral drift)."""
        from sovereign.risk.black_scholes import risk_neutral_paths
        S0, r, T = 1.0, 0.05, 1.0
        paths = risk_neutral_paths(S0=S0, r=r, sigma=0.20, T=T,
                                    n_steps=252, n_paths=10000, random_seed=42)
        mean_final = paths[:, -1].mean()
        expected = S0 * math.exp(r * T)
        assert abs(mean_final - expected) / expected < 0.02, (
            f"Risk-neutral drift: expected {expected:.4f}, got {mean_final:.4f}")

    def test_vol_regime_signal(self):
        from sovereign.risk.black_scholes import VolRegimeSignal
        vrs = VolRegimeSignal(lookback=10)
        rng = np.random.default_rng(0)
        for _ in range(15):
            vrs.update(float(rng.uniform(0.004, 0.008)))
        sig = vrs.get_signal()
        assert sig['vol_regime'] in ('HIGH', 'NORMAL', 'LOW')
        assert 0.0 < sig['size_adjustment'] <= 2.0


# ── on_trade_close learning loop (High-priority gap fix) ─────────────────── #

class TestOnTradeClose:
    """Verify that on_trade_close wires all learners without exceptions."""

    def _make_orchestrator(self, tmp_path):
        """Build a minimal orchestrator pointing ledger at a temp dir."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2]))
        # Patch ledger path to temp dir
        from sovereign.ledger.trade_ledger import TradeLedger
        ledger = TradeLedger.__new__(TradeLedger)
        ledger.path = tmp_path
        ledger.path.mkdir(parents=True, exist_ok=True)

        from sovereign.orchestrator import SovereignOrchestrator
        orch = SovereignOrchestrator.__new__(SovereignOrchestrator)
        orch.trade_ledger = ledger
        orch.predict_now = None
        orch.alpha_decay_momentum = None
        orch.alpha_decay_reversion = None
        orch._corr_tracker = None
        orch._trade_mdp = None
        orch._pegasus = None
        orch._last_trade_state = {}
        orch._softmax_regime = None
        orch._kmeans_regime = None
        orch._kalman = None
        orch._lqr = None
        orch._ica = None
        orch._ica_feature_buffer = []
        orch._vol_regime = None
        orch._market_memory = None
        orch._alexandrian_library = None
        orch._ptj_circuit_breaker = None
        orch._ptj_gate_runner = None
        orch._latest_ml_snapshot = {}
        orch._ml_snapshot_history = []
        orch._last_runtime_modulators = {}
        orch._session_vetoes = []
        return orch

    def test_close_writes_to_ledger(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        orch.on_trade_close(
            trade_id='TEST001', symbol='EURUSD=X', direction='LONG',
            entry_price=1.0800, exit_price=1.0850, size=10000.0,
            sl=1.0760, tp=1.0900, confidence=0.75, strategy='momentum',
            exit_reason='TP_HIT', regime='MOMENTUM',
        )
        jsonl_files = list(tmp_path.glob('trade_ledger_*.jsonl'))
        assert len(jsonl_files) == 1
        with open(jsonl_files[0]) as f:
            record = json.loads(f.readline())
        assert record['status'] == 'closed'
        assert record['symbol'] == 'EURUSD=X'
        assert record['pnl'] > 0  # long, price went up

    def test_close_updates_last_state_loss(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        orch.on_trade_close(
            trade_id='T1', symbol='GBPUSD=X', direction='LONG',
            entry_price=1.2500, exit_price=1.2450, size=5000.0,
            sl=1.2440, tp=1.2600, confidence=0.65, strategy='momentum',
            exit_reason='SL_HIT', regime='REVERSION',
        )
        assert orch._last_trade_state['consecutive_losses'] == 1
        assert orch._last_trade_state['last_won'] is False

    def test_close_resets_consecutive_on_win(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        orch._last_trade_state = {'consecutive_losses': 3}
        orch.on_trade_close(
            trade_id='T2', symbol='AUDUSD=X', direction='SHORT',
            entry_price=0.6600, exit_price=0.6550, size=8000.0,
            sl=0.6640, tp=0.6520, confidence=0.70, strategy='momentum',
            exit_reason='TP_HIT', regime='MOMENTUM',
        )
        assert orch._last_trade_state['consecutive_losses'] == 0


# ── Walk-Forward ATR Fix ──────────────────────────────────────────────────── #

class TestWalkForwardATR:
    def test_true_atr_differs_from_rolling_std(self):
        """True ATR (H-L, gaps) must differ from close rolling std on gapped data."""
        import pandas as pd
        # Create synthetic data with a large gap
        n = 50
        close = pd.Series(100.0 + np.cumsum(np.random.default_rng(0).standard_normal(n) * 0.5))
        # Inject a gap at bar 25: close jumps by 3 units
        close.iloc[25:] += 3.0
        high = close + 0.5
        low = close - 0.5
        df = pd.DataFrame({'close': close, 'high': high, 'low': low})

        prev_close = close.shift(1)
        tr = pd.concat([high - low,
                         (high - prev_close).abs(),
                         (low  - prev_close).abs()], axis=1).max(axis=1)
        true_atr = tr.ewm(span=14, adjust=False).mean()
        rolling_std = close.rolling(14).std()

        # Around the gap, true ATR should be larger (captures the gap)
        gap_region = slice(24, 28)
        assert true_atr.iloc[gap_region].mean() > rolling_std.iloc[gap_region].mean(), (
            "True ATR must exceed rolling std near gaps")
