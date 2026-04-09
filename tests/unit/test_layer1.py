"""Unit tests for Layer 1 modules."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from contracts.types import (
    Direction,
    Magnitude,
    BiasOutput,
    RegimeState,
    VolRegime,
    TrendRegime,
    RiskAppetite,
    MomentumRegime,
    EventRisk,
    AccountState,
    RiskOutput,
)
from layer1.feature_builder import FeatureBuilder, FeatureVector
from layer1.regime_classifier import RegimeClassifier
from layer1.bias_engine import BiasEngine, SHAPExplainer
from layer1.hard_constraints import HardConstraints, ConstraintCheck


class TestFeatureBuilder:
    """Test FeatureBuilder."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Generate sample OHLCV data."""
        np.random.seed(42)
        n = 50
        base_price = 20000

        data = {
            "open": base_price + np.random.randn(n).cumsum() * 50,
            "high": base_price + np.random.randn(n).cumsum() * 50 + 100,
            "low": base_price + np.random.randn(n).cumsum() * 50 - 100,
            "close": base_price + np.random.randn(n).cumsum() * 50,
            "volume": np.random.randint(1000000, 2000000, n),
        }

        # Ensure high >= low
        for i in range(n):
            data["high"][i] = max(
                data["high"][i], data["low"][i], data["close"][i], data["open"][i]
            )
            data["low"][i] = min(
                data["low"][i], data["high"][i], data["close"][i], data["open"][i]
            )

        return pd.DataFrame(data)

    def test_build_features(self, sample_ohlcv):
        """Test feature building."""
        builder = FeatureBuilder()
        features = builder.build_features(sample_ohlcv, vix_value=20.0)

        assert isinstance(features, FeatureVector)
        assert features.vix_level == 20.0

    def test_feature_vector_to_dict(self, sample_ohlcv):
        """Test FeatureVector conversion to dict."""
        builder = FeatureBuilder()
        features = builder.build_features(sample_ohlcv)

        feature_dict = features.to_dict()
        assert isinstance(feature_dict, dict)
        assert len(feature_dict) >= 43
        assert "rsi_14" in feature_dict

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        builder = FeatureBuilder()
        features = builder.build_features(pd.DataFrame())

        assert isinstance(features, FeatureVector)
        assert features.rsi_14 == 50.0  # Default value

    def test_rsi_calculation(self, sample_ohlcv):
        """Test RSI calculation."""
        builder = FeatureBuilder()
        features = builder.build_features(sample_ohlcv)

        assert 0 <= features.rsi_14 <= 100

    def test_atr_calculation(self, sample_ohlcv):
        """Test ATR calculation."""
        builder = FeatureBuilder()
        features = builder.build_features(sample_ohlcv)

        assert features.atr_14 > 0


class TestRegimeClassifier:
    """Test RegimeClassifier."""

    @pytest.fixture
    def sample_features(self):
        """Create sample feature vector."""
        return FeatureVector(
            atr_14=100,
            adx_14=30,
            rsi_14=60,
            price_vs_sma_20=0.02,
            price_vs_sma_50=0.05,
            volatility_regime=2,
        )

    def test_classify(self, sample_features):
        """Test regime classification."""
        classifier = RegimeClassifier()
        regime = classifier.classify(sample_features, vix_value=20.0)

        assert isinstance(regime, RegimeState)
        assert 0 <= regime.composite_score <= 1

    def test_volatility_regime(self, sample_features):
        """Test volatility regime classification."""
        classifier = RegimeClassifier()

        # Low VIX with low historical vol
        low_vol_features = FeatureVector(volatility_regime=1)
        regime = classifier.classify(low_vol_features, vix_value=12.0)
        assert regime.volatility == VolRegime.LOW

        # High VIX
        high_vol_features = FeatureVector(volatility_regime=4)
        regime = classifier.classify(high_vol_features, vix_value=40.0)
        assert regime.volatility == VolRegime.EXTREME

    def test_trend_classification(self):
        """Test trend regime classification."""
        classifier = RegimeClassifier()

        # Strong trend
        features = FeatureVector(adx_14=35, price_vs_sma_20=0.05, price_vs_sma_50=0.08)
        regime = classifier.classify(features, vix_value=20.0)
        assert regime.trend == TrendRegime.STRONG_TREND

        # Choppy
        features = FeatureVector(adx_14=10, price_vs_sma_20=0.0)
        regime = classifier.classify(features, vix_value=20.0)
        assert regime.trend == TrendRegime.CHOPPY


class TestBiasEngine:
    """Test BiasEngine."""

    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        return FeatureVector(
            price_vs_sma_20=0.03,
            price_vs_sma_50=0.05,
            macd_histogram=50,
            rsi_14=65,
            adx_14=28,
            atr_14=100,
            distance_to_support=200,
            distance_to_resistance=500,
            market_breadth_ratio=1.3,
        )

    @pytest.fixture
    def sample_regime(self):
        """Create sample regime."""
        return RegimeState(
            volatility=VolRegime.NORMAL,
            trend=TrendRegime.STRONG_TREND,
            risk_appetite=RiskAppetite.RISK_ON,
            momentum=MomentumRegime.ACCELERATING,
            event_risk=EventRisk.CLEAR,
            composite_score=0.75,
        )

    def test_get_daily_bias(self, sample_features, sample_regime):
        """Test bias generation."""
        engine = BiasEngine()
        bias = engine.get_daily_bias("NAS100", sample_features, sample_regime)

        assert isinstance(bias, BiasOutput)
        assert bias.model_version == "v1.0"
        assert len(bias.rationale) > 0

    def test_direction_prediction(self, sample_features):
        """Test direction prediction."""
        engine = BiasEngine()

        # Bullish features
        bullish_features = FeatureVector(
            price_vs_sma_20=0.05, rsi_14=65, macd_histogram=100
        )
        direction, confidence = engine._predict_direction(bullish_features)
        assert direction in [Direction.LONG, Direction.NEUTRAL]

        # Bearish features
        bearish_features = FeatureVector(
            price_vs_sma_20=-0.05, rsi_14=35, macd_histogram=-100
        )
        direction, confidence = engine._predict_direction(bearish_features)
        assert direction in [Direction.SHORT, Direction.NEUTRAL]

    def test_magnitude_determination(self):
        """Test magnitude determination."""
        engine = BiasEngine()

        # High confidence + high ADX = Large
        features = FeatureVector(adx_14=35)
        mag = engine._determine_magnitude(0.85, features)
        assert mag == Magnitude.LARGE

        # Low confidence = Small
        mag = engine._determine_magnitude(0.55, features)
        assert mag == Magnitude.SMALL

    def test_regime_override(self, sample_features):
        """Test regime override detection."""
        engine = BiasEngine()

        # Extreme event risk
        regime = RegimeState(
            VolRegime.NORMAL,
            TrendRegime.STRONG_TREND,
            RiskAppetite.RISK_ON,
            MomentumRegime.ACCELERATING,
            EventRisk.EXTREME,
            0.5,
        )
        override = engine._check_regime_override(regime)
        assert override is True

        # Clear conditions
        regime = RegimeState(
            VolRegime.NORMAL,
            TrendRegime.STRONG_TREND,
            RiskAppetite.RISK_ON,
            MomentumRegime.ACCELERATING,
            EventRisk.CLEAR,
            0.75,
        )
        override = engine._check_regime_override(regime)
        assert override is False

    def test_rationale_generation(self):
        """Test rationale generation."""
        engine = BiasEngine()

        # Features that trigger multiple rationales
        features = FeatureVector(
            atr_percent_14=0.03,
            adx_14=30,
            rsi_slope_5=5,
            distance_to_support=50,
            atr_14=100,
        )
        rationale = engine._generate_rationale(features)

        assert len(rationale) > 0
        # Should contain canonical group names
        assert all(isinstance(r, str) for r in rationale)


class TestSHAPExplainer:
    """Test SHAPExplainer."""

    def test_explain_prediction(self):
        """Test feature importance explanation."""
        explainer = SHAPExplainer()

        features = FeatureVector(atr_14=150, adx_14=35, rsi_14=70)

        importances = explainer.explain_prediction(features, Direction.LONG)

        assert isinstance(importances, dict)
        assert len(importances) > 0
        # Should sum to approximately 1
        assert abs(sum(importances.values()) - 1.0) < 0.01


class TestHardConstraints:
    """Test HardConstraints."""

    @pytest.fixture
    def sample_account(self):
        """Create sample account state."""
        return AccountState(
            account_id="test",
            equity=100000,
            balance=100000,
            open_positions=1,
            daily_pnl=500,
            daily_loss_pct=0.005,
            margin_used=10000,
            margin_available=40000,
            timestamp=datetime.utcnow(),
        )

    @pytest.fixture
    def sample_risk(self):
        """Create sample risk output."""
        return RiskOutput(
            position_size=1000,
            kelly_fraction=0.15,
            stop_price=19500,
            stop_method="atr",
            tp1_price=20500,
            tp2_price=21000,
            trail_config={},
            expected_value=0.05,
            ev_positive=True,
            size_breakdown={},
        )

    @pytest.fixture
    def sample_bias(self):
        """Create sample bias output."""
        return BiasOutput(
            direction=Direction.LONG,
            magnitude=Magnitude.NORMAL,
            confidence=0.75,
            regime_override=False,
            rationale=["TREND_STRENGTH"],
            model_version="v1.0",
            feature_snapshot={},
        )

    @pytest.fixture
    def sample_regime(self):
        """Create sample regime state."""
        return RegimeState(
            volatility=VolRegime.NORMAL,
            trend=TrendRegime.STRONG_TREND,
            risk_appetite=RiskAppetite.RISK_ON,
            momentum=MomentumRegime.ACCELERATING,
            event_risk=EventRisk.CLEAR,
            composite_score=0.75,
        )

    def test_all_constraints_pass(
        self, sample_bias, sample_risk, sample_regime, sample_account
    ):
        """Test that valid inputs pass all constraints."""
        constraints = HardConstraints()
        # Use a valid trading time (10:30 AM)
        trading_time = datetime(2024, 1, 15, 10, 30)
        check = constraints.check_all_constraints(
            sample_bias, sample_risk, sample_regime, sample_account, trading_time
        )

        assert check.passed is True

    def test_daily_loss_limit(self, sample_account):
        """Test daily loss limit constraint."""
        constraints = HardConstraints()

        # Account at loss limit
        sample_account.daily_loss_pct = 0.03
        check = constraints.check_daily_loss_limit(sample_account)
        assert check.passed is False
        assert check.severity == "BLOCK"

        # Account within limits
        sample_account.daily_loss_pct = 0.01
        check = constraints.check_daily_loss_limit(sample_account)
        assert check.passed is True

    def test_event_risk_block(self, sample_regime):
        """Test event risk blocking."""
        constraints = HardConstraints()

        # Extreme event risk
        sample_regime.event_risk = EventRisk.EXTREME
        check = constraints.check_event_risk(sample_regime)
        assert check.passed is False
        assert check.severity == "BLOCK"

    def test_max_positions(self, sample_account):
        """Test max positions constraint."""
        constraints = HardConstraints()

        sample_account.open_positions = 5
        check = constraints.check_max_positions(sample_account)
        assert check.passed is False

    def test_position_size_limit(self, sample_risk, sample_account):
        """Test position size constraint."""
        constraints = HardConstraints()

        sample_risk.position_size = 10000  # 10% of equity
        check = constraints.check_position_size(sample_risk, sample_account)
        assert check.passed is False

    def test_trading_hours_pre_market(
        self, sample_bias, sample_risk, sample_regime, sample_account
    ):
        """Test pre-market trading block."""
        constraints = HardConstraints()

        # 9 AM - pre-market
        pre_market = datetime(2024, 1, 15, 9, 0)
        check = constraints.check_trading_hours(pre_market)
        assert check.passed is False

    def test_trading_hours_after_hours(
        self, sample_bias, sample_risk, sample_regime, sample_account
    ):
        """Test after-hours trading block."""
        constraints = HardConstraints()

        # 4 PM - after hours
        after_hours = datetime(2024, 1, 15, 16, 0)
        check = constraints.check_trading_hours(after_hours)
        assert check.passed is False

    def test_positive_ev(self, sample_risk):
        """Test positive EV constraint."""
        constraints = HardConstraints()

        sample_risk.ev_positive = False
        check = constraints.check_positive_ev(sample_risk)
        assert check.passed is False

    def test_confidence_threshold(self, sample_bias):
        """Test confidence threshold."""
        constraints = HardConstraints()

        sample_bias.confidence = 0.5  # Below 0.55 threshold
        check = constraints.check_confidence_threshold(sample_bias)
        assert check.passed is False

    def test_bias_neutral(self, sample_bias):
        """Test neutral bias block."""
        constraints = HardConstraints()

        sample_bias.direction = Direction.NEUTRAL
        check = constraints.check_bias_neutral(sample_bias)
        assert check.passed is False

    def test_block_history(
        self, sample_bias, sample_risk, sample_regime, sample_account
    ):
        """Test block history tracking."""
        constraints = HardConstraints()

        # Trigger a block
        sample_account.daily_loss_pct = 0.05
        constraints.check_all_constraints(
            sample_bias, sample_risk, sample_regime, sample_account
        )

        history = constraints.get_block_history()
        assert len(history) > 0
        assert history[0]["severity"] == "BLOCK"
