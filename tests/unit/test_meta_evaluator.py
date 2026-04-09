"""Unit tests for Meta-Evaluator components."""

import pytest
from datetime import datetime, timedelta
from meta_evaluator.analyzer import (
    WeeklyAnalyzer,
    FeatureGroupTracker,
    RefitScheduler,
    MetaEvaluator,
)
from meta_evaluator.refit_scheduler import RefitScheduler as RefitSchedulerV2
from meta_evaluator.feature_group_tracker import FeatureGroupTracker as FeatureTrackerV2


class TestWeeklyAnalyzer:
    """Test WeeklyAnalyzer."""

    def test_analyze_week_with_trades(self):
        analyzer = WeeklyAnalyzer()

        trades = [
            {"realized_pnl": 100, "symbol": "NAS100"},
            {"realized_pnl": -50, "symbol": "NAS100"},
            {"realized_pnl": 200, "symbol": "NAS100"},
        ]

        result = analyzer.analyze_week(trades, datetime(2024, 1, 1))

        assert result["total_trades"] == 3
        assert result["winning_trades"] == 2
        assert result["losing_trades"] == 1
        assert result["win_rate"] == pytest.approx(0.6667, rel=0.01)
        assert result["profit_loss"] == 250
        assert result["status"] == "HEALTHY"

    def test_analyze_week_no_trades(self):
        analyzer = WeeklyAnalyzer()
        result = analyzer.analyze_week([], datetime(2024, 1, 1))

        assert result["total_trades"] == 0
        assert result["win_rate"] == 0
        assert result["status"] == "NO_DATA"

    def test_sharpe_calculation(self):
        analyzer = WeeklyAnalyzer()

        # Consistent returns = higher Sharpe
        trades = [{"realized_pnl": 100} for _ in range(10)]
        result = analyzer.analyze_week(trades, datetime(2024, 1, 1))

        # All same returns = infinite Sharpe, but we handle as 0 variance
        assert "sharpe" in result


class TestFeatureGroupTracker:
    """Test FeatureGroupTracker."""

    def test_set_baseline(self):
        tracker = FeatureGroupTracker()

        baseline = {
            "VOLATILITY_SPIKE": 0.25,
            "TREND_STRENGTH": 0.30,
            "MOMENTUM_SHIFT": 0.20,
        }

        tracker.set_baseline(baseline)

        assert tracker.baseline_importance == baseline

    def test_detect_drift_no_baseline(self):
        tracker = FeatureGroupTracker()

        drift = tracker.detect_drift()

        assert drift["drift_detected"] is False
        assert drift["status"] == "NO_BASELINE"

    def test_detect_drift_no_current(self):
        tracker = FeatureGroupTracker()
        tracker.set_baseline({"VOLATILITY_SPIKE": 0.25})

        drift = tracker.detect_drift()

        assert drift["drift_detected"] is False
        assert drift["status"] == "NO_DATA"

    def test_detect_drift_with_data(self):
        tracker = FeatureGroupTracker(drift_threshold=0.2)

        tracker.set_baseline(
            {
                "VOLATILITY_SPIKE": 0.25,
                "TREND_STRENGTH": 0.30,
            }
        )

        # Small change - no drift
        tracker.update_current(
            {
                "VOLATILITY_SPIKE": 0.26,
                "TREND_STRENGTH": 0.31,
            }
        )

        drift = tracker.detect_drift()
        assert drift["drift_detected"] is False

        # Large change - drift detected
        tracker.update_current(
            {
                "VOLATILITY_SPIKE": 0.50,  # 100% increase
                "TREND_STRENGTH": 0.15,  # 50% decrease
            }
        )

        drift = tracker.detect_drift()
        assert drift["drift_detected"] is True
        assert drift["status"] in ["WARNING", "CRITICAL"]
        assert len(drift["feature_details"]) > 0

    def test_get_top_features(self):
        tracker = FeatureGroupTracker()

        tracker.update_current(
            {
                "VOLATILITY_SPIKE": 0.10,
                "TREND_STRENGTH": 0.50,
                "MOMENTUM_SHIFT": 0.30,
            }
        )

        top = tracker.get_top_features(2)

        assert len(top) == 2
        assert top[0]["feature_group"] == "TREND_STRENGTH"
        assert top[0]["rank"] == 1


class TestRefitScheduler:
    """Test RefitScheduler."""

    def test_should_refit_performance_degradation(self):
        scheduler = RefitScheduler()

        weekly = {"win_rate": 0.40, "sharpe": 0.3, "total_trades": 100}
        drift = {"drift_detected": False}

        result = scheduler.should_refit(weekly, drift)

        assert result["should_refit"] is True
        assert any("Win rate" in r for r in result["reasons"])

    def test_should_refit_drift_detected(self):
        scheduler = RefitScheduler()

        weekly = {"win_rate": 0.55, "sharpe": 0.8, "total_trades": 100}
        drift = {"drift_detected": True}

        result = scheduler.should_refit(weekly, drift)

        assert result["should_refit"] is True
        assert any("drift" in r.lower() for r in result["reasons"])

    def test_should_not_refit_healthy(self):
        scheduler = RefitScheduler()

        weekly = {"win_rate": 0.55, "sharpe": 0.8, "total_trades": 100}
        drift = {"drift_detected": False}

        result = scheduler.should_refit(weekly, drift)

        assert result["should_refit"] is False

    def test_should_not_refit_insufficient_trades(self):
        scheduler = RefitScheduler()

        weekly = {"win_rate": 0.30, "sharpe": 0.2, "total_trades": 10}
        drift = {"drift_detected": True}

        result = scheduler.should_refit(weekly, drift)

        # Should not recommend refit with insufficient data
        assert result["should_refit"] is False


class TestMetaEvaluator:
    """Test MetaEvaluator integration."""

    def test_run_weekly_evaluation(self):
        meta = MetaEvaluator()

        trades = [
            {"realized_pnl": 100, "symbol": "NAS100"},
            {"realized_pnl": 200, "symbol": "NAS100"},
        ]

        feature_importance = {
            "VOLATILITY_SPIKE": 0.25,
            "TREND_STRENGTH": 0.30,
        }

        report = meta.run_weekly_evaluation(
            trades, feature_importance, datetime(2024, 1, 1)
        )

        assert "performance" in report
        assert "feature_analysis" in report
        assert "recommendations" in report
        assert report["performance"]["total_trades"] == 2


class TestRefitSchedulerV2:
    """Test RefitScheduler V2."""

    def test_evaluate_refit_need_performance(self):
        scheduler = RefitSchedulerV2()

        metrics = {"win_rate": 0.40, "sharpe": 0.3, "total_trades": 100}
        result = scheduler.evaluate_refit_need(metrics, drift_detected=False)

        assert result["should_refit"] is True
        assert result["urgency"] == "HIGH"

    def test_schedule_refit(self):
        scheduler = RefitSchedulerV2()

        evaluation = {
            "should_refit": True,
            "can_schedule": True,
            "reasons": ["Performance degraded"],
        }

        schedule = scheduler.schedule_refit(evaluation, "model_v2")

        assert schedule["scheduled"] is True
        assert schedule["model_version"] == "model_v2"
        assert "scheduled_time" in schedule


class TestFeatureTrackerV2:
    """Test FeatureGroupTracker V2."""

    def test_detect_drift_with_baseline(self):
        tracker = FeatureTrackerV2(drift_threshold=0.25)

        tracker.set_baseline(
            {
                "VOLATILITY_SPIKE": 0.25,
                "TREND_STRENGTH": 0.30,
                "MOMENTUM_SHIFT": 0.20,
            }
        )

        # No drift
        tracker.update_current(
            {
                "VOLATILITY_SPIKE": 0.26,
                "TREND_STRENGTH": 0.29,
                "MOMENTUM_SHIFT": 0.21,
            }
        )

        drift = tracker.detect_drift()
        assert drift["drift_detected"] is False
        assert drift["status"] == "HEALTHY"

        # With drift
        tracker.update_current(
            {
                "VOLATILITY_SPIKE": 0.50,  # 100% increase
                "TREND_STRENGTH": 0.15,  # 50% decrease
                "MOMENTUM_SHIFT": 0.20,
            }
        )

        drift = tracker.detect_drift()
        assert drift["drift_detected"] is True
        assert drift["features_drifted"] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
