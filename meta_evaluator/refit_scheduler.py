"""Refit Scheduler - Model retraining scheduler and tracker."""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

URGENCY_LEVELS = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


def _max_urgency(a: str, b: str) -> str:
    """Return the more severe urgency between two values."""
    return a if URGENCY_LEVELS.get(a, 0) >= URGENCY_LEVELS.get(b, 0) else b


class RefitScheduler:
    """Schedule and track model retraining based on performance metrics."""

    def __init__(self, firebase_client=None, config: Optional[Dict] = None):
        self.firebase = firebase_client
        self.config = config or {}
        self.last_refit: Optional[datetime] = None

        # Default thresholds
        self.min_trades_for_refit = self.config.get("min_trades", 50)
        self.win_rate_threshold = self.config.get("win_rate_threshold", 0.45)
        self.sharpe_threshold = self.config.get("sharpe_threshold", 0.5)
        self.max_days_between_refits = self.config.get("max_days", 30)

    def evaluate_refit_need(self, performance_metrics: Dict[str, Any], drift_detected: bool = False) -> Dict[str, Any]:
        """Evaluate if model retraining is needed.

        Args:
            performance_metrics: Dict with win_rate, sharpe, total_trades, etc.
            drift_detected: Whether feature drift was detected

        Returns:
            Dict with should_refit (bool), reasons (list), and urgency (str)
        """
        reasons = []
        urgency = "LOW"

        # Check win rate
        win_rate = performance_metrics.get("win_rate", 0)
        if win_rate < self.win_rate_threshold:
            reasons.append(f"Win rate {win_rate:.1%} below threshold {self.win_rate_threshold:.1%}")
            urgency = "HIGH"

        # Check Sharpe ratio
        sharpe = performance_metrics.get("sharpe", 0)
        if sharpe < self.sharpe_threshold:
            reasons.append(f"Sharpe {sharpe:.2f} below threshold {self.sharpe_threshold:.2f}")
            urgency = _max_urgency(urgency, "MEDIUM")

        # Check feature drift
        if drift_detected:
            reasons.append("Significant feature drift detected")
            urgency = "HIGH"

        # Check trade count sufficiency
        total_trades = performance_metrics.get("total_trades", 0)
        if total_trades < self.min_trades_for_refit:
            return {
                "should_refit": False,
                "reasons": [f"Insufficient trade history ({total_trades} < {self.min_trades_for_refit})"],
                "urgency": "LOW",
                "can_schedule": False,
            }

        # Check time-based refit
        if self.last_refit:
            days_since = (datetime.now() - self.last_refit).days
            if days_since >= self.max_days_between_refits:
                reasons.append(f"{days_since} days since last refit (max: {self.max_days_between_refits})")
                urgency = _max_urgency(urgency, "MEDIUM")

        should_refit = len(reasons) > 0 and urgency in ["HIGH", "MEDIUM"]

        return {
            "should_refit": should_refit,
            "reasons": reasons,
            "urgency": urgency,
            "can_schedule": total_trades >= self.min_trades_for_refit,
            "performance_summary": {
                "win_rate": win_rate,
                "sharpe": sharpe,
                "total_trades": total_trades,
            },
        }

    def schedule_refit(self, evaluation_result: Dict[str, Any], model_version: str) -> Dict[str, Any]:
        """Schedule a model refit if recommended.

        Returns:
            Schedule details or rejection reason
        """
        if not evaluation_result["should_refit"]:
            return {
                "scheduled": False,
                "reason": "No refit needed based on evaluation",
                "evaluation": evaluation_result,
            }

        if not evaluation_result["can_schedule"]:
            return {
                "scheduled": False,
                "reason": "Insufficient data for reliable retraining",
                "evaluation": evaluation_result,
            }

        # Schedule for next maintenance window (Friday 5 PM)
        now = datetime.now()
        days_until_friday = (4 - now.weekday()) % 7
        if days_until_friday == 0 and now.hour >= 17:
            days_until_friday = 7

        scheduled_time = now + timedelta(days=days_until_friday)
        scheduled_time = scheduled_time.replace(hour=17, minute=0, second=0)

        schedule = {
            "scheduled": True,
            "model_version": model_version,
            "scheduled_time": scheduled_time.isoformat(),
            "reasons": evaluation_result["reasons"],
            "urgency": evaluation_result.get("urgency", "MEDIUM"),
            "estimated_duration_minutes": 30,
            "status": "PENDING",
        }

        logger.info(f"Refit scheduled for {schedule['scheduled_time']}: {model_version}")

        if self.firebase:
            self.firebase.write("refit_schedule", schedule)

        return schedule

    def record_refit_completed(self, model_version: str, old_performance: Dict, new_performance: Dict) -> Dict[str, Any]:
        """Record completion of a model refit.

        Returns:
            Refit completion record
        """
        self.last_refit = datetime.now()

        win_rate_improvement = new_performance.get("win_rate", 0) - old_performance.get("win_rate", 0)
        sharpe_improvement = new_performance.get("sharpe", 0) - old_performance.get("sharpe", 0)

        record = {
            "timestamp": self.last_refit.isoformat(),
            "model_version": model_version,
            "old_performance": old_performance,
            "new_performance": new_performance,
            "improvements": {
                "win_rate_delta": round(win_rate_improvement, 4),
                "sharpe_delta": round(sharpe_improvement, 2),
                "successful": win_rate_improvement > 0 or sharpe_improvement > 0,
            },
        }

        logger.info(f"Refit completed: {model_version}, " f"win_rate improvement: {win_rate_improvement:+.1%}")

        if self.firebase:
            self.firebase.write("refit_history", record)

        return record

    def get_refit_history(self, limit: int = 10) -> list:
        """Get history of recent refits."""
        if not self.firebase:
            return []

        # Query Firebase for refit history
        return self.firebase.query("refit_history", limit=limit, order_by="timestamp")
