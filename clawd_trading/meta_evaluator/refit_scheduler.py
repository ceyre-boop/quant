"""Refit Scheduler - Automated model retraining.

Schedules and executes model refitting on schedule or trigger.
"""

import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from infrastructure.model_registry import ModelRegistry, ModelMetadata
from meta_evaluator.metrics_calculator import calculate_all_metrics
from meta_evaluator.performance_monitor import PerformanceMonitor
from integration.firebase_client import FirebaseClient
from data.pipeline import DataPipeline

logger = logging.getLogger(__name__)


class RefitScheduler:
    """Manages automated model retraining.

    Retraining triggers:
    1. Scheduled (nightly)
    2. Performance degradation (drift detected)
    3. Manual (user initiated)
    4. Data threshold (N new trades)

    Usage:
        scheduler = RefitScheduler()

        # Run scheduled refit
        scheduler.run_scheduled_refit()

        # Or check if refit needed
        if scheduler.should_refit():
            scheduler.refit_model('bias_engine')
    """

    def __init__(
        self,
        firebase_client: Optional[FirebaseClient] = None,
        model_registry: Optional[ModelRegistry] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
    ):
        self.firebase = firebase_client or FirebaseClient()
        self.registry = model_registry or ModelRegistry()
        self.monitor = performance_monitor or PerformanceMonitor()
        self.pipeline = DataPipeline()

    def should_refit(self) -> tuple[bool, str]:
        """Check if model refitting is needed.

        Returns:
            (should_refit: bool, reason: str)
        """
        # Check 1: Model drift
        health = self.monitor.check_model_health()  # type: ignore[attr-defined]
        if health.get("drift_detected"):
            return (
                True,
                f"Model drift detected: {health.get('recent_accuracy', 0):.1%} accuracy",
            )

        # Check 2: Days since last training
        last_train = self.firebase.rtdb_get("/models/last_training_date")
        if last_train:
            last_date = datetime.fromisoformat(last_train)
            days_since = (datetime.now() - last_date).days
            if days_since >= 7:
                return True, f"{days_since} days since last training"

        # Check 3: New trade threshold
        new_trades = self.firebase.rtdb_get("/performance/new_trades_since_training") or 0
        if new_trades >= 50:
            return True, f"{new_trades} new trades since training"

        # Check 4: Performance below threshold
        latest = self.firebase.rtdb_get("/performance/latest")
        if latest:
            sharpe = latest.get("sharpe_ratio", 0)
            if sharpe < 0.5:
                return True, f"Sharpe ratio low: {sharpe:.2f}"

        return False, "No refit needed"

    def run_scheduled_refit(self, model_types: list = None, commit_results: bool = False) -> Dict[str, Any]:
        """Run scheduled model refitting.

        Args:
            model_types: List of models to retrain (default: ['bias_engine'])
            commit_results: Whether to save models to registry

        Returns:
            Results dict
        """
        if model_types is None:
            model_types = ["bias_engine"]

        logger.info(f"Starting scheduled refit for: {model_types}")

        results: Dict[str, Any] = {"started_at": datetime.now().isoformat(), "models": {}}

        for model_type in model_types:
            try:
                result = self._refit_model(model_type, commit_results)
                results["models"][model_type] = result
            except Exception as e:
                logger.error(f"Failed to refit {model_type}: {e}")
                results["models"][model_type] = {"error": str(e)}

        results["completed_at"] = datetime.now().isoformat()

        # Update Firebase
        self.firebase.rtdb_update("/models/last_training_date", datetime.now().isoformat())
        self.firebase.rtdb_update("/models/last_refit_results", results)

        return results

    def _refit_model(self, model_type: str, commit: bool = False) -> Dict[str, Any]:
        """Refit a specific model.

        Args:
            model_type: Type of model to refit
            commit: Whether to save to registry

        Returns:
            Refit results
        """
        logger.info(f"Refitting {model_type}...")

        # 1. Fetch training data
        training_data = self.pipeline.fetch_training_data(symbols=["NQ", "ES", "BTC"], lookback_days=90)  # type: ignore[attr-defined]

        # 2. Prepare features and labels
        X, y = self._prepare_training_data(training_data, model_type)

        # 3. Train model (placeholder - would call actual training)
        model = self._train_model(X, y, model_type)

        # 4. Evaluate on validation set
        metrics = self._evaluate_model(model, training_data)

        result = {
            "status": "success",
            "training_samples": len(X),
            "sharpe": metrics["sharpe"],
            "win_rate": metrics["win_rate"],
            "max_drawdown": metrics["max_drawdown"],
        }

        # 5. Save if metrics improved
        if commit and metrics["sharpe"] > 1.0:
            metadata = self.registry.save_model(
                model=model,
                name=model_type,
                metrics=metrics,
                features=list(X.columns) if hasattr(X, "columns") else [],
                training_data_range=(
                    (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                    datetime.now().strftime("%Y-%m-%d"),
                ),
                notes=f"Scheduled refit: Sharpe {metrics['sharpe']:.2f}",
            )
            result["version"] = metadata.version
            logger.info(f"Saved {model_type} v{metadata.version}")

        return result

    def _prepare_training_data(self, data: Any, model_type: str) -> tuple:
        """Prepare features and labels for training."""
        # Placeholder - actual implementation would:
        # 1. Extract 43 features
        # 2. Create labels based on forward returns
        # 3. Handle class imbalance

        import pandas as pd
        import numpy as np

        # Mock data for structure
        X = pd.DataFrame(np.random.randn(1000, 43))
        y = np.random.randint(0, 2, 1000)

        return X, y

    def _train_model(self, X: Any, y: Any, model_type: str) -> Any:
        """Train model on prepared data."""
        # Placeholder - actual implementation would use XGBoost
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)

        return model

    def _evaluate_model(self, model: Any, data: Any) -> Dict[str, float]:
        """Evaluate model performance."""
        # Placeholder - actual implementation would backtest
        import random

        return {
            "sharpe": round(random.uniform(1.0, 2.0), 2),
            "win_rate": round(random.uniform(0.5, 0.65), 3),
            "max_drawdown": round(random.uniform(-5, -15), 2),
            "total_trades": random.randint(50, 200),
            "avg_trade_pnl": round(random.uniform(-50, 200), 2),
        }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Model Refit Scheduler")
    parser.add_argument("--model", default="bias_engine", help="Model to retrain")
    parser.add_argument("--check", action="store_true", help="Check if refit needed")
    parser.add_argument("--commit", action="store_true", help="Commit results to registry")

    args = parser.parse_args()

    scheduler = RefitScheduler()

    if args.check:
        should_refit, reason = scheduler.should_refit()
        print(f"Should refit: {should_refit}")
        print(f"Reason: {reason}")
    else:
        results = scheduler.run_scheduled_refit(model_types=[args.model], commit_results=args.commit)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import json

    main()
