"""Feature Group Tracker - Monitor feature importance and detect drift."""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FeatureGroupTracker:
    """Track feature importance and detect drift over time."""

    # Canonical feature group names from blueprint
    FEATURE_GROUPS = [
        "VOLATILITY_SPIKE",
        "TREND_STRENGTH",
        "MOMENTUM_SHIFT",
        "SUPPORT_RESISTANCE",
        "MARKET_BREADTH",
        "SENTIMENT_EXTREME",
        "REGIME_ALIGNMENT",
    ]

    def __init__(self, drift_threshold: float = 0.25):
        self.drift_threshold = drift_threshold
        self.baseline_importance = {}
        self.current_importance = {}
        self.importance_history = []

    def set_baseline(self, feature_importance: Dict[str, float]):
        """Set baseline feature importance from initial model training.

        Args:
            feature_importance: Dict mapping feature group names to importance scores
        """
        self.baseline_importance = {k: float(v) for k, v in feature_importance.items() if k in self.FEATURE_GROUPS}
        logger.info(f"Baseline set for {len(self.baseline_importance)} feature groups")

    def update_current(self, feature_importance: Dict[str, float]):
        """Update current feature importance from recent SHAP values.

        Args:
            feature_importance: Dict mapping feature group names to importance scores
        """
        self.current_importance = {k: float(v) for k, v in feature_importance.items() if k in self.FEATURE_GROUPS}

        # Add to history
        self.importance_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "importance": self.current_importance.copy(),
            }
        )

        # Keep only last 90 days
        if len(self.importance_history) > 90:
            self.importance_history = self.importance_history[-90:]

    def detect_drift(self) -> Dict[str, Any]:
        """Detect drift between baseline and current feature importance.

        Returns:
            Dict with drift status, severity, and per-feature analysis
        """
        if not self.baseline_importance:
            return {
                "drift_detected": False,
                "status": "NO_BASELINE",
                "message": "Baseline not set, cannot detect drift",
                "timestamp": datetime.now().isoformat(),
            }

        if not self.current_importance:
            return {
                "drift_detected": False,
                "status": "NO_DATA",
                "message": "No current importance data",
                "timestamp": datetime.now().isoformat(),
            }

        drifted_features = []
        total_drift_score = 0

        for feature in self.FEATURE_GROUPS:
            baseline = self.baseline_importance.get(feature, 0)
            current = self.current_importance.get(feature, 0)

            if baseline > 0:
                relative_change = abs(current - baseline) / baseline
                absolute_change = abs(current - baseline)

                drift_entry = {
                    "feature_group": feature,
                    "baseline_importance": round(baseline, 4),
                    "current_importance": round(current, 4),
                    "relative_change": round(relative_change, 2),
                    "absolute_change": round(absolute_change, 4),
                    "direction": "INCREASED" if current > baseline else "DECREASED",
                }

                if relative_change > self.drift_threshold:
                    drift_entry["drift_detected"] = True
                    drift_entry["severity"] = "HIGH" if relative_change > 0.5 else "MEDIUM"
                    drifted_features.append(drift_entry)
                    total_drift_score += relative_change
                else:
                    drift_entry["drift_detected"] = False
                    drift_entry["severity"] = "LOW"
            else:
                # New feature appearing
                if current > 0.01:  # Threshold for significance
                    drifted_features.append(
                        {
                            "feature_group": feature,
                            "baseline_importance": 0,
                            "current_importance": round(current, 4),
                            "relative_change": float("inf"),
                            "absolute_change": round(current, 4),
                            "direction": "NEW",
                            "drift_detected": True,
                            "severity": "MEDIUM",
                        }
                    )

        # Determine overall status
        high_severity = [f for f in drifted_features if f.get("severity") == "HIGH"]
        medium_severity = [f for f in drifted_features if f.get("severity") == "MEDIUM"]

        if high_severity:
            status = "CRITICAL"
        elif medium_severity:
            status = "WARNING"
        elif [f for f in drifted_features if f.get("drift_detected")]:
            status = "ATTENTION"
        else:
            status = "HEALTHY"

        return {
            "drift_detected": len([f for f in drifted_features if f.get("drift_detected")]) > 0,
            "status": status,
            "drift_score": round(total_drift_score, 2),
            "features_analyzed": len(self.FEATURE_GROUPS),
            "features_drifted": len([f for f in drifted_features if f.get("drift_detected")]),
            "high_severity_count": len(high_severity),
            "medium_severity_count": len(medium_severity),
            "feature_details": drifted_features,
            "timestamp": datetime.now().isoformat(),
        }

    def get_importance_trend(self, feature_group: str, days: int = 30) -> List[Dict]:
        """Get historical importance trend for a specific feature group.

        Returns:
            List of {date, importance} dicts
        """
        recent_history = self.importance_history[-days:] if len(self.importance_history) > days else self.importance_history

        return [
            {
                "date": entry["timestamp"],
                "importance": entry["importance"].get(feature_group, 0),
            }
            for entry in recent_history
        ]

    def get_top_features(self, n: int = 5, use_current: bool = True) -> List[Dict[str, Any]]:
        """Get top N most important features.

        Args:
            n: Number of features to return
            use_current: If True, use current importance; else use baseline

        Returns:
            List of feature dicts sorted by importance
        """
        source = self.current_importance if use_current else self.baseline_importance

        sorted_features = sorted(source.items(), key=lambda x: x[1], reverse=True)

        return [
            {"feature_group": name, "importance": round(importance, 4), "rank": i + 1}
            for i, (name, importance) in enumerate(sorted_features[:n])
        ]

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive feature importance report.

        Returns:
            Full report with drift status, trends, and top features
        """
        drift = self.detect_drift()

        return {
            "report_date": datetime.now().isoformat(),
            "drift_analysis": drift,
            "current_top_features": self.get_top_features(5, use_current=True),
            "baseline_top_features": (self.get_top_features(5, use_current=False) if self.baseline_importance else []),
            "summary": {
                "total_features_tracked": len(self.FEATURE_GROUPS),
                "baseline_established": bool(self.baseline_importance),
                "days_of_history": len(self.importance_history),
                "recommendation": ("REFIT" if drift["status"] in ["CRITICAL", "WARNING"] else "MONITOR"),
            },
        }
