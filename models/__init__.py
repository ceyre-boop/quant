"""Model Registry - Version control for ML models.

Manages model versions, metrics tracking, and registry updates.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Model version registry."""
    
    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        
        return {
            "current_version": "v1.0",
            "models": [],
            "registry_info": {
                "description": "Model version registry",
                "update_command": "python -m meta_evaluator.refit_scheduler --register-model"
            }
        }
    
    def register_model(
        self,
        version: str,
        model_path: str,
        metrics: Optional[Dict[str, Any]] = None,
        commit: Optional[str] = None
    ) -> None:
        """Register a new model version.
        
        Args:
            version: Version string (e.g., 'v1.1')
            model_path: Path to model file
            metrics: Performance metrics (sharpe, win_rate, etc.)
            commit: Git commit hash
        """
        model_entry = {
            "version": version,
            "path": model_path,
            "trained_at": datetime.now().strftime("%Y-%m-%d"),
            "commit": commit or "unknown",
            "metrics": metrics or {},
            "status": "active"
        }
        
        # Deactivate previous models
        for model in self.registry["models"]:
            model["status"] = "archived"
        
        # Add new model
        self.registry["models"].append(model_entry)
        self.registry["current_version"] = version
        
        self._save_registry()
        logger.info(f"Registered model {version}")
    
    def get_current_model(self) -> Optional[Dict[str, Any]]:
        """Get current active model."""
        current_version = self.registry.get("current_version")
        
        for model in self.registry["models"]:
            if model["version"] == current_version and model["status"] == "active":
                return model
        
        return None
    
    def get_model(self, version: str) -> Optional[Dict[str, Any]]:
        """Get specific model version."""
        for model in self.registry["models"]:
            if model["version"] == version:
                return model
        
        return None
    
    def get_all_models(self) -> List[Dict[str, Any]]:
        """Get all registered models."""
        return self.registry["models"]
    
    def update_metrics(self, version: str, metrics: Dict[str, Any]) -> None:
        """Update metrics for a model version."""
        for model in self.registry["models"]:
            if model["version"] == version:
                model["metrics"].update(metrics)
                self._save_registry()
                logger.info(f"Updated metrics for {version}")
                return
        
        logger.warning(f"Model {version} not found")
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get performance history across versions."""
        history = []
        
        for model in sorted(self.registry["models"], key=lambda x: x["trained_at"]):
            if model["metrics"]:
                history.append({
                    "version": model["version"],
                    "trained_at": model["trained_at"],
                    "metrics": model["metrics"]
                })
        
        return history


def create_model_registry() -> ModelRegistry:
    """Factory function to create model registry."""
    return ModelRegistry()
