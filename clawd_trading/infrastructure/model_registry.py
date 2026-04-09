"""Model Registry - Version control for ML models.

Tracks trained models with metadata for reproducibility and rollback.
"""

import os
import json
import logging
import pickle
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""

    name: str
    version: int
    training_date: str
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    total_trades: int
    avg_trade_pnl: float
    features_used: List[str]
    hyperparameters: Dict[str, Any]
    training_data_start: str
    training_data_end: str
    git_commit: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        return cls(**data)


class ModelRegistry:
    """Registry for managing ML model versions.

    Handles:
    - Saving models with versioned filenames
    - Tracking metadata (performance, training params)
    - Loading specific versions
    - Listing available models

    Directory structure:
        models/
        ├── bias_engine_v1_2026-03-01.pkl
        ├── bias_engine_v1_2026-03-01.json
        ├── bias_engine_v2_2026-03-10.pkl
        ├── bias_engine_v2_2026-03-10.json
        └── README.md

    Usage:
        registry = ModelRegistry()

        # Save new model
        registry.save_model(
            model=xgb_model,
            name='bias_engine',
            metrics={'sharpe': 1.42, 'win_rate': 0.58}
        )

        # Load latest
        model, metadata = registry.load_model('bias_engine')

        # Load specific version
        model, metadata = registry.load_model('bias_engine', version=2)
    """

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            # Default to models/ directory relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")

        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

        # Create README if doesn't exist
        self._ensure_readme()

    def _ensure_readme(self):
        """Create README.md in models directory."""
        readme_path = os.path.join(self.models_dir, "README.md")
        if not os.path.exists(readme_path):
            readme_content = """# Model Registry

This directory contains versioned ML models for the trading system.

## Naming Convention

```
{model_name}_v{version}_{YYYY-MM-DD}.pkl  - Model file
{model_name}_v{version}_{YYYY-MM-DD}.json - Metadata
```

## Available Models

### bias_engine
XGBoost classifier for directional bias prediction.
- Input: 43 engineered features
- Output: Long/Short probability

### risk_model
Position sizing and risk calculation model.

### game_model
Game-theoretic execution model.

## Usage

```python
from infrastructure.model_registry import ModelRegistry

registry = ModelRegistry()

# Load latest model
model, metadata = registry.load_model('bias_engine')

# Load specific version
model, metadata = registry.load_model('bias_engine', version=3)
```

## Version History

| Version | Date | Sharpe | Win Rate | Notes |
|---------|------|--------|----------|-------|
"""
            with open(readme_path, "w") as f:
                f.write(readme_content)

    def _get_next_version(self, model_name: str) -> int:
        """Get next version number for a model."""
        existing = self.list_models(model_name)
        if not existing:
            return 1
        return max(m.version for m in existing) + 1

    def _get_model_path(self, model_name: str, version: int, date_str: str) -> str:
        """Get file path for model."""
        filename = f"{model_name}_v{version}_{date_str}"
        return os.path.join(self.models_dir, filename)

    def save_model(
        self,
        model: Any,
        name: str,
        metrics: Dict[str, float],
        features: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_data_range: Optional[tuple] = None,
        git_commit: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> ModelMetadata:
        """Save a model with metadata.

        Args:
            model: The trained model object
            name: Model name (e.g., 'bias_engine')
            metrics: Dict with 'sharpe', 'win_rate', 'max_drawdown', etc.
            features: List of feature names used
            hyperparameters: Training hyperparameters
            training_data_range: (start_date, end_date) tuple
            git_commit: Git commit hash
            notes: Additional notes

        Returns:
            ModelMetadata object
        """
        version = self._get_next_version(name)
        date_str = datetime.now().strftime("%Y-%m-%d")

        base_path = self._get_model_path(name, version, date_str)
        model_path = f"{base_path}.pkl"
        meta_path = f"{base_path}.json"

        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            training_date=date_str,
            sharpe_ratio=metrics.get("sharpe", 0.0),
            win_rate=metrics.get("win_rate", 0.0),
            max_drawdown=metrics.get("max_drawdown", 0.0),
            total_trades=int(metrics.get("total_trades", 0)),
            avg_trade_pnl=metrics.get("avg_trade_pnl", 0.0),
            features_used=features or [],
            hyperparameters=hyperparameters or {},
            training_data_start=training_data_range[0] if training_data_range else "",
            training_data_end=training_data_range[1] if training_data_range else "",
            git_commit=git_commit,
            notes=notes,
        )

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metadata
        with open(meta_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(
            f"Model saved: {name} v{version} " f"(Sharpe: {metadata.sharpe_ratio:.2f}, " f"Win: {metadata.win_rate:.1%})"
        )

        # Update README
        self._update_readme(metadata)

        return metadata

    def load_model(self, name: str, version: Optional[int] = None, date: Optional[str] = None) -> tuple[Any, ModelMetadata]:
        """Load a model by name and optional version/date.

        Args:
            name: Model name
            version: Specific version (loads latest if None)
            date: Specific date (loads latest if None)

        Returns:
            (model_object, metadata)
        """
        if version is None and date is None:
            # Load latest version
            models = self.list_models(name)
            if not models:
                raise ValueError(f"No models found for {name}")
            metadata = max(models, key=lambda m: m.version)
        else:
            # Find specific model
            models = self.list_models(name)
            for m in models:
                if version and m.version != version:
                    continue
                if date and m.training_date != date:
                    continue
                metadata = m
                break
            else:
                raise ValueError(f"Model {name} v{version} not found")

        base_path = self._get_model_path(name, metadata.version, metadata.training_date)
        model_path = f"{base_path}.pkl"

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Model loaded: {name} v{metadata.version}")
        return model, metadata

    def list_models(self, name: Optional[str] = None) -> List[ModelMetadata]:
        """List all models or models matching a name.

        Args:
            name: Filter by model name (optional)

        Returns:
            List of ModelMetadata objects
        """
        models = []

        for filename in os.listdir(self.models_dir):
            if not filename.endswith(".json"):
                continue

            # Parse filename: {name}_v{version}_{date}.json
            try:
                parts = filename.replace(".json", "").split("_")
                if len(parts) < 3:
                    continue

                model_name = "_".join(parts[:-2])  # Handle names with underscores

                if name and model_name != name:
                    continue

                # Load metadata
                meta_path = os.path.join(self.models_dir, filename)
                with open(meta_path, "r") as f:
                    data = json.load(f)

                models.append(ModelMetadata.from_dict(data))

            except Exception as e:
                logger.warning(f"Failed to parse model file {filename}: {e}")
                continue

        return sorted(models, key=lambda m: (m.name, m.version))

    def get_model_history(self, name: str) -> List[ModelMetadata]:
        """Get version history for a model."""
        return self.list_models(name)

    def delete_model(self, name: str, version: int) -> bool:
        """Delete a specific model version.

        Returns:
            True if deleted, False if not found
        """
        models = self.list_models(name)
        for m in models:
            if m.version == version:
                base_path = self._get_model_path(name, version, m.training_date)

                # Delete files
                for ext in [".pkl", ".json"]:
                    path = f"{base_path}{ext}"
                    if os.path.exists(path):
                        os.remove(path)

                logger.info(f"Deleted model: {name} v{version}")
                return True

        return False

    def _update_readme(self, metadata: ModelMetadata):
        """Add model entry to README."""
        readme_path = os.path.join(self.models_dir, "README.md")

        # Create new entry line
        entry = (
            f"| {metadata.name}_v{metadata.version} | "
            f"{metadata.training_date} | "
            f"{metadata.sharpe_ratio:.2f} | "
            f"{metadata.win_rate:.1%} | "
            f"{metadata.notes or 'Auto-trained'} |\n"
        )

        # Append to README
        with open(readme_path, "a") as f:
            f.write(entry)

    def compare_models(self, name: str) -> Dict[str, Any]:
        """Compare all versions of a model.

        Returns:
            Dict with comparison data
        """
        models = self.list_models(name)

        if len(models) < 2:
            return {"error": "Need at least 2 versions to compare"}

        comparison = {
            "model_name": name,
            "versions": [],
            "best_sharpe": None,
            "best_win_rate": None,
            "most_recent": None,
        }

        best_sharpe = max(models, key=lambda m: m.sharpe_ratio)
        best_win_rate = max(models, key=lambda m: m.win_rate)
        most_recent = max(models, key=lambda m: datetime.strptime(m.training_date, "%Y-%m-%d"))

        comparison["best_sharpe"] = f"v{best_sharpe.version} ({best_sharpe.sharpe_ratio:.2f})"
        comparison["best_win_rate"] = f"v{best_win_rate.version} ({best_win_rate.win_rate:.1%})"
        comparison["most_recent"] = f"v{most_recent.version} ({most_recent.training_date})"

        for m in models:
            comparison["versions"].append(
                {
                    "version": m.version,
                    "date": m.training_date,
                    "sharpe": m.sharpe_ratio,
                    "win_rate": m.win_rate,
                    "drawdown": m.max_drawdown,
                }
            )

        return comparison
