"""
XGBoost Training Pipeline
Train model to predict next-day direction using TimeSeriesSplit
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

# Load env vars before importing modules that need them
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.feature_generator import FeatureGenerator


def train_model(timeframe: str = "1D", days: int = 365, n_splits: int = 5):
    """
    Train XGBoost classifier with walk-forward validation

    Args:
        timeframe: '1D' or '1H'
        days: Number of days of history
        n_splits: Number of time series splits
    """
    print("=" * 70)
    print("XGBOOST TRAINING PIPELINE")
    print("=" * 70)

    # Generate features
    print("\n[1/4] Generating features...")
    gen = FeatureGenerator()
    df, feature_cols = gen.build_training_dataset(timeframe=timeframe, days=days)

    if len(df) < 100:
        print("ERROR: Insufficient data for training")
        return None

    # Prepare data
    X = df[feature_cols].values
    y = df["target"].values

    print(f"\n[2/4] Training with TimeSeriesSplit ({n_splits} folds)...")
    print(f"Samples: {len(X)}, Features: {len(feature_cols)}")

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        print(f"Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Evaluate
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        fold_scores.append(acc)

        print(f"Fold accuracy: {acc:.3f}")

    print(f"\n[3/4] Cross-validation complete")
    print(f"Mean accuracy: {np.mean(fold_scores):.3f} (+/- {np.std(fold_scores):.3f})")

    # Train final model on all data
    print("\n[4/4] Training final model on all data...")
    final_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    final_model.fit(X, y)

    # Feature importance
    importance = pd.DataFrame(
        {"feature": feature_cols, "importance": final_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\nTop 10 Feature Importances:")
    print(importance.head(10).to_string(index=False))

    # Save model
    model_path = Path(__file__).parent / "xgb_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {"model": final_model, "features": feature_cols, "importance": importance},
            f,
        )

    print(f"\n[OK] Model saved to {model_path}")

    return final_model, feature_cols, importance


if __name__ == "__main__":
    model, features, importance = train_model(timeframe="1D", days=365, n_splits=5)
