"""
SHAP Analysis for XGBoost Model
Interpret feature importance and model behavior
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

# Load env vars first
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.feature_generator import FeatureGenerator


def analyze_model():
    """Run SHAP analysis on trained model"""
    print("=" * 70)
    print("SHAP ANALYSIS")
    print("=" * 70)

    # Load model
    model_path = Path(__file__).parent / "xgb_model.pkl"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run train_xgb.py first")
        return

    with open(model_path, "rb") as f:
        data = pickle.load(f)
        model = data["model"]
        feature_cols = data["features"]

    print(f"Loaded model with {len(feature_cols)} features")

    # Generate fresh data for SHAP
    print("\nGenerating validation data...")
    gen = FeatureGenerator()
    df, _ = gen.build_training_dataset(timeframe="1D", days=180)

    X = df[feature_cols].values

    # Try to import SHAP
    try:
        import shap
    except ImportError:
        print("\nInstalling SHAP...")
        import subprocess

        subprocess.run([sys.executable, "-m", "pip", "install", "shap", "-q"])
        import shap

    # Calculate SHAP values
    print("\nCalculating SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(model)

    # Use a sample for speed
    sample_size = min(1000, len(X))
    X_sample = X[:sample_size]

    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    print("\nGenerating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)

    plot_path = Path(__file__).parent / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ SHAP plot saved to {plot_path}")

    # Feature importance from SHAP
    if isinstance(shap_values, list):
        # Binary classification - use class 1
        shap_values = shap_values[1]

    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame(
        {"feature": feature_cols, "mean_shap": mean_shap}
    ).sort_values("mean_shap", ascending=False)

    print("\nTop 10 Features by SHAP Importance:")
    print(shap_importance.head(10).to_string(index=False))

    # Save importance
    importance_path = Path(__file__).parent / "shap_importance.csv"
    shap_importance.to_csv(importance_path, index=False)
    print(f"\n✅ SHAP importance saved to {importance_path}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    top_features = shap_importance.head(5)["feature"].tolist()

    noise_features = ["day_of_week", "month", "random", "id"]
    real_features = [
        "rsi",
        "macd",
        "atr",
        "volume",
        "return",
        "spy",
        "vixy",
        "bb_",
        "sma_",
        "dist_",
    ]

    noise_count = sum(
        1 for f in top_features if any(nf in f.lower() for nf in noise_features)
    )
    real_count = sum(
        1 for f in top_features if any(rf in f.lower() for rf in real_features)
    )

    if real_count >= 3:
        print("✅ MODEL LEARNED REAL PATTERNS")
        print(f"   Top features include: {', '.join(top_features[:3])}")
        print("   The model is using price action, momentum, and market context")
    elif noise_count >= 2:
        print("⚠️  WARNING: Model may be learning noise")
        print(f"   Top features: {', '.join(top_features[:3])}")
        print("   Consider feature engineering or more data")
    else:
        print("ℹ️  REVIEW: Mixed signals in feature importance")
        print(f"   Top features: {', '.join(top_features[:5])}")

    return shap_importance


if __name__ == "__main__":
    analyze_model()
