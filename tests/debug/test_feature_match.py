# Test feature count
import pickle
from pathlib import Path
from training.feature_generator import FeatureGenerator
import pandas as pd
import numpy as np

# Load model features
with open("training/xgb_model.pkl", "rb") as f:
    saved = pickle.load(f)
    model_features = saved["features"]
print(f"Model expects {len(model_features)} features")

# Create sample data
df = pd.DataFrame(
    {
        "open": np.random.randn(100).cumsum() + 100,
        "high": np.random.randn(100).cumsum() + 102,
        "low": np.random.randn(100).cumsum() + 98,
        "close": np.random.randn(100).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 100),
    }
)

# With cross-asset data
market_data = {"SPY": df, "VIXY": df, "SQQQ": df, "SPXU": df, "GLD": df, "TLT": df}
gen = FeatureGenerator()
result = gen.generate_features(df, market_data)
feature_cols = [c for c in result.columns if c not in ["open", "high", "low", "close", "volume"]]
print(f"Generator produces {len(feature_cols)} features with cross-asset data")

# Check match
model_set = set(model_features)
gen_set = set(feature_cols)
print(f"\nMissing from generator: {model_set - gen_set}")
print(f"Extra in generator: {gen_set - model_set}")
