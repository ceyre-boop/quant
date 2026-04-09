"""Quick test of training pipeline"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from training.feature_generator import FeatureGenerator
from training.train_xgb import train_model

# Quick test with small dataset
print("Testing feature generation...")
gen = FeatureGenerator()
df, features = gen.build_training_dataset(timeframe="1D", days=90)
print(f"\nGenerated {len(df)} samples with {len(features)} features")
print(f"Features: {features[:10]}...")
print(f"\nTarget distribution:")
print(df["target"].value_counts())
