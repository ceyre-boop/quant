"""
Walk-Forward Validation with Chi-Squared Analysis

Validates XGBoost model predictions using chi-squared test on confidence buckets.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
import sys

sys.path.insert(0, str(Path(__file__).parent))

from training.feature_generator import FeatureGenerator
from data.alpaca_client import AlpacaDataClient


def chi_squared_test(predictions: np.ndarray, actuals: np.ndarray, confidence_buckets: int = 5) -> Dict:
    """
    Perform chi-squared test on prediction confidence buckets.

    H0: Model predictions are independent of actual outcomes (no skill)
    H1: Model predictions are related to actual outcomes (has skill)

    Args:
        predictions: Predicted probabilities (0-1)
        actuals: Actual outcomes (0 or 1)
        confidence_buckets: Number of confidence buckets

    Returns:
        Dict with chi2 statistic, p-value, and bucket analysis
    """
    # Create confidence buckets
    bucket_edges = np.linspace(0, 1, confidence_buckets + 1)
    bucket_labels = [f"{(bucket_edges[i]*100):.0f}-{(bucket_edges[i+1]*100):.0f}%" for i in range(confidence_buckets)]

    # Assign each prediction to bucket
    bucket_indices = np.digitize(predictions, bucket_edges[1:-1])

    # Calculate observed vs expected
    results = []
    for i in range(confidence_buckets):
        mask = bucket_indices == i
        if mask.sum() == 0:
            continue

        bucket_preds = predictions[mask]
        bucket_actuals = actuals[mask]

        # Observed: how many actually went up
        obs_up = bucket_actuals.sum()
        obs_down = len(bucket_actuals) - obs_up

        # Expected: based on average predicted probability
        avg_prob = bucket_preds.mean()
        exp_up = len(bucket_actuals) * avg_prob
        exp_down = len(bucket_actuals) * (1 - avg_prob)

        # Chi-squared contribution
        chi2_up = ((obs_up - exp_up) ** 2) / max(exp_up, 0.001)
        chi2_down = ((obs_down - exp_down) ** 2) / max(exp_down, 0.001)

        # Win rate
        win_rate = obs_up / len(bucket_actuals) if len(bucket_actuals) > 0 else 0

        results.append(
            {
                "bucket": bucket_labels[i],
                "count": len(bucket_actuals),
                "avg_confidence": avg_prob,
                "actual_up_rate": win_rate,
                "obs_up": obs_up,
                "obs_down": obs_down,
                "exp_up": exp_up,
                "exp_down": exp_down,
                "chi2_contrib": chi2_up + chi2_down,
            }
        )

    # Total chi-squared
    total_chi2 = sum(r["chi2_contrib"] for r in results)

    # Degrees of freedom = (buckets - 1) * (outcomes - 1) = (5-1)*(2-1) = 4
    df = (confidence_buckets - 1) * (2 - 1)

    # P-value
    p_value = 1 - stats.chi2.cdf(total_chi2, df)

    return {
        "chi2_statistic": total_chi2,
        "p_value": p_value,
        "degrees_of_freedom": df,
        "significant": p_value < 0.05,
        "buckets": results,
    }


def walk_forward_validation(
    symbol: str = "SPY",
    timeframe: str = "1D",
    train_days: int = 252,
    test_days: int = 63,
    step_days: int = 21,
    years: int = 5,
) -> Dict:
    """
    Walk-forward validation: Train on past, test on future, step forward.

    Args:
        symbol: Asset to test
        timeframe: '1D' or '1H'
        train_days: Training window size
        test_days: Testing window size
        step_days: How much to step forward each iteration
        years: Total years of data to use

    Returns:
        Dict with performance metrics and chi-squared results
    """
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, log_loss

    print("=" * 70)
    print("WALK-FORWARD VALIDATION")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Train: {train_days} days | Test: {test_days} days | Step: {step_days} days")
    print()

    # Fetch data
    client = AlpacaDataClient()
    df = client.get_historical_bars(symbol, timeframe=timeframe, days=years * 365)

    if len(df) < train_days + test_days:
        print(f"ERROR: Insufficient data ({len(df)} bars)")
        return None

    # Generate features
    gen = FeatureGenerator()
    features_df = gen.generate_features(df)

    if features_df.empty:
        print("ERROR: Feature generation failed")
        return None

    # Define feature columns (exclude OHLCV and target)
    exclude_cols = ["open", "high", "low", "close", "volume", "target"]
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]

    # Create target: did price go up next day?
    features_df["target"] = (features_df["close"].shift(-1) > features_df["close"]).astype(int)

    # Drop NaN
    features_df = features_df.dropna()

    print(f"Total samples: {len(features_df)}")
    print(f"Features: {len(feature_cols)}")
    print()

    # Walk forward
    results = []
    all_predictions = []
    all_actuals = []
    all_confidences = []

    start_idx = 0
    fold = 0

    while start_idx + train_days + test_days <= len(features_df):
        fold += 1

        # Define windows
        train_start = start_idx
        train_end = start_idx + train_days
        test_start = train_end
        test_end = min(test_start + test_days, len(features_df))

        # Split data
        train_df = features_df.iloc[train_start:train_end]
        test_df = features_df.iloc[test_start:test_end]

        if len(test_df) < 5:
            break

        print(f"--- Fold {fold} ---")
        print(f"  Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} samples)")
        print(f"  Test:  {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} samples)")

        # Prepare matrices
        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df["target"].values
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df["target"].values

        # Train model
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )

        model.fit(X_train, y_train, verbose=False)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of UP

        # Store for aggregate analysis
        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)
        all_confidences.extend(y_prob)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, model.predict_proba(X_test))

        print(f"  Accuracy: {acc:.3f} | Log Loss: {ll:.3f}")

        results.append(
            {
                "fold": fold,
                "train_start": train_df.index[0],
                "train_end": train_df.index[-1],
                "test_start": test_df.index[0],
                "test_end": test_df.index[-1],
                "accuracy": acc,
                "log_loss": ll,
            }
        )

        # Step forward
        start_idx += step_days

    print()
    print("=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    all_confidences = np.array(all_confidences)

    # Overall accuracy
    overall_acc = accuracy_score(all_actuals, all_predictions)
    print(f"Overall Accuracy: {overall_acc:.3f}")
    print(f"Total Predictions: {len(all_predictions)}")

    # Baseline (always predict majority class)
    baseline = max(np.mean(all_actuals), 1 - np.mean(all_actuals))
    print(f"Baseline (majority): {baseline:.3f}")
    print(f"Edge over baseline: {overall_acc - baseline:.3f}")
    print()

    # Chi-squared test
    print("=" * 70)
    print("CHI-SQUARED VALIDATION")
    print("=" * 70)

    chi2_result = chi_squared_test(all_confidences, all_actuals, confidence_buckets=5)

    print(f"Chi-Squared Statistic: {chi2_result['chi2_statistic']:.4f}")
    print(f"Degrees of Freedom: {chi2_result['degrees_of_freedom']}")
    print(f"P-Value: {chi2_result['p_value']:.6f}")
    print(f"Significant (p < 0.05): {chi2_result['significant']}")
    print()

    if chi2_result["significant"]:
        print("✓ Model shows STATISTICALLY SIGNIFICANT predictive power")
    else:
        print("✗ Model does NOT show significant predictive power")
    print()

    # Bucket breakdown
    print("Confidence Bucket Analysis:")
    print("-" * 70)
    print(f"{'Bucket':<15} {'Count':>8} {'Avg Conf':>10} {'Win Rate':>10} {'Chi2':>10}")
    print("-" * 70)

    for b in chi2_result["buckets"]:
        print(
            f"{b['bucket']:<15} {b['count']:>8} {b['avg_confidence']:>10.2f} "
            f"{b['actual_up_rate']:>10.2f} {b['chi2_contrib']:>10.4f}"
        )

    return {
        "symbol": symbol,
        "overall_accuracy": overall_acc,
        "baseline": baseline,
        "edge": overall_acc - baseline,
        "chi_squared": chi2_result,
        "fold_results": results,
        "predictions": all_predictions,
        "actuals": all_actuals,
        "confidences": all_confidences,
    }


if __name__ == "__main__":
    # Run walk-forward validation on SPY
    results = walk_forward_validation(
        symbol="SPY",
        timeframe="1D",
        train_days=252,  # 1 year training
        test_days=63,  # 3 month testing
        step_days=21,  # Monthly step
        years=5,  # 5 years total
    )
