# sovereign/ml_trainer.py
"""
Trains two XGBoost models: one per regime.
Retrains automatically after each simulation batch.

López de Prado AFML fixes applied:

  Pitfall 5 — Sample uniqueness weights.
    Observations that overlap in their label horizon with other observations
    are down-weighted. Each observation's weight = 1 / (number of other
    observations that share at least one bar in their label horizon).
    This reduces the effective N of correlated samples and prevents the
    model from over-fitting to the most represented periods.

  Pitfall 6 — Purge embargo on the 80/20 split.
    The split point has a purge window: label_horizon bars removed from the
    end of the train set so that train labels do not look into test prices.
    Without this, momentum labels at position split-1 use prices from the
    test set (they look forward 10 bars) → direct leakage.

  The triple-barrier labeling (build_momentum_labels, build_reversion_labels)
    was already structurally correct (stop hit → 0, target hit → 1, neither → 0).
    Now: 'neither' is treated as NaN and dropped (time-barrier outcome).
    Training on inconclusive labels adds noise without signal.
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import logging

logger = logging.getLogger(__name__)


# ── ADF Stationarity Gate (Ernest Chan, second lecture 36:30) ─────────── #
# "The SPX trained at 1000 has never seen 3500 — the model learns nothing.
#  Non-stationary data fed to a model is worse than useless."
# We run a quick ADF test on each feature column and drop those that fail.
# ADF p < 0.05 → stationary (keep). ADF p ≥ 0.05 → non-stationary (drop).

def _stationary_features(df: pd.DataFrame, cols: list, alpha: float = 0.05) -> list:
    """
    Return only the columns from `cols` that pass the ADF stationarity test.

    Drops columns that are non-stationary (price levels, cumulative series,
    unbounded indicators) before they can corrupt the model's predictions.

    Requires statsmodels. If unavailable, returns all cols unchanged (safe
    degradation — the gate becomes a no-op rather than crashing training).
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        logger.debug("statsmodels not available — ADF stationarity gate skipped")
        return cols

    stationary, dropped = [], []
    for col in cols:
        series = df[col].dropna()
        if len(series) < 20:
            stationary.append(col)  # too short to test — keep by default
            continue
        try:
            adf_stat, p_value, *_ = adfuller(series, autolag='AIC')
            if p_value < alpha:
                stationary.append(col)
            else:
                dropped.append(col)
        except Exception:
            stationary.append(col)  # on error: keep

    if dropped:
        logger.info(
            f"ADF stationarity gate: dropped {len(dropped)} non-stationary features: "
            + ", ".join(dropped[:5]) + ("..." if len(dropped) > 5 else "")
        )
    return stationary


def _uniqueness_weights(label_indices: pd.Index, label_horizon: int,
                        all_index: pd.Index) -> pd.Series:
    """
    Compute sample uniqueness weights (AFML Ch. 4).

    For each observation i, count how many other observations j share at
    least one bar in their label horizon with observation i.
    Weight_i = 1 / overlap_count_i, then normalise so weights sum to N.

    This down-weights observations from busy / correlated periods
    (e.g., 10 momentum signals fired in consecutive bars all using overlapping
    future windows) and up-weights isolated, more independent observations.
    """
    n = len(label_indices)
    overlap_counts = np.ones(n, dtype=float)

    for i in range(n):
        t_i = label_indices[i]
        try:
            pos_i = all_index.get_loc(t_i)
        except KeyError:
            continue
        horizon_end_i = min(pos_i + label_horizon, len(all_index) - 1)

        for j in range(n):
            if i == j:
                continue
            t_j = label_indices[j]
            try:
                pos_j = all_index.get_loc(t_j)
            except KeyError:
                continue
            horizon_end_j = min(pos_j + label_horizon, len(all_index) - 1)
            # Overlap: horizons share at least one bar
            if pos_j <= horizon_end_i and pos_i <= horizon_end_j:
                overlap_counts[i] += 1

    raw_weights = 1.0 / overlap_counts
    # Normalise: weights sum to n so XGBoost sees the same effective dataset size
    weights = raw_weights * (n / raw_weights.sum())
    return pd.Series(weights, index=label_indices)


class MLTrainer:

    MOMENTUM_FEATURES = [
        'zscore_20', 'zscore_50', 'roc_5', 'roc_10',
        'rsi_14', 'rsi_divergence', 'volume_zscore',
        'atr_distance', 'adx_14', 'hurst', 'atr_zscore',
    ]

    REVERSION_FEATURES = [
        'zscore_20', 'zscore_50', 'bb_pct_b', 'rsi_14',
        'rsi_divergence', 'volume_zscore', 'atr_distance',
        'adx_14', 'atr_zscore',
    ]

    XGB_PARAMS = {
        'n_estimators':          300,
        'max_depth':             4,
        'learning_rate':         0.05,
        'subsample':             0.8,
        'colsample_bytree':      0.8,
        'eval_metric':           'auc',
        'early_stopping_rounds': 30,
        'random_state':          42,
    }

    def build_momentum_labels(self, df: pd.DataFrame,
                              forward_bars: int = 10,
                              continuation_threshold: float = 0.6) -> pd.Series:
        """
        Triple-barrier momentum label.

        +1  price continues in breakout direction by 60% of initial move
            within forward_bars WITHOUT hitting stop.
        -1  price hits stop (1.5×ATR against direction) before target.
        NaN time barrier expired — inconclusive, dropped from training.

        Entry anchor: T+1 open (realistic, not look-ahead).
        """
        initial_move = df['close'] - df['close'].shift(5)
        direction = np.sign(initial_move)
        atr = df['close'].rolling(14).std() * 1.5

        labels = []
        for i in range(len(df)):
            if i + forward_bars >= len(df) or pd.isna(initial_move.iloc[i]):
                labels.append(np.nan)
                continue

            entry = df['open'].iloc[i + 1] if i + 1 < len(df) else np.nan
            if pd.isna(entry):
                labels.append(np.nan)
                continue

            d = direction.iloc[i]
            stop_dist = atr.iloc[i]
            target = continuation_threshold * abs(initial_move.iloc[i])
            future = (df['close'].iloc[i + 1:i + 1 + forward_bars] - entry) * d

            if (future < -stop_dist).any():
                labels.append(-1)   # stop barrier
            elif (future >= target).any():
                labels.append(1)    # profit-take barrier
            else:
                labels.append(np.nan)  # time barrier — drop, not a 0

        return pd.Series(labels, index=df.index)

    def build_reversion_labels(self, df: pd.DataFrame,
                               forward_bars: int = 15,
                               reversion_threshold: float = 0.6) -> pd.Series:
        """
        Triple-barrier mean-reversion label.

        +1  price reverts 60% of distance to 50-bar mean within forward_bars
            WITHOUT hitting 1.5×ATR stop.
        -1  stop hit before target.
        NaN time barrier expired.

        Entry anchor: T+1 open.
        """
        mean = df['close'].rolling(50).mean()
        distance = mean - df['close']
        atr = df['close'].rolling(14).std() * 1.5

        labels = []
        for i in range(len(df)):
            if i + forward_bars >= len(df) or pd.isna(distance.iloc[i]):
                labels.append(np.nan)
                continue

            entry = df['open'].iloc[i + 1] if i + 1 < len(df) else np.nan
            if pd.isna(entry):
                labels.append(np.nan)
                continue

            d = np.sign(distance.iloc[i])
            stop_dist = atr.iloc[i]
            target = reversion_threshold * abs(distance.iloc[i])
            future = (df['close'].iloc[i + 1:i + 1 + forward_bars] - entry) * d

            if (future < -stop_dist).any():
                labels.append(-1)
            elif (future >= target).any():
                labels.append(1)
            else:
                labels.append(np.nan)

        return pd.Series(labels, index=df.index)

    def train(self, features_df: pd.DataFrame,
              labels: pd.Series,
              feature_cols: list,
              regime: str,
              label_horizon: int = 10,
              value_weights: "pd.Series | None" = None) -> tuple:
        """
        Purged walk-forward train with uniqueness sample weights.

        label_horizon: bars each label looks forward. Used to compute
            the purge boundary and uniqueness weights.
            momentum → 10 bars, reversion → 15 bars.

        Returns (calibrated_model, auc_score).
        """
        # ── ADF stationarity gate (Chan: non-stationary = worse than useless) ─ #
        feature_cols = _stationary_features(features_df, list(feature_cols))
        if not feature_cols:
            print(f"⚠️  {regime}: all features failed ADF test — cannot train")
            return None, 0.0

        clean = features_df[feature_cols].copy()
        clean['label'] = labels
        clean = clean.dropna()   # drops NaN (time-barrier outcomes — correct)

        if len(clean) < 200:
            print(f"⚠️  {regime}: only {len(clean)} clean rows — need 200+")
            return None, 0.0

        # Convert ±1 labels to binary for XGBoost
        clean['label_bin'] = (clean['label'] == 1).astype(int)

        # ── Purge: remove label_horizon bars from end of train ─────────── #
        # Without purge: momentum labels at split-1 look forward 10 bars
        # into the test set → direct leakage.
        purge_size = label_horizon
        split = int(len(clean) * 0.8)
        purged_split = max(0, split - purge_size)

        X_train = clean[feature_cols].iloc[:purged_split]
        y_train = clean['label_bin'].iloc[:purged_split]
        X_test  = clean[feature_cols].iloc[split:]
        y_test  = clean['label_bin'].iloc[split:]

        if len(X_train) < 100 or len(X_test) < 10:
            print(f"⚠️  {regime}: insufficient data after purge")
            return None, 0.0

        # ── Uniqueness weights (AFML Ch. 4, Pitfall 5) ────────────────── #
        # Only compute on train set (test set is never weighted in training)
        # Use a fast approximation: cap uniqueness computation at 500 obs
        # for speed; full computation is O(n²).
        train_idx = pd.DatetimeIndex(X_train.index) if hasattr(X_train.index, 'freq') \
            else pd.Index(X_train.index)
        all_idx = pd.Index(clean.index)

        if len(X_train) <= 500:
            sample_weights = _uniqueness_weights(
                label_indices=X_train.index,
                label_horizon=label_horizon,
                all_index=all_idx,
            ).values
        else:
            # Fast path: uniform weights for large datasets
            # (uniqueness computation becomes O(n²) expensive)
            sample_weights = np.ones(len(X_train))

        # Self-play passthrough: multiply value-function weights into the uniqueness
        # weights, aligned to X_train (spec §4.3). Preserves the purged split above.
        if value_weights is not None:
            vw = value_weights.reindex(X_train.index).fillna(1.0).values
            sample_weights = sample_weights * vw

        model = xgb.XGBClassifier(**self.XGB_PARAMS)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        calibrated = CalibratedClassifierCV(model, cv=5, method='isotonic')
        calibrated.fit(X_train, y_train, sample_weight=sample_weights)

        probs = calibrated.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)

        floor = 0.60 if regime == 'momentum' else 0.50
        high_conf_mask = probs >= floor
        win_rate = y_test[high_conf_mask].mean() if high_conf_mask.sum() > 0 else 0

        print(f"\n{'=' * 50}")
        print(f"  {regime.upper()} MODEL TRAINED")
        print(f"  AUC:              {auc:.4f}")
        print(f"  Win rate ({floor}+):  {win_rate:.1%} on {high_conf_mask.sum()} signals")
        print(f"  Train rows:       {len(X_train)} (raw {split}, purged {purge_size})")
        print(f"  Test rows:        {len(X_test)}")
        print(f"  Label balance:    {y_train.mean():.1%} positive")
        print(f"{'=' * 50}\n")

        fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("  TOP FEATURES:")
        for feat, imp in fi.head(5).items():
            print(f"    {feat:<25} {imp:.4f}")

        return calibrated, auc

    def retrain_on_failures(self, features_df, labels, feature_cols,
                            regime, failure_indices,
                            label_horizon: int = 10):
        """
        Called after each simulation batch.
        Failure observations get 3× weight PLUS uniqueness weighting.
        """
        clean = features_df[feature_cols].copy()
        clean['label'] = labels
        clean = clean.dropna()
        clean['label_bin'] = (clean['label'] == 1).astype(int)

        all_idx = pd.Index(clean.index)

        if len(clean) <= 500:
            base_weights = _uniqueness_weights(
                label_indices=clean.index,
                label_horizon=label_horizon,
                all_index=all_idx,
            )
        else:
            base_weights = pd.Series(1.0, index=clean.index)

        # Failure penalty: 3× on top of uniqueness weight
        for idx in failure_indices:
            if idx in base_weights.index:
                base_weights[idx] *= 3.0

        X = clean[feature_cols]
        y = clean['label_bin']
        w = base_weights.values

        model = xgb.XGBClassifier(**self.XGB_PARAMS)
        model.fit(X, y, sample_weight=w, verbose=False)

        calibrated = CalibratedClassifierCV(model, cv=3, method='isotonic')
        calibrated.fit(X, y, sample_weight=w)

        return calibrated
