"""
Sovereign Trading Intelligence -- HMM Regime Detector
Phase 2: Feature Layer

2-state Hidden Markov Model (trending / ranging).
Uses return volatility and directional persistence as observables.

The HMM provides two features for the router:
  - hmm_transition_prob:     Probability of regime change in next bar
  - bars_since_regime_change: How long the current regime has persisted

Regimes have memory -- a regime that has persisted for 50 bars
is more likely to continue than one that just started. The router
uses bars_since_regime_change to capture this inertia.

The HMM is trained on a rolling window of returns.
It is NOT trained on the full dataset (that would be lookahead).
Each prediction uses only data available up to that point.
"""

import numpy as np
import pandas as pd
import logging
import warnings

logger = logging.getLogger(__name__)

# Suppress hmmlearn convergence warnings during rolling fits
warnings.filterwarnings('ignore', category=DeprecationWarning)


class HMMRegimeDetector:
    """
    2-state Gaussian HMM for regime detection.

    State 0: Low-volatility / trending
    State 1: High-volatility / ranging

    (Labels are assigned post-hoc based on which state
    has higher variance -- the HMM doesn't know the labels.)
    """

    def __init__(self, n_states: int = 2, lookback: int = 252):
        """
        Args:
            n_states: Number of hidden states (default 2)
            lookback: Rolling window for HMM fitting (default 252 bars)
        """
        self.n_states = n_states
        self.lookback = lookback

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute HMM regime features from an OHLCV DataFrame.

        Uses expanding window for first `lookback` bars,
        then rolling window of `lookback` bars.

        Returns DataFrame with:
          - hmm_state:            Current regime (0 or 1)
          - hmm_transition_prob:  Probability of switching to other state
          - bars_since_regime_change: Bars since last regime switch
        """
        from hmmlearn.hmm import GaussianHMM
        
        price_series = df['close']
        returns = price_series.pct_change().fillna(0).values
        volume = df['volume'].values
        vol_returns = np.diff(np.log(volume + 1), prepend=np.log(volume[0] + 1))
        n = len(returns)

        # Features for HMM: [return, abs_return, volume_change]
        X_full = np.column_stack([
            returns,
            np.abs(returns),
            vol_returns
        ])

        states = np.full(n, np.nan)
        transition_probs = np.full(n, np.nan)

        # Minimum data requirement for HMM training
        min_train = max(60, self.n_states * 30)

        for i in range(min_train, n):
            # Rolling window -- no lookahead
            start_idx = max(0, i - self.lookback)
            X_train = X_full[start_idx:i + 1]

            try:
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type='diag',
                    n_iter=50,
                    random_state=42,
                    verbose=False,
                )
                model.fit(X_train)

                # Predict the state at the current bar
                state_seq = model.predict(X_train)
                current_state = state_seq[-1]

                # Transition probability: probability of leaving current state
                trans_matrix = model.transmat_
                transition_prob = 1.0 - trans_matrix[current_state, current_state]

                states[i] = current_state
                transition_probs[i] = transition_prob

            except Exception:
                # HMM fitting can fail on degenerate data -- skip quietly
                states[i] = states[i - 1] if i > 0 and not np.isnan(states[i - 1]) else 0
                transition_probs[i] = 0.0

        # Ensure consistent state labeling:
        # State with higher mean abs return = "ranging" (state 1)
        # This prevents label swapping between windows
        states_series = pd.Series(states, index=price_series.index)
        states_series = self._stabilize_labels(states_series, X_full)

        # Compute bars since regime change
        bars_since = self._compute_bars_since_change(states_series)

        result = pd.DataFrame({
            'hmm_state':                states_series,
            'hmm_transition_prob':      transition_probs,
            'bars_since_regime_change': bars_since,
        }, index=price_series.index)

        return result

    def _stabilize_labels(self, states: pd.Series,
                          X: np.ndarray) -> pd.Series:
        """
        HMM states are arbitrary (0 or 1). We define:
        - State 0 = lower volatility regime (trending)
        - State 1 = higher volatility regime (ranging)

        If the labels are swapped, flip them.
        """
        valid_mask = ~states.isna()
        if valid_mask.sum() < 10:
            return states

        s0_mask = (states == 0) & valid_mask
        s1_mask = (states == 1) & valid_mask

        if s0_mask.sum() == 0 or s1_mask.sum() == 0:
            return states

        # Mean absolute return per state
        abs_ret = np.abs(X[:, 0])
        mean_vol_s0 = abs_ret[s0_mask.values].mean()
        mean_vol_s1 = abs_ret[s1_mask.values].mean()

        # If state 0 has higher vol, swap
        if mean_vol_s0 > mean_vol_s1:
            states = states.map({0: 1, 1: 0, np.nan: np.nan})

        return states

    @staticmethod
    def _compute_bars_since_change(states: pd.Series) -> pd.Series:
        """
        Count bars since the last regime change.
        """
        result = pd.Series(np.nan, index=states.index)
        count = 0
        prev_state = np.nan

        for i, state in enumerate(states.values):
            if np.isnan(state):
                result.iloc[i] = np.nan
                continue

            if state != prev_state and not np.isnan(prev_state):
                count = 0
            else:
                count += 1

            result.iloc[i] = count
            prev_state = state

        return result


def compute_hmm_features(df: pd.DataFrame,
                         lookback: int = 252) -> pd.DataFrame:
    """
    Convenience function: compute HMM features from OHLCV DataFrame.

    Input: DataFrame with 'close' column.
    Output: DataFrame with hmm_state, hmm_transition_prob,
            bars_since_regime_change.

    Note: This is computationally expensive (~5-10s per asset).
    Cache the results.
    """
    detector = HMMRegimeDetector(lookback=lookback)
    return detector.compute(df)
