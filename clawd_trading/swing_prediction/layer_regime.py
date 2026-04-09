"""
LAYER C — Regime Classification
Hurst Exponent, ADX, Volatility — determines if mean-reversion signals are valid
"""

import logging
from datetime import datetime
from typing import Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np


@dataclass
class RegimeResult:
    """Output from regime classification."""

    symbol: str
    hurst_exponent: float
    hurst_signal: str  # 'MEAN_REVERT' | 'NEUTRAL' | 'TRENDING'
    adx_value: float
    adx_regime: str  # 'MEAN_REVERT' | 'TRANSITIONAL' | 'TRENDING' | 'BREAKOUT'
    vol_percentile: float
    hmm_state: str  # 'LOW_VOL' | 'NORMAL' | 'HIGH_VOL_CRISIS'
    hmm_confidence: float
    composite_regime: str  # 'FAVORABLE' | 'NEUTRAL' | 'UNFAVORABLE'
    supports_reversion: bool  # True = proceed, False = flag for human review
    score: float
    timestamp: str


class RegimeLayer:
    """
    Layer C: Regime Classification
    A mean-reversion signal in a trending market is a death trap.
    """

    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config["layer_c_regime"]

    def compute_hurst_exponent(self, prices: np.ndarray, window: int = 252) -> float:
        """
        Hurst Exponent via R/S analysis on rolling window.

        Algorithm:
        1. Compute log returns: r = log(p[t] / p[t-1])
        2. For each sub-period length n in [10, 20, 40, 80, 160]:
           a. Divide series into chunks of length n
           b. For each chunk: compute range R (max - min of cumulative deviations from mean)
           c. For each chunk: compute std S of returns
           d. Compute mean(R/S) for this n
        3. Regress log(R/S) on log(n) → slope = Hurst exponent H

        Interpretation:
        H < 0.4: strongly mean-reverting
        0.4–0.5: mildly mean-reverting
        0.5: random walk
        0.5–0.6: mildly trending
        H > 0.6: strongly trending

        Args:
            prices: numpy array of price data
            window: rolling window (default 252 trading days = 1 year)

        Returns:
            Hurst exponent H
        """
        if len(prices) < window:
            self.logger.warning(
                f"Insufficient data for Hurst: {len(prices)} < {window}"
            )
            return 0.5

        # Use last 'window' prices
        series = prices[-window:]

        # Compute log returns
        log_returns = np.diff(np.log(series))

        # Sub-period lengths
        lags = [10, 20, 40, 80, 160]
        rs_values = []

        for lag in lags:
            if lag >= len(log_returns):
                continue

            # Divide into chunks
            n_chunks = len(log_returns) // lag
            rs_chunks = []

            for i in range(n_chunks):
                chunk = log_returns[i * lag : (i + 1) * lag]

                # Mean of chunk
                mean_chunk = np.mean(chunk)

                # Cumulative deviations from mean
                cumsum = np.cumsum(chunk - mean_chunk)

                # Range R
                R = np.max(cumsum) - np.min(cumsum)

                # Standard deviation S
                S = np.std(chunk)

                if S > 0:
                    rs_chunks.append(R / S)

            if rs_chunks:
                rs_values.append((np.log(lag), np.log(np.mean(rs_chunks))))

        if len(rs_values) < 3:
            return 0.5

        # Linear regression of log(R/S) on log(lag)
        x = np.array([v[0] for v in rs_values])
        y = np.array([v[1] for v in rs_values])

        # Slope = Hurst exponent
        A = np.vstack([x, np.ones(len(x))]).T
        H, _ = np.linalg.lstsq(A, y, rcond=None)[0]

        return round(H, 4)

    def compute_adx_regime(self, ohlcv: Dict) -> Tuple[str, float, float]:
        """
        ADX (Average Directional Index, period=14) + Volatility Regime Classification.

        ADX levels:
        - adx < 20: no trend, ranging
        - 20–25: weak trend forming
        - 25–40: established trend
        - > 40: strong trend

        Volatility percentile:
        hist_vol = std(log_returns, 20) * sqrt(252)
        vol_percentile = percentile_rank(hist_vol, 1yr)

        Regime classification:
        - MEAN_REVERT: adx < 25 AND vol_percentile < 35
        - TRANSITIONAL: adx 20–30 OR vol_percentile 30–50
        - TRENDING: adx > 30 AND vol_percentile > 50
        - BREAKOUT: adx > 40 AND vol_percentile > 75

        Only MEAN_REVERT and TRANSITIONAL support swing reversion trades.

        Returns:
            (regime_classification, adx_value, vol_percentile)
        """
        adx = ohlcv.get("adx", 20)
        vol_pct = ohlcv.get("volatility_percentile", 50)

        # Classify regime
        if adx > 40 and vol_pct > 75:
            regime = "BREAKOUT"
        elif adx > 30 and vol_pct > 50:
            regime = "TRENDING"
        elif adx < 25 and vol_pct < 35:
            regime = "MEAN_REVERT"
        else:
            regime = "TRANSITIONAL"

        return regime, adx, vol_pct

    async def compute(self, symbol: str, data: Dict[str, Any]) -> RegimeResult:
        """Classify current market regime."""
        self.logger.debug(f"Classifying regime for {symbol}")

        # Calculate Hurst Exponent if price data available
        hurst = 0.5
        if "prices" in data:
            prices = np.array(data["prices"])
            window = self.config.get("hurst_window", 252)
            hurst = self.compute_hurst_exponent(prices, window)
        else:
            hurst = data.get("hurst_exponent", 0.5)

        # ADX Regime
        ohlcv = data.get("ohlcv", {})
        adx_regime, adx, vol_pct = self.compute_adx_regime(ohlcv)

        # Determine regime from Hurst and ADX
        if hurst < 0.4 or (hurst < 0.5 and adx_regime == "MEAN_REVERT"):
            regime = "mean_reverting"
            score = self.config["regime_scores"]["mean_reverting"]
            is_valid = True
        elif hurst > 0.6 or adx_regime in ["TRENDING", "BREAKOUT"]:
            regime = "trending"
            score = self.config["regime_scores"]["trending"]
            is_valid = False
        else:
            regime = "random_walk"
            score = self.config["regime_scores"]["random_walk"]
            is_valid = adx_regime in ["MEAN_REVERT", "TRANSITIONAL"]

        # ADX confirmation - mean reversion valid in compression
        if adx < 25 and vol_pct < 35:
            if regime != "trending":
                is_valid = True
                score = max(score, 1.0)

        # HMM State (if returns data available)
        hmm_state = "NORMAL"
        hmm_confidence = 0.5
        if "returns" in data:
            returns = np.array(data["returns"])
            hmm_result = self.fit_hmm_regime(
                returns, n_states=self.config.get("hmm_states", 3)
            )
            hmm_state = hmm_result["state_label"]
            hmm_confidence = hmm_result["state_probability"]

        # Composite regime classification
        composite = self.classify_composite_regime(hurst, adx_regime, hmm_state)

        # Hurst signal label
        if hurst < 0.5:
            hurst_signal = "MEAN_REVERT"
        elif hurst > 0.6:
            hurst_signal = "TRENDING"
        else:
            hurst_signal = "NEUTRAL"

        return RegimeResult(
            symbol=symbol,
            hurst_exponent=hurst,
            hurst_signal=hurst_signal,
            adx_value=adx,
            adx_regime=adx_regime,
            vol_percentile=vol_pct,
            hmm_state=hmm_state,
            hmm_confidence=hmm_confidence,
            composite_regime=composite,
            supports_reversion=composite in ["FAVORABLE", "NEUTRAL"],
            score=score,
            timestamp=datetime.now().isoformat(),
        )

    def fit_hmm_regime(self, returns: np.ndarray, n_states: int = 3) -> dict:
        """
        Fit a Gaussian HMM to log returns to classify latent market regimes.

        Use hmmlearn GaussianHMM with n_states=3 (low vol, normal, high vol/crisis).

        Parameters:
            returns: log return array
            n_states: 3 default — config: swing_params.hmm_states
            covariance_type: 'full'
            n_iter: 100

        Output:
            current_state: int (0, 1, 2)
            state_label: 'LOW_VOL' | 'NORMAL' | 'HIGH_VOL_CRISIS'
            state_probability: float (confidence in current state)
            transition_matrix: np.ndarray (3×3)
            regime_duration: int (bars in current state)

        Note: state labels are assigned post-fit by sorting states by mean return variance.
        Refit monthly on 2yr rolling window. Cache fitted model in Firebase.
        """
        try:
            from hmmlearn import hmm
        except ImportError:
            self.logger.warning("hmmlearn not installed, returning default")
            return {
                "current_state": 1,
                "state_label": "NORMAL",
                "state_probability": 0.5,
                "transition_matrix": np.eye(3),
                "regime_duration": 0,
            }

        # Need at least 2 years of data (assuming 252 trading days/year)
        min_samples = 504
        if len(returns) < min_samples:
            self.logger.warning(
                f"Insufficient data for HMM: {len(returns)} < {min_samples}"
            )
            return {
                "current_state": 1,
                "state_label": "NORMAL",
                "state_probability": 0.5,
                "transition_matrix": np.eye(3),
                "regime_duration": 0,
            }

        # Reshape for hmmlearn (n_samples, n_features)
        X = returns.reshape(-1, 1)

        # Fit HMM
        model = hmm.GaussianHMM(
            n_components=n_states, covariance_type="full", n_iter=100, random_state=42
        )

        try:
            model.fit(X)
        except Exception as e:
            self.logger.error(f"HMM fit failed: {e}")
            return {
                "current_state": 1,
                "state_label": "NORMAL",
                "state_probability": 0.5,
                "transition_matrix": np.eye(3),
                "regime_duration": 0,
            }

        # Predict hidden states
        hidden_states = model.predict(X)
        current_state = hidden_states[-1]

        # Calculate state probabilities
        state_probs = model.predict_proba(X)[-1]
        state_probability = state_probs[current_state]

        # Sort states by variance to label them
        variances = model.covars_.flatten() if model.covars_ is not None else [1, 1, 1]
        sorted_indices = np.argsort(variances)

        # Map state index to label
        label_map = {}
        for rank, idx in enumerate(sorted_indices):
            if rank == 0:
                label_map[idx] = "LOW_VOL"
            elif rank == 1:
                label_map[idx] = "NORMAL"
            else:
                label_map[idx] = "HIGH_VOL_CRISIS"

        state_label = label_map.get(current_state, "NORMAL")

        # Count duration in current state
        regime_duration = 0
        for i in range(len(hidden_states) - 1, -1, -1):
            if hidden_states[i] == current_state:
                regime_duration += 1
            else:
                break

        return {
            "current_state": current_state,
            "state_label": state_label,
            "state_probability": round(state_probability, 4),
            "transition_matrix": model.transmat_.tolist(),
            "regime_duration": regime_duration,
        }

    def classify_composite_regime(
        self, hurst: float, adx_regime: str, hmm_state: str
    ) -> str:
        """
        Composite regime logic:

        FAVORABLE: hurst < 0.5 AND adx_regime in ['MEAN_REVERT', 'TRANSITIONAL']
                   AND hmm_state != 'HIGH_VOL_CRISIS'
        UNFAVORABLE: hurst > 0.6 OR adx_regime == 'BREAKOUT'
                     OR hmm_state == 'HIGH_VOL_CRISIS'
        NEUTRAL: everything else
        """
        favorable = (
            hurst < 0.5
            and adx_regime in ["MEAN_REVERT", "TRANSITIONAL"]
            and hmm_state != "HIGH_VOL_CRISIS"
        )

        unfavorable = (
            hurst > 0.6 or adx_regime == "BREAKOUT" or hmm_state == "HIGH_VOL_CRISIS"
        )

        if favorable:
            return "FAVORABLE"
        elif unfavorable:
            return "UNFAVORABLE"
        else:
            return "NEUTRAL"
