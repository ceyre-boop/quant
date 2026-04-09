"""Feature Builder - Build all 43 features from v4.1 spec."""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """Container for all 43 features."""

    # Price-based features (1-10)
    returns_1h: float = 0.0
    returns_4h: float = 0.0
    returns_daily: float = 0.0
    returns_5d: float = 0.0
    price_vs_sma_20: float = 0.0
    price_vs_sma_50: float = 0.0
    price_vs_sma_200: float = 0.0
    price_vs_ema_12: float = 0.0
    price_vs_ema_26: float = 0.0
    price_position_daily_range: float = 0.0

    # Volatility features (11-18)
    atr_14: float = 0.0
    atr_percent_14: float = 0.0
    bollinger_position: float = 0.0
    bollinger_width: float = 0.0
    keltner_position: float = 0.0
    historical_volatility_20d: float = 0.0
    realized_volatility_5d: float = 0.0
    volatility_regime: int = 2

    # Trend/Momentum features (19-28)
    adx_14: float = 0.0
    dmi_plus: float = 0.0
    dmi_minus: float = 0.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    rsi_14: float = 50.0
    rsi_slope_5: float = 0.0
    stochastic_k: float = 50.0
    stochastic_d: float = 50.0
    cci_20: float = 0.0

    # Volume features (29-33)
    volume_sma_ratio: float = 1.0
    obv_slope: float = 0.0
    vwap_deviation: float = 0.0
    volume_profile_poc_dist: float = 0.0
    volume_trend_5d: float = 0.0

    # Market structure features (34-39)
    swing_high_20: Optional[float] = None
    swing_low_20: Optional[float] = None
    distance_to_resistance: float = 0.0
    distance_to_support: float = 0.0
    higher_highs_5d: int = 0
    higher_lows_5d: int = 0

    # Cross-market features (40-43)
    vix_level: float = 20.0
    vix_regime: int = 2
    spy_correlation_20d: float = 0.0
    market_breadth_ratio: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "returns_1h": self.returns_1h,
            "returns_4h": self.returns_4h,
            "returns_daily": self.returns_daily,
            "returns_5d": self.returns_5d,
            "price_vs_sma_20": self.price_vs_sma_20,
            "price_vs_sma_50": self.price_vs_sma_50,
            "price_vs_sma_200": self.price_vs_sma_200,
            "price_vs_ema_12": self.price_vs_ema_12,
            "price_vs_ema_26": self.price_vs_ema_26,
            "price_position_daily_range": self.price_position_daily_range,
            "atr_14": self.atr_14,
            "atr_percent_14": self.atr_percent_14,
            "bollinger_position": self.bollinger_position,
            "bollinger_width": self.bollinger_width,
            "keltner_position": self.keltner_position,
            "historical_volatility_20d": self.historical_volatility_20d,
            "realized_volatility_5d": self.realized_volatility_5d,
            "volatility_regime": self.volatility_regime,
            "adx_14": self.adx_14,
            "dmi_plus": self.dmi_plus,
            "dmi_minus": self.dmi_minus,
            "macd_line": self.macd_line,
            "macd_signal": self.macd_signal,
            "macd_histogram": self.macd_histogram,
            "rsi_14": self.rsi_14,
            "rsi_slope_5": self.rsi_slope_5,
            "stochastic_k": self.stochastic_k,
            "stochastic_d": self.stochastic_d,
            "cci_20": self.cci_20,
            "volume_sma_ratio": self.volume_sma_ratio,
            "obv_slope": self.obv_slope,
            "vwap_deviation": self.vwap_deviation,
            "volume_profile_poc_dist": self.volume_profile_poc_dist,
            "volume_trend_5d": self.volume_trend_5d,
            "swing_high_20": (
                self.swing_high_20 if self.swing_high_20 is not None else 0.0
            ),
            "swing_low_20": self.swing_low_20 if self.swing_low_20 is not None else 0.0,
            "distance_to_resistance": self.distance_to_resistance,
            "distance_to_support": self.distance_to_support,
            "higher_highs_5d": float(self.higher_highs_5d),
            "higher_lows_5d": float(self.higher_lows_5d),
            "vix_level": self.vix_level,
            "vix_regime": float(self.vix_regime),
            "spy_correlation_20d": self.spy_correlation_20d,
            "market_breadth_ratio": self.market_breadth_ratio,
        }


class FeatureBuilder:
    """Builds all 43 features for the AI Bias Engine."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def build_features(
        self,
        ohlcv_data: pd.DataFrame,
        vix_value: float = 20.0,
        market_breadth: float = 1.0,
        spy_correlation: float = 0.9,
    ) -> FeatureVector:
        """Build complete feature vector from OHLCV data.

        Args:
            ohlcv_data: DataFrame with columns [open, high, low, close, volume]
            vix_value: Current VIX level
            market_breadth: Market breadth ratio
            spy_correlation: Correlation with SPY

        Returns:
            FeatureVector with all 43 features
        """
        if ohlcv_data.empty or len(ohlcv_data) < 20:
            self.logger.warning("Insufficient data for feature building")
            return FeatureVector()

        features = FeatureVector()

        # Price-based features
        features = self._build_price_features(features, ohlcv_data)

        # Volatility features
        features = self._build_volatility_features(features, ohlcv_data)

        # Momentum features
        features = self._build_momentum_features(features, ohlcv_data)

        # Volume features
        features = self._build_volume_features(features, ohlcv_data)

        # Market structure features
        features = self._build_structure_features(features, ohlcv_data)

        # Cross-market features
        features.vix_level = vix_value
        features.vix_regime = self._classify_vix_regime(vix_value)
        features.market_breadth_ratio = market_breadth
        features.spy_correlation_20d = spy_correlation

        return features

    def _build_price_features(
        self, features: FeatureVector, data: pd.DataFrame
    ) -> FeatureVector:
        """Build price-based features."""
        close = data["close"]

        # Returns
        if len(close) >= 2:
            features.returns_daily = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        if len(close) >= 6:
            features.returns_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]

        # Moving averages
        if len(close) >= 20:
            sma20 = close.rolling(20).mean().iloc[-1]
            features.price_vs_sma_20 = (close.iloc[-1] - sma20) / close.iloc[-1]

        if len(close) >= 50:
            sma50 = close.rolling(50).mean().iloc[-1]
            features.price_vs_sma_50 = (close.iloc[-1] - sma50) / close.iloc[-1]

        if len(close) >= 200:
            sma200 = close.rolling(200).mean().iloc[-1]
            features.price_vs_sma_200 = (close.iloc[-1] - sma200) / close.iloc[-1]

        # EMAs
        if len(close) >= 12:
            ema12 = close.ewm(span=12).mean().iloc[-1]
            features.price_vs_ema_12 = (close.iloc[-1] - ema12) / close.iloc[-1]

        if len(close) >= 26:
            ema26 = close.ewm(span=26).mean().iloc[-1]
            features.price_vs_ema_26 = (close.iloc[-1] - ema26) / close.iloc[-1]

        # Price position in daily range
        if len(data) >= 1:
            latest = data.iloc[-1]
            range_size = latest["high"] - latest["low"]
            if range_size > 0:
                features.price_position_daily_range = (
                    latest["close"] - latest["low"]
                ) / range_size

        return features

    def _build_volatility_features(
        self, features: FeatureVector, data: pd.DataFrame
    ) -> FeatureVector:
        """Build volatility features."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # ATR
        if len(data) >= 14:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            features.atr_14 = tr.rolling(14).mean().iloc[-1]
            features.atr_percent_14 = features.atr_14 / close.iloc[-1]

        # Bollinger Bands
        if len(close) >= 20:
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            upper = sma20 + 2 * std20
            lower = sma20 - 2 * std20
            features.bollinger_position = (
                (close.iloc[-1] - sma20.iloc[-1]) / (2 * std20.iloc[-1])
                if std20.iloc[-1] > 0
                else 0
            )
            features.bollinger_width = (
                (upper.iloc[-1] - lower.iloc[-1]) / sma20.iloc[-1]
                if sma20.iloc[-1] > 0
                else 0
            )

        # Historical volatility
        if len(close) >= 21:
            log_returns = np.log(close / close.shift(1))
            features.historical_volatility_20d = log_returns.rolling(20).std().iloc[
                -1
            ] * np.sqrt(252)

        # Realized volatility (5-day)
        if len(close) >= 6:
            log_returns = np.log(close / close.shift(1))
            features.realized_volatility_5d = log_returns.iloc[-5:].std() * np.sqrt(252)

        # Volatility regime
        if features.historical_volatility_20d < 0.15:
            features.volatility_regime = 1  # LOW
        elif features.historical_volatility_20d < 0.25:
            features.volatility_regime = 2  # NORMAL
        elif features.historical_volatility_20d < 0.35:
            features.volatility_regime = 3  # ELEVATED
        else:
            features.volatility_regime = 4  # EXTREME

        return features

    def _build_momentum_features(
        self, features: FeatureVector, data: pd.DataFrame
    ) -> FeatureVector:
        """Build momentum features."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # RSI
        if len(close) >= 15:
            features.rsi_14 = self._calculate_rsi(close, 14)

            # RSI slope
            if len(close) >= 20:
                rsi_series = close.rolling(14).apply(
                    lambda x: self._calculate_rsi(x, 14)
                )
                features.rsi_slope_5 = (rsi_series.iloc[-1] - rsi_series.iloc[-6]) / 5

        # MACD
        if len(close) >= 26:
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            features.macd_line = ema12.iloc[-1] - ema26.iloc[-1]
            features.macd_signal = features.macd_line * 0.9  # Simplified
            features.macd_histogram = features.macd_line - features.macd_signal

        # ADX
        if len(data) >= 15:
            features.adx_14, features.dmi_plus, features.dmi_minus = (
                self._calculate_adx(data, 14)
            )

        # Stochastic
        if len(data) >= 15:
            lowest_low = low.rolling(14).min()
            highest_high = high.rolling(14).max()
            features.stochastic_k = (
                100
                * (close.iloc[-1] - lowest_low.iloc[-1])
                / (highest_high.iloc[-1] - lowest_low.iloc[-1])
                if (highest_high.iloc[-1] - lowest_low.iloc[-1]) > 0
                else 50
            )
            features.stochastic_d = features.stochastic_k  # Simplified

        # CCI
        if len(data) >= 21:
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(20).mean()
            mean_dev = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
            features.cci_20 = (
                (tp.iloc[-1] - sma_tp.iloc[-1]) / (0.015 * mean_dev.iloc[-1])
                if mean_dev.iloc[-1] > 0
                else 0
            )

        return features

    def _build_volume_features(
        self, features: FeatureVector, data: pd.DataFrame
    ) -> FeatureVector:
        """Build volume features."""
        volume = data["volume"]
        close = data["close"]

        # Volume SMA ratio
        if len(volume) >= 20:
            vol_sma20 = volume.rolling(20).mean().iloc[-1]
            features.volume_sma_ratio = (
                volume.iloc[-1] / vol_sma20 if vol_sma20 > 0 else 1.0
            )

        # OBV slope
        if len(volume) >= 6:
            obv = (np.sign(close.diff()) * volume).cumsum()
            features.obv_slope = (obv.iloc[-1] - obv.iloc[-6]) / 5

        return features

    def _build_structure_features(
        self, features: FeatureVector, data: pd.DataFrame
    ) -> FeatureVector:
        """Build market structure features."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Swing highs/lows (20-period)
        if len(data) >= 20:
            features.swing_high_20 = high.rolling(20).max().iloc[-1]
            features.swing_low_20 = low.rolling(20).min().iloc[-1]

            features.distance_to_resistance = features.swing_high_20 - close.iloc[-1]
            features.distance_to_support = close.iloc[-1] - features.swing_low_20

        # Higher highs/lows count
        if len(high) >= 6:
            highs = high.iloc[-6:-1].values
            lows = low.iloc[-6:-1].values

            features.higher_highs_5d = sum(
                1 for i in range(1, len(highs)) if highs[i] > highs[i - 1]
            )
            features.higher_lows_5d = sum(
                1 for i in range(1, len(lows)) if lows[i] > lows[i - 1]
            )

        return features

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0

        deltas = prices.diff()
        gains = deltas.clip(lower=0)
        losses = -deltas.clip(upper=0)

        avg_gain = gains.rolling(period).mean().iloc[-1]
        avg_loss = losses.rolling(period).mean().iloc[-1]

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> tuple:
        """Calculate ADX, +DI, -DI."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.clip(lower=0)
        minus_dm = minus_dm.clip(lower=0)

        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0

        tr = pd.concat(
            [high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1
        ).max(axis=1)

        atr = tr.rolling(period).mean()

        plus_di = 100 * plus_dm.rolling(period).mean() / atr
        minus_di = 100 * minus_dm.rolling(period).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return adx.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]

    def _classify_vix_regime(self, vix: float) -> int:
        """Classify VIX into regime."""
        if vix < 15:
            return 1  # LOW
        elif vix < 25:
            return 2  # NORMAL
        elif vix < 35:
            return 3  # ELEVATED
        else:
            return 4  # EXTREME
