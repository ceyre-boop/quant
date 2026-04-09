"""
LAYER B — Positioning Data
COT Index, Put/Call ratios, Options Skew, Sentiment
"""

import logging
from typing import Dict, Any, Optional, Optional
from dataclasses import dataclass


@dataclass
class PositioningResult:
    """Output from positioning analysis."""

    symbol: str
    cot_index: Optional[float]  # None if no COT data
    cot_signal: str  # 'EXTREME_LONG' | 'LONG' | 'NEUTRAL' | 'SHORT' | 'EXTREME_SHORT'
    equity_sentiment: Optional[dict]  # pc_ratio, skew_percentile, aaii_percentile
    composite_score: float  # 0-1 normalized
    direction_bias: str  # 'LONG' | 'SHORT' | 'NEUTRAL'
    signal_strength: str  # 'NONE' | 'WEAK' | 'MODERATE' | 'STRONG' | 'EXTREME'
    raw_values: Dict[str, float]
    timestamp: str


class PositioningLayer:
    """
    Layer B: Positioning Data
    Commercial Hedgers (smart money) vs Speculators (dumb money)

    COT Index Formula:
    COT Index = percentile rank of current net commercial position over trailing N weeks
    commercial_net = commercial_longs - commercial_shorts
    cot_index = percentile_rank(commercial_net, trailing_window)

    Data source: CFTC Commitment of Traders report
    Primary: https://publicreporting.cftc.gov
    Fallback: Quandl/Nasdaq Data Link CFTC dataset
    """

    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config["layer_b_positioning"]

    def compute_equity_positioning(self, symbol: str, options_data: dict, sentiment_data: dict) -> dict:
        """
        For equity indices (no COT equivalent), use three proxies:

        1. Options Put/Call Ratio [Weight: 0.4]
           pc_ratio = put_volume / call_volume
           pc_z = (pc_ratio - mean(pc_ratio, 1yr)) / std(pc_ratio, 1yr)
           high pc_z (> 1.5) = fear extreme = contrarian bullish

        2. Options Skew (25-delta) [Weight: 0.4]
           skew = IV_25delta_put - IV_25delta_call
           skew_percentile = percentile_rank(skew, 1yr)
           skew_percentile > 85 = extreme fear = contrarian bullish

        3. AAII Sentiment Survey (weekly) [Weight: 0.2]
           bull_bear_spread = bull_pct - bear_pct
           spread_percentile = percentile_rank(bull_bear_spread, 5yr)
           spread_percentile < 10 = extreme bearishness = contrarian bullish

        Composite positioning_score = weighted average: [0.4, 0.4, 0.2]
        """
        from datetime import datetime

        scores = []
        weights = [0.4, 0.4, 0.2]
        details = {}

        # 1. Put/Call Ratio
        if "put_call_ratio" in options_data:
            pc_ratio = options_data["put_call_ratio"]
            pc_mean = options_data.get("pc_1yr_mean", 0.85)
            pc_std = options_data.get("pc_1yr_std", 0.15)

            if pc_std > 0:
                pc_z = (pc_ratio - pc_mean) / pc_std
                # High put/call = fear = contrarian bullish (negative score)
                scores.append(-pc_z)
                details["put_call_z"] = round(pc_z, 4)
                details["put_call_ratio"] = pc_ratio

        # 2. Options Skew
        if "skew_25d" in options_data:
            skew = options_data["skew_25d"]
            skew_history = options_data.get("skew_history", [])

            if len(skew_history) > 50:
                below_skew = sum(1 for s in skew_history if s < skew)
                skew_pct = (below_skew / len(skew_history)) * 100
                # High skew percentile = fear = contrarian bullish
                skew_score = -((skew_pct - 50) / 25)  # Normalize
                scores.append(skew_score)
                details["skew_percentile"] = round(skew_pct, 2)

        # 3. AAII Sentiment
        if "aaii_bull" in sentiment_data and "aaii_bear" in sentiment_data:
            bull_bear_spread = sentiment_data["aaii_bull"] - sentiment_data["aaii_bear"]
            spread_history = sentiment_data.get("spread_history", [])

            if len(spread_history) > 100:
                below_spread = sum(1 for s in spread_history if s < bull_bear_spread)
                spread_pct = (below_spread / len(spread_history)) * 100
                # Low spread percentile = extreme bearish = contrarian bullish
                spread_score = (spread_pct - 50) / 25
                scores.append(spread_score)
                details["aaii_spread_percentile"] = round(spread_pct, 2)

        # Weighted composite
        if scores:
            composite = sum(s * w for s, w in zip(scores, weights[: len(scores)])) / sum(weights[: len(scores)])
        else:
            composite = 0

        return {
            "symbol": symbol,
            "composite_score": round((composite + 3) / 6, 4),  # Normalize to 0-1
            "direction_bias": ("LONG" if composite < -1 else "SHORT" if composite > 1 else "NEUTRAL"),
            "signal_strength": self._get_signal_strength(abs(composite)),
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }

    def _get_signal_strength(self, magnitude: float) -> str:
        if magnitude >= 2.5:
            return "EXTREME"
        elif magnitude >= 2.0:
            return "STRONG"
        elif magnitude >= 1.5:
            return "MODERATE"
        elif magnitude >= 1.0:
            return "WEAK"
        else:
            return "NONE"

    def compute_cot_index(self, cot_data: Dict) -> float:
        """
        Compute COT Index from raw CFTC COT data.

        Args:
            cot_data: dict with keys:
                - commercial_longs: int
                - commercial_shorts: int
                - history: list of commercial_net values for trailing window

        Returns:
            COT Index (0-100 percentile)
        """
        if not all(k in cot_data for k in ["commercial_longs", "commercial_shorts"]):
            return 50.0  # Neutral if no data

        commercial_net = cot_data["commercial_longs"] - cot_data["commercial_shorts"]

        # Get trailing window (default 156 weeks = 3 years)
        history = cot_data.get("history", [])
        if len(history) < 20:  # Need minimum data
            return 50.0

        # Calculate percentile rank
        # Percentile = (number of values < current) / total * 100
        below_current = sum(1 for h in history if h < commercial_net)
        cot_index = (below_current / len(history)) * 100

        return round(cot_index, 2)

    async def compute(self, symbol: str, data: Dict[str, Any]) -> PositioningResult:
        """Compute positioning score from multiple sources."""
        from datetime import datetime

        self.logger.debug(f"Computing positioning for {symbol}")

        raw_values = {}

        # Check if this is an equity/index (no COT) or futures/forex/commodity
        asset_class = data.get("asset_class", "equity")

        if asset_class in ["equities", "equity", "index"]:
            # Use equity positioning method
            options_data = data.get("options", {})
            sentiment_data = data.get("sentiment", {})

            result = self.compute_equity_positioning(symbol, options_data, sentiment_data)

            return PositioningResult(
                symbol=symbol,
                cot_index=None,
                cot_signal="N/A",
                equity_sentiment=result["details"],
                composite_score=result["composite_score"],
                direction_bias=result["direction_bias"],
                signal_strength=result["signal_strength"],
                raw_values=result["details"],
                timestamp=result["timestamp"],
            )

        # For futures/forex/commodity - use COT data
        signals = []

        # COT Index (Commercials positioning)
        cot_score, cot_signal = self._analyze_cot(data)
        signals.append((cot_score, self.config["weights"]["cot_index"]))
        raw_values["cot_score"] = cot_score
        raw_values["cot_index"] = data.get("cot_index", 50)

        # Put/Call Ratio
        pc_score, pc_signal = self._analyze_put_call(data)
        signals.append((pc_score, self.config["weights"]["put_call"]))
        raw_values["put_call_score"] = pc_score
        raw_values["put_call_ratio"] = data.get("put_call_ratio", 1.0)

        # Options Skew
        skew_score, skew_signal = self._analyze_skew(data)
        signals.append((skew_score, self.config["weights"]["options_skew"]))
        raw_values["skew_score"] = skew_score
        raw_values["options_skew"] = data.get("options_skew", 1.0)

        # Fear & Greed
        fg_score, fg_signal = self._analyze_fear_greed(data)
        signals.append((fg_score, self.config["weights"]["fear_greed"]))
        raw_values["fear_greed_score"] = fg_score
        raw_values["fear_greed_index"] = data.get("fear_greed_index", 50)

        # Weighted composite
        total_weight = sum(w for _, w in signals)
        composite = sum(s * w for s, w in signals) / total_weight if total_weight > 0 else 0

        # Determine direction (contrarian interpretation)
        if composite <= -1.5:
            direction = "LONG"
            strength = self._get_signal_strength(abs(composite))
        elif composite >= 1.5:
            direction = "SHORT"
            strength = self._get_signal_strength(abs(composite))
        else:
            direction = "NEUTRAL"
            strength = "NONE"

        # Normalize composite to 0-1
        normalized_score = (composite + 3) / 6

        return PositioningResult(
            symbol=symbol,
            cot_index=data.get("cot_index", 50),
            cot_signal=cot_signal,
            equity_sentiment=None,
            composite_score=round(normalized_score, 4),
            direction_bias=direction,
            signal_strength=strength,
            raw_values=raw_values,
            timestamp=datetime.now().isoformat(),
        )

    def _analyze_cot(self, data: Dict) -> tuple[float, str]:
        """Analyze COT Index - commercials vs speculators."""
        cot = data.get("cot_index", 50)

        if cot <= self.config["cot_index"]["extreme_short"]:
            return -2.0, "commercials_extreme_short"
        elif cot >= self.config["cot_index"]["extreme_long"]:
            return 2.0, "commercials_extreme_long"
        elif cot <= self.config["cot_index"]["notable_short"]:
            return -1.0, "commercials_short"
        elif cot >= self.config["cot_index"]["notable_long"]:
            return 1.0, "commercials_long"
        else:
            return 0, "neutral"

    def _analyze_put_call(self, data: Dict) -> tuple[float, str]:
        """Analyze Put/Call ratio - contrarian indicator."""
        pcr = data.get("put_call_ratio", 1.0)

        if pcr >= self.config["put_call_ratio"]["extreme_bearish"]:
            return -1.5, "extreme_fear"
        elif pcr <= self.config["put_call_ratio"]["extreme_bullish"]:
            return 1.5, "extreme_greed"
        elif pcr > self.config["put_call_ratio"]["threshold"]:
            return -0.5, "fear"
        elif pcr < self.config["put_call_ratio"]["threshold"]:
            return 0.5, "greed"
        else:
            return 0, "neutral"

    def _analyze_skew(self, data: Dict) -> tuple[float, str]:
        """Analyze options skew - fear vs complacency."""
        skew = data.get("options_skew", 1.0)

        if skew >= self.config["options_skew"]["extreme_fear"]:
            return -1.5, "extreme_fear"
        elif skew <= self.config["options_skew"]["extreme_greed"]:
            return 1.5, "extreme_greed"
        elif skew > 1.1:
            return -0.5, "fear"
        elif skew < 0.95:
            return 0.5, "greed"
        else:
            return 0, "neutral"

    def _analyze_fear_greed(self, data: Dict) -> tuple[float, str]:
        """Analyze Fear & Greed Index - CNN style."""
        fgi = data.get("fear_greed_index", 50)

        if fgi <= self.config["fear_greed"]["extreme_fear"]:
            return -2.0, "extreme_fear"
        elif fgi >= self.config["fear_greed"]["extreme_greed"]:
            return 2.0, "extreme_greed"
        elif fgi < 40:
            return -0.5, "fear"
        elif fgi > 60:
            return 0.5, "greed"
        else:
            return 0, "neutral"
