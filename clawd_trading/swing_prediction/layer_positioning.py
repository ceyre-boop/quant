"""
LAYER B — Positioning Data
COT Index, Put/Call ratios, Options Skew, Sentiment
"""

import logging
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class PositioningResult:
    score: float  # -3 to +3
    direction: str
    cot_signal: str
    options_signal: str
    sentiment_signal: str
    raw_values: Dict[str, float]


class PositioningLayer:
    """
    Layer B: Positioning Data
    Commercial Hedgers (smart money) vs Speculators (dumb money)
    """
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config["layer_b_positioning"]
    
    async def compute(self, symbol: str, data: Dict[str, Any]) -> PositioningResult:
        """Compute positioning score from multiple sources."""
        self.logger.debug(f"Computing positioning for {symbol}")
        
        signals = []
        raw_values = {}
        
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
            direction = "long"  # Extreme bearish positioning = contrarian long
        elif composite >= 1.5:
            direction = "short"  # Extreme bullish positioning = contrarian short
        else:
            direction = "neutral"
        
        return PositioningResult(
            score=max(-3, min(3, composite)),
            direction=direction,
            cot_signal=cot_signal,
            options_signal=f"{pc_signal}/{skew_signal}",
            sentiment_signal=fg_signal,
            raw_values=raw_values
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
