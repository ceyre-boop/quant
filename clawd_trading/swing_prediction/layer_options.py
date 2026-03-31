"""
LAYER D — Options Market Signals
GEX, VIX term structure, IV/RV spread
"""

import logging
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class OptionsResult:
    score: float
    gex_signal: str
    vix_signal: str
    iv_signal: str
    expected_move: float


class OptionsLayer:
    """
    Layer D: Options Market as Forward Signal
    """
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config["layer_d_options"]
    
    async def compute(self, symbol: str, data: Dict[str, Any]) -> OptionsResult:
        """Analyze options market structure."""
        self.logger.debug(f"Analyzing options for {symbol}")
        
        signals = []
        
        # GEX Analysis
        gex = data.get("gamma_exposure", 0)
        gex_score, gex_sig = self._analyze_gex(gex)
        signals.append((gex_score, self.config["weights"]["gex"]))
        
        # VIX Term Structure
        vix_spread = data.get("vix_term_spread", 0)  # front - back month
        vix_score, vix_sig = self._analyze_vix(vix_spread)
        signals.append((vix_score, self.config["weights"]["vix_term"]))
        
        # IV Rank
        iv_rank = data.get("iv_rank", 50)
        iv_score, iv_sig = self._analyze_iv(iv_rank)
        signals.append((iv_score, self.config["weights"]["iv_rank"]))
        
        # IV/RV Spread
        iv_rv = data.get("iv_rv_spread", 0.1)
        ivrv_score, ivrv_sig = self._analyze_ivrv(iv_rv)
        signals.append((ivrv_score, self.config["weights"]["iv_rv_spread"]))
        
        # Weighted composite
        total_weight = sum(w for _, w in signals)
        score = sum(s * w for s, w in signals) / total_weight if total_weight > 0 else 0
        
        # Expected move calculation
        atm_iv = data.get("atm_iv", 0.2)
        days = 20
        expected_move = atm_iv * np.sqrt(days / 365)
        
        return OptionsResult(
            score=max(-3, min(3, score)),
            gex_signal=gex_sig,
            vix_signal=vix_sig,
            iv_signal=f"{iv_sig}/{ivrv_sig}",
            expected_move=expected_move
        )
    
    def _analyze_gex(self, gex: float) -> tuple[float, str]:
        """Gamma Exposure - dealer hedging effects."""
        if gex < self.config["gex"]["extreme_negative"]:
            return -2.0, "extreme_negative_amplify"
        elif gex > self.config["gex"]["extreme_positive"]:
            return 1.0, "extreme_positive_pin"
        elif gex < -self.config["gex"]["notable_threshold"]:
            return -1.0, "negative_amplify"
        elif gex > self.config["gex"]["notable_threshold"]:
            return 0.5, "positive_pin"
        else:
            return 0, "neutral"
    
    def _analyze_vix(self, spread: float) -> tuple[float, str]:
        """VIX term structure - contango vs backwardation."""
        if spread < self.config["vix_term_structure"]["extreme_backwardation"]:
            return -1.5, "extreme_fear"
        elif spread < self.config["vix_term_structure"]["backwardation"]:
            return -0.5, "fear"
        elif spread > self.config["vix_term_structure"]["steep_contango"]:
            return 0.5, "complacency"
        else:
            return 0, "normal"
    
    def _analyze_iv(self, iv_rank: float) -> tuple[float, str]:
        """IV Rank - options cheap or expensive."""
        if iv_rank >= self.config["iv_rank"]["expensive"]:
            return -1.0, "expensive"
        elif iv_rank <= self.config["iv_rank"]["cheap"]:
            return 1.0, "cheap"
        else:
            return 0, "fair"
    
    def _analyze_ivrv(self, spread: float) -> tuple[float, str]:
        """IV/RV spread - options over/under priced vs realized."""
        if spread > self.config["iv_rv_spread"]["overpriced"]:
            return -0.5, "overpriced"
        elif spread < self.config["iv_rv_spread"]["underpriced"]:
            return 0.5, "underpriced"
        else:
            return 0, "fair"
