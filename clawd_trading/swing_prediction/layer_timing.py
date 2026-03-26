"""
LAYER E — Calendar / Flow Timing
OpEx, FOMC, Quarter-end, Earnings — structural timing windows
"""

import logging
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class TimingResult:
    score: float
    opex_window: str
    fomc_window: str
    quarter_position: str
    composite_signal: str


class TimingLayer:
    """
    Layer E: Calendar and Flow Timing
    Knowing what will move doesn't tell you when.
    """
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config["layer_e_timing"]
    
    async def compute(self, symbol: str) -> TimingResult:
        """Determine current timing window."""
        self.logger.debug(f"Checking timing for {symbol}")
        
        now = datetime.now()
        
        # Get timing signals
        opex = self._get_opex_window(now)
        fomc = self._get_fomc_window(now)
        quarter = self._get_quarter_position(now)
        earnings = self._get_earnings_season(now)
        
        # Calculate scores
        opex_score = self.config["opex"][f"{opex}_score"]
        fomc_score = self.config["fomc"][f"{fomc}"]
        quarter_score = self.config["quarter_end"][f"{quarter}"]
        earnings_score = self.config["earnings"][f"{earnings}"]
        
        # Weighted composite
        weights = self.config["weights"]
        score = (
            opex_score * weights["opex"] +
            fomc_score * weights["fomc"] +
            quarter_score * weights["quarter_end"] +
            earnings_score * weights["earnings"]
        )
        
        # Composite signal description
        signals = []
        if opex_score > 0.5: signals.append("post-opex")
        elif opex_score < 0: signals.append("opex-pressure")
        if fomc_score > 0.5: signals.append("post-fomc")
        elif fomc_score < 0: signals.append("pre-fomc")
        if quarter_score > 0: signals.append("fresh-quarter")
        elif quarter_score < 0: signals.append("rebal-pressure")
        
        return TimingResult(
            score=max(-3, min(3, score * 3)),  # Scale to -3 to 3
            opex_window=opex,
            fomc_window=fomc,
            quarter_position=quarter,
            composite_signal=" | ".join(signals) if signals else "neutral"
        )
    
    def _get_opex_window(self, now: datetime) -> str:
        """Determine OpEx window."""
        # Monthly OpEx is typically the 3rd Friday
        # This is simplified - real implementation would calculate actual dates
        day = now.day
        weekday = now.weekday()
        
        # Approximate: 3rd Friday is day 15-21
        if 15 <= day <= 21 and weekday == 4:  # Friday
            return "during"
        elif 8 <= day <= 14:
            return "before"
        elif 22 <= day <= 28:
            return "after"
        else:
            return "neutral"
    
    def _get_fomc_window(self, now: datetime) -> str:
        """Determine FOMC cycle position."""
        # FOMC meetings 8x per year
        # This is simplified - real implementation would use actual FOMC dates
        # For now, return neutral - real data needed
        return "neutral"
    
    def _get_quarter_position(self, now: datetime) -> str:
        """Determine quarter-end proximity."""
        month = now.month
        day = now.day
        
        # Month-ends that are quarter-ends: Mar, Jun, Sep, Dec
        quarter_end_months = [3, 6, 9, 12]
        
        if month in quarter_end_months:
            if day >= 15:
                return "last_two_weeks" if day < 24 else "last_week"
        elif month in [m + 1 for m in quarter_end_months]:  # Month after quarter-end
            if day <= 7:
                return "first_week_new"
        
        return "neutral"
    
    def _get_earnings_season(self, now: datetime) -> str:
        """Determine earnings season phase."""
        month = now.month
        
        # Earnings seasons: Jan, Apr, Jul, Oct (mostly)
        peak_earnings = [1, 4, 7, 10]
        
        if month in peak_earnings:
            return "active"
        elif month in [m - 1 for m in peak_earnings if m > 1]:
            return "pre"
        elif month in [m + 1 for m in peak_earnings if m < 12]:
            return "post"
        else:
            return "neutral"
