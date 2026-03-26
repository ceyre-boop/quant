"""
SWING PREDICTION LAYER — Master Orchestrator
Runs monthly scan (and optional weekly rescore) across symbol universe.
Produces SwingBias objects that gate the intraday three-layer system.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio

# Import layer modules
from swing_prediction.layer_fv import FairValueLayer
from swing_prediction.layer_positioning import PositioningLayer
from swing_prediction.layer_regime import RegimeLayer
from swing_prediction.layer_options import OptionsLayer
from swing_prediction.layer_timing import TimingLayer
from swing_prediction.scorer import CompositeScorer
from swing_prediction.backtest_base_rates import BaseRateCalculator
from swing_prediction.firebase_writer import FirebaseWriter


class SwingDirection(Enum):
    STRONG_LONG = "strong_long"
    MODERATE_LONG = "moderate_long"
    NEUTRAL = "neutral"
    MODERATE_SHORT = "moderate_short"
    STRONG_SHORT = "strong_short"


@dataclass
class SwingBias:
    """Output object for swing prediction layer."""
    symbol: str
    timestamp: datetime
    tradeable: bool
    direction: SwingDirection
    composite_score: float
    confidence: float
    base_rate: Optional[float]
    avg_return_20d: Optional[float]
    avg_return_40d: Optional[float]
    avg_return_60d: Optional[float]
    max_drawdown: Optional[float]
    layers_aligned: int
    layer_scores: Dict[str, float]
    block_reason: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "tradeable": self.tradeable,
            "direction": self.direction.value,
            "composite_score": round(self.composite_score, 4),
            "confidence": round(self.confidence, 4) if self.confidence else None,
            "base_rate": round(self.base_rate, 4) if self.base_rate else None,
            "avg_return_20d": round(self.avg_return_20d, 4) if self.avg_return_20d else None,
            "avg_return_40d": round(self.avg_return_40d, 4) if self.avg_return_40d else None,
            "avg_return_60d": round(self.avg_return_60d, 4) if self.avg_return_60d else None,
            "max_drawdown": round(self.max_drawdown, 4) if self.max_drawdown else None,
            "layers_aligned": self.layers_aligned,
            "layer_scores": {k: round(v, 4) for k, v in self.layer_scores.items()},
            "block_reason": self.block_reason
        }


class SwingEngine:
    """
    Master orchestrator for monthly swing prediction scan.
    """
    
    def __init__(self, config_path: str = "config/swing_params.json"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Initialize layers
        self.fv_layer = FairValueLayer(self.config)
        self.positioning_layer = PositioningLayer(self.config)
        self.regime_layer = RegimeLayer(self.config)
        self.options_layer = OptionsLayer(self.config)
        self.timing_layer = TimingLayer(self.config)
        
        # Initialize scorer and base rate calculator
        self.scorer = CompositeScorer(self.config)
        self.base_rate_calc = BaseRateCalculator(self.config)
        self.firebase_writer = FirebaseWriter(self.config)
        
        self.logger.info("SwingEngine initialized with config from %s", config_path)
    
    def _load_config(self, path: str) -> Dict:
        """Load swing parameters from JSON config."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error("Failed to load config from %s: %s", path, e)
            raise
    
    async def run_monthly_scan(self) -> List[SwingBias]:
        """
        Run full monthly scan across all symbols in universe.
        Returns list of SwingBias objects.
        """
        self.logger.info("Starting monthly swing scan at %s", datetime.now())
        
        symbols = self._get_symbol_universe()
        results = []
        
        scan_start = datetime.now()
        scan_log = {
            "scan_type": "monthly",
            "start_time": scan_start.isoformat(),
            "symbols_scanned": len(symbols),
            "symbols_tradeable": 0,
            "errors": []
        }
        
        for symbol in symbols:
            try:
                bias = await self._score_symbol(symbol)
                results.append(bias)
                
                if bias.tradeable:
                    scan_log["symbols_tradeable"] += 1
                    
                # Write individual result to Firebase
                await self.firebase_writer.write_swing_bias(bias)
                
            except Exception as e:
                self.logger.error("Error scoring %s: %s", symbol, e)
                scan_log["errors"].append({"symbol": symbol, "error": str(e)})
        
        scan_log["end_time"] = datetime.now().isoformat()
        scan_log["duration_seconds"] = (datetime.now() - scan_start).total_seconds()
        
        # Write scan log
        await self.firebase_writer.write_scan_log(scan_log)
        
        self.logger.info("Monthly scan complete. Tradeable symbols: %d/%d", 
                        scan_log["symbols_tradeable"], len(symbols))
        
        return results
    
    async def run_weekly_rescore(self, symbols: Optional[List[str]] = None) -> List[SwingBias]:
        """
        Run weekly rescore for specified symbols (or all if None).
        Lighter weight than monthly - only updates dynamic layers.
        """
        self.logger.info("Starting weekly rescore at %s", datetime.now())
        
        if symbols is None:
            symbols = self._get_symbol_universe()
        
        results = []
        for symbol in symbols:
            try:
                bias = await self._rescore_symbol(symbol)
                results.append(bias)
                await self.firebase_writer.write_swing_bias(bias)
            except Exception as e:
                self.logger.error("Error rescoring %s: %s", symbol, e)
        
        return results
    
    async def _score_symbol(self, symbol: str) -> SwingBias:
        """
        Run full 5-layer scoring for a single symbol.
        """
        self.logger.debug("Scoring symbol: %s", symbol)
        
        asset_class = self._get_asset_class(symbol)
        
        # Fetch all required data
        data = await self._fetch_symbol_data(symbol, asset_class)
        
        # Run each layer
        fv_result = await self.fv_layer.compute(symbol, asset_class, data)
        positioning_result = await self.positioning_layer.compute(symbol, data)
        regime_result = await self.regime_layer.compute(symbol, data)
        options_result = await self.options_layer.compute(symbol, data)
        timing_result = await self.timing_layer.compute(symbol)
        
        # Aggregate layer scores
        layer_scores = {
            "fair_value": fv_result.score,
            "positioning": positioning_result.score,
            "regime": regime_result.score,
            "options": options_result.score,
            "timing": timing_result.score
        }
        
        # Count aligned layers (score magnitude above threshold)
        layers_aligned = sum(1 for s in layer_scores.values() if abs(s) >= 0.5)
        
        # Calculate composite score
        composite = self.scorer.calculate_composite(layer_scores)
        
        # Determine direction and tradeability
        direction = self.scorer.direction_from_score(composite)
        
        # Check base rates if we have enough signal
        base_rate_data = None
        if layers_aligned >= self.config["composite_scoring"]["min_layers_for_tradeable"]:
            base_rate_data = await self.base_rate_calc.get_base_rate(
                symbol, layer_scores, direction
            )
        
        # Determine if tradeable
        tradeable, block_reason = self._determine_tradeability(
            composite, layers_aligned, base_rate_data
        )
        
        # Build SwingBias object
        bias = SwingBias(
            symbol=symbol,
            timestamp=datetime.now(),
            tradeable=tradeable,
            direction=direction,
            composite_score=composite,
            confidence=base_rate_data.get("confidence") if base_rate_data else None,
            base_rate=base_rate_data.get("win_rate") if base_rate_data else None,
            avg_return_20d=base_rate_data.get("avg_return_20d") if base_rate_data else None,
            avg_return_40d=base_rate_data.get("avg_return_40d") if base_rate_data else None,
            avg_return_60d=base_rate_data.get("avg_return_60d") if base_rate_data else None,
            max_drawdown=base_rate_data.get("max_drawdown") if base_rate_data else None,
            layers_aligned=layers_aligned,
            layer_scores=layer_scores,
            block_reason=block_reason
        )
        
        return bias
    
    async def _rescore_symbol(self, symbol: str) -> SwingBias:
        """
        Lightweight weekly rescore - only updates dynamic layers.
        Keeps fair value and base rate from monthly scan.
        """
        # Fetch previous monthly bias
        prev_bias = await self.firebase_writer.get_latest_swing_bias(symbol)
        
        # Only recompute dynamic layers
        data = await self._fetch_symbol_data(symbol, self._get_asset_class(symbol))
        
        positioning_result = await self.positioning_layer.compute(symbol, data)
        options_result = await self.options_layer.compute(symbol, data)
        timing_result = await self.timing_layer.compute(symbol)
        
        # Merge with previous layer scores
        layer_scores = prev_bias.layer_scores if prev_bias else {}
        layer_scores["positioning"] = positioning_result.score
        layer_scores["options"] = options_result.score
        layer_scores["timing"] = timing_result.score
        
        # Recalculate composite
        composite = self.scorer.calculate_composite(layer_scores)
        direction = self.scorer.direction_from_score(composite)
        layers_aligned = sum(1 for s in layer_scores.values() if abs(s) >= 0.5)
        
        tradeable, block_reason = self._determine_tradeability(composite, layers_aligned, None)
        
        return SwingBias(
            symbol=symbol,
            timestamp=datetime.now(),
            tradeable=tradeable,
            direction=direction,
            composite_score=composite,
            confidence=prev_bias.confidence if prev_bias else None,
            base_rate=prev_bias.base_rate if prev_bias else None,
            avg_return_20d=prev_bias.avg_return_20d if prev_bias else None,
            avg_return_40d=prev_bias.avg_return_40d if prev_bias else None,
            avg_return_60d=prev_bias.avg_return_60d if prev_bias else None,
            max_drawdown=prev_bias.max_drawdown if prev_bias else None,
            layers_aligned=layers_aligned,
            layer_scores=layer_scores,
            block_reason=block_reason
        )
    
    def _get_symbol_universe(self) -> List[str]:
        """Get all symbols from config."""
        universe = self.config["symbol_universe"]
        symbols = []
        for asset_class in universe.values():
            symbols.extend(asset_class)
        return symbols
    
    def _get_asset_class(self, symbol: str) -> str:
        """Determine asset class from symbol."""
        universe = self.config["symbol_universe"]
        for asset_class, symbols in universe.items():
            if symbol in symbols:
                return asset_class
        return "equities"  # default
    
    async def _fetch_symbol_data(self, symbol: str, asset_class: str) -> Dict:
        """
        Fetch all required data for a symbol.
        This is a placeholder - actual implementation would call data providers.
        """
        # TODO: Implement actual data fetching
        # This would integrate with:
        # - Price data (Yahoo Finance, Polygon, etc.)
        # - COT data (CFTC)
        # - Options data (CBOE, Polygon)
        # - Calendar data (calculated)
        
        return {
            "symbol": symbol,
            "asset_class": asset_class,
            "timestamp": datetime.now().isoformat()
        }
    
    def _determine_tradeability(
        self, 
        composite: float, 
        layers_aligned: int,
        base_rate_data: Optional[Dict]
    ) -> tuple[bool, Optional[str]]:
        """
        Determine if symbol is tradeable based on composite score and base rates.
        Returns (tradeable, block_reason).
        """
        min_layers = self.config["composite_scoring"]["min_layers_for_tradeable"]
        
        if layers_aligned < min_layers:
            return False, f"Only {layers_aligned} layers aligned (min: {min_layers})"
        
        if abs(composite) < self.config["composite_scoring"]["thresholds"]["neutral"]:
            return False, f"Composite score {composite:.2f} too close to neutral"
        
        if base_rate_data:
            min_confidence = self.config["composite_scoring"]["min_confidence_for_tradeable"]
            if base_rate_data.get("confidence", 0) < min_confidence:
                return False, f"Base rate confidence {base_rate_data['confidence']:.2f} below threshold"
            if base_rate_data.get("win_rate", 0) < 0.52:
                return False, f"Historical win rate {base_rate_data['win_rate']:.2f} insufficient"
        
        return True, None


# Singleton instance
_swing_engine: Optional[SwingEngine] = None


def get_swing_engine() -> SwingEngine:
    """Get or create singleton SwingEngine instance."""
    global _swing_engine
    if _swing_engine is None:
        _swing_engine = SwingEngine()
    return _swing_engine


if __name__ == "__main__":
    # Test run
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        engine = SwingEngine()
        # results = await engine.run_monthly_scan()
        # print(f"Scanned {len(results)} symbols")
    
    asyncio.run(test())
