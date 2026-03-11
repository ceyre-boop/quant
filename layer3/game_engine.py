"""Layer 3: Game-Theoretic Engine

Liquidity pool mapping, trapped positions, Nash zones, and order flow analysis.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from contracts.types import BiasOutput, AdversarialRisk

logger = logging.getLogger(__name__)


@dataclass
class LiquidityPool:
    """A detected liquidity pool."""
    price: float
    strength: int
    swept: bool
    age_bars: int
    draw_probability: float
    pool_type: str  # 'equal_highs' or 'equal_lows'


@dataclass
class TrappedPosition:
    """Trapped position estimate."""
    price_level: float
    estimated_size: float
    pain_distance: float
    unwind_probability: float


@dataclass
class NashZone:
    """Nash equilibrium zone."""
    price_level: float
    zone_type: str  # 'hvn', 'sr', 'round_number'
    state: str  # 'HOLDING', 'TESTED', 'BREAKING'
    test_count: int
    conviction: float


class LiquidityMap:
    """Liquidity pool mapper."""
    
    def __init__(self, tolerance_atr_pct: float = 0.15):
        self.tolerance_atr_pct = tolerance_atr_pct
    
    def build_liquidity_map(
        self,
        ohlcv: pd.DataFrame,
        atr: float
    ) -> Dict[str, List[LiquidityPool]]:
        """Build liquidity map from OHLCV data.
        
        Args:
            ohlcv: DataFrame with OHLCV data
            atr: Current ATR value
        
        Returns:
            Dict with 'equal_highs' and 'equal_lows' pools
        """
        if ohlcv.empty:
            return {'equal_highs': [], 'equal_lows': []}
        
        tolerance = atr * self.tolerance_atr_pct
        
        # Find swing highs and lows
        highs = ohlcv['high'].values
        lows = ohlcv['low'].values
        
        # Find clusters
        high_clusters = self._find_clusters(highs, tolerance)
        low_clusters = self._find_clusters(lows, tolerance)
        
        # Build pools
        equal_highs = [
            LiquidityPool(
                price=cluster['price'],
                strength=cluster['count'],
                swept=self._is_swept(cluster['price'], ohlcv, atr, 'high'),
                age_bars=cluster['age'],
                draw_probability=0.0,  # Calculated later
                pool_type='equal_highs'
            )
            for cluster in high_clusters
        ]
        
        equal_lows = [
            LiquidityPool(
                price=cluster['price'],
                strength=cluster['count'],
                swept=self._is_swept(cluster['price'], ohlcv, atr, 'low'),
                age_bars=cluster['age'],
                draw_probability=0.0,
                pool_type='equal_lows'
            )
            for cluster in low_clusters
        ]
        
        return {
            'equal_highs': equal_highs,
            'equal_lows': equal_lows
        }
    
    def _find_clusters(
        self,
        prices: np.ndarray,
        tolerance: float
    ) -> List[Dict]:
        """Find price clusters within tolerance."""
        if len(prices) < 3:
            return []
        
        # Simple clustering - group prices within tolerance
        clusters = []
        used = set()
        
        for i, price in enumerate(prices):
            if i in used:
                continue
            
            cluster_prices = [price]
            cluster_indices = [i]
            
            for j, other_price in enumerate(prices[i+1:], start=i+1):
                if j in used:
                    continue
                if abs(price - other_price) <= tolerance:
                    cluster_prices.append(other_price)
                    cluster_indices.append(j)
            
            if len(cluster_prices) >= 2:  # At least 2 touches
                clusters.append({
                    'price': np.mean(cluster_prices),
                    'count': len(cluster_prices),
                    'age': len(prices) - max(cluster_indices)
                })
                used.update(cluster_indices)
        
        return clusters
    
    def _is_swept(
        self,
        price: float,
        ohlcv: pd.DataFrame,
        atr: float,
        level_type: str
    ) -> bool:
        """Check if a liquidity pool has been swept."""
        if len(ohlcv) < 4:
            return False
        
        # Check last 3 bars
        recent = ohlcv.iloc[-3:]
        
        if level_type == 'high':
            # Swept if price exceeded level and closed back below
            for _, bar in recent.iterrows():
                if bar['high'] > price + 0.5 * atr and bar['close'] < price:
                    return True
        else:  # low
            for _, bar in recent.iterrows():
                if bar['low'] < price - 0.5 * atr and bar['close'] > price:
                    return True
        
        return False
    
    def compute_draw_probability(
        self,
        pool: LiquidityPool,
        bias: BiasOutput,
        current_price: float
    ) -> float:
        """Compute probability of price being drawn to pool.
        
        Uses sigmoid function based on:
        - Bias direction alignment
        - Pool strength
        - Distance to pool
        """
        # Direction score
        if pool.pool_type == 'equal_highs':
            direction_score = 1 if bias.direction.value == 1 else -1
        else:  # equal_lows
            direction_score = -1 if bias.direction.value == 1 else 1
        
        # Distance factor
        distance = abs(current_price - pool.price)
        distance_score = -distance / max(current_price * 0.01, 1)  # Normalize
        
        # Strength factor
        strength_score = pool.strength / 5  # Normalize to 0-1
        
        # Sigmoid combination
        z = (direction_score * 0.4 + strength_score * 0.3 + distance_score * 0.3)
        prob = 1 / (1 + np.exp(-z))
        
        return prob


class TrappedPositionDetector:
    """Detect trapped positions and estimate forced moves."""
    
    def detect_trapped_longs(
        self,
        ohlcv: pd.DataFrame,
        vwap_history: Optional[List[float]] = None
    ) -> List[TrappedPosition]:
        """Detect trapped long positions.
        
        Longs are trapped when:
        - Price broke below a consolidation zone they entered
        - Distance below VWAP is significant
        """
        if ohlcv.empty:
            return []
        
        trapped = []
        current_price = ohlcv['close'].iloc[-1]
        
        # Find consolidation zones (simplified)
        if len(ohlcv) >= 10:
            recent = ohlcv.iloc[-10:]
            avg_price = recent['close'].mean()
            
            # If current price is well below recent average
            if current_price < avg_price * 0.98:
                pain_distance = (avg_price - current_price) / current_price
                
                trapped.append(TrappedPosition(
                    price_level=avg_price,
                    estimated_size=recent['volume'].sum(),
                    pain_distance=pain_distance,
                    unwind_probability=min(pain_distance * 10, 0.8)
                ))
        
        return trapped
    
    def detect_trapped_shorts(
        self,
        ohlcv: pd.DataFrame,
        vwap_history: Optional[List[float]] = None
    ) -> List[TrappedPosition]:
        """Detect trapped short positions."""
        if ohlcv.empty:
            return []
        
        trapped = []
        current_price = ohlcv['close'].iloc[-1]
        
        if len(ohlcv) >= 10:
            recent = ohlcv.iloc[-10:]
            avg_price = recent['close'].mean()
            
            # If current price is well above recent average
            if current_price > avg_price * 1.02:
                pain_distance = (current_price - avg_price) / current_price
                
                trapped.append(TrappedPosition(
                    price_level=avg_price,
                    estimated_size=recent['volume'].sum(),
                    pain_distance=pain_distance,
                    unwind_probability=min(pain_distance * 10, 0.8)
                ))
        
        return trapped
    
    def calculate_squeeze_probability(
        self,
        trapped_longs: List[TrappedPosition],
        trapped_shorts: List[TrappedPosition]
    ) -> float:
        """Calculate probability of a squeeze (forced move)."""
        long_pain = sum(t.pain_distance for t in trapped_longs)
        short_pain = sum(t.pain_distance for t in trapped_shorts)
        
        if long_pain > short_pain * 2:
            return min(long_pain * 5, 0.8)  # Short squeeze potential
        elif short_pain > long_pain * 2:
            return min(short_pain * 5, 0.8)  # Long squeeze potential
        
        return 0.0


class AdversarialLevelModel:
    """Nash equilibrium zone detection."""
    
    def compute_nash_zones(
        self,
        ohlcv: pd.DataFrame,
        volume_profile: Optional[Dict] = None
    ) -> List[NashZone]:
        """Compute Nash equilibrium zones.
        
        Zones are where multiple strategies converge:
        - High volume nodes (HVN)
        - Structural support/resistance
        - Round numbers
        """
        if ohlcv.empty:
            return []
        
        zones = []
        current_price = ohlcv['close'].iloc[-1]
        
        # Find high volume nodes
        if len(ohlcv) >= 20:
            avg_volume = ohlcv['volume'].mean()
            high_vol_bars = ohlcv[ohlcv['volume'] > avg_volume * 1.5]
            
            for _, bar in high_vol_bars.iterrows():
                zones.append(NashZone(
                    price_level=bar['close'],
                    zone_type='hvn',
                    state='HOLDING',
                    test_count=0,
                    conviction=0.6
                ))
        
        # Add round number levels
        magnitude = 10 ** (len(str(int(current_price))) - 2)
        round_levels = [
            round(current_price / magnitude) * magnitude,
            round(current_price / (magnitude / 2)) * (magnitude / 2)
        ]
        
        for level in round_levels:
            zones.append(NashZone(
                price_level=level,
                zone_type='round_number',
                state='HOLDING',
                test_count=0,
                conviction=0.4
            ))
        
        return zones


class OrderFlowAnalyzer:
    """Analyze order flow using Kyle's lambda proxy."""
    
    def compute_kyle_lambda(
        self,
        trades: List[Dict[str, Any]]
    ) -> float:
        """Estimate Kyle's lambda from trade data.
        
        Kyle lambda = price impact per unit of order flow imbalance.
        High lambda = informed trading dominant.
        """
        if len(trades) < 10:
            return 0.0
        
        # Calculate signed volume and price changes
        signed_volumes = []
        price_changes = []
        
        prev_price = None
        for trade in trades:
            price = trade.get('price', 0)
            size = trade.get('size', 0)
            side = trade.get('side', 'unknown')
            
            # Sign volume
            if side == 'buy':
                signed_vol = size
            elif side == 'sell':
                signed_vol = -size
            else:
                signed_vol = 0
            
            if prev_price is not None:
                price_changes.append(price - prev_price)
                signed_volumes.append(signed_vol)
            
            prev_price = price
        
        if not signed_volumes:
            return 0.0
        
        # OLS: price_change = lambda * signed_volume + noise
        x = np.array(signed_volumes)
        y = np.array(price_changes)
        
        # Add regularization
        variance = np.var(x) + 1e-10
        covariance = np.cov(x, y)[0, 1] if len(x) > 1 else 0
        
        lambda_estimate = covariance / variance
        return float(lambda_estimate)


class GameEngine:
    """Composite game-theoretic engine."""
    
    def __init__(self):
        self.liquidity_mapper = LiquidityMap()
        self.trapped_detector = TrappedPositionDetector()
        self.nash_model = AdversarialLevelModel()
        self.order_flow = OrderFlowAnalyzer()
    
    def analyze(
        self,
        ohlcv: pd.DataFrame,
        bias: BiasOutput,
        current_price: float
    ) -> Dict[str, Any]:
        """Run complete game-theoretic analysis.
        
        Returns:
            Dict with all Layer 3 outputs
        """
        # Calculate ATR for context
        if len(ohlcv) >= 14:
            atr = self._calculate_atr(ohlcv)
        else:
            atr = current_price * 0.01
        
        # Build liquidity map
        liquidity_map = self.liquidity_mapper.build_liquidity_map(ohlcv, atr)
        
        # Calculate draw probabilities
        for pool_list in liquidity_map.values():
            for pool in pool_list:
                pool.draw_probability = self.liquidity_mapper.compute_draw_probability(
                    pool, bias, current_price
                )
        
        # Find nearest unswept pool
        all_pools = liquidity_map['equal_highs'] + liquidity_map['equal_lows']
        unswept = [p for p in all_pools if not p.swept]
        nearest_unswept = max(unswept, key=lambda p: p.draw_probability) if unswept else None
        
        # Detect trapped positions
        trapped_longs = self.trapped_detector.detect_trapped_longs(ohlcv)
        trapped_shorts = self.trapped_detector.detect_trapped_shorts(ohlcv)
        
        squeeze_prob = self.trapped_detector.calculate_squeeze_probability(
            trapped_longs, trapped_shorts
        )
        
        # Find Nash zones
        nash_zones = self.nash_model.compute_nash_zones(ohlcv)
        
        # Determine game state alignment
        game_aligned = self._check_alignment(
            bias, liquidity_map, trapped_longs, trapped_shorts
        )
        
        # Determine adversarial risk
        adversarial_risk = self._calculate_adversarial_risk(
            squeeze_prob, len(trapped_longs) + len(trapped_shorts)
        )
        
        return {
            'liquidity_map': liquidity_map,
            'nearest_unswept_pool': nearest_unswept,
            'trapped_positions': {
                'longs': trapped_longs,
                'shorts': trapped_shorts,
                'squeeze_probability': squeeze_prob
            },
            'forced_move_probability': squeeze_prob,
            'nash_zones': nash_zones,
            'kyle_lambda': 0.0,  # Would need trade data
            'game_state_aligned': game_aligned,
            'game_state_summary': self._generate_summary(
                liquidity_map, trapped_longs, trapped_shorts
            ),
            'adversarial_risk': adversarial_risk
        }
    
    def _calculate_atr(self, ohlcv: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(period).mean().iloc[-1]
    
    def _check_alignment(
        self,
        bias: BiasOutput,
        liquidity_map: Dict,
        trapped_longs: List,
        trapped_shorts: List
    ) -> bool:
        """Check if game state aligns with bias."""
        # Aligned if bias direction matches trapped positions
        if bias.direction.value == 1 and trapped_shorts:  # Long bias + trapped shorts = aligned
            return True
        if bias.direction.value == -1 and trapped_longs:  # Short bias + trapped longs = aligned
            return True
        return False
    
    def _calculate_adversarial_risk(
        self,
        squeeze_prob: float,
        trapped_count: int
    ) -> str:
        """Calculate adversarial risk level."""
        if squeeze_prob > 0.65 or trapped_count > 5:
            return AdversarialRisk.EXTREME.value
        elif squeeze_prob > 0.5 or trapped_count > 3:
            return AdversarialRisk.HIGH.value
        elif squeeze_prob > 0.3:
            return AdversarialRisk.MEDIUM.value
        return AdversarialRisk.LOW.value
    
    def _generate_summary(
        self,
        liquidity_map: Dict,
        trapped_longs: List,
        trapped_shorts: List
    ) -> str:
        """Generate human-readable summary."""
        parts = []
        
        if trapped_shorts:
            parts.append("SHORTS_TRAPPED")
        if trapped_longs:
            parts.append("LONGS_TRAPPED")
        
        unswept_highs = sum(1 for p in liquidity_map.get('equal_highs', []) if not p.swept)
        unswept_lows = sum(1 for p in liquidity_map.get('equal_lows', []) if not p.swept)
        
        if unswept_highs > 0:
            parts.append(f"{unswept_highs}_UNSWEPT_HIGHS")
        if unswept_lows > 0:
            parts.append(f"{unswept_lows}_UNSWEPT_LOWS")
        
        return '_'.join(parts) if parts else 'NEUTRAL'
