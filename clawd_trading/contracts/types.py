"""Type Contracts - Single source of truth for all shared types.

All inter-layer interfaces are typed contracts. No layer may accept or return 
raw dicts in production code — always use typed dataclasses or Pydantic models.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


# =============================================================================
# Enums
# =============================================================================

class Direction(Enum):
    SHORT = -1
    NEUTRAL = 0
    LONG = 1


class Magnitude(Enum):
    SMALL = 1
    NORMAL = 2
    LARGE = 3


class VolRegime(Enum):
    LOW = 'LOW'
    NORMAL = 'NORMAL'
    ELEVATED = 'ELEVATED'
    EXTREME = 'EXTREME'


class TrendRegime(Enum):
    STRONG_TREND = 'STRONG_TREND'
    WEAK_TREND = 'WEAK_TREND'
    RANGING = 'RANGING'
    CHOPPY = 'CHOPPY'


class RiskAppetite(Enum):
    RISK_ON = 'RISK_ON'
    NEUTRAL = 'NEUTRAL'
    RISK_OFF = 'RISK_OFF'


class MomentumRegime(Enum):
    ACCELERATING = 'ACCELERATING'
    STEADY = 'STEADY'
    DECELERATING = 'DECELERATING'
    REVERSING = 'REVERSING'


class EventRisk(Enum):
    CLEAR = 'CLEAR'
    ELEVATED = 'ELEVATED'
    HIGH = 'HIGH'
    EXTREME = 'EXTREME'


class AdversarialRisk(Enum):
    LOW = 'LOW'
    MEDIUM = 'MEDIUM'
    HIGH = 'HIGH'
    EXTREME = 'EXTREME'


class FeatureGroup(Enum):
    """Canonical feature group names for rationale[]."""
    VOLATILITY_SPIKE = 'VOLATILITY_SPIKE'
    TREND_STRENGTH = 'TREND_STRENGTH'
    MOMENTUM_SHIFT = 'MOMENTUM_SHIFT'
    SUPPORT_RESISTANCE = 'SUPPORT_RESISTANCE'
    MARKET_BREADTH = 'MARKET_BREADTH'
    SENTIMENT_EXTREME = 'SENTIMENT_EXTREME'
    REGIME_ALIGNMENT = 'REGIME_ALIGNMENT'


# =============================================================================
# Regime State (5-axis classification)
# =============================================================================

@dataclass
class RegimeState:
    """5-axis regime classification output."""
    volatility: VolRegime
    trend: TrendRegime
    risk_appetite: RiskAppetite
    momentum: MomentumRegime
    event_risk: EventRisk
    composite_score: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'volatility': self.volatility.value,
            'trend': self.trend.value,
            'risk_appetite': self.risk_appetite.value,
            'momentum': self.momentum.value,
            'event_risk': self.event_risk.value,
            'composite_score': self.composite_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegimeState':
        return cls(
            volatility=VolRegime(data['volatility']),
            trend=TrendRegime(data['trend']),
            risk_appetite=RiskAppetite(data['risk_appetite']),
            momentum=MomentumRegime(data['momentum']),
            event_risk=EventRisk(data['event_risk']),
            composite_score=data['composite_score']
        )


# =============================================================================
# Layer 1: AI Bias Engine Output
# =============================================================================

@dataclass
class BiasOutput:
    """Core AI function output."""
    direction: Direction
    magnitude: Magnitude
    confidence: float  # 0.0 to 1.0
    regime_override: bool
    rationale: List[str]  # feature GROUP names only (from FeatureGroup enum)
    model_version: str
    feature_snapshot: Dict[str, Any]  # 3-component snapshot
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'direction': self.direction.value,
            'magnitude': self.magnitude.value,
            'confidence': self.confidence,
            'regime_override': self.regime_override,
            'rationale': self.rationale,
            'model_version': self.model_version,
            'feature_snapshot': self.feature_snapshot,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiasOutput':
        return cls(
            direction=Direction(data['direction']),
            magnitude=Magnitude(data['magnitude']),
            confidence=data['confidence'],
            regime_override=data['regime_override'],
            rationale=data['rationale'],
            model_version=data['model_version'],
            feature_snapshot=data['feature_snapshot'],
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None
        )


# =============================================================================
# Layer 2: Quant Risk Model Output
# =============================================================================

@dataclass
class RiskOutput:
    """Risk structure output from Layer 2."""
    position_size: float
    kelly_fraction: float
    stop_price: float
    stop_method: str  # 'atr' | 'structural' | 'ict_ob'
    tp1_price: float
    tp2_price: float
    trail_config: Dict[str, Any]
    expected_value: float
    ev_positive: bool
    size_breakdown: Dict[str, Any]  # all multipliers applied
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'position_size': self.position_size,
            'kelly_fraction': self.kelly_fraction,
            'stop_price': self.stop_price,
            'stop_method': self.stop_method,
            'tp1_price': self.tp1_price,
            'tp2_price': self.tp2_price,
            'trail_config': self.trail_config,
            'expected_value': self.expected_value,
            'ev_positive': self.ev_positive,
            'size_breakdown': self.size_breakdown,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskOutput':
        return cls(
            position_size=data['position_size'],
            kelly_fraction=data['kelly_fraction'],
            stop_price=data['stop_price'],
            stop_method=data['stop_method'],
            tp1_price=data['tp1_price'],
            tp2_price=data['tp2_price'],
            trail_config=data['trail_config'],
            expected_value=data['expected_value'],
            ev_positive=data['ev_positive'],
            size_breakdown=data['size_breakdown'],
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None
        )


# =============================================================================
# Layer 3: Game-Theoretic Engine Output
# =============================================================================

@dataclass
class LiquidityPool:
    """A detected liquidity pool."""
    price: float
    strength: int  # count of swing points within tolerance
    swept: bool
    age_bars: int
    draw_probability: float  # 0.0 to 1.0
    pool_type: str  # 'equal_highs' | 'equal_lows'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'price': self.price,
            'strength': self.strength,
            'swept': self.swept,
            'age_bars': self.age_bars,
            'draw_probability': self.draw_probability,
            'pool_type': self.pool_type
        }


@dataclass
class TrappedPositions:
    """Trapped position estimation."""
    trapped_longs: List[Dict[str, Any]]
    trapped_shorts: List[Dict[str, Any]]
    total_long_pain: float
    total_short_pain: float
    squeeze_probability: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trapped_longs': self.trapped_longs,
            'trapped_shorts': self.trapped_shorts,
            'total_long_pain': self.total_long_pain,
            'total_short_pain': self.total_short_pain,
            'squeeze_probability': self.squeeze_probability
        }


@dataclass
class NashZone:
    """Nash equilibrium zone."""
    price_level: float
    zone_type: str  # 'hvn' | 'sr' | 'round_number'
    state: str  # 'HOLDING' | 'TESTED' | 'BREAKING'
    test_count: int
    conviction: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'price_level': self.price_level,
            'zone_type': self.zone_type,
            'state': self.state,
            'test_count': self.test_count,
            'conviction': self.conviction
        }


@dataclass
class GameOutput:
    """Composite output from Layer 3 models."""
    liquidity_map: Dict[str, List[LiquidityPool]]
    nearest_unswept_pool: Optional[LiquidityPool]
    trapped_positions: TrappedPositions
    forced_move_probability: float
    nash_zones: List[NashZone]
    kyle_lambda: float
    game_state_aligned: bool
    game_state_summary: str
    adversarial_risk: AdversarialRisk
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'liquidity_map': {
                'equal_highs': [p.to_dict() for p in self.liquidity_map.get('equal_highs', [])],
                'equal_lows': [p.to_dict() for p in self.liquidity_map.get('equal_lows', [])]
            },
            'nearest_unswept_pool': self.nearest_unswept_pool.to_dict() if self.nearest_unswept_pool else None,
            'trapped_positions': self.trapped_positions.to_dict(),
            'forced_move_probability': self.forced_move_probability,
            'nash_zones': [z.to_dict() for z in self.nash_zones],
            'kyle_lambda': self.kyle_lambda,
            'game_state_aligned': self.game_state_aligned,
            'game_state_summary': self.game_state_summary,
            'adversarial_risk': self.adversarial_risk.value,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameOutput':
        return cls(
            liquidity_map={
                'equal_highs': [LiquidityPool(**p) for p in data['liquidity_map'].get('equal_highs', [])],
                'equal_lows': [LiquidityPool(**p) for p in data['liquidity_map'].get('equal_lows', [])]
            },
            nearest_unswept_pool=LiquidityPool(**data['nearest_unswept_pool']) if data.get('nearest_unswept_pool') else None,
            trapped_positions=TrappedPositions(**data['trapped_positions']),
            forced_move_probability=data['forced_move_probability'],
            nash_zones=[NashZone(**z) for z in data.get('nash_zones', [])],
            kyle_lambda=data['kyle_lambda'],
            game_state_aligned=data['game_state_aligned'],
            game_state_summary=data['game_state_summary'],
            adversarial_risk=AdversarialRisk(data['adversarial_risk']),
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None
        )


# =============================================================================
# Three Layer Context
# =============================================================================

@dataclass
class ThreeLayerContext:
    """Aggregated context from all three layers."""
    bias: BiasOutput
    risk: RiskOutput
    game: GameOutput
    regime: RegimeState
    
    def all_aligned(self) -> bool:
        """Three-layer agreement gate.
        
        Entry signal is only generated when:
        - Layer 1 bias direction != NEUTRAL
        - Layer 1 confidence >= 0.55
        - Layer 2 risk structure has positive EV
        - Layer 3 game state is NOT adversarial with EXTREME risk
        """
        return (
            self.bias.direction != Direction.NEUTRAL
            and self.bias.confidence >= 0.55
            and self.risk.ev_positive
            and not (not self.game.game_state_aligned and self.game.adversarial_risk == AdversarialRisk.EXTREME)
        )
    
    def block_reason(self) -> Optional[str]:
        """Return reason if not aligned."""
        if self.bias.direction == Direction.NEUTRAL:
            return "BIAS_NEUTRAL"
        if self.bias.confidence < 0.55:
            return "CONFIDENCE_TOO_LOW"
        if not self.risk.ev_positive:
            return "EV_NEGATIVE"
        if not self.game.game_state_aligned and self.game.adversarial_risk == AdversarialRisk.EXTREME:
            return "LAYER3_VETO"
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'bias': self.bias.to_dict(),
            'risk': self.risk.to_dict(),
            'game': self.game.to_dict(),
            'regime': self.regime.to_dict(),
            'aligned': self.all_aligned(),
            'block_reason': self.block_reason()
        }


# =============================================================================
# Data Layer Types
# =============================================================================

@dataclass
class FeatureRecord:
    """Complete feature record for a symbol at a timestamp."""
    symbol: str
    timestamp: datetime
    timeframe: str
    features: Dict[str, float]  # all 43 features
    raw_data: Dict[str, Any]  # OHLCV, etc.
    is_valid: bool
    validation_errors: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'timeframe': self.timeframe,
            'features': self.features,
            'raw_data': self.raw_data,
            'is_valid': self.is_valid,
            'validation_errors': self.validation_errors,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureRecord':
        return cls(
            symbol=data['symbol'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            timeframe=data['timeframe'],
            features=data['features'],
            raw_data=data['raw_data'],
            is_valid=data['is_valid'],
            validation_errors=data.get('validation_errors', []),
            metadata=data.get('metadata', {})
        )


@dataclass
class FeatureSnapshot:
    """3-component feature snapshot for bias output."""
    raw_features: Dict[str, float]
    feature_group_tags: Dict[str, Any]
    regime_at_inference: RegimeState
    inference_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'raw_features': self.raw_features,
            'feature_group_tags': self.feature_group_tags,
            'regime_at_inference': self.regime_at_inference.to_dict(),
            'inference_timestamp': self.inference_timestamp
        }


# =============================================================================
# Execution Types
# =============================================================================

@dataclass
class EntrySignal:
    """Entry signal from entry engine."""
    symbol: str
    direction: Direction
    entry_price: float
    position_size: float
    stop_loss: float
    tp1: float
    tp2: float
    confidence: float
    rationale: List[str]
    timestamp: datetime
    layer_context: ThreeLayerContext
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'direction': self.direction.value,
            'entry_price': self.entry_price,
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'tp1': self.tp1,
            'tp2': self.tp2,
            'confidence': self.confidence,
            'rationale': self.rationale,
            'timestamp': self.timestamp.isoformat(),
            'layer_context': self.layer_context.to_dict()
        }


@dataclass
class PositionState:
    """Current position state."""
    trade_id: str
    symbol: str
    direction: Direction
    entry_price: float
    position_size: float
    stop_loss: float
    tp1: float
    tp2: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    status: str  # 'OPEN' | 'CLOSED'
    opened_at: datetime
    closed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'direction': self.direction.value,
            'entry_price': self.entry_price,
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'tp1': self.tp1,
            'tp2': self.tp2,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'status': self.status,
            'opened_at': self.opened_at.isoformat(),
            'closed_at': self.closed_at.isoformat() if self.closed_at else None
        }


@dataclass
class AccountState:
    """Account state for risk calculations."""
    account_id: str
    equity: float
    balance: float
    open_positions: int
    daily_pnl: float
    daily_loss_pct: float
    margin_used: float
    margin_available: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'account_id': self.account_id,
            'equity': self.equity,
            'balance': self.balance,
            'open_positions': self.open_positions,
            'daily_pnl': self.daily_pnl,
            'daily_loss_pct': self.daily_loss_pct,
            'margin_used': self.margin_used,
            'margin_available': self.margin_available,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class MarketData:
    """Market data snapshot."""
    symbol: str
    current_price: float
    bid: float
    ask: float
    spread: float
    volume_24h: float
    atr_14: float
    timestamp: datetime
    ohlcv_1m: Optional[Any] = None
    ohlcv_5m: Optional[Any] = None
    ohlcv_15m: Optional[Any] = None
    ohlcv_1h: Optional[Any] = None
    ohlcv_4h: Optional[Any] = None
    ohlcv_daily: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'bid': self.bid,
            'ask': self.ask,
            'spread': self.spread,
            'volume_24h': self.volume_24h,
            'atr_14': self.atr_14,
            'timestamp': self.timestamp.isoformat()
        }
