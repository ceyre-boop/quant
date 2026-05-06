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
    grade: str = "C" # Default
    score: float = 0.0
    
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
            'layer_context': self.layer_context.to_dict(),
            'grade': self.grade,
            'score': self.score
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

# =============================================================================
# Phase 2: Sovereign Features Dataclasses
# =============================================================================

@dataclass
class RegimeFeatures:
    """Output of sovereign/features/regime/ — all three detectors combined."""
    hurst_short:         float    # short-window Hurst exponent
    hurst_long:          float    # long-window Hurst exponent
    hurst_signal:        str      # 'MEAN_REVERT' | 'NEUTRAL' | 'TRENDING'
    csd_score:           float    # Critical Slowing Down score
    csd_signal:          str      # 'EXHAUSTION' | 'NEUTRAL' | 'BUILDING'
    hmm_state:           int      # 0, 1, or 2
    hmm_state_label:     str      # 'LOW_VOL' | 'NORMAL' | 'HIGH_VOL_CRISIS'
    hmm_confidence:      float
    hmm_transition_prob: float    # P(regime change in next N bars)
    adx:                 float
    adx_signal:          str      # 'NO_TREND' | 'WEAK' | 'ESTABLISHED' | 'STRONG'


@dataclass
class MomentumFeatures:
    """Output of sovereign/features/momentum/"""
    logistic_ode_score:  float    # institutional accumulation ODE
    jt_momentum_12_1:    float    # Jegadeesh-Titman 12-1 factor
    volume_entropy:      float    # volume disorder measure
    rsi_14:              float
    rsi_signal:          str      # 'OVERSOLD' | 'NEUTRAL' | 'OVERBOUGHT'


@dataclass
class MacroFeatures:
    """Output of sovereign/features/macro/"""
    yield_curve_slope:   float
    yield_curve_velocity: float
    erp:                 float    # equity risk premium
    cape_zscore:         float    # NaN if data unavailable
    cot_zscore:          float    # NaN if data unavailable
    m2_velocity:         float
    hyg_spread_bps:      float
    macro_signal:        str      # 'RISK_ON' | 'NEUTRAL' | 'RISK_OFF' | 'FAULT'


@dataclass
class PetrolausDecision:
    """Output of sovereign/kimi/fault_detector.py"""
    fault_detected:      bool
    fault_reason:        Optional[str]   # None if no fault
    fault_frameworks:    List[str]       # which of the 6 frameworks triggered
    action:              str             # 'TRADE' | 'HALT' | 'CONCENTRATE'
    macro_features:      MacroFeatures


@dataclass
class SovereignFeatureRecord:
    """Complete feature record for one symbol at one timestamp."""
    symbol:              str
    timestamp:           str
    regime:              RegimeFeatures
    momentum:            MomentumFeatures
    macro:               MacroFeatures
    petroulas:           Optional[PetrolausDecision]
    bar_ohlcv:           Dict[str, Any]  # raw OHLCV for the current bar
    is_valid:            bool
    validation_errors:   List[str]


@dataclass
class RouterOutput:
    """Output of sovereign/router/regime_router.py"""
    symbol:              str
    timestamp:           str
    regime:              str             # 'MOMENTUM' | 'REVERSION' | 'FLAT'
    regime_confidence:   float
    specialist_to_run:   Optional[str]   # 'momentum' | 'reversion' | None
    feature_record:      SovereignFeatureRecord
    router_version:      str


@dataclass
class VetoRecord:
    """Entry in the veto ledger."""
    timestamp:           str
    symbol:              str
    veto_stage:          str             # 'PETROULAS' | 'ROUTER' | 'SPECIALIST' | 'RISK' | 'GAME' | 'HARD_CONSTRAINT'
    veto_reason:         str
    signal_that_was_vetoed: Optional[Dict[str, Any]]


# =============================================================================
# PresentState — The Six-Dimension Unified View
# =============================================================================

@dataclass
class PriceRegimeState:
    """Dimension 1: What price is doing right now."""
    hurst_short:         float   # short-window Hurst (< 0.45 = reversion, > 0.55 = trend)
    hurst_long:          float
    hurst_signal:        str     # 'TRENDING' | 'MEAN_REVERT' | 'NEUTRAL'
    hmm_state:           int     # 0 = low-vol/trend, 1 = high-vol/ranging
    hmm_state_label:     str
    hmm_transition_prob: float   # P(regime flip next bar)
    adx:                 float
    adx_signal:          str     # 'NO_TREND' | 'WEAK' | 'ESTABLISHED' | 'STRONG'
    regime:              str     # Router output: 'MOMENTUM' | 'REVERSION' | 'FLAT'
    regime_confidence:   float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hurst_short': self.hurst_short, 'hurst_long': self.hurst_long,
            'hurst_signal': self.hurst_signal, 'hmm_state': self.hmm_state,
            'hmm_state_label': self.hmm_state_label,
            'hmm_transition_prob': self.hmm_transition_prob,
            'adx': self.adx, 'adx_signal': self.adx_signal,
            'regime': self.regime, 'regime_confidence': self.regime_confidence,
        }


@dataclass
class MacroRegimeState:
    """Dimension 2: Why price is doing it (rates, inflation, CB policy)."""
    yield_curve_slope:   float
    yield_curve_velocity: float
    erp:                 float   # equity risk premium
    cape_zscore:         float
    cot_zscore:          float
    m2_velocity:         float
    hyg_spread_bps:      float
    macro_signal:        str     # 'RISK_ON' | 'NEUTRAL' | 'RISK_OFF' | 'FAULT'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'yield_curve_slope': self.yield_curve_slope,
            'yield_curve_velocity': self.yield_curve_velocity,
            'erp': self.erp, 'cape_zscore': self.cape_zscore,
            'cot_zscore': self.cot_zscore, 'm2_velocity': self.m2_velocity,
            'hyg_spread_bps': self.hyg_spread_bps, 'macro_signal': self.macro_signal,
        }


@dataclass
class PositioningState:
    """Dimension 3: Who is positioned how (COT, options skew)."""
    cot_zscore:          float   # net commercial positioning z-score (3yr window)
    cot_symbol:          str     # underlying futures symbol used ('NQ', 'ES', …)
    positioning_bias:    str     # 'LONG_HEAVY' | 'SHORT_HEAVY' | 'NEUTRAL'
    source:              str     # 'CFTC' | 'macro_features' | 'none'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cot_zscore': self.cot_zscore,
            'cot_symbol': self.cot_symbol,
            'positioning_bias': self.positioning_bias,
            'source': self.source,
        }


@dataclass
class NarrativeState:
    """Dimension 4: What the market believes (TradingAgents / Qwen3 / LLM layer).

    Stub until TradingAgents integration is wired.  Source = 'none' means no
    narrative signal is active and gates should ignore this dimension.
    """
    summary:             str     # Free-text narrative from LLM layer
    sentiment:           str     # 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    confidence:          float   # 0..1
    source:              str     # 'TradingAgents' | 'Qwen3' | 'none'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'summary': self.summary, 'sentiment': self.sentiment,
            'confidence': self.confidence, 'source': self.source,
        }


@dataclass
class HistoricalMatchState:
    """Dimension 5: What this has looked like before (Alexandrian Library).

    Stub until the Library is wired.  Source = 'none' means no match is active.
    """
    regime_label:        str     # e.g. 'ASIAN_CURRENCY_CONTAGION'
    similarity_score:    float   # 0..1  (0.927 = very close match)
    volumes_converging:  int     # volumes agreeing (out of 10)
    typical_outcome:     str     # historical tendency description
    lookback_period_days: int    # how many days back the match spans
    source:              str     # 'AlexandrianLibrary' | 'none'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime_label': self.regime_label,
            'similarity_score': self.similarity_score,
            'volumes_converging': self.volumes_converging,
            'typical_outcome': self.typical_outcome,
            'lookback_period_days': self.lookback_period_days,
            'source': self.source,
        }


@dataclass
class CatalystWindowState:
    """Dimension 6: When resolution is expected (OU model + CB calendar)."""
    ou_half_life_days:       float   # physics-derived reversion half-life
    ou_reversion_days:       float   # expected days to mean reversion
    ou_confidence:           str     # 'HIGH' | 'MEDIUM' | 'LOW'
    next_cb_event_bank:      Optional[str]   # e.g. 'FED'
    next_cb_event_days:      int             # calendar days until next CB meeting
    cb_blackout_active:      bool            # within 10-day blackout window
    catalyst_urgency:        str     # 'IMMINENT' (<7d) | 'NEAR' (<21d) | 'DISTANT' | 'NONE'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ou_half_life_days': self.ou_half_life_days,
            'ou_reversion_days': self.ou_reversion_days,
            'ou_confidence': self.ou_confidence,
            'next_cb_event_bank': self.next_cb_event_bank,
            'next_cb_event_days': self.next_cb_event_days,
            'cb_blackout_active': self.cb_blackout_active,
            'catalyst_urgency': self.catalyst_urgency,
        }


@dataclass
class PresentState:
    """The unified six-dimension snapshot the orchestrator reads before every decision.

    Not a gate — a shared substrate.  All existing gates already draw from pieces
    of this.  PresentState makes those pieces coherent, logged, and consistent.

    Alignment score counts how many of the 6 active dimensions point the same
    direction as the current trade bias.  Higher = more constrained future.
    """
    symbol:           str
    timestamp:        str
    price_regime:     PriceRegimeState
    macro_regime:     MacroRegimeState
    positioning:      PositioningState
    narrative:        NarrativeState
    historical_match: HistoricalMatchState
    catalyst_window:  CatalystWindowState

    # Derived coherence metrics
    dimensions_active:  int    # how many dimensions have live data (source != 'none')
    alignment_score:    float  # 0..1 (fraction of active dimensions aligned with trade bias)
    alignment_label:    str    # 'FULL' | 'STRONG' | 'PARTIAL' | 'WEAK' | 'NONE'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'price_regime': self.price_regime.to_dict(),
            'macro_regime': self.macro_regime.to_dict(),
            'positioning': self.positioning.to_dict(),
            'narrative': self.narrative.to_dict(),
            'historical_match': self.historical_match.to_dict(),
            'catalyst_window': self.catalyst_window.to_dict(),
            'dimensions_active': self.dimensions_active,
            'alignment_score': self.alignment_score,
            'alignment_label': self.alignment_label,
        }

    def summary(self) -> str:
        """One-line human-readable summary of the present state."""
        cb_str = (f"{self.catalyst_window.next_cb_event_bank}+{self.catalyst_window.next_cb_event_days}d"
                  if self.catalyst_window.next_cb_event_bank else 'no-CB')
        return (
            f"PRESENT[{self.symbol}] "
            f"regime={self.price_regime.regime}({self.price_regime.hurst_signal}) "
            f"macro={self.macro_regime.macro_signal} "
            f"pos={self.positioning.positioning_bias}(z={self.positioning.cot_zscore:.2f}) "
            f"narrative={self.narrative.sentiment}({self.narrative.source}) "
            f"match={self.historical_match.regime_label}({self.historical_match.similarity_score:.3f}) "
            f"catalyst={self.catalyst_window.catalyst_urgency}({cb_str}) "
            f"→ alignment={self.alignment_label}({self.alignment_score:.2f})"
        )
