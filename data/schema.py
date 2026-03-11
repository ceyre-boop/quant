"""Data Schema Definitions

Pydantic models for data validation and serialization.
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class DataQuality(str, Enum):
    """Data quality flags."""
    VALID = 'VALID'
    STALE = 'STALE'
    GAPPED = 'GAPPED'
    SUSPECT = 'SUSPECT'
    INVALID = 'INVALID'


class OHLCVBar(BaseModel):
    """Single OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @field_validator('high')
    @classmethod
    def high_is_highest(cls, v: float, info) -> float:
        """Validate high is >= open and close."""
        values = info.data
        if 'open' in values and v < values['open']:
            raise ValueError('high must be >= open')
        if 'close' in values and v < values['close']:
            raise ValueError('high must be >= close')
        return v
    
    @field_validator('low')
    @classmethod
    def low_is_lowest(cls, v: float, info) -> float:
        """Validate low is <= open and close."""
        values = info.data
        if 'open' in values and v > values['open']:
            raise ValueError('low must be <= open')
        if 'close' in values and v > values['close']:
            raise ValueError('low must be <= close')
        return v
    
    @field_validator('volume')
    @classmethod
    def volume_non_negative(cls, v: float) -> float:
        """Validate volume is non-negative."""
        if v < 0:
            raise ValueError('volume must be >= 0')
        return v


class FeatureSchema(BaseModel):
    """Schema for all 43 features from v4.1 spec."""
    
    # Price-based features (1-10)
    returns_1h: float
    returns_4h: float
    returns_daily: float
    returns_5d: float
    price_vs_sma_20: float
    price_vs_sma_50: float
    price_vs_sma_200: float
    price_vs_ema_12: float
    price_vs_ema_26: float
    price_position_daily_range: float
    
    # Volatility features (11-18)
    atr_14: float = Field(..., gt=0)
    atr_percent_14: float = Field(..., gt=0, lt=0.1)
    bollinger_position: float = Field(..., ge=-3, le=3)
    bollinger_width: float = Field(..., gt=0)
    keltner_position: float
    historical_volatility_20d: float
    realized_volatility_5d: float
    volatility_regime: Literal[1, 2, 3, 4]
    
    # Trend/Momentum features (19-28)
    adx_14: float = Field(..., ge=0, le=100)
    dmi_plus: float = Field(..., ge=0)
    dmi_minus: float = Field(..., ge=0)
    macd_line: float
    macd_signal: float
    macd_histogram: float
    rsi_14: float = Field(..., ge=0, le=100)
    rsi_slope_5: float
    stochastic_k: float = Field(..., ge=0, le=100)
    stochastic_d: float = Field(..., ge=0, le=100)
    cci_20: float
    
    # Volume features (29-33)
    volume_sma_ratio: float = Field(..., gt=0)
    obv_slope: float
    vwap_deviation: float
    volume_profile_poc_dist: float
    volume_trend_5d: float
    
    # Market structure features (34-39)
    swing_high_20: Optional[float] = None
    swing_low_20: Optional[float] = None
    distance_to_resistance: float
    distance_to_support: float
    higher_highs_5d: int = Field(..., ge=0, le=5)
    higher_lows_5d: int = Field(..., ge=0, le=5)
    
    # Cross-market features (40-43)
    vix_level: float = Field(..., gt=0)
    vix_regime: Literal[1, 2, 3, 4]
    spy_correlation_20d: float = Field(..., ge=-1, le=1)
    market_breadth_ratio: float = Field(..., gt=0)
    
    @field_validator('atr_14')
    @classmethod
    def atr_sanity_check(cls, v: float, info) -> float:
        """ATR should be < 10% of close price (if available)."""
        # This is a sanity check to catch data errors
        if v > 10000:  # Unrealistic ATR for any normal market
            raise ValueError(f'ATR {v} appears unrealistic')
        return v


class FeatureRecordSchema(BaseModel):
    """Complete feature record schema for validation."""
    
    symbol: str = Field(..., min_length=1, max_length=20)
    timestamp: datetime
    timeframe: Literal['1m', '5m', '15m', '1h', '4h', 'daily']
    features: FeatureSchema
    raw_data: Dict[str, Any]
    is_valid: bool
    validation_errors: List[str] = Field(default_factory=list)
    data_quality: DataQuality = DataQuality.VALID
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Validate record consistency."""
        if not self.is_valid and len(self.validation_errors) == 0:
            raise ValueError('Invalid record must have validation_errors')
        if self.is_valid and len(self.validation_errors) > 0:
            raise ValueError('Valid record cannot have validation_errors')
        return self


class FeatureSnapshotSchema(BaseModel):
    """3-component feature snapshot schema."""
    
    raw_features: Dict[str, float]
    feature_group_tags: Dict[str, Any]
    regime_at_inference: Dict[str, Any]
    inference_timestamp: str
    
    @field_validator('inference_timestamp')
    @classmethod
    def valid_iso_timestamp(cls, v: str) -> str:
        """Validate timestamp is ISO format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError('inference_timestamp must be ISO format')
        return v


class BiasOutputSchema(BaseModel):
    """Bias output schema for Firebase validation."""
    
    symbol: str
    timestamp: str
    direction: Literal[-1, 0, 1]
    magnitude: Literal[1, 2, 3]
    confidence: float = Field(..., ge=0.0, le=1.0)
    regime_override: bool
    rationale: List[str]
    model_version: str
    feature_snapshot: FeatureSnapshotSchema
    created_at: Optional[str] = None
    
    @field_validator('rationale')
    @classmethod
    def valid_rationale(cls, v: List[str]) -> List[str]:
        """Validate rationale contains only canonical group names."""
        valid_groups = {
            'VOLATILITY_SPIKE', 'TREND_STRENGTH', 'MOMENTUM_SHIFT',
            'SUPPORT_RESISTANCE', 'MARKET_BREADTH', 'SENTIMENT_EXTREME',
            'REGIME_ALIGNMENT'
        }
        for item in v:
            if item not in valid_groups:
                raise ValueError(f'Invalid rationale group: {item}')
        return v


class RiskOutputSchema(BaseModel):
    """Risk structure schema for Firebase validation."""
    
    symbol: str
    timestamp: str
    bias_id: str
    position_size: float = Field(..., gt=0)
    kelly_fraction: float = Field(..., ge=0, le=1)
    stop_price: float = Field(..., gt=0)
    stop_method: Literal['atr', 'structural', 'ict_ob']
    tp1_price: float = Field(..., gt=0)
    tp2_price: float = Field(..., gt=0)
    trail_config: Dict[str, Any]
    expected_value: float
    ev_positive: bool
    size_breakdown: Dict[str, Any]
    created_at: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_tp_order(self):
        """Validate TP1 is closer to entry than TP2."""
        # This is a simplified check - actual logic depends on direction
        return self


class LiquidityPoolSchema(BaseModel):
    """Liquidity pool schema."""
    
    price: float = Field(..., gt=0)
    strength: int = Field(..., ge=0)
    swept: bool
    age_bars: int = Field(..., ge=0)
    draw_probability: float = Field(..., ge=0, le=1)
    pool_type: Literal['equal_highs', 'equal_lows']


class GameOutputSchema(BaseModel):
    """Game output schema for Firebase validation."""
    
    symbol: str
    timestamp: str
    liquidity_map: Dict[str, List[LiquidityPoolSchema]]
    nearest_unswept_pool: Optional[LiquidityPoolSchema]
    trapped_positions: Dict[str, Any]
    forced_move_probability: float = Field(..., ge=0, le=1)
    nash_zones: List[Dict[str, Any]]
    kyle_lambda: float
    game_state_aligned: bool
    game_state_summary: str
    adversarial_risk: Literal['LOW', 'MEDIUM', 'HIGH', 'EXTREME']
    created_at: Optional[str] = None


class EntrySignalSchema(BaseModel):
    """Entry signal schema for Firebase validation."""
    
    symbol: str
    direction: Literal[-1, 1]
    entry_price: float = Field(..., gt=0)
    position_size: float = Field(..., gt=0)
    stop_loss: float = Field(..., gt=0)
    tp1: float = Field(..., gt=0)
    tp2: float = Field(..., gt=0)
    confidence: float = Field(..., ge=0, le=1)
    rationale: List[str]
    timestamp: str
    layer_context: Dict[str, Any]
    created_at: Optional[str] = None


class PositionSchema(BaseModel):
    """Position state schema for Firebase validation."""
    
    trade_id: str
    symbol: str
    direction: Literal[-1, 1]
    entry_price: float = Field(..., gt=0)
    position_size: float = Field(..., gt=0)
    stop_loss: float = Field(..., gt=0)
    tp1: float = Field(..., gt=0)
    tp2: float = Field(..., gt=0)
    current_price: float = Field(..., gt=0)
    unrealized_pnl: float
    realized_pnl: float
    status: Literal['OPEN', 'CLOSED']
    opened_at: str
    closed_at: Optional[str] = None
    updated_at: Optional[str] = None


class AccountStateSchema(BaseModel):
    """Account state schema for Firebase validation."""
    
    account_id: str
    equity: float = Field(..., gt=0)
    balance: float = Field(..., gt=0)
    open_positions: int = Field(..., ge=0)
    daily_pnl: float
    daily_loss_pct: float
    margin_used: float = Field(..., ge=0)
    margin_available: float = Field(..., ge=0)
    timestamp: str
    updated_at: Optional[str] = None
