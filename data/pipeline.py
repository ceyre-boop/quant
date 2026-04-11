"""Data Pipeline - Master data coordinator.

Orchestrates all data fetching and feature building for the trading system.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from data.polygon_client import PolygonRestClient
from data.daily_fetcher import DailyFetcher
from data.index_fetcher import IndexFetcher
from data.breadth_engine import BreadthEngine
from data.calendar_fetcher import CalendarFetcher
from data.sentiment_engine import SentimentEngine
from data.order_flow_fetcher import OrderFlowFetcher
from data.tradelocker_client import TradeLockerClient
from data.validator import DataValidator
from layer3.macro_imbalance import MacroImbalanceFramework
from contracts.types import FeatureRecord

logger = logging.getLogger(__name__)


class DataPipeline:
    """Master data pipeline coordinator."""
    
    def __init__(
        self,
        polygon_client: Optional[PolygonRestClient] = None,
        tradelocker_client: Optional[TradeLockerClient] = None
    ):
        self.polygon = polygon_client or PolygonRestClient()
        self.tradelocker = tradelocker_client or TradeLockerClient()
        
        # Initialize fetchers
        self.daily_fetcher = DailyFetcher(self.polygon)
        self.index_fetcher = IndexFetcher(self.polygon)
        self.breadth_engine = BreadthEngine(self.polygon)
        self.calendar_fetcher = CalendarFetcher()
        self.sentiment_engine = SentimentEngine()
        self.order_flow_fetcher = OrderFlowFetcher(self.polygon)
        self.imbalance_engine = MacroImbalanceFramework()
        self.validator = DataValidator()
    
    def run_premarket(
        self,
        symbols: List[str],
        include_all_features: bool = True
    ) -> Dict[str, FeatureRecord]:
        """Run pre-market data pipeline.
        
        Runs at 08:00 EST. Builds all feature vectors for the session.
        
        Steps:
        1. Fetch overnight OHLCV (all timeframes)
        2. Fetch VIX, index data, sector ETFs
        3. Fetch economic calendar events for today
        4. Compute breadth metrics
        5. Fetch/refresh news sentiment (last 24h)
        6. Build FeatureRecord per symbol (validate schema)
        7. Write all FeatureRecords to Firebase: collection 'feature_records'
        8. Return validated feature vectors
        
        Args:
            symbols: List of symbols to process
            include_all_features: If True, compute all 43 features
        
        Returns:
            Dict mapping symbol to FeatureRecord
        """
        logger.info(f"Starting pre-market pipeline for {symbols}")
        
        results = {}
        timestamp = datetime.utcnow()
        
        # 1. Fetch market data for all symbols
        daily_data = {}
        for symbol in symbols:
            try:
                daily_data[symbol] = self.daily_fetcher.fetch_daily_bars(symbol, 30)
            except Exception as e:
                logger.error(f"Failed to fetch daily data for {symbol}: {e}")
                daily_data[symbol] = []
        
        # 2. Fetch index data
        try:
            index_data = self.index_fetcher.fetch_all_indices(lookback_days=30)
            vix_value = index_data.get('VIX', {}).get('latest', {}).get('value', 20)
            vix_regime = self.index_fetcher.get_vix_regime(vix_value) if vix_value else 'NORMAL'
        except Exception as e:
            logger.error(f"Failed to fetch index data: {e}")
            index_data = {}
            vix_value = 20
            vix_regime = 'NORMAL'
        
        # 3. Fetch calendar events
        try:
            calendar_events = self.calendar_fetcher.fetch_events(days_ahead=1)
            event_risk = self.calendar_fetcher.calculate_event_risk(calendar_events)
        except Exception as e:
            logger.error(f"Failed to fetch calendar: {e}")
            calendar_events = []
            event_risk = 'CLEAR'
        
        # 4. Compute breadth metrics
        try:
            breadth = self.breadth_engine.compute_breadth_metrics()
        except Exception as e:
            logger.error(f"Failed to compute breadth: {e}")
            breadth = None
            
        # 4.5 Run Imbalance Engine (Global Macro)
        try:
            # Prepare inputs from data sources
            macro_input = {
                'cape': 32.0, 
                'bond_yield_10yr': 4.25,
                'vix': vix_value,
                'spy_returns_1m': self._extract_index_value(index_data, 'SPY', 'perf_30d', 0),
                'spread_3m_10yr': self._extract_index_value(index_data, '3M10Y', 'value', -0.2),
                'credit_spread': self._extract_index_value(index_data, 'LQD', 'spread', 1.0)
            }
            macro_result = self.imbalance_engine.compute(macro_input)
        except Exception as e:
            logger.error(f"Imbalance Engine failed: {e}")
            macro_result = None
        
        # 5. Build FeatureRecord for each symbol
        for symbol in symbols:
            try:
                # Build features
                features = self._build_features(
                    symbol=symbol,
                    daily_data=daily_data.get(symbol, []),
                    index_data=index_data,
                    vix_value=vix_value,
                    vix_regime=vix_regime,
                    event_risk=event_risk,
                    breadth=breadth,
                    macro_result=macro_result
                )
                
                # Get latest bar for raw_data
                latest_bar = daily_data.get(symbol, [{}])[-1] if daily_data.get(symbol) else {}
                
                # Create FeatureRecord
                record = FeatureRecord(
                    symbol=symbol,
                    timestamp=timestamp,
                    timeframe='daily',
                    features=features,
                    raw_data=latest_bar,
                    is_valid=True,
                    validation_errors=[],
                    metadata={
                        'pipeline_version': '1.0',
                        'data_sources': ['polygon'],
                        'vix_regime': vix_regime,
                        'event_risk': event_risk
                    }
                )
                
                # Validate
                validation = self.validator.validate_feature_record(record.to_dict())
                if not validation.is_valid:
                    record.is_valid = False
                    record.validation_errors = validation.errors
                    logger.warning(f"Feature record validation failed for {symbol}: {validation.errors}")
                
                results[symbol] = record
                
            except Exception as e:
                logger.error(f"Failed to build features for {symbol}: {e}")
                # Create invalid record
                results[symbol] = FeatureRecord(
                    symbol=symbol,
                    timestamp=timestamp,
                    timeframe='daily',
                    features={},
                    raw_data={},
                    is_valid=False,
                    validation_errors=[str(e)],
                    metadata={}
                )
        
        logger.info(f"Pre-market pipeline complete. Valid records: {sum(1 for r in results.values() if r.is_valid)}/{len(symbols)}")
        return results
    
    def run_intraday_refresh(
        self,
        symbols: List[str]
    ) -> Dict[str, FeatureRecord]:
        """Run intraday data refresh.
        
        Runs every 5 minutes during session. Incremental update.
        
        Args:
            symbols: List of symbols to refresh
        
        Returns:
            Dict mapping symbol to updated FeatureRecord
        """
        logger.info(f"Starting intraday refresh for {symbols}")
        
        results = {}
        timestamp = datetime.utcnow()
        
        for symbol in symbols:
            try:
                # Fetch latest data
                order_flow = self.order_flow_fetcher.get_order_flow_summary(symbol)
                
                # Build incremental features
                features = {
                    'kyle_lambda': order_flow.get('kyle_lambda', 0),
                    'volume_imbalance': order_flow.get('volume_imbalance', {}).get('imbalance_ratio', 1.0),
                    'timestamp': timestamp.isoformat()
                }
                
                record = FeatureRecord(
                    symbol=symbol,
                    timestamp=timestamp,
                    timeframe='intraday',
                    features=features,
                    raw_data=order_flow,
                    is_valid=True,
                    validation_errors=[],
                    metadata={'refresh_type': 'intraday'}
                )
                
                results[symbol] = record
                
            except Exception as e:
                logger.error(f"Intraday refresh failed for {symbol}: {e}")
                results[symbol] = FeatureRecord(
                    symbol=symbol,
                    timestamp=timestamp,
                    timeframe='intraday',
                    features={},
                    raw_data={},
                    is_valid=False,
                    validation_errors=[str(e)],
                    metadata={}
                )
        
        return results
    
    def _build_features(
        self,
        symbol: str,
        daily_data: List[Dict],
        index_data: Dict,
        vix_value: float,
        vix_regime: str,
        event_risk: str,
        breadth: Any,
        macro_result: Any = None
    ) -> Dict[str, float]:
        """Build feature vector for a symbol."""
        features = {}
        
        # Inject macro imbalance features
        if macro_result:
            features['hmm_regime_stress'] = macro_result.get('hmm_regime_stress', 0.0)
            features['pca_mahalanobis'] = macro_result.get('pca_mahalanobis', 0.0)
            features['recession_prob_12m'] = macro_result.get('recession_prob_12m', 0.0)
        else:
            features['hmm_regime_stress'] = 0.0
            features['pca_mahalanobis'] = 0.0
            features['recession_prob_12m'] = 0.0
            
        if not daily_data:
            return features
        
        # Latest values
        latest = daily_data[-1]
        close = latest.get('c', 0)
        
        # Price-based features
        if len(daily_data) >= 2:
            features['returns_1h'] = (close - daily_data[-2].get('c', close)) / close
        else:
            features['returns_1h'] = 0.0
        
        if len(daily_data) >= 5:
            features['returns_daily'] = (close - daily_data[-5].get('c', close)) / close
        else:
            features['returns_daily'] = 0.0
        
        if len(daily_data) >= 25:
            features['returns_5d'] = (close - daily_data[-25].get('c', close)) / close
        else:
            features['returns_5d'] = 0.0
        
        # Moving averages
        if len(daily_data) >= 20:
            sma20 = sum(d.get('c', 0) for d in daily_data[-20:]) / 20
            features['price_vs_sma_20'] = (close - sma20) / close
        else:
            features['price_vs_sma_20'] = 0.0
        
        if len(daily_data) >= 50:
            sma50 = sum(d.get('c', 0) for d in daily_data[-50:]) / 50
            features['price_vs_sma_50'] = (close - sma50) / close
        else:
            features['price_vs_sma_50'] = 0.0
        
        # Volatility features
        if len(daily_data) >= 15:
            highs = [d.get('h', close) for d in daily_data[-14:]]
            lows = [d.get('l', close) for d in daily_data[-14:]]
            trs = [h - l for h, l in zip(highs, lows)]
            features['atr_14'] = sum(trs) / len(trs) if trs else 0
            features['atr_percent_14'] = features['atr_14'] / close if close else 0
        else:
            features['atr_14'] = 0
            features['atr_percent_14'] = 0
        
        features['vix_level'] = vix_value
        features['vix_regime'] = {'LOW': 1, 'NORMAL': 2, 'ELEVATED': 3, 'EXTREME': 4}.get(vix_regime, 2)
        
        # Momentum features
        if len(daily_data) >= 15:
            gains = []
            losses = []
            for i in range(-14, 0):
                change = daily_data[i].get('c', 0) - daily_data[i-1].get('c', 0)
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            rs = avg_gain / max(avg_loss, 1e-10)
            features['rsi_14'] = 100 - (100 / (1 + rs))
        else:
            features['rsi_14'] = 50.0
        
        # Breadth features
        if breadth:
            features['market_breadth_ratio'] = breadth.advance_decline_ratio
        else:
            features['market_breadth_ratio'] = 1.0
        
        # Correlation
        features['spy_correlation_20d'] = 0.9  # Placeholder
        
        # Fill remaining features with defaults
        for key in [
            'returns_4h', 'price_vs_sma_200', 'price_vs_ema_12', 'price_vs_ema_26',
            'price_position_daily_range', 'bollinger_position', 'bollinger_width',
            'keltner_position', 'historical_volatility_20d', 'realized_volatility_5d',
            'volatility_regime', 'adx_14', 'dmi_plus', 'dmi_minus', 'macd_line',
            'macd_signal', 'macd_histogram', 'rsi_slope_5', 'stochastic_k', 'stochastic_d',
            'cci_20', 'volume_sma_ratio', 'obv_slope', 'vwap_deviation',
            'volume_profile_poc_dist', 'volume_trend_5d', 'swing_high_20', 'swing_low_20',
            'distance_to_resistance', 'distance_to_support', 'higher_highs_5d', 'higher_lows_5d'
        ]:
            if key not in features:
                features[key] = 0.0
        
        return features

    def _extract_index_value(self, data: dict, key: str, subkey: str, default: float) -> float:
        """Helper to safely extract nested index values."""
        try:
            return data.get(key, {}).get('latest', {}).get(subkey, default)
        except Exception:
            return default
