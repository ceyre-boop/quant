"""
Feature Generator for XGBoost Training
Generates features for all 46 assets from Alpaca data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Load env vars first
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.alpaca_client import AlpacaDataClient


class FeatureGenerator:
    """Generate technical and cross-asset features for ML training"""
    
    def __init__(self):
        self.client = AlpacaDataClient()
        
    def generate_features(self, df: pd.DataFrame, all_market_data: Optional[Dict] = None, min_bars: int = 30) -> pd.DataFrame:
        """
        Generate features for a single asset's OHLCV data
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
            all_market_data: Dict of all assets' DataFrames for cross-asset features
            min_bars: Minimum bars required (default 30)
            
        Returns:
            DataFrame with feature columns added
        """
        if df.empty or len(df) < min_bars:
            return df
            
        df = df.copy()
        
        # === TECHNICAL FEATURES ===
        
        # Price-based
        df['returns_1d'] = df['close'].pct_change()
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_20d'] = df['close'].pct_change(20)
        
        # Moving averages (as % distance from price)
        df['sma_20'] = df['close'].rolling(20, min_periods=10).mean()
        df['sma_50'] = df['close'].rolling(50, min_periods=25).mean()
        if len(df) >= 100:
            df['sma_200'] = df['close'].rolling(200, min_periods=50).mean()
            df['dist_sma_200'] = (df['close'] - df['sma_200']) / df['sma_200']
        else:
            df['dist_sma_200'] = 0  # Neutral if insufficient data
        df['dist_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['dist_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        # EMA
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['dist_ema_12'] = (df['close'] - df['ema_12']) / df['ema_12']
        df['dist_ema_26'] = (df['close'] - df['ema_26']) / df['ema_26']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=7).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_5'] = df['rsi_14'].rolling(5, min_periods=3).mean()
        
        # MACD
        df['macd_line'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20, min_periods=10).mean()
        df['bb_std'] = df['close'].rolling(20, min_periods=10).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        bb_range = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = np.where(bb_range > 0, (df['close'] - df['bb_lower']) / bb_range, 0.5)
        df['bb_width'] = np.where(df['bb_middle'] > 0, (df['bb_upper'] - df['bb_lower']) / df['bb_middle'], 0)
        
        # ATR (normalized)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr_14'] = true_range.rolling(14, min_periods=7).mean()
        df['atr_pct'] = df['atr_14'] / df['close']
        
        # Volume
        df['volume_sma_20'] = df['volume'].rolling(20, min_periods=10).mean()
        df['volume_ratio'] = np.where(df['volume_sma_20'] > 0, df['volume'] / df['volume_sma_20'], 1.0)
        
        # Daily range
        df['daily_range_pct'] = (df['high'] - df['low']) / df['close']
        
        # Gap
        df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Price position in daily range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # === CROSS-ASSET FEATURES ===
        if all_market_data and 'SPY' in all_market_data:
            spy_df = all_market_data['SPY']
            df['spy_return_1d'] = spy_df['close'].pct_change().reindex(df.index)
            df['spy_return_5d'] = spy_df['close'].pct_change(5).reindex(df.index)
            
        if all_market_data and 'VIXY' in all_market_data:
            vixy_df = all_market_data['VIXY']
            df['vixy_level'] = vixy_df['close'].reindex(df.index)
            df['vixy_change'] = vixy_df['close'].pct_change().reindex(df.index)
            
        # Inverse ETF pressure (SQQQ/SPXU)
        if all_market_data and 'SQQQ' in all_market_data:
            sqqq_df = all_market_data['SQQQ']
            df['sqqq_return'] = sqqq_df['close'].pct_change().reindex(df.index)
            
        if all_market_data and 'SPXU' in all_market_data:
            spxu_df = all_market_data['SPXU']
            df['spxu_return'] = spxu_df['close'].pct_change().reindex(df.index)
        
        # Sector relative strength vs SPY
        if 'spy_return_1d' in df.columns:
            df['relative_strength'] = df['returns_1d'] - df['spy_return_1d']
        
        # Risk-on vs risk-off (GLD/TLT ratio)
        if all_market_data and 'GLD' in all_market_data and 'TLT' in all_market_data:
            gld_df = all_market_data['GLD']
            tlt_df = all_market_data['TLT']
            df['gld_tlt_ratio'] = (gld_df['close'] / tlt_df['close']).reindex(df.index)
        
        # Drop intermediate columns
        cols_to_drop = ['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26', 
                       'bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 'volume_sma_20']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        return df
    
    def build_training_dataset(self, timeframe: str = '1D', days: int = 365) -> pd.DataFrame:
        """
        Build complete training dataset for all assets
        
        Returns:
            DataFrame with all features and labels
        """
        print(f"[FeatureGen] Fetching data for {len(self.client.ALL_SYMBOLS)} assets...")
        all_data = self.client.get_all_assets(timeframe=timeframe, days=days)
        
        print(f"[FeatureGen] Generating features...")
        all_features = []
        
        for symbol, df in all_data.items():
            if df.empty or len(df) < 50:
                continue
                
            # Generate features
            features_df = self.generate_features(df, all_data)
            features_df['symbol'] = symbol
            
            # Generate label (next day return direction)
            features_df['next_day_return'] = features_df['close'].shift(-1) / features_df['close'] - 1
            features_df['target'] = (features_df['next_day_return'] > 0).astype(int)
            
            all_features.append(features_df)
        
        # Combine all
        combined = pd.concat(all_features, ignore_index=True)
        
        # Drop rows with NaN in features or target
        feature_cols = [c for c in combined.columns if c not in ['symbol', 'target', 'next_day_return', 'open', 'high', 'low', 'close', 'volume']]
        combined = combined.dropna(subset=feature_cols + ['target'])
        
        print(f"[FeatureGen] Training dataset: {len(combined)} samples x {len(feature_cols)} features")
        print(f"[FeatureGen] Symbols: {combined['symbol'].nunique()}")
        print(f"[FeatureGen] Target distribution: {combined['target'].value_counts().to_dict()}")
        
        return combined, feature_cols


if __name__ == '__main__':
    gen = FeatureGenerator()
    df, features = gen.build_training_dataset(timeframe='1D', days=180)
    print(f"\nFeature columns: {features}")
    print(f"\nSample data:")
    print(df[['symbol', 'close'] + features[:5] + ['target']].head(10))
