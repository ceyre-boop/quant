"""
Feature Generator for XGBoost Training (Warp Drive 2.0)
Optimized for high-speed parallel generation and Parquet IO.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os
from pathlib import Path
from joblib import Parallel, delayed
import logging

# Load env vars first
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.alpaca_client import AlpacaDataClient

logger = logging.getLogger(__name__)

class FeatureGenerator:
    """High-performance feature generation for Sovereign Intelligence."""
    
    def __init__(self, n_jobs: int = -1):
        self.client = AlpacaDataClient()
        self.n_jobs = n_jobs if n_jobs != -1 else os.cpu_count()
        logger.info(f"Warp Drive 2.0 initialized with {self.n_jobs} cores.")
        
    def generate_features(self, df: pd.DataFrame, all_market_data: Optional[Dict] = None, min_bars: int = 30) -> pd.DataFrame:
        """Vectorized indicator calculation for a single asset."""
        if df.empty or len(df) < min_bars:
            return df
            
        df = df.copy()
        
        # 1. PRICE MOMENTUM (Vectorized)
        df['returns_1d'] = df['close'].pct_change()
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_20d'] = df['close'].pct_change(20)
        
        # 2. MOVING AVERAGES
        df['sma_20'] = df['close'].rolling(20, min_periods=10).mean()
        df['sma_50'] = df['close'].rolling(50, min_periods=25).mean()
        df['dist_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['dist_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        # 3. VOLATILITY (ATR normalized)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        df['tr'] = np.max([high_low, high_close, low_close], axis=0)
        df['atr_14'] = df['tr'].rolling(14, min_periods=7).mean()
        df['atr_pct'] = df['atr_14'] / df['close']
        
        # 4. RSI (Optimized)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=7).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 5. CROSS-ASSET CORRELATION (If provided)
        if all_market_data:
            if 'SPY' in all_market_data:
                spy_ret = all_market_data['SPY']['close'].pct_change()
                df['relative_strength'] = df['returns_1d'] - spy_ret.reindex(df.index)
            if 'VIXY' in all_market_data:
                df['vix_level'] = all_market_data['VIXY']['close'].reindex(df.index)

        # Cleanup intermediate
        df.drop(columns=['tr', 'sma_20', 'sma_50'], inplace=True, errors='ignore')
        return df

    def _process_symbol(self, symbol: str, df: pd.DataFrame, all_market_data: Dict) -> pd.DataFrame:
        """Parallel helper for symbol processing."""
        try:
            feat_df = self.generate_features(df, all_market_data)
            feat_df['symbol'] = symbol
            
            # Label generation
            feat_df['next_day_return'] = feat_df['close'].shift(-1) / feat_df['close'] - 1
            feat_df['target'] = (feat_df['next_day_return'] > 0.001).astype(int) # 10bps min move
            return feat_df
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return pd.DataFrame()

    def build_training_dataset(self, timeframe: str = '1Day', days: int = 365) -> Tuple[pd.DataFrame, List[str]]:
        """Multi-process harvesting of the entire universe."""
        logger.info(f"Warp Drive: Harvesting {len(self.client.ALL_SYMBOLS)} assets...")
        
        # Fetching is still IO bound - standard client use for now
        all_data = self.client.get_all_assets(timeframe=timeframe, days=days)
        
        logger.info(f"Warp Drive: Parallelizing feature generation across {self.n_jobs} cores...")
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_symbol)(symbol, df, all_data) 
            for symbol, df in all_data.items() if not df.empty and len(df) >= 50
        )
        
        # Fast Concatenation
        combined = pd.concat(results, ignore_index=True)
        
        # Define feature columns
        exclude = ['symbol', 'target', 'next_day_return', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in combined.columns if c not in exclude]
        
        # Clean Dataset
        combined.dropna(subset=feature_cols + ['target'], inplace=True)
        
        logger.info(f"Dataset Build Complete: {len(combined)} samples | {len(feature_cols)} features.")
        return combined, feature_cols

    def save_dataset(self, df: pd.DataFrame, filename: str = 'master_dataset.parquet'):
        """Fast Parquet export."""
        path = Path('data/processed') / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine='pyarrow', compression='snappy')
        logger.info(f"Dataset successfully persisted to {path} (Parquet)")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    gen = FeatureGenerator()
    df, features = gen.build_training_dataset(timeframe='1Day', days=252) # 1 year
    gen.save_dataset(df)
    print(f"Features: {features}")
    print(df.head())
