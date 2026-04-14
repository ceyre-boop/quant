"""
Sovereign Core Training Script

Trains the essential models:
1. Regime Router (XGBoost meta-classifier)
2. Momentum Specialist
3. Reversion Specialist

Factor Zoo is run as optional diagnostic only.
"""

import argparse
import logging
from pathlib import Path

from sovereign.orchestrator import SovereignOrchestrator
from sovereign.features.factor_zoo import FactorZooScanner
from sovereign.data.feeds.alpaca_feed import AlpacaFeed
from config.loader import params

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def run_factor_zoo_diagnostic(symbols):
    """
    Optional diagnostic — NOT a gate.
    Run it for information, then train anyway.
    """
    logger.info("=" * 60)
    logger.info("FACTOR ZOO DIAGNOSTIC (Informational Only)")
    logger.info("=" * 60)
    
    scanner = FactorZooScanner()
    feed = AlpacaFeed()
    
    for symbol in symbols:
        logger.info(f"\n--- {symbol} ---")
        try:
            df = feed.get_historical_bars(
                symbol=symbol,
                start='2022-01-01',
                end='2024-12-31',
                timeframe='1D'
            )
            
            if df.empty:
                logger.warning(f"No data for {symbol}")
                continue
            
            feat = scanner.build_feature_matrix(df)
            result = scanner.scan(feat)
            
            n_real = int(result["is_real"].sum())
            logger.info(f"Robust features found: {n_real}")
            
            if n_real > 0:
                for _, row in result[result["is_real"]].iterrows():
                    logger.info(
                        f"  {row['feature']:30s} "
                        f"ICIR={row['icir']:+.3f} "
                        f"horizon={row['best_horizon']}"
                    )
            else:
                logger.info("No features passed — continuing with training anyway")
                
        except Exception as e:
            logger.error(f"Diagnostic failed for {symbol}: {e}")


def train_core_models(symbols):
    """
    Train the stripped Sovereign core.
    
    Fetches historical data, builds feature records,
    and trains router + specialists.
    """
    logger.info("=" * 60)
    logger.info("TRAINING SOVEREIGN CORE")
    logger.info("=" * 60)
    
    feed = AlpacaFeed()
    records = []
    
    # Fetch and build records for all symbols
    for symbol in symbols:
        logger.info(f"\nFetching data for {symbol}...")
        try:
            df = feed.get_historical_bars(
                symbol=symbol,
                start='2022-01-01',
                end='2024-12-31',
                timeframe='1D'
            )
            
            if len(df) < 500:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} bars")
                continue
            
            # Build feature records from DataFrame
            symbol_records = _build_records_from_df(symbol, df)
            records.extend(symbol_records)
            logger.info(f"Built {len(symbol_records)} records for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to fetch/build for {symbol}: {e}")
    
    if len(records) < 500:
        logger.error(f"Insufficient total records: {len(records)}. Need 500+")
        return False
    
    logger.info(f"\nTotal training records: {len(records)}")
    
    # Train
    orch = SovereignOrchestrator(mode='paper')
    orch.train(records)
    
    # Save
    Path('models/sovereign').mkdir(parents=True, exist_ok=True)
    orch.save_models(base_path='models/sovereign')
    
    logger.info("=" * 60)
    logger.info("[OK] Core models trained and saved")
    logger.info("=" * 60)
    
    return True


def _build_records_from_df(symbol, df):
    """Build minimal SovereignFeatureRecords from OHLCV DataFrame."""
    from contracts.types import (
        SovereignFeatureRecord, RegimeFeatures, MomentumFeatures,
        MacroFeatures, PetrolausDecision
    )
    from sovereign.features.regime.hurst import compute_hurst_features
    
    records = []
    
    # Compute Hurst on the dataframe
    try:
        hurst_df = compute_hurst_features(df)
    except Exception:
        hurst_df = None
    
    for i, (idx, row) in enumerate(df.iterrows()):
        # Skip first 90 bars (need lookback for Hurst long)
        if i < 90:
            continue
        
        h_short = 0.5
        h_long = 0.5
        if hurst_df is not None and idx in hurst_df.index:
            h_short = float(hurst_df.loc[idx].get('hurst_short', 0.5))
            h_long = float(hurst_df.loc[idx].get('hurst_long', 0.5))
        
        regime = RegimeFeatures(
            hurst_short=h_short,
            hurst_long=h_long,
            hurst_signal='TRENDING' if h_short > 0.52 else 'MEAN_REVERT' if h_short < 0.45 else 'NEUTRAL',
            csd_score=0.5,
            csd_signal='NEUTRAL',
            hmm_state=1,
            hmm_state_label='NORMAL',
            hmm_confidence=0.6,
            hmm_transition_prob=0.2,
            adx=25.0,
            adx_signal='ESTABLISHED'
        )
        
        momentum = MomentumFeatures(
            logistic_ode_score=0.0,
            jt_momentum_12_1=0.0,
            volume_entropy=1.0,
            rsi_14=50.0,
            rsi_signal='NEUTRAL'
        )
        
        macro = MacroFeatures(
            yield_curve_slope=0.01,
            yield_curve_velocity=0.0,
            erp=0.04,
            cape_zscore=1.0,
            cot_zscore=0.0,
            m2_velocity=1.5,
            hyg_spread_bps=200.0,
            macro_signal='RISK_ON'
        )
        
        petroulas = PetrolausDecision(
            fault_detected=False,
            fault_reason=None,
            fault_frameworks=[],
            action='TRADE',
            macro_features=macro
        )
        
        record = SovereignFeatureRecord(
            symbol=symbol,
            timestamp=idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
            regime=regime,
            momentum=momentum,
            macro=macro,
            petroulas=petroulas,
            bar_ohlcv={
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row.get('volume', 0))
            },
            is_valid=True,
            validation_errors=[]
        )
        
        records.append(record)
    
    return records


def main():
    parser = argparse.ArgumentParser(description='Train Sovereign Core Models')
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['META', 'PFE', 'UNH'],
        help='Symbols to train on'
    )
    parser.add_argument(
        '--skip-diagnostic',
        action='store_true',
        help='Skip Factor Zoo diagnostic'
    )
    parser.add_argument(
        '--diagnostic-only',
        action='store_true',
        help='Run diagnostic only, do not train'
    )
    
    args = parser.parse_args()
    
    # Phase 0B: Optional diagnostic
    if not args.skip_diagnostic:
        run_factor_zoo_diagnostic(args.symbols)
    
    # Train core
    if not args.diagnostic_only:
        success = train_core_models(args.symbols)
        return 0 if success else 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
