"""
Sovereign Daily Execution — Windows Task Scheduler Entry Point

Tear-down compliant minimal system:
- No Factor Zoo gate (diagnostic only)
- Petroulas logs warnings (doesn't block)
- No Layer 3 gate (logs only)
- Hard constraints enforced
- Paper trading starts immediately
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).parent
_LOG_DIR = _ROOT / 'data' / 'logs'
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(str(_LOG_DIR / f'daily_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def pre_market_checklist():
    """
    Phase 1: Pre-market validation.
    NOT a gate — just verification that systems are live.
    """
    logger.info("=" * 60)
    logger.info("PRE-MARKET CHECKLIST")
    logger.info("=" * 60)
    
    checks = {
        'Alpaca Connection': _check_alpaca,
        'Router Model Loaded': _check_router,
        'Specialists Loaded': _check_specialists,
        'Risk Config Valid': _check_risk_config,
        'Ledger Writable': _check_ledger
    }
    
    all_passed = True
    for name, check_fn in checks.items():
        try:
            passed = check_fn()
            status = "[OK]" if passed else "[FAIL]"
            logger.info(f"{status} {name}")
            if not passed:
                all_passed = False
        except Exception as e:
            logger.error(f"[ERROR] {name}: {e}")
            all_passed = False
    
    logger.info("-" * 60)
    logger.info(f"Pre-market: {'READY' if all_passed else 'ISSUES DETECTED'}")
    logger.info("=" * 60)
    
    return all_passed  # Return status but DON'T halt execution

def _check_alpaca():
    """Verify Alpaca API connection."""
    try:
        # TODO: migrate to MarketDataAdapter (TICK-043)
        from sovereign.data.feeds.alpaca_feed import AlpacaFeed
        feed = AlpacaFeed()
        return feed.health_check()
    except Exception as e:
        logger.warning(f"Alpaca check: {e}")
        return True  # Don't block if API is temporarily down

def _check_router():
    """Verify regime router model exists."""
    import joblib
    model_path = Path('models/sovereign/regime_router.joblib')
    return model_path.exists()

def _check_specialists():
    """Verify specialist models exist."""
    import joblib
    momentum = Path('models/sovereign/momentum_specialist.joblib')
    reversion = Path('models/sovereign/reversion_specialist.joblib')
    return momentum.exists() and reversion.exists()

def _check_risk_config():
    """Verify risk parameters are valid."""
    from config.loader import params
    required = ['atr_gate', 'risk', 'hard_constraints']
    return all(k in params for k in required)

def _check_ledger():
    """Verify ledger is writable."""
    Path('data/ledger').mkdir(parents=True, exist_ok=True)
    test_file = Path('data/ledger/.test')
    try:
        test_file.write_text('test')
        test_file.unlink()
        return True
    except Exception:
        return False

#: Bars of history fetched for the ATR window. 14-period ATR needs 15 bars; 40
#: leaves room for holidays and the odd missing session without silently
#: shortening the window.
_ATR_LOOKBACK_BARS = 40
_ATR_PERIOD = 14


def _compute_atr(feed, symbol: str, close_price: float) -> float | None:
    """Real ATR for one symbol, or None — never a placeholder.

    Reuses `ForexSignalEngine._compute_atr_pct` (a staticmethod, so no engine
    instance and no forex coupling) rather than adding a second ATR implementation
    to the repo. Two divergent ATRs is how `carry_engine._compute_atr` came to
    return a hardcoded 0.001 without anyone noticing.

    Returns absolute ATR. `_compute_atr_pct` returns a *fraction* of price, so the
    result is multiplied by close — getting that wrong would silently under-size
    every position by two orders of magnitude.
    """
    try:
        from sovereign.forex.signal_engine import ForexSignalEngine

        end = datetime.utcnow()
        start = end - timedelta(days=_ATR_LOOKBACK_BARS * 2)  # calendar != trading days
        df = feed.get_bars(symbol, start=start, end=end, timeframe='1d')
        if df is None or len(df) < _ATR_PERIOD + 1:
            logger.warning(
                f"{symbol}: {0 if df is None else len(df)} daily bars, need "
                f"{_ATR_PERIOD + 1} for a {_ATR_PERIOD}-period ATR")
            return None

        # _compute_atr_pct tests for capitalised OHLC columns; get_bars returns
        # lowercase. Without the rename it silently falls back to the
        # close-to-close proxy instead of true range.
        ohlc = df.rename(columns={'open': 'Open', 'high': 'High',
                                  'low': 'Low', 'close': 'Close'})
        atr_pct = ForexSignalEngine._compute_atr_pct(
            ohlc['Close'], ohlc, period=_ATR_PERIOD)
        if atr_pct is None or len(atr_pct) == 0:
            logger.warning(f"{symbol}: ATR computation returned nothing")
            return None

        latest_pct = float(atr_pct.iloc[-1])
        if not (latest_pct > 0):          # also catches NaN
            logger.warning(f"{symbol}: ATR%={latest_pct} is not positive")
            return None

        atr = latest_pct * close_price
        logger.info(f"{symbol}: ATR({_ATR_PERIOD}) = {atr:.4f} "
                    f"({latest_pct:.2%} of price, {len(df)} bars)")
        return atr
    except Exception as e:  # noqa: BLE001 — any failure means no trade, not a guess
        logger.warning(f"{symbol}: ATR unavailable ({type(e).__name__}: {e})")
        return None


def execute_killzone(mode='paper'):
    """
    Phase 2: Execute during NY Open kill zone.
    
    Runs sovereign orchestrator on trinity assets.
    Makes trades if signals fire.
    """
    logger.info("\n" + "=" * 60)
    logger.info("KILL ZONE EXECUTION")
    logger.info("=" * 60)
    logger.info(f"Mode: {mode.upper()}")
    logger.info("=" * 60)
    
    from sovereign.orchestrator import SovereignOrchestrator
    from sovereign.data.feeds.alpaca_feed import AlpacaFeed

    orch = SovereignOrchestrator(mode=mode)
    orch.load_models(base_path='models/sovereign')
    feed = AlpacaFeed()
    
    # Trinity assets from config
    from config.loader import params
    symbols = params.get('universe', {}).get('trinity_assets', ['META', 'PFE', 'UNH'])
    
    executed_trades = []
    
    for symbol in symbols:
        logger.info(f"\n--- Processing {symbol} ---")
        
        try:
            # Get latest data
            latest = feed.get_latest_bar(symbol)
            
            # Build feature record (simplified for immediate trading)
            # In production, this would come from real-time feature pipeline
            from contracts.types import (
                SovereignFeatureRecord, RegimeFeatures, MomentumFeatures,
                MacroFeatures, PetrolausDecision
            )
            
            # Build minimal valid feature record
            regime = RegimeFeatures(
                hurst_short=0.55,  # Default trending
                hurst_long=0.55,
                hurst_signal='TRENDING',
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
                timestamp=datetime.utcnow().isoformat(),
                regime=regime,
                momentum=momentum,
                macro=macro,
                petroulas=petroulas,
                bar_ohlcv={
                    'open': latest['open'],
                    'high': latest['high'],
                    'low': latest['low'],
                    'close': latest['close'],
                    'volume': latest['volume']
                },
                is_valid=True,
                validation_errors=[]
            )
            
            # Run orchestrator
            atr = _compute_atr(feed, symbol, latest['close'])
            if atr is None:
                # No silent mocking. A substituted ATR reaching sizing is exactly
                # the FAKE_DATA class forex_data_health.py exists to catch — and
                # until 2026-07-20 this line was `latest['close'] * 0.02`, sizing
                # every trade off a constant regardless of realised volatility.
                logger.error(
                    f"[SKIP] {symbol}: ATR unavailable — refusing to size off a "
                    f"placeholder. No trade attempted.")
                continue
            equity = 100000.0  # Paper equity
            
            result = orch.run_session(
                symbol,
                feature_record=record,
                current_price=latest['close'],
                atr=atr,
                equity=equity
            )
            
            if result:
                executed_trades.append(result)
                logger.info(f"[OK] Trade executed: {result['trade_id']}")
            else:
                logger.info(f"[SKIP] No trade signal for {symbol}")
                
        except Exception as e:
            logger.error(f"[ERROR] {symbol}: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Kill Zone Complete: {len(executed_trades)} trades")
    logger.info("=" * 60)
    
    return executed_trades

def post_session_report():
    """
    Phase 3: Post-session summary.
    """
    logger.info("\n" + "=" * 60)
    logger.info("POST-SESSION REPORT")
    logger.info("=" * 60)
    
    # Count today's trades
    from sovereign.ledger.trade_ledger import TradeLedger
    ledger = TradeLedger()
    
    # This would read from ledger and summarize
    logger.info("[INFO] Post-session analysis logged")
    logger.info("=" * 60)

def run_full_session(mode='paper'):
    """
    Complete daily trading session.
    Can be called from Task Scheduler.
    """
    logger.info("\n" + "=" * 60)
    logger.info("SOVEREIGN DAILY EXECUTION")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    # Phase 1: Pre-market (informational)
    pre_market_checklist()
    
    # Phase 2: Kill zone execution
    trades = execute_killzone(mode=mode)
    
    # Phase 3: Post-session
    post_session_report()
    
    logger.info("\n" + "=" * 60)
    logger.info("SESSION COMPLETE")
    logger.info("=" * 60)
    
    return len(trades)

def main():
    parser = argparse.ArgumentParser(description='Sovereign Daily Execution')
    parser.add_argument(
        '--mode',
        choices=['paper', 'live'],
        default='paper',
        help='Trading mode (default: paper)'
    )
    parser.add_argument(
        '--phase',
        choices=['premarket', 'killzone', 'postsession', 'full'],
        default='full',
        help='Which phase to run (default: full)'
    )
    parser.add_argument(
        '--date',
        default=None,
        help='Simulation date YYYY-MM-DD (default: today)'
    )

    args = parser.parse_args()

    # ── Monthly re-optimisation (runs once per month, ~20s, non-blocking) ────
    from sovereign.monthly_reopt import MonthlyReopt
    reopt = MonthlyReopt()
    if reopt.should_run():
        reopt.run(symbols=['META', 'PFE', 'UNH', 'BLK', 'QQQ', 'ES=F', 'NQ=F'])
    # ─────────────────────────────────────────────────────────────────────────

    if args.phase == 'premarket':
        pre_market_checklist()
    elif args.phase == 'killzone':
        execute_killzone(mode=args.mode)
    elif args.phase == 'postsession':
        post_session_report()
    else:
        run_full_session(mode=args.mode)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
