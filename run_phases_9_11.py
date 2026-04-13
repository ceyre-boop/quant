"""
Sovereign Phase 9-11 Integration Runner

Master script to execute:
  Phase 9: Veto Rate Diagnostic
  Phase 10: Backtest Engine
  Phase 11: Paper Trading Campaign

Usage:
  python run_phases_9_11.py --phase 9
  python run_phases_9_11.py --phase 10 --symbols META PFE UNH --start 2023-01-01 --end 2024-12-31
  python run_phases_9_11.py --phase 11 --start-campaign
  python run_phases_9_11.py --phase 11 --daily-cycle
  python run_phases_9_11.py --full-pipeline
"""

import argparse
import logging
import sys
from pathlib import Path

from sovereign.validation.veto_diagnostic import VetoRateDiagnostic
from sovereign.validation.backtest_engine import SovereignBacktest
from sovereign.validation.paper_trading_runner import PaperTradingRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)

logger = logging.getLogger(__name__)


def run_phase_9():
    """Run Veto Rate Diagnostic."""
    logger.info("=" * 60)
    logger.info("EXECUTING PHASE 9: VETO RATE DIAGNOSTIC")
    logger.info("=" * 60)
    
    diagnostic = VetoRateDiagnostic()
    result = diagnostic.run_diagnostic(days=30)
    diagnostic.generate_report_file(result)
    
    logger.info(f"\nHealth Score: {result.health_score:.0f}/100")
    logger.info(f"Execution Rate: {result.execution_rate:.1f}%")
    logger.info(f"Status: {'HEALTHY' if result.is_healthy else 'NEEDS ATTENTION'}")
    
    return 0 if result.is_healthy else 1


def run_phase_10(symbols, start_date, end_date):
    """Run Backtest Engine."""
    logger.info("=" * 60)
    logger.info("EXECUTING PHASE 10: BACKTEST ENGINE")
    logger.info("=" * 60)
    
    backtest = SovereignBacktest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        starting_equity=100000.0,
        slippage=0.001
    )
    
    result = backtest.run()
    backtest.save_results(result)
    
    logger.info(f"\nTotal Trades: {result.total_trades}")
    logger.info(f"Win Rate: {result.win_rate:.1%}")
    logger.info(f"Total Return: {result.total_return:.2%}")
    logger.info(f"3x Slippage Test: {'PASSED' if result.passed_3x_slippage else 'FAILED'}")
    
    return 0 if result.passed_3x_slippage else 1


def run_phase_11(start_campaign=False, daily_cycle=False):
    """Run Paper Trading Campaign."""
    logger.info("=" * 60)
    logger.info("EXECUTING PHASE 11: PAPER TRADING")
    logger.info("=" * 60)
    
    runner = PaperTradingRunner(
        symbols=['META', 'PFE', 'UNH'],
        starting_equity=100000.0,
        campaign_name='sovereign_paper_v1'
    )
    
    if start_campaign:
        runner.start()
        logger.info("Paper trading campaign started successfully")
    
    if daily_cycle:
        summary = runner.run_daily_cycle()
        status = runner.get_status()
        
        logger.info(f"\nTotal Trades: {status.total_trades}")
        logger.info(f"Win Rate: {status.win_rate:.1%}")
        logger.info(f"Total PnL: ${status.total_pnl:,.2f}")
        logger.info(f"Trades to Live: {status.trades_to_live}")
        logger.info(f"Can Go Live: {'YES' if status.can_go_live else 'NO'}")
        
        if status.can_go_live:
            logger.info("🚀 LIVE TRANSITION CRITERIA MET — AWAITING HUMAN APPROVAL")
    
    return 0


def run_full_pipeline():
    """Run all phases sequentially."""
    logger.info("=" * 60)
    logger.info("RUNNING FULL SOVEREIGN PIPELINE (PHASES 9-11)")
    logger.info("=" * 60)
    
    # Phase 9
    exit_code = run_phase_9()
    if exit_code != 0:
        logger.error("Phase 9 failed — pipeline halted")
        return exit_code
    
    # Phase 10
    exit_code = run_phase_10(
        symbols=['META', 'PFE', 'UNH'],
        start_date='2023-01-01',
        end_date='2024-12-31'
    )
    if exit_code != 0:
        logger.warning("Phase 10: Backtest did not pass 3x slippage — review before paper trading")
        # Don't halt — let user decide
    
    # Phase 11
    exit_code = run_phase_11(start_campaign=True, daily_cycle=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("FULL PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    return exit_code


def main():
    parser = argparse.ArgumentParser(
        description='Sovereign Phase 9-11 Integration Runner'
    )
    parser.add_argument(
        '--phase',
        type=int,
        choices=[9, 10, 11],
        help='Which phase to run'
    )
    parser.add_argument(
        '--full-pipeline',
        action='store_true',
        help='Run phases 9, 10, and 11 sequentially'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['META', 'PFE', 'UNH'],
        help='Symbols to trade (for backtest/paper)'
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2023-01-01',
        help='Backtest start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default='2024-12-31',
        help='Backtest end date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--start-campaign',
        action='store_true',
        help='Start new paper trading campaign (Phase 11)'
    )
    parser.add_argument(
        '--daily-cycle',
        action='store_true',
        help='Run daily cycle for existing paper campaign (Phase 11)'
    )
    
    args = parser.parse_args()
    
    if args.full_pipeline:
        return run_full_pipeline()
    
    if args.phase == 9:
        return run_phase_9()
    elif args.phase == 10:
        return run_phase_10(args.symbols, args.start, args.end)
    elif args.phase == 11:
        return run_phase_11(args.start_campaign, args.daily_cycle)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
