"""
Run ClawdBrain - Full 3-Layer Trading System

Usage:
    python run_clawd.py [--paper|--live]
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from clawd_brain_v31 import ClawdBrain
from ai_trading_bridge import AITradingBridge


def main():
    parser = argparse.ArgumentParser(description='Run ClawdBrain Trading System')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='Trading mode (default: paper)')
    parser.add_argument('--symbols', type=str, default='SPY,QQQ,IWM,NVDA,AAPL,MSFT,TSLA,AMD',
                       help='Comma-separated symbols to trade')
    parser.add_argument('--confidence', type=float, default=0.55,
                       help='Minimum confidence threshold (default: 0.55)')
    parser.add_argument('--kelly', type=float, default=0.25,
                       help='Kelly fraction (default: 0.25 = Quarter Kelly)')
    parser.add_argument('--max-position', type=float, default=0.10,
                       help='Max position size as pct of equity (default: 0.10)')
    
    args = parser.parse_args()
    
    print("="*70)
    print(" CLAWD BRAIN - 3 LAYER TRADING SYSTEM")
    print("="*70)
    print(f" Mode: {args.mode.upper()}")
    print(f" Symbols: {args.symbols}")
    print(f" Min Confidence: {args.confidence}")
    print(f" Kelly Fraction: {args.kelly}")
    print(f" Max Position: {args.max_position*100:.0f}%")
    print("="*70)
    print()
    
    if args.mode == 'live':
        print("⚠️  WARNING: LIVE TRADING MODE")
        print("This will use REAL MONEY!")
        confirm = input("Type 'LIVE' to confirm: ")
        if confirm != 'LIVE':
            print("Aborted.")
            return
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Create ClawdBrain with 3-layer architecture
    print("[Setup] Initializing ClawdBrain...")
    brain = ClawdBrain(
        kelly_fraction=args.kelly,
        max_position_pct=args.max_position,
        min_confidence=args.confidence,
        paper=(args.mode == 'paper')
    )
    
    # Create bridge
    print("[Setup] Connecting to Alpaca...")
    bridge = AITradingBridge(
        brain=brain,
        symbols=symbols,
        timeframe="1D",
        lookback_days=60,
        paper=(args.mode == 'paper'),
        min_confidence=args.confidence
    )
    
    print("[Setup] Ready!")
    print()
    
    # Run trading cycle
    result = bridge.run_cycle()
    
    # Summary
    print("\n" + "="*70)
    print(" TRADING CYCLE COMPLETE")
    print("="*70)
    print(f" Brain: {result['brain']}")
    print(f" Symbols Analyzed: {result['symbols_data']}")
    print(f" Signals Generated: {result['signals']}")
    print(f" Trades Executed: {result['executed']}")
    print("="*70)
    
    if result['executed'] > 0:
        print("\n✓ Orders submitted to Alpaca!")
        print("  Check your Alpaca dashboard for status.")
    else:
        print("\n○ No trades executed this cycle.")
        print("  (Signals filtered by 3-layer validation)")


if __name__ == '__main__':
    main()
