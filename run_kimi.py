"""
Run KimiBrain - LLM-Powered Trading with Learning

Usage:
    python run_kimi.py [--paper|--live] [--symbols SPY,QQQ,...]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from kimi_brain import KimiBrain
from ai_trading_bridge import AITradingBridge


def main():
    parser = argparse.ArgumentParser(description="Run KimiBrain LLM Trading")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,QQQ,IWM,NVDA,AAPL",
        help="Comma-separated symbols to analyze",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold (default: 0.6)",
    )
    parser.add_argument(
        "--learn",
        action="store_true",
        default=True,
        help="Enable learning mode (default: True)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" KIMI BRAIN - LLM-POWERED TRADING")
    print("=" * 70)
    print(f" Mode: {args.mode.upper()}")
    print(f" Symbols: {args.symbols}")
    print(f" Min Confidence: {args.confidence}")
    print(f" Learning: {'ON' if args.learn else 'OFF'}")
    print("=" * 70)
    print()

    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Create KimiBrain
    print("[Setup] Initializing KimiBrain...")
    try:
        brain = KimiBrain(
            learning_mode=args.learn, temperature=0.3  # Lower = more consistent
        )
    except ValueError as e:
        print(f"[ERROR] {e}")
        print("Make sure KIMI_API_KEY is set in your .env file")
        return

    # Create bridge
    print("[Setup] Connecting to Alpaca...")
    bridge = AITradingBridge(
        brain=brain,
        symbols=symbols,
        timeframe="1D",
        lookback_days=30,
        paper=(args.mode == "paper"),
        min_confidence=args.confidence,
    )

    print("[Setup] Ready!")
    print()

    # Run trading cycle
    print("[RUNNING] Analyzing markets with Kimi LLM...")
    print("(This may take 30-60 seconds for LLM responses)")
    print()

    result = bridge.run_cycle()

    # Summary
    print("\n" + "=" * 70)
    print(" TRADING CYCLE COMPLETE")
    print("=" * 70)
    print(f" Brain: {result['brain']}")
    print(f" Symbols Analyzed: {result['symbols_data']}")
    print(f" Signals Generated: {result['signals']}")
    print(f" Trades Executed: {result['executed']}")
    print("=" * 70)

    if result["executed"] > 0:
        print("\n✓ Orders submitted to Alpaca!")
        print("  Kimi will learn from the outcomes of these trades.")
    else:
        print("\n○ No trades executed.")
        print("  Kimi didn't find high-confidence setups this cycle.")

    print("\n[Tip] Run again in a few hours to see if Kimi learned anything!")


if __name__ == "__main__":
    main()
