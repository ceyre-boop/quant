"""Signal Analyzer - Analyzes archived signals for insights.

Provides analysis tools for understanding signal quality and patterns.
"""

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta

from data.signal_archive import SignalArchive

logger = logging.getLogger(__name__)


class SignalAnalyzer:
    """Analyzes trading signals to find patterns and insights.

    Capabilities:
    - Win rate by regime
    - Win rate by time of day
    - Feature importance analysis
    - Signal quality scoring

    Usage:
        analyzer = SignalAnalyzer()

        # Analyze by regime
        regime_stats = analyzer.analyze_by_regime()

        # Best performing hours
        hourly_stats = analyzer.analyze_by_hour()
    """

    def __init__(self, signal_archive: Optional[SignalArchive] = None):
        self.archive = signal_archive or SignalArchive()

    def analyze_by_regime(self, days_back: int = 30) -> Dict[str, Any]:
        """Analyze performance by market regime.

        Args:
            days_back: Days of history to analyze

        Returns:
            Stats by regime
        """
        end = datetime.now()
        start = end - timedelta(days=days_back)

        signals = self.archive.get_signals_for_range(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

        # Group by regime
        regimes = defaultdict(lambda: {"trades": [], "pnl": 0})

        for s in signals:
            outcome = s.get("outcome")
            if not outcome:
                continue

            regime = s.get("bias", {}).get("regime", "unknown")
            pnl = outcome.get("pnl", 0)

            regimes[regime]["trades"].append(pnl)
            regimes[regime]["pnl"] += pnl

        # Calculate stats per regime
        results = {}
        for regime, data in regimes.items():
            trades = data["trades"]
            wins = sum(1 for p in trades if p > 0)

            results[regime] = {
                "total_trades": len(trades),
                "winning_trades": wins,
                "win_rate": round(wins / len(trades), 3) if trades else 0,
                "total_pnl": round(data["pnl"], 2),
                "avg_trade": round(data["pnl"] / len(trades), 2) if trades else 0,
            }

        return results

    def analyze_by_hour(self, days_back: int = 30) -> Dict[int, Dict[str, Any]]:
        """Analyze performance by hour of day.

        Args:
            days_back: Days of history to analyze

        Returns:
            Stats by hour (0-23)
        """
        end = datetime.now()
        start = end - timedelta(days=days_back)

        signals = self.archive.get_signals_for_range(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

        # Group by hour
        hours = defaultdict(lambda: {"trades": [], "pnl": 0})

        for s in signals:
            outcome = s.get("outcome")
            if not outcome:
                continue

            ts = s.get("timestamp", "")
            if "T" in ts:
                hour = int(ts.split("T")[1].split(":")[0])
            else:
                continue

            pnl = outcome.get("pnl", 0)
            hours[hour]["trades"].append(pnl)
            hours[hour]["pnl"] += pnl

        # Calculate stats per hour
        results = {}
        for hour in sorted(hours.keys()):
            data = hours[hour]
            trades = data["trades"]
            wins = sum(1 for p in trades if p > 0)

            results[hour] = {
                "total_trades": len(trades),
                "winning_trades": wins,
                "win_rate": round(wins / len(trades), 3) if trades else 0,
                "total_pnl": round(data["pnl"], 2),
                "avg_trade": round(data["pnl"] / len(trades), 2) if trades else 0,
            }

        return results

    def analyze_by_symbol(self, days_back: int = 30) -> Dict[str, Any]:
        """Analyze performance by symbol."""
        end = datetime.now()
        start = end - timedelta(days=days_back)

        signals = self.archive.get_signals_for_range(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

        symbols = defaultdict(lambda: {"trades": [], "pnl": 0})

        for s in signals:
            outcome = s.get("outcome")
            if not outcome:
                continue

            symbol = s.get("symbol", "unknown")
            pnl = outcome.get("pnl", 0)

            symbols[symbol]["trades"].append(pnl)
            symbols[symbol]["pnl"] += pnl

        results = {}
        for symbol, data in symbols.items():
            trades = data["trades"]
            wins = sum(1 for p in trades if p > 0)

            results[symbol] = {
                "total_trades": len(trades),
                "winning_trades": wins,
                "win_rate": round(wins / len(trades), 3) if trades else 0,
                "total_pnl": round(data["pnl"], 2),
                "avg_trade": round(data["pnl"] / len(trades), 2) if trades else 0,
            }

        return results

    def get_best_setup(self, days_back: int = 30) -> Dict[str, Any]:
        """Identify the best performing setup combination.

        Returns:
            Best regime + hour combination
        """
        regime_stats = self.analyze_by_regime(days_back)
        hour_stats = self.analyze_by_hour(days_back)

        # Find best regime
        best_regime = max(
            regime_stats.items(),
            key=lambda x: (x[1].get("win_rate", 0) if x[1].get("total_trades", 0) >= 5 else 0),
            default=(None, {}),
        )

        # Find best hour (with at least 5 trades)
        best_hour = max(
            hour_stats.items(),
            key=lambda x: (x[1].get("win_rate", 0) if x[1].get("total_trades", 0) >= 5 else 0),
            default=(None, {}),
        )

        return {
            "best_regime": {"name": best_regime[0], "stats": best_regime[1]},
            "best_hour": {"hour": best_hour[0], "stats": best_hour[1]},
            "recommendation": self._generate_recommendation(best_regime, best_hour),
        }

    def _generate_recommendation(self, best_regime: tuple, best_hour: tuple) -> str:
        """Generate a recommendation based on analysis."""
        parts = []

        if best_regime[0] and best_regime[1].get("win_rate", 0) > 0.6:
            parts.append(f"Focus on {best_regime[0]} regime " f"({best_regime[1]['win_rate']:.0%} win rate)")

        if best_hour[0] is not None and best_hour[1].get("win_rate", 0) > 0.6:
            parts.append(f"Best performance at {best_hour[0]:02d}:00 " f"({best_hour[1]['win_rate']:.0%} win rate)")

        return " | ".join(parts) if parts else "No clear patterns identified yet"

    def generate_full_report(self, days_back: int = 30) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "period_days": days_back,
            "by_regime": self.analyze_by_regime(days_back),
            "by_hour": self.analyze_by_hour(days_back),
            "by_symbol": self.analyze_by_symbol(days_back),
            "best_setup": self.get_best_setup(days_back),
        }

        # Archive stats
        report["archive_stats"] = self.archive.get_statistics()

        return report


def main():
    """CLI for signal analysis."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Signal Analyzer")
    parser.add_argument("--days", type=int, default=30, help="Days to analyze")
    parser.add_argument("--output", help="Output JSON file")

    args = parser.parse_args()

    analyzer = SignalAnalyzer()
    report = analyzer.generate_full_report(args.days)

    # Print summary
    print(f"\n{'='*60}")
    print("Signal Analysis Report")
    print(f"{'='*60}")
    print(f"Period: Last {args.days} days")
    print(f"\nPerformance by Regime:")
    for regime, stats in report["by_regime"].items():
        print(f"  {regime}: {stats['win_rate']:.1%} win rate ({stats['total_trades']} trades)")

    print(f"\nPerformance by Symbol:")
    for symbol, stats in report["by_symbol"].items():
        print(f"  {symbol}: {stats['win_rate']:.1%} win rate (${stats['total_pnl']:+.2f})")

    print(f"\nRecommendation: {report['best_setup']['recommendation']}")
    print(f"{'='*60}\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
