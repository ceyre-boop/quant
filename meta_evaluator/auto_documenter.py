"""
Auto-Documentation Module for Clawd Trading

Automatically logs backtest results, live performance, and generates reports.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("docs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestResult:
    """Backtest result data."""

    timestamp: str
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    total_trades: int
    statistical_significance: bool
    r_squared: Optional[float] = None
    chi_squared: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LiveTrade:
    """Single live trade record."""

    trade_id: str
    timestamp: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    pnl: Optional[float]
    pnl_pct: Optional[float]
    status: str  # OPEN, CLOSED
    entry_model: str
    dominant_participant: str
    regime: str
    gates_passed: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AutoDocumenter:
    """Automatically document trading results and performance."""

    def __init__(self):
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"AutoDocumenter initialized: {self.results_dir}")

    def log_backtest_result(self, result: BacktestResult) -> str:
        """Log backtest result to docs/results/."""
        # Save as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_{timestamp}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Update RESULTS.md
        self._update_results_md(result)

        logger.info(f"Backtest result logged: {filepath}")
        return str(filepath)

    def _update_results_md(self, result: BacktestResult) -> None:
        """Update docs/RESULTS.md with latest backtest."""
        results_md = Path("docs/RESULTS.md")

        # Create header if file doesn't exist
        if not results_md.exists():
            header = """# Clawd Trading Results

Auto-generated performance tracking.

## Latest Backtest Results

"""
            results_md.write_text(header)

        # Append latest result
        sig_indicator = "✅ SIGNIFICANT" if result.statistical_significance else "❌ Not significant"

        entry = f"""### {result.timestamp}

| Metric | Value | Assessment |
|--------|-------|------------|
| **Sharpe Ratio** | {result.sharpe_ratio:.2f} | {'Excellent (>1.0)' if result.sharpe_ratio > 1.0 else 'Good (>0.5)' if result.sharpe_ratio > 0.5 else 'Marginal'} |
| **Win Rate** | {result.win_rate:.1%} | {'Solid edge (>55%)' if result.win_rate > 0.55 else 'Moderate (50-55%)'} |
| **Max Drawdown** | {result.max_drawdown:.1%} | {'Fantastic (<5%)' if result.max_drawdown > -0.05 else 'Good (<10%)' if result.max_drawdown > -0.10 else 'Needs work'} |
| **Total Trades** | {result.total_trades} | {'Decent sample' if result.total_trades > 100 else 'Small sample'} |
| **Statistical Significance** | {sig_indicator} | {'Edge is real' if result.statistical_significance else 'More data needed'} |

**Notes:** {result.notes}

---

"""

        # Prepend to file (latest first)
        current_content = results_md.read_text()
        # Find insertion point after "## Latest Backtest Results"
        insert_point = current_content.find("## Latest Backtest Results\n\n") + len("## Latest Backtest Results\n\n")
        new_content = current_content[:insert_point] + entry + current_content[insert_point:]
        results_md.write_text(new_content)

    def log_live_trade(self, trade: LiveTrade) -> None:
        """Log a live trade to the trades database."""
        trades_file = self.results_dir / "live_trades.jsonl"

        with open(trades_file, "a") as f:
            f.write(json.dumps(trade.to_dict()) + "\n")

        logger.info(f"Live trade logged: {trade.trade_id}")

    def compare_live_vs_backtest(self) -> Dict[str, Any]:
        """Compare live performance vs backtest expectations."""
        trades_file = self.results_dir / "live_trades.jsonl"

        if not trades_file.exists():
            return {"error": "No live trades yet"}

        # Load all trades
        trades = []
        with open(trades_file, "r") as f:
            for line in f:
                if line.strip():
                    trades.append(json.loads(line))

        if not trades:
            return {"error": "No live trades yet"}

        # Calculate live metrics
        closed_trades = [t for t in trades if t["status"] == "CLOSED"]
        if not closed_trades:
            return {"error": "No closed trades yet"}

        wins = sum(1 for t in closed_trades if t.get("pnl", 0) > 0)
        live_win_rate = wins / len(closed_trades)
        live_pnl = sum(t.get("pnl", 0) for t in closed_trades)

        # Load latest backtest for comparison
        backtest_files = sorted(self.results_dir.glob("backtest_*.json"))
        if not backtest_files:
            return {
                "live_trades": len(closed_trades),
                "live_win_rate": live_win_rate,
                "live_pnl": live_pnl,
                "backtest_comparison": "No backtest data available",
            }

        with open(backtest_files[-1], "r") as f:
            backtest = json.load(f)

        return {
            "live_trades": len(closed_trades),
            "live_win_rate": live_win_rate,
            "backtest_win_rate": backtest["win_rate"],
            "win_rate_diff": live_win_rate - backtest["win_rate"],
            "live_pnl": live_pnl,
            "backtest_sharpe": backtest["sharpe_ratio"],
            "assessment": (
                "Live performance tracking on track"
                if abs(live_win_rate - backtest["win_rate"]) < 0.1
                else "Deviation detected"
            ),
        }

    def generate_performance_report(self) -> str:
        """Generate a full performance report."""
        comparison = self.compare_live_vs_backtest()

        report_path = self.results_dir / f"report_{datetime.now().strftime('%Y%m%d')}.md"

        report = f"""# Performance Report - {datetime.now().strftime('%Y-%m-%d')}

## Live vs Backtest Comparison

```json
{json.dumps(comparison, indent=2)}
```

## Key Insights

"""

        if "error" in comparison:
            report += "- No live trading data available yet.\n"
        else:
            report += f"- **Live Win Rate:** {comparison['live_win_rate']:.1%}\n"
            report += f"- **Backtest Win Rate:** {comparison['backtest_win_rate']:.1%}\n"
            report += f"- **Difference:** {comparison['win_rate_diff']:+.1%}\n"
            report += f"\n**Assessment:** {comparison['assessment']}\n"

        report += f"\n---\n*Generated: {datetime.now().isoformat()}*\n"

        report_path.write_text(report)
        logger.info(f"Performance report generated: {report_path}")

        return str(report_path)


# Global instance
_auto_doc = None


def get_auto_documenter() -> AutoDocumenter:
    """Get or create global AutoDocumenter instance."""
    global _auto_doc
    if _auto_doc is None:
        _auto_doc = AutoDocumenter()
    return _auto_doc


def log_backtest_result(
    sharpe_ratio: float,
    win_rate: float,
    max_drawdown: float,
    total_trades: int,
    statistical_significance: bool = False,
    notes: str = "",
) -> str:
    """Quick function to log backtest result."""
    doc = get_auto_documenter()
    result = BacktestResult(
        timestamp=datetime.now().isoformat(),
        sharpe_ratio=sharpe_ratio,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        total_trades=total_trades,
        statistical_significance=statistical_significance,
        notes=notes,
    )
    return doc.log_backtest_result(result)


def log_live_trade(**kwargs) -> None:
    """Quick function to log a live trade."""
    doc = get_auto_documenter()
    trade = LiveTrade(**kwargs)
    doc.log_live_trade(trade)


def compare_performance() -> Dict[str, Any]:
    """Compare live vs backtest."""
    return get_auto_documenter().compare_live_vs_backtest()
