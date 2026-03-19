"""Performance Monitor - Weekly performance tracking and reporting.

Monitors trading performance, detects model degradation, and pushes
metrics to Firebase for dashboards.
"""

import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import asdict

from integration.firebase_client import FirebaseClient
from meta_evaluator.metrics_calculator import (
    calculate_all_metrics,
    PerformanceMetrics,
    detect_model_drift
)
from data.signal_archive import SignalArchive

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitors and reports trading performance metrics.
    
    Runs weekly to calculate:
    - Sharpe ratio, win rate, max drawdown
    - Model drift detection
    - Trade statistics
    
    Pushes results to Firebase at /performance/weekly_metrics/
    
    Usage:
        monitor = PerformanceMonitor()
        
        # Run weekly analysis
        metrics = monitor.run_weekly_analysis()
        
        # Or check for model drift
        drift_detected = monitor.check_model_health()
    """
    
    def __init__(
        self,
        firebase_client: Optional[FirebaseClient] = None,
        signal_archive: Optional[SignalArchive] = None
    ):
        self.firebase = firebase_client or FirebaseClient()
        self.archive = signal_archive or SignalArchive()
        self._baseline_accuracy: Optional[float] = None
    
    def run_weekly_analysis(
        self,
        days_back: int = 7,
        push_to_firebase: bool = True
    ) -> PerformanceMetrics:
        """Run complete weekly performance analysis.
        
        Args:
            days_back: Days to analyze (default: 7)
            push_to_firebase: Whether to push results to Firebase
            
        Returns:
            PerformanceMetrics object
        """
        logger.info(f"Running weekly analysis for last {days_back} days")
        
        # Get date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Get trades from archive
        trades = self._get_trades_for_period(start_str, end_str)
        
        if not trades:
            logger.warning("No trades found for analysis period")
            # Return empty metrics
            metrics = self._empty_metrics(start_str, end_str)
        else:
            # Calculate metrics
            metrics = calculate_all_metrics(
                trades=trades,
                period_days=days_back
            )
        
        # Add week identifier
        week_id = end_str  # Week ending date
        
        # Check model drift
        drift_detected, accuracy = self._check_drift(trades)
        metrics.model_drift_detected = drift_detected
        metrics.prediction_accuracy = accuracy
        
        # Push to Firebase
        if push_to_firebase:
            self._push_metrics(week_id, metrics)
        
        logger.info(
            f"Weekly analysis complete: Sharpe={metrics.sharpe_ratio}, "
            f"WinRate={metrics.win_rate}%, Trades={metrics.total_trades}"
        )
        
        return metrics
    
    def _get_trades_for_period(
        self,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """Get completed trades for a date range."""
        signals = self.archive.get_signals_for_range(start_date, end_date)
        
        # Filter to trades with outcomes
        trades = [
            {
                'symbol': s.get('symbol'),
                'direction': s.get('direction'),
                'entry_price': s.get('entry_price'),
                'exit_price': s.get('outcome', {}).get('exit_price'),
                'pnl': s.get('outcome', {}).get('pnl'),
                'timestamp': s.get('timestamp'),
                'exit_reason': s.get('outcome', {}).get('exit_reason')
            }
            for s in signals
            if s.get('outcome') and s['outcome'].get('pnl') is not None
        ]
        
        return trades
    
    def _check_drift(
        self,
        trades: List[Dict[str, Any]]
    ) -> tuple[bool, float]:
        """Check for model performance drift.
        
        Compares recent accuracy to baseline.
        """
        # Get baseline from Firebase
        if self._baseline_accuracy is None:
            try:
                baseline = self.firebase.rtdb_get('/performance/baseline_accuracy')
                self._baseline_accuracy = baseline or 0.55  # Default 55%
            except:
                self._baseline_accuracy = 0.55
        
        # Need predictions vs actuals - simplified for now
        # In real implementation, would compare signal confidence to outcomes
        if len(trades) < 10:
            return False, 0.0
        
        # Simple accuracy: did direction match outcome?
        correct = sum(
            1 for t in trades
            if (t.get('pnl', 0) > 0)
        )
        accuracy = correct / len(trades) if trades else 0
        
        # Detect drift (10% drop from baseline)
        drift = (self._baseline_accuracy - accuracy) > 0.1
        
        return drift, round(accuracy, 3)
    
    def _push_metrics(self, week_id: str, metrics: PerformanceMetrics):
        """Push metrics to Firebase."""
        try:
            data = asdict(metrics)
            data['week_id'] = week_id
            data['calculated_at'] = datetime.now().isoformat()
            
            self.firebase.rtdb_update(f'/performance/weekly_metrics/{week_id}', data)
            
            # Also update latest
            self.firebase.rtdb_update('/performance/latest', data)
            
            logger.info(f"Metrics pushed to Firebase for week {week_id}")
            
        except Exception as e:
            logger.error(f"Failed to push metrics: {e}")
    
    def _empty_metrics(
        self,
        start_date: str,
        end_date: str
    ) -> PerformanceMetrics:
        """Create empty metrics when no trades."""
        return PerformanceMetrics(
            total_return=0.0, avg_daily_return=0.0,
            volatility=0.0, max_drawdown=0.0, max_drawdown_duration=0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0,
            profit_factor=0.0, avg_trade_pnl=0.0,
            model_drift_detected=False, prediction_accuracy=0.0,
            start_date=start_date, end_date=end_date
        )
    
    def check_model_health(self) -> Dict[str, Any]:
        """Quick health check for model performance.
        
        Returns:
            Health status dict
        """
        # Get last 30 trades
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        trades = self._get_trades_for_period(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        drift_detected, accuracy = self._check_drift(trades)
        
        health = {
            'status': 'HEALTHY',
            'recent_trades': len(trades),
            'recent_accuracy': accuracy,
            'drift_detected': drift_detected,
            'checked_at': datetime.now().isoformat()
        }
        
        if drift_detected:
            health['status'] = 'DEGRADED'
            health['alert'] = 'Model performance has degraded'
        
        if len(trades) < 5:
            health['status'] = 'INSUFFICIENT_DATA'
        
        return health
    
    def export_report(self, filepath: str, days_back: int = 7):
        """Export performance report to JSON file.
        
        Args:
            filepath: Path to save report
            days_back: Days to include
        """
        metrics = self.run_weekly_analysis(days_back=days_back, push_to_firebase=False)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'period_days': days_back,
            'metrics': asdict(metrics)
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report exported to {filepath}")


def main():
    """CLI entry point for running analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance Monitor')
    parser.add_argument('--period', choices=['daily', 'weekly', 'monthly'], default='weekly')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--no-push', action='store_true', help='Skip Firebase push')
    
    args = parser.parse_args()
    
    # Map period to days
    days_map = {'daily': 1, 'weekly': 7, 'monthly': 30}
    days = days_map[args.period]
    
    monitor = PerformanceMonitor()
    metrics = monitor.run_weekly_analysis(
        days_back=days,
        push_to_firebase=not args.no_push
    )
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Performance Report - {args.period.upper()}")
    print(f"{'='*50}")
    print(f"Sharpe Ratio:      {metrics.sharpe_ratio:.2f}")
    print(f"Win Rate:          {metrics.win_rate:.1f}%")
    print(f"Max Drawdown:      {metrics.max_drawdown:.1f}%")
    print(f"Total Trades:      {metrics.total_trades}")
    print(f"Avg Trade P&L:     ${metrics.avg_trade_pnl:.2f}")
    print(f"Model Drift:       {'YES' if metrics.model_drift_detected else 'NO'}")
    print(f"{'='*50}\n")
    
    if args.output:
        monitor.export_report(args.output, days)


if __name__ == '__main__':
    main()
