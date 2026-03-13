"""Performance Monitoring - Weekly metrics and model drift detection.

Pushes performance data to Firebase for dashboard visualization.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from firebase.client import FirebaseClient

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and report trading performance metrics."""
    
    def __init__(self, firebase_client: Optional[FirebaseClient] = None):
        self.firebase = firebase_client or FirebaseClient()
        self.metrics_path = Path("metrics/performance_history.json")
    
    def compute_weekly_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """Compute weekly performance metrics.
        
        Args:
            trades: List of trade records with pnl, duration, etc.
        
        Returns:
            Dict with computed metrics
        """
        if not trades:
            return self._empty_metrics()
        
        # Extract PnL values
        pnls = [t.get('realized_pnl', 0) for t in trades if t.get('realized_pnl') is not None]
        
        if not pnls:
            return self._empty_metrics()
        
        # Calculate metrics
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / len(pnls)
        
        # Win rate
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls)
        
        # Sharpe ratio (simplified)
        if len(pnls) > 1:
            import statistics
            std_pnl = statistics.stdev(pnls)
            sharpe = (avg_pnl / std_pnl) * (252 ** 0.5) if std_pnl > 0 else 0
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = 0
        peak = 0
        max_drawdown = 0
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            drawdown = (peak - cumulative) / peak if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average risk-reward
        rr_ratios = [t.get('rr_achieved', 0) for t in trades if t.get('rr_achieved')]
        avg_rr = sum(rr_ratios) / len(rr_ratios) if rr_ratios else 0
        
        return {
            'total_trades': len(trades),
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_rr': avg_rr,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'period_start': trades[0].get('opened_at', datetime.now().isoformat()),
            'period_end': trades[-1].get('closed_at', datetime.now().isoformat()),
            'computed_at': datetime.now().isoformat()
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            'total_trades': 0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'profit_factor': 0.0,
            'avg_rr': 0.0,
            'computed_at': datetime.now().isoformat()
        }
    
    def check_feature_drift(self, current_features: Optional[Dict] = None) -> Dict[str, Any]:
        """Check for feature drift that might indicate model needs retraining.
        
        Args:
            current_features: Current feature distribution stats
        
        Returns:
            Dict with drift analysis
        """
        # Get historical feature stats from Firebase
        try:
            historical = self.firebase.read_realtime('/features/historical_stats')
        except:
            historical = None
        
        if not historical or not current_features:
            return {
                'needs_refit': False,
                'drift_score': 0.0,
                'message': 'Insufficient data for drift detection'
            }
        
        # Calculate drift score (simplified PSI - Population Stability Index)
        drift_scores = []
        for feature_name, current_value in current_features.items():
            if feature_name in historical:
                hist_mean = historical[feature_name].get('mean', 0)
                hist_std = historical[feature_name].get('std', 1)
                
                if hist_std > 0:
                    z_score = abs((current_value - hist_mean) / hist_std)
                    drift_scores.append(z_score)
        
        avg_drift = sum(drift_scores) / len(drift_scores) if drift_scores else 0
        
        # Threshold: if average drift > 2 standard deviations, recommend refit
        needs_refit = avg_drift > 2.0
        
        return {
            'needs_refit': needs_refit,
            'drift_score': avg_drift,
            'feature_drifts': dict(zip(current_features.keys(), drift_scores)) if drift_scores else {},
            'threshold': 2.0,
            'checked_at': datetime.now().isoformat()
        }
    
    def publish_weekly_metrics(self, metrics: Dict[str, Any], week: Optional[str] = None) -> None:
        """Publish weekly metrics to Firebase.
        
        Args:
            metrics: Computed metrics dict
            week: Week identifier (e.g., '2026-W11')
        """
        if week is None:
            week = datetime.now().strftime('%Y-W%W')
        
        try:
            # Write to performance history
            self.firebase.write(f'performance/weekly_metrics/{week}', 'metrics', metrics)
            
            # Update latest metrics in Realtime DB
            self.firebase.update_realtime('/performance/latest', {
                'week': week,
                'sharpe': metrics.get('sharpe', 0),
                'win_rate': metrics.get('win_rate', 0),
                'total_trades': metrics.get('total_trades', 0),
                'total_pnl': metrics.get('total_pnl', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'last_update': datetime.now().isoformat()
            })
            
            logger.info(f"Published weekly metrics for {week}")
            
        except Exception as e:
            logger.error(f"Failed to publish metrics: {e}")
    
    def publish_drift_report(self, drift_report: Dict[str, Any]) -> None:
        """Publish drift detection report to Firebase."""
        try:
            self.firebase.update_realtime('/performance/model_drift', {
                'needs_refit': drift_report.get('needs_refit', False),
                'drift_score': drift_report.get('drift_score', 0),
                'feature_drifts': drift_report.get('feature_drifts', {}),
                'last_check': datetime.now().isoformat()
            })
            
            if drift_report.get('needs_refit'):
                logger.warning("Model drift detected - retraining recommended")
            
        except Exception as e:
            logger.error(f"Failed to publish drift report: {e}")
    
    def get_performance_history(self, weeks: int = 12) -> List[Dict[str, Any]]:
        """Get performance history for the last N weeks."""
        history = []
        
        try:
            # This would read from Firebase
            # For now, return empty list
            pass
        except Exception as e:
            logger.error(f"Failed to fetch performance history: {e}")
        
        return history
    
    def save_local_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to local file for backup."""
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.metrics_path.exists():
            with open(self.metrics_path, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(metrics)
        
        # Keep last 52 weeks
        history = history[-52:]
        
        with open(self.metrics_path, 'w') as f:
            json.dump(history, f, indent=2)


def create_performance_monitor() -> PerformanceMonitor:
    """Factory function to create performance monitor."""
    return PerformanceMonitor()
