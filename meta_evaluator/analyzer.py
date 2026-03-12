"""Meta-Evaluator - Weekly model performance analysis and retraining scheduler.

Tracks model drift, feature importance changes, regime shifts, and auto-retraining.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class WeeklyAnalyzer:
    """Weekly analysis of model performance."""
    
    def __init__(self, firebase_client=None):
        self.firebase = firebase_client
        self.performance_history = []
    
    def analyze_week(self, trade_records: List[Dict], start_date: datetime) -> Dict[str, Any]:
        """Analyze one week of trading performance.
        
        Returns:
            Dict with win_rate, sharpe, max_drawdown, profit_factor
        """
        if not trade_records:
            return {
                'week_start': start_date.isoformat(),
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_loss': 0.0,
                'sharpe': 0.0,
                'status': 'NO_DATA'
            }
        
        wins = [t for t in trade_records if t.get('realized_pnl', 0) > 0]
        losses = [t for t in trade_records if t.get('realized_pnl', 0) <= 0]
        
        win_rate = len(wins) / len(trade_records) if trade_records else 0
        total_pnl = sum(t.get('realized_pnl', 0) for t in trade_records)
        
        # Calculate Sharpe (simplified)
        returns = [t.get('realized_pnl', 0) for t in trade_records]
        avg_return = sum(returns) / len(returns) if returns else 0
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns) if returns else 0
        std_dev = variance ** 0.5
        sharpe = (avg_return / std_dev) if std_dev > 0 else 0
        
        analysis = {
            'week_start': start_date.isoformat(),
            'total_trades': len(trade_records),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': round(win_rate, 4),
            'profit_loss': round(total_pnl, 2),
            'sharpe': round(sharpe, 2),
            'avg_win': round(sum(t.get('realized_pnl', 0) for t in wins) / len(wins), 2) if wins else 0,
            'avg_loss': round(sum(t.get('realized_pnl', 0) for t in losses) / len(losses), 2) if losses else 0,
            'status': 'HEALTHY' if win_rate > 0.5 and sharpe > 0.5 else 'DEGRADED'
        }
        
        self.performance_history.append(analysis)
        logger.info(f"Weekly analysis: {analysis['win_rate']:.1%} win rate, PnL: ${analysis['profit_loss']:.2f}")
        
        return analysis


class FeatureGroupTracker:
    """Track feature importance and detect drift."""
    
    def __init__(self):
        self.baseline_importance = {}
        self.current_importance = {}
        self.drift_threshold = 0.2  # 20% change triggers alert
    
    def set_baseline(self, feature_importance: Dict[str, float]):
        """Set baseline feature importance from training."""
        self.baseline_importance = feature_importance.copy()
        logger.info(f"Baseline set for {len(feature_importance)} feature groups")
    
    def update_current(self, feature_importance: Dict[str, float]):
        """Update current feature importance from recent predictions."""
        self.current_importance = feature_importance.copy()
    
    def detect_drift(self) -> Dict[str, Any]:
        """Detect feature importance drift from baseline.
        
        Returns:
            Dict with drift status and details per feature group
        """
        if not self.baseline_importance:
            return {'status': 'NO_BASELINE', 'drift_detected': False}
        
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': False,
            'feature_drifts': [],
            'status': 'HEALTHY'
        }
        
        for feature, baseline_val in self.baseline_importance.items():
            current_val = self.current_importance.get(feature, 0)
            if baseline_val > 0:
                change_pct = abs(current_val - baseline_val) / baseline_val
                
                if change_pct > self.drift_threshold:
                    drift_report['drift_detected'] = True
                    drift_report['feature_drifts'].append({
                        'feature_group': feature,
                        'baseline': round(baseline_val, 4),
                        'current': round(current_val, 4),
                        'change_pct': round(change_pct, 2),
                        'severity': 'HIGH' if change_pct > 0.5 else 'MEDIUM'
                    })
        
        if drift_report['drift_detected']:
            drift_report['status'] = 'DRIFT_DETECTED'
            logger.warning(f"Feature drift detected in {len(drift_report['feature_drifts'])} groups")
        
        return drift_report
    
    def get_top_features(self, n: int = 5) -> List[Dict]:
        """Get top N most important features currently."""
        sorted_features = sorted(
            self.current_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            {'feature_group': name, 'importance': round(val, 4)}
            for name, val in sorted_features[:n]
        ]


class RefitScheduler:
    """Schedule model retraining based on performance and drift."""
    
    def __init__(self, firebase_client=None):
        self.firebase = firebase_client
        self.last_refit = None
        self.min_trades_for_refit = 50
        self.performance_threshold = 0.45  # Win rate below this triggers retraining
    
    def should_refit(self, weekly_analysis: Dict, drift_report: Dict) -> Dict[str, Any]:
        """Determine if model should be retrained.
        
        Returns:
            Dict with recommendation and reasoning
        """
        reasons = []
        should_refit = False
        
        # Check performance degradation
        if weekly_analysis.get('win_rate', 1) < self.performance_threshold:
            should_refit = True
            reasons.append(f"Win rate {weekly_analysis['win_rate']:.1%} below threshold {self.performance_threshold:.1%}")
        
        # Check feature drift
        if drift_report.get('drift_detected'):
            should_refit = True
            high_drifts = [d for d in drift_report.get('feature_drifts', []) if d['severity'] == 'HIGH']
            reasons.append(f"{len(high_drifts)} high-severity feature drifts detected")
        
        # Check minimum trade count
        if weekly_analysis.get('total_trades', 0) < self.min_trades_for_refit:
            should_refit = False
            reasons.append(f"Insufficient trades ({weekly_analysis['total_trades']} < {self.min_trades_for_refit})")
        
        # Check time since last refit (max 30 days)
        if self.last_refit:
            days_since = (datetime.now() - self.last_refit).days
            if days_since > 30:
                should_refit = True
                reasons.append(f"30 days since last refit")
        
        recommendation = {
            'should_refit': should_refit and len(reasons) > 0,
            'timestamp': datetime.now().isoformat(),
            'reasons': reasons,
            'confidence': 'HIGH' if len([r for r in reasons if 'drift' in r]) > 0 else 'MEDIUM',
            'weekly_win_rate': weekly_analysis.get('win_rate'),
            'drift_detected': drift_report.get('drift_detected')
        }
        
        if recommendation['should_refit']:
            logger.info(f"Refit recommended: {reasons}")
        
        return recommendation
    
    def record_refit(self, model_version: str, performance_before: Dict, performance_after: Dict):
        """Record a completed model refit."""
        self.last_refit = datetime.now()
        
        refit_record = {
            'timestamp': self.last_refit.isoformat(),
            'model_version': model_version,
            'performance_before': performance_before,
            'performance_after': performance_after,
            'improvement': {
                'win_rate_delta': round(performance_after.get('win_rate', 0) - performance_before.get('win_rate', 0), 4),
                'sharpe_delta': round(performance_after.get('sharpe', 0) - performance_before.get('sharpe', 0), 2)
            }
        }
        
        logger.info(f"Refit recorded: {model_version}, improvement: {refit_record['improvement']}")
        
        if self.firebase:
            self.firebase.write('refit_history', refit_record)
        
        return refit_record


class MetaEvaluator:
    """Main meta-evaluator orchestrating weekly analysis, drift detection, and refit scheduling."""
    
    def __init__(self, firebase_client=None):
        self.firebase = firebase_client
        self.weekly_analyzer = WeeklyAnalyzer(firebase_client)
        self.feature_tracker = FeatureGroupTracker()
        self.refit_scheduler = RefitScheduler(firebase_client)
    
    def run_weekly_evaluation(self, trade_records: List[Dict], 
                             current_feature_importance: Dict[str, float],
                             start_date: datetime) -> Dict[str, Any]:
        """Run complete weekly evaluation cycle.
        
        Returns:
            Full evaluation report with recommendations
        """
        # Step 1: Weekly performance analysis
        weekly_analysis = self.weekly_analyzer.analyze_week(trade_records, start_date)
        
        # Step 2: Update and check feature drift
        self.feature_tracker.update_current(current_feature_importance)
        drift_report = self.feature_tracker.detect_drift()
        
        # Step 3: Check if refit is needed
        refit_recommendation = self.refit_scheduler.should_refit(weekly_analysis, drift_report)
        
        # Compile full report
        full_report = {
            'evaluation_date': datetime.now().isoformat(),
            'week_start': start_date.isoformat(),
            'performance': weekly_analysis,
            'feature_analysis': {
                'drift': drift_report,
                'top_features': self.feature_tracker.get_top_features(5)
            },
            'recommendations': {
                'refit_model': refit_recommendation['should_refit'],
                'refit_reasons': refit_recommendation['reasons'],
                'refit_confidence': refit_recommendation['confidence']
            },
            'action_required': refit_recommendation['should_refit']
        }
        
        # Save to Firebase if available
        if self.firebase:
            self.firebase.write('meta_evaluations', full_report)
        
        logger.info(f"Weekly evaluation complete. Action required: {full_report['action_required']}")
        
        return full_report
