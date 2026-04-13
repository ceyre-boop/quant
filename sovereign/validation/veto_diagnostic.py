"""
Phase 9 — Veto Rate Diagnostic (V1.0)
Analyzes veto ledger to ensure system is not over-filtering.
Generates diagnostic report with filter health metrics.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VetoDiagnostic:
    """Results of veto rate diagnostic."""
    total_signals_processed: int
    total_vetos: int
    execution_rate: float  # % of signals that became trades
    veto_breakdown: Dict[str, int]
    healthy_bounds: Dict[str, int]
    violations: List[str]
    health_score: float  # 0-100
    is_healthy: bool
    timestamp: str


class VetoRateDiagnostic:
    """
    Phase 9 Diagnostic Tool
    
    Analyzes veto ledger to detect:
    - Over-filtering (too many signals rejected)
    - Filter imbalance (one stage blocking too much)
    - Signal loss (healthy opportunities discarded)
    """
    
    # Healthy bounds per spec
    HEALTHY_BOUNDS = {
        'PETROULAS': {'max': 5, 'description': 'Macro fault rate'},
        'ROUTER/FLAT': {'max': 40, 'description': 'Dead zone filtering'},
        'ROUTER': {'max': 35, 'description': 'All router vetos'},
        'SPECIALIST': {'max': 10, 'description': 'Neutral bias rate'},
        'RISK/EV': {'max': 20, 'description': 'Risk gate blocks'},
        'GAME': {'max': 5, 'description': 'Game theory vetos'},
        'HARD_CONSTRAINT': {'max': 15, 'description': 'Hard limit blocks'},
    }
    
    def __init__(self, ledger_path: Optional[Path] = None):
        self.ledger_path = ledger_path or Path('data/ledger')
        self.ledger_path.mkdir(parents=True, exist_ok=True)
        
    def run_diagnostic(self, days: int = 30) -> VetoDiagnostic:
        """
        Run full veto diagnostic over specified period.
        
        Args:
            days: Lookback period in days
            
        Returns:
            VetoDiagnostic with full analysis
        """
        logger.info(f"Running Veto Rate Diagnostic ({days} days)...")
        
        # Collect all vetos and estimates
        vetos = self._collect_vetos(days)
        estimated_signals = self._estimate_total_signals(days)
        
        # Analyze breakdown
        veto_breakdown = self._analyze_breakdown(vetos)
        
        # Check bounds
        violations = self._check_bounds(veto_breakdown, estimated_signals)
        
        # Calculate health metrics
        execution_rate = self._calculate_execution_rate(vetos, estimated_signals)
        health_score = self._calculate_health_score(veto_breakdown, violations)
        
        diagnostic = VetoDiagnostic(
            total_signals_processed=estimated_signals,
            total_vetos=len(vetos),
            execution_rate=execution_rate,
            veto_breakdown=veto_breakdown,
            healthy_bounds={k: v['max'] for k, v in self.HEALTHY_BOUNDS.items()},
            violations=violations,
            health_score=health_score,
            is_healthy=len(violations) == 0 and health_score >= 80,
            timestamp=datetime.utcnow().isoformat()
        )
        
        self._log_report(diagnostic)
        return diagnostic
    
    def _collect_vetos(self, days: int) -> List[Dict]:
        """Collect all veto records from ledger files."""
        vetos = []
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Check current and previous month files
        for month_offset in [0, -1]:
            month_date = datetime.utcnow() + timedelta(days=month_offset * 30)
            month_str = month_date.strftime('%Y_%m')
            log_file = self.ledger_path / f'veto_ledger_{month_str}.jsonl'
            
            if not log_file.exists():
                continue
                
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry['timestamp'])
                        if entry_time >= cutoff:
                            vetos.append(entry)
            except Exception as e:
                logger.warning(f"Error reading {log_file}: {e}")
        
        return vetos
    
    def _estimate_total_signals(self, days: int) -> int:
        """
        Estimate total signals processed.
        
        For Sovereign: signals = vetos + executed trades
        """
        # Count vetos
        vetos = len(self._collect_vetos(days))
        
        # Count executed trades from trade ledger
        trades = self._count_trades(days)
        
        return vetos + trades
    
    def _count_trades(self, days: int) -> int:
        """Count executed trades from trade ledger."""
        trades = 0
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        for month_offset in [0, -1]:
            month_date = datetime.utcnow() + timedelta(days=month_offset * 30)
            month_str = month_date.strftime('%Y_%m')
            log_file = self.ledger_path / f'trade_ledger_{month_str}.jsonl'
            
            if not log_file.exists():
                continue
                
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry['entry_time'])
                        if entry_time >= cutoff:
                            trades += 1
            except Exception as e:
                logger.warning(f"Error reading trade ledger: {e}")
        
        return trades
    
    def _analyze_breakdown(self, vetos: List[Dict]) -> Dict[str, int]:
        """Analyze veto counts by stage."""
        counts = Counter()
        
        for veto in vetos:
            stage = veto.get('stage', 'UNKNOWN')
            # Normalize stage names
            if 'ROUTER' in stage:
                counts['ROUTER'] += 1
            counts[stage] += 1
        
        return dict(counts)
    
    def _check_bounds(self, breakdown: Dict[str, int], total_signals: int) -> List[str]:
        """Check if veto rates exceed healthy bounds."""
        violations = []
        
        for stage, count in breakdown.items():
            # Find matching bound
            bound_key = None
            for key in self.HEALTHY_BOUNDS:
                if key in stage or stage in key:
                    bound_key = key
                    break
            
            if not bound_key:
                continue
                
            max_allowed = self.HEALTHY_BOUNDS[bound_key]['max']
            
            if count > max_allowed:
                pct = (count / max(total_signals, 1)) * 100
                violations.append(
                    f"{stage}: {count} vetos (max {max_allowed}) — "
                    f"{pct:.1f}% of signals"
                )
        
        # Check execution rate
        total_vetos = sum(breakdown.values())
        execution_rate = ((total_signals - total_vetos) / max(total_signals, 1)) * 100
        
        if execution_rate < 5:  # Less than 5% execution = over-filtering
            violations.append(
                f"LOW_EXECUTION_RATE: {execution_rate:.1f}% signals executed "
                f"(system may be over-filtering)"
            )
        
        return violations
    
    def _calculate_execution_rate(self, vetos: List[Dict], total_signals: int) -> float:
        """Calculate percentage of signals that became trades."""
        if total_signals == 0:
            return 0.0
        executed = total_signals - len(vetos)
        return (executed / total_signals) * 100
    
    def _calculate_health_score(self, breakdown: Dict[str, int], violations: List[str]) -> float:
        """Calculate overall filter health score (0-100)."""
        base_score = 100.0
        
        # Deduct for violations
        base_score -= len(violations) * 15
        
        # Deduct for high veto rates
        total_vetos = sum(breakdown.values())
        if total_vetos > 0:
            # Check if any single filter dominates
            max_stage = max(breakdown.items(), key=lambda x: x[1])
            max_pct = max_stage[1] / total_vetos
            if max_pct > 0.5:  # One filter doing >50% of vetos
                base_score -= 20
        
        return max(0.0, base_score)
    
    def _log_report(self, diagnostic: VetoDiagnostic):
        """Log diagnostic report."""
        logger.info("\n" + "=" * 60)
        logger.info("SOVEREIGN VETO RATE DIAGNOSTIC REPORT")
        logger.info("=" * 60)
        logger.info(f"Period: Last 30 days")
        logger.info(f"Signals Processed: {diagnostic.total_signals_processed}")
        logger.info(f"Total Vetos: {diagnostic.total_vetos}")
        logger.info(f"Execution Rate: {diagnostic.execution_rate:.1f}%")
        logger.info(f"Health Score: {diagnostic.health_score:.0f}/100")
        logger.info(f"Status: {'✅ HEALTHY' if diagnostic.is_healthy else '⚠️ NEEDS ATTENTION'}")
        logger.info("-" * 60)
        logger.info("Veto Breakdown:")
        for stage, count in sorted(diagnostic.veto_breakdown.items()):
            bound = diagnostic.healthy_bounds.get(stage, '?')
            status = "✅" if count <= bound else "⚠️"
            logger.info(f"  {stage:20s}: {count:4d} (max: {bound}) {status}")
        logger.info("-" * 60)
        
        if diagnostic.violations:
            logger.info("⚠️ VIOLATIONS DETECTED:")
            for v in diagnostic.violations:
                logger.info(f"  • {v}")
        else:
            logger.info("✅ All filter rates within healthy bounds")
        
        logger.info("=" * 60)
    
    def generate_report_file(self, diagnostic: VetoDiagnostic, path: Optional[Path] = None):
        """Save diagnostic report to file."""
        if path is None:
            path = Path('data/ledger/veto_diagnostic_latest.json')
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': diagnostic.timestamp,
            'health_score': diagnostic.health_score,
            'is_healthy': diagnostic.is_healthy,
            'total_signals': diagnostic.total_signals_processed,
            'total_vetos': diagnostic.total_vetos,
            'execution_rate': diagnostic.execution_rate,
            'veto_breakdown': diagnostic.veto_breakdown,
            'violations': diagnostic.violations
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Diagnostic report saved to {path}")


# ─── Standalone execution ─────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    diagnostic = VetoRateDiagnostic()
    result = diagnostic.run_diagnostic(days=30)
    diagnostic.generate_report_file(result)
    
    # Exit code based on health
    exit(0 if result.is_healthy else 1)
