"""
Phase 11 — Sovereign Paper Trading Runner (V1.0)
30-day validation period. Tracks PnL, counts trades.
Auto-transitions to live after 200 qualifying trades.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from sovereign.orchestrator import SovereignOrchestrator
from sovereign.ledger.trade_ledger import TradeLedger
from sovereign.ledger.veto_ledger import VetoLedger
from sovereign.validation.veto_diagnostic import VetoRateDiagnostic
from contracts.types import SovereignFeatureRecord, MarketData
from config.loader import params

logger = logging.getLogger(__name__)


@dataclass
class PaperTradingStatus:
    """Current status of paper trading campaign."""
    is_running: bool
    start_date: str
    days_elapsed: int
    total_trades: int
    win_rate: float
    total_pnl: float
    total_return_pct: float
    max_drawdown: float
    trades_to_live: int  # 200 - total_trades
    daily_pnl: float
    last_trade_date: Optional[str]
    health_checks: List[str]
    can_go_live: bool


class PaperTradingRunner:
    """
    Phase 11 — 30-Day Paper Trading Campaign
    
    Responsibilities:
    1. Run Sovereign in paper mode
    2. Track every trade for 30 days
    3. Count trades toward 200-trade live threshold
    4. Run daily health checks (veto diagnostic)
    5. Auto-evaluate live-readiness
    
    Transition to live requires:
    - 30 days elapsed
    - 200+ trades
    - Positive equity curve
    - Veto diagnostic healthy
    """
    
    # Minimum requirements for live transition
    MIN_TRADES_FOR_LIVE = 200
    MIN_DAYS_FOR_LIVE = 30
    
    def __init__(
        self,
        symbols: List[str],
        starting_equity: float = 100000.0,
        campaign_name: Optional[str] = None
    ):
        self.symbols = symbols
        self.starting_equity = starting_equity
        self.campaign_name = campaign_name or f"paper_{datetime.utcnow().strftime('%Y%m%d')}"
        
        # Core components
        self.orchestrator = SovereignOrchestrator(mode='paper')
        self.trade_ledger = TradeLedger()
        self.veto_ledger = VetoLedger()
        self.diagnostic = VetoRateDiagnostic()
        
        # State tracking
        self.state_file = Path(f'data/paper_trading/{self.campaign_name}_state.json')
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        
    def _load_state(self) -> Dict:
        """Load campaign state from disk."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'campaign_name': self.campaign_name,
            'started_at': datetime.utcnow().isoformat(),
            'is_running': False,
            'daily_summaries': [],
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'peak_equity': self.starting_equity,
            'current_equity': self.starting_equity,
            'max_drawdown': 0.0,
            'health_check_history': []
        }
    
    def _save_state(self):
        """Persist campaign state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def start(self):
        """Start the 30-day paper trading campaign."""
        if self.state['is_running']:
            logger.info("Paper trading campaign already running")
            return
        
        self.state['is_running'] = True
        self.state['started_at'] = datetime.utcnow().isoformat()
        self._save_state()
        
        logger.info("=" * 60)
        logger.info("SOVEREIGN PAPER TRADING — CAMPAIGN START")
        logger.info("=" * 60)
        logger.info(f"Campaign: {self.campaign_name}")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Starting Equity: ${self.starting_equity:,.2f}")
        logger.info(f"Live Threshold: {self.MIN_TRADES_FOR_LIVE} trades + {self.MIN_DAYS_FOR_LIVE} days")
        logger.info("=" * 60)
    
    def run_daily_cycle(self):
        """
        Execute one full daily trading cycle.
        Called once per trading day (e.g., via cron or scheduler).
        """
        if not self.state['is_running']:
            logger.warning("Campaign not running. Call start() first.")
            return
        
        today = datetime.utcnow().date()
        logger.info(f"\n--- DAILY CYCLE: {today} ---")
        
        # 1. Count current stats
        trade_count = self._count_recent_trades(days=30)
        win_count = self._count_wins(days=30)
        daily_pnl = self._calculate_daily_pnl()
        
        # Update state
        self.state['total_trades'] = trade_count
        self.state['winning_trades'] = win_count
        self.state['total_pnl'] += daily_pnl
        self.state['current_equity'] = self.starting_equity + self.state['total_pnl']
        
        # Update peak/drawdown
        if self.state['current_equity'] > self.state['peak_equity']:
            self.state['peak_equity'] = self.state['current_equity']
        
        dd = (self.state['peak_equity'] - self.state['current_equity']) / self.state['peak_equity']
        if dd > self.state['max_drawdown']:
            self.state['max_drawdown'] = dd
        
        # 2. Optional veto diagnostic (informational only, not a gate)
        veto_score = None
        try:
            veto_result = self.diagnostic.run_diagnostic(days=7)
            veto_score = veto_result.health_score
            self.state['health_check_history'].append({
                'date': str(today),
                'health_score': veto_result.health_score,
                'is_healthy': veto_result.is_healthy
            })
        except Exception as e:
            logger.warning(f"Veto diagnostic failed (non-blocking): {e}")
        
        # 3. Record daily summary
        summary = {
            'date': str(today),
            'trades_today': self._count_recent_trades(days=1),
            'daily_pnl': daily_pnl,
            'total_trades': trade_count,
            'total_pnl': self.state['total_pnl'],
            'current_equity': self.state['current_equity'],
            'win_rate': (win_count / max(trade_count, 1)),
            'veto_health_score': veto_score,
            'max_drawdown': self.state['max_drawdown']
        }
        
        self.state['daily_summaries'].append(summary)
        
        self._save_state()
        
        # 4. Log summary
        self._log_daily_summary(summary)
        
        # 5. Check live readiness
        if self._can_go_live():
            logger.info("🚀 LIVE TRANSITION CRITERIA MET")
            self._generate_live_transition_report()
        
        return summary
    
    def _count_recent_trades(self, days: int) -> int:
        """Count trades from ledger in last N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        count = 0
        
        for month_offset in [0, -1, -2]:
            month_date = datetime.utcnow() + timedelta(days=month_offset * 30)
            month_str = month_date.strftime('%Y_%m')
            log_file = Path('data/ledger') / f'trade_ledger_{month_str}.jsonl'
            
            if not log_file.exists():
                continue
            
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry['entry_time'])
                        if entry_time >= cutoff:
                            count += 1
            except Exception as e:
                logger.warning(f"Error reading trade ledger: {e}")
        
        return count
    
    def _count_wins(self, days: int) -> int:
        """Count winning trades from closed positions."""
        # This requires access to closed trade PnL
        # For now, we'll estimate from the paper trading engine summary
        # In production, this would scan for trade closure records
        # Returning a reasonable estimate based on trade ledger entries
        # that have been matched with closes
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        wins = 0
        total = 0
        
        # Check paper_trades log if it exists
        paper_log = Path('data/ledger/paper_trades.jsonl')
        if paper_log.exists():
            try:
                with open(paper_log, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        if entry.get('status') != 'CLOSED':
                            continue
                        entry_time = datetime.fromisoformat(entry['entry_time'])
                        if entry_time >= cutoff:
                            total += 1
                            if entry.get('pnl', 0) > 0:
                                wins += 1
            except Exception:
                pass
        
        return wins
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate PnL from today's closed trades."""
        today = datetime.utcnow().date()
        pnl = 0.0
        
        paper_log = Path('data/ledger/paper_trades.jsonl')
        if paper_log.exists():
            try:
                with open(paper_log, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        if entry.get('status') != 'CLOSED':
                            continue
                        close_time = entry.get('exit_time')
                        if close_time:
                            close_date = datetime.fromisoformat(close_time).date()
                            if close_date == today:
                                pnl += entry.get('pnl', 0)
            except Exception:
                pass
        
        return pnl
    
    def _can_go_live(self) -> bool:
        """Check if all criteria for live trading are met."""
        start = datetime.fromisoformat(self.state['started_at'])
        days_elapsed = (datetime.utcnow() - start).days
        
        # Core criteria (per teardown: no veto diagnostic requirement)
        criteria = [
            days_elapsed >= self.MIN_DAYS_FOR_LIVE,
            self.state['total_trades'] >= self.MIN_TRADES_FOR_LIVE,
            self.state['total_pnl'] > 0,  # Positive equity curve
            self.state['max_drawdown'] < 0.10,  # <10% drawdown
        ]
        
        return all(criteria)
    
    def get_status(self) -> PaperTradingStatus:
        """Get current campaign status."""
        start = datetime.fromisoformat(self.state['started_at'])
        days_elapsed = (datetime.utcnow() - start).days
        
        trade_count = self.state['total_trades']
        win_count = self.state['winning_trades']
        win_rate = win_count / max(trade_count, 1)
        
        last_trade = None
        if self.state['daily_summaries']:
            last_trade = self.state['daily_summaries'][-1]['date']
        
        health_checks = []
        if not self.state['is_running']:
            health_checks.append("Campaign not started")
        if trade_count < self.MIN_TRADES_FOR_LIVE:
            health_checks.append(f"Need {self.MIN_TRADES_FOR_LIVE - trade_count} more trades")
        if days_elapsed < self.MIN_DAYS_FOR_LIVE:
            health_checks.append(f"Need {self.MIN_DAYS_FOR_LIVE - days_elapsed} more days")
        if self.state['total_pnl'] <= 0:
            health_checks.append("Equity curve not yet positive")
        if self.state['max_drawdown'] >= 0.10:
            health_checks.append("Drawdown exceeds 10% limit")
        
        return PaperTradingStatus(
            is_running=self.state['is_running'],
            start_date=self.state['started_at'],
            days_elapsed=days_elapsed,
            total_trades=trade_count,
            win_rate=win_rate,
            total_pnl=self.state['total_pnl'],
            total_return_pct=self.state['total_pnl'] / self.starting_equity,
            max_drawdown=self.state['max_drawdown'],
            trades_to_live=max(0, self.MIN_TRADES_FOR_LIVE - trade_count),
            daily_pnl=self._calculate_daily_pnl(),
            last_trade_date=last_trade,
            health_checks=health_checks if health_checks else ["All checks passed"],
            can_go_live=self._can_go_live()
        )
    
    def _log_daily_summary(self, summary: Dict):
        """Log daily summary."""
        logger.info("=" * 60)
        logger.info(f"DAILY SUMMARY: {summary['date']}")
        logger.info("=" * 60)
        logger.info(f"Trades Today: {summary['trades_today']}")
        logger.info(f"Daily PnL: ${summary['daily_pnl']:,.2f}")
        logger.info(f"Total Trades: {summary['total_trades']}")
        logger.info(f"Total PnL: ${summary['total_pnl']:,.2f}")
        logger.info(f"Current Equity: ${summary['current_equity']:,.2f}")
        logger.info(f"Win Rate: {summary['win_rate']:.1%}")
        logger.info(f"Max Drawdown: {summary['max_drawdown']:.2%}")
        if summary.get('veto_health_score') is not None:
            logger.info(f"Veto Health: {summary['veto_health_score']:.0f}/100")
        logger.info("=" * 60)
    
    def _generate_live_transition_report(self):
        """Generate report when live criteria are met."""
        report = {
            'campaign_name': self.campaign_name,
            'transition_date': datetime.utcnow().isoformat(),
            'start_date': self.state['started_at'],
            'total_trades': self.state['total_trades'],
            'total_pnl': self.state['total_pnl'],
            'total_return_pct': self.state['total_pnl'] / self.starting_equity,
            'max_drawdown': self.state['max_drawdown'],
            'daily_summaries': self.state['daily_summaries'],
            'approval_status': 'PENDING_HUMAN_REVIEW'
        }
        
        path = self.state_file.parent / f'{self.campaign_name}_live_transition.json'
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n🎯 LIVE TRANSITION REPORT SAVED: {path}")
        logger.info("HUMAN APPROVAL REQUIRED BEFORE SWITCHING TO LIVE MODE")
    
    def stop(self):
        """Stop the paper trading campaign."""
        self.state['is_running'] = False
        self.state['stopped_at'] = datetime.utcnow().isoformat()
        self._save_state()
        logger.info("Paper trading campaign stopped")


# ─── Standalone execution ─────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    # Example: Start paper trading campaign
    runner = PaperTradingRunner(
        symbols=['META', 'PFE', 'UNH'],  # Trinity assets
        starting_equity=100000.0,
        campaign_name='sovereign_paper_v1'
    )
    
    runner.start()
    
    # In production, run_daily_cycle() would be called once per trading day
    # via cron/scheduler at market close
    summary = runner.run_daily_cycle()
    
    status = runner.get_status()
    logger.info(f"\nStatus: {status}")
    logger.info(f"Can Go Live: {status.can_go_live}")
