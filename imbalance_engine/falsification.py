"""Falsification Discipline — Every Petroulas thesis must die cleanly if wrong.

"Small wrong losses protect the capital for the correct structural calls."

Architecture:
  - Every PetroulsasDecision creates a FalsificationEntry in the trade log
  - Weekly automated check: has the kill test triggered?
  - If triggered → immediate exit signal, trade recorded as loss
  - Win/loss record feeds into Kelly fraction calculation
  - No thesis survives beyond its 30-day horizon without review

The discipline enforces Popper on every position:
  "If your thesis cannot be falsified, it is not a thesis — it is a narrative."
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

FALSIFICATION_LOG_PATH = Path('data/petroulas_log.json')


# =============================================================================
# Contracts
# =============================================================================

@dataclass
class FalsificationEntry:
    """Single Petroulas thesis in the log."""
    thesis_id: str
    symbol: str
    direction: str
    entry_date: str
    entry_price: float
    
    # Thesis
    consensus_blindspot: str      # What consensus missed
    arithmetic_proof: str         # The specific math
    falsification_test: str       # Observable kill test (30d window)
    time_horizon_days: int
    deadline_date: str            # entry_date + time_horizon_days
    
    # Scores
    magnitude: int                # Kimi magnitude (or 0 if framework-only)
    conviction: int               # Kimi conviction (or 0 if framework-only)
    fault_quality: float          # composite
    position_size_pct: float
    
    # Status tracking
    status: str = 'ACTIVE'        # 'ACTIVE' | 'FALSIFIED' | 'CONFIRMED' | 'EXPIRED'
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    falsification_triggered: bool = False
    falsification_reason: Optional[str] = None
    
    # Weekly check log
    weekly_checks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class KellyStats:
    """Win rate and Kelly fraction for Petroulas trades."""
    total_trades: int
    wins: int
    losses: int
    win_rate: float              # empirical win rate
    avg_win_pct: float           # average winning trade %
    avg_loss_pct: float          # average losing trade %
    kelly_fraction: float        # full Kelly (use 0.25x in practice)
    quarter_kelly: float         # recommended bet size
    expected_value: float        # per trade EV


# =============================================================================
# Main Discipline Engine
# =============================================================================

class FalsificationDiscipline:
    """Manages the Petroulas trade log and falsification checking.
    
    Usage:
        discipline = FalsificationDiscipline()
        
        # On trade entry (from PetroulsasGate):
        entry = discipline.open_thesis(
            decision=petroulas_decision,
            kimi_score=kimi_score
        )
        
        # Weekly scheduled check:
        kills = discipline.weekly_check(current_market_data)
        for kill in kills:
            emit_exit_signal(kill.symbol)
        
        # On trade close:
        discipline.close_thesis(thesis_id, exit_price)
        
        # Kelly stats for position sizing:
        stats = discipline.get_kelly_stats()
        print(f"Win rate: {stats.win_rate:.1%} | Quarter Kelly: {stats.quarter_kelly:.1%}")
    """
    
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or FALSIFICATION_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log: List[FalsificationEntry] = self._load_log()
    
    # -------------------------------------------------------------------------
    # Opening a thesis
    # -------------------------------------------------------------------------
    
    def open_thesis(
        self,
        decision,            # PetroulsasDecision
        kimi_score=None      # KimiFaultScore or None
    ) -> FalsificationEntry:
        """Register a new Petroulas thesis in the log."""
        
        entry_date = date.today()
        horizon = kimi_score.time_horizon_days if kimi_score else 30
        deadline = entry_date + timedelta(days=horizon)
        
        entry = FalsificationEntry(
            thesis_id=decision.thesis_id,
            symbol=decision.symbol,
            direction='',   # filled in by caller
            entry_date=entry_date.isoformat(),
            entry_price=0.0,  # filled in by caller
            consensus_blindspot=kimi_score.consensus_blindspot if kimi_score else 'N/A',
            arithmetic_proof=kimi_score.arithmetic_proof if kimi_score else 'Framework-only',
            falsification_test=kimi_score.falsification_test if kimi_score else 'Price breaks entry by > 2%',
            time_horizon_days=horizon,
            deadline_date=deadline.isoformat(),
            magnitude=kimi_score.magnitude if kimi_score else 0,
            conviction=kimi_score.conviction if kimi_score else 0,
            fault_quality=decision.fault_quality,
            position_size_pct=decision.position_size_pct,
            status='ACTIVE'
        )
        
        self._log.append(entry)
        self._save_log()
        
        logger.info(
            f"[Falsification] Opened thesis {entry.thesis_id} | "
            f"{entry.symbol} | Deadline: {entry.deadline_date} | "
            f"Kill test: {entry.falsification_test[:80]}"
        )
        
        return entry
    
    def set_entry_details(self, thesis_id: str, direction: str, entry_price: float):
        """Update direction and price after execution confirms."""
        for entry in self._log:
            if entry.thesis_id == thesis_id:
                entry.direction = direction
                entry.entry_price = entry_price
                self._save_log()
                return
        logger.warning(f"[Falsification] Thesis {thesis_id} not found")
    
    # -------------------------------------------------------------------------
    # Weekly falsification check
    # -------------------------------------------------------------------------
    
    def weekly_check(self, market_snapshot: Dict[str, Any]) -> List[FalsificationEntry]:
        """Run weekly falsification check across all active theses.
        
        Args:
            market_snapshot: dict with current market data
                {
                    'prices': {'SPY': 520.0, 'AAPL': 185.0, ...},
                    'yield_curve_2_10': -15,  # bps
                    'vix': 22.0,
                    'date': '2026-04-08'
                }
        
        Returns:
            List of FalsificationEntry objects that triggered kill tests
            → Caller should emit exit signals for these.
        """
        today = date.today()
        kills: List[FalsificationEntry] = []
        
        for entry in self._log:
            if entry.status != 'ACTIVE':
                continue
            
            # Deadline check
            deadline = date.fromisoformat(entry.deadline_date)
            if today > deadline:
                entry.status = 'EXPIRED'
                entry.falsification_reason = f"Thesis expired past deadline {entry.deadline_date}"
                kills.append(entry)
                logger.warning(f"[Falsification] {entry.symbol} thesis EXPIRED: {entry.thesis_id}")
                continue
            
            # Price-based falsification (simple rule: if price moved against us beyond 2%)
            prices = market_snapshot.get('prices', {})
            current_price = prices.get(entry.symbol)
            
            if current_price and entry.entry_price > 0:
                long_falsified = (
                    entry.direction == 'LONG' and 
                    current_price < entry.entry_price * 0.98
                )
                short_falsified = (
                    entry.direction == 'SHORT' and 
                    current_price > entry.entry_price * 1.02
                )
                
                if long_falsified or short_falsified:
                    entry.status = 'FALSIFIED'
                    entry.falsification_triggered = True
                    entry.falsification_reason = (
                        f"Price breach: entry={entry.entry_price:.2f}, "
                        f"current={current_price:.2f} (>{2:.0f}% adverse)"
                    )
                    kills.append(entry)
                    logger.warning(
                        f"[Falsification] KILL SIGNAL: {entry.symbol} | "
                        f"{entry.falsification_reason}"
                    )
            
            # Record weekly check
            entry.weekly_checks.append({
                'date': today.isoformat(),
                'price': current_price,
                'vix': market_snapshot.get('vix'),
                'yield_curve': market_snapshot.get('yield_curve_2_10'),
                'status': entry.status
            })
        
        self._save_log()
        return kills
    
    # -------------------------------------------------------------------------
    # Closing a thesis
    # -------------------------------------------------------------------------
    
    def close_thesis(
        self,
        thesis_id: str,
        exit_price: float,
        forced_by_falsification: bool = False
    ) -> Optional[FalsificationEntry]:
        """Record exit and compute P&L."""
        for entry in self._log:
            if entry.thesis_id == thesis_id and entry.status == 'ACTIVE':
                entry.exit_date = date.today().isoformat()
                entry.exit_price = exit_price
                
                if entry.entry_price > 0:
                    if entry.direction == 'LONG':
                        entry.pnl_pct = (exit_price - entry.entry_price) / entry.entry_price * 100
                    else:
                        entry.pnl_pct = (entry.entry_price - exit_price) / entry.entry_price * 100
                
                if forced_by_falsification or entry.falsification_triggered:
                    entry.status = 'FALSIFIED'
                elif entry.pnl_pct and entry.pnl_pct > 0:
                    entry.status = 'CONFIRMED'   # Thesis confirmed, got paid
                else:
                    entry.status = 'FALSIFIED'
                
                self._save_log()
                
                logger.info(
                    f"[Falsification] Closed {thesis_id} | "
                    f"Status: {entry.status} | P&L: {entry.pnl_pct:.2f}%"
                )
                return entry
        
        logger.warning(f"[Falsification] Could not find active thesis {thesis_id}")
        return None
    
    # -------------------------------------------------------------------------
    # Kelly Stats
    # -------------------------------------------------------------------------
    
    def get_kelly_stats(self) -> KellyStats:
        """Compute empirical win rate and Kelly fraction from closed Petroulas trades.
        
        Kelly Criterion: f* = (bp - q) / b
        where:
            b = avg_win / avg_loss (odds ratio)
            p = win rate
            q = 1 - p
        
        Use quarter-Kelly (f*/4) to be conservative.
        """
        closed = [e for e in self._log if e.status in ('CONFIRMED', 'FALSIFIED') and e.pnl_pct is not None]
        
        if not closed:
            return KellyStats(
                total_trades=0, wins=0, losses=0, win_rate=0.0,
                avg_win_pct=0.0, avg_loss_pct=0.0,
                kelly_fraction=0.0, quarter_kelly=0.02,
                expected_value=0.0
            )
        
        wins = [e for e in closed if e.pnl_pct > 0]
        losses = [e for e in closed if e.pnl_pct <= 0]
        
        win_rate = len(wins) / len(closed)
        avg_win = sum(e.pnl_pct for e in wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(e.pnl_pct for e in losses) / len(losses)) if losses else 1.0
        
        if avg_loss == 0:
            avg_loss = 1.0
        
        b = avg_win / avg_loss        # odds ratio
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b if b > 0 else 0.0
        kelly = max(0.0, kelly)       # never bet negative
        quarter_k = kelly / 4.0
        
        ev = p * avg_win - q * avg_loss
        
        return KellyStats(
            total_trades=len(closed),
            wins=len(wins),
            losses=len(losses),
            win_rate=round(win_rate, 4),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            kelly_fraction=round(kelly, 4),
            quarter_kelly=round(quarter_k, 4),
            expected_value=round(ev, 3)
        )
    
    def get_active_theses(self) -> List[FalsificationEntry]:
        return [e for e in self._log if e.status == 'ACTIVE']
    
    def get_all_theses(self) -> List[FalsificationEntry]:
        return list(self._log)
    
    def print_report(self):
        """Print full Petroulas trade log."""
        kelly = self.get_kelly_stats()
        active = self.get_active_theses()
        
        print("\n" + "=" * 70)
        print("PETROULAS TRADE LOG — FALSIFICATION REPORT")
        print("=" * 70)
        print(f"Total Trades: {kelly.total_trades} | Wins: {kelly.wins} | Losses: {kelly.losses}")
        print(f"Win Rate: {kelly.win_rate:.1%} | Avg Win: +{kelly.avg_win_pct:.1f}% | Avg Loss: -{kelly.avg_loss_pct:.1f}%")
        print(f"Kelly Fraction: {kelly.kelly_fraction:.1%} | Quarter-Kelly: {kelly.quarter_kelly:.1%}")
        print(f"Expected Value per trade: {kelly.expected_value:+.2f}%")
        print(f"\nActive Theses: {len(active)}")
        for t in active:
            print(f"  [{t.symbol}] {t.thesis_id} | Deadline: {t.deadline_date} | Size: {t.position_size_pct:.1f}%")
            print(f"    Proof: {t.arithmetic_proof[:80]}")
            print(f"    Kill test: {t.falsification_test[:80]}")
        print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    
    def _load_log(self) -> List[FalsificationEntry]:
        if not self.log_path.exists():
            return []
        try:
            with open(self.log_path, 'r') as f:
                data = json.load(f)
            entries = []
            for d in data:
                # Reconstruct dataclass from dict
                e = FalsificationEntry(**d)
                entries.append(e)
            return entries
        except Exception as ex:
            logger.error(f"[Falsification] Failed to load log: {ex}")
            return []
    
    def _save_log(self):
        try:
            with open(self.log_path, 'w') as f:
                json.dump([asdict(e) for e in self._log], f, indent=2, default=str)
        except Exception as ex:
            logger.error(f"[Falsification] Failed to save log: {ex}")
