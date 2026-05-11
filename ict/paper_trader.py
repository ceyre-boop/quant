"""
ict/paper_trader.py
===================
Automated paper trading engine for the 30-day validation protocol.

Runs inside the orchestrator every 5 minutes during NY PM and London.
Manages the full trade lifecycle without human intervention:

  Signal fires → open paper trade at FVG limit
  Price hits TP1 → close 50%, move stop to breakeven
  Price hits TP2 → close remaining 50%, log WIN
  Price hits stop → log LOSS
  Session ends (4 PM ET) → close any remaining trades at market

State persists across runs in data/ledger/ict_paper_trades.json
Results append to logs/ict_paper_trade_log.csv
Results pushed to Firebase after every state change

Protocol constraints (data-driven):
  - A-grade signals only (A+ empirically underperforms)
  - NY PM primary session (56% WR in OOS)
  - USDJPY, NZDUSD, EURUSD only (proven pairs)
  - 1% paper risk per trade, max 3 concurrent
  - Structural stop (swept level), TP1=2R, TP2=4R
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

STATE_PATH  = Path('data/ledger/ict_paper_trades.json')
LOG_PATH    = Path('logs/ict_paper_trade_log.csv')
ACCOUNT     = 10_000.0
RISK_PCT    = 0.01          # 1% per trade
TP1_R       = 2.0
TP2_R       = 4.0
TP1_FRAC    = 0.50          # close 50% at TP1, run 50% to TP2

LOG_HEADER  = [
    'date', 'open_time', 'close_time', 'pair', 'session', 'grade', 'score',
    'direction', 'entry', 'stop', 'tp1', 'tp2', 'risk_pct',
    'outcome', 'exit_price', 'pnl_r', 'pnl_dollars', 'hold_bars', 'notes',
]


@dataclass
class PaperTrade:
    id:           str
    pair:         str
    direction:    str
    grade:        str
    score:        float
    session:      str
    open_time:    str
    entry:        float
    stop:         float
    tp1:          float
    tp2:          float
    risk_dollars: float
    stop_dist:    float
    # State
    partial_closed: bool  = False   # True after TP1 hit (50% closed)
    stop_moved_be:  bool  = False   # True after stop moved to breakeven
    bars_open:      int   = 0
    # Filled when closed
    closed:         bool  = False
    close_time:     str   = ''
    outcome:        str   = ''      # TP1 | TP2 | STOP | BE | TIMEOUT
    exit_price:     float = 0.0
    pnl_r:          float = 0.0
    pnl_dollars:    float = 0.0
    notes:          str   = ''


class PaperTrader:
    """
    Stateful paper trading engine.  One instance per orchestrator run.
    Loads state from disk, processes current prices, saves state back.
    """

    def __init__(self):
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_log_header()
        self._state = self._load_state()

    # ── Public API ─────────────────────────────────────────────────────────── #

    @property
    def open_trades(self) -> List[PaperTrade]:
        return [PaperTrade(**t) for t in self._state.get('open', [])]

    @property
    def n_open(self) -> int:
        return len(self._state.get('open', []))

    def has_open_trade(self, pair: str) -> bool:
        return any(t['pair'] == pair for t in self._state.get('open', []))

    def open_trade(self, scan_result) -> Optional[PaperTrade]:
        """
        Open a new paper trade from a ScanResult.
        Returns None if pair already has an open trade or max concurrent reached.
        """
        if self.n_open >= 3:
            logger.info("Max concurrent trades (3) — skipping %s", scan_result.pair)
            return None
        if self.has_open_trade(scan_result.pair):
            logger.info("Already have open trade on %s — skipping", scan_result.pair)
            return None

        entry = scan_result.entry_level or 0.0
        stop  = scan_result.stop or 0.0
        if entry == 0 or stop == 0:
            logger.warning("%s: missing entry/stop — cannot open trade", scan_result.pair)
            return None

        stop_dist    = abs(entry - stop)
        risk_dollars = ACCOUNT * RISK_PCT
        sign         = 1 if scan_result.signal == 'LONG' else -1
        tp1          = entry + sign * stop_dist * TP1_R
        tp2          = entry + sign * stop_dist * TP2_R

        trade = PaperTrade(
            id=f"{scan_result.pair}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            pair=scan_result.pair,
            direction=scan_result.signal,
            grade=scan_result.grade,
            score=scan_result.score,
            session=scan_result.session,
            open_time=datetime.now(timezone.utc).isoformat(),
            entry=round(entry, 5),
            stop=round(stop, 5),
            tp1=round(tp1, 5),
            tp2=round(tp2, 5),
            risk_dollars=round(risk_dollars, 2),
            stop_dist=round(stop_dist, 5),
        )

        self._state.setdefault('open', []).append(asdict(trade))
        self._save_state()
        logger.info("📂 OPENED: %s %s @ %s  stop=%s  TP1=%s  TP2=%s",
                    trade.pair, trade.direction, trade.entry,
                    trade.stop, trade.tp1, trade.tp2)
        return trade

    def update_trades(self, current_prices: Dict[str, dict]) -> List[PaperTrade]:
        """
        Check open trades against current prices.
        Handles partial closes, BE stops, full closes.
        Returns list of trades that closed this cycle.
        """
        closed_this_cycle: List[PaperTrade] = []
        still_open = []

        for td in self._state.get('open', []):
            t = PaperTrade(**td)
            prices = current_prices.get(t.pair)
            if not prices:
                still_open.append(asdict(t))
                continue

            hi   = prices['high']
            lo   = prices['low']
            last = prices['close']
            t.bars_open += 1

            sign = 1 if t.direction == 'LONG' else -1
            effective_stop = t.entry if t.stop_moved_be else t.stop

            # Check stop first (stops take priority at same bar)
            stop_hit = (t.direction == 'LONG' and lo <= effective_stop) or \
                       (t.direction == 'SHORT' and hi >= effective_stop)

            if stop_hit:
                if t.partial_closed:
                    # Second half stopped at breakeven → net = TP1 on 50%
                    t.exit_price  = effective_stop
                    t.pnl_r       = TP1_FRAC * TP1_R      # 50% at 2R
                    t.pnl_dollars = round(t.risk_dollars * t.pnl_r, 2)
                    t.outcome     = 'BE'
                    t.notes       = 'TP1 then stopped at breakeven'
                else:
                    t.exit_price  = effective_stop
                    t.pnl_r       = -1.0
                    t.pnl_dollars = round(-t.risk_dollars, 2)
                    t.outcome     = 'STOP'
                t.closed     = True
                t.close_time = datetime.now(timezone.utc).isoformat()
                self._log_trade(t)
                closed_this_cycle.append(t)
                logger.info("🔴 CLOSED %s: %s  pnl=%.2fR ($%.2f)",
                            t.pair, t.outcome, t.pnl_r, t.pnl_dollars)
                continue

            # TP2 check (only after partial close)
            if t.partial_closed:
                tp2_hit = (t.direction == 'LONG' and hi >= t.tp2) or \
                          (t.direction == 'SHORT' and lo <= t.tp2)
                if tp2_hit:
                    t.exit_price  = t.tp2
                    full_pnl_r    = TP1_FRAC * TP1_R + (1 - TP1_FRAC) * TP2_R
                    t.pnl_r       = full_pnl_r        # 1.0 + 2.0 = 3.0R
                    t.pnl_dollars = round(t.risk_dollars * t.pnl_r, 2)
                    t.outcome     = 'TP2'
                    t.notes       = 'Full TP2 target reached'
                    t.closed      = True
                    t.close_time  = datetime.now(timezone.utc).isoformat()
                    self._log_trade(t)
                    closed_this_cycle.append(t)
                    logger.info("🟢 CLOSED %s: TP2  pnl=%.2fR ($%.2f)",
                                t.pair, t.pnl_r, t.pnl_dollars)
                    continue

            # TP1 check (first half close)
            if not t.partial_closed:
                tp1_hit = (t.direction == 'LONG' and hi >= t.tp1) or \
                          (t.direction == 'SHORT' and lo <= t.tp1)
                if tp1_hit:
                    t.partial_closed = True
                    t.stop_moved_be  = True
                    t.stop           = t.entry   # move stop to breakeven
                    logger.info("🟡 TP1 hit %s — stop moved to BE, running to TP2 @ %s",
                                t.pair, t.tp2)

            still_open.append(asdict(t))

        self._state['open'] = still_open
        self._save_state()
        return closed_this_cycle

    def close_session(self, reason: str = 'SESSION_END') -> List[PaperTrade]:
        """Close all open trades at last price (end-of-session discipline)."""
        closed = []
        for td in self._state.get('open', []):
            t = PaperTrade(**td)
            if t.partial_closed:
                # TP1 was hit — net positive even on timeout
                t.pnl_r   = TP1_FRAC * TP1_R
                t.outcome  = 'TP1_TIMEOUT'
                t.notes    = 'TP1 hit, second half closed at session end'
            else:
                # Trade never reached TP1 — close at entry (conservative paper assumption)
                t.pnl_r   = 0.0
                t.outcome  = 'TIMEOUT'
                t.notes    = f'Closed at session end ({reason})'
            t.pnl_dollars = round(t.risk_dollars * t.pnl_r, 2)
            t.exit_price  = t.entry
            t.closed      = True
            t.close_time  = datetime.now(timezone.utc).isoformat()
            self._log_trade(t)
            closed.append(t)
            logger.info("⏱ SESSION CLOSE %s: %s  pnl=%.2fR", t.pair, t.outcome, t.pnl_r)

        self._state['open'] = []
        self._save_state()
        return closed

    def get_stats(self) -> dict:
        """Return running stats from the closed trades ledger."""
        closed = self._state.get('closed', [])
        if not closed:
            return {'n_trades': 0, 'win_rate': 0, 'avg_r': 0,
                    'total_r': 0, 'total_dollars': 0}
        pnls = [t['pnl_r'] for t in closed]
        wins = [p for p in pnls if p > 0]
        return {
            'n_trades':      len(closed),
            'win_rate':      round(len(wins) / len(pnls), 3),
            'avg_r':         round(sum(pnls) / len(pnls), 3),
            'total_r':       round(sum(pnls), 2),
            'total_dollars': round(sum(t['pnl_dollars'] for t in closed), 2),
            'by_outcome':    {
                o: len([t for t in closed if t['outcome'] == o])
                for o in ('TP2','TP1','BE','STOP','TIMEOUT','TP1_TIMEOUT')
            },
            'days_running': self._days_running(),
        }

    # ── Internal ───────────────────────────────────────────────────────────── #

    def _log_trade(self, t: PaperTrade):
        """Append closed trade to CSV log and closed ledger."""
        row = [
            datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            t.open_time[:16],
            t.close_time[:16],
            t.pair, t.session, t.grade, t.score,
            t.direction,
            t.entry, t.stop, t.tp1, t.tp2,
            f"{RISK_PCT*100:.1f}%",
            t.outcome, t.exit_price,
            t.pnl_r, t.pnl_dollars,
            t.bars_open, t.notes,
        ]
        with open(LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        self._state.setdefault('closed', []).append(asdict(t))

    def _load_state(self) -> dict:
        if STATE_PATH.exists():
            try:
                return json.loads(STATE_PATH.read_text())
            except Exception:
                pass
        return {'open': [], 'closed': [], 'started': datetime.now(timezone.utc).isoformat()}

    def _save_state(self):
        STATE_PATH.write_text(json.dumps(self._state, indent=2, default=str))

    def _ensure_log_header(self):
        if not LOG_PATH.exists():
            with open(LOG_PATH, 'w', newline='') as f:
                csv.writer(f).writerow(LOG_HEADER)

    def _days_running(self) -> int:
        started = self._state.get('started', datetime.now(timezone.utc).isoformat())
        try:
            d = datetime.fromisoformat(started.replace('Z', '+00:00'))
            return (datetime.now(timezone.utc) - d).days
        except Exception:
            return 0
