"""
ict/orchestrator.py
===================
ICT Micro-Edge Orchestrator — paper trading loop.

Proven universe (1yr backtest, Sharpe 1.47):
  Pairs:   USDJPY, NZDUSD, EURUSD
  Session: NY PM (13:30–16:00 ET) — 53% WR, +0.556R avg
  Grade:   A only (A+ is over-confirmed and runs late)

Usage:
    python3 -m ict.orchestrator --once          # single scan, print results
    python3 -m ict.orchestrator --watch         # run every 5min during NY PM
    python3 -m ict.orchestrator --session all   # include London too

ISOLATION: writes only to signals/ICT_ENGINE/* in Firebase.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ET session end hour — close all paper trades at 4 PM ET
SESSION_CLOSE_HOUR_ET = 16

# ── Proven configuration (data-driven, not theory) ─────────────────────────── #

# Dead pairs removed: AUDUSD (12% WR), GBPUSD (22% WR), GBPJPY (27% WR)
PROVEN_PAIRS = ['USDJPY=X', 'NZDUSD=X', 'EURUSD=X']

# NY PM is the only session with positive EV (53% WR, +0.556R avg)
# London included as secondary (31% WR, slightly negative — monitor only)
ACTIVE_SESSIONS = {'NY_PM', 'London'}
PRIMARY_SESSION = 'NY_PM'   # where real edge lives

FIREBASE_ROOT   = 'signals/ICT_ENGINE'
ACCOUNT_SIZE    = 10_000.0


# ── Result ─────────────────────────────────────────────────────────────────── #

@dataclass
class ScanResult:
    pair:             str
    timestamp:        str
    signal:           str        # LONG | SHORT | FLAT | VETO
    grade:            str
    score:            float
    session:          str
    is_primary:       bool       # True if NY PM
    actionable:       bool       # A-grade in primary session
    entry_level:      Optional[float]
    stop:             Optional[float]
    tp1:              Optional[float]
    tp2:              Optional[float]
    risk_pct:         float
    adr_pct:          float
    confirmations:    List[str]  = field(default_factory=list)
    missing:          List[str]  = field(default_factory=list)
    component_scores: dict       = field(default_factory=dict)
    veto_reason:      str        = ''


# ── Firebase publisher ─────────────────────────────────────────────────────── #

class ICTPublisher:
    def __init__(self):
        self._db    = None
        self._ref   = None
        self._init()

    def _init(self):
        try:
            import firebase_admin
            from firebase_admin import db as rtdb
            sa = os.environ.get('FIREBASE_SERVICE_ACCOUNT', 'config/firebase_service_account.json')
            if not firebase_admin._apps and Path(sa).exists():
                from firebase_admin import credentials
                firebase_admin.initialize_app(
                    credentials.Certificate(sa),
                    {'databaseURL': 'https://clawd-trading-7b8de-default-rtdb.firebaseio.com'}
                )
            self._db  = rtdb.reference('/')
            self._ref = self._db
            logger.info("Firebase: connected")
        except Exception as e:
            logger.warning("Firebase unavailable (%s) — offline mode", e)

    def push(self, result: ScanResult) -> bool:
        if self._db is None:
            return False
        try:
            path = f"{FIREBASE_ROOT}/{result.pair.replace('=X','')}"
            self._db.child(path).set(asdict(result))
            return True
        except Exception as e:
            logger.error("Firebase push failed for %s: %s", result.pair, e)
            return False

    def push_system(self, data: dict):
        if self._db is None:
            return
        try:
            self._db.child(f"{FIREBASE_ROOT}/_system").set(data)
        except Exception as e:
            logger.error("Firebase system push failed: %s", e)


# ── Orchestrator ───────────────────────────────────────────────────────────── #

class ICTOrchestrator:

    def __init__(self, pairs: Optional[List[str]] = None):
        self.pairs     = pairs or PROVEN_PAIRS
        self.publisher = ICTPublisher()

        from ict.pipeline import ICTPipeline
        from ict.micro_risk import MicroRiskParams
        from ict.session_classifier import SessionClassifier
        from ict.paper_trader import PaperTrader
        from ict.library_bridge import query_library
        from ict.daily_bias import DailyBiasEngine
        self._pipeline    = ICTPipeline()
        self._params      = MicroRiskParams
        self._sess_clf    = SessionClassifier()
        self._paper       = PaperTrader()
        self._query_lib   = query_library
        self._bias_engine = DailyBiasEngine()
        # Wire Firebase into paper trader after publisher is ready
        if self.publisher._db:
            self._paper.set_firebase(self.publisher._db)

    def scan_once(self, log_all: bool = True) -> List[ScanResult]:
        import yfinance as yf
        import pandas as pd
        from ict.pipeline import ICTSignal, ICTVeto, ICTGrade
        from ict._atr_utils import compute_atr

        now     = datetime.now(timezone.utc)
        session = self._sess_clf.classify(now)
        results = []
        actionable = []

        # ── Macro context (once per scan, not per pair) ───────────────────
        library_ctx = self._query_lib()
        pair_biases = self._bias_engine.get_biases(library_ctx)
        logger.info("Library: regime=%s threat=%s size=%.2f×",
                    library_ctx['regime'], library_ctx['threat'],
                    library_ctx['size_modifier'])
        for p, b in pair_biases.items():
            flag = ' [BLACKOUT]' if b['blackout'] else f" bias={b['bias']} conf={b['confidence']:.2f}"
            logger.info("  %s:%s", p, flag)

        for pair in self.pairs:
            clean = pair.replace('=X', '')
            try:
                df = yf.download(pair, period='5d', interval='1h',
                                 progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.rename(columns=str.capitalize)[['Open','High','Low','Close']].dropna()
                df.index = pd.to_datetime(df.index, utc=True)

                if len(df) < 30:
                    logger.warning("%s: insufficient data", clean)
                    continue

                atr     = compute_atr(df)
                account = self._params(account_size=ACCOUNT_SIZE)

                # Try both directions, take higher-scoring A-grade result
                best = None
                for direction in ('LONG', 'SHORT'):
                    try:
                        r = self._pipeline.evaluate(
                            symbol=clean, direction=direction,
                            df=df, timestamp=now, account=account, atr=atr,
                        )
                        if best is None or r.score > best.score:
                            best = r
                    except Exception as e:
                        logger.debug("%s %s: %s", clean, direction, e)

                if best is None:
                    continue

                sess_name = session.kill_zone_name or 'OFF-HOURS'
                is_primary = sess_name == PRIMARY_SESSION

                if isinstance(best, ICTSignal):
                    sz = best.sizing
                    pair_bias = pair_biases.get(clean, {})
                    bias_dir  = pair_bias.get('bias', 'NEUTRAL')
                    blackout  = pair_bias.get('blackout', False)

                    # Bias agreement: NEUTRAL = don't block, directional = must match
                    bias_agrees = (
                        bias_dir == 'NEUTRAL'
                        or bias_dir == best.direction
                    )

                    # A-grade + primary session + no blackout + bias agrees = actionable
                    is_actionable = (
                        best.passed
                        and best.grade == ICTGrade.A
                        and sess_name in ACTIVE_SESSIONS
                        and not blackout
                        and bias_agrees
                    )
                    if not bias_agrees:
                        logger.info("%s: bias=%s conflicts signal=%s — skipping",
                                    clean, bias_dir, best.direction)
                    r = ScanResult(
                        pair=clean, timestamp=now.isoformat(),
                        signal=best.direction, grade=best.grade.value,
                        score=round(best.score, 2), session=sess_name,
                        is_primary=is_primary, actionable=is_actionable,
                        entry_level=getattr(best, 'entry_level', None),
                        stop=getattr(sz, 'stop_loss', None),
                        tp1=getattr(sz, 'tp1', None),
                        tp2=getattr(sz, 'tp2', None),
                        risk_pct=getattr(sz, 'risk_pct', 0.0),
                        adr_pct=0.0,
                        confirmations=best.confirmations,
                        missing=best.missing,
                        component_scores=best.component_scores,
                    )
                    if is_actionable:
                        actionable.append(r)
                else:
                    r = ScanResult(
                        pair=clean, timestamp=now.isoformat(),
                        signal='VETO', grade=best.grade.value,
                        score=round(best.score, 2), session=sess_name,
                        is_primary=is_primary, actionable=False,
                        entry_level=None, stop=None, tp1=None, tp2=None,
                        risk_pct=0.0, adr_pct=0.0,
                        confirmations=best.confirmations,
                        missing=best.missing,
                        component_scores=best.component_scores,
                        veto_reason=best.reason,
                    )

                results.append(r)
                self.publisher.push(r)

                # ── Paper trading ─────────────────────────────────────────
                # Update open trades with current bar prices
                bar = df.iloc[-1]
                self._paper.update_trades({
                    clean: {
                        'high':  float(bar['High']),
                        'low':   float(bar['Low']),
                        'close': float(bar['Close']),
                    }
                })

                # Open new trade if signal is actionable and no position open
                if isinstance(r, ScanResult) and r.actionable:
                    from ict.pipeline import ICTGrade
                    opened = self._paper.open_trade(r)
                    if opened:
                        logger.info("📋 Paper trade opened: %s %s grade=%s score=%.1f",
                                    clean, r.signal, r.grade, r.score)

            except Exception as e:
                logger.error("%s scan failed: %s", clean, e)

        # Close all trades if NY PM session has ended (4 PM ET)
        et_hour = int(datetime.now().astimezone(
            __import__('zoneinfo').ZoneInfo('America/New_York')
        ).strftime('%H'))
        if et_hour >= SESSION_CLOSE_HOUR_ET:
            closed = self._paper.close_session('NY_PM_END')
            if closed:
                logger.info("Session ended — closed %d trade(s)", len(closed))

        # Push running paper stats + macro context to Firebase
        stats = self._paper.get_stats()
        self.publisher.push_system({
            'updated_at':    now.isoformat(),
            'session':       session.kill_zone_name or 'OFF-HOURS',
            'is_primary':    session.kill_zone_name == PRIMARY_SESSION,
            'pairs_scanned': len(results),
            'actionable':    len(actionable),
            'open_trades':   self._paper.n_open,
            'paper_stats':   stats,
            'library':       library_ctx,
            'pair_biases':   pair_biases,
            'version':       'v2',
        })
        # Write auto-trading config flag for dashboard
        if self.publisher._db:
            try:
                self.publisher._db.child('signals/ICT_ENGINE/config').set({
                    'auto_paper_trading_enabled': True,
                    'active_pairs': [p.replace('=X','') for p in self.pairs],
                    'primary_session': PRIMARY_SESSION,
                    'grade_filter': 'A',
                })
            except Exception:
                pass

        if log_all:
            _print_results(results, actionable, now, session)

        return results

    def watch(self, interval: int = 300):
        """Scan every `interval` seconds. Ctrl-C to stop."""
        logger.info("ICT watch mode — scanning every %ds", interval)
        logger.info("Active pairs: %s", ', '.join(p.replace('=X','') for p in self.pairs))
        logger.info("Primary session: %s (1:30–4:00 PM ET)", PRIMARY_SESSION)
        while True:
            try:
                self.scan_once()
            except KeyboardInterrupt:
                logger.info("Stopped.")
                break
            except Exception as e:
                logger.error("Scan error: %s", e)
            time.sleep(interval)


# ── Display ────────────────────────────────────────────────────────────────── #

def _print_results(results, actionable, now, session):
    sess_name = session.kill_zone_name or 'OFF-HOURS'
    print(f"\n{'='*58}")
    print(f"  ICT ENGINE  |  {now.strftime('%Y-%m-%d %H:%M')} UTC  |  {sess_name}")
    print(f"{'='*58}")

    for r in results:
        signal_str = f"{r.signal:<6}" if r.signal != 'VETO' else 'VETO  '
        grade_str  = f"[{r.grade}]"
        flag = ' ← TRADE' if r.actionable else ''
        print(f"  {r.pair:<8}  {signal_str}  {grade_str}  score={r.score:.1f}{flag}")
        if r.actionable:
            print(f"           entry={r.entry_level:.5f}  stop={r.stop:.5f}  "
                  f"TP1={r.tp1:.5f}  TP2={r.tp2:.5f}")

    if not actionable:
        if sess_name not in ACTIVE_SESSIONS:
            print(f"\n  Session {sess_name} not active — engine monitors but does not trade.")
            print(f"  Next primary session: NY PM (13:30–16:00 ET)")
        else:
            print(f"\n  No A-grade signals this scan.")
    else:
        print(f"\n  {len(actionable)} actionable signal(s) — A-grade in {PRIMARY_SESSION}")

    print(f"{'='*58}\n")


# ── CLI ────────────────────────────────────────────────────────────────────── #

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description='ICT Micro-Edge Scanner')
    parser.add_argument('--once',    action='store_true', help='Single scan and exit')
    parser.add_argument('--watch',   action='store_true', help='Scan every 5min continuously')
    parser.add_argument('--interval',type=int, default=300)
    parser.add_argument('--pairs',   nargs='+', default=None, help='Override pair list')
    args = parser.parse_args()

    orch = ICTOrchestrator(pairs=args.pairs)

    if args.watch:
        orch.watch(interval=args.interval)
    else:
        orch.scan_once()
