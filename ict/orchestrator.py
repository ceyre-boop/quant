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

# Macro-filtered universe (forex v004 results — all pairs macro-gated via library):
#   GBPUSD: Sharpe 1.094, WR 59.4%, PF 1.76 — BOE/FED rate differential
#   EURUSD: Sharpe 0.982, WR 54.5%, PF 1.62 — ECB/FED rate differential
#   AUDUSD: Sharpe 0.896, WR 52.5%, PF 1.56 — RBA/FED rate differential
# ICT entry precision + macro direction filter = compounding edge
# AUDNZD removed 2026-07-01: 0 signals, 100% displacement-gate fail
PROVEN_PAIRS = ['GBPUSD=X', 'EURUSD=X', 'AUDUSD=X']

# Forensics 2026-05-18: NY_PM averages -0.283R across all pairs.
# London averages +0.471R. London-only MC pass rate: 90.3% vs 58.4%.
# NY_PM vetoed in pipeline.py stage 5.7. Orchestrator still schedules it
# (veto fires inside pipeline before signal is issued — no code change needed here).
# PRIMARY_SESSION kept as NY_PM for scheduling continuity; pipeline does the filtering.
ACTIVE_SESSIONS   = {'NY_PM', 'London'}
PRIMARY_SESSION   = 'London'    # London is now the primary edge session
LONDON_PAIRS      = {'GBPUSD', 'EURUSD', 'AUDUSD'}  # AUDNZD removed 2026-07-01: 0 signals, 100% displacement-gate fail

FIREBASE_ROOT   = 'signals/ICT_ENGINE'
ACCOUNT_SIZE    = 10_000.0


# ── Veto sub-reasons ─────────────────────────────────────────────────────────── #

def build_gate_veto_reason(
    *,
    mem_veto: bool,
    mem_similarity: float,
    mem_historical_wr: float,
    floor_ok: bool,
    decision_score: float,
    score_floor: Optional[float],
    heatmap_conflict: bool,
    heatmap_detail: Optional[str],
    bias_agrees: bool,
    bias_dir: str,
    signal_direction: str,
    session_ok: bool,
    session_name: str,
    pair: str,
    blackout: bool,
    grade: str,
    score: float,
    time_utc: str,
) -> tuple[str, str]:
    """Map a blocked A-grade signal to ``(veto_stage, veto_reason)``.

    The stage is the coarse category the ledger buckets on; the reason carries
    the SPECIFIC triggering values (similarity, floor, magnet, bias direction,
    time…) so the veto ledger records WHY the gate fired, not just its name.
    Order mirrors the ``is_actionable`` short-circuit in ``scan()``.
    """
    if mem_veto:
        return "memory", (
            f"MEMORY_VETO: sim={mem_similarity:.2f} "
            f"historical_wr={mem_historical_wr:.2f} (losing/low-similarity cluster)"
        )
    if not floor_ok:
        _floor = f"{score_floor:.2f}" if score_floor is not None else "n/a"
        return "memory", (
            f"MEMORY_FLOOR: score {decision_score:.2f} < cluster floor {_floor}"
        )
    if heatmap_conflict:
        return "heatmap", (
            f"HEATMAP_CONFLICT: {heatmap_detail}" if heatmap_detail
            else "HEATMAP_CONFLICT: opposing magnet closer than TP1"
        )
    if not bias_agrees:
        return "bias", (
            f"BIAS_CONFLICT: library bias={bias_dir} vs signal={signal_direction}"
        )
    if not session_ok:
        return "session", (
            f"SESSION_BLOCK: {session_name} not tradeable for {pair} @ {time_utc}"
        )
    if blackout:
        return "gate", f"BLACKOUT: {pair} in high-impact event blackout"
    return "gate", (
        f"GATE: grade={grade} score={score:.2f} failed post-pipeline actionability check"
    )


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
        sa = os.environ.get('FIREBASE_SERVICE_ACCOUNT', 'config/firebase_service_account.json')

        # Fail loud when the service account is missing. The live scanner path
        # publishes to signals/ICT_ENGINE/* — without credentials every push
        # silently no-ops and the dashboard goes stale unnoticed. A missing SA
        # is a misconfiguration, not a normal state, so raise unless offline mode
        # is explicitly requested via ICT_FIREBASE_OFFLINE (local/dev runs).
        if not Path(sa).exists():
            offline_ok = os.getenv('ICT_FIREBASE_OFFLINE', '').lower() in ('1', 'true', 'yes')
            msg = (
                f"Firebase service account not found at '{sa}'. The ICT scanner "
                f"publishes live signals to {FIREBASE_ROOT}/* and cannot write "
                f"without it. Provide the key (config/firebase_service_account.json "
                f"or FIREBASE_SERVICE_ACCOUNT=/path/to/key.json), or set "
                f"ICT_FIREBASE_OFFLINE=1 to run intentionally offline."
            )
            if not offline_ok:
                logger.critical("FIREBASE SERVICE ACCOUNT MISSING — %s", msg)
                raise RuntimeError(msg)
            logger.warning("Firebase offline (ICT_FIREBASE_OFFLINE set) — %s", msg)
            return

        # Service account present — behaviour unchanged from before.
        try:
            import firebase_admin
            from firebase_admin import db as rtdb
            if not firebase_admin._apps:
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

    def __init__(self, pairs: Optional[List[str]] = None, ny_am_mode: bool = False):
        self.pairs      = pairs or PROVEN_PAIRS
        self.ny_am_mode = ny_am_mode
        self.publisher  = ICTPublisher()

        from ict.pipeline import ICTPipeline
        from ict.micro_risk import MicroRiskParams
        from ict.session_classifier import SessionClassifier
        from ict.paper_trader import PaperTrader
        from ict.library_bridge import query_library
        from ict.daily_bias import DailyBiasEngine
        from ict.memory_engine import ICTMemoryEngine
        from ict.regime_execution import get_regime_targets
        from ict.ict_veto_ledger import ICTVetoLedger
        from execution.funderpro_executor import FunderProExecutor
        self._pipeline       = ICTPipeline()
        self._params         = MicroRiskParams
        self._sess_clf       = SessionClassifier()
        self._paper          = PaperTrader()
        self._query_lib      = query_library
        self._bias_engine    = DailyBiasEngine()
        self._memory         = ICTMemoryEngine()
        self._get_targets    = get_regime_targets
        self._veto_ledger    = ICTVetoLedger()
        self._executor       = FunderProExecutor(account_size=ACCOUNT_SIZE)
        # Wire Firebase into paper trader after publisher is ready
        if self.publisher._db:
            self._paper.set_firebase(self.publisher._db)

    def scan_once(self, log_all: bool = True) -> List[ScanResult]:
        import yfinance as yf
        import pandas as pd
        from ict.pipeline import ICTSignal, ICTVeto, ICTGrade
        from ict._atr_utils import compute_atr

        # ICT_SKIP_TIMING_GATE=1: allow out-of-hours paper trade runs by bypassing
        # the hard ET-03:xx timing gate in pipeline Stage 0c. Retries any TIMING_GATE
        # veto with a synthetic 03:15 ET timestamp (data analysis is timestamp-agnostic).
        if os.getenv('ICT_SKIP_TIMING_GATE', '') in ('1', 'true', 'yes'):
            _orig_eval = self._pipeline.__class__.evaluate

            def _timing_skipped(self_p, symbol, direction, df, timestamp, account, atr=None):
                result = _orig_eval(self_p, symbol, direction, df, timestamp, account, atr)
                if isinstance(result, ICTVeto) and 'TIMING_GATE' in result.reason:
                    try:
                        import zoneinfo as _zi
                        from datetime import timezone as _tz
                        ts_3am = (
                            timestamp.astimezone(_zi.ZoneInfo("America/New_York"))
                            .replace(hour=3, minute=15, second=0, microsecond=0)
                            .astimezone(_tz.utc)
                        )
                        return _orig_eval(self_p, symbol, direction, df, ts_3am, account, atr)
                    except Exception:
                        pass
                return result

            self._pipeline.__class__.evaluate = _timing_skipped

        now     = datetime.now(timezone.utc)
        session = self._sess_clf.classify(now)
        results = []
        actionable = []

        # ── Macro context (once per scan, not per pair) ───────────────────
        library_ctx    = self._query_lib()
        pair_biases    = self._bias_engine.get_biases(library_ctx)
        regime_targets = self._get_targets(
            library_ctx['regime'], library_ctx['threat']
        )
        logger.info("Library: regime=%s threat=%s size=%.2f× | TP %.1fR/%.1fR [%s]",
                    library_ctx['regime'], library_ctx['threat'],
                    library_ctx['size_modifier'],
                    regime_targets['tp1_r'], regime_targets['tp2_r'],
                    regime_targets['mode'])
        for p, b in pair_biases.items():
            flag = ' [BLACKOUT]' if b['blackout'] else f" bias={b['bias']} conf={b['confidence']:.2f}"
            logger.info("  %s:%s", p, flag)

        # ── Weekly trend data (once per scan, fed to pipeline Stage 5.6) ──────────
        weekly_data: dict[str, Optional[pd.DataFrame]] = {}
        for _wp in self.pairs:
            try:
                _wdf = yf.download(_wp, period="2y", interval="1wk",
                                   progress=False, auto_adjust=True)
                if not _wdf.empty:
                    if isinstance(_wdf.columns, pd.MultiIndex):
                        _wdf.columns = _wdf.columns.get_level_values(0)
                    _wdf = _wdf.rename(columns=str.capitalize)
                    _wdf.index = pd.to_datetime(_wdf.index, utc=True)
                    weekly_data[_wp] = _wdf
                else:
                    weekly_data[_wp] = None
            except Exception as _exc:
                logger.debug("Weekly data for %s failed: %s", _wp, _exc)
                weekly_data[_wp] = None

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
                            ny_am_mode=self.ny_am_mode,
                            weekly_df=weekly_data.get(pair),
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

                    # Lever 4: Memory skip — low similarity or losing cluster
                    mem_check   = self._memory.match(
                        type('R', (), {'pair': clean, 'signal': best.direction,
                                       'grade': best.grade.value if hasattr(best.grade,'value') else str(best.grade),
                                       'score': best.score, 'session': sess_name,
                                       'adr_pct': 0.0, 'risk_pct': 0.0,
                                       'confirmations': best.confirmations,
                                       'missing': best.missing})()
                    )
                    mem_assessment = self._memory.assess_match(mem_check)
                    mem_veto = mem_assessment["hard_veto"]
                    soft_penalty_value = mem_assessment["penalty"]
                    # Start with original score, reduce only for soft-veto clusters.
                    decision_score = best.score
                    if mem_veto:
                        logger.info("%s: memory veto sim=%.2f wr=%.2f",
                                    clean, mem_check.similarity, mem_check.historical_wr)
                    elif mem_assessment["soft_veto"]:
                        decision_score = max(0.0, best.score - soft_penalty_value)
                        logger.info(
                            "%s: memory soft veto wr=%.2f penalty=%.2f score %.2f→%.2f",
                            clean, mem_check.historical_wr, soft_penalty_value,
                            best.score, decision_score,
                        )

                    # Lever 5: Heatmap conflict — opposing magnet closer than TP1
                    heatmap_conflict = False
                    heatmap_detail: Optional[str] = None
                    try:
                        from ict.liquidity_heatmap import compute_heatmap
                        hm = compute_heatmap(clean, df, atr)
                        top = hm.get('top_magnet')
                        if top and sz:
                            entry_p = getattr(best, 'entry_level', None)
                            tp1_p   = getattr(sz, 'tp1', None)
                            if entry_p and tp1_p and top['prob'] > 0.75:
                                magnet_dist = abs(top['price'] - entry_p)
                                tp1_dist    = abs(tp1_p - entry_p)
                                # Magnet between entry and TP1 = conflict
                                opposing_side = (
                                    (best.direction == 'LONG'  and top['price'] < entry_p) or
                                    (best.direction == 'SHORT' and top['price'] > entry_p)
                                )
                                if not opposing_side and magnet_dist < tp1_dist:
                                    heatmap_conflict = True
                                    heatmap_detail = (
                                        f"magnet {top['price']:.5f} prob={top['prob']:.2f} "
                                        f"closer than TP1 (dist {magnet_dist:.5f} < {tp1_dist:.5f})"
                                    )
                                    logger.info("%s: heatmap conflict — magnet %.5f prob=%.2f blocks TP1",
                                                clean, top['price'], top['prob'])
                    except Exception:
                        pass

                    # London session: only GBPUSD trades (has BOE/FED structure)
                    # NY AM mode: session gate handled by launcher script (UTC check)
                    session_ok = (
                        self.ny_am_mode
                        or sess_name == PRIMARY_SESSION
                        or (sess_name == 'London' and clean in LONDON_PAIRS)
                    )

                    # A-grade (+ Grade B in NY AM) + session + no blackout + bias + memory + heatmap
                    _accepted_grades = (ICTGrade.A_PLUS, ICTGrade.A, ICTGrade.B) if self.ny_am_mode else (ICTGrade.A_PLUS, ICTGrade.A)
                    # score_floor is None when the memory cluster is inactive or
                    # healthy (no real veto) — skip the floor check in that case.
                    _score_floor = mem_assessment["score_floor"]
                    _floor_ok = _score_floor is None or decision_score >= _score_floor
                    is_actionable = (
                        best.grade in _accepted_grades
                        and session_ok
                        and not blackout
                        and bias_agrees
                        and not mem_veto
                        and _floor_ok
                        and not heatmap_conflict
                    )
                    if not bias_agrees:
                        logger.info("%s: bias=%s conflicts signal=%s — skipping",
                                    clean, bias_dir, best.direction)
                    r = ScanResult(
                        pair=clean, timestamp=now.isoformat(),
                        signal=best.direction, grade=best.grade.value,
                        score=round(decision_score, 2), session=sess_name,
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
                        # A-grade signal that didn't clear post-pipeline gates —
                        # record to veto ledger for retroactive labeling. The
                        # reason carries the specific triggering values (WHY the
                        # gate fired), not just the stage category.
                        veto_stage, veto_reason = build_gate_veto_reason(
                            mem_veto=mem_veto,
                            mem_similarity=getattr(mem_check, "similarity", 0.0),
                            mem_historical_wr=getattr(mem_check, "historical_wr", 0.0),
                            floor_ok=_floor_ok,
                            decision_score=decision_score,
                            score_floor=_score_floor,
                            heatmap_conflict=heatmap_conflict,
                            heatmap_detail=heatmap_detail,
                            bias_agrees=bias_agrees,
                            bias_dir=bias_dir,
                            signal_direction=best.direction,
                            session_ok=session_ok,
                            session_name=sess_name,
                            pair=clean,
                            blackout=blackout,
                            grade=r.grade,
                            score=r.score,
                            time_utc=f"{now:%H:%M} UTC",
                        )
                        self._veto_ledger.record_veto(
                            pair=clean,
                            session=sess_name,
                            signal=best.direction,
                            grade=r.grade,
                            score=r.score,
                            veto_reason=veto_reason,
                            veto_stage=veto_stage,
                            entry_level=r.entry_level,
                            stop=r.stop,
                            tp1=r.tp1,
                            tp2=r.tp2,
                            intended_direction=best.direction,
                            intended_entry=r.entry_level,
                            structural_stop=r.stop,
                            adr_pct=r.adr_pct,
                            risk_pct=r.risk_pct,
                            confirmations=list(r.confirmations),
                            missing=list(r.missing),
                            component_scores=dict(r.component_scores),
                            timestamp=r.timestamp,
                        )
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
                    # Pipeline-level veto (grade B/C or hard gate failure)
                    self._veto_ledger.record_veto(
                        pair=clean,
                        session=sess_name,
                        signal=r.signal,
                        grade=r.grade,
                        score=r.score,
                        veto_reason=best.reason,
                        veto_stage="grade",
                        entry_level=getattr(best, 'entry_level', None),
                        stop=getattr(best, 'stop', None),
                        tp1=None,
                        tp2=None,
                        # `signal` reads "VETO" here; best.direction holds the
                        # pipeline's evaluated LONG/SHORT — recover it so the
                        # record is directionally labelable. entry/stop are
                        # present only for gates that fire after Stage 3.
                        intended_direction=getattr(best, 'direction', None),
                        intended_entry=getattr(best, 'entry_level', None),
                        structural_stop=getattr(best, 'stop', None),
                        adr_pct=getattr(best, 'adr_pct', 0.0),
                        risk_pct=0.0,
                        confirmations=list(best.confirmations),
                        missing=list(best.missing),
                        component_scores=dict(best.component_scores),
                        timestamp=now.isoformat(),
                    )

                # ── Memory: record scan, compute match ────────────────────
                self._memory.record_scan(r)
                memory_match = self._memory.match(r)

                # Push heatmap to Firebase
                try:
                    from ict.liquidity_heatmap import compute_heatmap
                    heatmap = compute_heatmap(clean, df, atr)
                    if self.publisher._db and heatmap.get('available'):
                        self.publisher._db.child(
                            f'signals/ICT_ENGINE/heatmap/{clean}'
                        ).set(heatmap)
                except Exception as he:
                    logger.debug("Heatmap failed for %s: %s", clean, he)

                results.append(r)
                self.publisher.push(r)

                # Push memory match to Firebase
                if self.publisher._db:
                    try:
                        path = f"signals/ICT_ENGINE/{clean}/memory"
                        self.publisher._db.child(path).set(memory_match.to_dict())
                    except Exception:
                        pass

                # ── Paper trading ─────────────────────────────────────────
                bar = df.iloc[-1]
                closed_updates = self._paper.update_trades({
                    clean: {
                        'high':  float(bar['High']),
                        'low':   float(bar['Low']),
                        'close': float(bar['Close']),
                    }
                })
                for t in closed_updates:
                    self._memory.record_outcome(
                        trade_id=t.id,
                        pair=t.pair,
                        outcome=t.outcome,
                        pnl_r=t.pnl_r,
                    )

                # Open new trade if actionable — apply regime TP ratios
                if isinstance(r, ScanResult) and r.actionable:
                    if regime_targets['skip']:
                        logger.info("%s: regime says SKIP (%s)", clean, regime_targets['reason'])
                        # G1_LIBRARY gate: the macro library regime vetoed an
                        # otherwise-actionable signal. Record it with the regime
                        # value so the ledger says WHY (not just that it skipped).
                        self._veto_ledger.record_veto(
                            pair=clean,
                            session=sess_name,
                            signal=r.signal,
                            grade=r.grade,
                            score=r.score,
                            veto_reason=(
                                f"G1_LIBRARY: regime={library_ctx['regime']} "
                                f"threat={library_ctx['threat']} — {regime_targets['reason']}"
                            ),
                            veto_stage="gate",
                            entry_level=r.entry_level,
                            stop=r.stop,
                            tp1=r.tp1,
                            tp2=r.tp2,
                            intended_direction=r.signal,
                            intended_entry=r.entry_level,
                            structural_stop=r.stop,
                            adr_pct=r.adr_pct,
                            risk_pct=r.risk_pct,
                            confirmations=list(r.confirmations),
                            missing=list(r.missing),
                            component_scores=dict(r.component_scores),
                            timestamp=r.timestamp,
                        )
                    else:
                        # Pass regime targets + risk multiplier into paper trader
                        opened = self._paper.open_trade(
                            r,
                            tp1_r=regime_targets['tp1_r'],
                            tp2_r=regime_targets['tp2_r'],
                            risk_mult=regime_targets.get('risk_mult', 1.0),
                        )
                        if opened:
                            logger.info(
                                "📋 Paper trade opened: %s %s grade=%s score=%.1f "
                                "TP %.1fR/%.1fR [%s]",
                                clean, r.signal, r.grade, r.score,
                                regime_targets['tp1_r'], regime_targets['tp2_r'],
                                regime_targets['mode'],
                            )
                        # Forward to prop executor (OFF by default)
                        self._executor.submit(
                            r,
                            tp1_r=regime_targets['tp1_r'],
                            tp2_r=regime_targets['tp2_r'],
                        )

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
            for t in closed:
                self._memory.record_outcome(
                    trade_id=t.id,
                    pair=t.pair,
                    outcome=t.outcome,
                    pnl_r=t.pnl_r,
                )

        # Push running paper stats + macro context to Firebase
        stats           = self._paper.get_stats()
        executor_status = self._executor.get_status().to_dict()
        self.publisher.push_system({
            'updated_at':      now.isoformat(),
            'session':         session.kill_zone_name or 'OFF-HOURS',
            'is_primary':      session.kill_zone_name == PRIMARY_SESSION,
            'pairs_scanned':   len(results),
            'actionable':      len(actionable),
            'open_trades':     self._paper.n_open,
            'paper_stats':     stats,
            'library':         library_ctx,
            'pair_biases':     pair_biases,
            'regime_targets':  regime_targets,
            'executor':        executor_status,
            'version':         'v3',
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

        self._write_scanner_state(results, session, library_ctx, pair_biases, now)
        return results

    def _write_scanner_state(self, results, session, library_ctx, pair_biases, now):
        """Write logs/scanner_state.json after every scan for the live dashboard."""
        from pathlib import Path as _P
        pairs_data = {}
        for r in results:
            gates = {
                'kill_zone':    any('Kill Zone' in c for c in r.confirmations),
                'sweep':        any('Sweep' in c for c in r.confirmations),
                'displacement': any('Displacement' in c or 'displacement' in c.lower()
                                    for c in r.confirmations),
                'fvg':          any('FVG' in c or 'fvg' in c.lower() for c in r.confirmations),
                'session':      r.session not in ('OFF-HOURS', ''),
            }
            blocking_gate = None
            blocking_reason = r.veto_reason or ''
            for m in r.missing:
                ml = m.lower()
                if 'fvg' in ml or 'ob' in ml:
                    blocking_gate = 'fvg'; blocking_reason = m; break
                if 'sweep' in ml:
                    blocking_gate = 'sweep'; blocking_reason = m; break
                if 'displacement' in ml:
                    blocking_gate = 'displacement'; blocking_reason = m; break
            bias_info = pair_biases.get(r.pair, {})
            blackout  = bias_info.get('blackout', False)
            status = ('BLACKOUT' if blackout
                      else 'WATCHING' if r.grade in ('A+', 'A', 'B')
                      else 'SCANNING')
            pairs_data[r.pair] = {
                'score': r.score, 'grade': r.grade, 'status': status,
                'signal': r.signal, 'session': r.session,
                'gates': gates, 'blocking_gate': blocking_gate,
                'blocking_reason': blocking_reason,
                'entry_if_fires': r.entry_level,
                'stop_if_fires':  r.stop,
                'tp1_if_fires':   r.tp1,
                'tp2_if_fires':   r.tp2,
                'confirmations':  list(r.confirmations),
                'missing':        list(r.missing),
                'component_scores': dict(r.component_scores),
                'veto_reason':    r.veto_reason,
            }
        import zoneinfo as _zi
        from datetime import timedelta as _td, timezone as _tz
        et_tz  = _zi.ZoneInfo('America/New_York')
        et_now = now.astimezone(et_tz)
        equity_scan = et_now.replace(hour=13, minute=35, second=0, microsecond=0)
        if et_now >= equity_scan:
            equity_scan += _td(days=1)
        state = {
            'last_scan':        now.isoformat(),
            'session_status':   session.kill_zone_name or 'OFF-HOURS',
            'library':          library_ctx,
            'pairs':            pairs_data,
            'next_equity_scan': equity_scan.astimezone(_tz.utc).isoformat(),
        }
        try:
            _P('logs/scanner_state.json').write_text(__import__('json').dumps(state, indent=2))
        except Exception:
            pass

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
    parser.add_argument('--ny-am',   action='store_true', help='NY AM mode: Grade B accepted, 0.50%% risk, bypasses NY_PM veto')
    args = parser.parse_args()

    if args.ny_am:
        ny_am_pairs = args.pairs or [
            # AUDNZD removed 2026-07-01: 0 signals, 100% displacement-gate fail
            'GBPUSD=X', 'EURUSD=X', 'AUDUSD=X', 'USDJPY=X'
        ]
        orch = ICTOrchestrator(pairs=ny_am_pairs, ny_am_mode=True)
    else:
        orch = ICTOrchestrator(pairs=args.pairs)

    if args.watch:
        orch.watch(interval=args.interval)
    else:
        orch.scan_once()
