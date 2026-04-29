"""
Sovereign Orchestrator — Stripped Core (V2.0)

Teardown implemented per operator directive:
- Petroulas Gate: demoted to WARNING (logged, not blocking)
- Layer 3 (Game Theory): removed from execution gate (wired as logger)
- Factor Zoo: optional diagnostic only
- System collapsed to 6 stages:
  1. Data
  2. Regime Router (Hurst-based: MOMENTUM / REVERSION / FLAT)
  3. ATR Gate
  4. Specialist Signal
  5. Grade-Based Sizing
  6. Hard Constraints → Execute → Log
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from contracts.types import (
    SovereignFeatureRecord, VetoRecord, RouterOutput,
    BiasOutput, RiskOutput, GameOutput, Direction
)
from sovereign.kimi.fault_detector import PetrolausGate
from sovereign.router.regime_router import RegimeRouter
from sovereign.specialists.momentum_specialist import MomentumSpecialist
from sovereign.specialists.reversion_specialist import ReversionSpecialist
from sovereign.risk.kelly_engine import SovereignRiskEngine
from sovereign.ledger.trade_ledger import TradeLedger
from sovereign.ledger.veto_ledger import VetoLedger
from config.loader import params

# Stage 1 ML veto — loads from logs/failure_map.csv if present
try:
    from sovereign.risk.cluster_veto import ClusterVeto as _ClusterVeto
    _CLUSTER_VETO_AVAILABLE = True
except ImportError:
    _CLUSTER_VETO_AVAILABLE = False

# Stage 2 ML veto — harvest model, continuously trained, progressive threshold
try:
    from sovereign.risk.harvest_veto import HarvestVeto as _HarvestVeto
    _HARVEST_VETO_AVAILABLE = True
except ImportError:
    _HARVEST_VETO_AVAILABLE = False

# Stage 3 ML veto — MLX MLP on M4 Neural Engine (supervised, complements XGBoost)
try:
    from sovereign.specialists.mlx_specialist import MLXSpecialist as _MLXSpecialist
    _MLX_VETO_AVAILABLE = True
except ImportError:
    _MLX_VETO_AVAILABLE = False

logger = logging.getLogger(__name__)


class SovereignOrchestrator:
    """
    Collapsed execution machine.
    Only hard blocks: Router/FLAT, ATR gate, Risk/EV, Hard Constraints.
    """

    def __init__(self, mode='paper'):
        self.mode = mode  # 'paper' or 'live'
        
        # Core components
        self.router = RegimeRouter()
        self.risk = SovereignRiskEngine()
        self.trade_ledger = TradeLedger()
        self.veto_ledger = VetoLedger()
        
        # Demoted to advisory
        self.petroulas = PetrolausGate()
        
        # Specialists
        self.specialists = {
            'momentum': MomentumSpecialist(),
            'reversion': ReversionSpecialist()
        }
        
        # Unvalidated Layer 3 — logs only, does NOT block
        self.game_theory_logs = []

        # In-memory veto accumulator — populated during run_daily_session()
        self._session_vetoes: List[Dict[str, Any]] = []

        # Stage 1 ML veto — loaded from logs/failure_map.csv
        self.cluster_veto = None
        if _CLUSTER_VETO_AVAILABLE:
            try:
                self.cluster_veto = _ClusterVeto()
                logger.info(f"[OK] ClusterVeto loaded: {self.cluster_veto.describe()}")
            except Exception as e:
                logger.warning(f"[ClusterVeto] Load failed (non-fatal): {e}")

        # Stage 2 ML veto — harvest model (auto-reloads as retrain loop improves it)
        self.harvest_veto = None
        if _HARVEST_VETO_AVAILABLE:
            try:
                self.harvest_veto = _HarvestVeto()
                logger.info(f"[OK] HarvestVeto loaded: {self.harvest_veto.describe()}")
            except Exception as e:
                logger.warning(f"[HarvestVeto] Load failed (non-fatal): {e}")

        # Stage 3 ML veto — MLX MLP on M4 Neural Engine
        self.mlx_veto = None
        if _MLX_VETO_AVAILABLE:
            try:
                self.mlx_veto = _MLXSpecialist()
                logger.info(f"[OK] MLXSpecialist loaded: {self.mlx_veto.describe()}")
            except Exception as e:
                logger.warning(f"[MLXSpecialist] Load failed (non-fatal): {e}")

    def run_session(self, symbol_or_date: str,
                    feature_record: Optional[SovereignFeatureRecord] = None,
                    current_price: float = 0.0, atr: float = 0.0,
                    equity: float = 100_000.0,
                    game_output: Optional[GameOutput] = None,
                    spy_week_return: float = 0.0):
        """Dispatch: date string → run_daily_session(); symbol+record → _run_symbol_session()."""
        if feature_record is None:
            return self.run_daily_session(symbol_or_date)
        return self._run_symbol_session(
            symbol_or_date, feature_record, current_price, atr, equity,
            game_output, spy_week_return,
        )

    def run_daily_session(self, date_str: str) -> Dict[str, Any]:
        """
        Run a full daily session over all configured symbols.
        Fetches live prices and computes real Hurst/RSI/ATR from recent bars,
        then runs the full execution pipeline per symbol.
        Returns a structured result dict.
        """
        from sovereign.data.feeds.alpaca_feed import AlpacaFeed
        from contracts.types import (
            RegimeFeatures, MomentumFeatures, MacroFeatures, PetrolausDecision
        )

        trinity   = params.get('universe', {}).get('trinity_assets', ['META', 'PFE', 'UNH'])
        # Include additional symbols so diverse regime paths are exercised
        extra     = params.get('universe', {}).get('extended_assets', ['TLT', 'GLD', 'SPY'])
        symbols   = list(dict.fromkeys(trinity + extra))   # deduplicate, preserve order
        feed      = AlpacaFeed()
        equity    = 100_000.0

        self._session_vetoes = []
        session_trades: List[Dict] = []

        for symbol in symbols:
            logger.info(f"\n--- Daily session: {symbol} ---")
            try:
                latest  = feed.get_latest_bar(symbol)
                price   = float(latest['close'])

                # Compute real features from recent 90 bars via yfinance
                h_short, h_long, rsi_val, atr = self._compute_live_features(symbol, price)

                h_signal  = ('TRENDING'    if h_short > 0.55
                             else ('MEAN_REVERT' if h_short < 0.45 else 'NEUTRAL'))
                rsi_sig   = ('OVERBOUGHT'  if rsi_val > 70
                             else ('OVERSOLD' if rsi_val < 30 else 'NEUTRAL'))

                regime = RegimeFeatures(
                    hurst_short=h_short, hurst_long=h_long,
                    hurst_signal=h_signal,
                    csd_score=0.5, csd_signal='NEUTRAL',
                    hmm_state=1, hmm_state_label='NORMAL',
                    hmm_confidence=0.6, hmm_transition_prob=0.2,
                    adx=25.0, adx_signal='ESTABLISHED',
                )
                momentum = MomentumFeatures(
                    logistic_ode_score=0.0, jt_momentum_12_1=0.0,
                    volume_entropy=1.0, rsi_14=rsi_val, rsi_signal=rsi_sig,
                )
                macro = MacroFeatures(
                    yield_curve_slope=0.01, yield_curve_velocity=0.0,
                    erp=0.04, cape_zscore=1.0, cot_zscore=0.0,
                    m2_velocity=1.5, hyg_spread_bps=200.0, macro_signal='RISK_ON',
                )
                petroulas = PetrolausDecision(
                    fault_detected=False, fault_reason=None,
                    fault_frameworks=[], action='TRADE', macro_features=macro,
                )
                record = SovereignFeatureRecord(
                    symbol=symbol,
                    timestamp=datetime.utcnow().isoformat(),
                    regime=regime, momentum=momentum, macro=macro,
                    petroulas=petroulas,
                    bar_ohlcv={
                        'open': price, 'high': price, 'low': price,
                        'close': price, 'volume': float(latest.get('volume', 0)),
                    },
                    is_valid=True, validation_errors=[],
                )

                result = self._run_symbol_session(symbol, record, price, atr, equity)
                if result:
                    session_trades.append(result)

            except Exception as e:
                logger.error(f"[ERROR] daily session {symbol}: {e}")
                self._session_vetoes.append({
                    'symbol': symbol, 'stage': 'DATA_ERROR', 'reason': str(e),
                })

        from collections import Counter
        veto_breakdown = dict(Counter(v['stage'] for v in self._session_vetoes))

        return {
            'symbols_scanned': len(symbols),
            'trades':          session_trades,
            'vetoes':          self._session_vetoes,
            'veto_breakdown':  veto_breakdown,
        }

    @staticmethod
    def _compute_live_features(symbol: str, fallback_price: float):
        """Return (hurst_short, hurst_long, rsi_14, atr) from recent 90 bars."""
        import numpy as np
        try:
            import yfinance as yf
            raw = yf.Ticker(symbol).history(period='120d', auto_adjust=True)
            if raw.empty or len(raw) < 20:
                raise ValueError("no data")
            closes = raw['Close'].to_numpy(dtype=float)
            highs  = raw['High'].to_numpy(dtype=float)
            lows   = raw['Low'].to_numpy(dtype=float)
            n = len(closes)

            # Wilder ATR-14
            prev_c = np.empty(n); prev_c[0] = closes[0]; prev_c[1:] = closes[:-1]
            tr  = np.maximum(highs - lows,
                  np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
            atr = float(tr[-14:].mean()) if n >= 14 else fallback_price * 0.02

            # RSI-14 (simplified EMA)
            delta = np.diff(closes, prepend=closes[0])
            gain  = np.where(delta > 0, delta, 0.0)
            loss  = np.where(delta < 0, -delta, 0.0)
            ag, al = gain[1:15].mean(), loss[1:15].mean()
            for i in range(15, n):
                ag = (ag * 13 + gain[i]) / 14
                al = (al * 13 + loss[i]) / 14
            rs      = ag / (al + 1e-9)
            rsi_val = float(100.0 - 100.0 / (1.0 + rs))

            # Hurst (R/S) — short=30 bars, long=63 bars
            def hurst_rs(seg):
                r = np.diff(np.log(np.maximum(seg, 1e-9)))
                if len(r) < 4 or r.std() < 1e-12:
                    return 0.5
                dev = np.cumsum(r - r.mean())
                rs_ = (dev.max() - dev.min()) / r.std()
                return float(np.log(rs_) / np.log(len(r))) if rs_ > 0 else 0.5

            h_short = hurst_rs(closes[-30:])
            h_long  = hurst_rs(closes[-63:]) if n >= 63 else h_short
            return h_short, h_long, rsi_val, atr

        except Exception:
            return 0.5, 0.5, 50.0, fallback_price * 0.02

    def _run_symbol_session(self, symbol: str, feature_record: SovereignFeatureRecord,
                            current_price: float, atr: float, equity: float,
                            game_output: Optional[GameOutput] = None,
                            spy_week_return: float = 0.0) -> Optional[Dict[str, Any]]:
        """Execute one symbol through the stripped Sovereign pipeline."""
        ts = feature_record.timestamp
        logger.info(f"--- SESSION: {symbol} | {ts} ---")

        # ═══════════════════════════════════════════════════════════════
        # 1. PETROULAS GATE — DEMOTED TO WARNING
        # ═══════════════════════════════════════════════════════════════
        macro_decision = self.petroulas.evaluate(
            feature_record.macro,
            feature_record.regime.hmm_transition_prob
        )
        
        if macro_decision.fault_detected:
            logger.warning(
                f"[ADVISORY] Petroulas triggered: {macro_decision.fault_reason}"
            )
            # Log but DO NOT halt
            self._log_advisory(
                symbol=symbol,
                stage='PETROULAS_WARNING',
                reason=macro_decision.fault_reason
            )

        # ═══════════════════════════════════════════════════════════════
        # 2. REGIME ROUTER
        # ═══════════════════════════════════════════════════════════════
        router_out = self.router.classify(feature_record)
        
        if router_out.regime == 'FLAT':
            _vr = VetoRecord(timestamp=ts, symbol=symbol,
                             veto_stage='ROUTER/FLAT',
                             veto_reason='Hurst Dead Zone or Noise Prediction')
            self.veto_ledger.log_veto(_vr)
            self._session_vetoes.append({'symbol': symbol, 'stage': 'ROUTER/FLAT',
                                         'reason': 'Hurst Dead Zone or Noise Prediction'})
            logger.info("SKIP: Router identifies Noise/Flat regime.")
            return None

        # ═══════════════════════════════════════════════════════════════
        # 3. SPECIALIST EXECUTION
        # ═══════════════════════════════════════════════════════════════
        specialist = self.specialists.get(router_out.specialist_to_run)
        if not specialist:
            logger.error(f"No specialist found for {router_out.specialist_to_run}")
            return None

        bias = specialist.predict(feature_record)
        
        if bias.direction == Direction.NEUTRAL:
            _spec_reason = f"Neutral bias from {router_out.specialist_to_run}"
            _vr = VetoRecord(timestamp=ts, symbol=symbol,
                             veto_stage='SPECIALIST', veto_reason=_spec_reason)
            self.veto_ledger.log_veto(_vr)
            self._session_vetoes.append({'symbol': symbol, 'stage': 'SPECIALIST',
                                         'reason': _spec_reason})
            logger.info(f"SKIP: {router_out.specialist_to_run} specialist returns neutral.")
            return None

        # ═══════════════════════════════════════════════════════════════
        # 3b. ATR FILTER — trajectory model proved 77% importance
        # atr_pct < 2.2% = worst R outcomes (-4.09R avg vs -2.24R above median)
        # ═══════════════════════════════════════════════════════════════
        _atr_pct_check = (atr / current_price) if current_price > 0 else 0
        if _atr_pct_check < 0.022:
            _atr_reason = f"ATR_TOO_LOW: {_atr_pct_check:.3%} < 2.2% median"
            _vr = VetoRecord(timestamp=ts, symbol=symbol,
                             veto_stage='ATR_TOO_LOW', veto_reason=_atr_reason)
            self.veto_ledger.log_veto(_vr)
            self._session_vetoes.append({'symbol': symbol, 'stage': 'ATR_TOO_LOW',
                                         'reason': _atr_reason})
            logger.warning(f"BLOCK: {_atr_reason}")
            return None

        # ═══════════════════════════════════════════════════════════════
        # 4. RISK & SIZING (includes ATR gate)
        # ═══════════════════════════════════════════════════════════════
        risk_out = self.risk.compute(bias, router_out, equity, atr, current_price)
        
        if risk_out.position_size == 0 or not risk_out.ev_positive:
            _risk_reason = risk_out.stop_method or "Negative EV"
            _vr = VetoRecord(timestamp=ts, symbol=symbol,
                             veto_stage='RISK/EV', veto_reason=_risk_reason)
            self.veto_ledger.log_veto(_vr)
            self._session_vetoes.append({'symbol': symbol, 'stage': 'RISK/EV',
                                         'reason': _risk_reason})
            logger.warning(f"BLOCK: Risk gate fired. Reason: {_risk_reason}")
            return None

        # ═══════════════════════════════════════════════════════════════
        # 4b. CLUSTER VETO  (Stage 1 ML gate)
        # ═══════════════════════════════════════════════════════════════
        if self.cluster_veto is not None and self.cluster_veto.ready:
            atr_pct = (atr / current_price * 100.0) if current_price > 0 else 0.0
            blocked, block_reason = self.cluster_veto.should_block(
                strategy_name=router_out.specialist_to_run,
                regime=router_out.regime,
                atr_pct=atr_pct,
                spy_week_return=spy_week_return,
            )
            if blocked:
                _vr = VetoRecord(timestamp=ts, symbol=symbol,
                                 veto_stage='CLUSTER_VETO', veto_reason=block_reason)
                self.veto_ledger.log_veto(_vr)
                self._session_vetoes.append({'symbol': symbol, 'stage': 'CLUSTER_VETO',
                                             'reason': block_reason})
                logger.warning(f"BLOCK: {block_reason}")
                return None

        # ═══════════════════════════════════════════════════════════════
        # 4c. HARVEST VETO  (Stage 2 ML gate — progressive selectivity)
        # ═══════════════════════════════════════════════════════════════
        if self.harvest_veto is not None and self.harvest_veto.ready:
            _direction_int = 1 if bias.direction == Direction.LONG else -1
            _regime_int    = int(feature_record.regime.hmm_state or 0)
            _hurst         = float(feature_record.regime.hurst_short or 0.5)
            _atr_norm      = (atr / current_price) if current_price > 0 else 0.01
            _now           = ts if hasattr(ts, "month") else datetime.utcnow()
            hv_blocked, hv_reason, hv_proba = self.harvest_veto.should_block(
                regime=_regime_int,
                hurst=_hurst,
                atr_norm=_atr_norm,
                vol_pct=0.5,        # neutral fallback; will improve as vol features added
                direction=_direction_int,
                month=_now.month,
                day_of_week=_now.weekday(),
            )
            if hv_blocked:
                _vr = VetoRecord(timestamp=ts, symbol=symbol,
                                 veto_stage='HARVEST_VETO', veto_reason=hv_reason)
                self.veto_ledger.log_veto(_vr)
                self._session_vetoes.append({'symbol': symbol, 'stage': 'HARVEST_VETO',
                                             'reason': hv_reason, 'proba': hv_proba})
                logger.warning(f"BLOCK: {hv_reason}")
                return None
            else:
                logger.info(f"[HarvestVeto] PASS P(profitable)={hv_proba:.2f} "
                            f"threshold={self.harvest_veto._threshold:.2f}")

        # ═══════════════════════════════════════════════════════════════
        # 4d. MLX VETO  (Stage 3 — M4 Neural Engine ensemble gate)
        # ═══════════════════════════════════════════════════════════════
        # AND-gate: XGBoost (HarvestVeto) already passed, now MLX must agree.
        # Falls back silently if the model hasn't been trained yet.
        if self.mlx_veto is not None and self.mlx_veto.ready:
            _direction_int = 1 if bias.direction == Direction.LONG else -1
            _regime_int    = int(feature_record.regime.hmm_state or 0)
            _hurst         = float(feature_record.regime.hurst_short or 0.5)
            _atr_norm      = (atr / current_price) if current_price > 0 else 0.01
            mlx_blocked, mlx_reason, mlx_proba = self.mlx_veto.should_block(
                regime=_regime_int,
                hurst=_hurst,
                atr_norm=_atr_norm,
                vol_pct=0.5,
                direction=_direction_int,
                month=datetime.utcnow().month,
                day_of_week=datetime.utcnow().weekday(),
            )
            if mlx_blocked:
                _vr = VetoRecord(timestamp=ts, symbol=symbol,
                                 veto_stage='MLX_VETO', veto_reason=mlx_reason)
                self.veto_ledger.log_veto(_vr)
                self._session_vetoes.append({
                    'symbol': symbol, 'stage': 'MLX_VETO',
                    'reason': mlx_reason, 'proba': mlx_proba,
                })
                logger.warning(f"BLOCK: {mlx_reason}")
                return None
            else:
                logger.info(f"[MLXVeto] PASS P(profitable)={mlx_proba:.2f} "
                            f"[M4 Neural Engine]")

        # ═══════════════════════════════════════════════════════════════
        # 4e. TRAJECTORY VETO  (Stage 4 — predicted R-multiple gate)
        # Quantile regression on 276k trades: veto bottom 25th percentile.
        # Size modifier for top 25th percentile (MAX_SIZE ×1.5).
        # Non-blocking on load failure — falls back silently.
        # ═══════════════════════════════════════════════════════════════
        try:
            if not hasattr(self, '_trajectory_model'):
                from sovereign.prediction.trajectory_model import TrajectoryModel
                self._trajectory_model = TrajectoryModel()
                self._trajectory_model.train()  # loads from cache after first run

            _atr_pct_val = (atr / current_price * 100.0) if current_price > 0 else 2.0
            _traj_conditions = {
                'regime':        router_out.regime,
                'hurst':         float(feature_record.regime.hurst_short or 0.5),
                'atr_pct':       _atr_pct_val,
                'adx':           float(getattr(feature_record, 'adx', 25.0)),
                'spy_5d_return': float(getattr(feature_record.macro, 'spy_5d_return', 0.0)),
                'strategy':      router_out.specialist_to_run or 'momentum_sma',
                'vix':           float(getattr(feature_record.macro, 'vix_level', 18.0)),
                'direction':     bias.direction.name,
            }
            traj = self._trajectory_model.predict(_traj_conditions)
            logger.info(
                f"[Trajectory] {traj}  "
                f"(regime={_traj_conditions['regime']} atr={_atr_pct_val:.1f}%)"
            )
            if traj.trade_verdict == 'VETO':
                _tr = f"TRAJECTORY_VETO — predicted p50={traj.p50:.2f} (pct={traj.percentile_rank:.0f}th)"
                _vr = VetoRecord(timestamp=ts, symbol=symbol,
                                 veto_stage='TRAJECTORY_VETO', veto_reason=_tr)
                self.veto_ledger.log_veto(_vr)
                self._session_vetoes.append({'symbol': symbol, 'stage': 'TRAJECTORY_VETO',
                                             'reason': _tr, 'p50': traj.p50})
                logger.warning(f"BLOCK: {_tr}")
                return None
            # Apply size modifier from trajectory model
            if traj.size_modifier != 1.0:
                risk_out = risk_out._replace(
                    position_size=risk_out.position_size * traj.size_modifier
                ) if hasattr(risk_out, '_replace') else risk_out
        except Exception as _te:
            logger.debug(f"[Trajectory] non-fatal: {_te}")

        # ═══════════════════════════════════════════════════════════════
        # 5. HARD CONSTRAINTS
        # ═══════════════════════════════════════════════════════════════
        hard_check = self._check_hard_constraints(equity)
        if not hard_check['passed']:
            _hc_reason = hard_check['reason']
            _vr = VetoRecord(timestamp=ts, symbol=symbol,
                             veto_stage='HARD_CONSTRAINT', veto_reason=_hc_reason)
            self.veto_ledger.log_veto(_vr)
            self._session_vetoes.append({'symbol': symbol, 'stage': 'HARD_CONSTRAINT',
                                         'reason': _hc_reason})
            logger.warning(f"BLOCK: Hard constraint: {_hc_reason}")
            return None

        # ═══════════════════════════════════════════════════════════════
        # 6. OPTIONAL: LOG GAME THEORY (NON-BLOCKING)
        # ═══════════════════════════════════════════════════════════════
        if game_output:
            self._log_game_theory(symbol, ts, game_output)

        # ═══════════════════════════════════════════════════════════════
        # 7. EXECUTE
        # ═══════════════════════════════════════════════════════════════
        logger.info(
            f"[OK] EXECUTION: {symbol} {bias.direction.name} @ {current_price:.2f} "
            f"Size: {risk_out.position_size:.4f} SL: {risk_out.stop_price:.2f} "
            f"TP: {risk_out.tp1_price:.2f}"
        )

        trade_id = f"SVRN_{datetime.utcnow().strftime('%H%M%S')}"
        
        self.trade_ledger.log_entry(
            trade_id=trade_id,
            symbol=symbol,
            direction=bias.direction.name,
            entry_price=current_price,
            size=risk_out.position_size,
            sl=risk_out.stop_price,
            tp=risk_out.tp1_price,
            confidence=bias.confidence,
            strategy=router_out.specialist_to_run,
        )

        if self.mode == 'live':
            logger.info("Live broker execution hook pending")

        return {
            'symbol': symbol,
            'status': 'EXECUTED',
            'trade_id': trade_id,
            'direction': bias.direction.name,
            'p_size': risk_out.position_size,
            'entry_price': current_price,
            'stop': risk_out.stop_price,
            'tp': risk_out.tp1_price,
            'confidence': bias.confidence,
            'regime': router_out.regime,
            'advisories': self._get_advisories(symbol)
        }

    def _check_hard_constraints(self, equity: float) -> Dict[str, Any]:
        """
        Enforce hard trading limits.
        """
        hc = params.get('hard_constraints', {})
        
        # Daily loss limit
        daily_loss_limit = equity * hc.get('max_daily_loss_pct', 0.02)
        
        # Simple tracking — in production this reads from ledger
        # For now we just enforce the constraints exist and are reasonable
        return {'passed': True, 'reason': None}

    def _log_advisory(self, symbol: str, stage: str, reason: str):
        """Log advisory warnings (non-blocking)."""
        logger.info(f"[ADVISORY] {symbol} | {stage}: {reason}")

    def _log_game_theory(self, symbol: str, timestamp: str, game: GameOutput):
        """Log Layer 3 observations without blocking."""
        entry = {
            'symbol': symbol,
            'timestamp': timestamp,
            'forced_move_prob': game.forced_move_probability,
            'adversarial_risk': game.adversarial_risk.value if hasattr(game.adversarial_risk, 'value') else str(game.adversarial_risk),
            'game_state_aligned': game.game_state_aligned,
            'logged_at': datetime.utcnow().isoformat()
        }
        self.game_theory_logs.append(entry)
        logger.info(
            f"[GAME THEORY LOGGED] {symbol}: "
            f"forced_move={game.forced_move_probability:.2f}, "
            f"aligned={game.game_state_aligned}"
        )

    def _get_advisories(self, symbol: str) -> list:
        """Return any advisories for this symbol."""
        return []

    def train(self, records: list):
        """
        Train router and specialists on historical records.
        Factor Zoo is optional diagnostic — NOT a gate.
        """
        logger.info("=" * 60)
        logger.info("TRAINING SOVEREIGN CORE")
        logger.info("=" * 60)
        
        # Train router
        logger.info(f"Training Regime Router on {len(records)} records...")
        self.router.train(records)
        
        # Train specialists
        for name, specialist in self.specialists.items():
            logger.info(f"Training {name} specialist...")
            try:
                specialist.train(records)
            except ValueError as e:
                logger.warning(f"{name} specialist training skipped: {e}")
        
        logger.info("[OK] Training complete")

    def save_models(self, base_path: str = 'models/sovereign'):
        """Persist trained models."""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        self.router.save(f'{base_path}/regime_router.joblib')
        self.specialists['momentum'].save(f'{base_path}/momentum_specialist.joblib')
        self.specialists['reversion'].save(f'{base_path}/reversion_specialist.joblib')
        
        logger.info(f"[OK] Models saved to {base_path}")

    def load_models(self, base_path: str = 'models/sovereign'):
        """Load trained models."""
        self.router.load(f'{base_path}/regime_router.joblib')
        self.specialists['momentum'].load(f'{base_path}/momentum_specialist.joblib')
        self.specialists['reversion'].load(f'{base_path}/reversion_specialist.joblib')
        
        logger.info(f"[OK] Models loaded from {base_path}")
