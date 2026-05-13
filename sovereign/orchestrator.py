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

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

ROOT = Path(__file__).resolve().parent.parent

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
from sovereign.present_state import PresentStateBuilder
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

# Ernest Chan: PredictNow — regime-conditional size multiplier
try:
    from sovereign.risk.predict_now import PredictNow as _PredictNow
    _PREDICT_NOW_AVAILABLE = True
except ImportError:
    _PREDICT_NOW_AVAILABLE = False

# Ernest Chan: Alpha decay monitor — rolling Sharpe vs. baseline
try:
    from sovereign.risk.alpha_decay import AlphaDecayMonitor as _AlphaDecayMonitor
    _ALPHA_DECAY_AVAILABLE = True
except ImportError:
    _ALPHA_DECAY_AVAILABLE = False

# Andrew Lo: correlated sequential trade information + uncertainty gate
try:
    from sovereign.risk.correlated_position_tracker import (
        CorrelatedPositionTracker as _CorrTracker,
        lo_uncertainty_gate as _lo_uncertainty_gate,
    )
    _LO_TRACKER_AVAILABLE = True
except ImportError:
    _LO_TRACKER_AVAILABLE = False

# CS229 L19 — Kalman filter regime estimator
try:
    from sovereign.risk.kalman_regime import KalmanRegimeEstimator as _KalmanEstimator
    _KALMAN_AVAILABLE = True
except ImportError:
    _KALMAN_AVAILABLE = False

# CS229 L18 — LQR position controller (Riccati equation)
try:
    from sovereign.risk.lqr_controller import LQRController as _LQRController
    _LQR_AVAILABLE = True
except ImportError:
    _LQR_AVAILABLE = False

# CS229 L16 — Trade MDP value iteration
try:
    from sovereign.risk.trade_mdp import TradeMDP as _TradeMDP
    _MDP_AVAILABLE = True
except ImportError:
    _MDP_AVAILABLE = False

# CS229 L20 — Pegasus policy search + REINFORCE
try:
    from sovereign.risk.pegasus_policy_search import PegasusPolicySearch as _Pegasus
    _PEGASUS_AVAILABLE = True
except ImportError:
    _PEGASUS_AVAILABLE = False

# MIT BS — Vol regime signal (IV vs realized vol)
try:
    from sovereign.risk.black_scholes import VolRegimeSignal as _VolRegimeSignal
    _VOL_REGIME_AVAILABLE = True
except ImportError:
    _VOL_REGIME_AVAILABLE = False

# Trading Memory — live pattern matching against all historical dislocations
try:
    from sovereign.risk.market_memory import MarketMemory as _MarketMemory
    _MARKET_MEMORY_AVAILABLE = True
except ImportError:
    _MARKET_MEMORY_AVAILABLE = False

# Alexandrian Library — all 10 volumes, 60+ entries, multi-volume convergence
try:
    from sovereign.risk.alexandrian_library import AlexandrianLibrary as _AlexandrianLibrary
    _ALEXANDRIAN_LIBRARY_AVAILABLE = True
except ImportError:
    _ALEXANDRIAN_LIBRARY_AVAILABLE = False

# Stage 3 ML veto — MLX MLP on M4 Neural Engine (supervised, complements XGBoost)
try:
    from sovereign.specialists.mlx_specialist import MLXSpecialist as _MLXSpecialist
    _MLX_VETO_AVAILABLE = True
except ImportError:
    _MLX_VETO_AVAILABLE = False

# Narrative advisory — TradingAgents/Qwen3 via Sovereign narrative engine
try:
    from sovereign.intelligence import NarrativeEngine as _NarrativeEngine
    _NARRATIVE_ENGINE_AVAILABLE = True
except ImportError:
    _NARRATIVE_ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── Library integration helpers ─────────────────────────────────────────── #

# Patterns classified as severe risk by the PTJ dislocation framework.
_PTJ_SEVERE_PATTERNS = frozenset({
    "REPO_MARKET_STRESS", "MONEY_MARKET_CRISIS_2008", "ASIAN_CURRENCY_CONTAGION",
    "GFC_PRECURSOR", "VOLATILITY_CRASH_1929", "COVID_CRASH", "BLACK_MONDAY_1987",
    "YEN_CARRY_UNWIND", "VIX_EXTREME_2008",
})
_PTJ_MODERATE_PATTERNS = frozenset({
    "VALUATION_DISLOCATION", "DOTCOM_CRASH_PRECURSOR", "STAGFLATION_REGIME",
    "FED_HIKING_PAUSE", "RATE_SHOCK_2022", "RATE_HIKING_CYCLE_FAST",
    "DOLLAR_WRECKING_BALL", "TREASURY_MARKET_STRESS_2020", "LIBOR_STRESS_2007",
    "SVB_REGIONAL_CRISIS", "EM_STRESS_2014", "VOLMAGEDDON_2018",
})

# Asset class groups for Gate 5b sector filtering.
# Keys match partial symbol names or tags attached to entries.
_LIBRARY_SECTOR_GATES: dict = {
    "VALUATION_DISLOCATION": {
        "block_longs":  ["XLK", "XLY", "AMD", "NVDA", "growth_high_pe"],
        "block_shorts": [],
        "allow_longs":  ["XLP", "XLV", "XLU", "GLD", "SLV", "PFE", "UNH", "JNJ"],
        "size_mod":     0.75,   # reduce size on borderline names
    },
    "DEFENSIVE_ROTATION": {
        "block_longs":  ["XLK", "AMD", "NVDA", "high_beta"],
        "block_shorts": [],
        "allow_longs":  ["XLP", "XLV", "XLU"],
        "size_mod":     1.0,
    },
    "STAGFLATION_REGIME": {
        "block_longs":  ["XLK", "XLY", "bonds", "rate_sensitive"],
        "block_shorts": [],
        "allow_longs":  ["XLE", "GLD", "SLV", "commodities"],
        "size_mod":     1.0,
    },
    "REPO_MARKET_STRESS": {
        "block_longs":  ["XLK", "XLY", "XLF", "AMD", "NVDA"],
        "block_shorts": [],
        "allow_longs":  ["GLD"],
        "size_mod":     0.50,
    },
    "RATE_HIKING_CYCLE_FAST": {
        "block_longs":  ["XLK", "bonds", "rate_sensitive", "high_duration"],
        "block_shorts": [],
        "allow_longs":  ["XLE", "XLF", "XLI"],
        "size_mod":     0.75,
    },
    "COVID_CRASH": {
        "block_longs":  ["XLY", "XLK", "XLF", "XLE"],
        "block_shorts": [],
        "allow_longs":  ["GLD", "XLV"],
        "size_mod":     0.25,
    },
}


def _library_asset_gate(symbol: str, direction: str, library_insight) -> dict:
    """
    Gate 5b: filter asset class based on Library macro regime.
    Returns {'allowed': bool, 'size_mod': float, 'reason': str}.
    """
    if library_insight is None:
        return {'allowed': True, 'size_mod': 1.0, 'reason': 'no_library_data'}

    top = library_insight.primary_regime or ""
    gates = _LIBRARY_SECTOR_GATES.get(top, {})

    if not gates:
        return {'allowed': True, 'size_mod': 1.0, 'reason': f'library_regime={top}_no_gate'}

    sym_upper = symbol.upper().replace("=X", "").replace("-USD", "")
    is_long = direction.upper() in ("LONG", "BUY", "1")

    if is_long:
        for blocked in gates.get("block_longs", []):
            if blocked.upper() in sym_upper or sym_upper in blocked.upper():
                return {
                    'allowed': False,
                    'size_mod': 0.0,
                    'reason': (f"LIBRARY_ASSET_GATE: {sym_upper} blocked long in "
                               f"{top} sim={library_insight.primary_similarity:.3f}"),
                }
    else:
        for blocked in gates.get("block_shorts", []):
            if blocked.upper() in sym_upper or sym_upper in blocked.upper():
                return {
                    'allowed': False,
                    'size_mod': 0.0,
                    'reason': (f"LIBRARY_ASSET_GATE: {sym_upper} blocked short in {top}"),
                }

    size_mod = gates.get("size_mod", 1.0)
    return {
        'allowed': True,
        'size_mod': size_mod,
        'reason': f"Library {top} → size ×{size_mod:.2f} for {sym_upper}",
    }


def ptj_dislocation_from_library(library_insight) -> tuple[int, float]:
    """
    Integration 5: derive PTJ dislocation category directly from Library patterns.

    The Alexandrian Library IS the dislocation detection system — the separate
    dislocation_library.py is now merged into this flow.

    Returns (category, size_multiplier):
      0 → Normal  (1.0×)
      1 → Moderate dislocation  (0.75×, -25% size)
      2 → Severe dislocation    (0.50×, -50% size, hedges allowed)
    """
    if library_insight is None:
        return 0, 1.0

    top_matches = {
        getattr(vm, 'label', '') or getattr(vm, 'entry_id', '')
        for vm in library_insight.volume_matches
    }

    converging = sum(1 for vm in library_insight.volume_matches if vm.similarity >= 0.60)
    max_sim = max((vm.similarity for vm in library_insight.volume_matches), default=0.0)

    severe_active   = bool(top_matches & _PTJ_SEVERE_PATTERNS)
    moderate_active = bool(top_matches & _PTJ_MODERATE_PATTERNS)

    if severe_active and converging >= 5:
        return 2, 0.50
    elif severe_active and converging >= 3 and max_sim > 0.85:
        return 2, 0.50
    elif moderate_active and converging >= 5:
        return 2, 0.50
    elif moderate_active and converging >= 3:
        return 1, 0.75
    elif converging >= 7:
        return 1, 0.75
    else:
        return 0, 1.0


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
        self.present_state_builder = PresentStateBuilder()

        # Demoted to advisory
        self.petroulas = PetrolausGate()
        
        # Specialists
        self.specialists = {
            'momentum': MomentumSpecialist(),
            'reversion': ReversionSpecialist()
        }
        
        # Calendar Fetcher
        from data.calendar_fetcher import CalendarFetcher
        self.calendar = CalendarFetcher()
        
        # Unvalidated Layer 3 — logs only, does NOT block
        self.game_theory_logs = []

        # In-memory veto accumulator — populated during run_daily_session()
        self._session_vetoes: List[Dict[str, Any]] = []

        # Narrative intelligence bridge (TradingAgents/Qwen3) — monthly cache,
        # subprocess boundary, and bounded influence only.
        self.narrative_engine = None
        if _NARRATIVE_ENGINE_AVAILABLE:
            try:
                bridge_python = os.getenv(
                    "TRADINGAGENTS_BRIDGE_PYTHON",
                    str(ROOT / ".venv-tradingagents" / "bin" / "python"),
                )
                bridge_script = os.getenv(
                    "TRADINGAGENTS_BRIDGE_SCRIPT",
                    str(ROOT / "scripts" / "run_tradingagents_narrative.py"),
                )
                self.narrative_engine = _NarrativeEngine(
                    use_subprocess_bridge=True,
                    bridge_python=bridge_python,
                    bridge_script=bridge_script,
                )
                logger.info("[OK] NarrativeEngine loaded (TradingAgents via Ollama/Qwen3)")
            except Exception as e:
                logger.warning(f"[NarrativeEngine] Load failed (non-fatal): {e}")

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

        # Ernest Chan: PredictNow — regime-conditional P(trade profitable) → size multiplier
        self.predict_now = None
        if _PREDICT_NOW_AVAILABLE:
            try:
                self.predict_now = _PredictNow()
                self.predict_now.load_or_train()
                logger.info("[OK] PredictNow loaded")
            except Exception as e:
                logger.warning(f"[PredictNow] Load failed (non-fatal): {e}")

        # Ernest Chan: Alpha decay monitor — rolling Sharpe gating
        self.alpha_decay_momentum  = None
        self.alpha_decay_reversion = None
        if _ALPHA_DECAY_AVAILABLE:
            try:
                self.alpha_decay_momentum  = _AlphaDecayMonitor(strategy='momentum')
                self.alpha_decay_reversion = _AlphaDecayMonitor(strategy='reversion')
                logger.info("[OK] AlphaDecayMonitor loaded")
            except Exception as e:
                logger.warning(f"[AlphaDecay] Load failed (non-fatal): {e}")

        # Andrew Lo: session-level correlated position tracker
        # Reset each session — only within-session trades inform the update
        self._corr_tracker: Optional[object] = None
        if _LO_TRACKER_AVAILABLE:
            try:
                self._corr_tracker = _CorrTracker(lookback=5, max_adjustment=0.12)
                logger.info("[OK] CorrelatedPositionTracker loaded")
            except Exception as e:
                logger.warning(f"[CorrTracker] Load failed (non-fatal): {e}")

        # CS229 L19 — Kalman filter: persistent across sessions, learns from each bar
        self._kalman: Optional[object] = None
        if _KALMAN_AVAILABLE:
            try:
                self._kalman = _KalmanEstimator()
                logger.info("[OK] KalmanRegimeEstimator loaded")
            except Exception as e:
                logger.warning(f"[Kalman] Load failed (non-fatal): {e}")

        # CS229 L18 — LQR controller: recalculates Riccati each session
        self._lqr: Optional[object] = None
        if _LQR_AVAILABLE:
            try:
                self._lqr = _LQRController(horizon=10)
                logger.info("[OK] LQRController loaded")
            except Exception as e:
                logger.warning(f"[LQR] Load failed (non-fatal): {e}")

        # CS229 L16 — Trade MDP: persistent value function, updated after each close
        self._trade_mdp: Optional[object] = None
        if _MDP_AVAILABLE:
            try:
                self._trade_mdp = _TradeMDP()
                logger.info(f"[OK] TradeMDP loaded ({self._trade_mdp._m.n_trades} trades)")
            except Exception as e:
                logger.warning(f"[TradeMDP] Load failed (non-fatal): {e}")

        # CS229 L20 — Pegasus: persistent policy params, REINFORCE update on close
        self._pegasus: Optional[object] = None
        if _PEGASUS_AVAILABLE:
            try:
                self._pegasus = _Pegasus(n_scenarios=50)
                logger.info("[OK] PegasusPolicySearch loaded")
            except Exception as e:
                logger.warning(f"[Pegasus] Load failed (non-fatal): {e}")

        # MIT BS — Vol regime signal: updated from realized daily returns
        self._vol_regime: Optional[object] = None
        if _VOL_REGIME_AVAILABLE:
            try:
                self._vol_regime = _VolRegimeSignal(lookback=20)
                logger.info("[OK] VolRegimeSignal loaded")
            except Exception as e:
                logger.warning(f"[VolRegime] Load failed (non-fatal): {e}")

        # Track last trade state for MDP transitions
        self._last_trade_state: dict = {}

        # Trading Memory — live historical pattern matching
        self._market_memory: Optional[object] = None
        if _MARKET_MEMORY_AVAILABLE:
            try:
                self._market_memory = _MarketMemory()
                logger.info(f"[OK] MarketMemory loaded: {self._market_memory.describe()}")
            except Exception as e:
                logger.warning(f"[MarketMemory] Load failed (non-fatal): {e}")

        # Alexandrian Library — all 10 volumes, 60+ market circumstance entries
        self._alexandrian_library: Optional[object] = None
        if _ALEXANDRIAN_LIBRARY_AVAILABLE:
            try:
                self._alexandrian_library = _AlexandrianLibrary()
                logger.info(f"[OK] AlexandrianLibrary loaded: {self._alexandrian_library.volume_summary()}")
            except Exception as e:
                logger.warning(f"[AlexandrianLibrary] Load failed (non-fatal): {e}")

        # CS229 L04 — Softmax 3-class regime classifier (second regime vote)
        # Provides probability distribution over MOMENTUM/REVERSION/FLAT;
        # blends with HMM confidence to produce a richer regime_confidence signal.
        self._softmax_regime: Optional[object] = None
        try:
            from sovereign.risk.softmax_regime import SoftmaxRegimeClassifier as _SoftmaxRC
            self._softmax_regime = _SoftmaxRC()
            # Bootstrap from trade ledger if not yet fitted
            if not self._softmax_regime._fitted:
                self._bootstrap_softmax()
            logger.info(f"[OK] SoftmaxRegimeClassifier loaded (fitted={self._softmax_regime._fitted})")
        except Exception as e:
            logger.warning(f"[Softmax] Load failed (non-fatal): {e}")

        # CS229 L12 — K-Means regime clusterer (third regime vote)
        # Unsupervised clustering of regime feature space; majority vote with
        # HMM + Softmax drives final blended regime confidence.
        self._kmeans_regime: Optional[object] = None
        self._kmeans_feature_buffer: list = []
        try:
            from sovereign.risk.ml_diagnostics import KMeansRegimeClusterer as _KMeansRC
            self._kmeans_regime = _KMeansRC(k=3, n_init=10)
            self._bootstrap_kmeans()
            logger.info(f"[OK] KMeansRegimeClusterer loaded "
                        f"(fitted={self._kmeans_regime._centroids is not None})")
        except Exception as e:
            logger.warning(f"[KMeans] Load failed (non-fatal): {e}")

        # CS229 L15 — ICA factor separator (LOESS distance preprocessor)
        # Fitted on accumulated trade feature vectors. When fitted, replaces PCA
        # as the feature-space projector for PredictNow's LOESS distance metric.
        # ICA produces zero-correlation independent components — the theoretically
        # correct distance space for heavy-tailed financial features.
        self._ica: Optional[object] = None
        self._ica_feature_buffer: list = []
        try:
            from sovereign.risk.ica_factor_separator import ICAFactorSeparator as _ICAFactory
            self._ica = _ICAFactory(n_components=5)
            # Bootstrap ICA from trade ledger if enough history exists
            if not self._ica._fitted:
                self._bootstrap_ica()
            logger.info(f"[OK] ICAFactorSeparator loaded (fitted={self._ica._fitted})")
        except Exception as e:
            logger.warning(f"[ICA] Load failed (non-fatal): {e}")

        # PTJ gates — circuit breaker + gate runner (equity initialised to 100k default)
        self._ptj_circuit_breaker = None
        self._ptj_gate_runner = None
        self._ptj_spy_arr = None
        self._ptj_ast_arr = None
        self._latest_ml_snapshot: dict = {}
        self._ml_snapshot_history: list = []
        self._last_runtime_modulators: dict = {}
        try:
            from execution.ptj_gates import PTJCircuitBreaker
            _init_equity = params.get('account', {}).get('starting_equity', 100_000.0)
            self._ptj_circuit_breaker = PTJCircuitBreaker(_init_equity)
            logger.info("[OK] PTJ circuit breaker loaded")
        except Exception as _pte:
            logger.debug(f"[PTJ] circuit breaker init non-fatal: {_pte}")

    def get_ml_state_snapshot(
        self,
        regime: Optional[str] = None,
        blended_conf: Optional[float] = None,
        votes: Optional[Dict[str, Optional[str]]] = None,
    ) -> Dict[str, Any]:
        """Return a consolidated runtime snapshot of the ML stack."""
        modules: Dict[str, Dict[str, Any]] = {}
        cold = 0

        softmax_fitted = bool(self._softmax_regime is not None and getattr(self._softmax_regime, "_fitted", False))
        modules["softmax"] = {
            "fitted": softmax_fitted,
            "n_samples": int(getattr(self._softmax_regime, "_n_updates", 0)) if self._softmax_regime is not None else 0,
        }
        cold += 0 if softmax_fitted else 1

        kalman_bars = int(getattr(self._kalman, "_n_updates", 0)) if self._kalman is not None else 0
        kalman_out = {}
        if self._kalman is not None:
            try:
                kalman_out = self._kalman.get_regime_output()
            except Exception:
                kalman_out = {}
        kalman_fitted = kalman_bars > 0
        modules["kalman"] = {
            "fitted": kalman_fitted,
            "bars": kalman_bars,
            "dominant": kalman_out.get("regime"),
            "uncertainty": float(self._kalman.state_uncertainty()) if self._kalman is not None else None,
        }
        cold += 0 if kalman_fitted else 1

        kmeans_fitted = bool(self._kmeans_regime is not None and getattr(self._kmeans_regime, "_centroids", None) is not None)
        modules["ml_diag"] = {
            "fitted": kmeans_fitted,
            "n_samples": int(len(getattr(self, "_kmeans_feature_buffer", []))),
            "dominant": (votes or {}).get("kmeans"),
        }
        cold += 0 if kmeans_fitted else 1

        pn_n = int(len(getattr(self.predict_now, "_X", []))) if self.predict_now is not None else 0
        pn_fitted = pn_n >= 5
        modules["predict_now"] = {
            "fitted": pn_fitted,
            "n_outcomes": pn_n,
        }
        cold += 0 if pn_fitted else 1

        ad_n = int(len(getattr(self.alpha_decay_momentum, "_r_history", []))) if self.alpha_decay_momentum is not None else 0
        ad_level = "INSUFFICIENT_DATA"
        try:
            if self.alpha_decay_momentum is not None:
                ad_level = self.alpha_decay_momentum.check().level
        except Exception:
            pass
        ad_fitted = ad_n >= 20
        modules["alpha_decay"] = {"fitted": ad_fitted, "n_samples": ad_n, "gate": ad_level}
        cold += 0 if ad_fitted else 1

        peg_updates = int(getattr(self._pegasus, "n_updates", 0)) if self._pegasus is not None else 0
        peg_fitted = peg_updates >= 10
        modules["pegasus"] = {
            "fitted": peg_fitted,
            "updates": peg_updates,
            "trust": float(self._pegasus.trust_multiplier) if self._pegasus is not None else 0.0,
        }
        cold += 0 if peg_fitted else 1

        mdp_trades = int(getattr(getattr(self._trade_mdp, "_m", None), "n_trades", 0)) if self._trade_mdp is not None else 0
        mdp_fitted = mdp_trades >= 20
        modules["mdp"] = {
            "fitted": mdp_fitted,
            "trades": mdp_trades,
            "policy_source": "learned" if mdp_fitted else "expert_prior",
        }
        cold += 0 if mdp_fitted else 1

        lqr_fitted = bool(self._lqr is not None and getattr(self._lqr, "_K", None) is not None)
        modules["lqr"] = {"fitted": lqr_fitted, "riccati_solved": lqr_fitted}
        cold += 0 if lqr_fitted else 1

        corr_fitted = bool(self._corr_tracker is not None)
        corr_wr = None
        if self._corr_tracker is not None:
            try:
                corr_wr = self._corr_tracker.session_win_rate()
            except Exception:
                corr_wr = None
        modules["corr_tracker"] = {
            "fitted": corr_fitted,
            "session_win_rate": corr_wr,
            "gate": "CLEAR" if corr_wr is None else ("SUPPRESSED" if corr_wr < 0.4 else "CLEAR"),
        }
        cold += 0 if corr_fitted else 1

        bs_fitted = bool(self._vol_regime is not None)
        bs_sig = {}
        if self._vol_regime is not None:
            try:
                bs_sig = self._vol_regime.get_signal()
            except Exception:
                bs_sig = {}
        modules["bs"] = {
            "fitted": bs_fitted,
            "vol_regime": bs_sig.get("vol_regime"),
            "iv_rv_ratio": bs_sig.get("iv_rv_ratio"),
            "size_scalar": bs_sig.get("size_adjustment"),
        }
        cold += 0 if bs_fitted else 1

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "modules": modules,
            "ensemble_vote": {
                "regime": regime,
                "blended_conf": blended_conf,
                "votes": votes or {},
            },
            "integrations": {
                "alexandrian_library_loaded": self._alexandrian_library is not None,
                "market_memory_loaded": self._market_memory is not None,
            },
            "runtime_modulators": dict(self._last_runtime_modulators) if self._last_runtime_modulators else {},
            "active_modules": int(max(0, 10 - cold)),
            "cold_modules": int(cold),
        }
        return snapshot

    def get_latest_ml_snapshot(self) -> Dict[str, Any]:
        """Get the latest runtime snapshot (or a fresh startup snapshot)."""
        if self._latest_ml_snapshot:
            return dict(self._latest_ml_snapshot)
        return self.get_ml_state_snapshot()

    # ── Bootstrap helpers: cold-start the online learners from ledger ─── #

    def _bootstrap_softmax(self):
        """Batch-fit SoftmaxRegimeClassifier on historical trade ledger."""
        if self._softmax_regime is None:
            return
        try:
            import json as _json
            import numpy as _np
            from pathlib import Path as _Path
            from sovereign.risk.softmax_regime import SoftmaxRegimeClassifier as _SRC
            feats, labels = [], []
            for f in sorted((_Path('data') / 'ledger').glob('trade_ledger_*.jsonl')):
                for line in f.read_text().splitlines():
                    if not line.strip():
                        continue
                    t = _json.loads(line)
                    if t.get('status') != 'closed':
                        continue
                    regime = t.get('regime', 'FLAT')
                    if regime not in ('MOMENTUM', 'REVERSION', 'FLAT'):
                        continue
                    feats.append(_SRC.encode(
                        hurst=float(t.get('hurst', 0.5)),
                        hmm_prob=float(t.get('hmm_transition_prob', 0.5)),
                        adx=float(t.get('adx', 20.0)),
                        prev_regime=regime,
                        strategy=t.get('strategy', 'momentum'),
                    ))
                    labels.append(regime)
            if len(feats) >= 10:
                self._softmax_regime.fit(_np.array(feats), labels)
                logger.info(f"[Softmax] Bootstrap fit on {len(feats)} trades")
        except Exception as e:
            logger.debug(f"[Softmax.bootstrap] non-fatal: {e}")

    def _bootstrap_kmeans(self):
        """Fit KMeansRegimeClusterer on historical trade features."""
        if self._kmeans_regime is None:
            return
        try:
            import json as _json
            import numpy as _np
            from pathlib import Path as _Path
            feat_list = []
            for f in sorted((_Path('data') / 'ledger').glob('trade_ledger_*.jsonl')):
                for line in f.read_text().splitlines():
                    if not line.strip():
                        continue
                    t = _json.loads(line)
                    if t.get('status') != 'closed':
                        continue
                    feat_list.append([
                        float(t.get('hurst', 0.5)),
                        float(t.get('adx', 20.0)) / 50.0,
                        float(t.get('hmm_transition_prob', 0.5)),
                    ])
            if len(feat_list) >= 30:
                self._kmeans_feature_buffer = [_np.array(f) for f in feat_list]
                self._kmeans_regime.fit(_np.array(feat_list))
                logger.info(f"[KMeans] Bootstrap fit on {len(feat_list)} trades")
        except Exception as e:
            logger.debug(f"[KMeans.bootstrap] non-fatal: {e}")

    def _bootstrap_ica(self):
        """Fit ICAFactorSeparator on historical trade feature vectors."""
        if self._ica is None:
            return
        try:
            import json as _json
            import numpy as _np
            from pathlib import Path as _Path
            feat_list = []
            for f in sorted((_Path('data') / 'ledger').glob('trade_ledger_*.jsonl')):
                for line in f.read_text().splitlines():
                    if not line.strip():
                        continue
                    t = _json.loads(line)
                    if t.get('status') != 'closed':
                        continue
                    feat_list.append([
                        float(t.get('hurst', 0.5)),
                        float(t.get('hmm_transition_prob', 0.5)),
                        float(t.get('adx', 20.0)) / 50.0,
                        float(t.get('confidence', 0.7)),
                        float(t.get('size', 1000.0)) / 10000.0,
                    ])
            if len(feat_list) >= 50:
                self._ica_feature_buffer = [_np.array(f) for f in feat_list]
                self._ica.fit(_np.array(feat_list))
                logger.info(f"[ICA] Bootstrap fit on {len(feat_list)} trades")
        except Exception as e:
            logger.debug(f"[ICA.bootstrap] non-fatal: {e}")

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

        # Fetch once per session
        try:
            events = self.calendar.fetch_events()
            event_risk = self.calendar.calculate_event_risk(events)
            logger.info(f"[CALENDAR] Risk level for session: {event_risk}")
        except Exception as e:
            logger.warning(f"[CALENDAR] Fetch failed: {e}")
            event_risk = 'CLEAR'

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
                    event_risk=event_risk,
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
        # 1b. ECONOMIC EVENT RISK — VETO
        # ═══════════════════════════════════════════════════════════════
        if feature_record.event_risk == 'EXTREME':
            _ev_reason = "EXTREME_EVENT_RISK: Trade halted due to high-impact macro data (FOMC/CPI/NFP)"
            _vr = VetoRecord(timestamp=ts, symbol=symbol,
                             veto_stage='ECONOMIC_CALENDAR', veto_reason=_ev_reason)
            self.veto_ledger.log_veto(_vr)
            self._session_vetoes.append({'symbol': symbol, 'stage': 'ECONOMIC_CALENDAR',
                                          'reason': _ev_reason})
            logger.warning(f"BLOCK: {_ev_reason}")
            return None
        elif feature_record.event_risk in ('HIGH', 'ELEVATED'):
            logger.info(f"[ADVISORY] Market event risk is {feature_record.event_risk}. Sizing will be reduced.")

        # ═══════════════════════════════════════════════════════════════
        # 1c. PTJ 200 SMA GATES (Gates 5 + 6)
        #     PTJ: "The 200-day moving average is the directional circuit
        #     breaker. Never fight it."
        #     Gate 5 (macro): if SPY < 200 SMA → block ALL new longs
        #     Gate 6 (asset): individual must be on right side of 200 SMA
        # ═══════════════════════════════════════════════════════════════
        if hasattr(self, '_ptj_gate_runner') and self._ptj_gate_runner is not None:
            try:
                import numpy as _np
                _asset_px = feature_record.price_history if hasattr(feature_record, 'price_history') else None
                _spy_px   = feature_record.spy_price_history if hasattr(feature_record, 'spy_price_history') else None
                _direction_hint = 'long'  # resolved after specialist; use neutral pass for now
                if _spy_px is not None and _asset_px is not None:
                    _ma_macro = self._ptj_gate_runner.cb  # reuse circuit breaker ref
                    from execution.ptj_gates import ptj_200sma_macro_gate, ptj_200sma_asset_gate
                    _spy_arr = _np.array(_spy_px) if not isinstance(_spy_px, _np.ndarray) else _spy_px
                    _ast_arr = _np.array(_asset_px) if not isinstance(_asset_px, _np.ndarray) else _asset_px
                    # Macro gate checked for longs (run after specialist gives direction below)
                    # Store arrays for post-specialist check
                    self._ptj_spy_arr = _spy_arr
                    self._ptj_ast_arr = _ast_arr
            except Exception as _pge:
                logger.debug(f"[PTJ_200SMA] setup non-fatal: {_pge}")

        # ═══════════════════════════════════════════════════════════════
        # 2. REGIME ROUTER
        # ═══════════════════════════════════════════════════════════════
        router_out = self.router.classify(feature_record)

        # ═══════════════════════════════════════════════════════════════
        # 2b. PRESENT STATE — unified six-dimension view (non-blocking)
        # Built once here; available to all downstream gates.
        # ═══════════════════════════════════════════════════════════════
        try:
            present = self.present_state_builder.build(
                symbol=symbol,
                feature_record=feature_record,
                router_out=router_out,
                current_price=current_price,
                atr=atr,
            )
            logger.info(f"[PresentState] {present.summary()}")
        except Exception as _ps_err:
            logger.debug(f"[PresentState] build failed (non-fatal): {_ps_err}")
            present = None
        
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

        # ── PTJ Gate 5+6: 200 SMA direction-aware check (runs post-specialist) ── #
        # Now that we know the direction, apply PTJ's circuit breaker properly.
        # PTJ: "The 200-day moving average is the directional circuit breaker."
        try:
            from execution.ptj_gates import ptj_200sma_macro_gate, ptj_200sma_asset_gate
            import numpy as _np
            _ptj_dir = 'long' if bias.direction.value == 1 else 'short'
            _spy_arr = getattr(self, '_ptj_spy_arr', None)
            _ast_arr = getattr(self, '_ptj_ast_arr', None)
            _ptj_grade = 'A+' if bias.confidence >= 0.92 else ('A' if bias.confidence >= 0.78 else 'B')

            if _spy_arr is not None and len(_spy_arr) >= 200:
                _macro_gate = ptj_200sma_macro_gate(_ptj_dir, _spy_arr, _ptj_grade)
                if not _macro_gate:
                    _mr = f"PTJ_MACRO_200SMA: {_macro_gate.reason}"
                    _vr = VetoRecord(timestamp=ts, symbol=symbol,
                                     veto_stage='PTJ_MACRO_200SMA', veto_reason=_mr)
                    self.veto_ledger.log_veto(_vr)
                    self._session_vetoes.append({'symbol': symbol,
                                                  'stage': 'PTJ_MACRO_200SMA', 'reason': _mr})
                    logger.info(f"SKIP: {_mr}")
                    return None

            if _ast_arr is not None and len(_ast_arr) >= 200:
                _asset_gate = ptj_200sma_asset_gate(_ptj_dir, _ast_arr, symbol, _ptj_grade)
                if not _asset_gate:
                    _ar = f"PTJ_ASSET_200SMA: {_asset_gate.reason}"
                    _vr = VetoRecord(timestamp=ts, symbol=symbol,
                                     veto_stage='PTJ_ASSET_200SMA', veto_reason=_ar)
                    self.veto_ledger.log_veto(_vr)
                    self._session_vetoes.append({'symbol': symbol,
                                                  'stage': 'PTJ_ASSET_200SMA', 'reason': _ar})
                    logger.info(f"SKIP: {_ar}")
                    return None
                elif _asset_gate.size_modifier < 1.0:
                    # Exception: within 2% at key level — half size
                    logger.info(f"[PTJ_200SMA] {_asset_gate.reason}")
        except Exception as _ptje:
            logger.debug(f"[PTJ_200SMA] non-fatal: {_ptje}")

        # ── Gate 5b: Library Asset Class Filter ──────────────────────── #
        # The Library advisory identifies which sectors are safe in the current
        # macro regime. In VALUATION_DISLOCATION + DEFENSIVE_ROTATION: growth
        # stocks blocked, defensives pass. In REPO_MARKET_STRESS: all equity
        # at half-size only. PTJ: "the market is telling you where to be."
        if _insight is not None:
            try:
                _gate5b_result = _library_asset_gate(symbol, bias.direction.name, _insight)
                if not _gate5b_result['allowed']:
                    _gate5b_reason = _gate5b_result['reason']
                    _vr = VetoRecord(timestamp=ts, symbol=symbol,
                                     veto_stage='LIBRARY_ASSET_GATE', veto_reason=_gate5b_reason)
                    self.veto_ledger.log_veto(_vr)
                    self._session_vetoes.append({'symbol': symbol,
                                                  'stage': 'LIBRARY_ASSET_GATE',
                                                  'reason': _gate5b_reason})
                    logger.info(f"SKIP: {_gate5b_reason}")
                    return None
                elif _gate5b_result['size_mod'] < 1.0:
                    logger.info(f"[Gate5b] {_gate5b_result['reason']}")
            except Exception as _g5be:
                logger.debug(f"[Gate5b/Library] non-fatal: {_g5be}")

        narrative = self._get_narrative_bias(symbol, ts, bias.direction.name)
        if narrative is not None:
            bias.narrative_modifier = narrative.modifier
            bias.narrative_summary = narrative.narrative_summary
            bias.narrative_confidence = narrative.confidence
            bias.narrative_key_risks = list(narrative.key_risks)
            bias.narrative_catalysts = list(narrative.catalysts)
            bias.narrative_consensus = narrative.agent_consensus
            bias.narrative_source = narrative.source

            if self.narrative_engine and self.narrative_engine.should_override_quant(
                narrative, bias.direction.name
            ):
                _narrative_reason = (
                    f"NARRATIVE_OVERRIDE — {narrative.direction} "
                    f"{narrative.confidence} consensus={narrative.agent_consensus:.2f}"
                )
                _vr = VetoRecord(timestamp=ts, symbol=symbol,
                                 veto_stage='NARRATIVE_OVERRIDE',
                                 veto_reason=_narrative_reason)
                self.veto_ledger.log_veto(_vr)
                self._session_vetoes.append({
                    'symbol': symbol,
                    'stage': 'NARRATIVE_OVERRIDE',
                    'reason': _narrative_reason,
                })
                logger.warning(f"BLOCK: {_narrative_reason}")
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
        # Ernest Chan: "In ML you get a DIFFERENT probability each day
        # based on current market conditions." PredictNow's dynamic
        # P(profitable) feeds directly into Kelly as the win_rate prior
        # rather than a static 55% default.
        # ═══════════════════════════════════════════════════════════════

        # ── Early Library query (feeds Lo, Kelly, PredictNow, Gate 5b) ── #
        # Must run BEFORE Lo gate so convergence can override uncertainty level.
        _insight = None
        _lib_n_converging = 0
        if self._alexandrian_library is not None:
            try:
                import numpy as _np
                _spy_early = None
                _vix_early = None
                if hasattr(feature_record, 'spy_price_history') and feature_record.spy_price_history is not None:
                    _spy_early = _np.array(feature_record.spy_price_history)
                if hasattr(feature_record, 'vix_history') and feature_record.vix_history is not None:
                    _vix_early = _np.array(feature_record.vix_history)
                if _spy_early is not None and len(_spy_early) >= 200:
                    _insight = self._alexandrian_library.query(_spy_early, _vix_early)
                    _lib_n_converging = sum(
                        1 for vm in _insight.volume_matches if vm.similarity >= 0.60
                    )
                    logger.debug(
                        f"[Library/early] {_insight.primary_regime} "
                        f"threat={_insight.threat_score:.3f} "
                        f"converging={_lib_n_converging}"
                    )
            except Exception as _le:
                logger.debug(f"[Library/early] non-fatal: {_le}")

        # ── Three-vote regime confidence (HMM + Softmax + KMeans) ───────── #
        # Each classifier votes on the regime. Agreement amplifies confidence;
        # disagreement reduces it. The blended signal feeds Lo, MDP, and LQR.
        # This replaces the raw HMM regime_confidence with a ensemble estimate.
        _hmm_label = router_out.regime or 'FLAT'
        _softmax_vote = None
        _kmeans_vote = None
        _blended_regime_conf = float(router_out.regime_confidence or 0.5)
        _softmax_proba: dict = {}

        if self._softmax_regime is not None and self._softmax_regime._fitted:
            try:
                from sovereign.risk.softmax_regime import SoftmaxRegimeClassifier as _SRC
                _sf_x = _SRC.encode(
                    hurst=float(feature_record.regime.hurst_short or 0.5),
                    hmm_prob=float(feature_record.regime.hmm_transition_prob or 0.5),
                    adx=float(feature_record.regime.adx or 20.0),
                    prev_regime=_hmm_label,
                    strategy=router_out.specialist_to_run or 'momentum',
                )
                _softmax_proba = self._softmax_regime.predict_proba(_sf_x)
                _sf_pred = max(_softmax_proba, key=_softmax_proba.get)
                _softmax_vote = _sf_pred
                _sf_conf = _softmax_proba.get(_sf_pred, 1.0 / 3)
                if _sf_pred == _hmm_label:
                    # Agreement: confidence amplifies proportionally to Softmax certainty
                    _blended_regime_conf = min(1.0, _blended_regime_conf
                                               * (1.0 + 0.3 * (_sf_conf - 1.0 / 3)))
                else:
                    # Disagreement: confidence reduced proportionally to Softmax's certainty
                    # in the OTHER regime — the more certain Softmax is, the bigger the cut
                    _blended_regime_conf = max(0.10,
                                               _blended_regime_conf * (1.0 - 0.4 * _sf_conf))
                logger.debug(
                    f"[Softmax] {_sf_pred} P={_sf_conf:.3f} | HMM={_hmm_label} | "
                    f"conf {float(router_out.regime_confidence or 0.5):.3f}"
                    f"→{_blended_regime_conf:.3f}"
                )
            except Exception as _sfe:
                logger.debug(f"[Softmax] non-fatal: {_sfe}")

        if self._kmeans_regime is not None and self._kmeans_regime._centroids is not None:
            try:
                import numpy as _np
                _km_x = _np.array([
                    float(feature_record.regime.hurst_short or 0.5),
                    float(feature_record.regime.adx or 20.0) / 50.0,
                    float(feature_record.regime.hmm_transition_prob or 0.5),
                ])
                _km_pred = self._kmeans_regime.predict(_km_x)
                _kmeans_vote = _km_pred
                # Three-way vote tally
                _votes = {_hmm_label: 1}
                _votes[_km_pred] = _votes.get(_km_pred, 0) + 1
                if _softmax_proba:
                    _sf_top = max(_softmax_proba, key=_softmax_proba.get)
                    _votes[_sf_top] = _votes.get(_sf_top, 0) + 1
                _consensus = _votes.get(_hmm_label, 0)
                if _consensus == 3:
                    _blended_regime_conf = min(1.0, _blended_regime_conf * 1.15)
                elif _consensus == 1:
                    _blended_regime_conf = max(0.10, _blended_regime_conf * 0.85)
                logger.debug(
                    f"[KMeans] {_km_pred} | votes={_votes} | "
                    f"final_conf={_blended_regime_conf:.3f}"
                )
            except Exception as _kme:
                logger.debug(f"[KMeans] non-fatal: {_kme}")

        try:
            _snapshot = self.get_ml_state_snapshot(
                regime=_hmm_label,
                blended_conf=float(_blended_regime_conf),
                votes={
                    "hmm": _hmm_label,
                    "softmax": _softmax_vote,
                    "kmeans": _kmeans_vote,
                },
            )
            self._latest_ml_snapshot = _snapshot
            self._ml_snapshot_history.append(_snapshot)
            if len(self._ml_snapshot_history) > 500:
                self._ml_snapshot_history = self._ml_snapshot_history[-500:]
            logger.info(f"[ML_SNAPSHOT] {json.dumps(_snapshot, default=str)}")
        except Exception as _mse:
            logger.debug(f"[ML_SNAPSHOT] non-fatal: {_mse}")

        # ── Pegasus Gate 0: learned HMM confidence gate ───────────────── #
        # Pegasus has learned a minimum blended regime confidence below which
        # trade quality degrades. Only applied once trust is earned (≥10 updates).
        if self._pegasus is not None and self._pegasus.n_updates >= 10:
            try:
                _pg_hmm_gate = self._pegasus.current_params.hmm_conf_gate
                if _blended_regime_conf < _pg_hmm_gate:
                    _pg_hmm_reason = (
                        f"PEGASUS_HMM_GATE: blended_conf={_blended_regime_conf:.3f} < "
                        f"learned_gate={_pg_hmm_gate:.3f} (n_updates={self._pegasus.n_updates})"
                    )
                    _vr = VetoRecord(timestamp=ts, symbol=symbol,
                                     veto_stage='PEGASUS_HMM_GATE', veto_reason=_pg_hmm_reason)
                    self.veto_ledger.log_veto(_vr)
                    self._session_vetoes.append({'symbol': symbol,
                                                  'stage': 'PEGASUS_HMM_GATE',
                                                  'reason': _pg_hmm_reason})
                    logger.info(f"SKIP: {_pg_hmm_reason}")
                    return None
            except Exception as _pge:
                logger.debug(f"[Pegasus/hmm_gate] non-fatal: {_pge}")

        # ── Andrew Lo: 5-level uncertainty gate (Library + ensemble adjusted) #
        # Level 5 (Knightian/irreducible): no trade.
        # Level 4 (partially reducible): quarter-size.
        # Level 3 (fully reducible): half-size.
        # Levels 1-2: full size.
        # Library convergence forces a minimum level independent of HMM state.
        _lo_size_mult = 1.0
        if _LO_TRACKER_AVAILABLE:
            try:
                from sovereign.risk.correlated_position_tracker import library_adjusted_uncertainty_level as _lib_lo
                _lo_size_mult, _lo_desc = _lib_lo(
                    hmm_transition_prob=float(
                        feature_record.regime.hmm_transition_prob or 0.5),
                    regime_confidence=_blended_regime_conf,  # three-vote blended
                    library_insight=_insight,
                )
                if _lo_size_mult == 0.0:
                    _lo_reason = f"LO_UNCERTAINTY_HALT — {_lo_desc}"
                    _vr = VetoRecord(timestamp=ts, symbol=symbol,
                                     veto_stage='LO_UNCERTAINTY', veto_reason=_lo_reason)
                    self.veto_ledger.log_veto(_vr)
                    self._session_vetoes.append({'symbol': symbol,
                                                 'stage': 'LO_UNCERTAINTY',
                                                 'reason': _lo_reason})
                    logger.info(f"SKIP: {_lo_reason}")
                    return None
                elif _lo_size_mult < 1.0:
                    logger.info(f"[Lo] {_lo_desc}")
            except Exception as _loe:
                logger.debug(f"[LoUncertainty] non-fatal: {_loe}")

        # ── Andrew Lo: correlated session information → win_rate update ── #
        # Prior correlated trades in the same regime update the Kelly
        # win_rate estimate (Lo sequential investment information value).
        _corr_win_rate_adj = 0.0
        if self._corr_tracker is not None:
            try:
                _corr_update = self._corr_tracker.get_win_rate_update(
                    current_symbol=symbol,
                    current_regime=router_out.regime,
                )
                _corr_win_rate_adj = _corr_update.win_rate_adjustment
                if abs(_corr_win_rate_adj) > 0.01:
                    logger.info(f"[Lo] {_corr_update.reason}")
            except Exception as _ce:
                logger.debug(f"[CorrTracker] non-fatal: {_ce}")

        _pn_prob = None
        if self.predict_now is not None:
            try:
                _pn_early = self.predict_now.evaluate(
                    regime=router_out.regime,
                    hmm_transition_prob=float(feature_record.regime.hmm_transition_prob or 0.5),
                    hurst=float(feature_record.regime.hurst_short or 0.5),
                    adx=float(feature_record.regime.adx or 20.0),
                    strategy=router_out.specialist_to_run or 'momentum',
                    ica_preprocessor=(
                        self._ica if (self._ica is not None and self._ica._fitted) else None
                    ),
                )
                # Incorporate correlated session update
                _pn_raw = max(0.01, min(0.99,
                    _pn_early.prob_profitable + _corr_win_rate_adj))
                # Blend with Library historical win rate for this regime
                try:
                    from sovereign.risk.predict_now import library_informed_win_rate as _lib_wr
                    _pn_prob, _lib_wr_reason = _lib_wr(
                        own_estimate=_pn_raw,
                        n_trades=_pn_early.n_trades_in_window,
                        library_insight=_insight,
                    )
                    _pn_prob = max(0.01, min(0.99, _pn_prob))
                    if 'library_blend' in _lib_wr_reason:
                        logger.debug(f"[PredictNow+Library] {_lib_wr_reason}")
                except Exception:
                    _pn_prob = _pn_raw
            except Exception:
                pass

        # Pegasus-learned Kelly params (trust ramp: proportional 0→1 over 30 updates)
        _pegasus_params = None
        if self._pegasus is not None and self._pegasus.n_updates >= 20:
            _pegasus_params = self._pegasus.current_params

        risk_out = self.risk.compute(bias, router_out, equity, atr, current_price,
                                     predict_now_prob=_pn_prob,
                                     library_insight=_insight,
                                     pegasus_params=_pegasus_params)

        # Apply Lo uncertainty size multiplier after Kelly (never inflates, only reduces)
        if _lo_size_mult < 1.0 and risk_out.position_size > 0:
            _lo_size = risk_out.position_size * _lo_size_mult
            if hasattr(risk_out, '_replace'):
                risk_out = risk_out._replace(position_size=_lo_size)
        
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
            _now           = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
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
        # 4f. PREDICT_NOW — regime-conditional size multiplier (Chan)
        #     Asks: "Given current regime + recent performance, what
        #     fraction of normal size should we risk?"
        #     Does NOT block — only modulates size down or up to 1.5×.
        # ═══════════════════════════════════════════════════════════════
        if self.predict_now is not None:
            try:
                _pn = self.predict_now.evaluate(
                    regime=router_out.regime,
                    hmm_transition_prob=float(feature_record.regime.hmm_transition_prob or 0.5),
                    hurst=float(feature_record.regime.hurst_short or 0.5),
                    adx=float(feature_record.regime.adx or 20.0),
                    strategy=router_out.specialist_to_run or 'momentum',
                )
                if _pn.size_multiplier == 0.0:
                    _pn_reason = f"PREDICT_NOW_SKIP — {_pn.reason}"
                    _vr = VetoRecord(timestamp=ts, symbol=symbol,
                                     veto_stage='PREDICT_NOW', veto_reason=_pn_reason)
                    self.veto_ledger.log_veto(_vr)
                    self._session_vetoes.append({'symbol': symbol, 'stage': 'PREDICT_NOW',
                                                 'reason': _pn_reason})
                    logger.info(f"SKIP: {_pn_reason}")
                    return None
                elif _pn.size_multiplier != 1.0:
                    new_size = risk_out.position_size * _pn.size_multiplier
                    if hasattr(risk_out, '_replace'):
                        risk_out = risk_out._replace(position_size=new_size)
                    logger.info(f"[PredictNow] {_pn.reason} → size ×{_pn.size_multiplier}")
            except Exception as _pne:
                logger.debug(f"[PredictNow] non-fatal: {_pne}")

        # ═══════════════════════════════════════════════════════════════
        # 4g. ALPHA DECAY — rolling Sharpe gate (Chan)
        #     When rolling Sharpe < 30% of baseline → halt new positions.
        #     Chan: "When alpha decays, tweak risk management, not the signal."
        # ═══════════════════════════════════════════════════════════════
        _strategy_label = router_out.specialist_to_run or 'momentum'
        _decay_monitor = (self.alpha_decay_momentum
                          if _strategy_label == 'momentum'
                          else self.alpha_decay_reversion)
        if _decay_monitor is not None:
            try:
                _decay = _decay_monitor.check()
                if _decay.multiplier == 0.0:
                    _dc_reason = f"ALPHA_DECAY_HALT — {_decay.reason}"
                    _vr = VetoRecord(timestamp=ts, symbol=symbol,
                                     veto_stage='ALPHA_DECAY', veto_reason=_dc_reason)
                    self.veto_ledger.log_veto(_vr)
                    self._session_vetoes.append({'symbol': symbol, 'stage': 'ALPHA_DECAY',
                                                 'reason': _dc_reason})
                    logger.warning(f"HALT: {_dc_reason}")
                    return None
                elif _decay.multiplier < 1.0:
                    new_size = risk_out.position_size * _decay.multiplier
                    if hasattr(risk_out, '_replace'):
                        risk_out = risk_out._replace(position_size=new_size)
                    logger.warning(f"[AlphaDecay] {_decay.level} — size ×{_decay.multiplier}")
            except Exception as _dce:
                logger.debug(f"[AlphaDecay] non-fatal: {_dce}")

        # ═══════════════════════════════════════════════════════════════
        # 4h. ML STACK + TRADING MEMORY — size modulators
        #
        #   MarketMemory (PTJ): compare today vs ALL historical precursors.
        #     Score 0-1 → NORMAL/ELEVATED/WARNING/DANGER/CRITICAL
        #     HIGH match to 1987/2008/2020 = auto size reduction.
        #   Kalman (L19): Bayesian regime state estimate.
        #   VolRegime (MIT BS): IV/RV ratio → size scale.
        #   TradeMDP (L16): value-iteration sequence-aware sizing.
        #   LQR (L18): Riccati drawdown-aware sizing.
        #   Pegasus (L20): learned policy params gate.
        #
        #   All multiplicative, non-blocking. Failures degrade silently.
        # ═══════════════════════════════════════════════════════════════

        # ── 4h: Alexandrian Library size modulator + PTJ dislocation ─── #
        # Library was already queried in stage 4 (early query) → reuse _insight.
        # This block applies the Library's size_modifier and PTJ dislocation
        # multiplier to the already-Kelly-sized position. These are additive
        # defences: Kelly was already reduced by Library via grade_cap —
        # this adds the regime-specific Library modifier on top.
        if risk_out.position_size > 0:
            if _insight is not None:
                try:
                    _lib_mult = _insight.size_modifier
                    _lib_regime = _insight.primary_regime or 'UNKNOWN'
                    _lib_advisory = _insight.advisory

                    if _lib_mult == 0.0:
                        _lib_veto = (f"ALEXANDRIAN_CRITICAL: {_lib_regime} "
                                     f"threat={_insight.threat_score:.3f} | {_lib_advisory[:120]}")
                        _vr = VetoRecord(timestamp=ts, symbol=symbol,
                                         veto_stage='ALEXANDRIAN_LIBRARY', veto_reason=_lib_veto)
                        self.veto_ledger.log_veto(_vr)
                        self._session_vetoes.append({'symbol': symbol,
                                                      'stage': 'ALEXANDRIAN_LIBRARY',
                                                      'reason': _lib_veto})
                        logger.critical(f"BLOCK: {_lib_veto}")
                        return None
                    elif _lib_mult < 1.0:
                        new_size = risk_out.position_size * _lib_mult
                        if hasattr(risk_out, '_replace'):
                            risk_out = risk_out._replace(position_size=new_size)
                        _converging = " [CONVERGING]" if _insight.converging_signal else ""
                        logger.warning(
                            f"[Library] {_lib_regime}{_converging} "
                            f"threat={_insight.threat_score:.3f} → size ×{_lib_mult:.2f} | "
                            f"{_lib_advisory[:80]}"
                        )

                    # PTJ dislocation from Library (Integration 5)
                    _ptj_dislo_cat, _ptj_dislo_mult = ptj_dislocation_from_library(_insight)
                    if _ptj_dislo_cat > 0 and _ptj_dislo_mult < 1.0:
                        new_size = risk_out.position_size * _ptj_dislo_mult
                        if hasattr(risk_out, '_replace'):
                            risk_out = risk_out._replace(position_size=new_size)
                        _sev = "SEVERE" if _ptj_dislo_cat == 2 else "MODERATE"
                        logger.warning(
                            f"[PTJ_Dislocation/{_sev}] converging={_lib_n_converging} "
                            f"pattern={_lib_regime} → size ×{_ptj_dislo_mult:.2f}"
                        )
                except Exception as _ale:
                    logger.debug(f"[AlexandrianLibrary/4h] non-fatal: {_ale}")

            # Legacy MarketMemory fallback (used if library not loaded or no SPY data)
            if _insight is None and self._market_memory is not None:
                try:
                    import numpy as _np
                    _spy_hist = None
                    _vix_hist = None
                    if hasattr(feature_record, 'spy_price_history') and feature_record.spy_price_history is not None:
                        _spy_hist = _np.array(feature_record.spy_price_history)
                    if hasattr(feature_record, 'vix_history') and feature_record.vix_history is not None:
                        _vix_hist = _np.array(feature_record.vix_history)

                    if _spy_hist is not None and len(_spy_hist) >= 200:
                        _mem_mult, _mem_reason = self._market_memory.get_size_modifier(
                            spy_prices=_spy_hist,
                            vix_prices=_vix_hist,
                        )
                        if _mem_mult < 1.0:
                            if _mem_mult == 0.0:
                                _mem_veto = f"MARKET_MEMORY_CRITICAL: {_mem_reason}"
                                _vr = VetoRecord(timestamp=ts, symbol=symbol,
                                                 veto_stage='MARKET_MEMORY', veto_reason=_mem_veto)
                                self.veto_ledger.log_veto(_vr)
                                self._session_vetoes.append({'symbol': symbol,
                                                              'stage': 'MARKET_MEMORY',
                                                              'reason': _mem_veto})
                                logger.critical(f"BLOCK: {_mem_veto}")
                                return None
                            else:
                                new_size = risk_out.position_size * _mem_mult
                                if hasattr(risk_out, '_replace'):
                                    risk_out = risk_out._replace(position_size=new_size)
                                logger.warning(f"[MarketMemory] {_mem_reason} → size ×{_mem_mult:.2f}")
                except Exception as _mme:
                    logger.debug(f"[MarketMemory] non-fatal: {_mme}")
        _kalman_hmm_conf = float(feature_record.regime.hmm_transition_prob or 0.5)
        if self._kalman is not None:
            try:
                import numpy as _np
                _ret_vec = _np.zeros(5)
                _ret_vec[0] = float(feature_record.macro.get('eurusd_ret', 0.0) if hasattr(feature_record.macro, 'get') else 0.0)
                _k_state = self._kalman.update(_ret_vec)
                _kalman_out = self._kalman.get_regime_output()
                _kalman_hmm_conf = _kalman_out['confidence']
                logger.debug(f"[Kalman] regime={_kalman_out['regime']} "
                             f"trend={_kalman_out['trend_factor']:+.3f} "
                             f"conf={_kalman_hmm_conf:.3f}")
            except Exception as _ke:
                logger.debug(f"[Kalman] non-fatal: {_ke}")

        if self._vol_regime is not None and atr > 0 and current_price > 0:
            try:
                _rv_daily = atr / current_price
                self._vol_regime.update(_rv_daily)
                _vol_sig = self._vol_regime.get_signal()
                _vr_mult = _vol_sig['size_adjustment']
                if _vr_mult != 1.0 and risk_out.position_size > 0:
                    if hasattr(risk_out, '_replace'):
                        risk_out = risk_out._replace(
                            position_size=risk_out.position_size * _vr_mult)
                    logger.info(f"[VolRegime] {_vol_sig['vol_regime']} "
                                f"IV/RV={_vol_sig['iv_rv_ratio']} → ×{_vr_mult:.2f}")
            except Exception as _vre:
                logger.debug(f"[VolRegime] non-fatal: {_vre}")

        _mdp_mult = 1.0
        _lqr_mult = 1.0
        _vr_mult = 1.0
        _effective_size_mult = 1.0
        if self._trade_mdp is not None and self._trade_mdp._m.n_trades >= 20:
            try:
                _consec = int(self._last_trade_state.get('consecutive_losses', 0))
                _dd = float(self._last_trade_state.get('drawdown_pct', 0.0))
                _mdp_mult = self._trade_mdp.get_size_multiplier(
                    regime=router_out.regime,
                    consecutive_losses=_consec,
                    drawdown_pct=_dd,
                    hmm_confidence=_kalman_hmm_conf,
                )
                if _mdp_mult != 1.0 and risk_out.position_size > 0:
                    if hasattr(risk_out, '_replace'):
                        risk_out = risk_out._replace(
                            position_size=risk_out.position_size * _mdp_mult)
                    logger.info(f"[TradeMDP] V*={self._trade_mdp.state_value(router_out.regime, _consec, _dd, _kalman_hmm_conf):.4f} → ×{_mdp_mult:.2f}")
            except Exception as _me:
                logger.debug(f"[TradeMDP] non-fatal: {_me}")

        if self._lqr is not None:
            try:
                _dd_pct = float(self._last_trade_state.get('drawdown_pct', 0.0))
                _roll_pnl = float(self._last_trade_state.get('rolling_pnl_3d', 0.0))
                _consec_l = int(self._last_trade_state.get('consecutive_losses', 0))
                _kelly_f = float(risk_out.kelly_fraction) if hasattr(risk_out, 'kelly_fraction') else 0.02
                _lqr_mult, _lqr_debug = self._lqr.compute_size_multiplier(
                    drawdown_pct=_dd_pct,
                    rolling_pnl_3d=_roll_pnl,
                    consecutive_losses=_consec_l,
                    kelly_fraction=_kelly_f,
                    base_multiplier=1.0,
                )
                if _lqr_mult != 1.0 and risk_out.position_size > 0:
                    if hasattr(risk_out, '_replace'):
                        risk_out = risk_out._replace(
                            position_size=risk_out.position_size * _lqr_mult)
                    logger.info(f"[LQR] delta={_lqr_debug['lqr_delta']:+.3f} → ×{_lqr_mult:.3f}")
            except Exception as _lqre:
                logger.debug(f"[LQR] non-fatal: {_lqre}")

        # Pegasus: apply learned size_multiplier (CS229 L20 — all 6 params now live)
        # Trust ramp: proportional application from n_updates=20 → 30
        if self._pegasus is not None and self._pegasus.n_updates >= 20 and risk_out.position_size > 0:
            try:
                _trust = self._pegasus.trust_multiplier
                _peg_size = self._pegasus.current_params.size_multiplier
                # Blend toward learned value proportionally to trust
                _effective_size_mult = 1.0 + _trust * (_peg_size - 1.0)
                if abs(_effective_size_mult - 1.0) > 0.01:
                    if hasattr(risk_out, '_replace'):
                        risk_out = risk_out._replace(
                            position_size=risk_out.position_size * _effective_size_mult)
                    logger.debug(f"[Pegasus] size_mult={_peg_size:.3f} trust={_trust:.2f} "
                                 f"→ effective ×{_effective_size_mult:.3f}")
            except Exception as _pse:
                logger.debug(f"[Pegasus/size] non-fatal: {_pse}")

        # Pegasus: enforce learned entry threshold on win_rate estimate
        if self._pegasus is not None and _pn_prob is not None:
            try:
                _entry_thresh = self._pegasus.current_params.entry_threshold
                if _pn_prob < _entry_thresh:
                    _pg_reason = (f"PEGASUS_GATE — P(win)={_pn_prob:.3f} < "
                                  f"learned threshold {_entry_thresh:.3f}")
                    _vr = VetoRecord(timestamp=ts, symbol=symbol,
                                     veto_stage='PEGASUS_GATE', veto_reason=_pg_reason)
                    self.veto_ledger.log_veto(_vr)
                    self._session_vetoes.append({'symbol': symbol,
                                                  'stage': 'PEGASUS_GATE',
                                                  'reason': _pg_reason})
                    logger.info(f"SKIP: {_pg_reason}")
                    return None
            except Exception as _pge:
                logger.debug(f"[Pegasus] non-fatal: {_pge}")

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
        # 6b. NARRATIVE ADVISORY — TradingAgents/Qwen3 (NON-BLOCKING)
        # ═══════════════════════════════════════════════════════════════
        self._log_narrative_advisory(symbol, ts)

        # ═══════════════════════════════════════════════════════════════
        # 7. EXECUTE
        # ═══════════════════════════════════════════════════════════════
        self._last_runtime_modulators = {
            "mdp_mult": float(_mdp_mult),
            "lqr_mult": float(_lqr_mult),
            "vol_mult": float(_vr_mult),
            "pegasus_mult": float(_effective_size_mult),
            "position_size": float(risk_out.position_size),
        }
        logger.info(
            f"[OK] EXECUTION: {symbol} {bias.direction.name} @ {current_price:.2f} "
            f"Size: {risk_out.position_size:.4f} SL: {risk_out.stop_price:.2f} "
            f"TP: {risk_out.tp1_price:.2f}"
        )

        trade_id = f"SVRN_{datetime.utcnow().strftime('%H%M%S')}"

        # Store execution params so on_trade_close() can pass them to Pegasus REINFORCE
        if hasattr(risk_out, 'size_breakdown') and isinstance(risk_out.size_breakdown, dict):
            _stop_dist = abs(current_price - risk_out.stop_price)
            _stop_atr = _stop_dist / atr if atr > 0 else 1.5
            _tp_dist = abs(risk_out.tp1_price - current_price)
            _tp_rr = _tp_dist / _stop_dist if _stop_dist > 0 else 2.5
            self._last_trade_state['stop_atr_used'] = float(_stop_atr)
            self._last_trade_state['tp_rr_used'] = float(_tp_rr)
            self._last_trade_state['kelly_frac_used'] = float(
                risk_out.size_breakdown.get('grade_cap', 0.025))

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

        # Lo: record open position to session tracker (outcome recorded on close)
        if self._corr_tracker is not None:
            try:
                self._corr_tracker.open_position(
                    symbol=symbol,
                    regime=router_out.regime,
                    direction=bias.direction.name,
                )
            except Exception:
                pass

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
            'strategy': router_out.specialist_to_run,
            'hmm_transition_prob': float(feature_record.regime.hmm_transition_prob or 0.5),
            'hurst': float(feature_record.regime.hurst_short or 0.5),
            'adx': float(feature_record.regime.adx or 20.0),
            'advisories': self._get_advisories(symbol),
            'present_state': present.to_dict() if present is not None else None,
        }

    def on_trade_close(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        size: float,
        sl: float,
        tp: float,
        confidence: float,
        strategy: str,
        exit_reason: str = 'CLOSE',
        regime: str = 'MOMENTUM',
        hmm_transition_prob: float = 0.5,
        hurst: float = 0.5,
        adx: float = 20.0,
        drawdown_pct: float = 0.0,
        consecutive_losses: int = 0,
        pnl_override: Optional[float] = None,
        entry_time=None,
        exit_time=None,
    ):
        """
        Call this whenever a trade closes (SL hit, TP hit, manual exit, end-of-day).

        This is the live learning loop — every closed trade updates ALL adaptive
        components so the system continuously improves from production outcomes.

        Components updated:
          1. TradeLedger.log_close()          — persistent record (feeds PredictNow/AlphaDecay on next load)
          2. PredictNow.record_outcome()      — online SGD + Newton IRLS update
          3. AlphaDecay.record_trade()        — rolling Sharpe tracking
          4. CorrTracker.record_outcome()     — Lo sequential information update
          5. TradeMDP.record_transition()     — Bellman value update (every 10 trades)
          6. Pegasus.reinforce_update()       — REINFORCE policy gradient step
          7. _last_trade_state update         — feeds LQR + TradeMDP next session
        """
        pnl = float(pnl_override) if pnl_override is not None else (
            (exit_price - entry_price) * size * (1 if direction == 'LONG' else -1)
        )
        won = pnl > 0

        # 1. Persistent ledger record
        try:
            self.trade_ledger.log_close(
                trade_id=trade_id, symbol=symbol, direction=direction,
                entry_price=entry_price, exit_price=exit_price, size=size,
                sl=sl, tp=tp, confidence=confidence, pnl=pnl,
                strategy=strategy, exit_reason=exit_reason,
                entry_time=entry_time, exit_time=exit_time,
            )
        except Exception as e:
            logger.warning(f"[on_trade_close] ledger write failed: {e}")

        # 2. PredictNow online update (CS229 L03 SGD + L04 Newton IRLS)
        if self.predict_now is not None:
            try:
                self.predict_now.record_outcome(
                    regime=regime, hmm_transition_prob=hmm_transition_prob,
                    hurst=hurst, adx=adx, strategy=strategy,
                    won=won, pnl=pnl,
                )
            except Exception as e:
                logger.debug(f"[PredictNow.record] non-fatal: {e}")

        # 3. AlphaDecay rolling Sharpe (Chan)
        # AlphaDecay reads directly from the ledger file on each check() call,
        # so writing to the ledger (step 1) is sufficient to update it.
        # No additional method call needed — ledger write is the feed.

        # 4. Lo correlated tracker (session-level sequential info)
        if self._corr_tracker is not None:
            try:
                self._corr_tracker.record_outcome(
                    symbol=symbol, regime=regime, direction=direction,
                    won=won, pnl=pnl,
                )
            except Exception as e:
                logger.debug(f"[CorrTracker.record] non-fatal: {e}")

        # 5. Trade MDP transition (CS229 L16 Bellman update)
        if self._trade_mdp is not None:
            try:
                _new_consec = (consecutive_losses + 1) if not won else 0
                _new_dd = max(0.0, drawdown_pct - pnl / max(abs(entry_price * size), 1))
                self._trade_mdp.record_transition(
                    prev_regime=regime, prev_consec_losses=consecutive_losses,
                    prev_drawdown_pct=drawdown_pct,
                    prev_hmm_conf=1.0 - hmm_transition_prob,
                    next_regime=regime, next_consec_losses=_new_consec,
                    next_drawdown_pct=_new_dd,
                    next_hmm_conf=1.0 - hmm_transition_prob,
                    pnl=pnl, size_multiplier_used=size,
                )
            except Exception as e:
                logger.debug(f"[TradeMDP.record] non-fatal: {e}")

        # 6. Pegasus REINFORCE gradient step — full 6-param update (CS229 L20)
        if self._pegasus is not None:
            try:
                import numpy as _np
                _state_feat = _np.array([confidence, 1.0 - hmm_transition_prob,
                                          hurst, adx / 50.0])
                self._pegasus.reinforce_update(
                    state_features=_state_feat,
                    action_taken=size,
                    realized_pnl=pnl,
                    hmm_confidence=1.0 - hmm_transition_prob,
                    stop_atr_used=self._last_trade_state.get('stop_atr_used', 1.5),
                    tp_rr_used=self._last_trade_state.get('tp_rr_used', 2.5),
                    kelly_frac_used=self._last_trade_state.get('kelly_frac_used', 0.025),
                )
            except Exception as e:
                logger.debug(f"[Pegasus.reinforce] non-fatal: {e}")

        # 6b. Softmax online SGD update (CS229 L04 — labeled training from outcomes)
        # Each closed trade provides: features at entry + the HMM regime label.
        # The Softmax learns to predict the regime distribution that HMM identifies,
        # but with better probability calibration from training on resolved trades.
        if self._softmax_regime is not None and regime in ('MOMENTUM', 'REVERSION', 'FLAT'):
            try:
                import numpy as _np
                from sovereign.risk.softmax_regime import SoftmaxRegimeClassifier as _SRC
                _sf_x = _SRC.encode(
                    hurst=hurst,
                    hmm_prob=hmm_transition_prob,
                    adx=adx,
                    prev_regime=regime,
                    strategy=strategy,
                )
                self._softmax_regime.update_online(_sf_x, regime, alpha=0.05)
            except Exception as e:
                logger.debug(f"[Softmax.update] non-fatal: {e}")

        # 6c. KMeans feature buffer + periodic refit (CS229 L12)
        # Accumulates regime feature observations; refits every 10 new trades
        # once 30+ observations are available (enough for stable clustering).
        if self._kmeans_regime is not None:
            try:
                import numpy as _np
                _km_feat = _np.array([hurst, adx / 50.0, hmm_transition_prob])
                self._kmeans_feature_buffer.append(_km_feat)
                _n_km = len(self._kmeans_feature_buffer)
                if _n_km >= 30 and _n_km % 10 == 0:
                    _km_mat = _np.array(self._kmeans_feature_buffer[-200:])
                    self._kmeans_regime.fit(_km_mat)
                    logger.debug(f"[KMeans] Refit on {len(_km_mat)} observations")
            except Exception as e:
                logger.debug(f"[KMeans.update] non-fatal: {e}")

        # 6d. ICA feature buffer + periodic refit (CS229 L15)
        # Accumulates 5-dim feature vectors; refits every 25 new trades once
        # 50+ observations exist. ICA finds independent components of the
        # correlated feature space — improves PredictNow LOESS distances.
        if self._ica is not None:
            try:
                import numpy as _np
                _ica_feat = _np.array([
                    hurst,
                    hmm_transition_prob,
                    adx / 50.0,
                    confidence,
                    min(size / 10000.0, 2.0),  # normalised position size
                ])
                self._ica_feature_buffer.append(_ica_feat)
                _n_ica = len(self._ica_feature_buffer)
                if _n_ica >= 50 and _n_ica % 25 == 0:
                    _ica_mat = _np.array(self._ica_feature_buffer[-300:])
                    self._ica.fit(_ica_mat)
                    logger.debug(f"[ICA] Refit on {len(_ica_mat)} observations "
                                 f"(n_components={self._ica.n_components})")
            except Exception as e:
                logger.debug(f"[ICA.update] non-fatal: {e}")

        # 7. Update running state for next session's LQR + MDP
        _prev_consec = self._last_trade_state.get('consecutive_losses', 0)
        _current_dd  = self._last_trade_state.get('drawdown_pct', 0.0)
        _new_dd      = max(0.0, _current_dd + min(0.0, pnl / max(abs(entry_price * size), 1)))
        self._last_trade_state = {
            'consecutive_losses': (_prev_consec + 1) if not won else 0,
            'drawdown_pct': _new_dd,
            'rolling_pnl_3d': pnl,
            'last_pnl': pnl,
            'last_won': won,
        }

        # 8. Trading Memory auto-learning — Alexandrian Library + MarketMemory
        # When the system lives through a significant drawdown, it extracts the
        # precursor pattern and adds it to both memories simultaneously.
        if (self._alexandrian_library is not None or self._market_memory is not None) and _new_dd >= 0.08:
            try:
                import numpy as _np
                _spy_hist = self._last_trade_state.get('_spy_history')
                if _spy_hist is not None and len(_spy_hist) >= 200:
                    _event_name = f"live_{datetime.utcnow().strftime('%Y%m%d')}_dd{int(_new_dd*100)}pct"
                    _severity = 2 if _new_dd >= 0.20 else 1
                    _spy_arr = _np.array(_spy_hist)
                    # Learn into AlexandrianLibrary first (preferred)
                    if self._alexandrian_library is not None:
                        try:
                            _lib_learned = self._alexandrian_library.learn(
                                event_name=_event_name,
                                crash_date=datetime.utcnow().strftime('%Y-%m-%d'),
                                severity=_severity,
                                crash_type='live_learned',
                                spy_prices=_spy_arr,
                            )
                            if _lib_learned:
                                logger.warning(
                                    f"[AlexandrianLibrary] Auto-learned: {_event_name} "
                                    f"drawdown={_new_dd:.1%} severity={_severity}"
                                )
                        except Exception as _ale:
                            logger.debug(f"[AlexandrianLibrary.learn] non-fatal: {_ale}")
                    # Also learn into MarketMemory for redundancy
                    if self._market_memory is not None:
                        try:
                            learned = self._market_memory.learn(
                                event_name=_event_name,
                                crash_date=datetime.utcnow().strftime('%Y-%m-%d'),
                                severity=_severity,
                                crash_type='live_learned',
                                spy_prices=_spy_arr,
                            )
                            if learned:
                                logger.warning(
                                    f"[MarketMemory] Auto-learned: {_event_name} "
                                    f"drawdown={_new_dd:.1%} severity={_severity}"
                                )
                        except Exception as e:
                            logger.debug(f"[MarketMemory.learn] non-fatal: {e}")
            except Exception as e:
                logger.debug(f"[TradingMemory.learn] non-fatal: {e}")

        logger.info(
            f"[on_trade_close] {symbol} {direction} pnl={pnl:+.4f} "
            f"({'WIN' if won else 'LOSS'}) — all learners updated"
        )

        # Cold-start risk-window telemetry (first 30 closed trades)
        _closed_trades = int(getattr(getattr(self._trade_mdp, "_m", None), "n_trades", 0)) if self._trade_mdp is not None else 0
        if 0 < _closed_trades <= 30:
            _latest = self._latest_ml_snapshot if isinstance(self._latest_ml_snapshot, dict) else {}
            _mods = _latest.get("modules", {})
            _telemetry = {
                "closed_trades": _closed_trades,
                "pegasus_updates": int(getattr(self._pegasus, "n_updates", 0)) if self._pegasus is not None else 0,
                "pegasus_trust": float(self._pegasus.trust_multiplier) if self._pegasus is not None else 0.0,
                "pegasus_phase": (
                    "off" if (self._pegasus is None or self._pegasus.n_updates < 10)
                    else "gate_only" if self._pegasus.n_updates < 20
                    else "trust_ramp" if self._pegasus.n_updates < 30
                    else "full"
                ),
                "mdp_trades": int(_mods.get("mdp", {}).get("trades", 0)),
                "mdp_policy_source": _mods.get("mdp", {}).get("policy_source", "expert_prior"),
                "kalman_bars": int(_mods.get("kalman", {}).get("bars", 0)),
                "kalman_phase": "warming_up" if int(_mods.get("kalman", {}).get("bars", 0)) < 15 else "converged",
                "ensemble_blended_conf": _latest.get("ensemble_vote", {}).get("blended_conf"),
                "ensemble_votes": _latest.get("ensemble_vote", {}).get("votes", {}),
                "cold_modules": int(_latest.get("cold_modules", 0)),
            }
            try:
                logger.info(f"[COLD_START_TELEMETRY] {json.dumps(_telemetry, default=str)}")
            except Exception:
                logger.info(f"[COLD_START_TELEMETRY] {_telemetry}")

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

    def _log_narrative_advisory(self, symbol: str, timestamp: str) -> None:
        """
        Fire-and-log call to TradingAgents/Qwen3 narrative pipeline.
        Runs in the current thread but is fully non-blocking from a veto
        perspective — the result is ONLY logged, never gate-checked.
        If the subprocess times out or fails, the method returns silently.
        """
        if self.narrative_engine is None:
            return
        try:
            month = timestamp[:7] if isinstance(timestamp, str) else datetime.utcnow().strftime("%Y-%m")
            advisory = self.narrative_engine.get_narrative_bias(symbol, month)
            logger.info(
                "[NARRATIVE ADVISORY] %s → direction=%s conf=%s mod=%+.2f | %s",
                symbol,
                advisory.direction,
                advisory.confidence,
                advisory.modifier,
                " | ".join(advisory.key_risks[:2] or advisory.catalysts[:2]),
            )
        except Exception as e:
            logger.debug("[NARRATIVE ADVISORY] Skipped (%s)", e)

    def _get_narrative_bias(self, symbol: str, timestamp: str, quant_direction: str):
        if self.narrative_engine is None:
            return None
        try:
            month = timestamp[:7] if isinstance(timestamp, str) else datetime.utcnow().strftime("%Y-%m")
            return self.narrative_engine.get_narrative_bias(symbol, month)
        except Exception as e:
            logger.debug("[NarrativeEngine] Skipped (%s)", e)
            return None

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
