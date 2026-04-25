"""
Harvest Veto — Stage 4c Live Execution Gate
============================================
Uses the XGBoost model trained by training/retrain_loop.py on millions of
backtested trades to predict P(this trade is profitable | current conditions).

Blocks the trade if P(profitable) < current_threshold.

The threshold starts at 0.50 and rises automatically every 4 hours — but only
when the retrain loop confirms that raising it produced MORE monthly PnL than
the prior month. Selectivity earns its keep or it doesn't happen.

Model and threshold auto-reload when the retrain loop writes new files,
so the live system gets smarter without any restarts.

Usage:
    from sovereign.risk.harvest_veto import HarvestVeto

    veto = HarvestVeto()
    blocked, reason, proba = veto.should_block(
        regime=0,
        hurst=0.62,
        atr_norm=0.011,
        vol_pct=0.45,
        direction=1,
        month=4,
        day_of_week=1,
        stop_atr_mult=2.0,
        tp_rr=2.0,
    )
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parent.parent.parent
MODEL_PATH   = ROOT / "models" / "xgb_veto.json"
THRESH_PATH  = ROOT / "models" / "current_threshold.json"

# Reload model from disk at most once every 60 seconds (cheap check)
_RELOAD_INTERVAL = 60.0

FEATURE_ORDER = [
    "stop_atr_mult", "tp_rr", "atr_period",
    "regime", "hurst", "atr_norm", "vol_pct",
    "month", "day_of_week", "direction",
]


class HarvestVeto:
    """
    Live veto gate backed by the continuously-trained harvest model.

    Thread-safe for single-threaded orchestrator use.
    Not suitable for multiprocessing — each process should create its own instance.
    """

    def __init__(self) -> None:
        self._model = None
        self._threshold: float = 0.50
        self._model_mtime: float = 0.0
        self._thresh_mtime: float = 0.0
        self._last_check: float = 0.0
        self.ready: bool = False
        self._try_load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _try_load(self) -> None:
        """Load or reload model + threshold if files have changed."""
        now = time.monotonic()
        if now - self._last_check < _RELOAD_INTERVAL:
            return
        self._last_check = now

        # Model
        if MODEL_PATH.exists():
            mtime = MODEL_PATH.stat().st_mtime
            if mtime != self._model_mtime:
                try:
                    import xgboost as xgb
                    m = xgb.XGBClassifier()
                    m.load_model(str(MODEL_PATH))
                    self._model = m
                    self._model_mtime = mtime
                    logger.info(f"[HarvestVeto] Model reloaded from {MODEL_PATH.name}")
                except Exception as e:
                    logger.warning(f"[HarvestVeto] Model load failed: {e}")

        # Threshold
        if THRESH_PATH.exists():
            mtime = THRESH_PATH.stat().st_mtime
            if mtime != self._thresh_mtime:
                try:
                    data = json.loads(THRESH_PATH.read_text())
                    self._threshold = float(data["threshold"])
                    self._thresh_mtime = mtime
                    logger.info(f"[HarvestVeto] Threshold reloaded: {self._threshold:.2f}")
                except Exception as e:
                    logger.warning(f"[HarvestVeto] Threshold load failed: {e}")

        self.ready = self._model is not None

    # ── Veto check ────────────────────────────────────────────────────────────

    def should_block(
        self,
        regime: int,
        hurst: float,
        atr_norm: float,
        vol_pct: float,
        direction: int,
        month: int,
        day_of_week: int,
        stop_atr_mult: float = 2.0,
        tp_rr: float = 2.0,
        atr_period: int = 14,
    ) -> Tuple[bool, str, float]:
        """
        Returns (blocked, reason, proba).
          blocked: True = veto this trade
          reason:  human-readable veto explanation
          proba:   model's P(profitable) estimate (0–1)
        """
        self._try_load()

        if not self.ready:
            return False, "harvest model not ready", 0.5

        x = np.array([[
            stop_atr_mult, tp_rr, atr_period,
            regime, hurst, atr_norm, vol_pct,
            month, day_of_week, direction,
        ]], dtype=np.float32)

        try:
            proba = float(self._model.predict_proba(x)[0, 1])
        except Exception as e:
            logger.warning(f"[HarvestVeto] Predict failed: {e}")
            return False, f"predict error: {e}", 0.5

        blocked = proba < self._threshold
        if blocked:
            reason = (
                f"HarvestVeto: P(profitable)={proba:.2f} < threshold={self._threshold:.2f} "
                f"| regime={regime} hurst={hurst:.2f} atr_norm={atr_norm:.4f}"
            )
        else:
            reason = ""

        return blocked, reason, proba

    def describe(self) -> str:
        return (
            f"HarvestVeto ready={self.ready} "
            f"threshold={self._threshold:.2f} "
            f"model={'loaded' if self._model else 'missing'}"
        )
