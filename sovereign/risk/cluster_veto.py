"""
Cluster Veto — Stage 1 live execution gate

Loads the historical failure map, trains a KNeighborsClassifier (k=10)
to predict which failure cluster a candidate trade's conditions match,
then blocks it if that cluster has avg_loss_r < -2.5.

This is the feedback loop made executable:
  failure_map.csv (2,363 losses) → KNN → should_block() → veto

Usage:
    from sovereign.risk.cluster_veto import ClusterVeto

    veto = ClusterVeto()
    blocked, reason = veto.should_block(
        strategy_name="momentum",
        regime="MOMENTUM",
        atr_pct=2.3,           # ATR as % of price
        spy_week_return=-1.2   # SPY 5-day return in %
    )
    if blocked:
        # log veto and skip trade
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

# MACRO_REVERSAL hard rule thresholds (independent of KNN)
# spy_week_return is a decimal fraction (e.g. -0.021 = -2.1%)
_MACRO_REVERSAL_SPY_THRESHOLD  = -0.015  # SPY 5-day return, decimal (-0.015 = -1.5% drop)
_MACRO_REVERSAL_MAE_MFE_RATIO  = 2.0     # prior MAE/MFE ratio above this = exits too late
_ENRICHED_PATH = Path("logs/live_trades_enriched.csv")

logger = logging.getLogger(__name__)

# Canonical names used in failure_map.csv
_STRATEGY_ALIASES: dict = {
    # Live specialists → backtest strategy names
    "momentum":           "momentum_sma",
    "reversion":          "bb_reversion",
    # Passthrough for backtest names
    "momentum_sma":       "momentum_sma",
    "bb_reversion":       "bb_reversion",
    "donchian_breakout":  "donchian_breakout",
    "atr_channel":        "atr_channel",
}

_REGIME_ALIASES: dict = {
    "MOMENTUM":  "MOMENTUM",
    "REVERSION": "REVERSION",
    "FLAT":      "NEUTRAL",
    "NEUTRAL":   "NEUTRAL",
}

_KNOWN_REGIMES    = ["MOMENTUM", "NEUTRAL", "REVERSION"]
_KNOWN_STRATEGIES = ["atr_channel", "bb_reversion", "donchian_breakout", "momentum_sma"]

# Block threshold: clusters with avg_loss_r below this are vetoed
BLOCK_THRESHOLD = -2.5


class ClusterVeto:
    """KNN-based live trade veto fitted on historical failure conditions.

    Attributes:
        ready (bool): True once the classifier is fitted and ready to veto.
        cluster_info (dict): cluster_id → cluster description dict.
    """

    def __init__(
        self,
        failure_map_path:  str = "logs/failure_map.csv",
        clusters_path:     str = "logs/failure_clusters.json",
    ):
        self.ready = False
        self.cluster_info: dict = {}
        self._knn = None
        self._col_min: Optional[np.ndarray]   = None
        self._col_range: Optional[np.ndarray] = None
        self._le_regime: Optional[object]     = None
        self._le_strategy: Optional[object]   = None

        self._load_and_fit(failure_map_path, clusters_path)
        self._prior_mae_mfe_ratio: float = self._load_prior_mae_mfe_ratio()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load_and_fit(self, fm_path: str, clusters_path: str) -> None:
        try:
            import pandas as pd
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.preprocessing import LabelEncoder
        except ImportError as e:
            logger.error(f"[ClusterVeto] Missing dependency: {e}. Veto disabled.")
            return

        fm = Path(fm_path)
        cp = Path(clusters_path)

        if not fm.exists():
            logger.warning(f"[ClusterVeto] {fm} not found — veto disabled.")
            return
        if not cp.exists():
            logger.warning(f"[ClusterVeto] {cp} not found — veto disabled.")
            return

        df = pd.read_csv(fm)
        required = {"regime_at_entry", "atr_pct", "market_condition",
                    "strategy", "failure_cluster"}
        if not required.issubset(df.columns):
            logger.error(f"[ClusterVeto] failure_map missing columns: {required - set(df.columns)}")
            return

        self.cluster_info = {
            c["cluster"]: c for c in json.loads(cp.read_text())
        }

        # Label encoders — fit on the full known label sets (not just what's in df)
        le_r = LabelEncoder()
        le_r.fit(_KNOWN_REGIMES)
        le_s = LabelEncoder()
        le_s.fit(_KNOWN_STRATEGIES)
        self._le_regime   = le_r
        self._le_strategy = le_s

        df = df[df["regime_at_entry"].isin(_KNOWN_REGIMES)]
        df = df[df["strategy"].isin(_KNOWN_STRATEGIES)]
        df = df.dropna(subset=["atr_pct", "market_condition", "failure_cluster"])

        if len(df) < 10:
            logger.warning("[ClusterVeto] Not enough clean failure rows. Veto disabled.")
            return

        X = self._encode(
            df["regime_at_entry"].tolist(),
            df["atr_pct"].tolist(),
            df["market_condition"].tolist(),
            df["strategy"].tolist(),
        )
        y = df["failure_cluster"].to_numpy(dtype=np.int32)

        # Normalise
        self._col_min   = X.min(axis=0)
        self._col_range = X.max(axis=0) - X.min(axis=0) + 1e-8
        X_norm = (X - self._col_min) / self._col_range

        n_neighbors = min(10, len(df))
        self._knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean")
        self._knn.fit(X_norm, y)

        self.ready = True
        logger.info(
            f"[ClusterVeto] Ready — fitted on {len(df)} failures, "
            f"{len(self.cluster_info)} clusters. "
            f"Block threshold: avg_loss_r < {BLOCK_THRESHOLD}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def should_block(
        self,
        strategy_name:        str,
        regime:               str,
        atr_pct:              float,
        spy_week_return:      float,
        prior_mae_mfe_ratio:  Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Determine if a candidate trade should be blocked.

        Args:
            strategy_name:        Specialist name ('momentum', 'reversion') or backtest
                                  strategy name ('momentum_sma', 'bb_reversion', …).
            regime:               Current regime: 'MOMENTUM', 'REVERSION', 'FLAT', or 'NEUTRAL'.
            atr_pct:              Current ATR as a percentage of price (e.g. 2.3 for 2.3 %).
            spy_week_return:      SPY return over the last 5 trading days, as a decimal fraction
                                  (e.g. -0.021 = -2.1% drop).
            prior_mae_mfe_ratio:  Override for the file-loaded MAE/MFE ratio. If None,
                                  the value loaded from live_trades_enriched.csv is used.

        Returns:
            (block: bool, reason: str)
            block=True  → trade conditions match a high-loss cluster; skip the trade.
            block=False → conditions do not match a dangerous cluster; allow.
        """
        # Hard rule: MACRO_REVERSAL — runs before KNN, no model required
        ratio = prior_mae_mfe_ratio if prior_mae_mfe_ratio is not None else self._prior_mae_mfe_ratio
        blocked, reason = self._check_macro_reversal(spy_week_return, ratio)
        if blocked:
            return True, reason

        if not self.ready:
            return False, ""

        canon_strategy = _STRATEGY_ALIASES.get(strategy_name)
        if canon_strategy is None:
            return False, ""

        canon_regime = _REGIME_ALIASES.get(regime, "NEUTRAL")

        X = self._encode(
            [canon_regime], [atr_pct], [spy_week_return], [canon_strategy]
        )
        X_norm = (X - self._col_min) / self._col_range
        predicted_cluster = int(self._knn.predict(X_norm)[0])

        info = self.cluster_info.get(predicted_cluster, {})
        mean_loss_r = float(info.get("mean_loss_r", 0.0))

        if mean_loss_r < BLOCK_THRESHOLD:
            avoid_cond = info.get("avoid_condition", f"Cluster {predicted_cluster} failure pattern")
            reason = (
                f"CLUSTER_VETO[{predicted_cluster}] "
                f"avg_loss={mean_loss_r:.2f}R — {avoid_cond}"
            )
            return True, reason

        return False, ""

    def describe(self) -> str:
        """Human-readable summary of loaded clusters."""
        if not self.ready:
            return "ClusterVeto: not ready (missing data)"
        lines = ["ClusterVeto — 5 failure patterns loaded:"]
        for cid, info in sorted(self.cluster_info.items()):
            marker = "BLOCKS" if info.get("mean_loss_r", 0) < BLOCK_THRESHOLD else "allows"
            lines.append(
                f"  [{cid}] {marker:6s}  n={info['size']:>4}  "
                f"avg={info['mean_loss_r']:+.2f}R  "
                f"{info.get('avoid_condition','')[:60]}"
            )
        return "\n".join(lines)

    # ── Hard Rules ───────────────────────────────────────────────────────────

    def _load_prior_mae_mfe_ratio(self) -> float:
        """Load MAE/MFE ratio from last row of live_trades_enriched.csv. Default 1.0."""
        try:
            import pandas as pd
            if not _ENRICHED_PATH.exists():
                return 1.0
            df = pd.read_csv(_ENRICHED_PATH)
            if df.empty or "mae_mfe_ratio" not in df.columns:
                return 1.0
            val = df["mae_mfe_ratio"].dropna().iloc[-1]
            return float(val)
        except Exception:
            return 1.0

    def _check_macro_reversal(self, spy_week_return: float, prior_mae_mfe_ratio: float = None) -> Tuple[bool, str]:
        if prior_mae_mfe_ratio is None:
            prior_mae_mfe_ratio = self._prior_mae_mfe_ratio
        """Hard rule: block when SPY is in sharp weekly drawdown AND prior exits were late.

        Catches events like March 16-22 where GLD/SLV/metals stopped 8 consecutive
        times during a macro risk-off week. The KNN clusters miss this because the
        pattern cuts across strategy and regime.

        Conditions (both must be true):
          1. spy_week_return < -1.5%   (macro risk-off environment)
          2. prior_mae_mfe_ratio > 2.0 (previous trades showed adverse excursion > 2× MFE)
        """
        if (spy_week_return < _MACRO_REVERSAL_SPY_THRESHOLD and
                prior_mae_mfe_ratio > _MACRO_REVERSAL_MAE_MFE_RATIO):
            reason = (
                f"MACRO_REVERSAL — SPY {spy_week_return*100:.2f}% week "
                f"(threshold {_MACRO_REVERSAL_SPY_THRESHOLD*100:.1f}%) + "
                f"prior MAE/MFE {prior_mae_mfe_ratio:.2f}x "
                f"(threshold {_MACRO_REVERSAL_MAE_MFE_RATIO}x)"
            )
            return True, reason
        return False, ""

    # ── Internal ──────────────────────────────────────────────────────────────

    def _encode(
        self,
        regimes:    list,
        atrs:       list,
        spy_rets:   list,
        strategies: list,
    ) -> np.ndarray:
        r_enc = self._le_regime.transform(regimes).astype(np.float32)
        s_enc = self._le_strategy.transform(strategies).astype(np.float32)
        return np.column_stack([
            r_enc,
            np.array(atrs,     dtype=np.float32),
            np.array(spy_rets, dtype=np.float32),
            s_enc,
        ])
