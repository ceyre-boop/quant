"""
TrajectoryModel — predicts the full distribution of R-multiples at entry.

Not win/loss. Not direction. The shape of what happens after you enter.

Three quantile regressors trained on 345k historical trades:
  p10 — pessimistic: 10% of outcomes fall below this
  p50 — median: the most likely single outcome
  p90 — optimistic: 90% of outcomes fall below this

Entry filter (replaces gut-feel conviction threshold):
  p50 < 0.20  → veto (expected median is negative)
  p10 < -2.50 → reduce size 50% (tail risk too large)
  p90 > 2.0 AND p50 > 0.50 → max conviction size

Trained on: logs/full_trade_map.csv (345,830 trades, 2018–2024)
Target: exit_r (R-multiple = profit / initial risk)

Reference: López de Prado — Advances in Financial Machine Learning,
           Chapter 3 (labelling) and Chapter 8 (feature importance)
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

MODEL_DIR   = Path(__file__).parents[2] / 'models' / 'prediction'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRADE_MAP   = Path(__file__).parents[2] / 'logs' / 'full_trade_map.csv'
MODEL_PATH  = MODEL_DIR / 'trajectory_model.pkl'
META_PATH   = MODEL_DIR / 'trajectory_meta.json'

# Entry filter thresholds — percentile based, set after training
# These are updated by _calibrate_thresholds() using the training distribution
# Default values are placeholders overwritten at train time
VETO_PERCENTILE          = 25     # veto if predicted p50 is in bottom 25% of all predictions
HALF_SIZE_PERCENTILE     = 15     # half size if in bottom 15% (tail risk)
MAX_SIZE_PERCENTILE      = 75     # max size if in top 25%

STRATEGY_ENCODING = {
    'momentum_sma': 0, 'donchian_breakout': 1,
    'bb_reversion': 2, 'atr_channel': 3,
    'macro_divergence': 4, 'ict_swing': 5,
}
REGIME_ENCODING = {'FLAT': 0, 'MOMENTUM': 1, 'REVERSION': 2}


@dataclass
class TrajectoryPrediction:
    p10: float
    p50: float
    p90: float
    percentile_rank: float   # where p50 sits in training distribution (0–100)
    veto_threshold: float
    half_size_threshold: float
    max_size_threshold: float

    @property
    def trade_verdict(self) -> str:
        if self.p50 <= self.veto_threshold:
            return 'VETO'
        if self.p50 <= self.half_size_threshold:
            return 'HALF_SIZE'
        if self.p50 >= self.max_size_threshold:
            return 'MAX_SIZE'
        return 'NORMAL'

    @property
    def size_modifier(self) -> float:
        v = self.trade_verdict
        return 0.0 if v == 'VETO' else (0.5 if v == 'HALF_SIZE' else (1.5 if v == 'MAX_SIZE' else 1.0))

    def __str__(self) -> str:
        return (f'R[p10={self.p10:+.2f} | p50={self.p50:+.2f} | p90={self.p90:+.2f}]'
                f'  pct={self.percentile_rank:.0f}th'
                f'  →  {self.trade_verdict}  (size ×{self.size_modifier})')


class TrajectoryModel:

    def __init__(self):
        self._p10: Optional[GradientBoostingRegressor] = None
        self._p50: Optional[GradientBoostingRegressor] = None
        self._p90: Optional[GradientBoostingRegressor] = None
        self._meta: dict = {}
        self._fitted = False
        # Calibrated from training distribution — updated at train time
        self._veto_threshold:      float = -3.0
        self._half_size_threshold: float = -4.0
        self._max_size_threshold:  float = -1.0
        self._training_p50_dist:   np.ndarray = np.array([])

    # ── Public API ────────────────────────────────────────────────────── #

    def train(self, force: bool = False) -> dict:
        if not force and MODEL_PATH.exists():
            self._load()
            logger.info(f"TrajectoryModel loaded from cache (trained on {self._meta.get('n_train','?')} trades)")
            return self._meta

        logger.info("TrajectoryModel: loading trade map…")
        df = pd.read_csv(TRADE_MAP)
        X, y = self._build_features(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, shuffle=True
        )

        logger.info(f"TrajectoryModel: fitting on {len(X_train):,} trades…")
        self._p10 = self._fit_quantile(X_train, y_train, alpha=0.10)
        self._p50 = self._fit_quantile(X_train, y_train, alpha=0.50)
        self._p90 = self._fit_quantile(X_train, y_train, alpha=0.90)
        self._fitted = True

        # Validation on holdout
        pred_p50 = self._p50.predict(X_test)
        corr = float(np.corrcoef(pred_p50, y_test)[0, 1])
        veto_prevented = int((pred_p50 < self._veto_threshold).sum())
        actual_neg_among_vetoed = float(
            (y_test[pred_p50 < self._veto_threshold] < 0).mean()
        ) if veto_prevented else 0.0

        feat_names = self._feature_names()
        importances = sorted(
            zip(feat_names, self._p50.feature_importances_),
            key=lambda x: x[1], reverse=True
        )

        self._meta = {
            'n_train':               len(X_train),
            'n_test':                len(X_test),
            'holdout_p50_corr':      round(corr, 4),
            'veto_rate_on_holdout':  round(veto_prevented / len(X_test), 4),
            'veto_accuracy':         round(actual_neg_among_vetoed, 4),
            'top_features':          [(f, round(float(i), 4)) for f, i in importances[:8]],
            'p50_train_mean':        round(float(self._p50.predict(X_train).mean()), 4),
            'actual_train_mean':     round(float(y_train.mean()), 4),
        }
        # Calibrate AFTER meta is built so calibrated_thresholds merges into it
        self._calibrate_thresholds(X_train)

        self._save()
        self._print_summary()
        return self._meta

    def predict(self, conditions: dict) -> TrajectoryPrediction:
        if not self._fitted:
            self.train()
        x    = self._conditions_to_array(conditions)
        p10  = round(float(self._p10.predict(x)[0]), 3)
        p50  = round(float(self._p50.predict(x)[0]), 3)
        p90  = round(float(self._p90.predict(x)[0]), 3)

        # Percentile rank: where does this p50 sit in the training distribution?
        pct_rank = float(np.searchsorted(
            np.sort(self._training_p50_dist), p50
        ) / max(len(self._training_p50_dist), 1) * 100) if hasattr(self, '_training_p50_dist') else 50.0

        return TrajectoryPrediction(
            p10=p10, p50=p50, p90=p90,
            percentile_rank=round(pct_rank, 1),
            veto_threshold=self._veto_threshold,
            half_size_threshold=self._half_size_threshold,
            max_size_threshold=self._max_size_threshold,
        )

    def predict_from_record(self, record) -> TrajectoryPrediction:
        """Accept a SovereignFeatureRecord or dict-like from the live system."""
        conditions = {
            'regime':          getattr(record, 'regime_label', 'FLAT') if hasattr(record, 'regime_label') else record.get('regime', 'FLAT'),
            'hurst':           getattr(record, 'hurst', 0.5) if hasattr(record, 'hurst') else record.get('hurst_at_entry', 0.5),
            'atr_pct':         getattr(record, 'atr_pct', 1.5) if hasattr(record, 'atr_pct') else record.get('atr_pct', 1.5),
            'adx':             getattr(record, 'adx', 25.0) if hasattr(record, 'adx') else record.get('adx_at_entry', 25.0),
            'spy_5d_return':   record.get('spy_5d_return', 0.0) if isinstance(record, dict) else 0.0,
            'vix':             record.get('vix', 18.0) if isinstance(record, dict) else 18.0,
            'strategy':        record.get('strategy', 'momentum_sma') if isinstance(record, dict) else 'momentum_sma',
            'direction':       record.get('direction', 'LONG') if isinstance(record, dict) else 'LONG',
            'n_signals':       record.get('n_signals_aligned', 2) if isinstance(record, dict) else 2,
        }
        return self.predict(conditions)

    # ── Threshold calibration ─────────────────────────────────────────── #

    def _calibrate_thresholds(self, X_train: np.ndarray) -> None:
        """
        Set veto/half-size/max-size thresholds from the actual distribution
        of p50 predictions on the training set.

        Veto     = bottom VETO_PERCENTILE of predictions
        HalfSize = between HALF_SIZE_PERCENTILE and VETO_PERCENTILE
        MaxSize  = top (100 - MAX_SIZE_PERCENTILE) of predictions
        """
        p50_preds = self._p50.predict(X_train)
        self._training_p50_dist = np.sort(p50_preds)

        self._veto_threshold      = float(np.percentile(p50_preds, VETO_PERCENTILE))
        self._half_size_threshold = float(np.percentile(p50_preds, HALF_SIZE_PERCENTILE))
        self._max_size_threshold  = float(np.percentile(p50_preds, MAX_SIZE_PERCENTILE))

        self._meta['calibrated_thresholds'] = {
            'veto_p50_below':      round(self._veto_threshold, 4),
            'half_size_p50_below': round(self._half_size_threshold, 4),
            'max_size_p50_above':  round(self._max_size_threshold, 4),
            'p50_dist_median':     round(float(np.median(p50_preds)), 4),
            'p50_dist_mean':       round(float(np.mean(p50_preds)), 4),
        }

    # ── Feature engineering ───────────────────────────────────────────── #

    def _build_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df = df.copy()
        df['vix_level'] = df['vix_level'].fillna(df['vix_level'].median())

        df['regime_enc']   = df['regime'].map(REGIME_ENCODING).fillna(0).astype(int)
        df['strategy_enc'] = df['strategy'].map(STRATEGY_ENCODING).fillna(0).astype(int)
        df['direction_enc'] = (df['direction'] == 'LONG').astype(int)

        # Clamp extreme R-multiples — beyond ±20 are data artifacts
        y = df['exit_r'].clip(-20, 10).values

        X = df[self._feature_names()].values
        return X, y

    @staticmethod
    def _feature_names() -> list[str]:
        # STRICT entry conditions only — no future information.
        # mfe_pct and mae_pct are OUTCOMES and must not appear here.
        return [
            'hurst_at_entry',   # is this market trending or mean-reverting?
            'atr_pct',          # volatility at entry
            'adx_at_entry',     # trend strength
            'spy_5d_return',    # recent market momentum
            'vix_level',        # fear/risk sentiment
            'regime_enc',       # FLAT / MOMENTUM / REVERSION
            'strategy_enc',     # which strategy triggered
            'direction_enc',    # LONG=1 / SHORT=0
            'day_of_week',      # Tuesday/Thursday are best trend days
            'month',            # seasonal awareness
        ]

    def _conditions_to_array(self, c: dict) -> np.ndarray:
        regime_enc    = REGIME_ENCODING.get(str(c.get('regime', 'FLAT')), 0)
        strategy_enc  = STRATEGY_ENCODING.get(str(c.get('strategy', 'momentum_sma')), 0)
        direction_enc = 1 if str(c.get('direction', 'LONG')) == 'LONG' else 0

        import datetime
        now = datetime.datetime.now()

        x = np.array([[
            float(c.get('hurst',         0.50)),
            float(c.get('atr_pct',       1.50)),
            float(c.get('adx',           25.0)),
            float(c.get('spy_5d_return',  0.0)),
            float(c.get('vix',           18.0)),
            float(regime_enc),
            float(strategy_enc),
            float(direction_enc),
            float(now.weekday()),
            float(now.month),
        ]])
        return x

    @staticmethod
    def _fit_quantile(X: np.ndarray, y: np.ndarray, alpha: float) -> GradientBoostingRegressor:
        model = GradientBoostingRegressor(
            loss='quantile',
            alpha=alpha,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=50,
            random_state=42,
        )
        model.fit(X, y)
        return model

    # ── Persistence ───────────────────────────────────────────────────── #

    def _save(self) -> None:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'p10': self._p10, 'p50': self._p50, 'p90': self._p90,
                'veto_threshold':       self._veto_threshold,
                'half_size_threshold':  self._half_size_threshold,
                'max_size_threshold':   self._max_size_threshold,
                'training_p50_dist':    self._training_p50_dist,
            }, f)
        with open(META_PATH, 'w') as f:
            json.dump(self._meta, f, indent=2)

    def _load(self) -> None:
        with open(MODEL_PATH, 'rb') as f:
            d = pickle.load(f)
        self._p10, self._p50, self._p90 = d['p10'], d['p50'], d['p90']
        self._veto_threshold      = d.get('veto_threshold',      -3.0)
        self._half_size_threshold = d.get('half_size_threshold', -4.0)
        self._max_size_threshold  = d.get('max_size_threshold',  -1.0)
        self._training_p50_dist   = d.get('training_p50_dist',   np.array([]))
        if META_PATH.exists():
            with open(META_PATH) as f:
                self._meta = json.load(f)
        self._fitted = True

    def _print_summary(self) -> None:
        m   = self._meta
        cal = m.get('calibrated_thresholds', {})
        print(f"\n{'═'*60}")
        print(f"  TRAJECTORY MODEL — trained on {m['n_train']:,} trades")
        print(f"{'─'*60}")
        print(f"  Holdout p50 correlation:  {m['holdout_p50_corr']:+.4f}")
        print(f"  Veto rate on holdout:     {m['veto_rate_on_holdout']:.1%}")
        print(f"  Veto accuracy (neg rate): {m['veto_accuracy']:.1%}")
        print(f"\n  CALIBRATED THRESHOLDS (percentile-based):")
        print(f"  VETO  if p50 < {cal.get('veto_p50_below', '?'):>7}  (bottom {VETO_PERCENTILE}th pct)")
        print(f"  HALF  if p50 < {cal.get('half_size_p50_below', '?'):>7}  (bottom {HALF_SIZE_PERCENTILE}th pct)")
        print(f"  MAX   if p50 > {cal.get('max_size_p50_above', '?'):>7}  (top {100-MAX_SIZE_PERCENTILE}th pct)")
        print(f"\n  TOP FEATURE IMPORTANCES (p50 model):")
        for name, imp in m['top_features']:
            bar = '█' * int(imp * 80)
            print(f"  {name:22s} {bar:<30s} {imp:.4f}")
        print(f"{'═'*60}\n")
