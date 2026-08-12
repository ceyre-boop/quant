"""P3: the variant universe + A1/A2/A3 selection engines (HYP-090).

VariantUniverse: 5,775 variants = 385 configs x 15 non-empty pair subsets;
variant daily return = equal-notional mean over subset pairs of the costed M2M
series. Trailing-window Sharpe for every (variant, day, window) via prefix sums
— the daily adaptive loop never re-backtests.

Causality: window scores at day t use returns <= t; A2's map rows are days
s <= t-1; selections apply from t+1 (replay.py). Truncation-safe by
construction — verified by tests/test_lookahead.py on rebuilt truncated inputs.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from research.modern._lib import OUT_DIR, TRADING_DAYS, seed_from, to_dtindex

N_CONFIGS = 385
PAIR_SUBSETS = [s for r in range(1, 5) for s in itertools.combinations(range(4), r)]
assert len(PAIR_SUBSETS) == 15
WINDOWS = [90, 180, 365]           # calendar days
V015_VARIANT = None                 # resolved in VariantUniverse (config 384, all pairs)
KNN_K = 25
MIN_WINDOW_OBS = 20


class VariantUniverse:
    def __init__(self, npz_path: Path = None):
        z = np.load(npz_path or (OUT_DIR / "daily_returns.npz"))
        self.costed = z["costed"]                    # (385, 4, D) float64
        self.position = z["position"]                # (385, 4, D) int8
        self.close_by_pair = z["close_by_pair"]      # (4, D)
        self.index = to_dtindex(z["union_index"])
        self.D = self.costed.shape[2]

        # variant table: (config_id, subset_tuple)
        self.variants = [(cid, sub) for cid in range(N_CONFIGS) for sub in PAIR_SUBSETS]
        self.n_variants = len(self.variants)         # 5775
        global V015_VARIANT
        V015_VARIANT = self.variants.index((384, (0, 1, 2, 3)))
        self.v015_variant = V015_VARIANT

        # V: (n_variants, D) variant daily returns
        subs_mask = np.zeros((15, 4), dtype=float)
        for si, sub in enumerate(PAIR_SUBSETS):
            subs_mask[si, list(sub)] = 1.0 / len(sub)
        # einsum: for each config, subset: mean over subset pairs
        V = np.einsum("sp,cpd->csd", subs_mask, self.costed)   # (385, 15, D)... wait order
        self.V = V.reshape(self.n_variants, self.D)

        # prefix sums for rolling window stats
        self._cum1 = np.cumsum(self.V, axis=1)
        self._cum2 = np.cumsum(self.V ** 2, axis=1)
        # window start index per (t, W): first day index with date > date[t] - W days
        self._starts = {}
        dates = self.index.values
        for W in WINDOWS:
            cutoff = dates - np.timedelta64(W, "D")
            self._starts[W] = np.searchsorted(dates, cutoff, side="right")

    def window_scores(self, W: int) -> np.ndarray:
        """(n_variants, D) trailing-W annualized Sharpe ending at each day t (inclusive).
        Degenerate windows (n < MIN_WINDOW_OBS or zero variance) score 0."""
        starts = self._starts[W]
        t_idx = np.arange(self.D)
        n = (t_idx - starts + 1).astype(float)                       # obs per window
        c1, c2 = self._cum1, self._cum2
        s1 = c1[:, t_idx] - np.where(starts > 0, c1[:, np.maximum(starts - 1, 0)], 0.0)
        s2 = c2[:, t_idx] - np.where(starts > 0, c2[:, np.maximum(starts - 1, 0)], 0.0)
        mean = s1 / n
        var = np.maximum(s2 / n - mean ** 2, 0.0) * (n / np.maximum(n - 1, 1.0))
        std = np.sqrt(var)
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe = np.where(std > 1e-12, mean / std * np.sqrt(TRADING_DAYS), 0.0)
        sharpe[:, n < MIN_WINDOW_OBS] = 0.0
        return sharpe


def a1_select(scores: np.ndarray, replay_days: np.ndarray) -> np.ndarray:
    """Recent-winner: argmax variant per day (ties -> lowest variant id = canonical
    manifest order). scores: (n_variants, D)."""
    return np.argmax(scores[:, replay_days], axis=0)


def a2_select(scores: np.ndarray, vectors: np.ndarray, replay_days: np.ndarray,
              v015_variant: int, k: int = KNN_K) -> np.ndarray:
    """Regime-matched k-NN: for day t, find k nearest PRIOR days (s <= t-1, valid
    vectors + valid scores), score variants by mean window-Sharpe across those
    days, argmax. Cold start -> v015 variant."""
    D = scores.shape[1]
    valid_row = ~np.isnan(vectors).any(axis=1)
    sel = np.full(len(replay_days), v015_variant, dtype=np.int64)
    for i, t in enumerate(replay_days):
        pool = np.where(valid_row[:t])[0]           # strictly s <= t-1
        if pool.size < k:
            continue
        dist = np.linalg.norm(vectors[pool] - vectors[t], axis=1)
        nearest = pool[np.argpartition(dist, k - 1)[:k]]
        sel[i] = int(np.argmax(scores[:, nearest].mean(axis=1)))
    return sel


def a3_select(n_variants: int, replay_days: np.ndarray, seed_parts) -> np.ndarray:
    """Placebo: uniform-random variant per day through identical machinery."""
    rng = seed_from("HYP-090-A3", *seed_parts)
    return rng.integers(0, n_variants, size=len(replay_days))
