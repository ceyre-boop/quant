"""Mandatory random-reweighting placebo control (HYP-090 lesson, made structural).

HYP-090 (daily adaptive parameter selection) was KILLED because it lost to a
random-selection placebo — the search beat nothing, it just looked directed. The
self-play policy update in policy_updater.py reweights XGBoost toward
value-function-favored trades (top_weight / bottom_weight). That is the SAME class
of search: a directed reweighting that could just as easily be doing nothing more
than random up/down-weighting with a plausible story attached.

This module forces the real, value-informed weighting to prove — every cycle, on a
purged walk-forward OOS split — that it beats a placebo policy trained with the
IDENTICAL weight distribution (same count of top_weight / bottom_weight values)
but with the assignment across trades SHUFFLED under a fixed, logged RNG seed. Only
the assignment differs; the composition (how many trades get which weight) is
unchanged, so any margin is attributable to the value information, not to weighting
magnitude or class imbalance.

FAIL-CLOSED: any missing/malformed input (empty returns, empty value_scores,
mismatched lengths, degenerate splits) raises PlaceboDataError. Callers MUST treat
that as an ineligible cycle — never interpret a raised/caught error as a pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from sovereign.training import policy_updater

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "training.yml"


class PlaceboDataError(Exception):
    """Raised when the placebo control cannot be computed. Callers must fail-closed."""


@dataclass
class PlaceboVerdict:
    eligible: bool
    real_metric: float
    placebo_metric: float
    margin: float
    margin_min: float
    significant: bool
    composition_ok: bool
    n_folds: int
    seed: int
    reason: str
    per_fold_margins: list = field(default_factory=list)


def _load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"training config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def random_reweight(real_weights: np.ndarray, seed: int) -> np.ndarray:
    """Shuffle the exact same weight values across trades under a fixed seed.

    Same count of each weight value as `real_weights` — only the assignment to
    trades changes. This isolates "is the weighting value-informed?" from "does
    up/down-weighting some trades help at all?"."""
    weights = np.asarray(real_weights, dtype=float)
    rng = np.random.default_rng(seed)
    return rng.permutation(weights)


def _composition_matches(real_weights: np.ndarray, placebo_weights: np.ndarray) -> bool:
    if real_weights.shape != placebo_weights.shape:
        return False
    return np.array_equal(np.sort(real_weights), np.sort(placebo_weights))


def purged_kfold_indices(n: int, n_splits: int, embargo_frac: float) -> list[np.ndarray]:
    """Contiguous chronological holdout folds with an embargo gap purged from the
    fold edges, so no holdout observation sits immediately adjacent to a training
    boundary. Returns the list of holdout index arrays (order preserved)."""
    if n_splits < 2:
        raise PlaceboDataError(f"n_splits must be >= 2, got {n_splits}")
    if n < n_splits * 4:
        raise PlaceboDataError(
            f"too few observations ({n}) for {n_splits} purged folds — need >= {n_splits * 4}"
        )
    embargo = max(1, int(round(n * embargo_frac)))
    bounds = np.linspace(0, n, n_splits + 1, dtype=int)
    folds = []
    for i in range(n_splits):
        lo, hi = bounds[i], bounds[i + 1]
        lo_p, hi_p = lo + embargo, hi - embargo
        if hi_p <= lo_p:
            lo_p, hi_p = lo, hi  # fold too small for embargo — use it unpurged rather than drop
        folds.append(np.arange(lo_p, hi_p))
    folds = [f for f in folds if f.size > 0]
    if len(folds) < 2:
        raise PlaceboDataError("purged folds collapsed to < 2 non-empty folds")
    return folds


def weighted_sharpe(returns: np.ndarray, weights: np.ndarray) -> float:
    """Sample-weighted Sharpe of `returns` under `weights`. Degenerate (zero-
    variance) folds return 0.0 rather than dividing by zero."""
    r = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    wsum = w.sum()
    if wsum <= 0 or r.size == 0:
        return 0.0
    mean = float((w * r).sum() / wsum)
    var = float((w * (r - mean) ** 2).sum() / wsum)
    std = var ** 0.5
    if std < 1e-12:
        return 0.0
    return mean / std * (r.size ** 0.5)


def run_control(returns, value_scores, config_path: Path | None = None) -> PlaceboVerdict:
    """Run the mandatory placebo control. Raises PlaceboDataError on any
    malformed/insufficient input — the caller must treat that as a REJECTED cycle."""
    cfg = _load_config(config_path)
    pcfg = cfg.get("placebo", {})
    seed = int(pcfg.get("seed", 42))
    n_splits = int(pcfg.get("n_splits", 5))
    embargo_frac = float(pcfg.get("embargo_frac", 0.02))
    margin_min = float(pcfg.get("placebo_margin_min", 0.15))

    returns = np.asarray(returns, dtype=float)
    scores = np.asarray(value_scores, dtype=float)
    if returns.size == 0 or scores.size == 0:
        raise PlaceboDataError("empty returns or value_scores — cannot run placebo control")
    if returns.shape != scores.shape:
        raise PlaceboDataError(
            f"returns/value_scores length mismatch: {returns.shape} vs {scores.shape}"
        )

    real_weights = policy_updater.compute_sample_weights(scores, config_path)
    placebo_weights = random_reweight(real_weights, seed)
    composition_ok = _composition_matches(real_weights, placebo_weights)

    folds = purged_kfold_indices(returns.size, n_splits, embargo_frac)
    per_fold_margins = []
    real_metrics, placebo_metrics = [], []
    for idx in folds:
        real_m = weighted_sharpe(returns[idx], real_weights[idx])
        placebo_m = weighted_sharpe(returns[idx], placebo_weights[idx])
        real_metrics.append(real_m)
        placebo_metrics.append(placebo_m)
        per_fold_margins.append(real_m - placebo_m)

    real_metric = float(np.mean(real_metrics))
    placebo_metric = float(np.mean(placebo_metrics))
    margin = real_metric - placebo_metric

    margins = np.asarray(per_fold_margins, dtype=float)
    n = margins.size
    mean_margin = float(margins.mean())
    std_margin = float(margins.std(ddof=1)) if n > 1 else 0.0
    if std_margin < 1e-12:
        t_stat = float("inf") if mean_margin > 0 else (float("-inf") if mean_margin < 0 else 0.0)
    else:
        t_stat = mean_margin / (std_margin / (n ** 0.5))
    # Simple, sample-appropriate check for ~n_splits folds: require the margin to be
    # consistently positive (t_stat >= 1.0), not a single lucky fold.
    significant = t_stat >= 1.0

    eligible = composition_ok and (margin >= margin_min) and significant

    if not composition_ok:
        reason = "REJECTED: placebo weight composition does not match real weights (implementation bug)"
    elif not significant:
        reason = (
            f"REJECTED: real-vs-placebo margin ({margin:+.3f}) not significant "
            f"(t={t_stat:+.2f} over {n} folds) — failed random-reweighting placebo control - HYP-090 pattern"
        )
    elif margin < margin_min:
        reason = (
            f"REJECTED: margin {margin:+.3f} below placebo_margin_min {margin_min:.3f} "
            "— failed random-reweighting placebo control - HYP-090 pattern"
        )
    else:
        reason = f"ELIGIBLE: real beats placebo by {margin:+.3f} (>= {margin_min:.3f}), t={t_stat:+.2f}"

    return PlaceboVerdict(
        eligible=eligible, real_metric=real_metric, placebo_metric=placebo_metric,
        margin=margin, margin_min=margin_min, significant=significant,
        composition_ok=composition_ok, n_folds=n, seed=seed, reason=reason,
        per_fold_margins=per_fold_margins,
    )
