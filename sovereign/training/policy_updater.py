"""Policy update (Phase 3) — refit the XGBoost policy on top-quartile trades.

This is NOT a new corpus: it is a reweighting of the existing one (spec §4.3). High
value-function trades get weight `top_weight`, bottom-quartile get `bottom_weight`.
The purged walk-forward splits and feature set are unchanged — only sample weights
change. The refit is delegated to sovereign/ml_trainer.py via its sample_weight
passthrough.

GATE-BOUND: when the ignition gate is CLOSED, compute_sample_weights still runs (it
is pure arithmetic, useful for the director diff), but refit_policy REFUSES to train
or write any production policy. It returns a dry report only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "training.yml"


@dataclass
class UpdateResult:
    dry: bool
    n_top: int
    n_bottom: int
    threshold: float
    note: str


def _load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"training config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def compute_sample_weights(value_scores, config_path: Path | None = None) -> np.ndarray:
    """Map value-function scores to XGBoost sample weights (spec §4.3).

    Top-quartile (score > pct percentile) → top_weight; the rest → bottom_weight.
    Pure arithmetic; safe to run regardless of gate state."""
    cfg = _load_config(config_path)
    r = cfg.get("reward", {})
    scores = np.asarray(value_scores, dtype=float)
    if scores.size == 0:
        return scores
    threshold = np.percentile(scores, float(r.get("top_quartile_pct", 75)))
    return np.where(
        scores > threshold,
        float(r.get("top_weight", 2.0)),
        float(r.get("bottom_weight", 0.5)),
    )


def refit_policy(value_scores, *, gate_open: bool,
                 config_path: Path | None = None) -> UpdateResult:
    """Refit the policy on reweighted trades — or refuse in DRY mode.

    gate_open=False → compute weights, return a dry report, train NOTHING, write
        NOTHING. gate_open=True → delegate the reweighted purged-walk-forward refit
        to ml_trainer (not wired until ignition; raises loudly if reached)."""
    cfg = _load_config(config_path)
    r = cfg.get("reward", {})
    weights = compute_sample_weights(value_scores, config_path)
    scores = np.asarray(value_scores, dtype=float)
    threshold = (
        float(np.percentile(scores, float(r.get("top_quartile_pct", 75))))
        if scores.size else 0.0
    )
    n_top = int((weights == float(r.get("top_weight", 2.0))).sum())
    n_bottom = int((weights == float(r.get("bottom_weight", 0.5))).sum())

    if not gate_open:
        return UpdateResult(
            dry=True, n_top=n_top, n_bottom=n_bottom, threshold=threshold,
            note="DRY: sample weights computed; NO refit, NO production policy written.",
        )

    raise NotImplementedError(
        "LIVE policy refit is not wired yet. It must call ml_trainer.train(..., "
        "sample_weight=weights) preserving the purged walk-forward split, once the "
        "ignition gate opens. Until then the loop runs DRY. See spec §4.3, §9."
    )
