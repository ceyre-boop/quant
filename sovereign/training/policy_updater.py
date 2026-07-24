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
    placebo: "object | None" = None
    eligible: bool = False


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


def refit_policy(value_scores, returns=None, *, gate_open: bool,
                 config_path: Path | None = None) -> UpdateResult:
    """Refit the policy on reweighted trades — or refuse in DRY mode.

    MANDATORY placebo control (HYP-090 lesson, structural — spec addendum): before
    any refit is even considered eligible, the value-informed weighting must beat a
    random-reweighting placebo (same weight composition, shuffled assignment) on a
    purged walk-forward OOS split, by at least config `placebo.placebo_margin_min`.
    Fail-closed: missing/malformed placebo inputs (e.g. no `returns` passed) produce
    an ineligible verdict, never a pass. This runs regardless of the ignition gate,
    so the mechanism is exercised in DRY mode too.

    gate_open=False → compute weights + placebo verdict, return a dry report, train
        NOTHING, write NOTHING. gate_open=True → refuse (REJECTED, no commit) if the
        placebo verdict is ineligible; otherwise delegate the reweighted purged-
        walk-forward refit to ml_trainer (not wired until ignition; raises loudly if
        reached)."""
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

    from sovereign.training import placebo_control  # local import: avoid import cycle

    try:
        if returns is None:
            raise placebo_control.PlaceboDataError("no `returns` provided to refit_policy")
        verdict = placebo_control.run_control(returns, value_scores, config_path)
    except placebo_control.PlaceboDataError as exc:
        verdict = placebo_control.PlaceboVerdict(
            eligible=False, real_metric=0.0, placebo_metric=0.0, margin=0.0,
            margin_min=float(cfg.get("placebo", {}).get("placebo_margin_min", 0.15)),
            significant=False, composition_ok=False, n_folds=0,
            seed=int(cfg.get("placebo", {}).get("seed", 42)),
            reason=f"FAIL-CLOSED: placebo control could not run — {exc}",
        )

    if not gate_open:
        return UpdateResult(
            dry=True, n_top=n_top, n_bottom=n_bottom, threshold=threshold,
            note="DRY: sample weights computed; NO refit, NO production policy written.",
            placebo=verdict, eligible=False,
        )

    if not verdict.eligible:
        return UpdateResult(
            dry=True, n_top=n_top, n_bottom=n_bottom, threshold=threshold,
            note=f"REJECTED — {verdict.reason}",
            placebo=verdict, eligible=False,
        )

    raise NotImplementedError(
        "LIVE policy refit is not wired yet. It must call ml_trainer.train(..., "
        "sample_weight=weights) preserving the purged walk-forward split, once the "
        "ignition gate opens. Until then the loop runs DRY. See spec §4.3, §9."
    )
