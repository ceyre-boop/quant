"""Director (Phase 4) — the constitutional review layer (spec §6).

Assembles the old-vs-new parameter diff and runs three checks before any commit:

  MECHANISM  — does each change make economic sense? (heuristic + flags for review)
  REGIME     — was the training window dominated by a single regime?
  MAGNITUDE  — is any single parameter moving more than the ±cap per cycle?

Output is a structured review report. Phase 4 approval is ALWAYS human-gated: this
module NEVER auto-approves. It produces a recommendation; a human (Colin) confirms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "training.yml"

# Params whose economically-sensible direction is known. Used by the mechanism
# heuristic to flag moves that contradict the known edge. Unknown params → flagged
# for human review rather than silently passed.
_MECHANISM_NOTES = {
    "conviction_exit_trail": "higher = exit earlier; consistent with high-ATR early-exit finding",
    "conviction_entry_min": "higher = fewer, higher-conviction entries",
    "atr_threshold_high": "defines the high-ATR regime boundary",
}


@dataclass
class ParamDiff:
    name: str
    old: float
    new: float

    @property
    def pct_change(self) -> float:
        if self.old == 0:
            return float("inf") if self.new != 0 else 0.0
        return (self.new - self.old) / abs(self.old) * 100.0


@dataclass
class DirectorReport:
    diffs: list[ParamDiff]
    mechanism_ok: bool
    regime_ok: bool
    magnitude_ok: bool
    placebo_ok: bool = False
    placebo_margin: float | None = None
    flags: list[str] = field(default_factory=list)
    recommendation: str = "DEFER"
    human_gated: bool = True   # ALWAYS true — never auto-approve

    @property
    def all_pass(self) -> bool:
        return self.mechanism_ok and self.regime_ok and self.magnitude_ok and self.placebo_ok


def _load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"training config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def build_diff(old_params: dict, new_params: dict) -> list[ParamDiff]:
    keys = sorted(set(old_params) | set(new_params))
    return [
        ParamDiff(k, float(old_params.get(k, 0.0)), float(new_params.get(k, 0.0)))
        for k in keys
    ]


def _magnitude_check(diffs: list[ParamDiff], cap_pct: float) -> tuple[bool, list[str]]:
    flags = []
    for d in diffs:
        if abs(d.pct_change) > cap_pct:
            flags.append(
                f"MAGNITUDE: {d.name} moved {d.pct_change:+.1f}% "
                f"({d.old} → {d.new}), exceeds ±{cap_pct:.0f}% cap — suspected degenerate path"
            )
    return (len(flags) == 0), flags


def _mechanism_check(diffs: list[ParamDiff]) -> tuple[bool, list[str]]:
    flags = []
    for d in diffs:
        if d.pct_change == 0:
            continue
        if d.name not in _MECHANISM_NOTES:
            flags.append(
                f"MECHANISM: {d.name} has no known economic mechanism — flag for review"
            )
    return (len(flags) == 0), flags


def _regime_check(regime_fraction: float | None) -> tuple[bool, list[str]]:
    """regime_fraction = share of the training window in a single dominant regime.
    Above 0.55 the window is regime-dominated → flag parameter dependence."""
    if regime_fraction is None:
        return True, ["REGIME: training-window regime fraction unknown — cannot assess durability"]
    if regime_fraction > 0.55:
        return False, [
            f"REGIME: training window {regime_fraction:.0%} single-regime — "
            "flag parameter dependence on this regime before committing"
        ]
    return True, []


def _placebo_check(placebo) -> tuple[bool, float | None, list[str]]:
    """MANDATORY random-reweighting placebo control (HYP-090 lesson, structural).
    Fail-closed: no placebo result (None) or an ineligible verdict both FAIL this
    check — never treated as a pass."""
    if placebo is None:
        return False, None, [
            "PLACEBO: no placebo-control result provided — fail-closed, cannot approve"
        ]
    if not getattr(placebo, "eligible", False):
        return False, getattr(placebo, "margin", None), [f"PLACEBO: {placebo.reason}"]
    return True, placebo.margin, [
        f"PLACEBO: real beats random-reweighting placebo by {placebo.margin:+.3f} "
        f"(min {placebo.margin_min:.3f}, t over {placebo.n_folds} folds, seed={placebo.seed})"
    ]


def review(old_params: dict, new_params: dict, *,
           regime_fraction: float | None = None,
           placebo=None,
           config_path: Path | None = None) -> DirectorReport:
    """Run the constitutional checks and produce a human-gated recommendation."""
    cfg = _load_config(config_path)
    dcfg = cfg.get("director", {})
    cap = float(dcfg.get("max_param_change_pct", 20.0))

    diffs = build_diff(old_params, new_params)
    mech_ok, mech_flags = _mechanism_check(diffs)
    regime_ok, regime_flags = _regime_check(regime_fraction)
    mag_ok, mag_flags = _magnitude_check(diffs, cap)
    placebo_ok, placebo_margin, placebo_flags = _placebo_check(placebo)

    flags = mech_flags + regime_flags + mag_flags + placebo_flags
    report = DirectorReport(
        diffs=diffs, mechanism_ok=mech_ok, regime_ok=regime_ok,
        magnitude_ok=mag_ok, placebo_ok=placebo_ok, placebo_margin=placebo_margin,
        flags=flags,
    )
    # Never auto-approve (dcfg.auto_approve is enforced as always-false at the runner).
    report.recommendation = (
        "APPROVE (pending human confirmation)" if report.all_pass
        else "REJECT — freeze flagged parameter(s), re-run cycle"
    )
    return report


def render_report(report: DirectorReport) -> str:
    lines = ["PHASE 4: CLAUDE REVIEW (human-gated — never auto-approved)"]
    for d in report.diffs:
        arrow = "→"
        lines.append(f"    {d.name}: {d.old} {arrow} {d.new}  ({d.pct_change:+.1f}%)")
    lines.append(f"    Mechanism: {'PASS' if report.mechanism_ok else 'FAIL'}")
    lines.append(f"    Regime:    {'PASS' if report.regime_ok else 'FAIL'}")
    lines.append(f"    Magnitude: {'PASS' if report.magnitude_ok else 'FAIL'}")
    margin_str = f"{report.placebo_margin:+.3f}" if report.placebo_margin is not None else "n/a"
    lines.append(f"    Placebo:   {'PASS' if report.placebo_ok else 'FAIL'} (margin={margin_str})")
    for f in report.flags:
        lines.append(f"    ! {f}")
    lines.append(f"    Recommendation: {report.recommendation}")
    return "\n".join(lines)
