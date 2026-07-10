"""Shared plumbing for research/prop_funnel (TICK-022).

Path anchors, evidence stamps, seeded RNG, canonical JSON (for the determinism
test), and environment recording. Everything this module writes stays under
data/research/prop_funnel/ — write-safety is a ticket acceptance criterion.
"""
from __future__ import annotations

import json
import platform
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = ROOT / "data" / "research" / "prop_funnel"
PARITY_DIR = OUT_DIR / "parity"
CHARTS_DIR = OUT_DIR / "charts"

# Keys stripped by canonical() so determinism compares only substance.
VOLATILE_KEYS = {"generated_at", "env", "runtime_seconds"}


class EvidenceStamp(str, Enum):
    """Every verdict row carries exactly one of these. The stamp is the honesty
    layer: PROVEN means the underlying edge cleared the repo's gauntlet;
    everything else is explicitly not that."""

    PROVEN_REGIME_FRAGILE = "PROVEN_REGIME_FRAGILE"   # carry: real, but regime-conditional
    UNPROVEN = "UNPROVEN"                             # ICT: permutation p=0.52
    LOW_N_SANITY_ONLY = "LOW_N_SANITY_ONLY"           # ict_live: n=27 closed outcomes
    UNVALIDATED = "UNVALIDATED"                       # futures ORB: n_real=0
    SYNTHETIC = "SYNTHETIC"                           # frontier grid: no claim of existence
    SCENARIO = "SCENARIO"                             # forward-Sharpe band: "if carry forward-Sharpe were X"


def get_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def env_record() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def canonical(obj: Any) -> Any:
    """Recursively drop volatile keys so two runs with the same seed compare equal."""
    if isinstance(obj, dict):
        return {k: canonical(v) for k, v in sorted(obj.items()) if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [canonical(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 12)
    return obj


def canonical_dumps(obj: Any) -> str:
    return json.dumps(canonical(obj), sort_keys=True)
