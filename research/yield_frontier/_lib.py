"""Yield Frontier shared lib — stamp, mined-N counter, path constants.

Plan: Plans/immutable-wondering-alpaca.md (TICK-030/031).
Every mining artifact carries STAMP; every evaluated configuration increments
the append-only mined-N counter, which is the gauntlet's honest DSR n_trials.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.modern._lib import (  # noqa: E402,F401  (re-exports)
    seed_from, canonical_hash, sha256_file, daily_sharpe,
    block_bootstrap_sharpe_diff_p,
)

OUT = REPO / "data/research/yield_frontier"
STAMP = "MINING — look-back, descriptive, uncorrected; NOT evidence; candidate-generation only"
MINED_N_PATH = OUT / "mined_n.json"


def record_mined(universe: str, family: str, n_cells: int) -> None:
    """Append-only trial counter. Never decrements; repeated runs of the same
    family overwrite that family's entry (same cells re-run, not new trials)."""
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(MINED_N_PATH.read_text()) if MINED_N_PATH.exists() else {}
    key = f"{universe}:{family}"
    prev = d.get(key, 0)
    if n_cells < prev:
        raise ValueError(f"mined-N would shrink for {key}: {prev} -> {n_cells}")
    d[key] = n_cells
    d["_total"] = sum(v for k, v in d.items() if not k.startswith("_"))
    MINED_N_PATH.write_text(json.dumps(d, indent=2, sort_keys=True))


def mined_total() -> int:
    if not MINED_N_PATH.exists():
        return 0
    return json.loads(MINED_N_PATH.read_text()).get("_total", 0)
