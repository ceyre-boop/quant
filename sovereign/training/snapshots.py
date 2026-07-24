"""Cycle snapshot + undo/restore for the self-play training scaffold.

WHY THIS EXISTS: Colin wants a bad training cycle to be reversible from the
dashboard (undo_last_cycle / restore_last_cycle). This module is the storage
layer behind those two actions.

SCOPE: this module ONLY reads/writes files under data/training/ (checkpoints,
snapshots, and the "current policy params" pointer). It NEVER writes
config/training.yml, NEVER touches sovereign/training/gate.py, and NEVER makes
a legitimacy decision of its own — it only stores and replays policy-param
states that the pipeline (sovereign_train.py, gated by gate.py + director.py +
placebo_control.py) already produced. Undo/restore therefore cannot become a
path to activate a cycle that failed placebo or human approval: they only move
between states the gated pipeline itself wrote, and while the ignition gate is
CLOSED every one of those states is a DRY/no-op state by construction (params
before == params after).

Undo/redo semantics, standard two-stack model:
  - `applied`: stack of cycle records currently "in effect", oldest first.
  - `undone`: stack of cycle records popped off `applied` by undo_last_cycle,
    most-recently-undone last — restore_last_cycle pops this stack.
  - recording a NEW cycle clears `undone` (a fresh cycle invalidates any
    pending redo chain — the same rule text editors use).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAP_DIR = ROOT / "data" / "training" / "snapshots"
HISTORY_PATH = SNAP_DIR / "history.json"
CURRENT_PARAMS_PATH = ROOT / "data" / "training" / "current_policy_params.json"
DEFAULT_CONFIG = ROOT / "config" / "training.yml"


@dataclass
class CycleRecord:
    id: str
    timestamp: str
    cycle_ref: str
    committed: bool
    params_before: dict = field(default_factory=dict)
    params_after: dict = field(default_factory=dict)


def _load_history() -> dict:
    if not HISTORY_PATH.exists():
        return {"applied": [], "undone": []}
    try:
        data = json.loads(HISTORY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {"applied": [], "undone": []}
    data.setdefault("applied", [])
    data.setdefault("undone", [])
    return data


def _save_history(history: dict) -> None:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2) + "\n")


def _write_current_params(params: dict) -> None:
    CURRENT_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_PARAMS_PATH.write_text(json.dumps(params, indent=2) + "\n")


def get_current_params() -> dict:
    """The active policy-param state: current_policy_params.json if it exists,
    else the baseline `policy_params` block from config/training.yml."""
    if CURRENT_PARAMS_PATH.exists():
        try:
            return json.loads(CURRENT_PARAMS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    if DEFAULT_CONFIG.exists():
        import yaml

        cfg = yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}
        return dict(cfg.get("policy_params", {}))
    return {}


def record_cycle(
    params_before: dict,
    params_after: dict,
    cycle_ref: str,
    committed: bool,
    timestamp: str,
) -> CycleRecord:
    """Called by the training runner after a pipeline cycle completes (dry or
    committed). Stores the before/after policy-param state and makes
    `params_after` the new current state. Clears the redo (`undone`) stack."""
    history = _load_history()
    snap_id = f"{timestamp}_{len(history['applied']) + len(history['undone'])}"
    record = CycleRecord(
        id=snap_id,
        timestamp=timestamp,
        cycle_ref=cycle_ref,
        committed=committed,
        params_before=dict(params_before),
        params_after=dict(params_after),
    )

    snap_path = SNAP_DIR / snap_id
    snap_path.mkdir(parents=True, exist_ok=True)
    (snap_path / "record.json").write_text(json.dumps(asdict(record), indent=2) + "\n")

    history["applied"].append(asdict(record))
    history["undone"] = []  # new cycle invalidates any pending redo
    _save_history(history)
    _write_current_params(params_after)
    return record


def undo_last_cycle() -> dict:
    """Restore the policy-param state to what it was before the most recent
    (not-yet-undone) cycle. Returns {"ok": bool, "message": str, "params": dict|None}."""
    history = _load_history()
    if not history["applied"]:
        return {"ok": False, "message": "no cycle to undo", "params": None}
    record = history["applied"].pop()
    history["undone"].append(record)
    _save_history(history)
    _write_current_params(record["params_before"])
    return {
        "ok": True,
        "message": f"undone cycle {record['id']} (cycle_ref={record['cycle_ref']})",
        "params": record["params_before"],
    }


def restore_last_cycle() -> dict:
    """Re-apply the most recently undone cycle. Returns {"ok": bool, "message": str, "params": dict|None}."""
    history = _load_history()
    if not history["undone"]:
        return {"ok": False, "message": "no cycle to restore", "params": None}
    record = history["undone"].pop()
    history["applied"].append(record)
    _save_history(history)
    _write_current_params(record["params_after"])
    return {
        "ok": True,
        "message": f"restored cycle {record['id']} (cycle_ref={record['cycle_ref']})",
        "params": record["params_after"],
    }


def status() -> dict:
    history = _load_history()
    return {
        "applied_count": len(history["applied"]),
        "undone_count": len(history["undone"]),
        "last_applied": history["applied"][-1] if history["applied"] else None,
        "last_undone": history["undone"][-1] if history["undone"] else None,
        "current_params": get_current_params(),
    }
