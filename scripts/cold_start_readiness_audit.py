#!/usr/bin/env python3
"""
Cold-start readiness audit for Sovereign ML stack.

Checks:
1) Ledger depth + feature completeness (trade_ledger_*.jsonl)
2) Startup warm/cold state from orchestrator snapshot
3) Checkpoint persistence under models/*.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Required for ledger bootstrap + three-vote diagnostics:
# - regime/hurst/hmm_transition_prob/adx/strategy feed Softmax/KMeans/PredictNow
# - pnl is required to derive trade outcome labels (win/loss) for online learners
REQUIRED_CLOSED_FEATURES = [
    "regime",
    "hurst",
    "hmm_transition_prob",
    "adx",
    "strategy",
    "pnl",
]

CHECKPOINTS = [
    "softmax_regime.pkl",
    "kmeans_regime.pkl",
    "ica_factor_separator.pkl",
    "predict_now.pkl",
    "trade_mdp.pkl",
    "pegasus_policy.pkl",
    "kalman_regime.pkl",
]


def audit_ledger(repo_root: Path) -> Dict[str, Any]:
    ledger_dir = repo_root / "data" / "ledger"
    files = sorted(ledger_dir.glob("trade_ledger_*.jsonl"))
    closed = 0
    opened = 0
    missing = Counter()

    for fp in files:
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = str(row.get("status", "")).lower()
            if status == "closed":
                closed += 1
                for key in REQUIRED_CLOSED_FEATURES:
                    if row.get(key) is None:
                        missing[key] += 1
            elif status == "open":
                opened += 1

    complete_closed = closed > 0 and all(missing.get(k, 0) == 0 for k in REQUIRED_CLOSED_FEATURES)
    return {
        "files": [str(p) for p in files],
        "file_count": len(files),
        "closed_count": closed,
        "open_count": opened,
        "required_closed_features": REQUIRED_CLOSED_FEATURES,
        "missing_required_features": dict(missing),
        "has_bootstrap_depth_50": closed >= 50,
        "complete_closed_features": complete_closed,
    }


def audit_checkpoints(repo_root: Path) -> Dict[str, Any]:
    models_dir = repo_root / "models"
    statuses = {}
    for name in CHECKPOINTS:
        path = models_dir / name
        statuses[name] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
    return statuses


def run_audit(repo_root: Path) -> Dict[str, Any]:
    ledger = audit_ledger(repo_root)
    snapshot = {}
    snapshot_error = None
    try:
        from sovereign.orchestrator import SovereignOrchestrator
        orch = SovereignOrchestrator(mode="paper")
        snapshot = orch.get_latest_ml_snapshot()
    except Exception as e:
        snapshot_error = str(e)

    checkpoints = audit_checkpoints(repo_root)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "ledger": ledger,
        "startup_ml_snapshot": snapshot,
        "startup_ml_snapshot_error": snapshot_error,
        "checkpoints": checkpoints,
        "ready_for_warm_start": bool(
            ledger["has_bootstrap_depth_50"] and ledger["complete_closed_features"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cold-start readiness audit")
    parser.add_argument(
        "--out",
        default="data/reports/cold_start_readiness_latest.json",
        help="Output JSON path (relative to repo root).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = run_audit(repo_root)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
