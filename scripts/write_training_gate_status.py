#!/usr/bin/env python3
"""
write_training_gate_status.py — emit the self-play ignition gate status as JSON
for the dashboard.

WHY THIS EXISTS
---------------
sovereign/training/gate.py exposes evaluate_gate() (returns a GateStatus) and a
CLI that only prints a human banner — it writes no machine-readable file. The
dashboard needs the gate's OPEN/CLOSED state, mode, refusal reasons, and the
individual check results as data. This script imports the gate's PUBLIC API,
evaluates it (pure read — reads config + value board + hypothesis ledger, all
read-only), and writes data/agent/training_gate_status.json.

FREEZE-SAFE: this is a NEW, additive script. It does NOT modify gate.py or any
execution-path file. It only calls gate.py's public evaluate_gate() and writes
one new JSON file. No trading path, no parameter mutation.

Usage:  python3 scripts/write_training_gate_status.py
Output: data/agent/training_gate_status.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sovereign.training.gate import evaluate_gate  # noqa: E402

OUT = REPO / "data" / "agent" / "training_gate_status.json"


def main() -> None:
    status = evaluate_gate()  # pure read of gate.py's public API
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "open": status.open,
        "mode": status.mode,          # "LIVE" or "SCAFFOLD/DRY"
        "reasons": status.reasons,    # why the gate is CLOSED (empty if OPEN)
        "checks": status.checks,      # {check_name: bool}
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
