#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DIP Daily Orchestrator — chains the EXISTING revived components:
#   1. continuous_harvester --passes 1   (parent-prefetch → fork-pool compute → harvest.db)
#   2. retrain_loop --once               (XGBoost retrain + threshold optimization)
#
# This is a RESEARCH loop only. It never calls order_send or any MT5/OANDA bridge,
# never touches the frozen execution path (forex_exit_manager / decide_exit).
#
# No silent failures: each phase writes a checkpoint on success and an error file
# on failure, and the script aborts on the first failing phase (set -e).
#
# Usage: ./scripts/dip_daily.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/opt/homebrew/bin/python3}"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
mkdir -p logs data

_fail() { echo "{\"phase\":\"$1\",\"ts\":\"$STAMP\",\"error\":\"$2\"}" > "data/_dip_daily_error.json"; exit 1; }

echo "[DIP] === daily orchestrator start $STAMP ==="

echo "[DIP] phase 1/2 — harvest (single pass, full universe)"
$PYTHON scripts/continuous_harvester.py --passes 1 || _fail harvest "harvester exited non-zero"

echo "[DIP] phase 2/2 — retrain (XGBoost, single cycle)"
$PYTHON training/retrain_loop.py --once || _fail retrain "retrain exited non-zero"

echo "{\"ts\":\"$STAMP\",\"status\":\"ok\"}" > "data/_dip_daily_checkpoint.json"
rm -f "data/_dip_daily_error.json"
echo "[DIP] === daily orchestrator done $STAMP ==="
