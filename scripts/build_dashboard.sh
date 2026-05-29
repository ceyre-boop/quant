#!/bin/bash
# Run after every Oracle cycle to keep GitHub Pages current with latest data.
# Usage: bash scripts/build_dashboard.sh
set -e
cd "$(dirname "$0")/.."

# Sync latest reflection to stable path
LATEST=$(ls data/oracle/reflections/[0-9]*.json 2>/dev/null | sort | tail -1)
if [ -n "$LATEST" ]; then
  cp "$LATEST" data/oracle/latest_reflection.json
  echo "Reflection synced: $LATEST → data/oracle/latest_reflection.json"
else
  echo "No dated reflection files found — skipping reflection sync"
fi

# Stage all dashboard data files
git add \
  data/agent/prop_challenge_state.json \
  data/agent/g2_progress.json \
  data/agent/hypothesis_ledger.json \
  data/agent/research_queue.json \
  data/agent/health.json \
  data/agent/checklist_state.json \
  data/oracle/latest_reflection.json \
  data/oracle/version_history.json \
  2>/dev/null || true

git diff --staged --quiet && echo "Nothing to commit — data is up to date" && exit 0
git commit -m "Dashboard data sync $(date -u +%Y-%m-%dT%H:%M)"
git push origin HEAD:master
echo "Done — GitHub Pages will update in ~60 seconds"
