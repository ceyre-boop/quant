#!/bin/bash
# install_heartbeat_plists.sh — install or reload all heartbeat LaunchAgent plists.
# Run on the host Mac (not in the Cowork sandbox).
# Usage: bash ~/quant/scripts/install_heartbeat_plists.sh

set -euo pipefail

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
AGENTS=~/Library/LaunchAgents
mkdir -p "$AGENTS"
mkdir -p ~/quant/logs ~/quant/data/health

# Plists to install: (script_name  label)
PLISTS=(
    "com.clawd.ny_am_scanner         com.clawd.ny_am_scanner"
    "com.sovereign.reddit_sentiment  com.sovereign.reddit_sentiment"
    "com.alta.oracle.session_close   com.alta.oracle.session_close"
    "com.alta.alexandria_health      com.alta.alexandria_health"
    "com.alta.system_health          com.alta.system_health"
)

chmod +x "$SCRIPTS/launch_ny_scanner.sh" 2>/dev/null || true
chmod +x "$SCRIPTS/launch_reddit_scraper.sh" 2>/dev/null || true
chmod +x "$SCRIPTS/check_alexandria_health.py" 2>/dev/null || true
chmod +x "$SCRIPTS/health_check.py" 2>/dev/null || true

echo "=== Alta/Sovereign heartbeat plist installer ==="
echo ""

for entry in "${PLISTS[@]}"; do
    FILE=$(echo "$entry" | awk '{print $1}')
    LABEL=$(echo "$entry" | awk '{print $2}')
    SRC="$SCRIPTS/${FILE}.plist"
    DST="$AGENTS/${FILE}.plist"

    if [ ! -f "$SRC" ]; then
        echo "ERROR: $SRC not found — skipping $LABEL"; continue
    fi

    # Unload if loaded (ignore errors — may not be loaded yet)
    launchctl unload "$DST" 2>/dev/null && echo "  unloaded $LABEL" || true

    cp "$SRC" "$DST"
    launchctl load "$DST"
    echo "  ✓ loaded  $LABEL"
done

echo ""
echo "All heartbeat agents loaded."
echo "Verify with: launchctl list | grep -E 'clawd|sovereign|alta'"
