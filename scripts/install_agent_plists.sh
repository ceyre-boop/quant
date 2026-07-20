#!/bin/bash
# install_agent_plists.sh — one-shot install of the three autonomous Claude agent plists.
# Run once from ~/quant on the Mac (not from the Cowork bash sandbox).
#
# What this does:
#   1. Copies morning/eod/research agent plists to ~/Library/LaunchAgents/
#   2. launchctl load each one
#   3. Runs plist_watchdog.py --rebaseline (GREEN status)
#   4. Commits and pushes all the new files to sovereign-v2
#
# After this runs you never need to open Claude Code manually again.
# Logs land in ~/quant/logs/{morning_agent,eod_agent,research_agent}.log

set -euo pipefail
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$SCRIPTS")"
AGENTS=~/Library/LaunchAgents

echo "=== Alta autonomous agent plist installer ==="
echo ""

# --- Step 1: copy and load plists ---
for name in com.alta.morning_agent com.alta.eod_agent com.alta.research_agent; do
    src="$SCRIPTS/${name}.plist"
    dst="$AGENTS/${name}.plist"

    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found"; exit 1
    fi

    echo "  Loading $name..."
    cp "$src" "$dst"
    launchctl unload "$dst" 2>/dev/null || true
    launchctl load "$dst"
done

echo ""
echo "  All three agents loaded."

# --- Step 2: watchdog rebaseline ---
echo ""
echo "  Running plist_watchdog.py --rebaseline..."
cd "$REPO"
python3 scripts/plist_watchdog.py --rebaseline "loaded morning_agent + eod_agent + research_agent 2026-07-19"

# --- Step 3: git commit and push ---
echo ""
echo "  Committing new files..."
# Clear any stale lock (safe if no other git process is running)
rm -f "$REPO/.git/index.lock" 2>/dev/null || true

git -C "$REPO" add \
    AGENT_DIRECTIVE.md \
    scripts/com.alta.morning_agent.plist \
    scripts/com.alta.eod_agent.plist \
    scripts/com.alta.research_agent.plist \
    scripts/install_agent_plists.sh \
    research/weekly_pattern_update.md \
    SYSTEM_STATUS.md \
    NEXT.md

git -C "$REPO" commit -m "[AGENT] Add AGENT_DIRECTIVE.md + 3 autonomous launchd agent plists

Autonomous operation layer: Claude Code agents covering morning (07:55 ET),
EOD (16:00 ET), and nightly research (21:00 ET). AGENT_DIRECTIVE.md is the
standing order for every autonomous session — covers the four daily routines,
eight non-negotiable standing rules (frozen-hash guard, holdout protection,
commit-on-every-pass). install_agent_plists.sh installed and rebaselined."

git -C "$REPO" push origin sovereign-v2

echo ""
echo "=== Done. Pushed to sovereign-v2. ==="
echo ""
echo "Plists loaded (fires automatically from now on):"
echo "  com.alta.morning_agent    07:55 ET Mon-Fri"
echo "  com.alta.eod_agent        16:00 ET Mon-Fri"
echo "  com.alta.research_agent   21:00 ET Sun-Thu"
echo ""
echo "Logs: ~/quant/logs/{morning_agent,eod_agent,research_agent}.log"
echo ""
echo "Now run the test loop to verify the morning routine:"
echo ""
echo "  claude --print \"Read ~/quant/AGENT_DIRECTIVE.md and execute the 08:00 morning routine\" \\"
echo "    --allowedTools \"Bash,Read,Write,Edit,Glob,Grep\" \\"
echo "    2>&1 | tee ~/quant/logs/autonomous_test_2026-07-19.log"
