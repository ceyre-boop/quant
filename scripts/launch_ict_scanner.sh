#!/bin/bash
# ICT scanner auto-runner for NY PM and London sessions
# Triggered by launchd — runs every 5min
#
# Install:
#   chmod +x scripts/launch_ict_scanner.sh
#   cp scripts/com.clawd.ict_scanner.plist ~/Library/LaunchAgents/
#   launchctl load ~/Library/LaunchAgents/com.clawd.ict_scanner.plist

cd "$(dirname "$0")/.." || exit 1

LOG="logs/ict_scanner.log"
mkdir -p logs

# Loop-liveness heartbeat — written EVERY invocation, BEFORE the session gate, so the
# monitor distinguishes "launchd stopped firing me" (genuinely down) from "running fine,
# no qualifying signals" (healthy, just quiet). The decision log is NOT a heartbeat.
date -u +"%Y-%m-%dT%H:%M:%SZ" > logs/.heartbeat_ict_scanner

# ET time — use the 10# radix prefix so leading-zero hours/mins (08, 09) are read as
# base-10, never octal. Bulletproof vs the old `sed 's/^0//'` (which crashed the time
# gate and recurringly took the scanner DOWN — see NEXT.md).
ET_HOUR=$(TZ="America/New_York" date +"%H")
ET_MIN=$(TZ="America/New_York" date +"%M")
ET_TIME=$(( 10#$ET_HOUR * 60 + 10#$ET_MIN ))

# NY PM: 13:30–16:00 ET = 810–960 min
# London: 02:00–05:00 ET = 120–300 min
NY_PM_START=810
NY_PM_END=960
LONDON_START=120
LONDON_END=300

if [ "$ET_TIME" -ge "$NY_PM_START" ] && [ "$ET_TIME" -le "$NY_PM_END" ]; then
    SESSION="NY_PM"
elif [ "$ET_TIME" -ge "$LONDON_START" ] && [ "$ET_TIME" -le "$LONDON_END" ]; then
    SESSION="London"
else
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Session=$SESSION — scanning..." >> "$LOG"
python3 -m ict.orchestrator --once >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done." >> "$LOG"
