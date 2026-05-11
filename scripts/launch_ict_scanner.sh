#!/bin/bash
# ICT scanner auto-runner for NY PM session (1:30–4:00 PM ET)
# Triggered by launchd — runs every 5min during the window
#
# Install:
#   chmod +x scripts/launch_ict_scanner.sh
#   cp scripts/com.clawd.ict_scanner.plist ~/Library/LaunchAgents/
#   launchctl load ~/Library/LaunchAgents/com.clawd.ict_scanner.plist

cd "$(dirname "$0")/.." || exit 1

LOG="logs/ict_scanner.log"
mkdir -p logs

# ET hour check (works on macOS)
ET_HOUR=$(TZ="America/New_York" date +"%H")
ET_MIN=$(TZ="America/New_York" date +"%M")
ET_TIME=$((ET_HOUR * 60 + ET_MIN))

# NY PM: 13:30–16:00 ET = 810–960 minutes from midnight
# London: 02:00–05:00 ET = 120–300 minutes
NY_PM_START=810
NY_PM_END=960
LONDON_START=120
LONDON_END=300

if [ "$ET_TIME" -ge "$NY_PM_START" ] && [ "$ET_TIME" -le "$NY_PM_END" ]; then
    SESSION="NY_PM"
elif [ "$ET_TIME" -ge "$LONDON_START" ] && [ "$ET_TIME" -le "$LONDON_END" ]; then
    SESSION="London"
else
    # Outside session — don't run (save resources)
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Session=$SESSION — scanning..." >> "$LOG"
python3 -m ict.orchestrator --once >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done." >> "$LOG"
