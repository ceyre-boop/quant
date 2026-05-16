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

# ET time — strip leading zeros so bash doesn't treat 08/09 as octal
ET_HOUR=$(TZ="America/New_York" date +"%H" | sed 's/^0//')
ET_MIN=$(TZ="America/New_York" date +"%M" | sed 's/^0//')
ET_TIME=$(( ET_HOUR * 60 + ET_MIN ))

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
