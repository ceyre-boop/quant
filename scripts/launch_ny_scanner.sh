#!/bin/bash
# NY AM scanner — 08:00–11:00 ET (12:00–15:00 UTC) data-collection mode
# Grade B accepted, 0.50% risk, bypasses NY_PM veto.
# Triggered by launchd (com.clawd.ny_am_scanner.plist) every 30 min.

cd "$(dirname "$0")/.." || exit 1

LOG="logs/ny_scanner.log"
mkdir -p logs

# UTC minutes since midnight
UTC_HOUR=$(date -u +"%H" | sed 's/^0//')
UTC_MIN=$(date -u +"%M" | sed 's/^0//')
UTC_TIME=$(( UTC_HOUR * 60 + UTC_MIN ))

# 12:00–15:00 UTC = 720–900 min
if [ "$UTC_TIME" -lt 720 ] || [ "$UTC_TIME" -ge 900 ]; then
    exit 0
fi

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] NY AM scan starting..." >> "$LOG"
python3 -m ict.orchestrator --once --ny-am >> "$LOG" 2>&1
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Done." >> "$LOG"
