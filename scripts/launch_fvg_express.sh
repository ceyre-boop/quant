#!/bin/bash
# FVG express scanner — London (02:00–05:00 UTC) and NY AM (12:00–15:00 UTC)
# Triggered by launchd (com.sovereign.fvg_express.plist) every 60 min.

cd "$(dirname "$0")/.." || exit 1

LOG="logs/fvg_express.log"
mkdir -p logs

# UTC minutes since midnight
UTC_HOUR=$(date -u +"%H" | sed 's/^0//')
UTC_MIN=$(date -u +"%M" | sed 's/^0//')
UTC_TIME=$(( UTC_HOUR * 60 + UTC_MIN ))

# London:  02:00–05:00 UTC = 120–300 min
# NY AM:   12:00–15:00 UTC = 720–900 min
LONDON=(120 300)
NY_AM=(720 900)

IN_LONDON=$([ "$UTC_TIME" -ge "${LONDON[0]}" ] && [ "$UTC_TIME" -lt "${LONDON[1]}" ] && echo 1 || echo 0)
IN_NY_AM=$([ "$UTC_TIME" -ge "${NY_AM[0]}" ] && [ "$UTC_TIME" -lt "${NY_AM[1]}" ] && echo 1 || echo 0)

if [ "$IN_LONDON" -eq 0 ] && [ "$IN_NY_AM" -eq 0 ]; then
    exit 0
fi

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] FVG express scan starting..." >> "$LOG"
python3 scripts/fvg_express.py >> "$LOG" 2>&1
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Done." >> "$LOG"
