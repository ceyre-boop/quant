#!/bin/bash
# Reddit sentiment scraper — runs every 30 min, guards to market hours 09:00–17:00 ET.
# Triggered by launchd (com.sovereign.reddit_sentiment.plist).
# Output: data/cache/reddit_sentiment.json + data/health/reddit_status.json

cd "$(dirname "$0")/.." || exit 1

LOG="logs/reddit_scraper.log"
mkdir -p logs data/health

# Machine is on ET — use local hour directly.
HOUR=$(date +"%H" | sed 's/^0*//')   # strip leading zero for arithmetic
MINUTE=$(date +"%M")
TIME_MIN=$(( HOUR * 60 + MINUTE ))

# 09:00–17:00 ET = 540–1020 min since midnight
if [ "$TIME_MIN" -lt 540 ] || [ "$TIME_MIN" -ge 1020 ]; then
    # Outside market hours — skip silently
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Reddit scraper starting..." >> "$LOG"
.venv/bin/python3 -m sovereign.data.reddit_scraper >> "$LOG" 2>&1
STATUS=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Done (exit=$STATUS)." >> "$LOG"
exit $STATUS
