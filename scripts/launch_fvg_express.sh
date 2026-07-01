#!/bin/bash
# FVG express scanner — London (02:00–05:00 UTC) and NY AM (12:00–15:00 UTC)
# Triggered by launchd (com.sovereign.fvg_express.plist) every 60 min.

# ─── DISABLED 2026-07-01 — DO NOT RE-ENABLE WITHOUT A LOGGED param_change ──────
# This launcher drives scripts/fvg_express.py, which places live OANDA orders on
# FVG_PAIRS = [GBPUSD, EURUSD, AUDUSD, AUDNZD, USDJPY] with NO decision logging.
# Two NON-NEGOTIABLE violations (see quant/CLAUDE.md):
#   1. HYP-045: AUDNZD is excluded and must NEVER trade — this job traded it
#      (fills trade_id 25 on 2026-06-01, 119 on 2026-06-29, both ~03:0X UTC).
#   2. Oracle loop: _place_trade() never calls log_forex_decision(), so every
#      fill (incl. untracked USD_JPY wins) bypasses decision_logger → Oracle
#      learns from a survivorship-biased 100%-loss sample.
# The plist was moved to ~/Library/LaunchAgents/disabled/ and unloaded. This
# hard guard stays so a stray re-deploy/reload can't silently revive it.
# To revive: fix both violations in fvg_express.py, log the rationale in
# config/param_change_log, then delete this block.
echo "[launch_fvg_express] DISABLED 2026-07-01 (HYP-045 + Oracle-bypass). Exiting." >&2
exit 0
# ──────────────────────────────────────────────────────────────────────────────

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
