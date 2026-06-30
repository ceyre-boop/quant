#!/bin/bash
# evening_prep.sh — daily 7 PM coding-window prep
#
# Scheduled via launchd: com.alta.evening_prep
# Runs at 6:00 PM ET / 22:00 UTC daily (StartCalendarInterval Hour=22, Minute=0).
#
# Prepares the quant repo for the user's 7 PM coding window:
#   1. Sync dashboard data
#   2. Auto-commit pending data/ changes
#   3. Summarize today's commits
#   4. Print scanner health (last ICT scanner log line)
#
# Never blocks: every step is guarded and the script always exits 0.

REPO="${HOME}/quant"
cd "${REPO}" || exit 0

DATE="$(date +%Y-%m-%d)"

echo "════ EVENING PREP | ${DATE} ════"

# 1. Sync dashboard data (skip silently if the sync script is absent)
if [ -f "${REPO}/scripts/sync_dashboard_data.py" ]; then
  echo "── Syncing dashboard data ──"
  python3 "${REPO}/scripts/sync_dashboard_data.py" || echo "  (sync_dashboard_data.py exited non-zero — continuing)"
else
  echo "── sync_dashboard_data.py not found — skipping sync ──"
fi

# 2. Auto-commit any pending data/ changes
echo "── Auto-committing pending data/ changes ──"
git add data/ 2>/dev/null
if ! git diff --cached --quiet 2>/dev/null; then
  git commit -m "[AUTO] Evening data sync ${DATE}" || echo "  (commit failed — continuing)"
else
  echo "  No pending data/ changes to commit."
fi

# 3. Summary of today's commits
echo "── Today's commits (since 6am) ──"
git log --oneline --since="6am" HEAD || echo "  (git log failed — continuing)"

# 4. Health check — last line of the ICT scanner log
echo "── Scanner health (last ict_scanner.log line) ──"
if [ -f "${REPO}/logs/ict_scanner.log" ]; then
  tail -n 1 "${REPO}/logs/ict_scanner.log"
else
  echo "  logs/ict_scanner.log not found."
fi

echo "════ EVENING PREP COMPLETE ════"

# 5. Always exit clean — never block the coding window
exit 0
