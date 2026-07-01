# DEPRECATED 2026-06-30 — no-op stub.
#
# This script formerly pushed live system state to the Firebase Realtime
# Database (paths under /signals, /session, /system). No live dashboard reads
# those paths anymore: dashboards read the Render backend (/api/*) and committed
# JSON snapshots. The Firebase push path is dead code.
#
# The 200+ lines of push logic that used to live below `sys.exit(0)` were
# unreachable and have been removed (see [CLEANUP] commit). This file is kept
# only as a no-op so any stray caller (cron, launchd, docs) exits cleanly.
# Safe to delete once no dependents remain.
import sys

print("[push_to_firebase] DEPRECATED — no-op. Firebase push path is dead.", file=sys.stderr)
sys.exit(0)
