#!/usr/bin/env bash
# connect_all_plists.sh — install every new plist authored since July 20 baseline
# Run this ONCE on your Mac from the quant/ root:
#   cd ~/quant && bash scripts/connect_all_plists.sh
#
# What this does:
#   1. Copies each new plist to ~/Library/LaunchAgents/
#   2. Loads it (launchctl load)
#   3. Skips anything already loaded
#   4. Runs plist_watchdog --rebaseline at the end
#
# After this runs: `launchctl list | grep alta` should show ~53 jobs.

set -e
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
AGENTS="$HOME/Library/LaunchAgents"
QUANT="$(dirname "$SCRIPTS")"
LOG="$QUANT/logs/connect_all_plists.log"

mkdir -p "$AGENTS" "$(dirname "$LOG")"
echo "=== connect_all_plists.sh $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

# ── Plists to install (everything not in July-20 baseline)
# hyp107_shadow, ib_shortable excluded — need separate decisions
AUTO_LOAD=(
    com.alta.alexandria_health
    com.alta.dashboard_refresh
    com.alta.decision_backfill
    com.alta.dip_daily
    com.alta.dip_diffuse
    com.alta.dip_peak
    com.alta.dip_warmup
    com.alta.edge.factory
    com.alta.esnq.brief
    com.alta.forensics
    com.alta.gdelt_retry
    com.alta.intelligence_run
    com.alta.obsidian_sync
    com.alta.oracle.market_briefing
    com.alta.paper_accounts
    com.alta.petrules_gate
    com.alta.system_eod
    com.alta.system_health
    com.alta.system_health_verdict
    com.alta.system_morning
    com.alta.system_regime
)

LOADED=0
SKIPPED=0
FAILED=0

for label in "${AUTO_LOAD[@]}"; do
    plist_src="$SCRIPTS/${label}.plist"
    plist_dst="$AGENTS/${label}.plist"

    if ! [ -f "$plist_src" ]; then
        echo "  MISSING: $plist_src — skip" | tee -a "$LOG"
        continue
    fi

    # Check if already loaded
    if launchctl list "$label" &>/dev/null 2>&1; then
        echo "  SKIP (already loaded): $label" | tee -a "$LOG"
        ((SKIPPED++)) || true
        continue
    fi

    # Copy and load
    cp "$plist_src" "$plist_dst"
    if launchctl load "$plist_dst" 2>>"$LOG"; then
        echo "  LOADED: $label" | tee -a "$LOG"
        ((LOADED++)) || true
    else
        echo "  FAIL: $label (see $LOG)" | tee -a "$LOG"
        ((FAILED++)) || true
    fi
done

echo "" | tee -a "$LOG"
echo "Done: $LOADED loaded, $SKIPPED already running, $FAILED failed" | tee -a "$LOG"

# ── Petrules first live run (runs immediately so dashboard shows real data today)
echo "" | tee -a "$LOG"
echo "=== Running Petrules scanner first pass ===" | tee -a "$LOG"
python3 "$SCRIPTS/petrules_gate_scanner.py" 2>&1 | tee -a "$LOG" || echo "Petrules scan errored — check $LOG"

# ── Rebaseline watchdog
echo "" | tee -a "$LOG"
echo "=== Rebaselining plist watchdog ===" | tee -a "$LOG"
python3 "$SCRIPTS/plist_watchdog.py" --rebaseline "connect_all_plists.sh: installed $(( LOADED )) new plists (2026-07-24)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Verify: $(launchctl list | grep -c 'com.alta\|com.sovereign') alta/sovereign jobs loaded ===" | tee -a "$LOG"
echo "Full job list:" | tee -a "$LOG"
launchctl list | grep 'com.alta\|com.sovereign' | awk '{print $NF}' | sort | tee -a "$LOG"
