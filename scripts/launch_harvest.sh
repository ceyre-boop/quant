#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Launch Harvest + Retrain — runs both as background daemons
# Logs: logs/harvester.log  logs/retrain.log
#
# Usage:
#   ./scripts/launch_harvest.sh          # start both daemons
#   ./scripts/launch_harvest.sh status   # show process status + DB size
#   ./scripts/launch_harvest.sh stop     # kill both daemons
#   ./scripts/launch_harvest.sh report   # print DB trade count + monthly PnL
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
HARVESTER_PID="$ROOT/.harvester.pid"
RETRAIN_PID="$ROOT/.retrain.pid"

_is_running() {
    local pid_file="$1"
    [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

_start() {
    if _is_running "$HARVESTER_PID"; then
        echo "Harvester already running (pid $(cat $HARVESTER_PID))"
    else
        $PYTHON scripts/continuous_harvester.py \
            >> logs/harvester.log 2>&1 &
        echo $! > "$HARVESTER_PID"
        echo "Harvester started  pid=$(cat $HARVESTER_PID)"
    fi

    if _is_running "$RETRAIN_PID"; then
        echo "Retrain loop already running (pid $(cat $RETRAIN_PID))"
    else
        $PYTHON training/retrain_loop.py \
            >> logs/retrain.log 2>&1 &
        echo $! > "$RETRAIN_PID"
        echo "Retrain loop started  pid=$(cat $RETRAIN_PID)"
    fi

    echo ""
    echo "Watching harvester (Ctrl+C to detach — daemons keep running):"
    tail -f logs/harvester.log
}

_stop() {
    for f in "$HARVESTER_PID" "$RETRAIN_PID"; do
        if _is_running "$f"; then
            kill "$(cat "$f")" && echo "Stopped pid=$(cat $f)"
            rm -f "$f"
        else
            echo "Not running: $f"
        fi
    done
}

_status() {
    echo "=== Process Status ==="
    if _is_running "$HARVESTER_PID"; then
        echo "  Harvester: RUNNING  pid=$(cat $HARVESTER_PID)"
    else
        echo "  Harvester: STOPPED"
    fi
    if _is_running "$RETRAIN_PID"; then
        echo "  Retrain:   RUNNING  pid=$(cat $RETRAIN_PID)"
    else
        echo "  Retrain:   STOPPED"
    fi

    echo ""
    echo "=== DB Stats ==="
    if [[ -f "data/harvest.db" ]]; then
        du -sh data/harvest.db
        $PYTHON - << 'EOF'
import duckdb, sys
con = duckdb.connect("data/harvest.db", read_only=True)
r = con.execute("SELECT COUNT(*), AVG(CAST(is_profitable AS FLOAT))*100, SUM(pnl) FROM trades").fetchone()
print(f"  Total trades:  {r[0]:,}")
print(f"  Win rate:      {r[1]:.1f}%")
print(f"  Total PnL:     ${r[2]:,.0f}")
con.close()
EOF
    else
        echo "  No DB yet — harvester hasn't written data"
    fi

    echo ""
    echo "=== Active Threshold ==="
    if [[ -f "models/current_threshold.json" ]]; then
        cat models/current_threshold.json
    else
        echo "  Not set yet (retrain hasn't run)"
    fi
}

_report() {
    $PYTHON - << 'EOF'
import duckdb, json
from pathlib import Path
con = duckdb.connect("data/harvest.db", read_only=True)

# Load threshold
thr_file = Path("models/current_threshold.json")
thr = json.loads(thr_file.read_text())["threshold"] if thr_file.exists() else 0.5

# Load threshold history for progression trend
hist_file = Path("models/threshold_history.json")
history = json.loads(hist_file.read_text()) if hist_file.exists() else []

print(f"\n{'='*60}")
print(f"  HARVEST REPORT — Active threshold: {thr:.2f}")
print(f"{'='*60}")

# Month-by-month breakdown
rows = con.execute("""
    SELECT
        strftime(window_start, '%Y-%m') AS month,
        COUNT(*) AS n_trades,
        AVG(CAST(is_profitable AS FLOAT))*100 AS win_rate,
        SUM(pnl) AS total_pnl
    FROM trades
    GROUP BY 1
    ORDER BY 1
""").fetchall()

prev_pnl = None
print(f"\n  {'Month':<9} {'Trades':>8} {'WR':>7} {'PnL':>12}  {'Trend'}")
print(f"  {'-'*50}")
for month, n, wr, pnl in rows[-8:]:
    arrow = ""
    if prev_pnl is not None:
        arrow = "▲ BETTER" if pnl > prev_pnl else "▼ worse"
    print(f"  {month:<9} {n:>8,} {wr:>6.1f}% ${pnl:>11,.0f}  {arrow}")
    prev_pnl = pnl

# Strategy breakdown
print(f"\n  {'='*50}")
print(f"  Strategy Performance")
rows2 = con.execute("""
    SELECT strategy,
           COUNT(*) AS n,
           AVG(CAST(is_profitable AS FLOAT))*100 AS wr,
           SUM(pnl) AS pnl
    FROM trades
    GROUP BY strategy
    ORDER BY pnl DESC
""").fetchall()
for strat, n, wr, pnl in rows2:
    print(f"  {strat:<22} n={n:>8,}  WR={wr:5.1f}%  PnL=${pnl:>12,.0f}")

# Regime breakdown
print(f"\n  Regime 0=trending  1=ranging")
rows3 = con.execute("""
    SELECT regime,
           COUNT(*) AS n,
           AVG(CAST(is_profitable AS FLOAT))*100 AS wr,
           SUM(pnl) AS pnl
    FROM trades
    GROUP BY regime
    ORDER BY regime
""").fetchall()
for regime, n, wr, pnl in rows3:
    label = "trending" if regime == 0 else "ranging"
    print(f"  regime={regime} ({label:<9})  n={n:>8,}  WR={wr:5.1f}%  PnL=${pnl:>12,.0f}")

# Threshold progression
if len(history) > 1:
    print(f"\n  Threshold History (last 5 decisions):")
    for h in history[-5:]:
        direction = "▲" if h["upgraded"] else "—"
        print(f"  {h['ts'][:16]}  {h['prev_threshold']:.2f} → {h['new_threshold']:.2f}  {direction}")

con.close()
print()
EOF
}

CMD="${1:-start}"
case "$CMD" in
    start)  _start ;;
    stop)   _stop ;;
    status) _status ;;
    report) _report ;;
    *)
        echo "Usage: $0 {start|stop|status|report}"
        exit 1
        ;;
esac
