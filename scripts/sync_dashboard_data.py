"""
sync_dashboard_data.py — refresh GitHub Pages static data files

Run this locally before committing + pushing to master.
Loads .env so API keys are visible, runs the real health check,
converts fills jsonl → json, and syncs oracle reflections.

Usage:
    python3 scripts/sync_dashboard_data.py
    # then: git add data/ && git commit && git push origin HEAD:master
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")  # MUST be before agent_scheduler import so os.environ is populated

from scripts.agent_scheduler import run_health_check  # noqa: E402  (after dotenv)


def _sync_health() -> dict:
    print("Running API health check (real connections)…")
    health = run_health_check()
    out = ROOT / "data" / "agent" / "health.json"
    out.write_text(json.dumps(health, indent=2))
    green = sum(1 for v in health["components"].values() if v["status"] == "GREEN")
    yellow = sum(1 for v in health["components"].values() if v["status"] == "YELLOW")
    red = sum(1 for v in health["components"].values() if v["status"] == "RED")
    print(f"  Health: {health['overall']} | GREEN={green} YELLOW={yellow} RED={red}")
    for k, v in health["components"].items():
        icon = "✓" if v["status"] == "GREEN" else ("!" if v["status"] == "YELLOW" else "✗")
        print(f"    {icon} {k:<25} {v['status']:<8} {v.get('detail','')[:60]}")
    return health


def _sync_fills() -> int:
    src = ROOT / "data" / "ledger" / "oanda_fills.jsonl"
    dst = ROOT / "data" / "ledger" / "oanda_fills.json"
    fills = []
    if src.exists():
        for line in src.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    fills.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    dst.write_text(json.dumps(fills, indent=2))
    print(f"  OANDA fills: {len(fills)} trades → data/ledger/oanda_fills.json")
    return len(fills)


def _sync_checklist(fill_count: int) -> None:
    path = ROOT / "data" / "agent" / "checklist_state.json"
    if not path.exists():
        return
    state = json.loads(path.read_text())
    for gate in state.get("gates", []):
        if gate.get("id") == "G2":
            needed = 30
            have = fill_count
            remaining = max(0, needed - have)
            gate["value"] = f"{have} London+GradeA trades logged"
            gate["status"] = "GREEN" if have >= needed else ("YELLOW" if have >= 10 else "RED")
            gate["detail"] = (
                f"Gate clear!" if have >= needed
                else f"Need {remaining} more — OANDA bridge live since 2026-05-28"
            )
    state["timestamp"] = datetime.now(timezone.utc).isoformat()
    green = sum(1 for g in state.get("gates", []) if g["status"] == "GREEN")
    state["gates_green"] = green
    state["gates_yellow"] = sum(1 for g in state.get("gates", []) if g["status"] == "YELLOW")
    state["gates_red"] = sum(1 for g in state.get("gates", []) if g["status"] == "RED")
    state["overall"] = "GO" if green == len(state.get("gates", [])) else "WAIT"
    path.write_text(json.dumps(state, indent=2))
    print(f"  Checklist: G2 updated to {fill_count}/30 trades")


def _sync_tv_regime_signals() -> int:
    """Ensure tv_regime_signals.json exists and is tracked. Prune entries older than 48h."""
    path = ROOT / "data" / "agent" / "tv_regime_signals.json"
    if not path.exists():
        path.write_text("[]")
        print("  TV regime signals: created empty data/agent/tv_regime_signals.json")
        return 0

    try:
        signals = json.loads(path.read_text())
        if not isinstance(signals, list):
            signals = []
    except Exception:
        signals = []

    from datetime import timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    before = len(signals)
    signals = [
        s for s in signals
        if _parse_ts_utc(s.get("timestamp", "")) >= cutoff
    ]
    if len(signals) != before:
        path.write_text(json.dumps(signals, indent=2))
    print(f"  TV regime signals: {len(signals)} signals (last 48h) → data/agent/tv_regime_signals.json")
    return len(signals)


def _parse_ts_utc(ts_str: str):
    from datetime import timezone
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def _sync_indicators() -> tuple[bool, bool]:
    """Ensure indicator data files are tracked for GitHub Pages. Returns (memory_ok, snapshot_ok)."""
    ind_dir = ROOT / "data" / "indicators"
    memory_path = ind_dir / "oracle_indicator_memory.json"
    snap_path   = ind_dir / "live_snapshot.json"

    if not memory_path.exists():
        print("  Indicators: oracle_indicator_memory.json missing — run build_indicator_ontology.py first")
        memory_ok = False
    else:
        data = json.loads(memory_path.read_text())
        total_obs = data.get("total_observations", 0)
        green_cnt = data.get("green_conditions_found", 0)
        print(f"  Indicators: {total_obs:,} obs, {green_cnt} green conditions — data/indicators/oracle_indicator_memory.json")
        memory_ok = True

    if not snap_path.exists():
        print("  Indicators: live_snapshot.json missing — run pulse_check.py to generate it")
        snapshot_ok = False
    else:
        data = json.loads(snap_path.read_text())
        n_pairs = len(data.get("pairs", {}))
        ts = data.get("timestamp", "")[:16]
        print(f"  Indicators: {n_pairs} pairs live as of {ts} — data/indicators/live_snapshot.json")
        snapshot_ok = True

    return memory_ok, snapshot_ok


def _sync_reflections() -> str | None:
    ref_dir = ROOT / "data" / "oracle" / "reflections"
    files = sorted(ref_dir.glob("20??-??-??.json"))
    if not files:
        return None
    latest = files[-1]
    print(f"  Latest reflection: {latest.name}")
    return latest.stem


def main() -> None:
    print(f"\n{'='*54}")
    print(f"Sovereign Dashboard Sync — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*54}\n")

    _sync_health()
    print()
    fill_count = _sync_fills()
    _sync_checklist(fill_count)
    _sync_tv_regime_signals()
    _sync_indicators()
    ref_date = _sync_reflections()

    print(f"\n{'─'*54}")
    print("Files ready to commit:")
    print("  data/agent/health.json")
    print("  data/agent/checklist_state.json")
    print("  data/ledger/oanda_fills.json")
    print("  data/agent/tv_regime_signals.json")
    print("  data/indicators/oracle_indicator_memory.json  (if built)")
    print("  data/indicators/live_snapshot.json            (if pulse ran)")
    if ref_date:
        print(f"  data/oracle/reflections/{ref_date}.json")
    print("\nNext:")
    print("  git add data/agent/ data/ledger/ data/indicators/ data/oracle/reflections/ index.html")
    print("  git commit -m '[DASHBOARD] Add indicator ontology + consensus panel'")
    print("  git push origin HEAD:master")


if __name__ == "__main__":
    main()
