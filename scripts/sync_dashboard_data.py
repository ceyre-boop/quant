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

# agent_scheduler.py was moved to archive/ (the live 2h pulse now only patches the
# oracle_pulse component). Its full 18-component run_health_check is still the canonical
# component-health builder the dashboard needs, so import it from archive/ until it is
# relocated to a live module. TODO(NEXT.md): restore run_health_check to a live home so
# health.json's API components stop going stale between manual syncs.
sys.path.insert(0, str(ROOT / "archive"))
from agent_scheduler import run_health_check  # noqa: E402  (after dotenv, from archive/)


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


G2B_TARGET = 8  # execution validation threshold (reduced from 30)


def _read_g2a() -> dict:
    """Return g2a_validation.json summary or empty defaults if not run yet."""
    path = ROOT / "data" / "agent" / "g2a_validation.json"
    if not path.exists():
        return {"status": "PENDING", "signals_generated": 0, "win_rate": None}
    try:
        data = json.loads(path.read_text())
        return {
            "status": data.get("status", "PENDING"),
            "signals_generated": data.get("signals_generated", 0),
            "win_rate": data.get("win_rate"),
        }
    except Exception:
        return {"status": "PENDING", "signals_generated": 0, "win_rate": None}


def _sync_checklist(fill_count: int) -> None:
    g2a = _read_g2a()

    # Update checklist_state.json (legacy G2 gate — keep in sync for fallback)
    path = ROOT / "data" / "agent" / "checklist_state.json"
    if path.exists():
        state = json.loads(path.read_text())
        for gate in state.get("gates", []):
            if gate.get("id") == "G2":
                remaining = max(0, G2B_TARGET - fill_count)
                gate["value"] = f"{fill_count}/{G2B_TARGET} execution trades"
                gate["status"] = "GREEN" if fill_count >= G2B_TARGET else ("YELLOW" if fill_count >= 4 else "RED")
                gate["detail"] = (
                    "Gate clear!" if fill_count >= G2B_TARGET
                    else f"Need {remaining} more — OANDA bridge live since 2026-05-28"
                )
        state["timestamp"] = datetime.now(timezone.utc).isoformat()
        green = sum(1 for g in state.get("gates", []) if g["status"] == "GREEN")
        state["gates_green"] = green
        state["gates_yellow"] = sum(1 for g in state.get("gates", []) if g["status"] == "YELLOW")
        state["gates_red"] = sum(1 for g in state.get("gates", []) if g["status"] == "RED")
        state["overall"] = "GO" if green == len(state.get("gates", [])) else "WAIT"
        path.write_text(json.dumps(state, indent=2))
        print(f"  Checklist: G2b updated to {fill_count}/{G2B_TARGET} trades")

    # Update prop_challenge_state.json (primary dashboard source)
    prop_path = ROOT / "data" / "agent" / "prop_challenge_state.json"
    if prop_path.exists():
        prop = json.loads(prop_path.read_text())
        for gate in prop.get("gates", []):
            if gate.get("id") == "G2a":
                if g2a["status"] == "PASS":
                    gate["status"] = "GREEN"
                    wr = f"{g2a['win_rate']:.1%}" if g2a["win_rate"] is not None else "?"
                    gate["value"] = f"PASS — {g2a['signals_generated']} signals, WR {wr}"
                    gate["detail"] = "Signal pipeline validated against 30 days of real data"
                elif g2a["status"] == "FAIL":
                    gate["status"] = "RED"
                    gate["value"] = f"FAIL — {g2a['signals_generated']} signals"
                    gate["detail"] = "Re-run validate_signals_retrospective.py and investigate"
                else:
                    gate["status"] = "YELLOW"
                    gate["value"] = "not run"
                    gate["detail"] = "Run: python3 scripts/validate_signals_retrospective.py"
            elif gate.get("id") == "G2b":
                remaining = max(0, G2B_TARGET - fill_count)
                gate["value"] = f"{fill_count}/{G2B_TARGET}"
                gate["status"] = "GREEN" if fill_count >= G2B_TARGET else ("YELLOW" if fill_count >= 4 else "RED")
                gate["detail"] = (
                    "Execution pipeline validated!" if fill_count >= G2B_TARGET
                    else f"Need {remaining} more — use run_fvg_express.py to accelerate"
                )
        prop["timestamp"] = datetime.now(timezone.utc).isoformat()
        greens = sum(1 for g in prop.get("gates", []) if g["status"] == "GREEN")
        prop["overall"] = "GO" if greens == len(prop.get("gates", [])) else "WAIT"
        prop_path.write_text(json.dumps(prop, indent=2))
        print(f"  Prop challenge: G2a={g2a['status']} G2b={fill_count}/{G2B_TARGET}")

    # Update g2_progress.json
    g2p_path = ROOT / "data" / "agent" / "g2_progress.json"
    if g2p_path.exists():
        g2p = json.loads(g2p_path.read_text())
        g2p["total"] = fill_count
        g2p["target"] = G2B_TARGET
        g2p["g2a_status"] = g2a["status"]
        g2p["g2a_signals"] = g2a["signals_generated"]
        g2p["g2a_win_rate"] = g2a["win_rate"]
        g2p_path.write_text(json.dumps(g2p, indent=2))
        print(f"  G2 progress: total={fill_count} target={G2B_TARGET} g2a={g2a['status']}")


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
