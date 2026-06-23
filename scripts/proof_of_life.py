#!/usr/bin/env python3
"""
Proof of Life — daily "the system is alive and producing signal" summary.

Read-only. Answers the question Colin checks OANDA for ("is the system producing signals?")
WITHOUT real-money risk and WITHOUT manufacturing trades: it surfaces the would-be signals,
how close each pair is to firing, the strongest near-miss, and how long since a real fill —
the honest proof of life. No trades are placed; nothing here is a trading input.

Reads:  data/agent/forex_proximity.json (per-pair conviction/direction/proximity),
        data/ledger/oanda_fills.jsonl (last actual fill),
        data/oracle/loop_health_status.json (are the scanners alive).
Writes: data/agent/proof_of_life.json  (read by the MCP tool + dashboard).

Run:  python3 scripts/proof_of_life.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROX = ROOT / "data" / "agent" / "forex_proximity.json"
FILLS = ROOT / "data" / "ledger" / "oanda_fills.jsonl"
LOOPH = ROOT / "data" / "oracle" / "loop_health_status.json"
OUT = ROOT / "data" / "agent" / "proof_of_life.json"


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _age_hours(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - t).total_seconds() / 3600.0, 1)
    except Exception:
        return None


def _last_fill() -> dict | None:
    try:
        lines = [l for l in FILLS.read_text().splitlines() if l.strip()]
        if not lines:
            return None
        t = json.loads(lines[-1])
        age_h = _age_hours(t.get("timestamp"))
        return {
            "pair": t.get("pair"),
            "direction": t.get("direction"),
            "fill_price": t.get("fill_price"),
            "date": str(t.get("timestamp", ""))[:10],
            "days_ago": round(age_h / 24.0, 1) if age_h is not None else None,
        }
    except Exception:
        return None


def compute() -> dict:
    prox = _read_json(PROX) or {}
    lh = _read_json(LOOPH) or {}
    pairs_raw = prox.get("pairs", []) or []

    pairs = []
    for p in pairs_raw:
        conv = p.get("conviction")
        pairs.append({
            "pair": p.get("pair"),
            "conviction": conv,
            "direction": p.get("direction"),
            "pct_to_trigger": p.get("pct_to_trigger"),
            "regime": p.get("regime"),
            "rate_differential": p.get("rate_differential"),
        })
    # rank by conviction (the closest-to-firing read)
    ranked = sorted([p for p in pairs if isinstance(p.get("conviction"), (int, float))],
                    key=lambda x: x["conviction"], reverse=True)
    strongest = ranked[0] if ranked else None
    would_fire = [p for p in pairs if p.get("direction") not in (None, "NO_TRADE")]

    loops = lh.get("loops", {})
    down = lh.get("down", []) or []
    scan_age = _age_hours(prox.get("last_scan"))
    forex_alive = (loops.get("forex_scan", {}).get("status") == "ALIVE")
    alive = bool(loops) and not lh.get("frozen") and forex_alive

    last = _last_fill()

    # the satisfying one-liner — lead with a fresh fill (the real proof of life)
    fresh_fill = last and isinstance(last.get("days_ago"), (int, float)) and last["days_ago"] < 1.0
    if fresh_fill:
        s = f"🟢 TRADE FIRED today: {last['pair']} {last['direction']} @ {last['fill_price']}. "
    else:
        s = "System alive. "
    s += f"{len(pairs)} pairs scanned {scan_age}h ago — "
    if would_fire:
        wf = ", ".join(f"{p['pair']} {p['direction']}" for p in would_fire)
        s += f"{len(would_fire)} signalling ({wf}). "
    else:
        s += "all NO_TRADE (edge not firing — correct, not broken). "
    if strongest:
        s += f"Highest conviction: {strongest['pair']} {strongest['conviction']}. "
    if last and not fresh_fill:
        s += f"Last fill: {last['pair']} {last['direction']}, {last['days_ago']}d ago."

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "alive": alive,
        "fired_today": bool(fresh_fill),
        "frozen": lh.get("frozen"),
        "scan_age_hours": scan_age,
        "n_pairs_scanned": len(pairs),
        "n_would_fire_today": len(would_fire),
        "strongest_signal": strongest,
        "pairs": ranked or pairs,
        "last_fill": last,
        "loops_alive": sum(1 for v in loops.values() if v.get("status") == "ALIVE"),
        "loops_down": down,
        "summary_line": s,
        "note": "Honest proof-of-life: would-be signals + proximity, never a forced trade. Read-only.",
    }


def main() -> dict:
    pol = compute()
    OUT.write_text(json.dumps(pol, indent=2))
    return pol


if __name__ == "__main__":
    p = main()
    print(p["summary_line"])
    print(f"  alive={p['alive']} loops_alive={p['loops_alive']} down={p['loops_down']} "
          f"would_fire={p['n_would_fire_today']}")
