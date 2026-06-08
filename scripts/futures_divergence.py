#!/usr/bin/env python3
"""
Divergence tracker — paper ACTUAL net vs backtest EXPECTED net, one row per session.

The single most predictive month-1 metric: does live execution actually capture the edge the
model expects? Gate for live capital: 30-session running |divergence| < 20%.

  EXPECTED net/session ← data/futures/replay_report_*.json  sessions[].net_usd  (already costed)
  ACTUAL   net/session ← data/futures/trade_log.jsonl       (exit-entry)*dir*$/pt*contracts
                                                            minus the honest round-turn cost
  divergence% = (actual - expected) / |expected|            (NEUTRAL/no-trade days → NA)

Regenerated from source each run → deterministic + idempotent (no duplicate rows). Ready-but-empty
until paper trading writes trade_log.jsonl. Output: data/futures/divergence_log.csv

Usage:
  python3 scripts/futures_divergence.py            # rebuild the CSV
  python3 scripts/futures_divergence.py --summary  # print the running-30 gate number
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sovereign.futures.config import contract_spec, round_turn_cost_usd

REPLAY_GLOB = str(ROOT / "data" / "futures" / "replay_report_*.json")
TRADE_LOG = ROOT / "data" / "futures" / "trade_log.jsonl"
OUT = ROOT / "data" / "futures" / "divergence_log.csv"
GATE_PCT = 20.0
WINDOW = 30
COLS = ["session_date", "instrument", "expected_net_usd", "actual_net_usd",
        "divergence_pct", "running_30_avg_pct"]


def _norm(inst: str) -> str:
    return {"ES": "MES", "NQ": "MNQ"}.get((inst or "").upper(), (inst or "").upper())


def _expected() -> dict[tuple[str, str], float]:
    """{(instrument, day): expected_net_usd} from every replay report (later files win)."""
    exp: dict[tuple[str, str], float] = {}
    for f in sorted(glob.glob(REPLAY_GLOB)):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        inst = _norm(d.get("instrument", ""))
        for s in d.get("sessions", []):
            day = s.get("day")
            if day:
                exp[(inst, day)] = float(s.get("net_usd", 0.0))
    return exp


def _actual() -> dict[tuple[str, str], float]:
    """{(instrument, day): actual_net_usd} from paper trade_log (costed). Empty if no log."""
    act: dict[tuple[str, str], float] = {}
    if not TRADE_LOG.exists():
        return act
    for line in TRADE_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            t = json.loads(line)
        except Exception:
            continue
        entry, exit_p = t.get("entry"), t.get("exit")
        if entry is None or exit_p is None:
            continue  # open / un-reconciled trade — not yet comparable
        inst = _norm(t.get("instrument", ""))
        day = (t.get("ts", "") or "")[:10]
        if not day:
            continue
        direction = t.get("direction")
        if direction is None:
            direction = 1 if str(t.get("side", "")).upper() == "LONG" else -1
        n = int(t.get("size_contracts", 1) or 1)
        try:
            dpp = contract_spec(inst)["dollars_per_point"]
            gross = (float(exit_p) - float(entry)) * direction * dpp * n
            net = gross - round_turn_cost_usd(inst, n)
        except Exception:
            continue
        act[(inst, day)] = act.get((inst, day), 0.0) + net
    return act


def build_rows() -> list[dict]:
    exp, act = _expected(), _actual()
    keys = sorted(set(exp) | set(act), key=lambda k: (k[1], k[0]))  # by date, then instrument
    rows, recent = [], []  # recent = trailing list of divergence values
    for inst, day in keys:
        e = exp.get((inst, day))
        a = act.get((inst, day))
        div = None
        if e is not None and a is not None and abs(e) > 1e-9:
            div = (a - e) / abs(e) * 100.0
            recent.append(div)
        run30 = (sum(recent[-WINDOW:]) / len(recent[-WINDOW:])) if recent else None
        rows.append({
            "session_date": day, "instrument": inst,
            "expected_net_usd": round(e, 2) if e is not None else "NA",
            "actual_net_usd": round(a, 2) if a is not None else "NA",
            "divergence_pct": round(div, 1) if div is not None else "NA",
            "running_30_avg_pct": round(run30, 1) if run30 is not None else "NA",
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    rows = build_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    scored = [r for r in rows if r["divergence_pct"] != "NA"]
    latest_run30 = next((r["running_30_avg_pct"] for r in reversed(rows)
                         if r["running_30_avg_pct"] != "NA"), None)

    print(f"Divergence tracker → {OUT}")
    print(f"  sessions: {len(rows)} total | {len(scored)} with paper-vs-expected divergence")
    if latest_run30 is None:
        print("  running-30 divergence: NA (no paper trades yet — tracker armed, waiting on trade_log.jsonl)")
    else:
        flag = "RED — ABOVE GATE" if abs(latest_run30) >= GATE_PCT else "green"
        print(f"  running-30 divergence: {latest_run30:+.1f}%  [gate <{GATE_PCT:.0f}%] → {flag}")

    if args.summary and scored:
        print("\n  recent sessions:")
        for r in scored[-min(10, len(scored)):]:
            print(f"    {r['session_date']} {r['instrument']:4s} "
                  f"exp={r['expected_net_usd']:>8} act={r['actual_net_usd']:>8} "
                  f"div={r['divergence_pct']:>6}% run30={r['running_30_avg_pct']:>6}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
