"""
Prop Firm Deployment Checklist — Layer 5 of the Sovereign Intelligence Architecture.

Five gates must all be GREEN before buying the real Lucid LucidFlex $100k challenge.
Run this at any time to see current status. The scheduler posts a daily FYI.

Gates:
  G1  MC pass rate ≥ 70%                  (simulation — already validated 99.7-100%)
  G2  Live paper trades ≥ 30              (London+GradeA+committed, from paper_challenge)
  G3  Live walk-forward WR within 15%     (live WR vs backtest 41% — tolerance ±15%)
  G4  Bridge macro threat < 0.85          (Library not in TIGHTEN/HALT_NEW)
  G5  Minimum 5 consecutive non-bust days (system hasn't blown up recently)

All gates GREEN → 🟢 BUY — print purchase instructions
Any gate RED    → 🔴 WAIT — print exactly what's missing

Run:
    python3 sovereign/propfirm/deployment_checklist.py
    python3 sovereign/propfirm/deployment_checklist.py --verbose
    python3 sovereign/propfirm/deployment_checklist.py --json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]

# File references
PAPER_CHALLENGE_FILE = ROOT / "data" / "propfirm" / "active_challenge.json"
PAPER_TRADE_LOG      = ROOT / "data" / "ledger" / "ict_paper_trades.json"
MC_RESULTS_FILE      = ROOT / "logs" / "prop_challenge_sim.json"
BRIDGE_STATE_FILE    = ROOT / "data" / "forensics" / "cross_system_state.json"
VETO_LEDGER          = ROOT / "data" / "ledger" / "ict_veto_ledger_2026_05.jsonl"

# Thresholds
MIN_LIVE_TRADES     = 30
MIN_MC_PASS_RATE    = 0.70
MAX_BRIDGE_THREAT   = 0.85      # above this = WAIT (TIGHTEN/HALT_NEW territory)
WR_TOLERANCE        = 0.15      # live WR must be within ±15pp of backtest 41%
BACKTEST_WR         = 0.41      # London+GradeA backtest win rate
MIN_NON_BUST_DAYS   = 5         # consecutive paper days without bust

PURCHASE_URL     = "https://lucidtrader.com/lucid-flex"
CHALLENGE_PRICE  = 399          # approximate USD for $100k LucidFlex
CHALLENGE_SIZE   = 100_000


@dataclass
class GateResult:
    gate_id: str
    name: str
    status: str          # GREEN / RED / YELLOW / UNKNOWN
    value: str           # what we measured
    target: str          # what we need
    detail: str = ""


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# ── Gate evaluators ───────────────────────────────────────────────────────────

def gate_mc_pass_rate() -> GateResult:
    mc = _load_json(MC_RESULTS_FILE)
    if not mc:
        return GateResult("G1", "MC pass rate", "UNKNOWN",
                          "no simulation file", f"≥{MIN_MC_PASS_RATE*100:.0f}%",
                          "Run: python3 sovereign/propfirm/challenge_simulator.py --window london_a")
    rate = mc.get("pass_rate", 0.0)
    window = mc.get("window", "?")
    n = mc.get("n_simulations", 0)
    status = "GREEN" if rate >= MIN_MC_PASS_RATE else "RED"
    return GateResult(
        "G1", "MC pass rate", status,
        f"{rate*100:.1f}% ({n:,} sims, window={window})",
        f"≥{MIN_MC_PASS_RATE*100:.0f}%",
        f"Median days to pass: {mc.get('median_days_to_pass', '?')} | "
        f"P90: {mc.get('p90_days_to_pass', '?')} days",
    )


def gate_live_trades() -> GateResult:
    raw = _load_json(PAPER_TRADE_LOG)
    closed = raw.get("closed", [])

    # Filter to London+GradeA trades only (committed filter proxy)
    london_a = [
        t for t in closed
        if t.get("session", "") == "London"
        and t.get("grade", "")  in ("A", "A+")   # A+ counts — commitment detector already gated it
    ]
    n = len(london_a)
    status = "GREEN" if n >= MIN_LIVE_TRADES else "RED"
    remaining = max(0, MIN_LIVE_TRADES - n)
    return GateResult(
        "G2", "Live paper trades", status,
        f"{n} London+GradeA trades logged",
        f"≥{MIN_LIVE_TRADES} live paper trades",
        f"{'DONE' if status == 'GREEN' else f'Need {remaining} more — collecting live since 2026-05-18'}",
    )


def gate_walk_forward_wr() -> GateResult:
    raw = _load_json(PAPER_TRADE_LOG)
    closed = raw.get("closed", [])

    london_a = [
        t for t in closed
        if t.get("session", "") == "London"
        and t.get("grade", "") in ("A", "A+")
        and t.get("outcome") in ("TP1", "TP2", "STOP")
    ]

    if len(london_a) < 10:
        return GateResult(
            "G3", "Walk-forward WR alignment", "YELLOW",
            f"Only {len(london_a)} closed trades (need ≥10 to measure)",
            f"Live WR within ±{WR_TOLERANCE*100:.0f}pp of {BACKTEST_WR*100:.0f}%",
            "Collecting data — check again after 10+ closed trades",
        )

    wins = [t for t in london_a if t.get("outcome") in ("TP1", "TP2")]
    live_wr = len(wins) / len(london_a)
    lower = BACKTEST_WR - WR_TOLERANCE
    upper = BACKTEST_WR + WR_TOLERANCE
    status = "GREEN" if lower <= live_wr <= upper else (
        "YELLOW" if live_wr >= lower * 0.85 else "RED"
    )
    return GateResult(
        "G3", "Walk-forward WR alignment", status,
        f"Live WR={live_wr*100:.1f}% on {len(london_a)} trades",
        f"Backtest {BACKTEST_WR*100:.0f}% ±{WR_TOLERANCE*100:.0f}pp "
        f"[{lower*100:.0f}%–{upper*100:.0f}%]",
        f"{'Within tolerance' if status=='GREEN' else 'Outside tolerance — review setup quality'}",
    )


def gate_bridge_threat() -> GateResult:
    state = _load_json(BRIDGE_STATE_FILE)
    threat = state.get("library_threat_score", 0.0)
    mode   = state.get("ict_mode", "UNKNOWN")
    regime = state.get("library_primary_regime", "?")
    updated = state.get("last_updated", "never")[:16]

    if not state:
        return GateResult("G4", "Bridge macro threat", "UNKNOWN",
                          "no bridge state file",
                          f"threat < {MAX_BRIDGE_THREAT}",
                          "Run: python3 sovereign/intelligence/cross_system_bridge.py --update")

    status = "GREEN" if threat < MAX_BRIDGE_THREAT else (
        "YELLOW" if threat < 0.95 else "RED"
    )
    return GateResult(
        "G4", "Bridge macro threat", status,
        f"threat={threat:.2f} mode={mode} ({updated})",
        f"threat < {MAX_BRIDGE_THREAT} (NORMAL mode)",
        f"Regime: {regime} | {'Buy window open' if status=='GREEN' else 'Wait for macro to clear'}",
    )


def gate_consecutive_days() -> GateResult:
    ch = _load_json(PAPER_CHALLENGE_FILE)
    day_log = ch.get("day_log", [])
    trading_days = ch.get("trading_days", 0)

    if trading_days == 0:
        return GateResult(
            "G5", "Consecutive non-bust days", "YELLOW",
            "Paper challenge not yet started",
            f"≥{MIN_NON_BUST_DAYS} consecutive days without bust",
            "Paper challenge #1 active — days will accumulate",
        )

    # Check last N days for busts
    recent = day_log[-MIN_NON_BUST_DAYS:] if len(day_log) >= MIN_NON_BUST_DAYS else day_log
    busts = [d for d in recent if d.get("status") == "BUST"]

    if trading_days < MIN_NON_BUST_DAYS:
        status = "YELLOW"
        detail = f"Only {trading_days} days elapsed — wait for {MIN_NON_BUST_DAYS}"
    elif busts:
        status = "RED"
        detail = f"Bust detected in last {MIN_NON_BUST_DAYS} days — review risk"
    else:
        status = "GREEN"
        detail = f"{trading_days} days running, last {len(recent)} days clean"

    return GateResult(
        "G5", "Consecutive non-bust days", status,
        f"{trading_days} paper trading days elapsed",
        f"≥{MIN_NON_BUST_DAYS} consecutive non-bust days",
        detail,
    )


# ── Report ────────────────────────────────────────────────────────────────────

STATUS_ICON = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "UNKNOWN": "⚪"}


def run_checklist(verbose: bool = False) -> dict:
    gates = [
        gate_mc_pass_rate(),
        gate_live_trades(),
        gate_walk_forward_wr(),
        gate_bridge_threat(),
        gate_consecutive_days(),
    ]

    greens  = sum(1 for g in gates if g.status == "GREEN")
    yellows = sum(1 for g in gates if g.status == "YELLOW")
    reds    = sum(1 for g in gates if g.status in ("RED", "UNKNOWN"))
    go      = reds == 0 and yellows == 0 and greens == len(gates)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall": "GO" if go else "WAIT",
        "gates_green": greens,
        "gates_yellow": yellows,
        "gates_red": reds,
        "gates": [
            {
                "id": g.gate_id,
                "name": g.name,
                "status": g.status,
                "value": g.value,
                "target": g.target,
                "detail": g.detail,
            }
            for g in gates
        ],
        "purchase_url": PURCHASE_URL if go else None,
        "challenge_price_usd": CHALLENGE_PRICE,
        "challenge_size_usd": CHALLENGE_SIZE,
    }


def print_checklist(result: dict, verbose: bool = False) -> None:
    overall = result["overall"]
    ts = result["timestamp"][:16]
    print(f"\n{'='*62}")
    print(f"PROP FIRM DEPLOYMENT CHECKLIST — {ts}")
    print(f"{'='*62}")

    for g in result["gates"]:
        icon = STATUS_ICON.get(g["status"], "?")
        print(f"  {icon} {g['id']}: {g['name']}")
        print(f"       Measured: {g['value']}")
        print(f"       Target:   {g['target']}")
        if verbose and g["detail"]:
            print(f"       Detail:   {g['detail']}")
        print()

    print(f"  Gates GREEN: {result['gates_green']}/5")
    if result["gates_yellow"]:
        print(f"  Gates YELLOW: {result['gates_yellow']}/5  (collecting data)")
    if result["gates_red"]:
        print(f"  Gates RED:    {result['gates_red']}/5  (blocking)")
    print()

    if overall == "GO":
        print(f"  🟢 VERDICT: ALL GATES CLEAR — BUY THE CHALLENGE")
        print(f"  💰 Cost: ~${result['challenge_price_usd']:,} for ${result['challenge_size_usd']:,} account")
        print(f"  🔗 {result['purchase_url']}")
    else:
        reds = [g for g in result["gates"] if g["status"] in ("RED", "UNKNOWN")]
        yellows = [g for g in result["gates"] if g["status"] == "YELLOW"]
        print(f"  🔴 VERDICT: NOT YET — {len(reds)} gate(s) RED, {len(yellows)} YELLOW")
        if reds:
            print(f"\n  Blocking gates:")
            for g in reds:
                print(f"    {g['id']} {g['name']}: {g['detail']}")
        if yellows:
            print(f"\n  Collecting (non-blocking):")
            for g in yellows:
                print(f"    {g['id']} {g['name']}: {g['detail']}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prop firm deployment checklist")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detail lines for each gate")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON only (for agent consumption)")
    args = parser.parse_args()

    result = run_checklist(verbose=args.verbose)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_checklist(result, verbose=args.verbose)
