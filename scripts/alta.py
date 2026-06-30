#!/usr/bin/env python3
"""
alta — operator CLI for the system kill switch (master freeze for the live trading path).

    python3 scripts/alta.py freeze "reason" [--hard]   # freeze trading path (+ config if --hard)
    python3 scripts/alta.py thaw                        # resume
    python3 scripts/alta.py status                      # show state + what's blocked

soft freeze blocks the trading/signal path (forex_live_scan placement, DecisionChain).
hard freeze ALSO blocks approve_edge.py (live-config mutation).
Monitoring (pulse_check/loop_health) and Oracle cognition keep running either way.

Tip: add a shell alias so it works anywhere — `alias alta='python3 ~/quant/scripts/alta.py'`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.utils import kill_switch as ks

PROOF_DIR = ROOT / "data" / "proof"
LIVE_EQUITY = ROOT / "data" / "agent" / "equity_curve_live.jsonl"


def cmd_freeze(args) -> None:
    p = ks.freeze(args.reason, hard=args.hard, by=args.by)
    blocks = "trading/signal path" + (" + approve_edge (live config)" if p["mode"] == "hard" else "")
    print(f"🧊 SYSTEM FROZEN ({p['mode']}) — {p['reason']}")
    print(f"   at {p['frozen_at']} by {p['by']}")
    print(f"   blocks: {blocks}")
    print(f"   still running: pulse_check/loop_health (monitoring), Oracle reflect/briefing (cognition)")
    print(f"   file: {ks.KILL_SWITCH}")
    print(f"   thaw with: python3 scripts/alta.py thaw")


def cmd_thaw(args) -> None:
    prior = ks.thaw(by=args.by)
    if prior:
        print(f"☀️  THAWED — was {prior.get('mode')} freeze ({prior.get('reason', '')})")
        print("   trading path resumes on the next scheduled cycle.")
    else:
        print("Not frozen — nothing to thaw.")


def cmd_status(args) -> None:
    s = ks.state()
    if not s:
        print("🟢 RUNNING — no freeze active. Trading path live.")
        return
    print(f"🧊 FROZEN ({s.get('mode')}) — {s.get('reason', '')}")
    print(f"   since {s.get('frozen_at')} by {s.get('by')}")
    print(f"   blocked: trading/signal path (forex_live_scan placement, DecisionChain.evaluate)")
    if s.get("mode") == "hard":
        print(f"   blocked: approve_edge.py (live-config mutation)")
    print(f"   running: pulse_check/loop_health (monitoring), Oracle reflect/briefing (cognition)")


def cmd_prove(args) -> None:
    """Run the backtest proof engine (scripts/prove.py) and draw the equity curve."""
    cmd = [sys.executable, str(ROOT / "scripts" / "prove.py")]
    if args.start:
        cmd += ["--start", args.start]
    if args.end:
        cmd += ["--end", args.end]
    if args.pairs:
        cmd += ["--pairs", *args.pairs]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


def cmd_exit_policy_evolution(args) -> None:
    """Run the HYP-067 evolutionary exit-policy search (scripts/research/exit_policy_evolution.py)."""
    cmd = [sys.executable, str(ROOT / "scripts" / "research" / "exit_policy_evolution.py")]
    if args.sign:
        cmd += ["--sign"]
    if args.standalone:
        cmd += ["--standalone"]
    if args.pop is not None:
        cmd += ["--pop", str(args.pop)]
    if args.generations is not None:
        cmd += ["--generations", str(args.generations)]
    if args.perms is not None:
        cmd += ["--perms", str(args.perms)]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


def cmd_hyp_071_exit_value(args) -> None:
    """Run the HYP-071 tabular exit value function (scripts/research/hyp_071_exit_value_function.py)."""
    cmd = [sys.executable, str(ROOT / "scripts" / "research" / "hyp_071_exit_value_function.py")]
    if getattr(args, "standalone", False):
        cmd += ["--standalone"]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


def _read_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def cmd_money(args) -> None:
    """One screen: 'are we making money?' — backtest proof + live forward curve."""
    print("=" * 60)
    print("  ALTA — ARE WE MAKING MONEY?")
    print("=" * 60)

    proof = PROOF_DIR / "backtest_equity_latest.json"
    if proof.exists():
        c = json.loads(proof.read_text())
        s = c.get("stats", {})
        wsharpe = s.get("portfolio_sharpe_weighted")
        verdict = ("INSTITUTIONAL ✓" if (wsharpe or 0) >= 1.5
                   else "VIABLE EDGE" if (wsharpe or 0) >= 0.8 else "WEAK / REVIEW")
        print(f"  BACKTEST PROOF — {c.get('label')}")
        print(f"    Portfolio Sharpe : {wsharpe}   → {verdict}   (target ≥1.5, viable ≥0.8)")
        print(f"    Total return     : {s.get('total_return_pct')}%   over {s.get('years')}y, n={s.get('n_trades')} trades")
        print(f"    Max drawdown     : {s.get('max_drawdown_pct')}%  |  win {round((s.get('win_rate') or 0) * 100, 1)}%  |  PF {s.get('profit_factor')}")
    else:
        print("  BACKTEST PROOF — none yet.  Run:  alta prove")
    print()

    rows = _read_jsonl(LIVE_EQUITY)
    cur, src = None, "cached"
    try:
        from sovereign.execution.oanda_bridge import OandaBridge
        cur, src = OandaBridge().get_account_summary(), "live"
    except Exception:
        if rows:
            r = rows[-1]
            cur = {"nav": r.get("nav"), "balance": r.get("balance"),
                   "unrealized_pl": r.get("unrealized_pl"), "open_trade_count": r.get("open_trade_count")}
    print("  LIVE PAPER (OANDA practice) — forward confirmation")
    if cur:
        print(f"    NAV now          : ${cur['nav']:,.0f}  ({src})  |  open trades: {cur.get('open_trade_count')}")
        if cur.get("unrealized_pl") is not None:
            print(f"    Unrealized P&L   : {cur['unrealized_pl']:+,.2f}")
    if rows:
        from sovereign.reporting.equity_curve import build_from_nav
        lc = build_from_nav([{"t": r["t"], "nav": r["nav"]} for r in rows], label="live")
        ls = lc["stats"]
        print(f"    Since {rows[0]['t'][:10]}: {ls.get('total_return_pct')}%  over {ls.get('n_snapshots')} snapshots  (maxDD {ls.get('max_drawdown_pct')}%)")
    else:
        print("    No live snapshots yet — the 2h pulse populates this forward.")
    print()
    print("  The live edge fires ~4-14×/yr by design — backtests are the fast proof.")
    print("=" * 60)


def cmd_doctor(args) -> None:
    """Is the machine alive and honest? Loops, kill switch, outcome loop, launchd."""
    from datetime import datetime, timezone
    print("=" * 60)
    print("  ALTA DOCTOR")
    print("=" * 60)

    st = ks.state()
    print(f"  Kill switch:  {'🧊 FROZEN (' + str(st.get('mode')) + ')' if st else '🟢 running'}")

    try:
        from sovereign.oracle.loop_health import check_all_loops
        h = check_all_loops()
        down = h.get("down", []) if isinstance(h, dict) else []
        print(f"  Loops down:   {down or 'none 🟢'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  loop_health unavailable: {type(exc).__name__}: {exc}")

    try:
        month = datetime.now(timezone.utc).strftime("%Y_%m")
        p = ROOT / "data" / "decision_logs" / f"decisions_{month}.jsonl"
        fx = [r for r in _read_jsonl(p) if r.get("system") == "FOREX"]
        opened = sum(1 for r in fx if r.get("outcome") is None)
        closed = sum(1 for r in fx if r.get("outcome") not in (None, "EXPIRED"))
        flag = "  ⚠ check update_outcome" if (closed == 0 and len(fx) > 0) else ""
        print(f"  Outcome loop (FOREX {month}): {closed} closed, {opened} open, {len(fx)} total{flag}")
    except Exception as exc:  # noqa: BLE001
        print(f"  outcome-loop check failed: {type(exc).__name__}: {exc}")

    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=10).stdout
        jobs = sorted({l.split()[-1] for l in out.splitlines()
                       if "com.alta" in l or "com.clawd" in l})
        print(f"  launchd jobs loaded: {len(jobs)}")
        for j in jobs:
            print(f"    • {j}")
    except Exception as exc:  # noqa: BLE001
        print(f"  launchctl unavailable: {type(exc).__name__}: {exc}")
    print("=" * 60)


def cmd_discover(args) -> None:
    """Run the edge-discovery pipeline (discovery feeds the gate)."""
    cmd = [sys.executable, str(ROOT / "scripts" / "discover.py"), "--track", args.track]
    if args.selfcheck:
        cmd.append("--selfcheck")
    if args.validate:
        cmd += ["--validate", args.validate]
    if args.perms is not None:
        cmd += ["--perms", str(args.perms)]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


def cmd_validate(args) -> None:
    """Run the pipeline validation gate (audit the machinery; read-only)."""
    cmd = [sys.executable, str(ROOT / "scripts" / "validate_pipeline.py")]
    if args.drift_compare:
        cmd.append("--drift-compare")
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


def cmd_bench(args) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "bench_throughput.py"),
           "--seconds", str(args.seconds), "--tiers", args.tiers]
    if args.cores:
        cmd += ["--cores", str(args.cores)]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="alta", description="Alta operator CLI — proof, money, health, discover, validate, bench, kill switch.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("prove", help="run the backtest proof engine (equity curve + verdict)")
    pv.add_argument("--start", default=None)
    pv.add_argument("--end", default=None)
    pv.add_argument("--pairs", nargs="*", default=None)
    pv.set_defaults(fn=cmd_prove)

    mn = sub.add_parser("money", aliases=["pnl"], help="one-screen 'are we making money?'")
    mn.set_defaults(fn=cmd_money)

    dr = sub.add_parser("doctor", help="loop/launchd health + outcome-loop + kill-switch")
    dr.set_defaults(fn=cmd_doctor)

    dc = sub.add_parser("discover", help="edge-discovery pipeline (discovery feeds the gate)")
    dc.add_argument("--track", default="forex-daily",
                    choices=["forex-daily", "regime", "nq-intraday", "intraday-fx"])
    dc.add_argument("--perms", type=int, default=None)
    dc.add_argument("--validate", default=None, metavar="CANDIDATE")
    dc.add_argument("--selfcheck", action="store_true")
    dc.set_defaults(fn=cmd_discover)

    vd = sub.add_parser("validate", help="run the pipeline validation gate (audit the machinery)")
    vd.add_argument("--drift-compare", action="store_true", help="24h drift test vs last snapshot")
    vd.set_defaults(fn=cmd_validate)

    ev = sub.add_parser("exit-policy-evolution",
                        help="HYP-067 GA exit-policy search (gated on HYP-066; --standalone to override)")
    ev.add_argument("--sign", action="store_true", help="freeze the prereg hash (run once)")
    ev.add_argument("--standalone", action="store_true", help="run as independent HYP-067")
    ev.add_argument("--pop", type=int, default=None, help="GA population size")
    ev.add_argument("--generations", type=int, default=None, help="GA generations")
    ev.add_argument("--perms", type=int, default=None, help="holdout permutations")
    ev.set_defaults(fn=cmd_exit_policy_evolution)

    hv = sub.add_parser("hyp-071",
                        help="HYP-071 tabular exit value function (reconcile-gated; independent study)")
    hv.add_argument("--standalone", action="store_true", help="(HYP-071 is independent; kept for symmetry)")
    hv.set_defaults(fn=cmd_hyp_071_exit_value)

    f = sub.add_parser("freeze", help="freeze the trading path")
    f.add_argument("reason", help="why (logged to data/agent/param_change_log.jsonl)")
    f.add_argument("--hard", action="store_true",
                   help="also block approve_edge.py (live-config mutation)")
    f.add_argument("--by", default="colin", help="who issued the freeze (default: colin)")
    f.set_defaults(fn=cmd_freeze)

    t = sub.add_parser("thaw", help="remove the freeze")
    t.add_argument("--by", default="colin")
    t.set_defaults(fn=cmd_thaw)

    s = sub.add_parser("status", help="show freeze state")
    s.set_defaults(fn=cmd_status)

    bn = sub.add_parser("bench", help="measure backtest throughput (backtests/sec) — documented + history-tracked")
    bn.add_argument("--seconds", type=float, default=1.5, help="time budget per measurement cell")
    bn.add_argument("--tiers", default="90bar,daily,5min,1min")
    bn.add_argument("--cores", type=int, default=None)
    bn.set_defaults(fn=cmd_bench)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
