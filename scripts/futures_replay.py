#!/usr/bin/env python3
"""
Nightly ES/NQ replay — "backtest like it's live tonight."

Drives the EXACT live scalp + ORB logic (sovereign/futures/scalp_strategy.py) over
historical 1-min bars, simulates bracket fills with the REAL cost model, and enforces
the prop-firm trailing-drawdown / daily-loss / consistency rules. Writes a morning-ready
report so you walk in tomorrow knowing how the machine would have traded.

What you backtest here is what futures_monitor.py trades — same module, same params.

Usage:
    python3.13 scripts/futures_replay.py                         # MES, yfinance, last 5d
    python3.13 scripts/futures_replay.py --instrument MNQ --source yf
    python3.13 scripts/futures_replay.py --day 2026-06-06 --bias auto
    python3.13 scripts/futures_replay.py --source ib --lookback 5d
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.futures import scalp_strategy as strat          # noqa: E402
from sovereign.futures import bar_feed as bf                   # noqa: E402
from sovereign.futures.config import futures_params, contract_spec, round_turn_cost_usd  # noqa: E402

BIAS_LOG = ROOT / "data" / "futures" / "bias_log.jsonl"
REPORT_DIR = ROOT / "data" / "futures"


# ── bias for a replay day ─────────────────────────────────────────────────────

def _logged_bias(day: str, instrument: str) -> dict | None:
    norm = {"MES": "ES", "MNQ": "NQ"}.get(instrument.upper(), instrument.upper())
    if not BIAS_LOG.exists():
        return None
    found = None
    for line in BIAS_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        rinst = {"MES": "ES", "MNQ": "NQ"}.get(r.get("instrument", "").upper(), r.get("instrument", "").upper())
        if r.get("date") == day and rinst == norm:
            found = r
    return found


def _day_bias(day_df, day: str, prior_close: float | None,
              instrument: str, mode: str) -> tuple[str, dict]:
    """Return (bias_dir, key_levels). mode: auto|long|short|neutral."""
    if mode in ("long", "short"):
        return mode.upper(), {}
    if mode == "neutral":
        return "NEUTRAL", {}
    # auto: prefer the logged bias for that date
    logged = _logged_bias(day, instrument)
    if logged and logged.get("bias") in ("LONG", "SHORT"):
        return logged["bias"], logged.get("key_levels", {})
    # proxy: first RTH bar vs prior session close
    if prior_close is not None and len(day_df) > 0:
        first_close = float(day_df["Close"].iloc[0])
        return ("LONG" if first_close >= prior_close else "SHORT"), {}
    return "NEUTRAL", {}


# ── fill simulation ───────────────────────────────────────────────────────────

def _check_exit(pos: dict, high: float, low: float) -> tuple[float | None, str | None]:
    """Conservative same-bar resolution: stop is assumed to fill before target."""
    if pos["direction"] == "LONG":
        if low <= pos["stop"]:
            return pos["stop"], "STOPPED"
        if high >= pos["target"]:
            return pos["target"], "T1_HIT"
    else:
        if high >= pos["stop"]:
            return pos["stop"], "STOPPED"
        if low <= pos["target"]:
            return pos["target"], "T1_HIT"
    return None, None


def simulate_session(day_df, day: str, bias_dir: str, key_levels: dict,
                     instrument: str, orb_size: str, on_event=None) -> dict:
    """Run the live logic over one RTH session. Returns trades + per-day stats.

    `on_event(kind, payload)` (optional) is called for kind in {"bar","entry","exit"}
    so a renderer (scripts/futures_player.py) can animate the SAME engine the silent
    batch report uses — the two can never diverge. Default None = silent (batch)."""
    p = futures_params()
    spec = contract_spec(instrument)
    dpp = spec["dollars_per_point"]
    orb_cfg = p["orb"]

    def _emit(kind, payload):
        if on_event is not None:
            on_event(kind, payload)

    feed = bf.ReplayBarFeed(day_df, warmup=2)
    n_bars = len(feed)
    trades: list[dict] = []
    realized = 0.0
    hwm = 0.0
    max_dd = 0.0
    day_low_equity = 0.0

    position: dict | None = None
    prev_ind = None
    last_entry_time = None
    trades_taken = 0
    session_r = 0.0
    orb_hi = orb_lo = None
    orb_taken = False

    def _close(pos, exit_price, reason, ts):
        nonlocal realized, hwm, max_dd, day_low_equity, session_r, last_entry_time, trades_taken
        pts = (exit_price - pos["entry"]) if pos["direction"] == "LONG" else (pos["entry"] - exit_price)
        gross = pts * dpp * pos["contracts"]
        cost = round_turn_cost_usd(instrument, pos["contracts"])
        net = gross - cost
        r = strat.compute_r(pos["entry"], pos["stop"], exit_price, pos["direction"])
        realized += net
        hwm = max(hwm, realized)
        max_dd = max(max_dd, hwm - realized)
        day_low_equity = min(day_low_equity, realized)
        session_r += r
        last_entry_time = ts
        trades_taken += 1
        trades.append({
            "day": day, "instrument": instrument, "setup": pos["setup"],
            "direction": pos["direction"], "entry": round(pos["entry"], 2),
            "stop": round(pos["stop"], 2), "target": round(pos["target"], 2),
            "exit": round(exit_price, 2), "exit_reason": reason,
            "contracts": pos["contracts"], "gross_usd": round(gross, 2),
            "cost_usd": round(cost, 2), "net_usd": round(net, 2),
            "r_realized": r, "entry_ts": pos["entry_ts"], "exit_ts": ts.isoformat(),
        })
        _emit("exit", trades[-1])

    for i, (ts, window) in enumerate(feed.stream(), start=1):
        bar = window.iloc[-1]
        high, low, close = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
        try:
            ind = strat.compute_indicators(window)
        except ValueError:
            continue

        _emit("bar", {"ts": ts, "i": i, "n": n_bars, "ind": ind, "bias": bias_dir,
                      "realized": realized, "session_r": session_r,
                      "in_position": position is not None})

        # 1) manage open position on this bar
        if position is not None:
            ex, reason = _check_exit(position, high, low)
            if ex is not None:
                _close(position, ex, reason, ts)
                position = None

        # 2) capture ORB range
        if orb_hi is None and len(window) >= orb_cfg["minutes"]:
            rng = strat.orb_range(window, orb_cfg["minutes"])
            if rng:
                orb_hi, orb_lo = rng

        # 3) new entries only when flat
        if position is None and bias_dir in ("LONG", "SHORT"):
            # ORB macro (once per session)
            if not orb_taken and orb_hi is not None:
                d = strat.orb_break(bias_dir, close, orb_hi, orb_lo, ind)
                if d:
                    contracts = orb_cfg["big_contracts"] if orb_size == "big" else orb_cfg["safe_contracts"]
                    tgt_pts = orb_cfg["big_target_points"] if orb_size == "big" else orb_cfg["safe_target_points"]
                    stop = orb_lo if d == "LONG" else orb_hi
                    target = close + tgt_pts if d == "LONG" else close - tgt_pts
                    position = {"direction": d, "entry": close, "stop": stop, "target": target,
                                "contracts": contracts, "setup": "ORB", "entry_ts": ts.isoformat()}
                    orb_taken = True
                    _emit("entry", position)
            # micro scalp
            if position is None and prev_ind is not None:
                sig = strat.micro_signal(bias_dir, ind, prev_ind, now=ts,
                                         last_entry_time=last_entry_time, trades_taken=trades_taken)
                if sig:
                    stop = strat.compute_stop(sig, close, key_levels.get("overnight_low"),
                                              key_levels.get("overnight_high"))
                    target = strat.target_from_rr(sig, close, stop)
                    position = {"direction": sig, "entry": close, "stop": stop, "target": target,
                                "contracts": 1, "setup": "MICRO", "entry_ts": ts.isoformat()}
                    _emit("entry", position)

        prev_ind = ind

    # EOD: close any runner at last close
    if position is not None:
        last_close = float(day_df["Close"].iloc[-1])
        _close(position, last_close, "EOD", day_df.index[-1])

    return {
        "day": day, "bias": bias_dir, "trades": trades,
        "net_usd": round(realized, 2), "max_drawdown_usd": round(max_dd, 2),
        "day_low_equity_usd": round(day_low_equity, 2), "n_trades": len(trades),
    }


# ── aggregate + prop verdict ──────────────────────────────────────────────────

def _aggregate(sessions: list[dict], instrument: str) -> dict:
    p = futures_params()["prop"]
    all_trades = [t for s in sessions for t in s["trades"]]
    n = len(all_trades)
    net_total = round(sum(s["net_usd"] for s in sessions), 2)
    wins = [t for t in all_trades if t["net_usd"] > 0]
    losses = [t for t in all_trades if t["net_usd"] <= 0]
    gross_win = sum(t["net_usd"] for t in wins)
    gross_loss = abs(sum(t["net_usd"] for t in losses)) or 1e-9
    total_r = round(sum(t["r_realized"] for t in all_trades), 3)

    # run-level trailing drawdown on the cumulative realized curve
    cum = 0.0
    hwm = 0.0
    run_dd = 0.0
    for s in sessions:
        cum += s["net_usd"]
        hwm = max(hwm, cum)
        run_dd = max(run_dd, hwm - cum)

    day_nets = [s["net_usd"] for s in sessions]
    worst_day = min(day_nets) if day_nets else 0.0
    best_day = max(day_nets) if day_nets else 0.0

    hit_target = net_total >= p["profit_target_usd"]

    # Hard rules (blow the account anytime): trailing DD + daily loss.
    breaches = []
    if run_dd >= p["trailing_drawdown_usd"]:
        breaches.append(f"TRAILING DD ${run_dd:.0f} >= ${p['trailing_drawdown_usd']:.0f}")
    if worst_day <= -p["daily_loss_limit_usd"]:
        breaches.append(f"DAILY LOSS ${worst_day:.0f} <= -${p['daily_loss_limit_usd']:.0f}")

    # Consistency only matters at payout — soft warning until the target is hit.
    consistency_ok = True
    warnings_ = []
    if net_total > 0 and best_day / net_total > p["consistency_pct"]:
        consistency_ok = False
        msg = f"CONSISTENCY: best day {best_day/net_total:.0%} > {p['consistency_pct']:.0%}"
        (breaches if hit_target else warnings_).append(msg)

    verdict = "PASS" if (hit_target and not breaches) else ("BREACH" if breaches else "IN_PROGRESS")

    return {
        "instrument": instrument,
        "sessions": len(sessions),
        "n_trades": n,
        "net_total_usd": net_total,
        "win_rate": round(len(wins) / n, 3) if n else 0.0,
        "profit_factor": round(min(gross_win / gross_loss, 20.0), 3) if losses else None,
        "total_r": total_r,
        "avg_r": round(total_r / n, 3) if n else 0.0,
        "run_trailing_dd_usd": round(run_dd, 2),
        "best_day_usd": round(best_day, 2),
        "worst_day_usd": round(worst_day, 2),
        "prop_profit_target_usd": p["profit_target_usd"],
        "prop_trailing_dd_usd": p["trailing_drawdown_usd"],
        "prop_progress_pct": round(net_total / p["profit_target_usd"] * 100, 1) if p["profit_target_usd"] else 0.0,
        "prop_breaches": breaches,
        "warnings": warnings_,
        "consistency_ok": consistency_ok,
        "verdict": verdict,
    }


def _print_report(agg: dict, sessions: list[dict]) -> None:
    G, R, Y, BD, RS = "\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m"
    v = agg["verdict"]
    vc = G if v == "PASS" else (R if v == "BREACH" else Y)
    print(f"\n{BD}{'═'*64}{RS}")
    print(f"  {BD}REPLAY REPORT — {agg['instrument']}  |  {agg['sessions']} session(s){RS}")
    print(f"{BD}{'═'*64}{RS}")
    pf = "∞" if agg["profit_factor"] is None else f"{agg['profit_factor']:.2f}"
    print(f"  Trades: {agg['n_trades']}   Win rate: {agg['win_rate']:.0%}   PF: {pf}")
    print(f"  Net P&L (after costs): {G if agg['net_total_usd']>=0 else R}${agg['net_total_usd']:.2f}{RS}"
          f"   Total R: {agg['total_r']:+.2f}   Avg R: {agg['avg_r']:+.3f}")
    print(f"  Run trailing DD: ${agg['run_trailing_dd_usd']:.2f}"
          f"   Best/Worst day: ${agg['best_day_usd']:.2f} / ${agg['worst_day_usd']:.2f}")
    print(f"\n  {BD}Prop check ({futures_params()['prop']['firm']}):{RS}")
    print(f"    Profit target: ${agg['prop_profit_target_usd']:.0f}  "
          f"(progress {agg['prop_progress_pct']:.0f}%)")
    print(f"    Trailing DD limit: ${agg['prop_trailing_dd_usd']:.0f}")
    if agg["prop_breaches"]:
        for b in agg["prop_breaches"]:
            print(f"    {R}✗ {b}{RS}")
    else:
        print(f"    {G}✓ no hard-rule breaches{RS}")
    for w in agg.get("warnings", []):
        print(f"    {Y}⚠ {w} (soft — only blocks at payout){RS}")
    print(f"\n  {BD}Verdict: {vc}{v}{RS}")
    print(f"\n  Per session:")
    for s in sessions:
        print(f"    {s['day']}  bias {s['bias']:<7}  trades {s['n_trades']:>2}  "
              f"net ${s['net_usd']:+8.2f}  maxDD ${s['max_drawdown_usd']:.2f}")
    print(f"{BD}{'═'*64}{RS}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Nightly ES/NQ replay (backtest like live)")
    ap.add_argument("--instrument", default="MES", choices=["MES", "MNQ"])
    ap.add_argument("--source", default="yf", choices=["yf", "ib"],
                    help="yf=yfinance fallback (runs tonight w/o IB), ib=IB Gateway bars")
    ap.add_argument("--day", default=None, help="Single ET day YYYY-MM-DD (else all available)")
    ap.add_argument("--lookback", default="5d", help="yfinance lookback window (1m caps ~7d)")
    ap.add_argument("--bias", default="auto", choices=["auto", "long", "short", "neutral"])
    ap.add_argument("--orb-size", default="safe", choices=["safe", "big"])
    ap.add_argument("--no-log", action="store_true", help="Don't write the JSON report")
    args = ap.parse_args()

    print(f"Loading {args.instrument} 1-min history via {args.source}...", end=" ", flush=True)
    # Always load the full window so the prior session is present for prior-close/bias
    # context; --day only SELECTS which session(s) to replay.
    df = bf.load_history(args.instrument, source=args.source, day=None, lookback=args.lookback)
    if df is None or len(df) == 0:
        print(f"\n  No data returned. (yfinance 1m only covers ~7 days; try --source ib.)")
        sys.exit(1)
    print(f"done. {len(df)} bars.")

    all_days = bf.session_days(df)
    days = [args.day] if args.day else all_days
    sessions = []
    prior_close = None
    for day in all_days:                  # walk all for prior_close continuity
        day_df = df[df.index.tz_convert(bf.ET).strftime("%Y-%m-%d") == day]
        if len(day_df) < 3:
            prior_close = float(day_df["Close"].iloc[-1]) if len(day_df) else prior_close
            continue
        if day in days:
            bias_dir, key_levels = _day_bias(day_df, day, prior_close, args.instrument, args.bias)
            sessions.append(simulate_session(day_df, day, bias_dir, key_levels,
                                             args.instrument, args.orb_size))
        prior_close = float(day_df["Close"].iloc[-1])

    if not sessions:
        print("  No complete sessions to replay.")
        sys.exit(1)

    agg = _aggregate(sessions, args.instrument)
    _print_report(agg, sessions)

    if not args.no_log:
        tag = args.day or f"{days[0]}_to_{days[-1]}"
        out = REPORT_DIR / f"replay_report_{args.instrument}_{tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "instrument": args.instrument, "source": args.source,
            "bias_mode": args.bias, "orb_size": args.orb_size,
            "summary": agg, "sessions": sessions,
        }
        out.write_text(json.dumps(report, indent=2))
        print(f"  Report → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
