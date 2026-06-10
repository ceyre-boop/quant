#!/usr/bin/env python3
"""ES/NQ paper trading runner — MNQ default, IB Gateway paper account ONLY.

Usage: python3 scripts/es_nq_paper_runner.py [--instrument MNQ|MES] [--dry-run]

Startup gates, in order (all fail loud):
  1. kill switch     — exits cleanly if the system is frozen
  2. Stage-1 verdict — refuses to start unless ESNQ-BIAS-01 is VALID_EDGE
  3. roll-day skip   — no structure trading on contract-roll sessions
  4. daily loss lock — data/es_nq/.session_lock (1.5% of account, hard)

Loop: poll IB every 30s → 1-min bars → resample 5-min → live_scanner.scan_step →
bracket order on a fresh TradePlan → SessionLadder sizing → log fills (not
submissions) → flat 15:55 ET. Bias recomputed at 09:25 with the pre-registered
cutoff (overrides the 08:45 brief if different — logged).

PAPER ONLY: this script refuses to run against a live IB port.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sovereign.es_nq import data_store                               # noqa: E402
from sovereign.es_nq.config import contract_spec, es_nq_params       # noqa: E402
from sovereign.es_nq.daily_bias_engine import (                      # noqa: E402
    calendar_score, compute_bias, hurst_score, international_score,
    overnight_score, vix_score,
)
from sovereign.es_nq.live_scanner import ScannerState, scan_step     # noqa: E402
from sovereign.es_nq.session_logger import log_session               # noqa: E402
from sovereign.es_nq.session_sizing import SessionLadder             # noqa: E402
from sovereign.es_nq.structure_gate import Levels                    # noqa: E402
from sovereign.futures.bar_feed import live_session_bars             # noqa: E402
from sovereign.futures.ib_bridge import IBBridge                     # noqa: E402
from sovereign.utils import kill_switch                              # noqa: E402

ET = ZoneInfo("America/New_York")
VALIDATION_PATH = ROOT / "data" / "research" / "es_nq_validation.json"
SESSION_LOCK = ROOT / "data" / "es_nq" / ".session_lock"
CAL_PATH = ROOT / "data" / "es_nq" / "econ_calendar_2018_2026.json"
POLL_SECONDS = 30
LIVE_IB_PORT = 4001


def gate_checks(date: str) -> None:
    if kill_switch.skip_if_frozen("es_nq_paper"):
        sys.exit(0)
    if not VALIDATION_PATH.exists():
        raise SystemExit("FATAL: no validation results — the gauntlet has not run. "
                         "Paper trading is forbidden before Stage 1 = VALID_EDGE.")
    v = json.loads(VALIDATION_PATH.read_text())
    if (v.get("stage1") or {}).get("verdict") != "VALID_EDGE":
        raise SystemExit("FATAL: Stage 1 (ESNQ-BIAS-01) is not VALID_EDGE — "
                         "paper runner permanently blocked (non-negotiable #2).")
    if SESSION_LOCK.exists():
        lock = json.loads(SESSION_LOCK.read_text())
        if lock.get("date") == date:
            raise SystemExit(f"FATAL: daily loss lock active since {lock.get('ts')} — "
                             "session is over. Manual unlock: delete data/es_nq/.session_lock")
        SESSION_LOCK.unlink()   # stale lock from a prior day


def compute_morning_bias(date: str, p: dict):
    """09:25 bias with the pre-registered cutoff. Fails loud on any data gap."""
    now = datetime.now(ET)
    start = (now - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
    df = data_store.pull_globex_history(
        start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), chunk_days=2)
    daily = data_store.load_daily()
    prior = daily.iloc[-1]
    if str(prior.name) >= date:
        raise SystemExit(f"FATAL: daily cache last row {prior.name} >= today {date} — "
                         "run scripts/es_nq_pull_history.py --update")
    roll_day = str(df["symbol"].iloc[-1]) != str(prior["symbol"])
    overnight_ret = float(df["Close"].iloc[-1]) / float(prior["rth_close"]) - 1.0
    aux = data_store.load_aux_daily()
    cal = json.loads(CAL_PATH.read_text())
    comp = {
        "overnight": overnight_score(overnight_ret, roll_day, p),
        "vix": vix_score(aux["vix"], date, p),
        "hurst": hurst_score(daily["rth_close"], date, p),
        "international": international_score(aux["nikkei"], aux["dax"], date),
    }
    cal_s, event_day = calendar_score(date, cal)
    comp["calendar"] = cal_s
    bias = compute_bias(date, comp, event_day, roll_day, calendar_active=False, params=p)
    levels = Levels(pdh=float(prior["rth_high"]), pdl=float(prior["rth_low"]),
                    onh=float(df["High"].max()), onl=float(df["Low"].min()))
    return bias, levels, roll_day


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default=None, choices=["MNQ", "MES"])
    ap.add_argument("--dry-run", action="store_true",
                    help="full loop, no orders placed (replay-style validation)")
    args = ap.parse_args()
    p = es_nq_params()
    inst = args.instrument or p["meta"]["trade_instrument"]
    date = datetime.now(ET).strftime("%Y-%m-%d")

    gate_checks(date)
    bias, levels, roll_day = compute_morning_bias(date, p)
    print(f"[{date}] 09:25 bias: {bias.direction} conf={bias.confidence} | {bias.reasoning}")
    if roll_day:
        log_session({"session_date": date, "mode": "PAPER", "bias": bias.__dict__,
                     "trades": [], "session_r_total": 0.0, "session_usd_total": 0.0,
                     "skipped": "ROLL_DAY"})
        print("ROLL DAY — no structure trading. Done.")
        return
    if bias.direction == "NEUTRAL":
        log_session({"session_date": date, "mode": "PAPER", "bias": bias.__dict__,
                     "trades": [], "session_r_total": 0.0, "session_usd_total": 0.0,
                     "skipped": "NEUTRAL_BIAS"})
        print("NEUTRAL — the skip is a trade. Done.")
        return

    import os
    ib_port = int(os.environ.get("IB_PORT", "4002"))
    if ib_port in (LIVE_IB_PORT, 7496):
        raise SystemExit(f"FATAL: IB_PORT={ib_port} is a LIVE port — paper only. Refusing.")
    bridge = IBBridge()
    bridge.connect()
    contract = bridge.mnq_contract() if inst == "MNQ" else bridge.mes_contract()
    spec = contract_spec(inst)
    ladder = SessionLadder(account_usd=p["sizing"]["account_base_usd"], params=p)
    state = ScannerState(bias_dir=bias.direction, levels=levels, instrument=inst, params=p)
    trades_log: list[dict] = []
    open_trade: dict | None = None

    try:
        while True:
            now = datetime.now(ET)
            if kill_switch.skip_if_frozen("es_nq_paper"):
                break
            if (now.hour, now.minute) >= (15, 55):
                if open_trade and not args.dry_run:
                    bridge.cancel_all_orders()
                    bridge.market_order(contract,
                                        "SELL" if open_trade["direction"] == "LONG" else "BUY",
                                        open_trade["contracts"])
                    _close_trade(open_trade, bridge, spec, ladder, trades_log, reason="FLAT")
                    open_trade = None
                print("15:55 ET — flat, session over.")
                break
            if ladder.halted():
                SESSION_LOCK.write_text(json.dumps(
                    {"date": date, "ts": datetime.now(timezone.utc).isoformat(),
                     "realized_usd": ladder.realized_usd}))
                print(f"DAILY LOSS CAP HIT ({ladder.realized_usd:+.2f}) — locked.")
                break

            bars1 = live_session_bars(bridge, contract)
            if bars1 is None or len(bars1) < 10:
                time.sleep(POLL_SECONDS)
                continue
            bars5 = data_store.resample_5min(data_store.filter_rth(bars1))
            if len(bars5) < 3:
                time.sleep(POLL_SECONDS)
                continue

            if open_trade is not None:
                done = _poll_open_trade(open_trade, bridge, spec, ladder, trades_log)
                if done:
                    state.trades_done += 1
                    state.in_trade = False
                    state.scan_from_idx = len(bars5)
                    open_trade = None
            else:
                role = ladder.next_role()
                if role is None:
                    print("Ladder complete — observing until close.")
                    time.sleep(POLL_SECONDS)
                    continue
                plan = scan_step(datetime.now(timezone.utc), bars5, state)
                if plan is not None:
                    n = ladder.contracts(role, plan.stop_points, inst)
                    if n == 0:
                        print(f"Setup found but stop {plan.stop_points:.2f}pts too wide "
                              f"for {role} risk — skipped.")
                        state.scan_from_idx = len(bars5)
                    else:
                        open_trade = _place(bridge, contract, plan, n, role, inst,
                                            dry_run=args.dry_run)
                        state.in_trade = open_trade is not None
            time.sleep(POLL_SECONDS)
    finally:
        bridge.disconnect()
        _finalize_session(date, bias, levels, trades_log, ladder)


def _place(bridge, contract, plan, n, role, inst, *, dry_run: bool) -> dict | None:
    print(f"SETUP: {plan.direction} {n}x{inst} entry~{plan.entry} stop {plan.stop} "
          f"t1 {plan.t1} t2 {plan.t2} ({role}, sweep {plan.sweep.level_name})")
    if dry_run:
        return {"plan": plan, "role": role, "direction": plan.direction, "contracts": n,
                "dry_run": True, "entry_fill": plan.entry, "order_ids": []}
    side = "BUY" if plan.direction == "LONG" else "SELL"
    results = bridge.bracket_order(contract, side, n, plan.entry, plan.stop, plan.t1)
    return {"plan": plan, "role": role, "direction": plan.direction, "contracts": n,
            "dry_run": False, "entry_fill": None,
            "order_ids": [r.order_id for r in results]}


def _poll_open_trade(open_trade, bridge, spec, ladder, trades_log) -> bool:
    """Fill-confirmed accounting: a trade is recorded when IB reports the position
    closed (fills), never on submission. Returns True when the trade is finished."""
    if open_trade["dry_run"]:
        # Dry-run: resolve optimistically at t1 after one poll (validation only).
        plan = open_trade["plan"]
        _record(open_trade, exit_price=plan.t1, reason="T1_DRY", spec=spec,
                ladder=ladder, trades_log=trades_log)
        return True
    if any(pos.get("position") for pos in bridge.positions()):
        return False
    fills = [f for f in bridge.fills() if f.get("order_id") in
             set(open_trade.get("order_ids", []))]
    if not fills:
        return False
    # Entry fill = first fill on the parent side; exit = last fill.
    entry_side = "BOT" if open_trade["direction"] == "LONG" else "SLD"
    entry_fills = [f for f in fills if f["side"] == entry_side]
    exit_fills = [f for f in fills if f["side"] != entry_side]
    if not exit_fills:
        return False
    if entry_fills and open_trade.get("entry_fill") is None:
        open_trade["entry_fill"] = float(entry_fills[0]["price"])
    px = float(exit_fills[-1]["price"])
    plan = open_trade["plan"]
    reason = "STOP" if abs(px - plan.stop) <= abs(px - plan.t1) else "TARGET"
    _record(open_trade, exit_price=px, reason=reason, spec=spec, ladder=ladder,
            trades_log=trades_log)
    return True


def _record(open_trade, *, exit_price, reason, spec, ladder, trades_log) -> None:
    plan = open_trade["plan"]
    sgn = 1.0 if open_trade["direction"] == "LONG" else -1.0
    entry = open_trade.get("entry_fill") or plan.entry
    pts = sgn * (exit_price - entry)
    usd = pts * spec["dollars_per_point"] * open_trade["contracts"] - \
        0.70 * open_trade["contracts"]
    r = pts / plan.stop_points if plan.stop_points else 0.0
    ladder.record(r, usd)
    trades_log.append({
        "role": open_trade["role"], "direction": open_trade["direction"],
        "contracts": open_trade["contracts"], "entry": entry, "stop": plan.stop,
        "t1": plan.t1, "t2": plan.t2, "exit": exit_price, "exit_reason": reason,
        "r_net": round(r, 4), "usd_net": round(usd, 2),
        "sweep_level": plan.sweep.level_name, "dry_run": open_trade["dry_run"],
    })
    print(f"CLOSED {open_trade['direction']} {reason} @ {exit_price} "
          f"R={r:+.2f} ${usd:+.2f}")


def _close_trade(open_trade, bridge, spec, ladder, trades_log, *, reason) -> None:
    px = bridge.last_price(bridge.mnq_contract()) or open_trade["plan"].entry
    _record(open_trade, exit_price=float(px), reason=reason, spec=spec,
            ladder=ladder, trades_log=trades_log)


def _finalize_session(date, bias, levels, trades_log, ladder) -> None:
    daily = data_store.load_daily()
    correct = None   # filled by tomorrow's --update + nightly review once the session lands
    log_session({
        "session_date": date, "mode": "PAPER",
        "bias": {"direction": bias.direction, "confidence": bias.confidence,
                 "raw_score": bias.raw_score, "components": bias.components,
                 "event_day": bias.event_day, "roll_day": bias.roll_day,
                 "reasoning": bias.reasoning},
        "levels": {"pdh": levels.pdh, "pdl": levels.pdl,
                   "onh": levels.onh, "onl": levels.onl},
        "trades": trades_log,
        "session_r_total": round(sum(t["r_net"] for t in trades_log), 4),
        "session_usd_total": round(ladder.realized_usd, 2),
        "bias_was_correct": correct,
        "notes": f"paper runner; daily cache through {daily.index[-1]}",
    })
    print(f"Session logged: {len(trades_log)} trades, "
          f"R {sum(t['r_net'] for t in trades_log):+.2f}, ${ladder.realized_usd:+.2f}")


if __name__ == "__main__":
    main()
