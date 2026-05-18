"""
Live paper challenge runner — simulates a prop firm evaluation alongside
your real paper account using actual system signals.

Runs continuously. Every signal gets prop-firm risk sizing applied BEFORE
execution. EOD update happens automatically at 5pm ET via the scheduler.

State persists in data/propfirm/active_challenge.json.
History in data/propfirm/challenge_history.jsonl.

Run manually:
    python3 sovereign/propfirm/paper_challenge.py --status
    python3 sovereign/propfirm/paper_challenge.py --eod
    python3 sovereign/propfirm/paper_challenge.py --trade --r 2.5 --pair GBPUSD
    python3 sovereign/propfirm/paper_challenge.py --new --firm lucid --account 100000
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sovereign.propfirm.rules_engine import PropFirmRules

ROOT        = Path(__file__).resolve().parents[2]
STATE_PATH  = ROOT / "data" / "propfirm" / "active_challenge.json"
HISTORY_PATH = ROOT / "data" / "propfirm" / "challenge_history.jsonl"


def _save_state(rules: PropFirmRules, firm: str, account_size: float) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "firm": firm,
        "account_size": account_size,
        "balance": rules.balance,
        "drawdown_floor": rules.drawdown_floor,
        "peak_eod_balance": rules.peak_eod_balance,
        "trading_days": rules.trading_days,
        "trade_count": rules.trade_count,
        "day_open_balance": rules.day_open_balance,
        "is_active": rules.is_active,
        "outcome": rules.outcome,
        "profit_target": rules.profit_target,
        "max_dd": rules.max_dd,
        "risk_per_trade_pct": rules.risk_per_trade_pct,
        "min_trading_days": rules.min_trading_days,
        "trade_log": [
            {
                "trade_num": t.trade_num,
                "pair": t.pair,
                "direction": t.direction,
                "r_multiple": t.r_multiple,
                "pnl_dollars": round(t.pnl_dollars, 2),
                "balance_after": round(t.balance_after, 2),
                "blocked": t.blocked,
                "size_reduced": t.size_reduced,
                "note": t.note,
            }
            for t in rules.trade_log
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _load_state() -> Optional[tuple[PropFirmRules, str, float]]:
    if not STATE_PATH.exists():
        return None
    state = json.loads(STATE_PATH.read_text())
    if state.get("firm") == "mff":
        rules = PropFirmRules.mff(account_size=state["account_size"])
    else:
        rules = PropFirmRules.lucid(account_size=state["account_size"])

    rules.risk_per_trade_pct = state.get("risk_per_trade_pct", 0.0075)
    rules.balance             = state["balance"]
    rules.drawdown_floor      = state["drawdown_floor"]
    rules.peak_eod_balance    = state["peak_eod_balance"]
    rules.trading_days        = state["trading_days"]
    rules.trade_count         = state["trade_count"]
    rules.day_open_balance    = state["day_open_balance"]
    rules.is_active           = state["is_active"]
    rules.outcome             = state.get("outcome")
    return rules, state["firm"], state["account_size"]


def _archive_challenge(rules: PropFirmRules, firm: str) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "firm": firm,
        "account_size": rules.account_size,
        **rules.summary(),
    }
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def cmd_status(rules: PropFirmRules, firm: str, account_size: float) -> None:
    s = rules.summary()
    starting_dd = account_size * rules.max_dd
    target = account_size * (1 + rules.profit_target)
    to_target = target - rules.balance
    buffer_pct = rules.buffer_pct()

    if buffer_pct >= 0.40:
        status_icon = "🟢 ON_TRACK"
    elif buffer_pct >= 0.20:
        status_icon = "🟡 CAUTION"
    else:
        status_icon = "🔴 DANGER — halt new positions"

    print(f"\n{'='*55}")
    print(f"PAPER CHALLENGE — {firm.upper()} ${account_size:,.0f}")
    print(f"{'='*55}")
    print(f"Balance:          ${rules.balance:>12,.2f}")
    print(f"Target:           ${target:>12,.2f}  (need ${to_target:,.2f} more)")
    print(f"Floor:            ${rules.drawdown_floor:>12,.2f}")
    print(f"Buffer:           ${rules.balance - rules.drawdown_floor:>12,.2f}  "
          f"({buffer_pct*100:.0f}% of starting DD)")
    print(f"Trading days:     {rules.trading_days}")
    print(f"Trades taken:     {rules.trade_count}  "
          f"(blocked={s['trades_blocked']} reduced={s['trades_reduced']})")
    print(f"Return:           {s['return_pct']:+.2f}%")
    print(f"Status:           {status_icon}")

    if rules.is_passed():
        print(f"\n🎉 CHALLENGE PASSED! Ready to go live.")
    elif rules.is_bust():
        print(f"\n💀 CHALLENGE BUSTED. Starting fresh.")
    print()


def cmd_eod(rules: PropFirmRules, firm: str, account_size: float) -> None:
    day_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rec = rules.update_eod(day_str)

    icon = {"ON_TRACK": "🟢", "CAUTION": "🟡", "DANGER": "🔴", "PASSED": "🎉", "BUST": "💀"}.get(rec.status, "")
    print(f"\nEOD Update — Day {rec.day_num} ({day_str})")
    print(f"  Close:  ${rec.close_balance:,.2f}")
    print(f"  Floor:  ${rec.floor:,.2f}")
    print(f"  Buffer: ${rec.buffer:,.2f}  ({rec.buffer_pct_of_starting_dd*100:.0f}% of starting DD)")
    print(f"  Status: {icon} {rec.status}")

    if rec.status == "PASSED":
        print(f"\n🎉 CHALLENGE PASSED on day {rec.day_num}!")
        _archive_challenge(rules, firm)
        STATE_PATH.unlink(missing_ok=True)
        print("Starting new challenge automatically...")
        rules.open_challenge()
    elif rec.status == "BUST":
        print(f"\n💀 BUSTED on day {rec.day_num}.")
        _archive_challenge(rules, firm)
        STATE_PATH.unlink(missing_ok=True)
        print("Starting new challenge automatically...")
        rules.open_challenge()

    _save_state(rules, firm, account_size)


def cmd_trade(
    rules: PropFirmRules,
    firm: str,
    account_size: float,
    r_multiple: float,
    pair: str,
    direction: str,
    risk_pct: Optional[float],
) -> None:
    max_risk = rules.max_position_risk()
    risk_dollars = (risk_pct or rules.risk_per_trade_pct) * rules.balance
    buffer_before = rules.buffer_pct()

    rec = rules.apply_trade_pnl(
        r_multiple=r_multiple,
        pair=pair,
        direction=direction,
        requested_risk_pct=risk_pct or rules.risk_per_trade_pct,
    )

    outcome_icon = "✓" if rec.pnl_dollars >= 0 else "✗"
    print(f"\n{outcome_icon} Trade #{rec.trade_num} — {pair} {direction}")
    print(f"  R: {r_multiple:+.2f}  PnL: ${rec.pnl_dollars:+,.2f}")
    if rec.blocked:
        print(f"  ⚠ BLOCKED (no buffer remaining)")
    elif rec.size_reduced:
        print(f"  ⚠ SIZED DOWN: wanted ${risk_dollars:,.0f}, took ${rec.risk_dollars:,.0f}")
    print(f"  Balance: ${rec.balance_after:,.2f}")
    print(f"  Buffer:  ${rec.buffer_at_time:,.2f}  ({rules.buffer_pct()*100:.0f}% of starting DD)")

    if rules.outcome == "BUST":
        print(f"\n💀 BUSTED after this trade!")

    _save_state(rules, firm, account_size)


def _load_history() -> list:
    if not HISTORY_PATH.exists():
        return []
    records = []
    with open(HISTORY_PATH) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def cmd_history() -> None:
    records = _load_history()
    if not records:
        print("No completed challenges yet.")
        return
    passes  = sum(1 for r in records if r.get("outcome") == "PASSED")
    busts   = sum(1 for r in records if r.get("outcome") == "BUST")
    print(f"\nCompleted challenges: {len(records)}")
    print(f"  Passed: {passes}  Busted: {busts}  "
          f"Pass rate: {passes/len(records)*100:.0f}%")
    print()
    for r in records[-5:]:
        icon = "🎉" if r.get("outcome") == "PASSED" else "💀"
        print(f"  {icon} {r.get('completed_at','')[:10]}  "
              f"{r.get('outcome')}  {r.get('trading_days')}d  "
              f"{r.get('return_pct',0):+.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status",  action="store_true")
    parser.add_argument("--eod",     action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--trade",   action="store_true")
    parser.add_argument("--new",     action="store_true")
    parser.add_argument("--r",       type=float, help="R-multiple for --trade")
    parser.add_argument("--pair",    type=str, default="")
    parser.add_argument("--dir",     type=str, default="")
    parser.add_argument("--risk",    type=float, default=None, help="Risk pct override")
    parser.add_argument("--firm",    choices=["lucid","mff"], default="lucid")
    parser.add_argument("--account", type=float, default=100_000)
    args = parser.parse_args()

    if args.history:
        cmd_history()
    elif args.new:
        if args.firm == "mff":
            rules = PropFirmRules.mff(account_size=args.account)
        else:
            rules = PropFirmRules.lucid(account_size=args.account)
        rules.open_challenge()
        _save_state(rules, args.firm, args.account)
        print(f"New {args.firm.upper()} challenge started — ${args.account:,.0f} account")
        cmd_status(rules, args.firm, args.account)
    else:
        loaded = _load_state()
        if loaded is None:
            print("No active challenge. Run with --new to start one.")
        else:
            rules, firm, account_size = loaded
            if args.eod:
                cmd_eod(rules, firm, account_size)
            elif args.trade:
                if args.r is None:
                    print("--trade requires --r (R-multiple)")
                else:
                    cmd_trade(rules, firm, account_size, args.r, args.pair, args.dir, args.risk)
            else:
                cmd_status(rules, firm, account_size)
