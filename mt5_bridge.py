#!/usr/bin/env python3
"""MT5 execution bridge (DEMO-only) — CLI (TICK-056, spec: specs/mt5_bridge.md).

Turns a decoupled `order_intent` JSON contract into an MT5 order on a DEMO account,
with a human in the loop, without touching frozen execution code, and physically
incapable of routing live without an explicit, logged unlock.

Commands
  --selftest              connect, print account, ASSERT DEMO, route NOTHING
  --stage <intent_id>     load + validate intent, verify DEMO, print order card,
                          write a pending staging record. Does NOT route.
  --route <intent_id>     re-verify DEMO immediately before order_send. Requires
             --approve    the explicit --approve flag; without it this is a dry-run
                          that re-prints the card and exits 0 WITHOUT routing.

Isolation: imports NOTHING from the frozen execution path. Enforced by
tests/test_mt5_bridge.py::test_ast_isolation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import yaml

from sovereign.execution.mt5 import TRADE_MODE_NAMES
from sovereign.execution.mt5.connector import Connector, ConnectorError, MT5Connector
from sovereign.execution.mt5.contract import IntentError, OrderIntent
from sovereign.execution.mt5.guard import GuardError, assert_routable

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config" / "mt5.yml"

# MT5 constants (mirrored so the CLI reads without the Windows package present).
TRADE_ACTION_DEAL = 1
ORDER_TIME_GTC = 0
ORDER_FILLING_IOC = 1
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
TRADE_RETCODE_DONE = 10009


class BridgeError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Config + I/O                                                                 #
# --------------------------------------------------------------------------- #

def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise BridgeError(f"config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _p(cfg: dict, key: str, default: str) -> Path:
    return REPO_ROOT / cfg.get("paths", {}).get(key, default)


def already_routed(routed_ledger: Path, intent_id: str) -> bool:
    """True if intent_id appears in the append-only idempotency ledger."""
    if not routed_ledger.exists():
        return False
    for line in routed_ledger.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("intent_id") == intent_id:
            return True
    return False


def append_routed(routed_ledger: Path, record: dict) -> None:
    routed_ledger.parent.mkdir(parents=True, exist_ok=True)
    with routed_ledger.open("a") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- #
# Account helpers                                                             #
# --------------------------------------------------------------------------- #

def _attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def connect_and_verify(connector: Connector, cfg: dict) -> tuple[Any, str]:
    """Initialize the connector, fetch account, assert routable (DEMO). Returns
    (account, trade_mode_name). Raises loud on any failure."""
    connector.initialize()
    account = connector.account_info()
    unlock_path = _p(cfg, "live_unlock", "data/execution/mt5_LIVE_UNLOCK.json")
    mode_name = assert_routable(account, unlock_path=unlock_path)
    return account, mode_name


# --------------------------------------------------------------------------- #
# Commands                                                                     #
# --------------------------------------------------------------------------- #

def cmd_selftest(connector: Connector, cfg: dict) -> int:
    print("== MT5 bridge selftest ==")
    account, mode_name = connect_and_verify(connector, cfg)
    print(f"  login      : {_attr(account, 'login')}")
    print(f"  server     : {_attr(account, 'server')}")
    print(f"  trade_mode : {_attr(account, 'trade_mode')} ({mode_name})")
    print(f"  balance    : {_attr(account, 'balance')} {_attr(account, 'currency')}")
    if mode_name != "DEMO":
        # Only reachable if an explicit unlock was created — surface it loudly.
        print("  WARNING: account is NOT demo but an unlock is present.", file=sys.stderr)
    print("  ROUTED     : nothing (selftest never places an order).")
    return 0


def _load_intent(cfg: dict, intent_id: str) -> OrderIntent:
    intents_dir = _p(cfg, "intents_dir", "data/execution/mt5_intents")
    intent = OrderIntent.load(intents_dir / f"{intent_id}.json")
    lots = cfg.get("lots", {})
    symbol_map = cfg.get("symbol_map") or None
    allowed = set(symbol_map.keys()) if symbol_map else None
    intent.validate(
        min_lot=float(lots.get("min_lot", 0.01)),
        max_lot=float(lots.get("max_lot", 100.0)),
        allowed_symbols=allowed,
    )
    return intent


def cmd_stage(connector: Connector, cfg: dict, intent_id: str) -> int:
    intent = _load_intent(cfg, intent_id)

    routed_ledger = _p(cfg, "routed_ledger", "data/execution/mt5_routed.jsonl")
    if already_routed(routed_ledger, intent.intent_id):
        raise BridgeError(
            f"intent {intent.intent_id} already appears in {routed_ledger.name} — "
            f"refusing to re-stage (idempotency)."
        )

    account, mode_name = connect_and_verify(connector, cfg)
    print(intent.order_card(
        login=_attr(account, "login"),
        server=_attr(account, "server"),
        trade_mode_name=mode_name,
    ))

    pending_dir = _p(cfg, "pending_dir", "data/execution/mt5_pending")
    pending_dir.mkdir(parents=True, exist_ok=True)
    staging = {
        "intent_id": intent.intent_id,
        "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "account_login": _attr(account, "login"),
        "account_server": _attr(account, "server"),
        "trade_mode": _attr(account, "trade_mode"),
        "trade_mode_name": mode_name,
        "symbol": intent.symbol,
        "side": intent.side,
        "volume_lots": intent.volume_lots,
    }
    (pending_dir / f"{intent.intent_id}.json").write_text(json.dumps(staging, indent=2))
    print(f"\nStaged → {pending_dir / (intent.intent_id + '.json')}")
    print("Review, then route with:")
    print(f"  python mt5_bridge.py --route {intent.intent_id} --approve")
    return 0


def _build_order_request(cfg: dict, intent: OrderIntent, connector: Connector) -> dict:
    symbol_map = cfg.get("symbol_map") or {}
    broker_symbol = symbol_map.get(intent.symbol, intent.symbol)
    order_type = ORDER_TYPE_BUY if intent.side == "BUY" else ORDER_TYPE_SELL

    tick = connector.symbol_tick(broker_symbol)
    if tick is None:
        raise BridgeError(
            f"no tick for '{broker_symbol}' — cannot price a market order. "
            f"Is the symbol in Market Watch and the market open?"
        )
    price = tick.ask if intent.side == "BUY" else tick.bid

    request = {
        "action": TRADE_ACTION_DEAL,
        "symbol": broker_symbol,
        "volume": float(intent.volume_lots),
        "type": order_type,
        "price": price,
        "sl": float(intent.sl_price),
        "deviation": int(intent.max_slippage_points),
        "magic": int(intent.magic),
        "comment": intent.comment,
        "type_time": ORDER_TIME_GTC,
        "type_filling": ORDER_FILLING_IOC,
    }
    if intent.tp_price is not None:
        request["tp"] = float(intent.tp_price)
    return request


def cmd_route(connector: Connector, cfg: dict, intent_id: str, approve: bool) -> int:
    intent = _load_intent(cfg, intent_id)

    routed_ledger = _p(cfg, "routed_ledger", "data/execution/mt5_routed.jsonl")
    if already_routed(routed_ledger, intent.intent_id):
        raise BridgeError(
            f"intent {intent.intent_id} already routed (in {routed_ledger.name}) — "
            f"refusing to route twice (idempotency)."
        )

    # Guard runs AGAIN here, immediately before order_send (TOCTOU-safe, spec §7).
    account, mode_name = connect_and_verify(connector, cfg)
    print(intent.order_card(
        login=_attr(account, "login"),
        server=_attr(account, "server"),
        trade_mode_name=mode_name,
    ))

    if not approve:
        print("\nDRY-RUN: no --approve flag. Nothing routed. Exit 0.")
        print(f"To route to the {mode_name} account, re-run with --approve.")
        return 0

    request = _build_order_request(cfg, intent, connector)
    print(f"\nRouting to {mode_name} account (login {_attr(account, 'login')})…")
    result = connector.order_send(request)

    retcode = _attr(result, "retcode")
    record = {
        "intent_id": intent.intent_id,
        "routed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "account_login": _attr(account, "login"),
        "account_server": _attr(account, "server"),
        "trade_mode": _attr(account, "trade_mode"),
        "symbol": request["symbol"],
        "side": intent.side,
        "volume_lots": intent.volume_lots,
        "request_price": request["price"],
        "retcode": retcode,
        "order": _attr(result, "order"),
        "deal": _attr(result, "deal"),
        "fill_price": _attr(result, "price"),
        "comment": _attr(result, "comment"),
    }
    append_routed(routed_ledger, record)

    if retcode != TRADE_RETCODE_DONE:
        # Report verbatim; NO auto-retry (could double-fire) — spec §6.4/§8.
        print(f"NON-DONE result (retcode={retcode}): {_attr(result, 'comment')}", file=sys.stderr)
        print(f"Recorded to {routed_ledger.name}. No retry attempted.")
        return 2

    print(f"DONE. order={record['order']} deal={record['deal']} "
          f"fill_price={record['fill_price']}")
    print(f"Recorded → {routed_ledger}")
    return 0


# --------------------------------------------------------------------------- #
# Entry                                                                        #
# --------------------------------------------------------------------------- #

def build_connector(cfg: dict) -> Connector:
    """Construct the REAL connector. Off-Windows this fails loud on initialize()."""
    conn = cfg.get("connection", {})
    import os
    login = os.environ.get("ALTA_MT5_LOGIN")
    return MT5Connector(
        login=int(login) if login else None,
        password=os.environ.get("ALTA_MT5_PASSWORD"),
        server=os.environ.get("ALTA_MT5_SERVER") or conn.get("server"),
        terminal_path=conn.get("terminal_path"),
    )


def main(argv: Optional[list[str]] = None, connector: Optional[Connector] = None) -> int:
    parser = argparse.ArgumentParser(description="MT5 execution bridge (DEMO-only)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selftest", action="store_true", help="connect, assert DEMO, route nothing")
    group.add_argument("--stage", metavar="INTENT_ID", help="stage an intent (no routing)")
    group.add_argument("--route", metavar="INTENT_ID", help="route a staged intent")
    parser.add_argument("--approve", action="store_true", help="explicit approval; required to route")
    args = parser.parse_args(argv)

    try:
        cfg = load_config()
        conn = connector if connector is not None else build_connector(cfg)
        try:
            if args.selftest:
                return cmd_selftest(conn, cfg)
            if args.stage:
                return cmd_stage(conn, cfg, args.stage)
            if args.route:
                return cmd_route(conn, cfg, args.route, args.approve)
        finally:
            try:
                conn.shutdown()
            except Exception:
                pass
    except (BridgeError, IntentError, GuardError, ConnectorError) as e:
        print(f"ABORT: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
