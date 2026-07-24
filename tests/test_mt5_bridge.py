"""Unit tests for the MT5 execution bridge (TICK-056, spec §10).

All run WITHOUT a terminal (MockConnector). Covers: demo guard, live-unlock gate,
contract validation, idempotency, AST isolation from the frozen execution path.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import mt5_bridge
from sovereign.execution.mt5 import (
    ACCOUNT_TRADE_MODE_CONTEST,
    ACCOUNT_TRADE_MODE_DEMO,
    ACCOUNT_TRADE_MODE_REAL,
)
from sovereign.execution.mt5.connector import MockAccount, MockConnector, Tick
from sovereign.execution.mt5.contract import IntentError, OrderIntent
from sovereign.execution.mt5.guard import (
    LiveAccountError,
    NoConnectionError,
    assert_routable,
    live_routing_permitted,
    load_unlock,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

FROZEN_MODULES = [
    "forex_exit_manager",
    "decide_exit",
    "execution.harness",
    "carry_engine",
    "ict.pipeline",
]


def valid_intent_dict(**over):
    d = {
        "intent_id": "2026-07-29T14:31:00Z_EURUSD_SHORT",
        "created_at": "2026-07-29T14:31:00Z",
        "symbol": "EURUSD",
        "side": "SELL",
        "order_type": "MARKET",
        "volume_lots": 0.50,
        "sl_price": 1.0925,
        "tp_price": 1.0840,
        "comment": "v015_carry_HYP045",
        "magic": 5015,
        "source_strategy": "carry_v015",
        "signal_hash": "abc123",
        "max_slippage_points": 20,
    }
    d.update(over)
    return d


# --------------------------------------------------------------------------- #
# Demo-vs-live guard (spec §7)                                                  #
# --------------------------------------------------------------------------- #

def test_guard_allows_demo():
    acct = MockAccount(trade_mode=ACCOUNT_TRADE_MODE_DEMO)
    assert assert_routable(acct, env={}, unlock_path="/nonexistent") == "DEMO"


def test_guard_aborts_real():
    acct = MockAccount(trade_mode=ACCOUNT_TRADE_MODE_REAL)
    with pytest.raises(LiveAccountError):
        assert_routable(acct, env={}, unlock_path="/nonexistent")


def test_guard_aborts_contest():
    acct = MockAccount(trade_mode=ACCOUNT_TRADE_MODE_CONTEST)
    with pytest.raises(LiveAccountError):
        assert_routable(acct, env={}, unlock_path="/nonexistent")


def test_guard_aborts_none_account():
    with pytest.raises(NoConnectionError):
        assert_routable(None, env={}, unlock_path="/nonexistent")


def test_guard_aborts_unknown_trade_mode():
    acct = MockAccount(trade_mode=99)
    with pytest.raises(LiveAccountError):
        assert_routable(acct, env={}, unlock_path="/nonexistent")


# --------------------------------------------------------------------------- #
# Live-unlock gate — BOTH env AND file required (spec §7)                       #
# --------------------------------------------------------------------------- #

def _write_unlock(tmp_path, **over):
    data = {
        "rationale": "manual live test",
        "operator_signature": "colin",
        "authorized_at": "2026-07-29T00:00:00Z",
    }
    data.update(over)
    p = tmp_path / "mt5_LIVE_UNLOCK.json"
    p.write_text(json.dumps(data))
    return p


def test_live_gate_needs_both_env_and_file(tmp_path):
    unlock = _write_unlock(tmp_path)
    # only file, no env
    assert live_routing_permitted(env={}, unlock_path=unlock) is False
    # only env, no file
    assert live_routing_permitted(env={"ALTA_MT5_ALLOW_LIVE": "1"},
                                  unlock_path=tmp_path / "absent.json") is False
    # both
    assert live_routing_permitted(env={"ALTA_MT5_ALLOW_LIVE": "1"}, unlock_path=unlock) is True


def test_live_gate_default_unreachable():
    # Neither env flag nor file → live never permitted.
    assert live_routing_permitted(env={}, unlock_path="/nonexistent") is False


def test_guard_real_still_aborts_with_only_env(tmp_path):
    acct = MockAccount(trade_mode=ACCOUNT_TRADE_MODE_REAL)
    with pytest.raises(LiveAccountError):
        assert_routable(acct, env={"ALTA_MT5_ALLOW_LIVE": "1"},
                        unlock_path=tmp_path / "absent.json")


def test_guard_real_still_aborts_with_only_file(tmp_path):
    acct = MockAccount(trade_mode=ACCOUNT_TRADE_MODE_REAL)
    unlock = _write_unlock(tmp_path)
    with pytest.raises(LiveAccountError):
        assert_routable(acct, env={}, unlock_path=unlock)


def test_guard_real_permitted_with_both(tmp_path):
    acct = MockAccount(trade_mode=ACCOUNT_TRADE_MODE_REAL)
    unlock = _write_unlock(tmp_path)
    # Explicit, logged unlock present → returns mode name instead of raising.
    assert assert_routable(acct, env={"ALTA_MT5_ALLOW_LIVE": "1"},
                           unlock_path=unlock) == "REAL"


def test_malformed_unlock_treated_as_absent(tmp_path):
    # Missing a required field → NOT a valid unlock.
    p = _write_unlock(tmp_path, operator_signature="")
    assert load_unlock(p) is None
    p.write_text("{ not json")
    assert load_unlock(p) is None


# --------------------------------------------------------------------------- #
# Contract validation (spec §5)                                                 #
# --------------------------------------------------------------------------- #

def test_valid_intent_parses():
    intent = OrderIntent.from_dict(valid_intent_dict())
    intent.validate(min_lot=0.01, max_lot=5.0, allowed_symbols={"EURUSD"})
    assert intent.side == "SELL"
    assert intent.tp_price == 1.0840


def test_tp_price_null_allowed():
    intent = OrderIntent.from_dict(valid_intent_dict(tp_price=None))
    assert intent.tp_price is None


@pytest.mark.parametrize("field", [
    "intent_id", "created_at", "symbol", "side", "order_type",
    "volume_lots", "sl_price", "comment", "magic",
    "source_strategy", "signal_hash", "max_slippage_points",
])
def test_missing_required_field_rejected(field):
    d = valid_intent_dict()
    del d[field]
    with pytest.raises(IntentError) as e:
        OrderIntent.from_dict(d)
    assert field in str(e.value)


def test_bad_side_rejected():
    with pytest.raises(IntentError):
        OrderIntent.from_dict(valid_intent_dict(side="HOLD"))


def test_non_market_order_type_rejected():
    with pytest.raises(IntentError):
        OrderIntent.from_dict(valid_intent_dict(order_type="PENDING"))


def test_nonpositive_volume_rejected():
    with pytest.raises(IntentError):
        OrderIntent.from_dict(valid_intent_dict(volume_lots=0))
    with pytest.raises(IntentError):
        OrderIntent.from_dict(valid_intent_dict(volume_lots=-1))


def test_comment_too_long_rejected():
    with pytest.raises(IntentError):
        OrderIntent.from_dict(valid_intent_dict(comment="x" * 32))


def test_magic_must_be_int():
    with pytest.raises(IntentError):
        OrderIntent.from_dict(valid_intent_dict(magic="5015"))
    # bool is not a valid magic even though bool is int subclass
    with pytest.raises(IntentError):
        OrderIntent.from_dict(valid_intent_dict(magic=True))


def test_volume_out_of_config_bounds_rejected():
    intent = OrderIntent.from_dict(valid_intent_dict(volume_lots=10.0))
    with pytest.raises(IntentError):
        intent.validate(min_lot=0.01, max_lot=5.0)


def test_symbol_not_in_map_rejected():
    intent = OrderIntent.from_dict(valid_intent_dict(symbol="XAUUSD"))
    with pytest.raises(IntentError):
        intent.validate(min_lot=0.01, max_lot=5.0, allowed_symbols={"EURUSD"})


def test_sell_sl_must_be_above_tp():
    # SELL with SL below TP is invalid (SL must sit above a short's target).
    intent = OrderIntent.from_dict(valid_intent_dict(side="SELL", sl_price=1.05, tp_price=1.09))
    with pytest.raises(IntentError):
        intent.validate(min_lot=0.01, max_lot=5.0)


def test_buy_sl_must_be_below_tp():
    intent = OrderIntent.from_dict(
        valid_intent_dict(side="BUY", sl_price=1.09, tp_price=1.05)
    )
    with pytest.raises(IntentError):
        intent.validate(min_lot=0.01, max_lot=5.0)


def test_load_from_file(tmp_path):
    p = tmp_path / "i.json"
    p.write_text(json.dumps(valid_intent_dict()))
    intent = OrderIntent.load(p)
    assert intent.symbol == "EURUSD"


def test_load_missing_file_rejected(tmp_path):
    with pytest.raises(IntentError):
        OrderIntent.load(tmp_path / "nope.json")


# --------------------------------------------------------------------------- #
# CLI flow: stage / route / idempotency (spec §6) via MockConnector            #
# --------------------------------------------------------------------------- #

def _setup_repo(tmp_path, monkeypatch, trade_mode=ACCOUNT_TRADE_MODE_DEMO):
    """Build a throwaway config + intent tree and point the bridge at it."""
    intents = tmp_path / "intents"
    intents.mkdir()
    cfg = {
        "lots": {"min_lot": 0.01, "max_lot": 5.0},
        "symbol_map": {"EURUSD": "EURUSD"},
        "paths": {
            "intents_dir": str(intents),
            "pending_dir": str(tmp_path / "pending"),
            "routed_ledger": str(tmp_path / "routed.jsonl"),
            "live_unlock": str(tmp_path / "unlock.json"),
        },
    }
    (intents / "i1.json").write_text(json.dumps(valid_intent_dict(intent_id="i1")))
    # Patch _p to resolve against absolute paths from this cfg (not REPO_ROOT).
    monkeypatch.setattr(mt5_bridge, "_p", lambda c, k, d: Path(c["paths"].get(k, d)))
    monkeypatch.setattr(mt5_bridge, "load_config", lambda *a, **kw: cfg)
    conn = MockConnector(account=MockAccount(trade_mode=trade_mode))
    conn.ticks["EURUSD"] = Tick("EURUSD", bid=1.0850, ask=1.0851, time_msc=1)
    return cfg, conn


def test_stage_then_dryrun_route_places_no_order(tmp_path, monkeypatch):
    cfg, conn = _setup_repo(tmp_path, monkeypatch)
    assert mt5_bridge.main(["--stage", "i1"], connector=conn) == 0
    assert (tmp_path / "pending" / "i1.json").exists()
    # dry-run route (no --approve) must NOT send an order
    assert mt5_bridge.main(["--route", "i1"], connector=conn) == 0
    assert conn.sent_requests == []


def test_route_with_approve_sends_order_and_is_idempotent(tmp_path, monkeypatch):
    cfg, conn = _setup_repo(tmp_path, monkeypatch)

    class _Result:
        retcode = mt5_bridge.TRADE_RETCODE_DONE
        order = 111
        deal = 222
        price = 1.0850
        comment = "done"
    conn.send_result = _Result()

    assert mt5_bridge.main(["--route", "i1", "--approve"], connector=conn) == 0
    assert len(conn.sent_requests) == 1
    ledger = tmp_path / "routed.jsonl"
    assert ledger.exists()

    # Second route of same intent → refused (idempotency), no new send.
    rc = mt5_bridge.main(["--route", "i1", "--approve"], connector=conn)
    assert rc == 1
    assert len(conn.sent_requests) == 1


def test_route_on_real_account_aborts_no_order(tmp_path, monkeypatch):
    cfg, conn = _setup_repo(tmp_path, monkeypatch, trade_mode=ACCOUNT_TRADE_MODE_REAL)
    rc = mt5_bridge.main(["--route", "i1", "--approve"], connector=conn)
    assert rc == 1
    assert conn.sent_requests == []


def test_selftest_routes_nothing(tmp_path, monkeypatch):
    cfg, conn = _setup_repo(tmp_path, monkeypatch)
    assert mt5_bridge.main(["--selftest"], connector=conn) == 0
    assert conn.sent_requests == []


# --------------------------------------------------------------------------- #
# AST isolation (spec §10, CLAUDE.md NN#1)                                      #
# --------------------------------------------------------------------------- #

def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def test_ast_isolation_bridge_and_package():
    files = [
        REPO_ROOT / "mt5_bridge.py",
        REPO_ROOT / "scripts" / "fomc_window_logger.py",
        REPO_ROOT / "sovereign" / "execution" / "mt5" / "connector.py",
        REPO_ROOT / "sovereign" / "execution" / "mt5" / "contract.py",
        REPO_ROOT / "sovereign" / "execution" / "mt5" / "guard.py",
    ]
    for f in files:
        imports = _imports_of(f)
        for frozen in FROZEN_MODULES:
            leaf = frozen.split(".")[-1]
            for imp in imports:
                assert leaf not in imp.split("."), (
                    f"{f.name} imports frozen module '{imp}' (matches '{frozen}')"
                )
