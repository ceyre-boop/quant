"""tests/test_forex_exit_manager.py — L2 Step 3 validation for the shadow exit manager.

The load-bearing test is REPLAY-MATCH: drive the manager's live decision path (init_trade_state →
step_trade, the exact code that will run in production) over the same historical bars the backtester
steps over, and assert every exit lands on the SAME day with the SAME reason as fast_backtester's
ledger. Same entries → same exits, decision-for-decision. The rest pin the safety contracts: state
halts loudly when missing/corrupt, reconciliation drops/initializes correctly, and SHADOW mode makes
zero broker writes while flipping the toggle to LIVE executes the identical decisions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sovereign.forex.exit_machine import ExitConfig, ExitDecision
from sovereign.forex.fast_backtester import simulate_forex_trades_arrays
from sovereign.execution import forex_exit_manager as mgr
from sovereign.execution.forex_exit_manager import (
    Action, MarketBar, StateError, TradeState, cfg_for_pair, init_trade_state,
    init_state_file, load_state, reconcile, run_daily, save_state, step_trade,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "exit_machine_golden.npz"

# Map manager ExitDecision → the backtester's exit_reason string (fast_backtester.reason_map).
DECISION_TO_REASON = {
    ExitDecision.INITIAL_STOP: "stop",
    ExitDecision.REVERSAL: "reversal",
    ExitDecision.CB_REFRESH: "cb_refresh",
    ExitDecision.TIME: "time",
    ExitDecision.TRAILING_ATR: "trailing_stop",
    ExitDecision.DONCHIAN: "donchian_exit",
}


def _replay_match(opens, closes, signals, hold_days, atr_pcts, cfg: ExitConfig):
    """Run the backtester to get its ledger, then drive the manager's step_trade per trade and assert
    exit-day + reason parity. Returns (n_trades, n_matched) — they must be equal with zero divergence."""
    ledger = simulate_forex_trades_arrays(
        opens=opens, closes=closes, signals=signals, hold_days=hold_days, stop_pct=0.04,
        index=None,  # entry_date / exit_date come back as integer bar indices
        atr_pcts=atr_pcts, stop_atr_mult=cfg.stop_atr_mult, trailing_atr_mult=cfg.trailing_atr_mult,
        strict_mode=cfg.strict_mode, enable_cb_refresh=cfg.enable_cb_refresh,
    )
    matched = 0
    for t in ledger:
        e_idx, x_idx = int(t["entry_date"]), int(t["exit_date"])
        ts = init_trade_state(
            trade_id=f"t{e_idx}", pair="TEST", direction=int(t["direction"]),
            entry_price=float(opens[e_idx]),
            entry_atr_pct=float(atr_pcts[e_idx - 1]),       # ATR @ entry-signal bar (entry_idx-1)
            hold_days_at_entry=int(hold_days[e_idx - 1]),   # hold @ entry-signal bar
            entry_date=str(e_idx), cfg=cfg,
        )
        closed_at = None
        for i in range(e_idx, len(closes)):
            bar = MarketBar(pair="TEST", date=str(i), close=float(closes[i]),
                            atr_pct=float(atr_pcts[i]), signal=int(signals[i]),
                            hold_today=int(hold_days[i]))
            res = step_trade(ts, bar, cfg)
            ts = res.new_state
            if res.action == Action.CLOSE:
                closed_at = i
                assert i == x_idx, f"trade {e_idx}: manager closed at {i}, backtester at {x_idx}"
                assert DECISION_TO_REASON[res.decision] == t["exit_reason"], (
                    f"trade {e_idx}: manager reason {res.decision.name} "
                    f"({DECISION_TO_REASON[res.decision]}) != backtester {t['exit_reason']}")
                break
        assert closed_at is not None, f"trade {e_idx}: manager never closed (backtester exited at {x_idx})"
        matched += 1
    return len(ledger), matched


# ─── load-bearing replay-match vs the Step-1 golden fixture ──────────────────────────────────────

def test_replay_match_golden():
    if not FIXTURE.exists():
        pytest.skip("golden fixture not captured")
    g = np.load(FIXTURE)
    cfg = ExitConfig(stop_atr_mult=2.0, trailing_atr_mult=1.25, strict_mode=False, enable_cb_refresh=True)
    n, matched = _replay_match(g["opens"], g["closes"], g["signals"], g["hold_days"], g["atr_pcts"], cfg)
    assert n > 0 and matched == n, f"{n - matched}/{n} divergences vs backtester ledger"
    # the golden exercises >=5 exit types; confirm the replay actually walked a non-trivial ledger
    assert n >= 5


def test_replay_match_synthetic_trend_and_reversal():
    """Independent series: an up-trend that time-exits, then a clean signal reversal."""
    n = 90
    opens = np.linspace(1.00, 1.20, n).astype(np.float64)
    closes = opens + 0.001
    signals = np.zeros(n, dtype=np.int8)
    signals[0] = 1            # go long, rides the uptrend → time exit at hold_limit
    signals[75] = -1          # later flip → reversal on whatever is open
    hold_days = np.full(n, 60, dtype=np.int32)
    atr_pcts = np.full(n, 0.01, dtype=np.float64)
    cfg = ExitConfig(2.0, 1.25, False, True)
    total, matched = _replay_match(opens, closes, signals, hold_days, atr_pcts, cfg)
    assert total >= 1 and matched == total


# ─── state persistence: HALT loudly, never silently re-init ──────────────────────────────────────

def test_load_missing_state_halts(tmp_path):
    with pytest.raises(StateError, match="MISSING"):
        load_state(tmp_path / "nope.json")


def test_load_corrupt_state_halts(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{ this is not json ")
    with pytest.raises(StateError, match="CORRUPT"):
        load_state(p)


def test_load_wrong_shape_halts(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"version": 1}))   # no "trades"
    with pytest.raises(StateError, match="unexpected shape"):
        load_state(p)


def test_init_then_load_and_no_clobber(tmp_path):
    p = tmp_path / "state.json"
    init_state_file(p)
    st = load_state(p)
    assert st["trades"] == {} and st["version"] == mgr.STATE_VERSION
    with pytest.raises(StateError, match="refusing to overwrite"):
        init_state_file(p)            # second init must NOT clobber


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    ts = init_trade_state("9", "EUR_USD", -1, 1.1395, 0.006, 60, "2026-06-23T00:00:00Z",
                          cfg_for_pair("EUR_USD"))
    save_state({"trades": {"9": ts.to_dict()}}, p)
    back = load_state(p)
    assert TradeState.from_dict(back["trades"]["9"]) == ts


# ─── reconciliation: drop closed, initialize new ─────────────────────────────────────────────────

class _StubProvider:
    def __init__(self, atr=0.01, bar=None):
        self._atr, self._bar = atr, bar
    def entry_atr_pct(self, pair, entry_date):
        return self._atr
    def market_bar(self, pair):
        if self._bar is None:
            raise RuntimeError("no bar configured")
        return self._bar


def test_reconcile_drops_closed_and_inits_new():
    state = {"trades": {"OLD": init_trade_state("OLD", "GBP_USD", 1, 1.25, 0.01, 60, "x",
                                                cfg_for_pair("GBP_USD")).to_dict()}}
    open_trades = [{"id": "NEW", "instrument": "EUR_USD", "currentUnits": "-10000",
                    "price": "1.1395", "openTime": "2026-06-23T00:00:00Z"}]
    reconcile(state, open_trades, _StubProvider(atr=0.006))
    assert "OLD" not in state["trades"], "closed trade not dropped"
    assert "NEW" in state["trades"], "new trade not initialized"
    new = TradeState.from_dict(state["trades"]["NEW"])
    assert new.direction == -1                      # short (negative units)
    assert new.stop_price > new.entry_price         # short stop sits ABOVE entry
    assert new.hold_limit == mgr.HOLD_DAYS


# ─── the toggle: SHADOW writes nothing; LIVE executes the same decision ──────────────────────────

class _SpyBridge:
    def __init__(self, open_trades):
        self._ot = open_trades
        self.set_stop_calls: list = []
        self.close_trade_calls: list = []
    def get_open_trades(self):
        return self._ot
    def set_stop(self, tid, price):
        self.set_stop_calls.append((tid, price)); return {}
    def close_trade(self, tid):
        self.close_trade_calls.append(tid); return {"status": "CLOSED"}


def _long_trade_about_to_reverse(tmp_path):
    """A long EUR_USD trade plus a bar whose signal flips short → decide_exit REVERSAL → CLOSE."""
    open_trades = [{"id": "42", "instrument": "EUR_USD", "currentUnits": "10000",
                    "price": "1.10000", "openTime": "2026-06-01T00:00:00Z"}]
    bar = MarketBar(pair="EUR_USD", date="2026-06-26", close=1.10000, atr_pct=0.01,
                    signal=-1, hold_today=60)   # reversal; close above stop, trail not hit
    provider = _StubProvider(atr=0.01, bar=bar)
    state_p = tmp_path / "state.json"
    init_state_file(state_p)
    return open_trades, provider, state_p, tmp_path / "shadow.jsonl"


def test_shadow_mode_makes_zero_broker_writes(tmp_path):
    open_trades, provider, state_p, log_p = _long_trade_about_to_reverse(tmp_path)
    spy = _SpyBridge(open_trades)
    entries = run_daily(spy, provider, shadow=True, state_path=state_p, shadow_log_path=log_p)

    assert spy.set_stop_calls == [], "SHADOW must never call set_stop"
    assert spy.close_trade_calls == [], "SHADOW must never call close_trade"
    assert len(entries) == 1
    assert entries[0]["decision"] == "REVERSAL" and entries[0]["action"] == "CLOSE"
    assert entries[0]["mode"] == "SHADOW"
    # the decision was journaled to the shadow log
    logged = [json.loads(l) for l in log_p.read_text().splitlines()]
    assert logged[0]["action"] == "CLOSE"


def test_live_toggle_executes_same_decision(tmp_path):
    open_trades, provider, state_p, log_p = _long_trade_about_to_reverse(tmp_path)
    spy = _SpyBridge(open_trades)
    run_daily(spy, provider, shadow=False, state_path=state_p, shadow_log_path=log_p)
    # The ONLY difference from shadow is that the same CLOSE decision now hits the broker.
    assert spy.close_trade_calls == ["42"]
    assert spy.set_stop_calls == []


def test_module_default_is_shadow():
    assert mgr.SHADOW_MODE is True, "the committed default MUST be shadow"
