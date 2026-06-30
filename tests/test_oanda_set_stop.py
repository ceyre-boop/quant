"""tests/test_oanda_set_stop.py — L2 Step 2 PRACTICE-ACCOUNT integration test for set_stop.

This is a REAL round-trip against the OANDA practice account — NOT a mock. The whole point of
Step 2 is to confirm the TradeCRCDO stop-amend endpoint actually works end-to-end, so a mocked
response would prove nothing. If the practice account is unreachable (missing creds, live mode,
connectivity, or a closed forex market), the test SKIPS with the documented failure mode rather
than substituting a fake response.

    pytest tests/test_oanda_set_stop.py -v -s

Pair: USD_CAD — liquid, but deliberately NOT one of the four v015 live pairs
(EURUSD/GBPUSD/USDJPY/AUDUSD), so the place_trade FIFO gate can't collide with a real
forex_live_scan position. Size: 1 unit (smallest). The trade is opened with a far stop/TP that
cannot trigger, the stop is amended well below entry (cannot trigger), then the trade is closed
immediately in a finally block so nothing is left open on the account.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load the repo .env explicitly (py3.14 find_dotenv() is unreliable under some runners).
load_dotenv(dotenv_path=str(Path(__file__).resolve().parents[1] / ".env"))

TEST_PAIR = "USD_CAD"   # liquid, outside the v015 live set → no FIFO collision risk
TEST_UNITS = 1          # smallest position


def _bridge_or_skip():
    """Construct a practice-account bridge or skip with the documented failure mode."""
    if not os.environ.get("OANDA_API_KEY") or not os.environ.get("OANDA_ACCOUNT_ID"):
        pytest.skip("FAILURE MODE: missing OANDA practice creds (OANDA_API_KEY / "
                    "OANDA_ACCOUNT_ID not in .env) — cannot run a real round-trip.")
    if os.environ.get("OANDA_LIVE") == "1":
        pytest.skip("FAILURE MODE: OANDA_LIVE=1 — refusing to run an integration test against a "
                    "LIVE account. Practice only.")
    try:
        from sovereign.execution.oanda_bridge import OandaBridge
        bridge = OandaBridge()
    except Exception as exc:  # auth / connectivity at construction (NAV confirm)
        pytest.skip(f"FAILURE MODE: bridge init / connectivity failed — {type(exc).__name__}: {exc}")
    if getattr(bridge, "_environment", None) != "practice":
        pytest.skip(f"FAILURE MODE: bridge environment is {bridge._environment!r}, not 'practice'.")
    return bridge


def test_set_stop_practice_round_trip():
    """Open → set_stop → confirm broker accepted the amended stop → close. Real practice account."""
    bridge = _bridge_or_skip()

    # ── 1. Open a minimal position with a far, non-triggering stop/TP ──────────
    open_res = bridge.place_trade(
        pair=TEST_PAIR, direction="LONG", units=TEST_UNITS,
        stop_price=1.00000,   # far below USD_CAD (~1.3x) — cannot trigger
        tp1_price=2.00000,    # far above — cannot trigger
    )
    if open_res.get("status") != "FILLED":
        # Market closed (weekend), FIFO collision, or broker rejection — document & stop, don't mock.
        pytest.skip(f"FAILURE MODE: could not open {TEST_PAIR} test position — {open_res}. "
                    "(Forex market likely closed, or an existing position blocked FIFO.) "
                    "Re-run during market hours; no response was mocked.")

    trade_id = open_res["trade_id"]
    fill_price = float(open_res["fill_price"])
    print(f"\n[round-trip] opened {TEST_PAIR} trade {trade_id} @ {fill_price:.5f} ({TEST_UNITS}u)")

    try:
        # ── 2. Amend the stop to a new, valid, non-triggering level ────────────
        new_stop = round(fill_price * 0.90, 5)   # 10% below entry — valid long stop, cannot trigger
        print(f"[round-trip] set_stop({trade_id}, {new_stop}) …")
        resp = bridge.set_stop(trade_id, new_stop)

        # ── 3. Assert the API response confirms the amended stop price ─────────
        assert isinstance(resp, dict), f"set_stop must return the full API dict, got {type(resp)}"
        sl_txn = resp.get("stopLossOrderTransaction")
        assert sl_txn, f"no stopLossOrderTransaction in response — broker did not accept the amend: {resp}"
        confirmed = float(sl_txn["price"])
        print(f"[round-trip] broker confirmed stop = {confirmed:.5f} (sent {new_stop:.5f})")
        assert abs(confirmed - new_stop) < 1e-5, (
            f"confirmed stop {confirmed} != sent {new_stop} within 5dp")

        # ── 4. Secondary proof: the trade now actually HOLDS the amended stop ──
        trade = bridge.get_trade(trade_id)
        held = float(trade["stopLossOrder"]["price"])
        print(f"[round-trip] open trade now holds stopLossOrder price = {held:.5f}")
        assert abs(held - new_stop) < 1e-5, f"broker-held stop {held} != sent {new_stop} within 5dp"

    finally:
        # ── 5. Always close the position — leave nothing open on the account ──
        close_res = bridge.close_trade(trade_id)
        print(f"[round-trip] closed trade {trade_id}: {close_res.get('status')} "
              f"pl={close_res.get('pl')}")

    assert close_res.get("status") == "CLOSED", f"cleanup close failed: {close_res}"
