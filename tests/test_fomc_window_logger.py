"""Unit tests for the FOMC-window logger (TICK-056 companion).

Pure/window/slippage logic + a full sampled run against MockConnector that asserts
NO orders are ever placed and DEMO is enforced.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from scripts import fomc_window_logger as fwl
from sovereign.execution.mt5 import ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_REAL
from sovereign.execution.mt5.connector import MockAccount, MockConnector, Tick
from sovereign.execution.mt5.guard import LiveAccountError


def test_parse_center_naive_gets_tz():
    dt = fwl.parse_center("2026-07-29T14:00", "America/New_York")
    assert dt.tzinfo is not None
    assert dt.hour == 14
    # 2:00pm ET on 2026-07-29 is 18:00 UTC (EDT, UTC-4)
    assert dt.utctimetuple().tm_hour == 18


def test_compute_window():
    center = fwl.parse_center("2026-07-29T14:00", "America/New_York")
    start, end = fwl.compute_window(center, 15)
    assert (end - start).total_seconds() == 30 * 60
    assert start.minute == 45
    assert end.minute == 15


def test_load_intended_prices(tmp_path):
    ledger = tmp_path / "routed.jsonl"
    ledger.write_text(
        json.dumps({"symbol": "EURUSD", "magic": 5015, "request_price": 1.0850,
                    "routed_at": "2026-07-29T14:00:00Z"}) + "\n" +
        json.dumps({"symbol": "EURUSD", "magic": 5015, "request_price": 1.0860,
                    "routed_at": "2026-07-29T14:05:00Z"}) + "\n"
    )
    idx = fwl.load_intended_prices(ledger)
    # keeps the most recent for (symbol, magic)
    assert idx[("EURUSD", 5015)]["request_price"] == 1.0860


def test_position_slippage_buy_adverse():
    pos = {"type": 0, "price_open": 1.0855}  # BUY filled above intended → adverse
    slip = fwl.position_slippage(pos, intended_price=1.0850)
    assert slip["slippage_price_adverse"] == pytest.approx(0.0005)


def test_position_slippage_sell_adverse():
    pos = {"type": 1, "price_open": 1.0845}  # SELL filled below intended → adverse
    slip = fwl.position_slippage(pos, intended_price=1.0850)
    assert slip["slippage_price_adverse"] == pytest.approx(0.0005)


def test_position_slippage_none_when_no_intended():
    assert fwl.position_slippage({"type": 0, "price_open": 1.0}, None) is None


def test_build_sample_quotes_and_positions():
    conn = MockConnector(account=MockAccount())
    conn.ticks["EURUSD"] = Tick("EURUSD", bid=1.0850, ask=1.0851, time_msc=5)
    conn.positions = [{"ticket": 1, "symbol": "EURUSD", "magic": 5015, "type": 1,
                       "volume": 0.5, "price_open": 1.0845, "price_current": 1.0840}]
    intended = {("EURUSD", 5015): {"request_price": 1.0850}}
    sample = fwl.build_sample(conn, ["EURUSD"], intended, now_iso="2026-07-29T18:00:00Z")
    assert sample["quotes"][0]["spread_price"] == pytest.approx(0.0001)
    assert sample["open_positions"][0]["slippage"]["slippage_price_adverse"] == pytest.approx(0.0005)


def test_run_places_no_orders_and_writes_samples(tmp_path):
    conn = MockConnector(account=MockAccount(trade_mode=ACCOUNT_TRADE_MODE_DEMO))
    conn.ticks["EURUSD"] = Tick("EURUSD", bid=1.0850, ask=1.0851, time_msc=1)
    out = tmp_path / "fomc.jsonl"
    center = fwl.parse_center("2026-07-29T14:00", "America/New_York")

    # Fake clock inside the window so the loop runs a fixed number of samples.
    ticks = iter([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 999999999.0])
    fake_now = center.timestamp()

    def clock():
        try:
            return fake_now + next(ticks)
        except StopIteration:
            return fake_now + 999999999.0

    rc = fwl.run(conn, {}, center=center, window_min=15, interval_sec=0.0,
                 symbols=["EURUSD"], out_path=out, max_samples=3,
                 clock=clock, sleep=lambda s: None)
    assert rc == 0
    assert conn.sent_requests == []  # NEVER places an order
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
    assert json.loads(lines[0])["quotes"][0]["symbol"] == "EURUSD"


def test_run_refuses_non_demo(tmp_path):
    conn = MockConnector(account=MockAccount(trade_mode=ACCOUNT_TRADE_MODE_REAL))
    center = fwl.parse_center("2026-07-29T14:00", "America/New_York")
    with pytest.raises(LiveAccountError):
        fwl.run(conn, {}, center=center, window_min=1, interval_sec=0.0,
                symbols=["EURUSD"], out_path=tmp_path / "x.jsonl",
                max_samples=1, clock=lambda: center.timestamp(), sleep=lambda s: None)
    assert conn.sent_requests == []


def test_dry_run_no_connection(capsys):
    rc = fwl.main(["--dry-run"])
    assert rc == 0
    assert "DRY-RUN" in capsys.readouterr().out
