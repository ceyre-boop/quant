"""Harness wiring: replay universe sourcing, kill switch, output contract.

Network-free. The Alpaca surface is stubbed so these run offline and in CI.
"""
import json
from datetime import date
from unittest.mock import patch

import pytest

from execution import harness, scan
from execution.harness import FillRecord, run_session, write_daily_summary

DAY = date(2026, 6, 16)


def test_replay_refuses_live_screener_when_archive_is_empty(tmp_path):
    """THE BUG THIS GUARDS.

    `alpaca.movers()` has no historical mode — it returns whatever is moving
    right now. A replay that silently fell back to it would score today's movers
    against a past session and produce a plausible-looking, meaningless result.
    With no archived universe the harness must score nothing instead.
    """
    with patch.object(scan, "archived_symbols", return_value=[]), \
         patch.object(scan, "scan_universe") as scan_universe, \
         patch.object(harness.borrow, "load_locate", return_value=None):
        out = run_session(DAY, out_dir=tmp_path, replay=True, check_news=False)

    assert out == []
    scan_universe.assert_not_called(), "must not screen at all without an archive"

    with open(tmp_path / "daily_summary.csv") as fh:
        row = list(__import__("csv").DictReader(fh))[0]
    assert row["n_signals"] == "0"
    assert row["median_net_return"] == ""      # blank, never 0.0


def test_replay_uses_archived_universe(tmp_path):
    with patch.object(scan, "archived_symbols", return_value=["AAA", "BBB"]) as arch, \
         patch.object(scan, "scan_universe", return_value=[]) as scan_universe, \
         patch.object(harness.borrow, "load_locate", return_value=None):
        run_session(DAY, out_dir=tmp_path, replay=True, check_news=False)

    arch.assert_called_once_with(DAY)
    assert scan_universe.call_args.kwargs["symbols"] == ["AAA", "BBB"]


def test_live_run_uses_screener(tmp_path):
    """Live runs pass symbols=None so scan_universe uses the movers screener."""
    with patch.object(scan, "scan_universe", return_value=[]) as scan_universe, \
         patch.object(harness.borrow, "load_locate", return_value=None):
        run_session(DAY, out_dir=tmp_path, replay=False, check_news=False)
    assert scan_universe.call_args.kwargs["symbols"] is None


def test_exit_timestamp_is_bar_close_not_bar_open():
    """Frozen specs exit at b1030['c'] — the price at 10:31:00, not 10:30:00.

    Capturing the exit quote at the bar's start would price the wrong end of the
    minute and make vs_backtest_delta meaningless.
    """
    captured = {}

    def fake_quote_at(symbol, ts, **kw):
        captured.setdefault(symbol, []).append(ts)
        return None       # force SKIP_NO_QUOTE after recording the timestamps

    with patch.object(harness.quotes, "quote_at", side_effect=fake_quote_at), \
         patch.object(harness.halts, "halted_at", return_value=(False, "clear")):
        harness.price_leg("AAA", DAY, side="LONG", hypothesis="HYP-107",
                          entry_et="09:31", exit_et="10:30",
                          bars=[{"t": "2026-06-16T13:31:00Z", "o": 1, "h": 1,
                                 "l": 1, "c": 1, "v": 1}],
                          stop_pct=0.25)

    entry_ts = captured["AAA"][0]
    assert entry_ts.strftime("%H:%M") == "13:31"       # 09:31 ET open


def test_kill_switch_blocks_and_exits_clean(tmp_path):
    """Paper-only is not an exemption: precedent es_nq_paper_runner.py:54."""
    with patch.object(harness.kill_switch, "skip_if_frozen", return_value=True) as ks, \
         patch.object(harness, "run_session") as rs:
        rc = harness.main(["--live", "--out", str(tmp_path)])
    assert rc == 0, "a frozen skip is a clean exit, not an error"
    ks.assert_called_once()
    rs.assert_not_called()


def test_heartbeat_written_before_kill_switch_check(tmp_path):
    """Heartbeat must land even on a frozen day, so a freeze reads as FROZEN
    rather than a false DOWN."""
    order = []
    with patch.object(harness, "_write_heartbeat",
                      side_effect=lambda m: order.append("heartbeat")), \
         patch.object(harness.kill_switch, "skip_if_frozen",
                      side_effect=lambda *a, **k: order.append("killswitch") or True):
        harness.main(["--live", "--out", str(tmp_path)])
    assert order == ["heartbeat", "killswitch"]


def test_frozen_hash_verified_before_any_network_io(tmp_path):
    """Param drift must fail startup before a single request goes out."""
    with patch.object(harness, "verify_frozen_hash",
                      side_effect=RuntimeError("drift")), \
         patch.object(harness.alpaca, "get") as api:
        with pytest.raises(RuntimeError, match="drift"):
            harness.main(["--live", "--out", str(tmp_path)])
    api.assert_not_called()


def test_skip_rows_are_logged_not_dropped(tmp_path):
    """A skip is evidence. It belongs in the fill log with its reason."""
    rows = [FillRecord(ticker="AAA", date=str(DAY), signal_type="SKIP_NO_BORROW",
                       reason="no_locate_snapshot")]
    harness.record_fill(rows[0], tmp_path)
    rec = json.loads((tmp_path / "fill_log.jsonl").read_text().strip())
    assert rec["signal_type"] == "SKIP_NO_BORROW"
    assert rec["reason"] == "no_locate_snapshot"
    assert rec["net_return"] is None
