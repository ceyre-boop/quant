"""Tests for ALTA TERM.

The terminal's only claim is that it never lies about what it is showing: every
number carries its real age, a dead loop is labelled dead, and there is no order
path anywhere in the process. These pin exactly that.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from rich.console import Console

from sovereign.terminal import app, screens
from sovereign.terminal import sources as S


# ─── freshness: the whole honesty mechanism ──────────────────────────────────

def _aged(tmp_path: Path, name: str, hours: float) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({"x": 1}))
    ts = time.time() - hours * 3600
    os.utime(p, (ts, ts))
    return p


@pytest.mark.parametrize("hours,expected", [
    (0.5, "FRESH"), (23.0, "FRESH"), (25.0, "STALE"),
    (24 * 6, "STALE"), (24 * 8, "DEAD"), (24 * 60, "DEAD"),
])
def test_age_decides_status(tmp_path, hours, expected):
    assert S._classify(_aged(tmp_path, "f.json", hours))[0] == expected


def test_missing_file_is_missing_not_empty(tmp_path):
    status, ts = S._classify(tmp_path / "nope.json")
    assert status == "MISSING" and ts is None


def test_read_json_reports_reason_rather_than_returning_none(tmp_path, monkeypatch):
    monkeypatch.setitem(S.PATHS, "briefing", tmp_path / "gone.json")
    src = S.read_json("briefing")
    assert src.status == "MISSING" and src.note and src.data is None


def test_corrupt_json_is_an_error_not_a_silent_empty(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    monkeypatch.setitem(S.PATHS, "briefing", bad)
    src = S.read_json("briefing")
    assert src.status == "ERROR" and "JSONDecodeError" in src.note


def test_stale_is_readable_but_not_trustworthy(tmp_path):
    src = S.Source("x", "STALE", {"a": 1},
                   datetime.now(timezone.utc) - timedelta(days=3))
    assert src.ok is True
    assert src.trustworthy is False, "a 3-day-old file is history, not state"


def test_dead_is_neither_ok_nor_trustworthy():
    src = S.Source("x", "DEAD", {}, datetime.now(timezone.utc) - timedelta(days=30))
    assert not src.ok and not src.trustworthy


@pytest.mark.parametrize("delta,expected", [
    (timedelta(minutes=30), "30m"),
    (timedelta(hours=5), "5h"),
    (timedelta(days=9), "9d"),
])
def test_age_label_is_human(delta, expected):
    assert S.Source("x", "FRESH", {}, datetime.now(timezone.utc) - delta).age_label == expected


def test_live_reads_as_live_not_as_an_age():
    assert S.Source("x", "LIVE", {}, datetime.now(timezone.utc)).age_label == "live"
    assert str(screens.tag(S.Source("x", "LIVE", {}, datetime.now(timezone.utc)))) == "live"


# ─── risk budget comes from the constitution ─────────────────────────────────

def test_risk_caps_are_parsed_from_the_ratified_file():
    rb = S.risk_budget(100_000.0)
    assert rb.parsed_ok, rb.note
    assert rb.per_trade_pct == 0.75
    assert rb.carry_heat_pct == 2.5
    assert rb.ladder_pct == (3.5, 5.0, 6.5)


def test_risk_budget_converts_to_this_accounts_money():
    rb = S.risk_budget(200_000.0)
    assert rb.per_trade == pytest.approx(1500.0)
    assert rb.carry_heat == pytest.approx(5000.0)


def test_ladder_levels_are_below_nav_and_ordered():
    levels = [lv for _, _, lv in S.risk_budget(100_000.0).ladder_levels()]
    assert levels == sorted(levels, reverse=True)
    assert all(lv < 100_000.0 for lv in levels)


def test_unreadable_constitution_says_so_instead_of_inventing_caps(monkeypatch, tmp_path):
    monkeypatch.setitem(S.PATHS, "risk_const", tmp_path / "missing.md")
    rb = S.risk_budget(100_000.0)
    assert rb.parsed_ok is False
    assert "unreadable" in rb.note


# ─── loop health is recomputed, not trusted ──────────────────────────────────

def test_loop_health_reports_its_own_status_file_as_one_of_the_loops():
    lh = S.loop_health()
    labels = [r["loop"] for r in lh.data["rows"]]
    assert "loop health file" in labels, \
        "the status file is itself a loop and can be stale about its own staleness"
    assert lh.data["n_dead"] == sum(
        r["status"] in ("DEAD", "MISSING") for r in lh.data["rows"])


# ─── symbols ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("EURUSD", "EUR_USD"), ("eur_usd", "EUR_USD"),
    ("GBP/JPY", "GBP_JPY"), ("SPY", "SPY"),
])
def test_oanda_symbol_normalisation(raw, expected):
    assert S._oanda_symbol(raw) == expected


def test_audnzd_is_not_in_the_live_board():
    assert "AUDNZD" not in screens.LIVE_PAIRS, "excluded by HYP-045 — both legs RBA-driven"
    assert set(screens.LIVE_PAIRS) == {"EURUSD", "GBPUSD", "AUDUSD", "GBPJPY"}


# ─── read-only guarantee ─────────────────────────────────────────────────────

@pytest.mark.parametrize("module", [S, screens, app])
def test_no_order_path_anywhere_in_the_terminal(module):
    """Not 'disabled' — absent. No writer, no order endpoint, no broker POST."""
    src = Path(module.__file__).read_text()
    banned = ("/v3/accounts/{}/orders", "/orders", "place_order", "create_order",
              "method=\"POST\"", "method='POST'", "forex_exit_manager", "decide_exit")
    assert not [b for b in banned if b in src]


def test_terminal_only_reads_from_the_broker():
    src = Path(S.__file__).read_text()
    for endpoint in ("/summary", "/openPositions", "/openTrades", "/pricing"):
        assert endpoint in src
    assert "urllib.request.Request(" in src
    assert "data=" not in src.split("def account")[1].split("def quote")[0], \
        "a request body would make this a write"


# ─── dispatch + graceful degradation ─────────────────────────────────────────

@pytest.fixture
def offline_session(monkeypatch):
    """A session where every live source is down — the terminal must still render."""
    monkeypatch.setattr(S, "account",
                        lambda: S.Source("account", "ERROR", note="no broker in tests"))
    monkeypatch.setattr(S, "quote",
                        lambda sym: S.Source("quote", "ERROR", note=f"no quote for {sym}"))
    monkeypatch.setattr(S, "trade_context",
                        lambda sym, offline=True, with_library=True:
                        S.Source("context", "ERROR", note="context off in tests"))
    return app.Session(console=Console(width=100, no_color=True, force_terminal=False))


@pytest.mark.parametrize("cmd", [
    "", "HOME", "H", "HELP", "?", "POS", "RISK", "BOARD", "MKT",
    "HEALTH", "EDGE", "HYP", "HYP carry", "MACRO", "LIB", "DES EURUSD",
    "EURUSD", "SPY", "DES",
])
def test_every_command_renders_with_all_live_sources_down(offline_session, cmd):
    out = offline_session.console.render_str  # noqa: F841  (touch the console)
    assert app.render(offline_session, cmd) is not None


def test_unknown_command_is_reported_not_swallowed(offline_session):
    rendered = _text(offline_session, "ZZZ NOT A COMMAND")
    assert "unknown command" in rendered


def test_a_bare_symbol_is_treated_as_a_symbol(offline_session, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(app, "_instrument",
                        lambda sess, sym: seen.append(sym) or "x")
    app.render(offline_session, "gbpjpy")
    assert seen == ["GBPJPY"]


def _text(sess, cmd: str) -> str:
    console = Console(width=120, no_color=True, force_terminal=False, record=True)
    console.print(app.render(sess, cmd))
    return console.export_text()


def test_health_screen_names_the_stopped_writers(offline_session):
    out = _text(offline_session, "HEALTH")
    assert "SYSTEM HEALTH" in out
    assert "writers have stopped" in out or "All writers current" in out


def test_broker_failure_is_shown_not_rendered_as_zero(offline_session):
    out = _text(offline_session, "POS")
    assert "no broker in tests" in out
    assert "$0.00" not in out, "a dead broker must not render as a flat account"


def test_one_shot_mode_exits_zero(monkeypatch, offline_session):
    console = Console(width=100, no_color=True, force_terminal=False)
    monkeypatch.setattr(app, "Session", lambda console: offline_session)
    assert app.run(console, "HEALTH", once=True) == 0
