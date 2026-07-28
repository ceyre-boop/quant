"""TICK-092 — regression tests for the outcome-loop accounting contract.

Fences audit/fills_ledger_spec.md:
  F1 every examined trade lands in exactly one bucket (buckets sum to seen)
  F5 net_r is real or null — never fabricated
  F6 the fills-ledger producer never writes to the broker

The defect these guard against: `update_outcome()` returns False both when no decision
record exists AND when one exists but is already CLOSED. `pulse_check` treated the second
case as the first and stamped 13 of 20 closed trades `unmatchable` with the reason "the
entry path likely never called log_forex_decision()" — false for all 13, which carried real
WIN/LOSS outcomes. The sidecar then short-circuited them forever, so the wrong verdict was
self-preserving and successive sessions kept hunting an entry path that worked.
"""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ── F6: the producer is read-only against the broker ──────────────────────────

def test_rebuild_fills_ledger_never_writes_to_broker():
    """F6. A ledger rebuild that could place or amend an order is not a rebuild."""
    src = (ROOT / "scripts" / "rebuild_fills_ledger.py").read_text()
    tree = ast.parse(src)
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for forbidden in ("place_trade", "close_trade", "set_stop"):
        assert forbidden not in called, (
            f"rebuild_fills_ledger.py calls broker-write method {forbidden!r} — "
            "spec F6 requires read-only access"
        )


def test_rebuild_fills_ledger_emits_required_fields():
    """F3. Field names are dictated by the four existing readers; drift breaks them silently."""
    src = (ROOT / "scripts" / "rebuild_fills_ledger.py").read_text()
    for field in ("trade_id", "pair", "direction", "fill_price", "stop_price", "timestamp"):
        assert f'"{field}"' in src, f"producer must emit reader-expected field {field!r}"


# ── F5: net_r is real or null, never fabricated ───────────────────────────────

def test_net_r_credit_raises_return_above_gross():
    """A financing CREDIT must increase net R. Short EURUSD is the calibrated credit case."""
    from sovereign.oracle.pulse_check import _net_r_from_gross

    net, financing_r, rate = _net_r_from_gross(
        pair="EURUSD", side="SHORT", entry=1.14395, stop=1.16,
        gross_r=0.50, hold_hours=24 * 8, open_ts="2026-07-03",
    )
    assert rate is not None and rate > 0, "OANDA pays to hold short EURUSD — rate must be a credit"
    assert financing_r > 0
    assert net > 0.50, "a credit must raise net R above gross R"


def test_net_r_charge_lowers_return_below_gross():
    from sovereign.oracle.pulse_check import _net_r_from_gross

    net, financing_r, rate = _net_r_from_gross(
        pair="EURUSD", side="LONG", entry=1.14395, stop=1.13,
        gross_r=0.50, hold_hours=24 * 8, open_ts="2026-07-03",
    )
    assert rate < 0 and financing_r < 0
    assert net < 0.50, "a financing charge must lower net R below gross R"


@pytest.mark.parametrize("kwargs", [
    dict(pair="USDCAD", side="LONG", entry=1.36, stop=1.35, gross_r=1.0,
         hold_hours=48, open_ts="2026-07-03"),           # pair not calibrated
    dict(pair="EURUSD", side="LONG", entry=1.14, stop=1.13, gross_r=None,
         hold_hours=48, open_ts="2026-07-03"),           # no gross R
    dict(pair="EURUSD", side="LONG", entry=1.14, stop=1.14, gross_r=1.0,
         hold_hours=48, open_ts="2026-07-03"),           # zero risk
])
def test_net_r_returns_null_rather_than_guessing(kwargs):
    """F5. Un-priceable inputs yield None. Falling back to the static table that TICK-024
    proved ~9x wrong (with a sign flip) is exactly the silent-mocking failure to avoid."""
    from sovereign.oracle.pulse_check import _net_r_from_gross

    assert _net_r_from_gross(**kwargs) == (None, None, None)


# ── The conflation fix: already-closed is not the same as missing ─────────────

def test_find_recorded_outcome_distinguishes_closed_from_missing(tmp_path, monkeypatch):
    """The core defect. A CLOSED record must be reported as closed, not as absent."""
    from sovereign.intelligence import decision_logger as dl

    log_dir = tmp_path / "decision_logs"
    log_dir.mkdir()
    (log_dir / "decisions_2026_07.jsonl").write_text(
        json.dumps({"system": "FOREX", "pair": "EURUSD=X", "trade_id": "177",
                    "outcome": "LOSS", "r_realized": -1.0}) + "\n"
        + json.dumps({"system": "FOREX", "pair": "GBPUSD=X", "trade_id": "999",
                      "outcome": None}) + "\n"
    )
    monkeypatch.setattr(dl, "LOG_DIR", log_dir)

    closed = dl.find_recorded_outcome(pair="EUR_USD", trade_id="177", system="FOREX")
    assert closed is not None and closed["outcome"] == "LOSS", (
        "a closed record must be found — reporting it as missing is what produced the "
        "false 'entry path never called log_forex_decision()' verdict"
    )

    # Still OPEN → not a closed record; the normal backfill path should handle it.
    assert dl.find_recorded_outcome(pair="GBP_USD", trade_id="999", system="FOREX") is None
    # Genuinely absent → None, so a real gap is still reported as a gap.
    assert dl.find_recorded_outcome(pair="EUR_USD", trade_id="404", system="FOREX") is None


def test_find_recorded_outcome_normalises_pair_across_venues(tmp_path, monkeypatch):
    """Forex logs the yfinance ticker, OANDA uses underscores, ICT uses plain."""
    from sovereign.intelligence import decision_logger as dl

    log_dir = tmp_path / "decision_logs"
    log_dir.mkdir()
    (log_dir / "decisions_2026_07.jsonl").write_text(
        json.dumps({"system": "FOREX", "pair": "EURUSD=X", "trade_id": "1",
                    "outcome": "WIN"}) + "\n"
    )
    monkeypatch.setattr(dl, "LOG_DIR", log_dir)

    for venue_format in ("EUR_USD", "EURUSD", "EURUSD=X"):
        assert dl.find_recorded_outcome(
            pair=venue_format, trade_id="1", system="FOREX") is not None, venue_format


# ── F1: total accounting ──────────────────────────────────────────────────────

def test_backfill_accounts_for_every_trade_it_examines():
    """F1. Buckets must sum to trades seen — this is what makes the NEXT silent drain
    impossible, since a new skip path cannot be added without breaking the sum."""
    src = (ROOT / "sovereign" / "oracle" / "pulse_check.py").read_text()
    assert "OUTCOME_LOOP_UNACCOUNTED" in src, "F1 bucket-sum check must be present"
    assert "n_seen" in src and "accounted" in src


def test_no_stop_skip_is_counted_not_silent():
    """F2. The skip stays (never fabricate an R) but must increment a reported bucket."""
    src = (ROOT / "sovereign" / "oracle" / "pulse_check.py").read_text()
    assert "n_no_stop" in src and "OUTCOME_LOOP_NO_STOP" in src


def test_already_closed_trades_are_not_counted_as_stalls():
    """A trade whose record already carries an outcome is the loop WORKING. Counting it
    as a failure is what generated the false URGENT 'entry path is broken' alarm."""
    src = (ROOT / "sovereign" / "oracle" / "pulse_check.py").read_text()
    assert "n_failed = n_attempted - n_backfilled - n_already_closed" in src, (
        "already-closed trades must be excluded from the stall count"
    )
