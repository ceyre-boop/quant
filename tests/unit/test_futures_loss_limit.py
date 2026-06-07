"""Track-2 futures sandbox: the hard daily loss limit must lock at the threshold and stay locked
until manually unlocked. This is the non-negotiable safety floor — test it like one."""
import json
from datetime import datetime, timezone

from sovereign.futures import loss_limit as ll


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(ll, "LOCK_FILE", tmp_path / ".session_lock")
    monkeypatch.setattr(ll, "TRADE_LOG", tmp_path / "trade_log.jsonl")


def test_locks_at_limit_and_stays_locked(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert not ll.is_locked()
    assert ll.check_and_lock(-200.0, 500.0) is False     # within limit → no lock
    assert not ll.is_locked()
    assert ll.check_and_lock(-500.0, 500.0) is True       # at limit → lock
    assert ll.is_locked()
    # Once locked, the auto path stays blocked even if P&L recovers (only manual unlock clears it).
    assert ll.check_and_lock(+100.0, 500.0) is True
    ll.unlock()
    assert not ll.is_locked()


def test_session_pnl_fallback_from_trade_log(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    today = datetime.now(timezone.utc).isoformat()
    # MES long: 5000→4990 = −10pt × $5 × 1ct = −$50 ; MNQ short: 18000→17990 = +10pt × $2 × 2ct = +$40
    # Session P&L is the ACCOUNT TOTAL across instruments (per-trade point value): −50 + 40 = −10.
    ll.TRADE_LOG.write_text(
        json.dumps({"ts": today, "instrument": "MES", "direction": "LONG",
                    "entry": 5000, "exit": 4990, "size_contracts": 1}) + "\n" +
        json.dumps({"ts": today, "instrument": "MNQ", "direction": "SHORT",
                    "entry": 18000, "exit": 17990, "size_contracts": 2}) + "\n")
    assert abs(ll.session_pnl_usd(None, "MES") - (-10.0)) < 1e-6
    assert abs(ll.session_pnl_usd(None, "MNQ") - (-10.0)) < 1e-6   # total is instrument-agnostic


def test_ib_realized_pnl_preferred(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    class _Bridge:
        def account_summary(self):
            return {"RealizedPnL": -321.5}
    assert ll.session_pnl_usd(_Bridge(), "MES") == -321.5
