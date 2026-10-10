"""Thesis engine: hash stability, ledger refusals, scoring and Modified Dietz. Network-free."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sovereign.thesis import ledger
from sovereign.thesis.schema import canonical_hash, validate
from sovereign.thesis.score import basket_curve, max_drawdown, score
from sovereign.thesis.reconcile import daily_frame, modified_dietz, twr


def spec(**over):
    d = {
        "id": "THS-900", "title": "t", "thesis": "m", "horizon_months": 12,
        "basket": [{"ticker": "AAA", "weight": 0.5}, {"ticker": "BBB", "weight": 0.5}],
        "entry_date": "2024-01-02",
        "invalidation": {"rule": "dd 30%", "metric": "basket_drawdown", "threshold": 0.30},
        "source": "operator",
    }
    d.update(over)
    return d


def fake_loader(ticker):
    idx = pd.bdate_range("2024-01-02", periods=200)
    if ticker == "AAA":
        v = np.linspace(100, 200, 200)          # +100%
    elif ticker == "BBB":
        v = np.linspace(100, 50, 200)           # -50%
    else:                                        # SPY: +10%
        v = np.linspace(100, 110, 200)
    s = pd.Series(v, index=idx, name=ticker)
    return s


def test_hash_ignores_key_order_and_mutable_fields():
    a = spec(); b = dict(reversed(list(spec().items())))
    assert canonical_hash(a) == canonical_hash(b)
    assert canonical_hash(dict(a, status="LIVE", scorecards=[1])) == canonical_hash(a)
    assert canonical_hash(dict(a, entry_date="2024-01-03")) != canonical_hash(a)


def test_validate_requires_invalidation_and_weights():
    assert validate(spec()) == []
    assert any("invalidation" in e for e in validate(spec(invalidation={})))
    assert any("weights" in e for e in validate(spec(basket=[{"ticker": "A", "weight": 0.7}])))


def test_ledger_refuses_duplicate_and_sealed_mutation(tmp_path):
    L = tmp_path / "l.jsonl"; P = tmp_path / "pre"
    ledger.create(spec(), L)
    with pytest.raises(ledger.LedgerError):
        ledger.create(spec(), L)
    r = ledger.seal("THS-900", L, P)
    assert r["status"] == "SEALED" and ledger.verify("THS-900", L, P)
    with pytest.raises(ledger.LedgerError):
        ledger.seal("THS-900", L, P)
    bad = dict(ledger.get("THS-900", L), basket=[{"ticker": "AAA", "weight": 1.0}])
    with pytest.raises(ledger.LedgerError):
        ledger.update(bad, L)
    ok = ledger.set_status("THS-900", "LIVE", "entered", L)
    assert ok["status"] == "LIVE" and ok["hash_lock"] == r["hash_lock"]
    # tamper with prereg file → verify fails
    pf = P / "THS-900.json"; d = json.loads(pf.read_text()); d["entry_date"] = "1999-01-01"
    pf.write_text(json.dumps(d))
    assert not ledger.verify("THS-900", L, P)


def test_score_reproduces_hand_computed_basket():
    r = spec()
    card = score(r, fake_loader)
    # 0.5*(2.0) + 0.5*(0.5) = 1.25 → +25%
    assert card["basket_return"] == pytest.approx(0.25, abs=1e-9)
    assert card["benchmark_return"] == pytest.approx(0.10, abs=1e-9)
    assert card["excess_return"] == pytest.approx(0.15, abs=1e-9)
    assert card["invalidated"] is False
    assert card["marks"]["12m"]["basket"] is None  # 200 bdays < 12 months
    assert card["marks"]["3m"]["basket"] is not None
    c = basket_curve([{"ticker": "BBB", "weight": 1.0}], "2024-01-02", fake_loader)
    assert max_drawdown(c) == pytest.approx(-0.5, abs=1e-9)
    assert score(spec(basket=[{"ticker": "BBB", "weight": 1.0}]), fake_loader)["invalidated"] is True


def test_modified_dietz_and_twr_known_values():
    idx = pd.bdate_range("2024-01-01", periods=3)
    # day0: deposit 100 → value 100; day1: value 110 (+10%); day2: deposit 110 at open, value 242 (+10%)
    df = pd.DataFrame({"value": [100.0, 110.0, 242.0], "flow": [100.0, 0.0, 110.0]}, index=idx)
    t, _ = twr(df)
    assert t == pytest.approx(0.21, abs=1e-9)
    md = modified_dietz(df)
    assert md == pytest.approx(32 / 155, abs=1e-9)


def test_daily_frame_from_fills():
    rows = [
        {"ts": "2024-01-02", "type": "CASH", "subtype": "DEPOSIT", "net": 1000.0},
        {"ts": "2024-01-02", "type": "TRADE", "symbol": "AAA", "side": "BUY", "qty": 5, "price": 100.0, "net": -500.0},
        {"ts": "2024-01-04", "type": "TRADE", "symbol": "AAA", "side": "SELL", "qty": 5, "price": 100.67, "net": 503.35},
    ]
    df = daily_frame(rows, fake_loader, "2024-01-05")
    assert df["flow"].iloc[0] == 1000.0
    assert df["value"].iloc[-1] == pytest.approx(1003.35)
    sub = daily_frame(rows, fake_loader, "2024-01-05", tickers={"AAA"})
    assert sub["flow"].iloc[0] == 500.0 and sub["flow"].loc["2024-01-04"] == pytest.approx(-503.35)
