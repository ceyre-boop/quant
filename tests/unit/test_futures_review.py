"""Nightly + weekly review tests — costed P&L, pattern/hypothesis logic, null-safety."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from sovereign.futures import review_common as rc


def _trade(inst, direction, entry, exit_p, r, *, cvd_quality="HIGH", setup="ORB",
           confidence="MEDIUM", regime="TRENDING", time_gate="OPEN", confluence=1,
           ts="2026-06-08T14:00:00+00:00", learning=True):
    return {"ts": ts, "instrument": inst, "direction": direction, "entry": entry, "exit": exit_p,
            "r_realized": r, "size_contracts": 1, "setup_type": setup,
            "bias_direction": direction,
            "reasoning": {"cvd_quality": cvd_quality, "setup_type": setup, "confidence": confidence,
                          "regime": regime, "time_gate": time_gate, "confluence_score": confluence,
                          "learning_mode": learning}}


def test_pnl_and_winrate():
    win = _trade("MES", "LONG", 100.0, 110.0, 2.0)   # +10pts * $5 - cost
    loss = _trade("MES", "LONG", 100.0, 98.0, -1.0)
    assert rc.trade_pnl_usd(win) > 0 and rc.trade_pnl_usd(loss) < 0
    assert rc.trade_pnl_usd({"entry": 1, "exit": None}) is None      # open -> None
    assert rc.is_win(win) is True and rc.is_win(loss) is False
    assert rc.winrate([win, loss]) == 0.5


def _load(script):
    spec = importlib.util.spec_from_file_location(script.stem, script)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_nightly_appends_learnings(tmp_path, monkeypatch):
    log = tmp_path / "trade_log.jsonl"
    # 4 trades: 3 LOW-cvd losses + 1 HIGH win -> a LOW-cvd negative pattern
    trades = [_trade("MES", "LONG", 100, 98, -1.0, cvd_quality="LOW") for _ in range(3)]
    trades.append(_trade("MES", "LONG", 100, 110, 2.0, cvd_quality="HIGH"))
    log.write_text("\n".join(json.dumps(t) for t in trades) + "\n")
    nightly = _load(Path("scripts/futures_nightly_review.py"))
    oracle = tmp_path / "oracle.jsonl"
    monkeypatch.setattr(nightly, "TRADE_LOG", log)
    monkeypatch.setattr(nightly, "ORACLE_LOG", oracle)
    monkeypatch.setattr(nightly, "OUT", tmp_path / "nightly.jsonl")
    monkeypatch.setattr("sys.argv", ["x", "--date", "2026-06-08"])
    assert nightly.main() == 0
    block = json.loads(oracle.read_text().splitlines()[-1])
    assert block["type"] == "session_learnings" and block["closed"] == 4
    assert any("cvd_quality=LOW" in s for s in block["session_learnings"])


def test_weekly_proposes_hypothesis(tmp_path, monkeypatch):
    log = tmp_path / "trade_log.jsonl"
    # 12 HIGH-cvd wins + 12 LOW-cvd losses -> cvd_quality is a strong predictor (n>=10, >15pp)
    trades = [_trade("MES", "LONG", 100, 110, 2.0, cvd_quality="HIGH") for _ in range(12)]
    trades += [_trade("MES", "LONG", 100, 98, -1.0, cvd_quality="LOW") for _ in range(12)]
    log.write_text("\n".join(json.dumps(t) for t in trades) + "\n")
    weekly = _load(Path("scripts/futures_weekly_review.py"))
    hyp = tmp_path / "hyp.jsonl"
    monkeypatch.setattr(weekly, "TRADE_LOG", log)
    monkeypatch.setattr(weekly, "HYP_LOG", hyp)
    monkeypatch.setattr("sys.argv", ["x", "--days", "3650"])
    assert weekly.main() == 0
    proposed = [json.loads(l) for l in hyp.read_text().splitlines() if l.strip()]
    cvd = [h for h in proposed if h["cut"] == "cvd_quality"]
    assert cvd and all(h["status"] == "PROPOSED" for h in proposed)
    assert any(h["predictor"] == "POSITIVE" and h["bucket"] == "HIGH" for h in cvd)


def test_killzone_agreement(tmp_path, monkeypatch):
    import json as _json
    oracle = tmp_path / "oracle_mornings.jsonl"
    rows = [
        {"date": "2026-06-09", "synthesis_type": "daily_opus", "instrument": "MES", "bias": "LONG"},
        {"date": "2026-06-09", "synthesis_type": "killzone_sonnet", "killzone": "NY_AM", "instrument": "MES", "bias": "LONG"},   # agree
        {"date": "2026-06-09", "synthesis_type": "killzone_sonnet", "killzone": "NY_PM", "instrument": "MES", "bias": "SHORT"},  # disagree
        {"date": "2026-06-09", "synthesis_type": "killzone_sonnet", "killzone": "LONDON", "instrument": "MES", "bias": "NO_PREDICTION"},  # n/a
    ]
    oracle.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    nightly = _load(Path("scripts/futures_nightly_review.py"))
    monkeypatch.setattr(nightly, "ORACLE_LOG", oracle)
    out = nightly._killzone_agreement("2026-06-09")
    assert out["agreements"] == 1 and out["disagreements"] == 1
    assert out["daily_call_present"] is True
    assert any(c["agree"] is None for c in out["comparisons"])   # NO_PREDICTION -> n/a
    assert nightly._killzone_agreement("2026-01-01") is None       # no killzone rows that day


def test_review_empty_safe(tmp_path, monkeypatch):
    log = tmp_path / "empty.jsonl"
    nightly = _load(Path("scripts/futures_nightly_review.py"))
    monkeypatch.setattr(nightly, "TRADE_LOG", log)
    monkeypatch.setattr(nightly, "ORACLE_LOG", tmp_path / "o.jsonl")
    monkeypatch.setattr(nightly, "OUT", tmp_path / "n.jsonl")
    monkeypatch.setattr("sys.argv", ["x"])
    assert nightly.main() == 0          # no trades -> no crash
