"""Layer 6 — reconciliation arithmetic and the unfilled-GO headline."""
import json
from datetime import date

import pytest

from execution import eod
from execution.eod import reconcile, render

DAY = date(2026, 6, 16)


def _setup(tmp_path, signals, fills):
    sd, fd = tmp_path / "sig", tmp_path / "fill"
    sd.mkdir(); fd.mkdir()
    (sd / f"signals_{DAY}.json").write_text(json.dumps(
        {"date": str(DAY), "n_go": sum(1 for s in signals if s["decision"] == "GO"),
         "n_no_go": sum(1 for s in signals if s["decision"] != "GO"),
         "signals": signals}))
    with open(fd / "fill_log.jsonl", "w") as fh:
        for f in fills:
            fh.write(json.dumps(f) + "\n")
    return sd, fd


def _sig(t, h="HYP-107", d="GO"):
    return {"signal_id": f"{DAY}:{t}:{h}", "date": str(DAY), "ticker": t,
            "hypothesis": h, "decision": d, "reason": "pass", "rank": 1}


def _fill(t, h="HYP-107", stype="LONG", net=0.05, exp=0.03, sid=None, breached=None):
    return {"date": str(DAY), "ticker": t, "hypothesis": h, "signal_type": stype,
            "net_return": net, "backtest_expected_return": exp,
            "spread_cost": 0.01, "signal_id": sid or f"{DAY}:{t}:{h}",
            "risk_breached": breached or [], "risk_action": "ALLOW"}


def test_unfilled_go_is_detected(tmp_path):
    """The headline number: a GO that never became an attempt."""
    sd, fd = _setup(tmp_path, [_sig("AAA"), _sig("BBB")], [_fill("AAA")])
    r = reconcile(DAY, fill_dir=fd, signal_dir=sd)
    assert r["conversion"]["n_go"] == 2
    assert r["conversion"]["n_go_unfilled"] == 1
    assert r["conversion"]["unfilled_detail"][0]["ticker"] == "BBB"


def test_zero_go_is_reported_as_a_real_zero(tmp_path):
    sd, fd = _setup(tmp_path, [_sig("AAA", d="NO_GO")], [])
    r = reconcile(DAY, fill_dir=fd, signal_dir=sd)
    assert r["conversion"]["n_go"] == 0
    assert r["conversion"]["fill_rate"] is None      # not 0.0 — undefined
    assert "real zero" in render(r)


def test_skips_are_counted_by_reason(tmp_path):
    sd, fd = _setup(tmp_path, [_sig("AAA")],
                    [_fill("AAA", stype="SKIP_HALT"),
                     _fill("BBB", stype="SKIP_NO_BORROW")])
    r = reconcile(DAY, fill_dir=fd, signal_dir=sd)
    assert r["conversion"]["skip_reasons"]["SKIP_HALT"] == 1
    assert r["conversion"]["skip_reasons"]["SKIP_NO_BORROW"] == 1
    assert r["conversion"]["n_filled"] == 0


def test_skips_excluded_from_performance(tmp_path):
    sd, fd = _setup(tmp_path, [_sig("AAA"), _sig("BBB")],
                    [_fill("AAA", net=0.05), _fill("BBB", stype="SKIP_RISK", net=None)])
    r = reconcile(DAY, fill_dir=fd, signal_dir=sd)
    assert r["performance"]["n"] == 1
    assert r["performance"]["median_net"] == pytest.approx(0.05)


def test_vs_backtest_delta_arithmetic(tmp_path):
    sd, fd = _setup(tmp_path, [_sig("AAA")], [_fill("AAA", net=0.05, exp=0.03)])
    r = reconcile(DAY, fill_dir=fd, signal_dir=sd)
    assert r["performance"]["vs_backtest_delta"] == pytest.approx(0.02)


def test_risk_gate_firing_is_surfaced(tmp_path):
    sd, fd = _setup(tmp_path, [_sig("AAA")],
                    [_fill("AAA", stype="SKIP_RISK",
                           breached=["ART1_PER_TRADE 0.02>0.0075"])])
    r = reconcile(DAY, fill_dir=fd, signal_dir=sd)
    assert r["risk"]["n_gates_fired"] == 1
    assert "ART1_PER_TRADE" in render(r)


def test_render_warns_when_inputs_were_dark(tmp_path, monkeypatch):
    sd, fd = _setup(tmp_path, [_sig("AAA")], [_fill("AAA")])
    cd = tmp_path / "ctx"; cd.mkdir()
    (cd / f"morning_context_{DAY}.json").write_text(json.dumps(
        {"date": str(DAY),
         "health": {"n_sources": 7, "n_fresh": 2, "fraction_fresh": 0.29,
                    "by_status": {"FRESH": ["a", "b"], "UNAVAILABLE": ["c"]}}}))
    r = reconcile(DAY, fill_dir=fd, signal_dir=sd, ctx_dir=cd)
    body = render(r)
    assert "2/7 sources FRESH" in body
    assert "Most inputs were not live today" in body


def test_render_is_pure_markdown_no_crash_on_empty(tmp_path):
    sd, fd = _setup(tmp_path, [], [])
    body = render(reconcile(DAY, fill_dir=fd, signal_dir=sd))
    assert "## Conversion" in body and "## Risk gates" in body


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    from execution import obsidian
    monkeypatch.setattr(obsidian, "VAULT", tmp_path / "vault")
    monkeypatch.setattr(obsidian, "TRADING", tmp_path / "vault" / "Trading")
    p, text = obsidian.write_eod_note(DAY, "body", dry_run=True)
    assert not p.exists() and "body" in text
