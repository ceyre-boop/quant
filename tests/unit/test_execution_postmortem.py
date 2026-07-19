"""Post-mortem ledger: records, never infers; readiness cannot be flipped by code."""
import json
from datetime import date

import pytest

from execution import bias as bias_mod
from execution import postmortem as pm

DAY = date(2026, 6, 16)


def _fills(tmp_path, rows):
    d = tmp_path / "fill"; d.mkdir()
    with open(d / "fill_log.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return d


def _row(t, net=0.05, stype="LONG", sid=None):
    return {"date": str(DAY), "ticker": t, "hypothesis": "HYP-107",
            "signal_type": stype, "net_return": net, "gross_return": 0.06,
            "spread_cost": 0.01, "backtest_expected_return": 0.03,
            "signal_id": sid or f"{DAY}:{t}:HYP-107", "reason": "filled",
            "risk_action": "ALLOW", "risk_breached": [], "frozen_hash": "abc"}


# ── no inference ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("net,stype,expected", [
    (0.05, "LONG", "WIN"), (-0.05, "LONG", "LOSS"), (0.0, "LONG", "FLAT"),
    (None, "SKIP_HALT", "SKIPPED"), (None, "LONG", "UNKNOWN"),
])
def test_outcome_is_mechanical_only(net, stype, expected):
    assert pm.classify(net, stype) == expected


def test_module_generates_no_narrative_labels():
    """No 'why did this lose' inference. Labels of convenience are the trap.

    Checks IMPORTS and CALLS via AST, not source text. A substring scan fails on
    this module's own docstring, which explains at length why LLM-generated
    labels are dangerous -- a mention is not a use. Same lesson as the L1/L2 wall
    guard in test_bias_isolation.py.
    """
    import ast, inspect
    tree = ast.parse(inspect.getsource(pm))

    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            imported.add(base)
            imported.update(f"{base}.{a.name}" for a in n.names)
    for banned in ("anthropic", "openai", "Inference", "sovereign.oracle"):
        assert not any(banned in m for m in imported), (
            f"postmortem must not import an inference path ({banned})")

    called = {n.func.attr if isinstance(n.func, ast.Attribute) else
              (n.func.id if isinstance(n.func, ast.Name) else "")
              for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for banned in ("complete", "generate", "explain", "infer", "predict", "fit"):
        assert banned not in called, f"postmortem must not call {banned}()"


# ── readiness cannot be flipped ──────────────────────────────────────────────

def test_ready_for_modelling_is_never_true(tmp_path):
    """Reaching the row count is necessary, not sufficient. Code must not decide."""
    rows = [pm.PostMortem(signal_id=f"s{i}", date=str(DAY), ticker="X",
                          hypothesis="HYP-107", side="LONG",
                          filter_features={"gap": 0.4}, outcome="WIN",
                          net_return=0.05, is_live=True)
            for i in range(pm.MIN_ROWS_FOR_ANY_MODELLING + 50)]
    pm.record(rows, tmp_path)
    rd = pm.readiness(tmp_path)
    assert rd["n_live_feature_complete"] > rd["minimum_required"]
    assert rd["shortfall"] == 0
    assert rd["ready_for_modelling"] is False, (
        "even above the row count this must stay False — a prereg and the A3 "
        "placebo bar are also required, and code cannot assert those")


def test_replay_rows_excluded_from_readiness(tmp_path):
    """Backtest/replay must never inflate the live label count — that conflation
    is how 3,460 ICT replay records looked like live evidence."""
    pm.record([pm.PostMortem(signal_id="r", date=str(DAY), ticker="X",
                             hypothesis="HYP-107", side="LONG",
                             filter_features={"gap": 0.4}, outcome="WIN",
                             net_return=0.05, is_live=False)], tmp_path)
    rd = pm.readiness(tmp_path)
    assert rd["n_rows_total"] == 1
    assert rd["n_live_feature_complete"] == 0


def test_minimum_derives_from_power_analysis():
    from execution.drift import n_for_power
    assert pm.MIN_ROWS_FOR_ANY_MODELLING == n_for_power(0.60)


# ── source/outcome join ──────────────────────────────────────────────────────

def test_source_status_recorded_beside_outcome(tmp_path):
    """The join that PERMITS a future attribution study — it performs none."""
    fd = _fills(tmp_path, [_row("AAA")])
    cd = tmp_path / "ctx"; cd.mkdir()
    (cd / f"morning_context_{DAY}.json").write_text(json.dumps({
        "date": str(DAY),
        "health": {"n_sources": 7, "n_fresh": 2, "fraction_fresh": 0.29},
        "fields": {"fred_macro": {"status": "FRESH"}, "reddit": {"status": "SILENT_NULL"}},
    }))
    rows = pm.build(DAY, fill_dir=fd, ctx_dir=cd)
    assert rows[0].source_status == {"fred_macro": "FRESH", "reddit": "SILENT_NULL"}
    assert rows[0].context_health["fraction_fresh"] == 0.29


def test_skips_are_recorded_not_dropped(tmp_path):
    fd = _fills(tmp_path, [_row("AAA"), _row("BBB", net=None, stype="SKIP_NO_BORROW")])
    rows = pm.build(DAY, fill_dir=fd)
    assert len(rows) == 2
    assert {r.outcome for r in rows} == {"WIN", "SKIPPED"}


def test_feature_complete_requires_features_and_outcome():
    complete = pm.PostMortem(signal_id="a", date=str(DAY), ticker="X",
                             hypothesis="H", side="LONG",
                             filter_features={"gap": 0.4}, outcome="WIN",
                             net_return=0.05)
    assert complete.feature_complete is True
    no_feats = pm.PostMortem(signal_id="b", date=str(DAY), ticker="X",
                             hypothesis="H", side="LONG", outcome="WIN",
                             net_return=0.05)
    assert no_feats.feature_complete is False


# ── bias realised-direction ──────────────────────────────────────────────────

def test_neutral_band_is_definitional_not_fitted():
    """Tuning this after scoring would invalidate every prior score."""
    assert bias_mod.NEUTRAL_BAND == 0.0025
    assert bias_mod.REALISED_PROXY == "SPY"


def test_unknown_direction_is_not_neutral(monkeypatch):
    """An unscoreable day must not be recorded as NEUTRAL — NEUTRAL is an answer,
    UNKNOWN is the absence of one."""
    from execution import alpaca
    monkeypatch.setattr(alpaca, "daily_prev_close", lambda *a, **k: None)
    monkeypatch.setattr(alpaca, "minute_bars", lambda *a, **k: [])
    d, detail = bias_mod.realised_direction(DAY)
    assert d == "UNKNOWN"
    assert "no proxy bars" in detail["reason"]
