"""Tests for the swap-measurement harness.

This harness exists to answer one question with money on it: does the broker's
take leave anything of the carry premium. The tests pin the ways that answer
could be quietly wrong — a silent zero, a diluted average, a missing quote read
as free money, or an order path appearing in a read-only tool.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sovereign.financing import swap_probe as sp


def _pf(pair="EUR_USD", long_rate=-0.0248, short_rate=0.0046):
    return sp.PairFinancing(pair=pair, long_rate=long_rate, short_rate=short_rate,
                            weekly_days_charged=7, triple_day="WEDNESDAY")


# ─── the carry side, and the broker's take ───────────────────────────────────

def test_carry_side_is_the_side_financing_favours():
    assert _pf(long_rate=-0.0248, short_rate=0.0046).carry_side == "SHORT"
    assert _pf(long_rate=0.0100, short_rate=-0.0300).carry_side == "LONG"


def test_carry_rate_is_the_better_side_not_the_average():
    p = _pf(long_rate=-0.0248, short_rate=0.0046)
    assert p.carry_rate == pytest.approx(0.0046)


def test_both_sides_can_cost_and_carry_rate_stays_negative():
    """GBPUSD on 2026-09-20: −1.22% long, −0.81% short. There is no paying side."""
    p = _pf("GBP_USD", -0.0122, -0.0081)
    assert p.carry_rate == pytest.approx(-0.0081)
    assert p.carry_rate < 0, "no side pays; the report must not imply one does"


def test_broker_take_is_the_shortfall_from_symmetry():
    p = _pf(long_rate=-0.0248, short_rate=0.0046)
    assert p.broker_take == pytest.approx(-0.0202)


# ─── the silent zero ─────────────────────────────────────────────────────────

def test_zero_on_both_sides_is_an_absent_quote_not_zero_carry():
    assert _pf("GBP_JPY", 0.0, 0.0).quoted is False


def test_a_real_zero_on_one_side_is_still_a_quote():
    assert _pf("X_Y", 0.0, -0.0100).quoted is True


def _write_snaps(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _snap(date: str, pairs: dict[str, sp.PairFinancing]) -> dict:
    return {"date": date, "captured_at": f"{date}T00:00:00+00:00", "mode": "practice",
            "pairs": {k: v.to_dict() for k, v in pairs.items()}}


def test_unquoted_pair_is_excluded_from_the_book_average(tmp_path):
    """The failure this prevents: GBP_JPY's absent quote read as 0%/yr, pulling a
    −1.00% book toward −0.75% and making the broker look cheaper than it is."""
    path = tmp_path / "s.jsonl"
    _write_snaps(path, [_snap("2026-09-01", {
        "EUR_USD": _pf("EUR_USD", -0.02, -0.01),    # carry −1.00%/yr
        "GBP_JPY": _pf("GBP_JPY", 0.0, 0.0),        # absent
    })])
    rep = sp.build_report(path)
    assert rep.portfolio_carry_pct == pytest.approx(-1.00)
    byp = {v.pair: v for v in rep.pairs}
    assert byp["GBP_JPY"].verdict == "NOT_QUOTED"
    assert any("NOT QUOTED" in w for w in rep.warnings)
    assert any("3 of 4" in w or "1 of 2" in w for w in rep.warnings)


def test_all_pairs_unquoted_gives_no_average_rather_than_zero(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_snaps(path, [_snap("2026-09-01", {"GBP_JPY": _pf("GBP_JPY", 0.0, 0.0)})])
    rep = sp.build_report(path)
    assert rep.portfolio_carry_pct is None, "no quotes must not average to 0%"


def test_old_snapshots_without_the_quoted_field_are_still_classified(tmp_path):
    path = tmp_path / "s.jsonl"
    row = _pf("GBP_JPY", 0.0, 0.0).to_dict()
    row.pop("quoted")
    _write_snaps(path, [{"date": "2026-09-01", "mode": "practice",
                         "pairs": {"GBP_JPY": row}}])
    rep = sp.build_report(path)
    assert rep.pairs[0].verdict == "NOT_QUOTED"


# ─── model comparison (TICK-024) ─────────────────────────────────────────────

def test_sign_flip_against_the_modelled_table_is_named(tmp_path):
    """EURUSD SHORT: the model says pay, the broker says earn. TICK-024's finding."""
    path = tmp_path / "s.jsonl"
    _write_snaps(path, [_snap("2026-09-01", {"EUR_USD": _pf("EUR_USD", -0.0248, 0.0046)})])
    v = sp.build_report(path).pairs[0]
    assert v.carry_side == "SHORT"
    assert v.sign_agrees is False
    assert "SIGN FLIP" in v.note


def test_magnitude_gap_against_the_model_is_reported(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_snaps(path, [_snap("2026-09-01", {"GBP_USD": _pf("GBP_USD", -0.0122, -0.0081)})])
    v = sp.build_report(path).pairs[0]
    assert v.model_ratio is not None and v.model_ratio > 3
    assert "off by" in v.note


def test_modelled_table_is_quoted_not_imported():
    """Importing the backtester would put an execution-path module in this tool's
    import graph. The table is copied, with its source named in a string instead.

    Checked on the AST, not on the text — the module docstring legitimately names
    forex_backtester, and a substring test would either fail on that or pass on a
    real import hidden in a comment."""
    import ast
    tree = ast.parse(Path(sp.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    banned = ("forex_backtester", "exit_machine", "entry_engine", "forex_exit_manager",
              "sovereign.forex", "sovereign.execution")
    assert not [m for m in imported for b in banned if b in m], sorted(imported)
    assert sp.MODELLED_SOURCE.endswith("SWAP_RATES_ANNUAL")


# ─── the verdict ─────────────────────────────────────────────────────────────

def test_verdict_says_no_trade_when_financing_exceeds_the_premium(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_snaps(path, [_snap("2026-09-01", {"EUR_USD": _pf("EUR_USD", -0.10, -0.06)})])
    rep = sp.build_report(path, premium_pct=5.0)
    assert "NO TRADE AT THIS SIZE" in rep.headline


def test_verdict_says_the_premium_survives_when_financing_pays(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_snaps(path, [_snap("2026-09-01", {"EUR_USD": _pf("EUR_USD", -0.02, 0.01)})])
    rep = sp.build_report(path, premium_pct=5.0)
    assert "survives the broker" in rep.headline


def test_headline_is_marked_provisional_below_the_day_floor(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_snaps(path, [_snap("2026-09-01", {"EUR_USD": _pf()})])
    rep = sp.build_report(path)
    assert rep.enough_data is False
    assert rep.headline.startswith("PROVISIONAL")


def test_enough_data_flips_at_the_declared_floor(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_snaps(path, [_snap(f"2026-09-{d:02d}", {"EUR_USD": _pf()})
                        for d in range(1, sp.MIN_DAYS_FOR_VERDICT + 1)])
    rep = sp.build_report(path)
    assert rep.n_days == sp.MIN_DAYS_FOR_VERDICT
    assert rep.enough_data is True
    assert not rep.headline.startswith("PROVISIONAL")


def test_no_snapshots_reports_no_data_not_a_verdict(tmp_path):
    rep = sp.build_report(tmp_path / "absent.jsonl")
    assert rep.headline.startswith("NO DATA")
    assert rep.portfolio_carry_pct is None


def test_a_side_flip_during_the_window_is_flagged(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_snaps(path, [
        _snap("2026-09-01", {"EUR_USD": _pf("EUR_USD", -0.02, 0.01)}),   # SHORT pays
        _snap("2026-09-02", {"EUR_USD": _pf("EUR_USD", 0.01, -0.02)}),   # LONG pays
    ])
    rep = sp.build_report(path)
    assert any("flipped" in w for w in rep.warnings)


# ─── snapshots ───────────────────────────────────────────────────────────────

def test_snapshot_is_idempotent_per_day(tmp_path, monkeypatch):
    path = tmp_path / "s.jsonl"
    monkeypatch.setattr(sp, "fetch_quoted", lambda pairs=None: [_pf()])
    sp.snapshot(["EUR_USD"], path)
    sp.snapshot(["EUR_USD"], path)
    sp.snapshot(["EUR_USD"], path)
    assert len(sp.load_snapshots(path)) == 1, \
        "re-running the same day must not inflate n with one reading"


def test_snapshot_records_nothing_when_the_broker_is_down(tmp_path, monkeypatch):
    path = tmp_path / "s.jsonl"

    def _boom(pairs=None):
        raise sp.BrokerUnavailable("HTTP 503")

    monkeypatch.setattr(sp, "fetch_quoted", _boom)
    with pytest.raises(sp.BrokerUnavailable):
        sp.snapshot(["EUR_USD"], path)
    assert sp.load_snapshots(path) == [], "a failed read must not write a row"


def test_missing_financing_block_raises_rather_than_defaulting(monkeypatch):
    monkeypatch.setattr(sp, "_get",
                        lambda p, q=None: {"instruments": [{"name": "EUR_USD"}]})
    with pytest.raises(sp.BrokerUnavailable, match="no financing"):
        sp.fetch_quoted(["EUR_USD"])


def test_corrupt_snapshot_lines_are_skipped_not_fatal(tmp_path):
    path = tmp_path / "s.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_snap("2026-09-01", {"EUR_USD": _pf()})) + "\n{oops\n")
    assert len(sp.load_snapshots(path)) == 1


# ─── read-only guarantee ─────────────────────────────────────────────────────

def test_there_is_no_order_path_in_the_harness():
    src = Path(sp.__file__).read_text()
    banned = ("/orders", "place_order", "create_order", "POST", "urlopen(req, data",
              "method=", "trades/close", "positions/close")
    assert not [b for b in banned if b in src]


def test_realized_needs_a_position_and_says_so_rather_than_opening_one(monkeypatch):
    monkeypatch.setattr(sp, "_get", lambda p, q=None: {"transactions": [], "pages": []})
    res = sp.realized(30)
    assert res["available"] is False
    assert "will not open it" in res["reason"]


def test_realized_reports_broker_failure_as_unavailable(monkeypatch):
    def _boom(p, q=None):
        raise sp.BrokerUnavailable("HTTP 401")
    monkeypatch.setattr(sp, "_get", _boom)
    assert sp.realized(30)["available"] is False


# ─── CLI ─────────────────────────────────────────────────────────────────────

def test_report_json_is_parseable(tmp_path, monkeypatch, capsys):
    path = tmp_path / "s.jsonl"
    _write_snaps(path, [_snap("2026-09-01", {"EUR_USD": _pf()})])
    monkeypatch.setattr(sp, "SNAPSHOT_PATH", path)
    assert sp.main(["report", "--json"]) == 0
    assert "headline" in json.loads(capsys.readouterr().out)


def test_render_never_prints_a_number_for_an_unquoted_pair(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_snaps(path, [_snap("2026-09-01", {
        "EUR_USD": _pf(), "GBP_JPY": _pf("GBP_JPY", 0.0, 0.0)})])
    out = sp.render(sp.build_report(path))
    line = [ln for ln in out.splitlines() if ln.startswith("GBP_JPY")][0]
    assert "NOT QUOTED" in line and "+0.000" not in line
