"""Tests for the read-only trade-context assembler.

The assembler's whole value is that it never lies about what it could read.
These tests pin that: instrument identity, the noise floor, the closed-door
filter, and — most importantly — that a missing source degrades loudly rather
than silently returning an empty-but-confident packet.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sovereign.context import trade_context as tc


# ─── instrument identity ─────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,symbol,base,quote", [
    ("EURUSD",  "EURUSD", "EUR", "USD"),
    ("eur_usd", "EURUSD", "EUR", "USD"),
    ("EUR/USD", "EURUSD", "EUR", "USD"),
    ("GBPJPY",  "GBPJPY", "GBP", "JPY"),
    ("AUDNZD",  "AUDNZD", "AUD", "NZD"),
])
def test_fx_pairs_parse_to_one_identity(raw, symbol, base, quote):
    inst = tc.parse_instrument(raw)
    assert inst.symbol == symbol
    assert inst.asset_class == "forex"
    assert (inst.base, inst.quote) == (base, quote)
    assert "carry" in inst.tokens        # carry lessons must reach FX queries


def test_equity_is_not_mistaken_for_a_pair():
    inst = tc.parse_instrument("SPY")
    assert inst.asset_class == "equity_or_etf"
    assert inst.base is None and inst.quote is None
    assert "carry" not in inst.tokens


def test_six_letters_that_are_not_currencies_stay_equity():
    inst = tc.parse_instrument("GOOGLX")
    assert inst.asset_class == "equity_or_etf"


# ─── verdict classification ──────────────────────────────────────────────────

@pytest.mark.parametrize("status,expected", [
    ("**CONFIRMED, LIVE** — OOS Sharpe 1.25", "CONFIRMED"),
    ("NOT_SIGNIFICANT", "CLOSED"),
    ("REJECTED_OOS, p=0.50", "CLOSED"),
    ("**NOT AN EDGE** — one regime", "CLOSED"),
    ("**FRAGILE** — OOS 0.22", "CLOSED"),
    ("IC_ONLY", "CLOSED"),
    ("in research", "OPEN"),
])
def test_declassify_reads_the_ledgers_own_vocabulary(status, expected):
    assert tc._declassify(status) == expected


def test_fragile_is_closed_not_confirmed():
    """A FRAGILE benchmark is not something to trade on. HYP-115's whole point."""
    assert tc._declassify("**FRAGILE** — OOS 2007-14 Sharpe 0.22") == "CLOSED"


# ─── markdown table parsing ──────────────────────────────────────────────────

def test_md_rows_skips_separators_and_prose():
    text = "prose\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\nmore prose\n"
    assert tc._md_rows(text) == [["a", "b"], ["1", "2"], ["3", "4"]]


# ─── the real repo record ────────────────────────────────────────────────────

def test_edge_ledger_finds_the_one_confirmed_live_edge():
    inst = tc.parse_instrument("EURUSD")
    edges = tc.edge_context(inst)
    assert edges["available"], edges.get("reason")
    claims = " ".join(r["claim"] + r["status"] for r in edges["confirmed"]).lower()
    assert "carry" in claims, "v015 carry is the only CONFIRMED live edge; it must surface"


def test_the_retraction_travels_with_every_packet():
    """The operator asked that the fade never be described as an edge again.
    The packet carries the retraction verbatim so nothing downstream can forget."""
    edges = tc.edge_context(tc.parse_instrument("SPY"))
    assert edges["retraction"], "EDGE_LEDGER retraction paragraph must be extracted"
    assert "not an edge" in edges["retraction"].lower()


def test_closed_doors_are_populated_for_forex():
    inst = tc.parse_instrument("EURUSD")
    doors = tc.closed_doors(tc.edge_context(inst), tc.lessons_context(inst))
    assert doors, "FX has refuted hypotheses on record; the list must not be empty"
    assert all({"source", "what", "verdict"} <= set(d) for d in doors)


def test_risk_caps_are_read_from_the_constitution_not_hardcoded():
    risk = tc.risk_context()
    assert risk["available"], risk.get("reason")
    assert any("0.75%" in c for c in risk["clauses"]), \
        "per-trade cap must be quoted from RISK_CONSTITUTION.md at call time"


# ─── degradation is loud, never silent ───────────────────────────────────────

def test_missing_source_reports_the_reason(monkeypatch):
    monkeypatch.setattr(tc, "EDGE_LEDGER", Path("/nonexistent/EDGE_LEDGER.md"))
    edges = tc.edge_context(tc.parse_instrument("EURUSD"))
    assert edges["available"] is False
    assert edges["reason"], "an unreadable source must say why, not return empty"


def test_packet_flags_degradation_in_warnings_and_completeness(monkeypatch):
    monkeypatch.setattr(tc, "EDGE_LEDGER", Path("/nonexistent/a.md"))
    monkeypatch.setattr(tc, "LESSONS", Path("/nonexistent/b.md"))
    monkeypatch.setattr(tc, "RISK_CONSTITUTION", Path("/nonexistent/c.md"))
    p = tc.build_packet("EURUSD", with_library=False)
    assert p.completeness in ("DEGRADED", "PARTIAL")
    assert len(p.warnings) >= 4
    assert any("Edge ledger unavailable" in w for w in p.warnings)


def test_library_below_floor_abstains_rather_than_calling_a_regime(monkeypatch):
    """A 0.067 similarity once fired a real ICT veto. Below the floor the
    packet must say UNKNOWN and restore neutral sizing — never a false call."""
    class _Insight:
        primary_regime = "PANDEMIC_SHUTDOWN"
        primary_volume = "VOLUME_IX_GEOPOLITICAL"
        primary_similarity = 0.067
        threat_score = 0.9
        threat_level = "DANGER"
        size_modifier = 0.25
        converging_signal = True
        advisory = "reduce exposure"
        action_summary = "cut"
        top_matches: list = []

    class _Lib:
        n_patterns = 63
        n_volumes = 10
        def query(self, *a, **k):
            return _Insight()

    import sovereign.risk.alexandrian_library as real
    monkeypatch.setattr(real, "AlexandrianLibrary", lambda: _Lib())
    monkeypatch.setattr(tc, "_price_arrays",
                        lambda offline=False: ([0.0] * 400, None, None, None, ["test"]))

    lib = tc.library_context()
    assert lib["available"] is True
    assert lib["above_floor"] is False
    assert lib["primary_regime"] == "UNKNOWN"
    assert lib["size_modifier"] == 1.0
    assert "below the" in lib["advisory"]


def test_library_import_failure_is_reported_not_swallowed(monkeypatch):
    monkeypatch.setattr(tc, "_price_arrays",
                        lambda offline=False: (None, None, None, None, []))
    lib = tc.library_context()
    assert lib["available"] is False
    assert lib["reason"]


# ─── packet shape / serialisation ────────────────────────────────────────────

def test_packet_is_json_serialisable_for_agent_handoff():
    p = tc.build_packet("EURUSD", with_library=False)
    blob = json.dumps(p.to_dict(), default=str)
    back = json.loads(blob)
    assert set(back) == {
        "instrument", "asof", "library", "edges", "lessons",
        "closed_doors", "risk", "live", "warnings", "completeness",
    }


def test_markdown_render_names_the_instrument_and_completeness():
    p = tc.build_packet("GBPUSD", with_library=False)
    md = tc.to_markdown(p)
    assert md.startswith("# Trade context — GBPUSD")
    assert p.completeness in md
    assert "Closed doors" in md
    assert "Absence of a closed door is not evidence of an edge" in md


def test_cli_json_mode_writes_a_parseable_packet(capsys):
    assert tc.main(["EURUSD", "--json", "--no-library"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["instrument"]["symbol"] == "EURUSD"


# ─── isolation law ───────────────────────────────────────────────────────────

def test_module_imports_nothing_from_ict():
    """sovereign/ may read sovereign/; it must not become a second ICT bridge."""
    src = Path(tc.__file__).read_text()
    assert "from ict" not in src and "import ict" not in src
