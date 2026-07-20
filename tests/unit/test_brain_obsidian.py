"""
Tests for the Obsidian intelligence layer (sovereign/brain).

Covers: graceful degradation on a missing vault, the write->read round-trip for
each writer, and the ICT/sovereign-execution isolation invariant.
"""

import importlib
import os

import pytest


@pytest.fixture()
def brain(tmp_path, monkeypatch):
    """Point the brain at an empty temp vault and reload its path module."""
    monkeypatch.setenv("ALTA_VAULT", str(tmp_path / "vault"))
    monkeypatch.setenv("ALTA_REPO", str(tmp_path / "repo"))
    from sovereign.brain import _paths as P

    importlib.reload(P)
    from sovereign.brain import obsidian_reader, obsidian_writer

    importlib.reload(obsidian_reader)
    importlib.reload(obsidian_writer)
    yield obsidian_reader, obsidian_writer, P
    # Restore defaults for any later test module.
    for mod in ("ALTA_VAULT", "ALTA_REPO"):
        os.environ.pop(mod, None)
    importlib.reload(P)


def test_reader_never_crashes_on_missing_vault(brain):
    reader, _, _ = brain
    # Nothing exists yet — every function must return an empty structure.
    assert reader.load_recent_verdicts() == []
    assert reader.load_weakness_log() == []
    assert reader.load_trading_psychology() == []
    assert reader.load_edge_summary()["confirmed"] == []
    assert reader.load_regime_context()["regime_notes"] == []
    mc = reader.get_morning_context()
    assert mc["active_edges"] == [] and mc["recent_verdicts"] == []
    assert reader.get_research_context()["graveyard"] == []


def test_weakness_roundtrip(brain):
    reader, writer, _ = brain
    assert writer.write_weakness_note("overtrading", "4 trades in a VIX spike", "2026-07-19")
    log = reader.load_weakness_log()
    assert len(log) == 1
    assert log[0]["date"] == "2026-07-19"
    assert log[0]["type"] == "overtrading"
    assert "VIX spike" in log[0]["description"]
    # And the human-facing psychology view renders it.
    assert any("overtrading" in s for s in reader.load_trading_psychology())


def test_regime_roundtrip(brain):
    reader, writer, _ = brain
    assert writer.write_regime_observation("USDJPY", "above 158 post-BoJ", "scan")
    notes = reader.load_regime_context()["regime_notes"]
    assert any("USDJPY" in n and "158" in n for n in notes)


def test_verdict_and_briefs_write(brain):
    _, writer, P = brain
    assert writer.write_verdict("HYP-999", "NOT_SIGNIFICANT", {"p_value": 0.4}, "none", "smoke")
    assert P.VERDICT_LOG.exists()
    assert "HYP-999" in P.VERDICT_LOG.read_text()

    assert writer.write_morning_brief(["EURUSD SHORT"], "carry-positive", ["HYP-045"])
    assert writer.write_eod_summary(fills=2, pnl="+0.3%", notes="quiet", lessons=["held discipline"])
    ops = list((P.OPS).glob("Agent-Log-*.md"))
    assert ops, "morning/eod should create an Ops agent log"
    text = ops[0].read_text()
    assert "MORNING BRIEF" in text and "EOD SUMMARY" in text and "held discipline" in text


def test_brain_isolation_no_ict_import():
    """The brain must never import ict/ or the live execution path (NN#1)."""
    import pathlib

    brain_dir = pathlib.Path(__file__).resolve().parents[2] / "sovereign" / "brain"
    for py in brain_dir.glob("*.py"):
        src = py.read_text()
        for line in src.splitlines():
            s = line.strip()
            if s.startswith(("import ict", "from ict")) or "import ict." in s:
                pytest.fail(f"{py.name} imports ict: {s}")
