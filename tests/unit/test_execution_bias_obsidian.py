"""Obsidian writer safety + bias scoring."""
import pytest
from datetime import date
from execution import obsidian
from execution.bias import Bias, write_bias, score_bias, track_record, derive_bias
from execution.context import Status

def test_refuses_to_escape_trading_dir():
    with pytest.raises(ValueError, match="refusing to write outside"):
        obsidian._safe_path("../../../personal/secrets.md")

def test_dry_run_touches_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(obsidian, "VAULT", tmp_path / "nope")
    monkeypatch.setattr(obsidian, "TRADING", tmp_path / "nope" / "Trading")
    p, text = obsidian.write_note("Ops/x.md", "body", day="2026-07-18",
                                  title="T", dry_run=True)
    assert not p.exists()
    assert "body" in text

def test_missing_vault_raises_not_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(obsidian, "VAULT", tmp_path / "gone")
    monkeypatch.setattr(obsidian, "TRADING", tmp_path / "gone" / "Trading")
    with pytest.raises(obsidian.VaultUnavailable):
        obsidian.write_note("Ops/x.md", "b", day="2026-07-18", title="T")

def test_append_never_truncates(tmp_path, monkeypatch):
    monkeypatch.setattr(obsidian, "VAULT", tmp_path)
    monkeypatch.setattr(obsidian, "TRADING", tmp_path / "Trading")
    (tmp_path / "Trading").mkdir(parents=True)
    obsidian.append_block("Oracle-Log/d.md", "## first", day="2026-07-18", title="T")
    obsidian.append_block("Oracle-Log/d.md", "## second", day="2026-07-18", title="T")
    txt = (tmp_path / "Trading/Oracle-Log/d.md").read_text()
    assert "## first" in txt and "## second" in txt
    assert txt.count("---") >= 2   # frontmatter written once

def test_frontmatter_matches_template_shape():
    fm = obsidian.frontmatter("2026-07-18", "Oracle Log — 2026-07-18", "trading-log")
    for k in ("date:", "title:", "type:"):
        assert k in fm
    assert fm.startswith("---")

# ── bias scoring ──
def test_bias_scoring_records_outcome(tmp_path):
    b = Bias(date="2026-07-18", direction="BULLISH", confidence=0.5)
    write_bias(b, tmp_path)
    scored = score_bias("2026-07-18", "BULLISH", out_dir=tmp_path)
    assert scored.realised["correct"] is True
    scored2 = score_bias("2026-07-18", "BEARISH", out_dir=tmp_path)
    assert scored2.realised["correct"] is False

def test_neutral_is_unscoreable_not_wrong(tmp_path):
    b = Bias(date="2026-07-18", direction="NEUTRAL", confidence=0.0)
    write_bias(b, tmp_path)
    s = score_bias("2026-07-18", "BULLISH", out_dir=tmp_path)
    assert s.realised["correct"] is None, "NEUTRAL must not count as a miss"

def test_track_record_reports_n(tmp_path):
    for i, (d, real) in enumerate([("BULLISH","BULLISH"),("BEARISH","BULLISH"),("NEUTRAL","BULLISH")]):
        day = f"2026-07-{10+i:02d}"
        write_bias(Bias(date=day, direction=d, confidence=0.5), tmp_path)
        score_bias(day, real, out_dir=tmp_path)
    rec = track_record(tmp_path)
    assert rec["n_total"] == 3
    assert rec["n_scored"] == 2      # NEUTRAL excluded
    assert rec["hit_rate"] == 0.5

def test_confidence_scaled_by_context_freshness():
    """A bias built on a dark morning must be stated weakly, arithmetically."""
    ctx = {"date": "2026-07-18",
           "fields": {"briefing": {"status": Status.FRESH.value,
                                   "value": {"directional_bias": "BULLISH",
                                             "confidence": 0.8}}},
           "health": {"fraction_fresh": 0.25}}
    b = derive_bias(ctx)
    assert b.direction == "BULLISH"
    assert b.confidence == pytest.approx(0.2)   # 0.8 * 0.25

def test_stale_briefing_yields_neutral():
    ctx = {"date": "2026-07-18",
           "fields": {"briefing": {"status": Status.STALE.value,
                                   "value": {"directional_bias": "BULLISH",
                                             "confidence": 0.9}}},
           "health": {"fraction_fresh": 0.9}}
    b = derive_bias(ctx)
    assert b.direction == "NEUTRAL", "stale data must not produce a direction"
