"""Layer 1 — the degrade-never-fabricate contract.

The rule these tests defend: a source that fails must never look like a source
that reported zero.

These originally exercised the contract through `load_reddit`, because
`data/cache/reddit_sentiment.json` was the live example — timestamp minutes old,
`posts_scanned: 0` — and downstream code recorded that emptiness as data. Reddit
was RETIRED on 2026-07-20 (403 on both hosts, no OAuth app), so the contract is now
carried by `load_fred`, which still produces all five statuses. The vehicle changed;
the contract did not, and it must keep a live carrier or the coverage silently
lapses with whichever source is retired next.
"""
import json
from datetime import date

import pytest

from execution import context
from execution.context import Status, build_morning_context, load_fred, load_reddit


def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def test_silent_null_is_distinct_from_fresh(tmp_path, monkeypatch):
    """A source that ran, succeeded, and returned nothing is SILENT_NULL —
    even though its file is seconds old."""
    monkeypatch.setattr(context, "ROOT", tmp_path)
    _write(tmp_path / "data/macro/fred_economic_latest.json",
           {"date": "2026-07-18", "metrics": {}})
    f = load_fred()
    assert f.status is Status.SILENT_NULL
    assert f.value is None, "SILENT_NULL must not carry a fabricated value"
    assert f.age_seconds is not None and f.age_seconds < 3600, (
        "the file IS fresh — freshness alone would have called this healthy")


def test_populated_source_is_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "ROOT", tmp_path)
    _write(tmp_path / "data/macro/fred_economic_latest.json",
           {"date": "2026-07-18", "metrics": {"UNRATE": 4.1, "PAYEMS": 150000}})
    f = load_fred()
    assert f.status is Status.FRESH
    assert len(f.value["metrics"]) == 2


def test_missing_file_is_unavailable_not_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "ROOT", tmp_path)
    f = load_fred()
    assert f.status is Status.UNAVAILABLE
    assert f.value is None
    assert f.value != 0, "a missing reading must never become a zero reading"


def test_corrupt_source_is_error_not_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "ROOT", tmp_path)
    p = tmp_path / "data/macro/fred_economic_latest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json")
    f = load_fred()
    assert f.status is Status.ERROR
    assert "JSONDecodeError" in f.detail or "Expecting" in f.detail


@pytest.mark.parametrize("status", list(Status))
def test_only_fresh_is_usable(status):
    """STALE data is readable but using it must be a deliberate choice."""
    f = context.Field_(name="x", value={"a": 1}, status=status, source="t")
    assert f.usable is (status is Status.FRESH)


def test_context_reports_honest_health(tmp_path, monkeypatch):
    """With every source absent the context must report 0 fresh, not fake success."""
    monkeypatch.setattr(context, "ROOT", tmp_path)
    ctx = build_morning_context(date(2026, 7, 18))
    assert ctx["health"]["n_fresh"] == 0
    assert ctx["health"]["fraction_fresh"] == 0.0
    assert all(f["value"] is None for f in ctx["fields"].values())


def test_every_field_carries_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "ROOT", tmp_path)
    ctx = build_morning_context(date(2026, 7, 18))
    for name, f in ctx["fields"].items():
        assert f["status"] in {s.value for s in Status}, name
        assert f["source"], f"{name} must name its source"
        assert "value" in f, name


def test_all_declared_sources_present(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "ROOT", tmp_path)
    ctx = build_morning_context(date(2026, 7, 18))
    assert set(ctx["fields"]) == set(context.SOURCES)


def test_staleness_budget_flags_old_data(tmp_path, monkeypatch):
    """Old-but-real data stays readable, flagged STALE — not discarded."""
    monkeypatch.setattr(context, "ROOT", tmp_path)
    p = tmp_path / "data/macro/fred_economic_latest.json"
    _write(p, {"date": "2026-07-18", "metrics": {"UNRATE": 4.1}})
    import os, time
    old = time.time() - context.BUDGET_HOURS["fred_macro"] * 3600 - 60
    os.utime(p, (old, old))
    f = load_fred()
    assert f.status is Status.STALE
    assert f.value is not None, "STALE keeps its value; only FRESH-ness is lost"


# ── retirement ────────────────────────────────────────────────────────────────

def test_retired_source_is_not_silent_null(tmp_path, monkeypatch):
    """Reddit is RETIRED, not failing. The distinction matters: SILENT_NULL says
    'we asked and got nothing', RETIRED says 'we stopped asking, and here is why'."""
    monkeypatch.setattr(context, "ROOT", tmp_path)
    f = load_reddit()
    assert f.status is Status.RETIRED
    assert f.value is None
    assert "403" in f.detail, "the retirement reason must travel with the field"


def test_retired_source_leaves_the_health_denominator(tmp_path, monkeypatch):
    """A source nobody asks for cannot hold health permanently below 100%."""
    monkeypatch.setattr(context, "ROOT", tmp_path)
    ctx = build_morning_context(date(2026, 7, 20))
    assert ctx["health"]["n_sources"] == len(context.SOURCES) - 1
    assert "reddit" in ctx["health"]["retired"]
    assert ctx["health"]["n_retired"] == 1
