"""Tests for experience/precedents.py — L2a structured retrieval over canonical + annex."""
from experience import library_annex as la
from experience import precedent_service as svc
from experience import precedents as prec
from sovereign.sentiment import store


def _annex_entry(**kw):
    base = dict(entry_id="annex:review:2026-W27", volume=la.VOLUME_XI_LIVED,
                label="TEST_LIVED_EVENT", date="2026-07-02", description="desc",
                outcome="outcome", outcome_days=7, severity=0, tags=["zzz_unique_tag"],
                source_kind="review", source_ref="x")
    base.update(kw)
    return la.LivedEntry(**base)


class TestWeekBoardExtremes:
    def test_db_absent_or_locked_degrades_to_empty(self, monkeypatch):
        def _boom(**kwargs):
            raise RuntimeError("database is locked")
        monkeypatch.setattr(store, "connect", _boom)
        assert prec.week_board_extremes("2026-06-01", "2026-06-07") == []

    def test_query_failure_also_degrades_to_empty(self, monkeypatch):
        class FakeCon:
            def execute(self, *a, **kw):
                raise RuntimeError("no such table: sentiment_board_state")

            def close(self):
                pass
        monkeypatch.setattr(store, "connect", lambda **kw: FakeCon())
        assert prec.week_board_extremes("2026-06-01", "2026-06-07") == []

    def test_extremes_flagged_from_board_rows(self, monkeypatch):
        import pandas as pd

        class FakeResult:
            def __init__(self, df):
                self._df = df

            def df(self):
                return self._df

        class FakeCon:
            def execute(self, *a, **kw):
                return FakeResult(pd.DataFrame([
                    {"date": "2026-06-03", "pair": "EURUSD", "vix_level": 35.0,
                     "vix_regime": "SPIKE", "econ_surprise_z": 0.1, "cot_net_pct": 0.5,
                     "vrp_pct": 0.5, "macro_curve": 0.2},
                    {"date": "2026-06-04", "pair": "GBPUSD", "vix_level": 14.0,
                     "vix_regime": "NORMAL", "econ_surprise_z": 0.0, "cot_net_pct": 0.5,
                     "vrp_pct": 0.5, "macro_curve": 0.2},
                ]))

            def close(self):
                pass

        monkeypatch.setattr(store, "connect", lambda **kw: FakeCon())
        extremes = prec.week_board_extremes("2026-06-01", "2026-06-07")
        assert len(extremes) == 1                            # only the SPIKE/high-vix row breaches
        assert extremes[0]["pair"] == "EURUSD"
        assert "volatility" in extremes[0]["tags"]
        assert "vix_spike" in extremes[0]["tags"]


class TestWeekContextTags:
    def test_union_of_engine_thesis_cls_overlays_and_extremes(self):
        rows = [{"engine": "carry", "thesis": {"kind": "structural_carry"}}]
        atts = [{"cls": "thesis_confirmed", "overlays": ["execution_variance"]}]
        extremes = [{"tags": ["vix_spike"]}]
        tags = prec.week_context_tags(rows, atts, extremes)
        assert tags == {"carry", "structural_carry", "thesis_confirmed",
                        "execution_variance", "vix_spike"}

    def test_empty_inputs_yield_empty_set(self):
        assert prec.week_context_tags([], [], []) == set()


class TestFindPrecedents:
    def test_no_context_tags_returns_empty(self):
        assert prec.find_precedents(set(), top_k=3) == []

    def test_canonical_match_by_tag(self):
        results = prec.find_precedents({"carry"}, top_k=3)
        assert results
        assert all(r["source"] == "canonical" for r in results)
        assert all("carry" in r["matched_tags"] for r in results)
        assert len(results) <= 3
        expected_keys = {"entry_id", "source", "label", "event_date", "why", "what_followed",
                         "outcome_days", "severity", "matched_tags", "score"}
        assert set(results[0].keys()) == expected_keys

    def test_annex_and_canonical_retrieved_together(self, tmp_path, monkeypatch):
        monkeypatch.setattr(la, "ANNEX_PATH", tmp_path / "annex.jsonl")
        la.append_entries([_annex_entry(tags=["carry", "zzz_unique_tag"])])
        results = prec.find_precedents({"carry"}, top_k=10)
        sources = {r["source"] for r in results}
        assert "canonical" in sources
        assert "annex" in sources

    def test_top_k_respected_and_sorted_by_score_desc(self, tmp_path, monkeypatch):
        monkeypatch.setattr(la, "ANNEX_PATH", tmp_path / "annex.jsonl")
        results = prec.find_precedents({"carry", "low_vol"}, top_k=2)
        assert len(results) <= 2
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestPrecedentServiceStub:
    def test_stub_returns_empty_when_flag_off(self, monkeypatch):
        monkeypatch.setattr(prec, "decision_time_enabled", False)
        assert svc.query({"vix_regime": "SPIKE"}) == []

    def test_stub_queries_precedents_when_flag_on(self, monkeypatch):
        monkeypatch.setattr(prec, "decision_time_enabled", True)
        result = svc.query({"vix_regime": "SPIKE"})
        assert isinstance(result, list)

    def test_stub_empty_row_returns_empty_even_when_flag_on(self, monkeypatch):
        monkeypatch.setattr(prec, "decision_time_enabled", True)
        assert svc.query({}) == []
