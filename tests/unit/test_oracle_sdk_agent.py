"""Network-free tests for the SDK Oracle's plain tool functions."""
import json
from sovereign.oracle import sdk_agent as sa


def test_thesis_list_is_json_with_hashes():
    rows = json.loads(sa.t_thesis_list())
    assert rows and all(r["id"].startswith("THS-") for r in rows)
    assert all(r["hash_lock"] for r in rows if r["status"] in ("SEALED", "LIVE"))


def test_recent_decisions_shape():
    d = json.loads(sa.t_recent_decisions(7))
    assert set(d) >= {"days", "n", "records", "source"}


def test_draft_rejects_bad_weights_and_writes_good(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "DRAFTS", tmp_path)
    bad = {"title": "x", "thesis": "who pays", "horizon_months": 12,
           "basket": [{"ticker": "AAA", "weight": 0.7}, {"ticker": "BBB", "weight": 0.7}],
           "invalidation": {"rule": "dd", "metric": "basket_drawdown", "threshold": 0.3}}
    assert sa.t_propose_thesis_draft(bad).startswith("REJECTED")
    good = dict(bad, basket=[{"ticker": "AAA", "weight": 0.5}, {"ticker": "BBB", "weight": 0.5}])
    msg = sa.t_propose_thesis_draft(good)
    assert msg.startswith("DRAFT written") and list(tmp_path.glob("*.json"))
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["source"] == "ai-proposed"
