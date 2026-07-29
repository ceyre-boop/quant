"""
Offline structural tests for the Petrules Gate daily scanner (Phase 0).

No network. Proves the rule-based rubric, tiering, schema, sizing honesty
(BACKTEST ONLY), noise-drop, and the honest error-JSON path — everything that
can be verified without a live SEC EDGAR / Yahoo run.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import gate_scorer  # noqa: E402
import petrules_gate_scanner as scanner  # noqa: E402


@pytest.fixture
def cfg():
    return scanner.load_config()


def test_weights_sum_to_one(cfg):
    assert abs(sum(cfg["weights"].values()) - 1.0) < 1e-9


def test_tier4_screamer(cfg):
    feats = {
        "insider": {"n_buyers": 4, "n_sellers": 0, "total_buy_usd": 4_200_000,
                    "includes_csuite": True, "cluster_within_days": 3},
        "revision": {"n_upgrades": 6, "n_downgrades": 0},
        "options": {"best_vol_oi": 4.8, "best_premium_usd": 2_800_000,
                    "multiple_strikes_same_side": True},
        "activist": {"filing_type": "new_13d"},
    }
    scored = gate_scorer.score_instrument("NVDA", feats, cfg)
    assert scored["tier"] == 4
    assert scored["conviction_score"] >= cfg["tiers"]["tier4_min"]


def test_selling_penalty_pulls_score_down(cfg):
    base = {"n_buyers": 3, "n_sellers": 0, "total_buy_usd": 500_000,
            "includes_csuite": False, "cluster_within_days": 4}
    with_sells = dict(base, n_sellers=2)
    assert gate_scorer.score_insider_cluster(with_sells, cfg) < \
        gate_scorer.score_insider_cluster(base, cfg)


def test_downgrade_caps_revision(cfg):
    feats = {"n_upgrades": 6, "n_downgrades": 1}
    assert gate_scorer.score_revision_velocity(feats, cfg) <= \
        cfg["scoring"]["revision_velocity"]["downgrade_cap"]


def test_missing_data_scores_zero_not_fabricated(cfg):
    scored = gate_scorer.score_instrument("XYZ", {"insider": None, "revision": None,
                                                  "options": None, "activist": None}, cfg)
    assert scored["conviction_score"] == 0.0
    assert scored["tier"] == 1


def test_sizing_block_is_backtest_only_and_respects_fmax(cfg):
    block = gate_scorer.sizing_block(0.99, cfg)
    assert block["calibration_status"].startswith("BACKTEST ONLY")
    assert block["f"] <= block["f_max"]


def test_scan_schema_and_noise_drop(cfg):
    fixtures = {
        "NVDA": {"insider": {"n_buyers": 4, "n_sellers": 0, "total_buy_usd": 4_200_000,
                             "includes_csuite": True, "cluster_within_days": 3},
                 "revision": {"n_upgrades": 6, "n_downgrades": 0},
                 "options": {"best_vol_oi": 4.8, "best_premium_usd": 2_800_000,
                             "multiple_strikes_same_side": True},
                 "activist": {"filing_type": "new_13d"}},
        "KO": {"insider": {"n_buyers": 0, "n_sellers": 2, "total_buy_usd": 0,
                           "includes_csuite": False, "cluster_within_days": 0},
               "revision": {"n_upgrades": 0, "n_downgrades": 0},
               "options": None, "activist": None},
    }
    scan = scanner.run_scan(cfg, injected_features=fixtures)
    scan.pop("_tier2_plus_internal", None)
    for key in ("scanned_at", "instruments_scanned", "tier3_plus",
                "top_signal", "all_signals"):
        assert key in scan
    assert scan["instruments_scanned"] == 2
    assert scan["top_signal"]["symbol"] == "NVDA"
    # KO is Tier 1 noise — must not appear in surfaced all_signals
    assert all(s["tier"] >= 2 for s in scan["all_signals"])


def test_error_json_path(cfg, tmp_path, monkeypatch):
    # Redirect the scan output to a temp file and prove the honest error record.
    out = tmp_path / "scan.json"
    monkeypatch.setitem(cfg["paths"], "scan_output", str(out.relative_to(REPO))
                        if str(out).startswith(str(REPO)) else "data/agent/_test_err.json")
    # Simpler: call write_error_scan against a patched path map.
    p = scanner.write_error_scan(cfg, "RuntimeError: simulated source outage")
    rec = json.loads(Path(p).read_text())
    assert set(rec.keys()) == {"error", "scanned_at"}
    assert "simulated source outage" in rec["error"]
