"""Safeguard tests: holdout guard + walk-forward window construction.

These assert the ENFORCEMENT, not the strategies. If someone loosens the guard
or reintroduces a single-split validation path, these fail.
"""
import numpy as np
import pandas as pd
import pytest

from backtester import holdout_guard as hg
from backtester import walk_forward as wf


@pytest.fixture(autouse=True)
def _clean_sanction(monkeypatch):
    hg.revoke()
    monkeypatch.delenv("ALLOW_HOLDOUT_ACCESS", raising=False)
    yield
    hg.revoke()


# ── holdout guard ────────────────────────────────────────────────────────────

def test_mining_range_before_holdout_passes():
    hg.validate_date_range("2020-01-01", "2025-07-16", dataset="equities_daily")


def test_range_touching_holdout_raises():
    with pytest.raises(hg.HoldoutViolation, match="touches the"):
        hg.validate_date_range("2020-01-01", "2025-07-17", dataset="equities_daily")


def test_unbounded_end_is_treated_as_touching():
    # "9999" / None is exactly how holdout data leaks into a mining scan.
    with pytest.raises(hg.HoldoutViolation):
        hg.validate_date_range("2020-01-01", "9999", dataset="equities_daily")
    with pytest.raises(hg.HoldoutViolation):
        hg.validate_date_range("2020-01-01", None, dataset="equities_daily")


def test_env_override_sanctions_and_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(hg, "ACCESS_LOG", tmp_path / "access.jsonl")
    monkeypatch.setenv("ALLOW_HOLDOUT_ACCESS", "1")
    hg.validate_date_range("2020-01-01", "2026-07-17", context="verdict")
    assert (tmp_path / "access.jsonl").read_text().count("verdict") == 1


def test_in_process_sanction_allows_and_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(hg, "ACCESS_LOG", tmp_path / "access.jsonl")
    hg.sanction("HYP-999 prereg verified")
    hg.validate_date_range("2020-01-01", "2026-07-17")
    assert "HYP-999" in (tmp_path / "access.jsonl").read_text()


def test_unregistered_dataset_is_rejected():
    with pytest.raises(KeyError, match="unregistered dataset"):
        hg.validate_date_range("2020-01-01", "2020-06-01", dataset="made_up")


def test_every_registered_holdout_is_a_valid_iso_date():
    for name, d in hg.HOLDOUT_REGISTRY.items():
        pd.Timestamp(d)  # raises if malformed
        assert len(d) == 10, f"{name}: {d}"


def test_is_mining_safe_predicate():
    assert hg.is_mining_safe("2020-01-01", "2025-07-16")
    assert not hg.is_mining_safe("2020-01-01", "2026-01-01")


# ── walk-forward windows ─────────────────────────────────────────────────────

def test_windows_tile_test_slices_without_overlap_or_gap():
    w = wf.generate_windows(1000, train_window=252, test_window=63, min_train=126)
    assert w, "expected windows"
    for (_, _, lo, hi), (_, _, nlo, _) in zip(w, w[1:]):
        assert hi == nlo, "test slices must tile contiguously"
        assert hi - lo == 63


def test_test_slice_always_starts_at_train_end_no_leakage():
    for tr_lo, tr_hi, te_lo, te_hi in wf.generate_windows(800, 252, 63, 126):
        assert te_lo == tr_hi, "test must begin exactly where train ends"
        assert tr_lo < tr_hi <= te_lo < te_hi


def test_rolling_window_has_fixed_train_size():
    w = wf.generate_windows(1000, 252, 63, 126, anchored=False)
    assert {hi - lo for lo, hi, _, _ in w} == {252}


def test_anchored_window_actually_expands():
    """Regression for the backtest/walk_forward.py EXPANDING bug."""
    w = wf.generate_windows(1000, 252, 63, 126, anchored=True)
    sizes = [hi - lo for lo, hi, _, _ in w]
    assert all(lo == 0 for lo, _, _, _ in w)
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0], "train set must grow"


def test_short_data_yields_no_windows():
    assert wf.generate_windows(100, 252, 63, 126) == []


def test_invalid_window_params_rejected():
    with pytest.raises(ValueError):
        wf.generate_windows(1000, 0, 63, 126)
    with pytest.raises(ValueError):
        wf.generate_windows(1000, 100, 63, min_train=200)


# ── walk-forward run ─────────────────────────────────────────────────────────

def _frame(n=800, start="2020-01-02"):
    d = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(7)
    return pd.DataFrame({"date": d.strftime("%Y-%m-%d"),
                         "close": 100 + np.cumsum(rng.normal(0, 1, n))})


def test_walk_forward_reports_only_out_of_sample():
    """strategy_fn is handed train and test; only test returns reach the summary."""
    seen = []

    def strat(train, test):
        seen.append((len(train), len(test)))
        # deliberately return a constant so pooled n == sum of test lengths
        return {"rets": [0.01] * len(test), "params": {"thr": 1}}

    res = wf.walk_forward_backtest(strat, _frame(), train_window=252,
                                   test_window=63, min_train=126,
                                   check_holdout=False)
    assert res["n_windows"] == len(seen)
    assert res["out_of_sample"]["n"] == sum(t for _, t in seen)
    assert all(tr == 252 for tr, _ in seen)


def test_walk_forward_flags_refit_churn():
    counter = {"i": 0}

    def churny(train, test):
        counter["i"] += 1
        return {"rets": [0.001] * len(test), "params": {"thr": counter["i"]}}

    res = wf.walk_forward_backtest(churny, _frame(), check_holdout=False)
    assert res["param_stability"]["refit_churn"] == 1.0
    assert res["param_stability"]["distinct_param_sets"] == res["n_windows"]


def test_walk_forward_enforces_holdout_on_the_frame():
    df = _frame(n=800, start="2024-01-02")  # runs into the sealed window
    with pytest.raises(hg.HoldoutViolation):
        wf.walk_forward_backtest(lambda a, b: {"rets": []}, df,
                                 dataset="equities_daily")


def test_walk_forward_short_data_returns_error_not_crash():
    res = wf.walk_forward_backtest(lambda a, b: {"rets": []}, _frame(50),
                                   check_holdout=False)
    assert res["n_windows"] == 0 and "error" in res
