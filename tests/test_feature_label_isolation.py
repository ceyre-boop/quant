"""
tests/test_feature_label_isolation.py

The automated guard against the tautology that killed Sovereign Core ML and that is
STILL LIVE in layer1/bias_engine.py today (it trains on rsi_14 and derives its label
y = rsi_14 > 50 from the same rsi_14 — predicting a feature from itself).

This test encodes the rule from the HYP-064 pre-registration: every feature's data
window must end STRICTLY BEFORE the label's forward window begins. It validates the
declared spec (docs/layer1/feature_windows.json) and proves it has teeth by flagging
the tautology and a look-ahead leak, while passing the corrected forward-direction label.

Phase 1: this checks the declared windows. Phase 2 will reuse `is_isolated` to enforce
the same invariant on the actual feature builder at compute time.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "docs" / "layer1" / "feature_windows.json"
FEATURES_PARQUET = ROOT / "data" / "layer1" / "features_v1.parquet"
LABELS_PARQUET = ROOT / "data" / "layer1" / "labels_v1.parquet"
LABEL_HORIZON = 5


# ─── The reusable checker (pure; promote to sovereign/layer1 in Phase 2) ───────

def is_isolated(feature_window: dict, label_window: dict) -> bool:
    """True iff the feature's data ends strictly before the label's forward window.

    feature_window / label_window are dicts with 'earliest_offset_days' and
    'latest_offset_days' (trading days relative to the decision day t0; 0 = t0 close).
    Isolation holds iff feature.latest_offset < label.earliest_offset.
    """
    return feature_window["latest_offset_days"] < label_window["earliest_offset_days"]


def load_spec() -> dict:
    return json.loads(SPEC_PATH.read_text())


# ─── Spec presence / structure ────────────────────────────────────────────────

def test_spec_file_exists_and_parses():
    assert SPEC_PATH.exists(), f"feature window spec missing at {SPEC_PATH}"
    spec = load_spec()
    assert spec["features"], "spec must declare features"
    assert spec["label"]["window"], "spec must declare a label window"


def test_every_feature_has_source_and_window():
    spec = load_spec()
    for f in spec["features"]:
        assert f.get("name"), f"feature missing name: {f}"
        assert f.get("source"), f"feature {f['name']} missing source"
        cw = f.get("computation_window", {})
        assert "earliest_offset_days" in cw and "latest_offset_days" in cw, \
            f"feature {f['name']} missing a computation window"
        assert cw["earliest_offset_days"] <= cw["latest_offset_days"], \
            f"feature {f['name']} has an inverted window"


# ─── THE INVARIANT: every declared feature is isolated from the label ─────────

def test_real_spec_every_feature_is_isolated():
    spec = load_spec()
    label_w = spec["label"]["window"]
    violations = [
        f["name"] for f in spec["features"]
        if not is_isolated(f["computation_window"], label_w)
    ]
    assert not violations, (
        "Feature/label isolation VIOLATED — these features' data windows reach into the "
        f"label window {label_w}: {violations}. This is the tautology guard; fix the spec."
    )


def test_no_feature_uses_future_data():
    """A feature may never use data after t0 (latest_offset must be <= 0)."""
    spec = load_spec()
    leaks = [
        f["name"] for f in spec["features"]
        if f["computation_window"]["latest_offset_days"] > 0
    ]
    assert not leaks, f"features using future (post-t0) data: {leaks}"


# ─── TEETH: the checker must FLAG the known failure modes ──────────────────────

def test_tautology_red_case_is_flagged():
    """The exact live bug in layer1/bias_engine.py: label y = (rsi_14[t0] > 50), a
    function of a t0 feature. The label window includes offset 0, overlapping the
    rsi_14 feature window [-14, 0]. The checker MUST report this as NOT isolated.
    """
    rsi_feature_window = {"earliest_offset_days": -14, "latest_offset_days": 0}
    same_bar_label_window = {"earliest_offset_days": 0, "latest_offset_days": 0}
    assert not is_isolated(rsi_feature_window, same_bar_label_window), (
        "Checker FAILED to flag the rsi->rsi same-bar tautology — it has no teeth."
    )


def test_lookahead_feature_red_case_is_flagged():
    """A feature that peeks 3 days into the future overlaps the forward label window."""
    lookahead_feature = {"earliest_offset_days": -14, "latest_offset_days": 3}
    fwd_label = {"earliest_offset_days": 1, "latest_offset_days": 5}
    assert not is_isolated(lookahead_feature, fwd_label), (
        "Checker FAILED to flag a look-ahead feature."
    )


def test_corrected_forward_label_green_case_passes():
    """The corrected formulation: features end at t0, label is the next-5-day direction."""
    feature = {"earliest_offset_days": -14, "latest_offset_days": 0}
    fwd_label = {"earliest_offset_days": 1, "latest_offset_days": 5}
    assert is_isolated(feature, fwd_label), (
        "Corrected forward-direction label should be isolated and pass."
    )


# ─── Hurst must stay excluded (the prior tautology basis) ──────────────────────

def test_hurst_is_not_a_feature():
    spec = load_spec()
    names = {f["name"] for f in spec["features"]}
    hurst_like = {n for n in names if "hurst" in n.lower()}
    assert not hurst_like, (
        f"Hurst must not be a model input (prior tautology basis): found {hurst_like}"
    )


def test_hurst_is_documented_as_rejected():
    spec = load_spec()
    rejected = {r["name"] for r in spec.get("rejected_features", [])}
    assert "hurst_regime" in rejected, "hurst_regime must be documented in rejected_features"


# ─── Phase 2: the test must pass on the REAL parquet, not just the spec ─────────

class TestRealParquet:
    """Isolation guarantees verified against the actual built data (HYP-064 Phase 2)."""

    @pytest.fixture(scope="class")
    def data(self):
        if not (FEATURES_PARQUET.exists() and LABELS_PARQUET.exists()):
            pytest.skip("features_v1/labels_v1 parquet not built yet (run scripts/build_layer1_dataset.py)")
        pd = pytest.importorskip("pandas")
        feats = pd.read_parquet(FEATURES_PARQUET)
        labels = pd.read_parquet(LABELS_PARQUET)
        return feats, labels

    def test_feature_columns_are_all_declared(self, data):
        feats, _ = data
        declared = {f["name"] for f in load_spec()["features"]}
        undeclared = [c for c in feats.columns if c not in declared]
        assert not undeclared, f"parquet has undeclared (possibly leaked) feature columns: {undeclared}"

    def test_label_columns_present(self, data):
        _, labels = data
        assert "fwd_direction_5d" in labels.columns
        assert "fwd_direction_10d" in labels.columns

    def test_features_and_labels_share_index(self, data):
        feats, labels = data
        assert feats.index.equals(labels.index), "features and labels must share an identical (date,pair) index"

    def test_label_tail_is_nan_proving_forward_looking(self, data):
        """Per pair, the last LABEL_HORIZON rows of fwd_direction_5d must be NaN — there is no
        forward window at the series end. This is structural proof the label looks FORWARD (a
        same-bar transform like the old rsi>50 tautology would have no missing tail)."""
        _, labels = data
        lab = labels.sort_index()
        for pair, grp in lab.groupby(level=1):
            tail = grp["fwd_direction_5d"].iloc[-LABEL_HORIZON:]
            assert tail.isna().all(), f"{pair}: last {LABEL_HORIZON} labels must be NaN (forward window ran out)"
            body = grp["fwd_direction_5d"].iloc[:-LABEL_HORIZON]
            assert body.notna().any(), f"{pair}: body should contain real labels"

    def test_no_feature_is_a_proxy_for_the_label(self, data):
        """No feature may equal or near-perfectly correlate with the label — the data-level guard
        against the rsi->rsi leakage class (a feature predicting the label by construction)."""
        pd = pytest.importorskip("pandas")
        feats, labels = data
        y = labels["fwd_direction_5d"]
        mask = y.notna()
        y = y[mask]
        offenders = []
        for col in feats.columns:
            x = feats[col][mask]
            if x.notna().sum() < 100 or x.nunique() < 3:
                continue
            c = x.corr(y)
            if c is not None and abs(c) > 0.99:
                offenders.append((col, round(float(c), 4)))
        assert not offenders, f"feature(s) near-perfectly correlated with the label (leakage): {offenders}"

    def test_holdout_untouched(self, data):
        pd = pytest.importorskip("pandas")
        feats, _ = data
        max_date = feats.index.get_level_values(0).max()
        assert max_date <= pd.Timestamp("2023-12-31"), f"HOLDOUT LEAK: max date {max_date} > 2023-12-31"


META_PARQUET = ROOT / "data" / "layer1" / "meta_dataset_v1.parquet"


class TestMetaDataset:
    """Isolation guarantees for the META-LABELING dataset (HYP-064 revised, event-sampled)."""

    @pytest.fixture(scope="class")
    def md(self):
        if not META_PARQUET.exists():
            pytest.skip("meta_dataset_v1 not built (run sovereign/layer1/meta_label_builder.py)")
        pd = pytest.importorskip("pandas")
        return pd.read_parquet(META_PARQUET)

    AUX = {"meta_win", "realized_r", "exit_reason", "direction", "exit_date", "hold_days"}

    def test_feature_columns_are_all_declared(self, md):
        declared = {f["name"] for f in load_spec()["features"]}
        feat_cols = [c for c in md.columns if c not in self.AUX]
        undeclared = [c for c in feat_cols if c not in declared]
        assert not undeclared, f"meta-dataset has undeclared feature columns: {undeclared}"

    def test_label_is_binary(self, md):
        assert "meta_win" in md.columns
        assert set(md["meta_win"].dropna().unique()) <= {0, 1}

    def test_label_matches_realized_sign(self, md):
        """meta_win must equal (realized_r > 0) — the label is the net trade outcome, nothing else."""
        derived = (md["realized_r"] > 0).astype(int)
        assert (derived == md["meta_win"]).all(), "meta_win disagrees with sign(realized_r)"

    def test_label_is_forward_by_construction(self, md):
        """Every exit is at or after entry — the label depends on POST-entry prices (forward), so it
        cannot be a same-bar feature transform (the rsi->rsi tautology class)."""
        pd = pytest.importorskip("pandas")
        entry = md.index.get_level_values(0)
        assert (pd.to_datetime(md["exit_date"]).values >= entry.values).all(), \
            "a trade exits before it enters — label is not forward-looking"

    def test_no_feature_is_a_proxy_for_the_label(self, md):
        feat_cols = [c for c in md.columns if c not in self.AUX]
        y = md["meta_win"]
        offenders = []
        for col in feat_cols:
            x = md[col]
            if x.notna().sum() < 50 or x.nunique() < 3:
                continue
            c = x.corr(y)
            if c is not None and abs(c) > 0.99:
                offenders.append((col, round(float(c), 4)))
        assert not offenders, f"feature(s) near-perfectly correlated with meta_win (leakage): {offenders}"

    def test_holdout_untouched(self, md):
        pd = pytest.importorskip("pandas")
        max_entry = md.index.get_level_values(0).max()
        max_exit = pd.to_datetime(md["exit_date"]).max()
        assert max(max_entry, max_exit) <= pd.Timestamp("2023-12-31"), "HOLDOUT LEAK in meta-dataset"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
