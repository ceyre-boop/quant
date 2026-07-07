"""tests/test_geometry_runner.py — scripts/research/run_geometry_family.py (TICK-019 Phase 2).

Offline & deterministic: synthetic frames + tmp-path ledgers/DBs only (never the real
data/sentiment.db, the real data/agent/hypothesis_ledger.json, or the hash-locked prereg
files). Mirrors tests/test_positioning_event_study.py's TestLedgerAnnotate idiom for the
ledger/seal mechanics. Covers (per plan Phase 2 "Tests shipped with the runner"): gate-zero
mismatch aborts (tamper fixture); Gα residualization is trailing-only (truncation test on
beta_t); pip-floor arithmetic; Gβ event de-overlap + suppression on synthetic gaps; adjudication
verdict mapping (all outcome classes). Formation-events/fvg_count_20d parity lives in
tests/test_sentiment_geometry.py::TestFVGFormationEvents (it tests geometry_feed.py directly).
"""
from __future__ import annotations

import json
import shutil
from datetime import date

import numpy as np
import pandas as pd
import pytest

import scripts.research.run_geometry_family as run
from sovereign.research.positioning import event_study as es
from sovereign.sentiment import store

GEOM_CFG = {
    "corridor_window": 120, "fvg_max_age": 20, "fvg_min_atr_frac": 0.3,
    "tri_window": 20, "tri_pctile": 0.25, "start": "2000-01-01",
}


# ── gate zero ──────────────────────────────────────────────────────────────────────────────

class TestGateZero:
    def test_real_prereg_hashes_verify(self):
        # Read-only hash check against the real (committed, hash-locked) prereg files — never
        # writes anything. Confirms the 4-file gate the real runner enforces actually passes.
        checks = run.gate_zero()
        assert all(v["ok"] for v in checks.values())
        assert set(checks) == set(run.ALL_PREREG)

    def test_tamper_aborts(self, tmp_path, monkeypatch):
        for name in run.ALL_PREREG:
            shutil.copy2(run.PREREG / name, tmp_path / name)
        target = tmp_path / "HYP-082_fractal_beyond_carry.json"
        doc = json.loads(target.read_text())
        doc["thesis"] = doc["thesis"] + " TAMPERED"
        target.write_text(json.dumps(doc))
        monkeypatch.setattr(run, "PREREG", tmp_path)
        with pytest.raises(SystemExit, match="GATE ZERO"):
            run.gate_zero()

    def test_missing_hash_lock_aborts(self, tmp_path, monkeypatch):
        for name in run.ALL_PREREG:
            shutil.copy2(run.PREREG / name, tmp_path / name)
        target = tmp_path / "HYP-083_fvg_diversifier.json"
        doc = json.loads(target.read_text())
        doc["hash_lock"] = "0" * 64
        target.write_text(json.dumps(doc))
        monkeypatch.setattr(run, "PREREG", tmp_path)
        with pytest.raises(SystemExit, match="GATE ZERO"):
            run.gate_zero()


# ── Gα: carry_residual_frame — trailing-only beta ───────────────────────────────────────────

class TestCarryResidualFrame:
    def _series_pair(self, n=400, seed=5, start="2015-01-01"):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range(start, periods=n)
        pair_close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, n))), index=idx)
        v015_eq = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.003, n))), index=idx)
        return pair_close, v015_eq

    def test_beta_undefined_before_252_window(self):
        # pair_ret/v015_ret each have a leading NaN (from .shift(1)), so the trailing 252-window
        # first contains 252 valid (non-NaN) PAIRS at position 252 (0-indexed) — the window ending
        # at position 251 still has only 251 valid pairs (position 0's return is NaN).
        pair_close, v015_eq = self._series_pair()
        out = run.carry_residual_frame(pair_close, v015_eq, h=20, beta_window=252)
        assert out["beta"].iloc[:252].isna().all()
        assert not pd.isna(out["beta"].iloc[252])

    def test_beta_is_trailing_only_truncation_invariant(self):
        """The KEY no-look-ahead guarantee: beta_t at a given date must be identical whether
        computed on the full series or on a series truncated right after that date."""
        pair_close, v015_eq = self._series_pair(n=400)
        full = run.carry_residual_frame(pair_close, v015_eq, h=20, beta_window=252)
        t = 300
        trunc = run.carry_residual_frame(pair_close.iloc[: t + 1], v015_eq.iloc[: t + 1],
                                         h=20, beta_window=252)
        assert full["beta"].iloc[t] == pytest.approx(trunc["beta"].iloc[-1], rel=1e-9)

    def test_forward_fields_undefined_in_final_h_bars(self):
        pair_close, v015_eq = self._series_pair(n=300)
        out = run.carry_residual_frame(pair_close, v015_eq, h=20, beta_window=252)
        assert out["carry_residual"].iloc[-20:].isna().all()
        assert pd.isna(out["exit_date"].iloc[-1])

    def test_residual_formula(self):
        # Hand-computable: pair_fwd - beta*v015_fwd, using values read straight off the frame.
        pair_close, v015_eq = self._series_pair(n=400)
        out = run.carry_residual_frame(pair_close, v015_eq, h=20, beta_window=252)
        row = out.iloc[260]
        assert row["carry_residual"] == pytest.approx(row["pair_fwd"] - row["beta"] * row["v015_fwd"])


# ── Gα: every-5th-day observation grid ──────────────────────────────────────────────────────

class TestGammaObservations:
    def test_every_5th_position_and_filters(self):
        idx = pd.bdate_range("2015-01-01", periods=60)
        corridor_dev = np.arange(60, dtype=float)
        corridor_dev[10] = np.nan   # position 10 is one of the every-5th slots (0,5,10,...)
        geo = pd.DataFrame({"corridor_dev": corridor_dev}, index=idx)
        obs = run.gamma_observations(geo, start="2015-01-01")
        expected_positions = [p for p in range(0, 60, 5) if p != 10]
        assert list(obs.index) == [idx[p] for p in expected_positions]

    def test_start_filter_applied_after_grid(self):
        idx = pd.bdate_range("2015-01-01", periods=60)
        geo = pd.DataFrame({"corridor_dev": np.arange(60, dtype=float)}, index=idx)
        cutoff = idx[20]
        obs = run.gamma_observations(geo, start=str(cutoff.date()))
        assert (obs.index >= cutoff).all()
        # anchor unaffected by the start filter: still positions 0,5,10,... intersected with >= cutoff
        assert list(obs.index) == [idx[p] for p in range(0, 60, 5) if idx[p] >= cutoff]


# ── Gα: pooled IC permutation (two-sided on |IC|) ───────────────────────────────────────────

class TestPooledIcPermutation:
    def test_no_relationship_p_not_extreme(self):
        rng = np.random.default_rng(1)
        scored = {}
        for pair in ("EURUSD", "GBPUSD"):
            n = 80
            scored[pair] = pd.DataFrame({
                "corridor_dev": rng.normal(0, 1, n),
                "carry_residual": rng.normal(0, 1, n),
            })
        out = run.pooled_ic_permutation(scored, np.random.default_rng(42), n_perm=500)
        assert out["p"] is not None and 0.0 <= out["p"] <= 1.0
        assert abs(out["ic"]) < 0.5

    def test_injected_positive_relationship_detected(self):
        rng = np.random.default_rng(2)
        scored = {}
        for pair in ("EURUSD", "GBPUSD", "USDJPY"):
            n = 80
            feat = rng.normal(0, 1, n)
            resid = feat * 0.8 + rng.normal(0, 0.2, n)
            scored[pair] = pd.DataFrame({"corridor_dev": feat, "carry_residual": resid})
        out = run.pooled_ic_permutation(scored, np.random.default_rng(42), n_perm=999)
        assert out["ic"] > 0.5
        assert out["p"] < 0.05

    def test_two_sided_catches_negative_relationship_too(self):
        rng = np.random.default_rng(2)
        scored = {}
        for pair in ("EURUSD", "GBPUSD", "USDJPY"):
            n = 80
            feat = rng.normal(0, 1, n)
            resid = -feat * 0.8 + rng.normal(0, 0.2, n)
            scored[pair] = pd.DataFrame({"corridor_dev": feat, "carry_residual": resid})
        out = run.pooled_ic_permutation(scored, np.random.default_rng(42), n_perm=999)
        assert out["ic"] < -0.5
        assert out["p"] < 0.05      # two-sided on |IC| — a strong NEGATIVE IC must also be significant

    def test_too_few_obs_returns_none(self):
        scored = {"EURUSD": pd.DataFrame({"corridor_dev": [0.1], "carry_residual": [0.2]})}
        out = run.pooled_ic_permutation(scored, np.random.default_rng(42), n_perm=100)
        assert out["ic"] is None and out["p"] is None


# ── Gα: CPCV fold-sign consistency ──────────────────────────────────────────────────────────

class TestCpcvFoldSignConsistency:
    def _scored(self, n, relation, seed=3):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2015-01-01", periods=n)
        feat = rng.normal(0, 1, n)
        resid = relation(feat, np.arange(n)) + rng.normal(0, 0.05, n)
        exit_idx = np.minimum(np.arange(n) + 5, n - 1)
        return {"EURUSD": pd.DataFrame({"corridor_dev": feat, "carry_residual": resid,
                                       "exit_date": idx[exit_idx]}, index=idx)}

    def test_stable_relationship_all_same_sign(self):
        scored = self._scored(240, lambda f, t: 0.9 * f)
        out = run.cpcv_fold_sign_consistency(scored, n_groups=6, test_groups=1, embargo_frac=0.02)
        assert out["n_folds"] == 6
        assert out["all_same_sign"] is True

    def test_regime_flip_gives_inconsistent_signs(self):
        # positive relationship in the first half of time, negative in the second half —
        # 6 time-contiguous groups should split into sign-disagreeing folds.
        def relation(f, t):
            return np.where(t < len(t) / 2, 0.9 * f, -0.9 * f)
        scored = self._scored(240, relation)
        out = run.cpcv_fold_sign_consistency(scored, n_groups=6, test_groups=1, embargo_frac=0.02)
        assert out["all_same_sign"] is False

    def test_too_few_obs_returns_none_with_note(self):
        scored = {"EURUSD": pd.DataFrame({"corridor_dev": [0.1, 0.2], "carry_residual": [0.1, 0.2],
                                          "exit_date": pd.bdate_range("2015-01-01", periods=2)})}
        out = run.cpcv_fold_sign_consistency(scored, n_groups=6, test_groups=1)
        assert out["all_same_sign"] is None and "note" in out


# ── Gα: cost floor pip arithmetic ────────────────────────────────────────────────────────────

class TestCostFloor:
    def _closes_with_move(self, base, move, n=25, start="2020-01-01"):
        idx = pd.bdate_range(start, periods=n)
        prices = np.full(n, base, dtype=float)
        prices[20] = base + move
        return pd.Series(prices, index=idx), idx

    def test_pip_conversion_exact_floor_passes(self):
        eur_close, idx = self._closes_with_move(1.1000, 0.00030)     # exactly 3.0 pips @ 0.0001/pip
        jpy_close, _ = self._closes_with_move(110.00, 0.030)         # exactly 3.0 pips @ 0.01/pip
        closes = {"EURUSD": eur_close, "USDJPY": jpy_close}
        obs = {"EURUSD": pd.DataFrame({"corridor_dev": [2.0]}, index=[idx[0]]),
              "USDJPY": pd.DataFrame({"corridor_dev": [2.0]}, index=[idx[0]])}
        res = run.cost_floor(obs, closes, h=20)
        assert res["per_pair"]["EURUSD"]["median_pips"] == pytest.approx(3.0)
        assert res["per_pair"]["USDJPY"]["median_pips"] == pytest.approx(3.0)
        assert res["pooled_median_pips"] == pytest.approx(3.0)
        assert res["pass"] is True

    def test_below_floor_fails(self):
        eur_close, idx = self._closes_with_move(1.1000, 0.00010)     # 1.0 pip
        jpy_close, _ = self._closes_with_move(110.00, 0.010)         # 1.0 pip
        closes = {"EURUSD": eur_close, "USDJPY": jpy_close}
        obs = {"EURUSD": pd.DataFrame({"corridor_dev": [2.0]}, index=[idx[0]]),
              "USDJPY": pd.DataFrame({"corridor_dev": [2.0]}, index=[idx[0]])}
        res = run.cost_floor(obs, closes, h=20)
        assert res["pooled_median_pips"] == pytest.approx(1.0)
        assert res["pass"] is False

    def test_pooling_normalizes_each_pair_by_its_own_pip_size(self):
        # EURUSD raw move 0.0001 (1 pip) vs USDJPY raw move 0.04 (4 pips): a naive raw-price
        # pool would be dominated by USDJPY's larger raw number for the wrong reason (scale, not
        # pips) — the correct pip-normalized pooled median is (1+4)/2 = 2.5, which still fails
        # the 3.0 floor even though the raw USDJPY number (0.04) looks "big".
        eur_close, idx = self._closes_with_move(1.1000, 0.0001)
        jpy_close, _ = self._closes_with_move(110.00, 0.04)
        closes = {"EURUSD": eur_close, "USDJPY": jpy_close}
        obs = {"EURUSD": pd.DataFrame({"corridor_dev": [2.0]}, index=[idx[0]]),
              "USDJPY": pd.DataFrame({"corridor_dev": [2.0]}, index=[idx[0]])}
        res = run.cost_floor(obs, closes, h=20)
        assert res["pooled_median_pips"] == pytest.approx(2.5)
        assert res["pass"] is False

    def test_empty_obs_gives_no_pass(self):
        idx = pd.bdate_range("2020-01-01", periods=25)
        closes = {"EURUSD": pd.Series(1.1, index=idx)}
        obs = {"EURUSD": pd.DataFrame({"corridor_dev": pd.Series(dtype=float)})}
        res = run.cost_floor(obs, closes, h=20)
        assert res["pooled_median_pips"] is None and res["pass"] is False


# ── Gβ: formation log + de-overlap + suppression on synthetic gaps ─────────────────────────

def _flat_ohlc(n, start="2021-01-01", price=100.0):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Open": price, "High": price + 0.05, "Low": price - 0.05,
                        "Close": price}, index=idx)


def _inject_bull_gap(df, at, base, gap=1.0):
    """Overwrite bars [at, at+2] with a clean bullish 3-bar gap, then flatten everything after
    at+2 at the new price level so a SUBSEQUENT injection later in the frame starts continuous
    (no incidental seam gap between two independently-injected triplets)."""
    df = df.copy()
    loc = {c: df.columns.get_loc(c) for c in ("Open", "High", "Low", "Close")}
    df.iloc[at, [loc["Open"], loc["Close"], loc["High"], loc["Low"]]] = [base, base, base + 0.05, base - 0.05]
    df.iloc[at + 1, [loc["Open"], loc["Close"], loc["High"], loc["Low"]]] = \
        [base + 0.1, base + 0.15, base + 0.2, base]
    top = base + gap
    df.iloc[at + 2, [loc["Open"], loc["Close"], loc["High"], loc["Low"]]] = [top, top + 0.05, top + 0.1, top]
    level = top + 0.05
    if at + 3 < len(df):
        df.iloc[at + 3:, loc["Open"]] = level
        df.iloc[at + 3:, loc["Close"]] = level
        df.iloc[at + 3:, loc["High"]] = level + 0.05
        df.iloc[at + 3:, loc["Low"]] = level - 0.05
    return df, level


class TestFormationLogAndSuppression:
    def test_formation_log_finds_injected_gap_with_correct_direction_and_date(self):
        df = _flat_ohlc(60)
        df, _ = _inject_bull_gap(df, at=30, base=100.0)
        log = run.formation_log(df, GEOM_CFG)
        expected_date = pd.Timestamp(df.index[32]).date()
        assert (expected_date, 1) in log

    def test_suppression_drops_same_direction_within_5_trading_days(self):
        df = _flat_ohlc(60)
        df, level = _inject_bull_gap(df, at=30, base=100.0)   # forms at position 32
        df, _ = _inject_bull_gap(df, at=33, base=level)       # forms at position 35 (3d after 32)
        log = run.formation_log(df, GEOM_CFG)
        d1 = pd.Timestamp(df.index[32]).date()
        d2 = pd.Timestamp(df.index[35]).date()
        assert (d1, 1) in log and (d2, 1) in log   # both raw formations detected pre-suppression
        out = run.deoverlap_and_suppress(log, df.index, suppress=5)
        assert (d1, 1) in out
        assert (d2, 1) not in out   # same direction, only 3 trading days later — suppressed

    def test_suppression_keeps_events_5_or_more_trading_days_apart(self):
        df = _flat_ohlc(70)
        df, level = _inject_bull_gap(df, at=30, base=100.0)   # forms at position 32
        df, _ = _inject_bull_gap(df, at=40, base=level)       # forms at position 42 (10d after 32)
        log = run.formation_log(df, GEOM_CFG)
        out = run.deoverlap_and_suppress(log, df.index, suppress=5)
        d1 = pd.Timestamp(df.index[32]).date()
        d2 = pd.Timestamp(df.index[42]).date()
        assert (d1, 1) in out and (d2, 1) in out

    def test_opposite_direction_not_suppressed(self):
        df = _flat_ohlc(60)
        df, level = _inject_bull_gap(df, at=30, base=100.0)     # bullish, forms at 32
        loc = {c: df.columns.get_loc(c) for c in ("Open", "High", "Low", "Close")}
        # bearish gap 3 trading days later (opposite direction — suppression is same-side only)
        at = 33
        df.iloc[at, [loc["Open"], loc["Close"], loc["High"], loc["Low"]]] = [level, level, level + 0.05, level - 0.05]
        df.iloc[at + 1, [loc["Open"], loc["Close"], loc["High"], loc["Low"]]] = \
            [level - 0.1, level - 0.15, level, level - 0.2]
        bottom = level - 1.0
        df.iloc[at + 2, [loc["Open"], loc["Close"], loc["High"], loc["Low"]]] = \
            [bottom, bottom - 0.05, bottom, bottom - 0.1]
        log = run.formation_log(df, GEOM_CFG)
        out = run.deoverlap_and_suppress(log, df.index, suppress=5)
        d1 = pd.Timestamp(df.index[32]).date()
        d2 = pd.Timestamp(df.index[at + 2]).date()
        assert (d1, 1) in out
        assert (d2, -1) in out

    def test_deoverlap_dedupes_exact_duplicate_dates(self):
        idx = pd.bdate_range("2021-01-01", periods=10)
        events = [(idx[3].date(), 1), (idx[3].date(), 1), (idx[8].date(), 1)]
        out = run.deoverlap_and_suppress(events, idx, suppress=5)
        assert out.count((idx[3].date(), 1)) == 1


# ── Gβ: diversifier gate ─────────────────────────────────────────────────────────────────────

class TestDiversifierGate:
    def _setup(self, n=500, seed=9):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2018-01-01", periods=n)
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.006, n))), index=idx)
        return idx, close, rng

    def test_independent_benchmarks_pass(self):
        idx, close, rng = self._setup()
        closes = {"EURUSD": close}
        events = [es.Event("EURUSD", d.date(), 1, 1.0) for d in idx[50:450:15]]
        bench = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)   # independent of `close`
        gate = run.diversifier_gate(events, closes, idx, 10, bench, bench)
        assert gate["benchmarks"]["v015"]["pass"] is True
        assert gate["benchmarks"]["dbv"]["pass"] is True
        assert gate["pass"] is True

    def test_book_tracking_the_benchmark_fails(self):
        idx, close, rng = self._setup()
        closes = {"EURUSD": close}
        # dense, uniformly-long coverage of almost the whole tape: the "book" ends up tracking
        # EURUSD's own daily moves, so correlating against EURUSD's own daily return is a
        # deliberately-not-a-diversifier construction.
        events = [es.Event("EURUSD", d.date(), 1, 1.0) for d in idx[20:480:8]]
        bench = close.pct_change()
        gate = run.diversifier_gate(events, closes, idx, 10, bench, bench)
        assert gate["benchmarks"]["v015"]["full"]["rho"] is not None
        assert abs(gate["benchmarks"]["v015"]["full"]["rho"]) >= 0.25
        assert gate["benchmarks"]["v015"]["pass"] is False
        assert gate["pass"] is False

    def test_requires_both_benchmarks_to_pass(self):
        idx, close, rng = self._setup()
        closes = {"EURUSD": close}
        events = [es.Event("EURUSD", d.date(), 1, 1.0) for d in idx[20:480:8]]
        tracking_bench = close.pct_change()                          # will fail
        independent_bench = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)  # will pass
        gate = run.diversifier_gate(events, closes, idx, 10, independent_bench, tracking_bench)
        assert gate["benchmarks"]["v015"]["pass"] is True
        assert gate["benchmarks"]["dbv"]["pass"] is False
        assert gate["pass"] is False    # overall requires BOTH

    def test_crisis_window_below_min_n_excluded_not_penalized(self):
        idx, close, rng = self._setup()
        closes = {"EURUSD": close}
        events = [es.Event("EURUSD", d.date(), 1, 1.0) for d in idx[50:450:15]]
        # benchmark with NO data at all inside either crisis window -> n<20 there -> excluded
        bench = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
        mask = ((idx >= pd.Timestamp("2020-02-20")) & (idx <= pd.Timestamp("2020-04-30"))) | \
               ((idx >= pd.Timestamp("2022-01-01")) & (idx <= pd.Timestamp("2022-12-31")))
        bench = bench.where(~mask)   # NaN inside both crisis windows
        gate = run.diversifier_gate(events, closes, idx, 10, bench, bench)
        for w in gate["benchmarks"]["v015"]["crisis_windows"].values():
            assert w["rho"] is None
        assert gate["benchmarks"]["v015"]["max_crisis_abs_rho"] is None
        # full-period rho still independent -> pass on the crisis leg by vacuous truth
        assert gate["benchmarks"]["v015"]["pass"] is True


# ── ledger annotate + seal (mirrors test_positioning_event_study.py::TestLedgerAnnotate) ────

class TestLedgerAnnotateAndSeal:
    def _ledger(self, tmp_path):
        ledger = [{"id": "HYP-082", "status": "PREREGISTERED", "hash_lock": "abc", "verdict": None},
                  {"id": "HYP-999", "status": "REJECTED"}]
        lp = tmp_path / "ledger.json"
        lp.write_text(json.dumps(ledger))
        return lp

    def test_annotate_appends_only(self, tmp_path, monkeypatch):
        lp = self._ledger(tmp_path)
        monkeypatch.setattr(run, "LEDGER", lp)
        run._annotate_ledger("HYP-082", {"date": "2026-07-07", "by": "test", "note": "n"})
        out = json.loads(lp.read_text())
        e = next(x for x in out if x["id"] == "HYP-082")
        assert len(e["annotations"]) == 1 and e["status"] == "PREREGISTERED" and e["hash_lock"] == "abc"
        assert list(tmp_path.glob("*.bak-*.json"))

    def test_annotate_refuses_non_preregistered(self, tmp_path, monkeypatch):
        lp = self._ledger(tmp_path)
        monkeypatch.setattr(run, "LEDGER", lp)
        with pytest.raises(AssertionError):
            run._annotate_ledger("HYP-999", {"date": "x", "by": "t", "note": "n"})

    def test_annotate_sets_verdict_field(self, tmp_path, monkeypatch):
        lp = self._ledger(tmp_path)
        monkeypatch.setattr(run, "LEDGER", lp)
        run._annotate_ledger("HYP-082", {"date": "2026-07-07", "by": "test", "note": "n"},
                             verdict="CONFIRMED")
        out = json.loads(lp.read_text())
        e = next(x for x in out if x["id"] == "HYP-082")
        assert e["verdict"] == "CONFIRMED"

    def test_seal_dry_run_writes_artifact_but_not_ledger(self, tmp_path, monkeypatch):
        lp = self._ledger(tmp_path)
        monkeypatch.setattr(run, "LEDGER", lp)
        monkeypatch.setattr(run, "OUT", tmp_path / "geometry_family")
        run.seal("HYP-082", {"primary": {"raw_p": 0.01, "N": 100}, "sample_status": "OK"}, dry_run=True)
        assert (tmp_path / "geometry_family" / "HYP-082.json").exists()
        out = json.loads(lp.read_text())
        e = next(x for x in out if x["id"] == "HYP-082")
        assert e.get("annotations", []) == []

    def test_seal_writes_annotation_when_not_dry_run(self, tmp_path, monkeypatch):
        lp = self._ledger(tmp_path)
        monkeypatch.setattr(run, "LEDGER", lp)
        monkeypatch.setattr(run, "OUT", tmp_path / "geometry_family")
        run.seal("HYP-082", {"primary": {"raw_p": 0.01, "N": 100}, "sample_status": "OK"}, dry_run=False)
        out = json.loads(lp.read_text())
        e = next(x for x in out if x["id"] == "HYP-082")
        assert len(e["annotations"]) == 1


# ── adjudication verdict mapping (all outcome classes) ──────────────────────────────────────

class TestMemberVerdictMapping:
    def test_underpowered_overrides_everything(self):
        assert run._member_verdict("HYP-082", bh_pass=True, underpowered=True,
                                   doc={"gates": {"fold_sign_consistent": True, "ic_threshold_pass": True,
                                                  "cost_floor_pass": True}}) == "UNDERPOWERED"

    def test_bh_fail_is_not_significant(self):
        assert run._member_verdict("HYP-082", bh_pass=False, underpowered=False, doc={}) == "NOT_SIGNIFICANT"
        assert run._member_verdict("HYP-083", bh_pass=False, underpowered=False, doc={}) == "NOT_SIGNIFICANT"

    def test_gamma_fold_sign_unstable_is_not_robust(self):
        doc = {"gates": {"fold_sign_consistent": False, "ic_threshold_pass": True, "cost_floor_pass": True}}
        assert run._member_verdict("HYP-082", bh_pass=True, underpowered=False, doc=doc) == "NOT_ROBUST"

    def test_gamma_weak_ic_is_not_significant(self):
        doc = {"gates": {"fold_sign_consistent": True, "ic_threshold_pass": False, "cost_floor_pass": True}}
        assert run._member_verdict("HYP-082", bh_pass=True, underpowered=False, doc=doc) == "NOT_SIGNIFICANT"

    def test_gamma_cost_floor_fail_is_not_significant(self):
        doc = {"gates": {"fold_sign_consistent": True, "ic_threshold_pass": True, "cost_floor_pass": False}}
        assert run._member_verdict("HYP-082", bh_pass=True, underpowered=False, doc=doc) == "NOT_SIGNIFICANT"

    def test_gamma_all_green_is_confirmed(self):
        doc = {"gates": {"fold_sign_consistent": True, "ic_threshold_pass": True, "cost_floor_pass": True}}
        assert run._member_verdict("HYP-082", bh_pass=True, underpowered=False, doc=doc) == "CONFIRMED"

    def test_beta_gate_fail_is_not_diversifier(self):
        doc = {"diversifier_gate": {"pass": False}}
        assert run._member_verdict("HYP-083", bh_pass=True, underpowered=False, doc=doc) == "NOT_DIVERSIFIER"

    def test_beta_gate_pass_is_confirmed(self):
        doc = {"diversifier_gate": {"pass": True}}
        assert run._member_verdict("HYP-083", bh_pass=True, underpowered=False, doc=doc) == "CONFIRMED"


class TestAdjudicateIntegration:
    def _write_doc(self, out_dir, hyp, raw_p, n, extra):
        out_dir.mkdir(parents=True, exist_ok=True)
        doc = {"hyp": hyp, "primary": {"raw_p": raw_p, "N": n}, "sample_status": "OK", **extra}
        (out_dir / f"{hyp}.json").write_text(json.dumps(doc))
        return doc

    def _ledger(self, tmp_path):
        ledger = [{"id": "HYP-082", "status": "PREREGISTERED", "hash_lock": "a", "verdict": None},
                  {"id": "HYP-083", "status": "PREREGISTERED", "hash_lock": "b", "verdict": None}]
        lp = tmp_path / "ledger.json"
        lp.write_text(json.dumps(ledger))
        return lp

    def test_refuses_partial_family(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "geometry_family"
        monkeypatch.setattr(run, "OUT", out_dir)
        monkeypatch.setattr(run, "LEDGER", self._ledger(tmp_path))
        self._write_doc(out_dir, "HYP-082", 0.001, 100,
                        {"gates": {"fold_sign_consistent": True, "ic_threshold_pass": True, "cost_floor_pass": True}})
        assert run.adjudicate(dry_run=False) == 1

    def test_both_confirmed_writes_verdicts(self, tmp_path, monkeypatch, capsys):
        out_dir = tmp_path / "geometry_family"
        monkeypatch.setattr(run, "OUT", out_dir)
        lp = self._ledger(tmp_path)
        monkeypatch.setattr(run, "LEDGER", lp)
        self._write_doc(out_dir, "HYP-082", 0.001, 100,
                        {"gates": {"fold_sign_consistent": True, "ic_threshold_pass": True, "cost_floor_pass": True}})
        self._write_doc(out_dir, "HYP-083", 0.01, 60, {"diversifier_gate": {"pass": True}})
        rc = run.adjudicate(dry_run=False)
        assert rc == 0
        out = json.loads(lp.read_text())
        e082 = next(x for x in out if x["id"] == "HYP-082")
        e083 = next(x for x in out if x["id"] == "HYP-083")
        assert e082["verdict"] == "CONFIRMED" and e083["verdict"] == "CONFIRMED"
        assert len(e082["annotations"]) == 1 and len(e083["annotations"]) == 1
        printed = capsys.readouterr().out
        assert "E4 PROTOCOL" in printed

    def test_bh_fail_one_member_not_significant(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "geometry_family"
        monkeypatch.setattr(run, "OUT", out_dir)
        monkeypatch.setattr(run, "LEDGER", self._ledger(tmp_path))
        self._write_doc(out_dir, "HYP-082", 0.001, 100,
                        {"gates": {"fold_sign_consistent": True, "ic_threshold_pass": True, "cost_floor_pass": True}})
        self._write_doc(out_dir, "HYP-083", 0.80, 60, {"diversifier_gate": {"pass": True}})
        run.adjudicate(dry_run=False)
        out = json.loads((run.LEDGER).read_text())
        e083 = next(x for x in out if x["id"] == "HYP-083")
        assert e083["verdict"] == "NOT_SIGNIFICANT"

    def test_underpowered_member(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "geometry_family"
        monkeypatch.setattr(run, "OUT", out_dir)
        monkeypatch.setattr(run, "LEDGER", self._ledger(tmp_path))
        self._write_doc(out_dir, "HYP-082", 0.001, 100,
                        {"gates": {"fold_sign_consistent": True, "ic_threshold_pass": True, "cost_floor_pass": True}})
        self._write_doc(out_dir, "HYP-083", 0.02, 30, {"diversifier_gate": {"pass": True}})  # N<50
        run.adjudicate(dry_run=False)
        out = json.loads((run.LEDGER).read_text())
        e083 = next(x for x in out if x["id"] == "HYP-083")
        assert e083["verdict"] == "UNDERPOWERED"

    def test_dry_run_writes_no_annotations(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "geometry_family"
        monkeypatch.setattr(run, "OUT", out_dir)
        lp = self._ledger(tmp_path)
        monkeypatch.setattr(run, "LEDGER", lp)
        self._write_doc(out_dir, "HYP-082", 0.001, 100,
                        {"gates": {"fold_sign_consistent": True, "ic_threshold_pass": True, "cost_floor_pass": True}})
        self._write_doc(out_dir, "HYP-083", 0.01, 60, {"diversifier_gate": {"pass": True}})
        rc = run.adjudicate(dry_run=True)
        assert rc == 0
        out = json.loads(lp.read_text())
        assert all(e.get("annotations", []) == [] for e in out)


# ── load_geometry: reads sentiment_geometry_daily via a tmp-path DB (never the real one) ────

class TestLoadGeometry:
    def test_reads_and_pivots_by_pair(self, tmp_path, monkeypatch):
        db_path = tmp_path / "sentiment_test.db"
        monkeypatch.setattr(store, "DB_PATH", db_path)
        con = store.connect()
        try:
            rows = pd.DataFrame([
                {"date": date(2024, 1, 2), "pair": "EURUSD", "corridor_r2": 0.5, "corridor_dev": 1.1,
                 "corridor_window": 120, "fvg_count_20d": 1, "fvg_unfilled": 0, "tri_state": 0,
                 "days_in_consolidation": 0, "range_slope": 0.01, "src_last_bar_date": date(2024, 1, 2),
                 "fetched_at": pd.Timestamp.now("UTC")},
                {"date": date(2024, 1, 3), "pair": "USDJPY", "corridor_r2": 0.6, "corridor_dev": -0.9,
                 "corridor_window": 120, "fvg_count_20d": 2, "fvg_unfilled": 1, "tri_state": 1,
                 "days_in_consolidation": 3, "range_slope": -0.02, "src_last_bar_date": date(2024, 1, 3),
                 "fetched_at": pd.Timestamp.now("UTC")},
            ])
            store.upsert(con, "sentiment_geometry_daily", rows, ["date", "pair"])
        finally:
            con.close()
        out = run.load_geometry(run.PAIRS)
        assert set(out) == set(run.PAIRS)
        assert len(out["EURUSD"]) == 1 and out["EURUSD"]["corridor_dev"].iloc[0] == pytest.approx(1.1)
        assert len(out["USDJPY"]) == 1 and out["USDJPY"]["corridor_dev"].iloc[0] == pytest.approx(-0.9)
        assert out["GBPUSD"].empty and out["AUDUSD"].empty
