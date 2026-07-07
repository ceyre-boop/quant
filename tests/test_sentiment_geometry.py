"""tests/test_sentiment_geometry.py — G2 trailing price-geometry features (TICK-018).

Offline & deterministic: synthetic OHLC frames + an in-memory DuckDB (no network, no real
data/sentiment.db). Proves: (1) truncation-invariance, (2) no look-ahead (future bars never
change a past row), (3) the replicated FVG kernel matches the ict canon on truncated frames
(legal here — a test file, not sovereign/sentiment/geometry_feed.py, importing both sides),
(4) tri-state correctly reads contracting vs expanding range wedges, (5) the board's ASOF join
carries a geometry row forward without ever back-dating it, (6) the look-ahead auditor covers
the new table, and (7) the isolation wall (AST-checked) covers geometry_feed.
"""
from __future__ import annotations

import ast
import inspect
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from sovereign.sentiment import store, board_state, vix_feed, geometry_feed

NOW = datetime(2026, 6, 30, tzinfo=timezone.utc)

GEOM_CFG = {
    "corridor_window": 120, "fvg_max_age": 20, "fvg_min_atr_frac": 0.3,
    "tri_window": 20, "tri_pctile": 0.25, "start": "2000-01-01",
}


# ── synthetic OHLC builders ────────────────────────────────────────────────────────────────

def _make_ohlc(n, seed=0, start="2020-01-01", drift=0.0002, vol=0.006):
    """A realistic-shaped daily OHLC frame: geometric random walk + independent H/L jitter,
    internally consistent (High >= max(O,C), Low <= min(O,C))."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    rets = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    hi_jit = np.abs(rng.normal(0, vol * 0.3, n))
    lo_jit = np.abs(rng.normal(0, vol * 0.3, n))
    high = np.maximum(open_, close) * (1 + hi_jit)
    low = np.minimum(open_, close) * (1 - lo_jit)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)
    df.index.name = "Date"
    return df


def _inject_gap(df, at, kind="bull", gap_size=3.0):
    """Overwrite bars [at, at+2] with a clean 3-bar Fair Value Gap of the given kind."""
    df = df.copy()
    loc = {c: df.columns.get_loc(c) for c in ("Open", "High", "Low", "Close")}
    base = float(df["Close"].iloc[at])
    if kind == "bull":
        df.iloc[at, [loc["Open"], loc["Close"], loc["High"], loc["Low"]]] = [base, base, base + 0.2, base - 0.2]
        df.iloc[at + 1, [loc["Open"], loc["Close"], loc["High"], loc["Low"]]] = \
            [base + 0.3, base + 0.4, base + 0.5, base - 0.1]
        top = base + gap_size
        df.iloc[at + 2, [loc["Open"], loc["Close"], loc["High"], loc["Low"]]] = \
            [top, top + 0.3, top + 0.5, top]
    else:
        df.iloc[at, [loc["Open"], loc["Close"], loc["High"], loc["Low"]]] = [base, base, base + 0.2, base - 0.2]
        df.iloc[at + 1, [loc["Open"], loc["Close"], loc["High"], loc["Low"]]] = \
            [base - 0.3, base - 0.4, base + 0.1, base - 0.5]
        bottom = base - gap_size
        df.iloc[at + 2, [loc["Open"], loc["Close"], loc["High"], loc["Low"]]] = \
            [bottom, bottom - 0.3, bottom, bottom - 0.5]
    return df


def _make_wedge(n, start="2019-01-01", contracting=True, base_range=5.0, k=0.01, floor=0.5, price0=100.0):
    """A daily H-L range that monotonically contracts (or expands) around a flat close —
    a clean synthetic consolidation (or expansion) 'wedge' for tri_state_stats."""
    idx = pd.bdate_range(start, periods=n)
    t = np.arange(n, dtype=float)
    rng_vals = np.maximum(base_range - k * t, floor) if contracting else base_range + k * t
    close = np.full(n, price0, dtype=float)
    df = pd.DataFrame(
        {"Open": close, "High": close + rng_vals / 2, "Low": close - rng_vals / 2, "Close": close}, index=idx
    )
    df.index.name = "Date"
    return df


# ── (1) corridor_stats ──────────────────────────────────────────────────────────────────────

class TestCorridorStats:
    def test_warmup_null_below_window(self):
        close = pd.Series(np.linspace(100, 110, 50))
        r2, dev = geometry_feed.corridor_stats(close, 120)
        assert math.isnan(r2) and math.isnan(dev)

    def test_strong_linear_trend_gives_high_r2(self):
        n = 150
        rng = np.random.default_rng(42)
        t = np.arange(n)
        log_p = 4.6 + 0.001 * t + rng.normal(0, 0.002, n)   # clean trend, small noise
        close = pd.Series(np.exp(log_p))
        r2, dev = geometry_feed.corridor_stats(close, 120)
        assert 0.5 < r2 <= 1.0
        assert abs(dev) < 6

    def test_dev_sign_tracks_last_point_kick(self):
        n = 130
        rng = np.random.default_rng(1)
        t = np.arange(n, dtype=float)
        log_p = 4.6 + rng.normal(0, 0.003, n)   # flat trend, small noise
        log_p[-1] += 0.08                       # kick the LAST point clearly above the fit
        close = pd.Series(np.exp(log_p))
        r2, dev = geometry_feed.corridor_stats(close, 120)
        assert dev > 0

    def test_depends_only_on_its_trailing_window(self):
        # corridor_stats always reads close.iloc[-window:] — giving it MORE history before that
        # window must not change the answer at all (the function has no other way to look ahead;
        # this is the meaningful trailing-only guarantee at this function's own level of scope).
        close_full = pd.Series(_make_ohlc(300, seed=7)["Close"])
        t = 200
        with_extra_history = geometry_feed.corridor_stats(close_full.iloc[: t + 1], 120)
        exact_window_only = geometry_feed.corridor_stats(close_full.iloc[t + 1 - 120: t + 1], 120)
        assert with_extra_history[0] == pytest.approx(exact_window_only[0])
        assert with_extra_history[1] == pytest.approx(exact_window_only[1])


# ── (2)/(3) detect_fvgs_daily + FVG parity vs the ict canon ─────────────────────────────────

class TestFVGParity:
    """Legal here (a test file) to import both sides — geometry_feed.py itself never may."""

    def _assert_matches_canon(self, df, max_age=20, min_atr_frac=0.3):
        from ict.fvg_detector import FVGDetector
        det = FVGDetector()
        det._fvg_max_age = max_age
        det._fvg_min_atr = min_atr_frac
        fvgs, _obs = det.detect(df)
        canon_count = len(fvgs)
        canon_unfilled = sum(1 for f in fvgs if not f.filled)
        got_count, got_unfilled = geometry_feed.detect_fvgs_daily(df, max_age, min_atr_frac)
        assert got_count == canon_count, f"count mismatch: got {got_count} vs canon {canon_count}"
        assert got_unfilled == canon_unfilled, f"unfilled mismatch: got {got_unfilled} vs canon {canon_unfilled}"
        return canon_count

    def test_gap_up(self):
        df = _inject_gap(_make_ohlc(40, seed=11), at=30, kind="bull")
        assert self._assert_matches_canon(df) >= 1   # the engineered gap actually registered

    def test_gap_down(self):
        df = _inject_gap(_make_ohlc(40, seed=12), at=30, kind="bear")
        assert self._assert_matches_canon(df) >= 1

    def test_no_gap(self):
        idx = pd.bdate_range("2021-01-01", periods=40)
        df = pd.DataFrame({"Open": 100.0, "High": 100.3, "Low": 99.7, "Close": 100.0}, index=idx)
        assert self._assert_matches_canon(df) == 0

    def test_parity_holds_at_multiple_as_of_points(self):
        df = _inject_gap(_make_ohlc(90, seed=13), at=40, kind="bull")
        df = _inject_gap(df, at=60, kind="bear")
        for t in (45, 65, 89):
            self._assert_matches_canon(df.iloc[: t + 1])

    def test_warmup_null_below_full_lookback(self):
        df = _make_ohlc(15, seed=20)   # fewer than max_age(20)+2 = 22 bars
        count, unfilled = geometry_feed.detect_fvgs_daily(df, max_age=20, min_atr_frac=0.3)
        assert count is None and unfilled is None

    def test_true_zero_below_min_bars(self):
        df = _make_ohlc(3, seed=21)
        count, unfilled = geometry_feed.detect_fvgs_daily(df, max_age=20, min_atr_frac=0.3)
        assert count == 0 and unfilled == 0


# ── (4) tri_state_stats ──────────────────────────────────────────────────────────────────────

class TestTriState:
    def test_contracting_wedge_is_consolidating(self):
        n = 320
        df = _make_wedge(n, contracting=True)
        state, days, slope = geometry_feed.tri_state_stats(df, window=20, pctile=0.25)
        assert state is True
        assert slope < 0
        assert days == n - 252 + 1   # monotonic shrink -> every post-warmup day qualifies

    def test_expanding_wedge_is_not_consolidating(self):
        df = _make_wedge(320, contracting=False)
        state, days, slope = geometry_feed.tri_state_stats(df, window=20, pctile=0.25)
        assert state is False
        assert slope > 0
        assert days == 0

    def test_warmup_null_below_252_floor(self):
        df = _make_wedge(200, contracting=True)
        state, days, slope = geometry_feed.tri_state_stats(df, window=20, pctile=0.25)
        assert state is None and days is None and math.isnan(slope)


# ── compute_pair: truncation-invariance + look-ahead trap across all 7 columns at once ──────

class TestComputePair:
    def test_truncation_invariance(self):
        df = _make_ohlc(400, seed=30)
        full = geometry_feed.compute_pair(df, "EURUSD", GEOM_CFG)
        for t in (150, 300, 399):
            trunc = geometry_feed.compute_pair(df.iloc[: t + 1], "EURUSD", GEOM_CFG)
            pd.testing.assert_series_equal(full.iloc[t], trunc.iloc[-1], check_names=False)

    def test_look_ahead_trap_future_shock(self):
        df = _make_ohlc(300, seed=31)
        t = 200
        before = geometry_feed.compute_pair(df, "EURUSD", GEOM_CFG).iloc[t]
        last_close = float(df["Close"].iloc[-1])
        shock_idx = df.index[-1] + pd.Timedelta(days=1)
        shock = pd.DataFrame(
            {"Open": [last_close * 1000], "High": [last_close * 2000],
             "Low": [0.01], "Close": [last_close * 1500]},
            index=[shock_idx],
        )
        df2 = pd.concat([df, shock])
        after = geometry_feed.compute_pair(df2, "EURUSD", GEOM_CFG).iloc[t]
        pd.testing.assert_series_equal(before, after, check_names=False)

    def test_warmup_rows_never_dropped_always_null(self):
        df = _make_ohlc(50, seed=32)
        out = geometry_feed.compute_pair(df, "EURUSD", GEOM_CFG)
        assert len(out) == len(df)                          # never dropped
        assert pd.isna(out["corridor_r2"].iloc[0])
        assert pd.isna(out["tri_state"].iloc[0])

    def test_start_filters_output_coverage(self):
        df = _make_ohlc(500, seed=33, start="2014-01-01")
        cfg = dict(GEOM_CFG, start="2015-01-01")
        out = geometry_feed.compute_pair(df, "EURUSD", cfg)
        assert (out["date"] >= pd.Timestamp("2015-01-01").date()).all()


# ── board integration: ASOF visibility (absent at d-1, present d and d+1) ──────────────────

class TestBoardIntegration:
    def test_geometry_asof_visibility(self):
        con = store.connect(path=":memory:")
        try:
            dates = [pd.Timestamp(d).date() for d in ("2024-01-02", "2024-01-03", "2024-01-04")]
            vix = pd.DataFrame({"date": dates, "vix_close": [15.0, 15.5, 16.0]})
            vix["vix_5d_ago"] = vix["vix_close"]
            vix["vix_momentum"] = 0.0
            vix["vix_regime"] = [vix_feed.classify_regime(v) for v in vix["vix_close"]]
            vix["fetched_at"] = NOW
            store.upsert(con, "sentiment_vix_daily", vix, ["date"])

            geo = pd.DataFrame([{
                "date": dates[1], "pair": "EURUSD", "corridor_r2": 0.77, "corridor_dev": 1.2,
                "corridor_window": 120, "fvg_count_20d": 2, "fvg_unfilled": 1, "tri_state": 1,
                "days_in_consolidation": 4, "range_slope": -0.01, "src_last_bar_date": dates[1],
                "fetched_at": NOW,
            }])
            store.upsert(con, "sentiment_geometry_daily", geo, ["date", "pair"])

            board_state.rebuild(con=con)
            before = board_state.get_state(dates[0], "EURUSD", con=con)
            on = board_state.get_state(dates[1], "EURUSD", con=con)
            after = board_state.get_state(dates[2], "EURUSD", con=con)

            assert before is not None and pd.isna(before["corridor_r2"])   # not yet visible
            assert on["corridor_r2"] == pytest.approx(0.77)
            assert on["tri_state"] == 1
            assert after["corridor_r2"] == pytest.approx(0.77)             # carried forward (ASOF)
            assert after["tri_state"] == 1
        finally:
            con.close()

    def test_board_has_geometry_columns(self):
        for col in ("corridor_r2", "corridor_dev", "fvg_count_20d", "fvg_unfilled", "tri_state",
                    "days_in_consolidation", "range_slope"):
            assert col in board_state.REQUIRED_COLUMNS


# ── audit_look_ahead: geometry provenance check ─────────────────────────────────────────────

class TestGeometryAudit:
    def _seed_row(self, con, date, pair, src_last_bar_date):
        geo = pd.DataFrame([{
            "date": date, "pair": pair, "corridor_r2": 0.5, "corridor_dev": 0.1,
            "corridor_window": 120, "fvg_count_20d": 1, "fvg_unfilled": 0, "tri_state": 0,
            "days_in_consolidation": 0, "range_slope": 0.001, "src_last_bar_date": src_last_bar_date,
            "fetched_at": NOW,
        }])
        store.upsert(con, "sentiment_geometry_daily", geo, ["date", "pair"])

    def test_clean_geometry_zero_violations(self):
        con = store.connect(path=":memory:")
        try:
            d = pd.Timestamp("2024-03-01").date()
            self._seed_row(con, d, "EURUSD", d)   # src_last_bar_date == date
            from scripts.audit_look_ahead import audit
            results = audit(con)
            geo_results = [r for r in results if r["table"] == "sentiment_geometry_daily"]
            assert geo_results, "auditor did not check sentiment_geometry_daily"
            assert sum(r["violations"] for r in geo_results) == 0, geo_results
        finally:
            con.close()

    def test_catches_mismatched_src_last_bar_date(self):
        con = store.connect(path=":memory:")
        try:
            d = pd.Timestamp("2024-03-01").date()
            bad = pd.Timestamp("2024-02-28").date()
            self._seed_row(con, d, "LEAKPAIR", bad)   # src_last_bar_date != date
            from scripts.audit_look_ahead import audit
            viol = {(r["table"], r["check"]): r["violations"] for r in audit(con)}
            assert viol[("sentiment_geometry_daily", "provenance_last_bar_matches_date")] == 1
        finally:
            con.close()


# ── isolation wall: standalone AST check (belt-and-suspenders alongside the edited
#    tests/test_sentiment_board.py::test_sentiment_does_not_import_ict, which is the
#    MANDATORY coverage the plan calls for) ──────────────────────────────────────────────────

def test_geometry_feed_does_not_import_ict():
    forbidden_roots = {"ict", "ict_engine", "layer1", "layer2"}
    tree = ast.parse(inspect.getsource(geometry_feed))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    for name in names:
        assert name.split(".")[0] not in forbidden_roots, f"geometry_feed imports forbidden module {name!r}"


# ── update(): local-parquet feeder idiom (never touches the real data/sentiment.db) ─────────

class TestUpdate:
    def test_missing_parquet_skips_loudly(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(geometry_feed, "SPOT_CACHE", tmp_path)   # empty dir -> every pair missing
        con = store.connect(path=":memory:")
        try:
            cov = geometry_feed.update(con=con, pairs=["EURUSD"], start="2015-01-01")
            printed = capsys.readouterr().out
            assert "MISSING parquet" in printed
            assert cov["EURUSD"]["rows"] == 0
        finally:
            con.close()

    def test_writes_and_reads_back(self, tmp_path, monkeypatch):
        df = _make_ohlc(300, seed=40, start="2014-01-01")
        df.to_parquet(tmp_path / "EURUSD_ohlc.parquet")
        monkeypatch.setattr(geometry_feed, "SPOT_CACHE", tmp_path)
        con = store.connect(path=":memory:")
        try:
            cov = geometry_feed.update(con=con, pairs=["EURUSD"], start="2014-06-01")
            assert cov["EURUSD"]["rows"] > 0
            n = con.execute("SELECT COUNT(*) FROM sentiment_geometry_daily").fetchone()[0]
            assert n == cov["EURUSD"]["rows"]
        finally:
            con.close()
