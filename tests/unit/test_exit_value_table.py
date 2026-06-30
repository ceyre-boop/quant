"""Unit tests for sovereign/discovery/exit_value_table.py — the HYP-071 label generator core.

Network-free and deterministic. Synthetic per-pair caches drive the pool / retrace / rollout / CPCV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sovereign.forex.exit_machine import BarContext, ExitConfig, ExitDecision, PositionState, decide_exit
from sovereign.discovery import exit_value_table as evt

CFG = ExitConfig(stop_atr_mult=2.0, trailing_atr_mult=1.25, strict_mode=False, enable_cb_refresh=True)


def _make_cache(seed: int, n: int = 400, pair_idx: int = 0, drift: float = 0.0):
    """Synthetic single-pair cache with all the keys retrace/pool need."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.006, n)
    closes = 100.0 * np.cumprod(1.0 + rets)
    opens = closes / (1.0 + rng.normal(0, 0.001, n))
    high = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.004, n))
    low = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.004, n))
    tr = np.maximum(high - low, np.abs(high - np.r_[closes[0], closes[:-1]]))
    atr = pd.Series(tr).rolling(14).mean().bfill().to_numpy() / closes
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return {
        "pair": f"P{pair_idx}", "idx": idx, "pos": {ts: i for i, ts in enumerate(idx)},
        "opens": opens, "closes": closes, "atr": atr, "atr_pct": evt.trailing_pct(atr, 60),
        "signal": np.ones(n, dtype=np.int64), "hold": np.full(n, 60, dtype=np.int64),
        "rsi": evt.compute_rsi(closes), "tercile": evt.trailing_pct(atr, 60),
    }


def _reference_terminal(cache, epos, direction, cfg):
    """Mirror retrace's stepping with the scalar decide_exit to find the terminal (pos, reason)."""
    entry_price = float(cache["opens"][epos]); entry_atr = max(float(cache["atr"][epos - 1]), 1e-6)
    risk = cfg.stop_atr_mult * entry_atr * entry_price
    stop = entry_price - risk if direction == 1 else entry_price + risk
    hold_limit = max(int(cache["hold"][epos - 1]), 1)
    state = PositionState(direction, stop, entry_price, entry_price, 0, hold_limit)
    n_hold = 0
    for j in range(epos, len(cache["closes"])):
        res = decide_exit(state, BarContext(float(cache["closes"][j]), float(cache["atr"][j]),
                                            int(cache["signal"][j]), int(cache["hold"][j]), float("nan")), cfg)
        state = res.state
        if res.decision == ExitDecision.HOLD:
            n_hold += 1
        else:
            return j, evt.DECISION_TO_REASON[int(res.decision)], n_hold
    return None, None, n_hold


# ── value formulas ──────────────────────────────────────────────────────────────
class TestValue:
    def test_downside_dev(self):
        assert evt.downside_dev(np.array([-1.0, 1.0, 0.0])) == pytest.approx(np.sqrt(1 / 3))
        assert evt.downside_dev(np.array([1.0, 2.0])) == 0.0      # no downside

    def test_value(self):
        R = np.array([-1.0, 1.0])
        assert evt.value(R) == pytest.approx(0.0 - 0.5 * np.sqrt(0.5))

    def test_excursion_R_at_stop_is_minus_one(self):
        # R = dir*(close-entry)/(stop_atr_mult*atr*entry); at the stop price R == -1
        entry, atr, d = 100.0, 0.01, 1
        risk = CFG.stop_atr_mult * atr * entry
        stop = entry - risk
        R = d * (stop - entry) / risk
        assert R == pytest.approx(-1.0)
        # short side
        d = -1; stop_s = entry + risk
        assert d * (stop_s - entry) / risk == pytest.approx(-1.0)


# ── board ────────────────────────────────────────────────────────────────────────
class TestBoard:
    def test_cell_id_roundtrip(self):
        seen = set()
        for atr in range(3):
            for exc in range(3):
                for h in range(3):
                    for rx in range(2):
                        for ca in range(2):
                            cid = evt.cell_id(atr, exc, h, rx, ca)
                            seen.add(cid)
                            d = evt.decode_cell(cid)
                            assert d["atr_tercile"] == ["low", "mid", "high"][atr]
                            assert d["carry"] == ["aligned", "not_aligned"][ca]
        assert seen == set(range(108))

    def test_evaluated_is_aligned_half(self):
        assert sum(evt.is_evaluated(c) for c in range(108)) == 54
        assert evt.is_evaluated(evt.cell_id(1, 1, 1, 0, 0))       # aligned
        assert not evt.is_evaluated(evt.cell_id(1, 1, 1, 0, 1))   # not-aligned

    def test_buckets(self):
        assert (evt._excursion_bucket(-0.1), evt._excursion_bucket(0.5), evt._excursion_bucket(1.5)) == (0, 1, 2)
        assert (evt._hold_bucket(0.1), evt._hold_bucket(0.5), evt._hold_bucket(0.9)) == (0, 1, 2)
        assert evt._rsi_extreme(75, 1) == 1 and evt._rsi_extreme(75, -1) == 0
        assert evt._rsi_extreme(20, -1) == 1 and evt._carry_bucket(-1, 1) == 1 and evt._carry_bucket(0, 1) == 0


# ── pool + sampler ─────────────────────────────────────────────────────────────
class TestPool:
    def test_starts_in_tercile_and_geometric(self):
        pool = evt.build_return_pool([_make_cache(1), _make_cache(2, pair_idx=1)])
        for t in (0, 1, 2):
            assert len(pool.starts_by_tercile[t]) > 0
        rng = np.random.default_rng(0)
        terc = np.zeros(4000, dtype=np.int64)               # all tercile-0 continuations
        g = evt._sample_paths(terc, 40, pool, rng)
        # every block start (column 0) is in tercile 0
        assert np.isin(g[:, 0], pool.starts_by_tercile[0]).all()
        # mean run length ≈ L=5 (geometric restart prob 1/L)
        cont = (g[:, 1:] == g[:, :-1] + 1)
        run_continues = cont.mean()
        mean_run = 1.0 / (1.0 - run_continues)
        assert 3.5 < mean_run < 7.0

    def test_restrict_dates_shrinks_starts(self):
        c = _make_cache(3)
        full = evt.build_return_pool([c])
        half = evt.build_return_pool([c], restrict_dates=(c["idx"][0], c["idx"][199]))
        assert sum(len(half.starts_by_tercile[t]) for t in (0, 1, 2)) < \
               sum(len(full.starts_by_tercile[t]) for t in (0, 1, 2))


# ── retrace ───────────────────────────────────────────────────────────────────
class TestRetrace:
    def test_members_are_hold_bars_and_parity_matches(self):
        cache = _make_cache(7, drift=-0.002)   # downward drift → a long trips its stop/trail
        epos = 30
        exit_pos, reason, n_hold = _reference_terminal(cache, epos, 1, CFG)
        assert exit_pos is not None
        trade = {"entry_date": cache["idx"][epos], "exit_date": cache["idx"][exit_pos],
                 "direction": 1, "exit_reason": reason}
        members, parity = evt.retrace_members(cache, [trade], CFG, pair_idx=0)
        assert parity["matched"] == 1 and parity["dropped"] == 0
        # members are exactly the HOLD bars whose tercile is defined
        assert 0 < len(members) <= n_hold
        for m in members:
            assert evt.is_evaluated(m["cell_id"])             # HOLD bars are carry-aligned
            assert m["hold_count"] >= 1

    def test_excursion_sign(self):
        cache = _make_cache(9, drift=0.003)    # upward → long shows positive excursion somewhere
        epos = 20
        ep, _, _ = _reference_terminal(cache, epos, 1, CFG)
        trade = {"entry_date": cache["idx"][epos], "exit_date": cache["idx"][ep],
                 "direction": 1, "exit_reason": "x"}
        members, _ = evt.retrace_members(cache, [trade], CFG, pair_idx=0)
        if members:
            assert any(m["excursion_R"] > 0 for m in members)

    def test_unknown_entry_date_dropped(self):
        cache = _make_cache(5)
        trade = {"entry_date": "1999-01-01", "exit_date": "1999-02-01", "direction": 1, "exit_reason": "x"}
        members, parity = evt.retrace_members(cache, [trade], CFG, pair_idx=0)
        assert parity["dropped"] == 1 and members == []


# ── rollout + table ─────────────────────────────────────────────────────────────
def _cell_members(seed=11, n_members=200):
    """Build a MemberArrays for one cell by retracing several synthetic trades."""
    cache = _make_cache(seed, n=500)
    cfg_by_pair = {0: CFG}
    trades = []
    rng = np.random.default_rng(seed)
    for epos in rng.integers(20, 480, size=40):
        ep, reason, _ = _reference_terminal(cache, int(epos), 1, CFG)
        if ep is None:
            continue
        trades.append({"entry_date": cache["idx"][int(epos)], "exit_date": cache["idx"][ep],
                       "direction": 1, "exit_reason": reason})
    members, _ = evt.retrace_members(cache, trades, CFG, pair_idx=0)
    by_cell = evt.group_members_by_cell(members)
    pool = evt.build_return_pool([cache])
    return by_cell, pool, cfg_by_pair


class TestRolloutTable:
    def test_rollout_deterministic_and_finite(self):
        by_cell, pool, cfgs = _cell_members()
        cid, ma = max(by_cell.items(), key=lambda kv: len(kv[1]))  # most-populated cell
        a = evt.rollout_R(ma, pool, cfgs, 3000, np.random.default_rng(1), "live")
        b = evt.rollout_R(ma, pool, cfgs, 3000, np.random.default_rng(1), "live")
        np.testing.assert_array_equal(a, b)               # seeded → reproducible
        assert np.isfinite(a).all()                       # every continuation terminated (TIME cap)

    def test_v_exit_deterministic_no_rng(self):
        by_cell, pool, cfgs = _cell_members()
        cid, ma = next(iter(by_cell.items()))
        assert evt.value(ma.excursion_R) == evt.value(ma.excursion_R)   # no RNG dependence

    def test_compute_table_actions(self):
        by_cell, pool, cfgs = _cell_members()
        table = evt.compute_table(by_cell, pool, cfgs, n_cont=2000)
        assert len(table) > 0
        for cid, res in table.items():
            assert evt.is_evaluated(cid)
            assert res.optimal_action in ("HOLD_AND_TRAIL", "EXIT_NOW")
            assert res.margin == pytest.approx(res.V_hold - res.V_exit)

    def test_signal_frozen_runs(self):
        by_cell, pool, cfgs = _cell_members()
        cid, ma = max(by_cell.items(), key=lambda kv: len(kv[1]))
        frozen = evt.rollout_R(ma, pool, cfgs, 2000, np.random.default_rng(2), "frozen")
        assert np.isfinite(frozen).all()


# ── CPCV + divergence ────────────────────────────────────────────────────────────
class TestCPCVDivergence:
    def test_cpcv_insufficient_on_tiny_cell(self):
        by_cell, pool, cfgs = _cell_members()
        # fabricate a tiny cell
        cid, ma = next(iter(by_cell.items()))
        tiny = ma.subset(np.arange(min(5, len(ma))))
        out = evt.cpcv_sign_stability(tiny, [_make_cache(11, n=500)], cfgs, 0.1, 0, "live")
        assert out["status"] == "CPCV_INSUFFICIENT" and out["sign_stable"] is False

    def test_cpcv_runs_on_populated_cell(self):
        by_cell, pool, cfgs = _cell_members()
        caches = [_make_cache(11, n=500)]
        cid, ma = max(by_cell.items(), key=lambda kv: len(kv[1]))
        if len(ma) >= 2 * evt.MIN_MEMBERS_FOLD:
            full_margin = (evt.value(evt.rollout_R(ma, pool, cfgs, 2000, np.random.default_rng(3), "live"))
                           - evt.value(ma.excursion_R))
            out = evt.cpcv_sign_stability(ma, caches, cfgs, full_margin, 0, "live")
            assert out["status"] in ("OK", "CPCV_INSUFFICIENT")
            assert 0.0 <= (out["stability_fraction"] or 0.0) <= 1.0

    def test_divergence_classification(self):
        # sensible: EXIT_NOW in high-ATR / late / extended / rsi-extreme
        high_atr = evt.cell_id(2, 1, 0, 0, 0)
        assert evt.is_economically_sensible(high_atr, "EXIT_NOW")
        late = evt.cell_id(0, 1, 2, 0, 0)
        assert evt.is_economically_sensible(late, "EXIT_NOW")
        calm = evt.cell_id(0, 1, 0, 0, 0)                 # low-atr early modest not-extreme
        assert not evt.is_economically_sensible(calm, "EXIT_NOW")
        assert not evt.is_economically_sensible(high_atr, "HOLD_AND_TRAIL")
