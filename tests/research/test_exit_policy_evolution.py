"""tests/research/test_exit_policy_evolution.py — HYP-067 evolutionary exit-policy search.

Offline tests (no yfinance / no backtester I/O): the HYP-066 gate, replay parity vs the
HYP-066 single-trade replay, GA determinism + output shape, and the prove band. The GA is
exercised on synthetic trade specs so it runs in CI without network.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_epe():
    path = ROOT / "scripts" / "research" / "exit_policy_evolution.py"
    spec = importlib.util.spec_from_file_location("exit_policy_evolution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


epe = _load_epe()


# ── HYP-066 gate ──────────────────────────────────────────────────────────── #

def test_gate_halts_unless_valid_or_standalone():
    # default run halts whenever the prior is not VALID_EDGE
    assert epe.gate_should_halt("NOT_SIGNIFICANT", standalone=False) is True
    assert epe.gate_should_halt("ABSENT", standalone=False) is True
    # --standalone always overrides the gate
    assert epe.gate_should_halt("NOT_SIGNIFICANT", standalone=True) is False
    # a genuine prior pass runs without the flag
    assert epe.gate_should_halt("VALID_EDGE", standalone=False) is False


def test_hyp066_on_disk_is_not_significant():
    # the real prior result: this is exactly why the default run halts today
    assert epe.hyp066_verdict() == "NOT_SIGNIFICANT"


# ── replay parity vs the HYP-066 single-trade replay ──────────────────────── #

def _build_arr(closes, atr, signal, hold):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    return {
        "pair": "EURUSD=X", "idx": None, "pos": {},
        "opens": closes.copy(), "closes": closes,
        "atr": np.asarray(atr, dtype=float),
        "signal": np.asarray(signal, dtype=int),
        "hold": np.asarray(hold, dtype=int),
    }


def _spec_from_arr(arr, epos, direction):
    """Mirror build_trade_specs for one trade so replay_policy sees the same bars as replay_exit."""
    n = len(arr["closes"])
    end = min(epos + epe.MAX_BARS, n)
    nb = end - epos
    cells = np.zeros(nb, dtype=np.int16)  # constant policy → cell choice irrelevant
    entry_price = float(arr["opens"][epos])
    entry_atr = max(float(arr["atr"][epos - 1]), 1e-6)
    return {
        "index": 0, "pair": arr["pair"], "direction": direction,
        "entry_dt": np.datetime64("2015-01-01"), "exit_dt": np.datetime64("2015-02-01"),
        "entry_price": entry_price, "entry_atr": entry_atr,
        "bar_close": arr["closes"][epos:end], "bar_atr": arr["atr"][epos:end],
        "bar_sig": arr["signal"][epos:end], "bar_hold": arr["hold"][epos:end],
        "cells": cells, "entry_cell": 0, "marginal": 0, "visited": np.unique(cells), "static_pnl": 0.0,
    }


@pytest.mark.parametrize("direction", [1, -1])
def test_replay_policy_matches_hyp066_replay(direction):
    # a price path that wanders then drifts, with a mid-trade reversal signal
    rng = np.random.default_rng(3)
    n = 40
    closes = 100.0 * np.cumprod(1 + rng.normal(0, 0.004, n))
    atr = np.full(n, 0.01)
    signal = np.zeros(n, dtype=int)
    signal[12] = -direction  # reversal mid-trade
    hold = np.full(n, 18, dtype=int)
    arr = _build_arr(closes, atr, signal, hold)
    epos, stop, trail = 1, 2.0, 1.25

    ref = epe.replay_exit(arr, epos, direction, stop, trail)
    ref_costed = epe._apply_costs("EURUSD=X", ref["entry"], direction, ref["hold_days"], ref["pnl_pct"])

    spec = _spec_from_arr(arr, epos, direction)
    hold_at_entry = int(arr["hold"][epos])  # replay_exit uses hold[epos-1]; both 18 here
    const_policy = np.tile([stop, trail, hold_at_entry], (epe.N_CELLS, 1)).astype(float)
    got = epe.replay_policy(spec, const_policy)

    assert got == pytest.approx(ref_costed, abs=1e-9)


# ── GA determinism + output shape on synthetic specs ──────────────────────── #

def _make_spec(i, pair, direction, t0_days, rng):
    nb = int(rng.integers(8, 40))
    closes = 100.0 * np.cumprod(1 + rng.normal(0, 0.002, nb))
    sigs = np.zeros(nb, dtype=int)
    if rng.random() < 0.3:
        sigs[int(rng.integers(1, nb))] = -direction
    cells = rng.integers(0, epe.N_CELLS, nb).astype(np.int16)
    cells[0] = epe.cell_index(int(rng.integers(0, 3)), int(rng.integers(0, 3)), 0)
    entry_dt = np.datetime64("2015-01-01") + np.timedelta64(t0_days, "D")
    return {
        "index": i, "pair": pair, "direction": direction,
        "entry_dt": entry_dt, "exit_dt": entry_dt + np.timedelta64(nb, "D"),
        "entry_price": 100.0, "entry_atr": 0.01,
        "bar_close": closes, "bar_atr": np.full(nb, 0.01), "bar_sig": sigs, "bar_hold": np.full(nb, 60, dtype=int),
        "cells": cells, "entry_cell": int(cells[0]), "marginal": epe.cell_marginal(int(cells[0])),
        "visited": np.unique(cells), "static_pnl": float(rng.normal(0.001, 0.02)),
    }


def _synthetic_specs(n=40, seed=7):
    rng = np.random.default_rng(seed)
    return [_make_spec(i, ["EURUSD=X", "GBPUSD=X"][i % 2], 1 if rng.random() < 0.5 else -1, i * 9, rng)
            for i in range(n)]


def test_cpcv_ctx_builds_expected_paths():
    specs = _synthetic_specs()
    ctx = epe.build_fitness_ctx(specs)
    assert len(ctx["test_sets"]) == epe.n_cpcv_splits(epe.CPCV_GROUPS, epe.CPCV_TEST) == 15


def test_ga_is_deterministic_and_well_formed():
    specs = _synthetic_specs()
    counts = epe.visit_counts(specs)
    ctx = epe.build_fitness_ctx(specs)
    quiet = lambda *a, **k: None

    w1, arch1, hist1, n1 = epe.run_ga(specs, counts, ctx, n_pop=12, n_gen=4, decade_best_cfg=None, log=quiet)
    w2, arch2, hist2, n2 = epe.run_ga(specs, counts, ctx, n_pop=12, n_gen=4, decade_best_cfg=None, log=quiet)

    assert np.array_equal(w1, w2)
    assert hist1 == hist2 and n1 == n2
    assert w1.shape == (epe.N_CELLS, 3)
    # winner is snapped + within bounds
    assert epe.STOP_BOUNDS[0] <= w1[:, 0].min() and w1[:, 0].max() <= epe.STOP_BOUNDS[1]
    assert epe.HOLD_BOUNDS[0] <= w1[:, 2].min() and w1[:, 2].max() <= epe.HOLD_BOUNDS[1]

    front = epe.pareto_front(arch1)
    knee = epe.knee_of(front)
    assert 1 <= len(front) <= 20
    assert "policy_key" in knee


def test_min_cell_fallback_overwrites_thin_cells():
    base = epe.random_policy(np.random.default_rng(1))
    counts = np.zeros(epe.N_CELLS, dtype=int)
    keep = epe.cell_index(2, 2, 0)
    counts[keep] = 999  # only one populated cell
    eff = epe.resolve_fallback(base, counts)
    # every thin cell must equal the single populated cell's config (global fallback)
    for c in range(epe.N_CELLS):
        if c != keep:
            assert np.allclose(eff[c], eff[keep])


# ── prove band ────────────────────────────────────────────────────────────── #

def test_prove_band_two_sided():
    assert epe.prove_band_gate(0.69, 1.20)["cleared"] is True
    assert epe.prove_band_gate(0.69, 1.20)["full_decade_ok"] is True
    # above the band (the overfit tell) fails
    assert epe.prove_band_gate(2.50, 1.20)["cleared"] is False
    # OOS outside band fails
    assert epe.prove_band_gate(0.69, 0.50)["oos_ok"] is False
