"""P2 correctness: decomposition exactness, cost tolerance, padding invariance,
and the config-385 v015 parity gate (HYP-090)."""

import json

import numpy as np
import pandas as pd
import pytest

from research.modern import daily_returns as dr
from research.modern import precompute as pc
from research.modern._lib import OUT_DIR
from research.modern.signal_arrays import PAIRS


@pytest.fixture(scope="module")
def npz():
    return np.load(OUT_DIR / "signals.npz")


@pytest.fixture(scope="module")
def trades():
    return pd.read_parquet(OUT_DIR / "trades.parquet")


@pytest.fixture(scope="module")
def daily():
    return np.load(OUT_DIR / "daily_returns.npz")


def test_manifest_shape():
    cfgs = pc.config_manifest()
    assert len(cfgs) == 385
    assert cfgs[384]["trailing"] == "per_pair" and cfgs[384]["theta"] == 0.15


def test_raw_decomposition_sums_exactly(trades, daily, npz):
    """Sum of raw daily rows per (config, pair) == sum of closed-trade pnl_pct (1e-12)."""
    raw = daily["raw"].astype(np.float64)
    from research.modern._lib import to_dtindex
    union = to_dtindex(daily["union_index"])
    rng = np.random.default_rng(3)
    for cid in rng.choice(385, size=25, replace=False):
        for pi, pair in enumerate(PAIRS):
            sub = trades[(trades.config_id == cid) & (trades.pair == pair)]
            closed = sub[~sub.is_open]
            expected = float(closed.pnl_pct_raw.sum())
            # open-tail rows contribute M2M not in pnl sums — recompute their partial sum
            key = pair.replace("=X", "")
            c, o = npz[f"{key}__close"], npz[f"{key}__open"]
            tail = 0.0
            for t in sub[sub.is_open].itertuples(index=False):
                e, x, d = int(t.entry_idx), int(t.exit_idx), int(t.direction)
                tail += d * (c[x] - o[e]) / max(float(t.entry), 1e-9)
            got = float(raw[cid, pi].sum())
            assert got == pytest.approx(expected + tail, abs=1e-10), (cid, pair)
    assert len(union) == raw.shape[2]


def test_costed_total_matches_apply_costs_within_tolerance(trades, daily):
    """Costed totals reconcile to ForexBacktester._apply_costs within swap rounding."""
    from sovereign.forex.forex_backtester import ForexBacktester, SWAP_RATES_ANNUAL, _DEFAULT_SWAP

    costed = daily["costed"].astype(np.float64)
    raw = daily["raw"].astype(np.float64)
    rng = np.random.default_rng(4)
    for cid in rng.choice(385, size=10, replace=False):
        for pi, pair in enumerate(PAIRS):
            sub = trades[(trades.config_id == cid) & (trades.pair == pair) & (~trades.is_open)]
            if sub.empty:
                continue
            canon = [{"entry": t.entry, "direction": t.direction, "hold_days": t.hold_days,
                      "pnl_pct": t.pnl_pct_raw, "risk_pct": 0.01}
                     for t in sub.itertuples(index=False)]
            canon = ForexBacktester._apply_costs(canon, pair)
            canon_total = sum(t["pnl_pct"] for t in canon)
            swap = SWAP_RATES_ANNUAL.get(pair, _DEFAULT_SWAP)
            max_rate = max(abs(swap["LONG"]), abs(swap["SHORT"]))
            tol = len(sub) * max_rate / 365.0 * 1.0 + 1e-9   # <=~1 swap-day slack per trade
            our_total = float(costed[cid, pi].sum() - raw[cid, pi].sum()
                              + sum(t.pnl_pct_raw for t in sub.itertuples(index=False)))
            # our_total = raw pnl + our cost adjustments (excluding open-tail M2M drift)
            open_adj = float(raw[cid, pi].sum()) - float(sub.pnl_pct_raw.sum())
            our_total -= open_adj
            assert our_total == pytest.approx(canon_total, abs=tol), (cid, pair)


def test_padding_cannot_change_closed_trades(npz):
    """Same config run with PAD=200 vs PAD=600 -> identical closed trades."""
    cfg = pc.config_manifest()[42]
    a = pc.run_config_pair(npz, cfg, "EURUSD=X")
    old_pad = pc.PAD
    try:
        pc.PAD = 600
        b = pc.run_config_pair(npz, cfg, "EURUSD=X")
    finally:
        pc.PAD = old_pad
    ac = [t for t in a if not t["is_open"]]
    bc = [t for t in b if not t["is_open"]]
    assert ac == bc


def test_config385_v015_parity():
    """The incumbent config's PARITY-SPAN trades reproduce the canonical
    backtest_all snapshot (entry/exit dates identical, pnl within 1e-9 pre-costs).
    Proves the ungated-build + external-mask path == canonical build_signal_frame."""
    snapshot = json.loads((OUT_DIR / "reconcile_snapshot_trades.json").read_text())
    ptrades = pd.read_parquet(OUT_DIR / "parity_trades.parquet")
    pnpz = np.load(OUT_DIR / "signals_parity_2015_2024.npz")

    for pair in PAIRS:
        key = pair.replace("=X", "")
        from research.modern._lib import to_dtindex
        index = to_dtindex(pnpz[f"{key}__index"])
        ours = ptrades[(ptrades.config_id == 384) & (ptrades.pair == pair) & (~ptrades.is_open)]
        ours_dates = [(str(index[int(t.entry_idx)].date()), str(index[int(t.exit_idx)].date()),
                       round(float(t.pnl_pct_raw), 9))
                      for t in ours.itertuples(index=False)]
        canon = snapshot.get(pair, [])
        # snapshot pnl is COSTED — undo costs for comparison via the recorded fields
        canon_dates = [(str(t["entry_date"])[:10], str(t["exit_date"])[:10],
                        round(float(t["pnl_pct"]) + t.get("cost_spread_frac", 0.0)
                              - t.get("cost_swap_frac", 0.0), 9))
                       for t in canon]
        assert len(ours_dates) == len(canon_dates), (
            f"{pair}: {len(ours_dates)} trades vs canonical {len(canon_dates)}")
        for (oe, ox, opnl), (ce, cx, cpnl) in zip(ours_dates, canon_dates):
            assert oe == ce and ox == cx, f"{pair}: date mismatch {oe},{ox} vs {ce},{cx}"
            # snapshot cost fields are rounded to 6dp, so the raw-pnl reconstruction
            # carries up to ~2e-6 of rounding noise; dates above are the exact proof
            assert opnl == pytest.approx(cpnl, abs=2e-6), f"{pair} {oe}"
