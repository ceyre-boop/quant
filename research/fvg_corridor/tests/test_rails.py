"""HYP-098 rails — fence, causality (truncation equivalence), pivot delay,
FVG known-values, determinism, AST isolation.
Run: python3 -m pytest research/fvg_corridor/tests/ -q  (not main-suite collected)
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from research.fvg_corridor import core  # noqa: E402

ALLOWED = ("research.modern._lib", "research.yield_frontier", "research.fvg_corridor",
           "sovereign.discovery", "sovereign.reporting")
FORBIDDEN_ROOTS = ("ict", "config")


def test_isolation_ast():
    for py in HERE.glob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                root = m.split(".")[0]
                if root in ("sovereign", "research"):
                    assert m.startswith(ALLOWED), f"{py.name}: {m}"
                assert root not in FORBIDDEN_ROOTS, f"{py.name}: {m}"


def test_fences():
    mine = core.load_nq_5min("mining")
    assert str(mine.index.max()) <= "2024-07-01"
    hold = core.load_nq_5min("holdout")
    assert str(hold.index.min()) >= "2024-07-01"
    assert len(mine) > 100_000 and len(hold) > 50_000


def synth(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    c = 15000 + np.cumsum(rng.normal(0, 10, n))
    h = c + rng.uniform(1, 15, n)
    l = c - rng.uniform(1, 15, n)
    o = c + rng.normal(0, 3, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "v": 100}, index=idx)


def test_truncation_equivalence_causality():
    """Gold standard: features at bar t-1 identical whether or not bars >= t exist."""
    df = synth()
    for t in (700, 900, 1100):
        full = core.corridor_features(df)
        part = core.corridor_features(df.iloc[:t])
        for d in core.DEPTHS[:2]:            # depth 250 needs longer series
            a, b = full[d]["pos"][t - 1], part[d]["pos"][t - 1]
            assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9, (d, t)
        atr_f = core.causal_atr(df)[t - 1]
        atr_p = core.causal_atr(df.iloc[:t])[t - 1]
        assert (np.isnan(atr_f) and np.isnan(atr_p)) or abs(atr_f - atr_p) < 1e-9


def test_pivot_confirmation_delay():
    """A depth-d pivot at index i must not influence corridors before i+d."""
    n, d = 400, 10
    x = np.linspace(0, 1, n) * 0
    x += np.sin(np.arange(n) / 7) * 50 + 15000       # oscillating highs
    piv_i, conf_i, val = core.confirmed_pivots(x, d, True)
    assert len(piv_i) > 2
    assert ((conf_i - piv_i) == d).all()
    # corridor at conf time must only use pivots with conf_i <= t
    df = synth(400)
    up, lo = core.corridor_lines(df, d)
    ph, ch, vh = core.confirmed_pivots(df["h"].to_numpy(), d, True)
    if len(ph) >= 2:
        first_usable = ch[1]
        assert np.isnan(up[:first_usable]).all()
        assert np.isfinite(up[first_usable])


def test_fvg_known_values():
    # bull FVG: bar2.low > bar0.high
    rows = [
        dict(o=100, h=101, l=99, c=100.5),   # bar0
        dict(o=101, h=105, l=100.9, c=104.8),  # displacement
        dict(o=105, h=106, l=103, c=105.5),  # bar2: low 103 > bar0 high 101 -> bull FVG
        dict(o=105, h=105, l=104, c=104.5),
    ] + [dict(o=104, h=105, l=103.5, c=104) for _ in range(30)]
    idx = pd.date_range("2020-01-01", periods=len(rows), freq="5min", tz="UTC")
    df = pd.DataFrame(rows, index=idx).astype(float)
    df["v"] = 1
    atr = np.full(len(df), 1.0)
    fidx, side, top, bot = core.detect_fvgs(df, atr, 0.5)
    assert 2 in fidx
    k = list(fidx).index(2)
    assert side[k] == 1 and abs(top[k] - 103) < 1e-9 and abs(bot[k] - 101) < 1e-9


def test_determinism():
    df = synth()
    a = core.corridor_features(df)
    b = core.corridor_features(df)
    for d in core.DEPTHS:
        assert np.array_equal(a[d]["pos"], b[d]["pos"], equal_nan=True)
