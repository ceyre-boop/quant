"""M0 gate tests — isolation, fences, frictions, counters, determinism, VRP collision.

Run explicitly: python3 -m pytest research/yield_frontier/tests/ -q
(pytest.ini testpaths=tests keeps this out of the main suite.)
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from research.yield_frontier import _lib, frictions, holdout_guard, yield_board  # noqa: E402


# ---------- isolation (AST whitelist) ----------
ALLOWED_PREFIXES = (
    "research.modern._lib", "research.yield_frontier", "sovereign.discovery",
    "sovereign.reporting",
)
FORBIDDEN_ROOTS = ("ict", "config")


def _imports(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_isolation_ast_whitelist():
    for py in HERE.glob("*.py"):
        for mod in _imports(py):
            root = mod.split(".")[0]
            if root in ("sovereign", "research"):
                assert mod.startswith(ALLOWED_PREFIXES), f"{py.name} imports {mod}"
            assert root not in FORBIDDEN_ROOTS, f"{py.name} imports {mod}"


# ---------- holdout fences ----------
def test_nq_loader_cannot_return_holdout():
    df = holdout_guard.load_nq("daily")
    col = "session" if "session" in df.columns else df.columns[0]
    assert len(df) > 1000
    assert str(df[col].max())[:10] <= "2024-06-30"


def test_chain_files_respect_cutoff():
    files = holdout_guard.chain_files()
    assert len(files) > 500
    assert max(f.stem.split("_")[0] for f in files) <= holdout_guard.OPTIONS_QUOTE_CUTOFF


def test_equities_holdout_absent_and_cache_in_window():
    holdout_guard.assert_no_equities_holdout_on_disk()
    files = holdout_guard.gapper_grouped_files()
    assert len(files) > 200


def test_miners_contain_no_holdout_date_literals():
    for name in ("m1_equities.py", "m2_nq.py", "m3_options.py"):
        fp = HERE / name
        if not fp.exists():
            continue
        src = fp.read_text()
        for bad in ("2024-07", "2024-08", "2024-09", "2024-1", "2025-01", "2025-02",
                    "2023-10", "2023-11", "2023-12"):
            assert bad not in src, f"{name} hardcodes holdout-window date {bad}"


# ---------- frictions known-values ----------
def test_friction_known_values():
    assert frictions.htb_apr(1.2) == 3.00
    assert abs(frictions.short_borrow_cost(1.2, 1) - 3.0 / 365) < 1e-12
    assert abs(frictions.NQ_RT_COST_PTS["MNQ"] - 0.87) < 1e-9
    assert abs(frictions.NQ_RT_COST_PTS["NQ"] - 0.625) < 1e-9
    assert frictions.option_fill(1.00, 0.10, 0.5, selling=True) == 0.95
    assert frictions.option_fill(1.00, 0.10, 0.5, selling=False) == 1.05
    assert frictions.locate_fill_prob(1.5) == 0.50


# ---------- mined-N counter ----------
def test_mined_n_monotonic(tmp_path, monkeypatch):
    monkeypatch.setattr(_lib, "MINED_N_PATH", tmp_path / "mined_n.json")
    monkeypatch.setattr(_lib, "OUT", tmp_path)
    _lib.record_mined("eq", "F1", 10)
    _lib.record_mined("eq", "F1", 10)          # idempotent re-run ok
    _lib.record_mined("nq", "F2", 5)
    assert json.loads((tmp_path / "mined_n.json").read_text())["_total"] == 15
    with pytest.raises(ValueError):
        _lib.record_mined("eq", "F1", 9)       # shrink forbidden


# ---------- board row + determinism + stamp ----------
def test_board_row_math_and_stamp():
    rets = np.array([0.10, -0.05, 0.02, -0.01] * 20)
    r1 = yield_board.row("t", "f", "c", rets, events_per_day=1.0, capacity_usd=1e5)
    r2 = yield_board.row("t", "f", "c", rets, events_per_day=1.0, capacity_usd=1e5)
    assert r1 == r2
    assert r1["stamp"].startswith("MINING")
    assert abs(r1["gross_pct_day"] - rets.mean()) < 1e-9
    r3 = yield_board.row("t", "f", "c", rets, 1.0, 1e5, net_adjust=0.01, fill_prob=0.5)
    assert r3["net_pct_day"] < r1["net_pct_day"]


def test_seed_determinism():
    a = _lib.seed_from("yield", 1).integers(0, 1 << 30, 5)
    b = _lib.seed_from("yield", 1).integers(0, 1 << 30, 5)
    assert (a == b).all()


# ---------- VRP-001-v2 collision ----------
def test_no_options_cell_matches_vrp_v2():
    v2 = json.loads((REPO / "data/research/preregister/VRP-001-OPTIONS-v2.json").read_text())
    blob = json.dumps(v2).lower()
    # v2 is weekly-Monday cadence, 1-sigma RV20 strikes, 30-45 DTE band
    assert "monday" in blob or "weekly" in blob or "sigma" in blob  # spec sanity
    # our grid is daily cadence with delta-based strikes — assert the axes differ
    from research.yield_frontier import m3_options as m3
    for cell in m3.OP2_GRID:
        assert cell.get("cadence") == "daily"
        assert cell.get("strike_rule") == "moneyness"   # v2 uses 1sigma-RV20


# ---------- look-ahead canary ----------
def test_lookahead_canary_shows_absurd_yield():
    """A peek-next-bar rule must produce blatantly impossible yield through the
    same row machinery — proving the evaluator surfaces leaked information."""
    daily = holdout_guard.load_nq("daily")
    ret = (daily["rth_close"] / daily["rth_open"] - 1).dropna().to_numpy() \
        if {"rth_close", "rth_open"}.issubset(daily.columns) else None
    if ret is None:
        cols = [c for c in daily.columns if "close" in c.lower()]
        pytest.skip(f"nq_daily columns unexpected: {list(daily.columns)[:12]}")
    peek = np.abs(ret)  # 'knew the direction in advance'
    r = yield_board.row("canary", "peek", "cheat", peek, 1.0, 0)
    assert r["net_pct_day"] > 0.005, "canary failed to surface an absurd yield"
