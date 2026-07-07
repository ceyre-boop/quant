"""tests/unit/test_positioning_export.py — TICK-007 Step 1: positioning-board dashboard export.

Offline & deterministic: a tmp-path DuckDB (schema created via store.connect, same as production)
seeded with minimal sentiment_board_state rows. No network, never touches the real data/sentiment.db.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from scripts import export_positioning_board as epb
from sovereign.sentiment import store


def _seed(con, rows):
    """Insert minimal sentiment_board_state rows. Any column not given in a row dict defaults to
    NULL (mirrors production reality — most feeders lag/miss on any given day)."""
    for r in rows:
        cols = ", ".join(r.keys())
        placeholders = ", ".join(["?"] * len(r))
        con.execute(f"INSERT INTO sentiment_board_state ({cols}) VALUES ({placeholders})", list(r.values()))


@pytest.fixture
def con(tmp_path):
    """A tmp-file DuckDB with the real schema (store.connect / init_schema creates it), no network."""
    c = store.connect(path=str(tmp_path / "sentiment_test.db"))
    yield c
    c.close()


# 1 — schema: the envelope keys and every requested per-pair field key are present
def test_schema_keys(con):
    _seed(con, [{"date": date(2026, 7, 1), "pair": "EURUSD", "cot_net_pct_1y": 0.42, "vix_regime": "NORMAL"}])
    snap = epb.build_snapshot(con, today=date(2026, 7, 1))
    assert set(snap.keys()) == {"as_of", "built_at", "stale", "pairs"}
    assert "EURUSD" in snap["pairs"]
    assert set(snap["pairs"]["EURUSD"].keys()) == set(epb.FIELDS)


# 2 — staleness flag: fresh board -> False, old board -> True, and the ">3" boundary is exact
def test_staleness_true_and_false(con):
    _seed(con, [{"date": date(2026, 7, 1), "pair": "EURUSD"}])   # Wednesday

    fresh = epb.build_snapshot(con, today=date(2026, 7, 1))
    assert fresh["stale"] is False

    stale = epb.build_snapshot(con, today=date(2026, 7, 10))     # 7 trading days later
    assert stale["stale"] is True

    # exactly 3 trading days later must NOT be stale (">", not ">=")
    boundary = epb.build_snapshot(con, today=date(2026, 7, 6))   # Wed -> Mon = 3 busdays
    assert boundary["stale"] is False


# 3 — null passthrough: unset columns AND a stored NaN DOUBLE (distinct from SQL NULL) both render
#     as JSON null — never fabricated, and never a bare `NaN` token (which is not valid JSON)
def test_null_passthrough(con, tmp_path):
    con.execute("""
        INSERT INTO sentiment_board_state (date, pair, cot_net_pct_1y, vrp_signal, vix_regime)
        VALUES (DATE '2026-07-01', 'GBPUSD', 0.55, 'NaN'::DOUBLE, NULL)
    """)
    snap = epb.build_snapshot(con, today=date(2026, 7, 1))
    gbp = snap["pairs"]["GBPUSD"]
    assert gbp["cot_net_pct_1y"] == pytest.approx(0.55)
    assert gbp["vrp_signal"] is None    # NaN -> None, not the literal float
    assert gbp["vix_regime"] is None    # explicit NULL -> None
    assert gbp["rr25"] is None          # never-set column -> None

    out = tmp_path / "out.json"
    out.write_text(json.dumps(snap, indent=2))    # must not raise, must not contain a bare NaN token
    text = out.read_text()
    assert "NaN" not in text
    assert json.loads(text)["pairs"]["GBPUSD"]["vrp_signal"] is None


# 4 — each pair's "latest row" is independent; as_of is the GLOBAL latest board date
def test_latest_row_is_independent_per_pair(con):
    _seed(con, [
        {"date": date(2026, 6, 20), "pair": "EURUSD", "cot_net_z": 1.0},
        {"date": date(2026, 7, 1), "pair": "EURUSD", "cot_net_z": 2.0},
        {"date": date(2026, 6, 25), "pair": "AUDUSD", "cot_net_z": -1.0},
    ])
    snap = epb.build_snapshot(con, today=date(2026, 7, 1))
    assert snap["pairs"]["EURUSD"]["cot_net_z"] == pytest.approx(2.0)    # not the older 1.0 row
    assert snap["pairs"]["AUDUSD"]["cot_net_z"] == pytest.approx(-1.0)
    assert snap["as_of"] == "2026-07-01"


# 5 — empty board: no rows at all -> no pairs, no as_of, definitionally stale, never crashes
def test_empty_board(con):
    snap = epb.build_snapshot(con, today=date(2026, 7, 1))
    assert snap["pairs"] == {}
    assert snap["as_of"] is None
    assert snap["stale"] is True


# 6 — output path is monkeypatchable: export() falls back to the (patched) module-level OUTPUT_PATH,
#     so tests (and any future caller) never risk touching real data/agent/positioning_board.json
def test_output_path_monkeypatchable(con, tmp_path, monkeypatch):
    _seed(con, [{"date": date(2026, 7, 1), "pair": "EURUSD", "gdelt_tone": 0.1}])
    target = tmp_path / "nested" / "positioning_board.json"
    monkeypatch.setattr(epb, "OUTPUT_PATH", target)

    snapshot = epb.export(con=con)   # no output_path kwarg -> must use the patched OUTPUT_PATH

    assert target.exists()
    on_disk = json.loads(target.read_text())
    assert on_disk == snapshot
    assert on_disk["pairs"]["EURUSD"]["gdelt_tone"] == pytest.approx(0.1)


# 7 — export() also accepts an explicit output_path override
def test_export_explicit_output_path(con, tmp_path):
    _seed(con, [{"date": date(2026, 7, 1), "pair": "EURUSD"}])
    target = tmp_path / "explicit.json"
    epb.export(con=con, output_path=target)
    assert target.exists()
    assert json.loads(target.read_text())["as_of"] == "2026-07-01"


# 8 — standalone path (con=None) opens its own read-only connection against store.DB_PATH; verify
#     the wiring against a monkeypatched tmp DB, never the real data/sentiment.db
def test_export_standalone_uses_read_only_connection(tmp_path, monkeypatch):
    db_path = tmp_path / "standalone.db"
    seed_con = store.connect(path=str(db_path))
    seed_con.execute("INSERT INTO sentiment_board_state (date, pair) VALUES (DATE '2026-07-01', 'EURUSD')")
    seed_con.close()

    monkeypatch.setattr(store, "DB_PATH", db_path)
    out_target = tmp_path / "standalone_out.json"
    monkeypatch.setattr(epb, "OUTPUT_PATH", out_target)

    snapshot = epb.export()   # con=None -> opens its own read_only connection against (patched) DB_PATH
    assert snapshot["as_of"] == "2026-07-01"
    assert out_target.exists()
