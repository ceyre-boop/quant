#!/usr/bin/env python3
"""scripts/export_positioning_board.py — dashboard-facing snapshot of the sentiment
positioning board (TICK-007 Step 1).

Reads the latest per-pair row from `sentiment_board_state` and writes a small JSON
summary to `data/agent/positioning_board.json` for the (Step 2, separate session)
dashboard panel.

DISPLAY-ONLY (TICK-007 constraint / RISK_CONSTITUTION Article 6): this file feeds no
live gate, readiness score, or decision chain. Missing data is preserved as `null` —
never fabricated or interpolated. Board signals reach live decisions only behind a
CONFIRMED hypothesis-ledger verdict plus a logged param change, same as everywhere else.

    python3 scripts/export_positioning_board.py     # standalone (opens its own read-only connection)

Also invoked as a guarded tail-call at the end of scripts/update_sentiment.py::main(), reusing that
run's already-open connection so the export refreshes with every board rebuild without ever taking a
second lock on the same DuckDB file.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.sentiment import store

# Where the snapshot lands. Module-level so tests can monkeypatch it instead of touching real data.
OUTPUT_PATH = ROOT / "data" / "agent" / "positioning_board.json"

# The board columns the Step 2 dashboard panel needs, in display order. Every one of these already
# exists on sentiment_board_state (see sovereign/sentiment/store.py::SCHEMA) — this is a read-only
# projection, not a new computation.
FIELDS = [
    "cot_net_pct_1y", "cot_net_z", "cot_flush_1w", "tff_lev_net_pct",
    "vrp_signal", "vrp_pct", "rr25", "bf25", "atm_term_slope",
    "econ_surprise_z", "gdelt_tone", "vix_regime",
]

STALE_TRADING_DAYS = 3  # stale if the board's latest date is more than this many trading days old


def _clean(v):
    """SQL NULL already comes back as None from duckdb's fetchone(); this only catches a genuine
    stored NaN DOUBLE (distinct from NULL) and folds it to None too — json.dumps would otherwise
    emit a bare `NaN` token, which is not valid JSON and breaks strict parsers (e.g. JS JSON.parse).
    Never fabricates a value; only ever narrows "not a real number" down to "missing"."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _latest_row_for_pair(con, pair: str) -> tuple[date, dict] | None:
    """The most recent sentiment_board_state row for one pair, as (date, {field: value})."""
    cols = ", ".join(FIELDS)
    row = con.execute(
        f"SELECT date, {cols} FROM sentiment_board_state WHERE pair = ? ORDER BY date DESC LIMIT 1",
        [pair],
    ).fetchone()
    if row is None:
        return None
    row_date, *values = row
    return row_date, {field: _clean(v) for field, v in zip(FIELDS, values)}


def build_snapshot(con, today: date | None = None) -> dict:
    """Read the latest per-pair row from sentiment_board_state and shape the export dict.

    Pure with respect to disk: takes an already-open connection, returns a dict. `today` is
    injectable for deterministic staleness testing; defaults to the real current UTC date.
    """
    today = today or datetime.now(timezone.utc).date()

    pair_names = [r[0] for r in con.execute(
        "SELECT DISTINCT pair FROM sentiment_board_state ORDER BY pair").fetchall()]

    pairs_out: dict[str, dict] = {}
    latest_dates: list[date] = []
    for pair in pair_names:
        found = _latest_row_for_pair(con, pair)
        if found is None:
            continue
        row_date, fields = found
        latest_dates.append(row_date)
        pairs_out[pair] = fields

    if latest_dates:
        as_of_date = max(latest_dates)
        as_of_str = as_of_date.isoformat()
        stale = int(np.busday_count(as_of_date, today)) > STALE_TRADING_DAYS
    else:
        # No board data at all — nothing to report, and by definition not fresh.
        as_of_str = None
        stale = True

    return {
        "as_of": as_of_str,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "stale": stale,
        "pairs": pairs_out,
    }


def export(con=None, output_path: Path | str | None = None) -> dict:
    """Build the snapshot and write it to `output_path` (default OUTPUT_PATH).

    When `con` is omitted (the standalone-CLI path), opens — and closes — its own read-only
    connection. When called from update_sentiment.py, pass that run's already-open (read-write)
    `con` so we never open a second connection to the same DuckDB file.
    """
    own = con is None
    con = con or store.connect(read_only=True)
    try:
        snapshot = build_snapshot(con)
    finally:
        if own:
            con.close()

    path = Path(output_path) if output_path is not None else OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2))
    return snapshot


def main() -> dict:
    snapshot = export()
    print(f"[export_positioning_board] wrote {OUTPUT_PATH}  "
          f"as_of={snapshot['as_of']}  stale={snapshot['stale']}  pairs={len(snapshot['pairs'])}")
    return snapshot


if __name__ == "__main__":
    main()
