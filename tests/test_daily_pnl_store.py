"""TICK-044: reconstruct_daily_state reads what the staged patch will start writing.

Dormant module — not wired into any live path yet (see execution/daily_pnl_store.py
docstring). These tests just prove the reconstruction math is correct in isolation
so the July-28 apply has a green baseline to build on.
"""
from datetime import date
from pathlib import Path

from execution.daily_pnl_store import reconstruct_daily_state


def _write_fills(out_dir: Path, rows: list[dict]) -> None:
    fp = out_dir / "fill_log.jsonl"
    with open(fp, "w") as fh:
        for r in rows:
            fh.write(__import__("json").dumps(r) + "\n")


def test_no_file_returns_flat_state(tmp_path):
    st = reconstruct_daily_state(date(2026, 7, 24), tmp_path)
    assert st.daily_pnl_frac == 0.0
    assert st.consecutive_losses == 0
    assert st.n_fills_considered == 0


def test_sums_only_todays_non_skip_fills_with_effective_risk(tmp_path):
    day = date(2026, 7, 24)
    _write_fills(tmp_path, [
        {"date": "2026-07-24", "signal_type": "LONG", "net_return": -0.02,
         "effective_risk_frac": 0.005},
        {"date": "2026-07-24", "signal_type": "SKIP_RISK", "net_return": -0.05,
         "effective_risk_frac": 0.005},          # skipped fill — must not count
        {"date": "2026-07-23", "signal_type": "LONG", "net_return": -1.0,
         "effective_risk_frac": 0.005},          # wrong day — must not count
        {"date": "2026-07-24", "signal_type": "SHORT", "net_return": -0.01,
         "effective_risk_frac": 0.0075},
    ])
    st = reconstruct_daily_state(day, tmp_path)
    expected = (-0.02 * 0.005) + (-0.01 * 0.0075)
    assert abs(st.daily_pnl_frac - expected) < 1e-12
    assert st.consecutive_losses == 2
    assert st.n_fills_considered == 2


def test_rows_missing_effective_risk_frac_are_skipped_not_guessed(tmp_path):
    day = date(2026, 7, 24)
    _write_fills(tmp_path, [
        {"date": "2026-07-24", "signal_type": "LONG", "net_return": -0.02},
    ])
    st = reconstruct_daily_state(day, tmp_path)
    assert st.n_fills_considered == 0
    assert st.daily_pnl_frac == 0.0


def test_win_resets_loss_streak(tmp_path):
    day = date(2026, 7, 24)
    _write_fills(tmp_path, [
        {"date": "2026-07-24", "signal_type": "LONG", "net_return": -0.02,
         "effective_risk_frac": 0.005},
        {"date": "2026-07-24", "signal_type": "LONG", "net_return": 0.03,
         "effective_risk_frac": 0.005},
    ])
    st = reconstruct_daily_state(day, tmp_path)
    assert st.consecutive_losses == 0
