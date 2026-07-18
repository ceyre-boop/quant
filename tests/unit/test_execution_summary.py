"""Daily summary CSV contract, skip semantics, and restart idempotency."""
import csv
from datetime import date

import pytest

from execution.harness import (FillRecord, SUMMARY_COLUMNS, is_skip,
                               record_fill, write_daily_summary, _existing_keys)

DAY = date(2026, 7, 16)


def _filled(ticker, net, spread, expected, side="LONG"):
    return FillRecord(ticker=ticker, date=str(DAY), signal_type=side,
                      net_return=net, spread_cost=spread,
                      backtest_expected_return=expected, reason="filled")


def _skipped(ticker, kind="SKIP_HALT"):
    return FillRecord(ticker=ticker, date=str(DAY), signal_type=kind,
                      reason="test")


def test_header_is_exact(tmp_path):
    write_daily_summary(DAY, [], tmp_path)
    with open(tmp_path / "daily_summary.csv") as fh:
        assert next(csv.reader(fh)) == SUMMARY_COLUMNS


def test_no_readiness_column_exists(tmp_path):
    """The summary must never carry a funding/convergence verdict."""
    write_daily_summary(DAY, [], tmp_path)
    header = (tmp_path / "daily_summary.csv").read_text().splitlines()[0].lower()
    for banned in ("ready", "converged", "fund", "verdict", "go_live"):
        assert banned not in header


def test_zero_fill_day_writes_empty_not_zero(tmp_path):
    """A 0.0 would read as a real measurement of zero. It must be blank."""
    row = write_daily_summary(DAY, [_skipped("AAA")], tmp_path)
    assert row["median_net_return"] == ""
    assert row["median_spread_cost"] == ""
    assert row["vs_backtest_delta"] == ""
    assert row["n_filled"] == 0
    assert row["n_skipped"] == 1


def test_medians_and_delta(tmp_path):
    rows = [_filled("AAA", 0.05, 0.01, 0.03), _filled("BBB", 0.07, 0.02, 0.05)]
    row = write_daily_summary(DAY, rows, tmp_path)
    assert row["n_filled"] == 2
    assert float(row["median_net_return"]) == pytest.approx(0.06)
    assert float(row["median_spread_cost"]) == pytest.approx(0.015)
    assert float(row["vs_backtest_delta"]) == pytest.approx(0.02)


def test_skips_excluded_from_medians(tmp_path):
    rows = [_filled("AAA", 0.05, 0.01, 0.03), _skipped("BBB"),
            _skipped("CCC", "SKIP_NO_BORROW")]
    row = write_daily_summary(DAY, rows, tmp_path)
    assert row["n_signals"] == 3
    assert row["n_filled"] == 1
    assert row["n_skipped"] == 2
    assert float(row["median_net_return"]) == pytest.approx(0.05)


def test_rerun_rewrites_row_not_duplicates(tmp_path):
    write_daily_summary(DAY, [_filled("AAA", 0.05, 0.01, 0.03)], tmp_path)
    write_daily_summary(DAY, [_filled("AAA", 0.09, 0.01, 0.03)], tmp_path)
    with open(tmp_path / "daily_summary.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert float(rows[0]["median_net_return"]) == pytest.approx(0.09)


def test_multiple_days_sorted(tmp_path):
    write_daily_summary(date(2026, 7, 17), [_filled("B", 0.02, 0.01, 0.01)], tmp_path)
    write_daily_summary(date(2026, 7, 16), [_filled("A", 0.01, 0.01, 0.01)], tmp_path)
    with open(tmp_path / "daily_summary.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["date"] for r in rows] == ["2026-07-16", "2026-07-17"]


@pytest.mark.parametrize("kind,expected", [
    ("LONG", False), ("SHORT", False),
    ("SKIP_HALT", True), ("SKIP_NO_BORROW", True),
    ("SKIP_NO_QUOTE", True), ("SKIP_NO_DATA", True),
])
def test_is_skip(kind, expected):
    assert is_skip(kind) is expected


def test_idempotency_keys_roundtrip(tmp_path):
    """A restart must not double-record a decision already on disk."""
    record_fill(_filled("AAA", 0.05, 0.01, 0.03), tmp_path)
    record_fill(_skipped("BBB"), tmp_path)
    keys = _existing_keys(tmp_path, DAY)
    assert (str(DAY), "AAA", "LONG") in keys
    assert (str(DAY), "BBB", "SKIP_HALT") in keys
    assert (str(DAY), "CCC", "LONG") not in keys


def test_required_fields_always_present(tmp_path):
    import json
    record_fill(_filled("AAA", 0.05, 0.01, 0.03), tmp_path)
    rec = json.loads((tmp_path / "fill_log.jsonl").read_text().strip())
    for f in ["ticker", "date", "signal_type", "entry_time", "entry_bid",
              "entry_ask", "entry_fill", "exit_fill", "gross_return",
              "spread_cost", "net_return", "backtest_expected_return"]:
        assert f in rec, f"required field {f} missing from fill_log record"
    assert rec["logged_at"]
