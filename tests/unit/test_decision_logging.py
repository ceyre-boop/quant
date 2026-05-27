"""
tests/unit/test_decision_logging.py

Integration tests for the decision logging pipeline:
  signal → log → close → Oracle reads → reasoning analyzer clusters

Every test uses a temporary directory for log isolation so real paper
trade data is never touched.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import patch


# ─── Minimal ICTSignal stub ───────────────────────────────────────────────────

@dataclass
class _FVGStub:
    formed_at: datetime = None

    def __post_init__(self):
        if self.formed_at is None:
            self.formed_at = datetime.now(timezone.utc)


@dataclass
class _SizingStub:
    risk_pct: float = 0.0075
    risk_dollars: float = 75.0
    stop_loss: float = 1.2700
    tp1: float = 1.2825
    tp2: float = 1.2900


@dataclass
class _SignalStub:
    symbol: str = "GBPUSD"
    direction: str = "LONG"
    timestamp: datetime = None
    score: float = 5.0
    grade: str = "A"
    sizing: _SizingStub = None
    session_status: object = None
    entry_level: float = 1.2750
    component_scores: dict = None
    confirmations: list = None
    missing: list = None
    nearest_fvg: Optional[_FVGStub] = None
    nearest_ob: Optional[object] = None
    sweep: Optional[object] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.sizing is None:
            self.sizing = _SizingStub()
        if self.component_scores is None:
            self.component_scores = {"sweep": 1.0, "fvg": 1.0, "kill_zone": 1.0}
        if self.confirmations is None:
            self.confirmations = ["Liquidity sweep confirmed", "FVG tap at 1.2745", "LONDON session"]
        if self.missing is None:
            self.missing = []
        if self.nearest_fvg is None:
            self.nearest_fvg = _FVGStub()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _patch_log_dir(tmp_path: Path):
    """Context manager: redirect LOG_DIR to a temp directory."""
    import sovereign.intelligence.decision_logger as dl
    return patch.object(dl, "LOG_DIR", tmp_path)


def _read_records(log_dir: Path) -> list[dict]:
    records = []
    for f in sorted(log_dir.glob("decisions_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestICTFullLoop(unittest.TestCase):

    def test_ict_full_loop(self):
        """ICTSignal → log → update_outcome with truncated timestamp → verify backfill."""
        from sovereign.intelligence.decision_logger import log_ict_decision, update_outcome

        signal = _SignalStub()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with _patch_log_dir(tmp_path):
                rec = log_ict_decision(
                    signal=signal,
                    commitment_score=0.85,
                    bars_since_signal=1,
                )

                # Verify entry written
                records = _read_records(tmp_path)
                self.assertEqual(len(records), 1)
                entry = records[0]
                self.assertEqual(entry["pair"], "GBPUSD")
                self.assertEqual(entry["direction"], "LONG")
                self.assertEqual(entry["system"], "ICT")
                self.assertIsNone(entry["outcome"])

                # Simulate forensic engine using date-only timestamp
                date_only = entry["entry_timestamp"][:10]
                found = update_outcome(
                    pair="GBPUSD",
                    entry_timestamp=date_only,
                    outcome="TIMING_FAILURE",
                    r_realized=-0.85,
                    system="ICT",
                )

                self.assertTrue(found, "update_outcome should find the record via fuzzy match")
                records = _read_records(tmp_path)
                self.assertEqual(records[0]["outcome"], "TIMING_FAILURE")
                self.assertAlmostEqual(records[0]["r_realized"], -0.85)


class TestForexFullLoop(unittest.TestCase):

    def test_forex_full_loop(self):
        """log_forex_decision → update_outcome(system=FOREX) → verify backfill."""
        from sovereign.intelligence.decision_logger import log_forex_decision, update_outcome

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with _patch_log_dir(tmp_path):
                rec = log_forex_decision(
                    pair="USDJPY",
                    direction="LONG",
                    entry_level=148.50,
                    stop_loss=147.80,
                    hold_days=20,
                    risk_pct=0.0075,
                    signal_layers=["rate diff z=+2.5", "bull regime"],
                    rate_diff_z=2.5,
                    vix_at_entry=12.0,
                )

                records = _read_records(tmp_path)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["system"], "FOREX")
                self.assertIsNone(records[0]["outcome"])

                # Back-fill using space-separated datetime (forensic engine format)
                ts = rec.entry_timestamp
                space_ts = ts[:10] + " " + ts[11:16]  # "2026-05-26 23:12"
                found = update_outcome(
                    pair="USDJPY",
                    entry_timestamp=space_ts,
                    outcome="WIN",
                    r_realized=1.8,
                    system="FOREX",
                )
                self.assertTrue(found)
                records = _read_records(tmp_path)
                self.assertEqual(records[0]["outcome"], "WIN")


class TestTimestampRobustness(unittest.TestCase):

    def test_all_formats_normalize(self):
        from sovereign.utils.timestamps import normalize_timestamp, timestamps_match

        canonical = "2026-05-26T03:45:00+00:00"
        variants = [
            "2026-05-26T03:45:00+00:00",
            "2026-05-26T03:45:15+00:00",      # different seconds, same hour → should match
            "2026-05-26T03:45:15.123456+00:00",
            "2026-05-26 03:45",
            "2026-05-26 03:45:00",
        ]
        for v in variants:
            self.assertTrue(
                timestamps_match(v, canonical),
                f"Expected {v!r} to match {canonical!r}"
            )

    def test_different_days_no_match(self):
        from sovereign.utils.timestamps import timestamps_match
        self.assertFalse(timestamps_match("2026-05-26", "2026-05-27"))

    def test_different_hours_no_match(self):
        from sovereign.utils.timestamps import timestamps_match
        self.assertFalse(
            timestamps_match("2026-05-26T03:45:00+00:00", "2026-05-26T04:00:00+00:00")
        )


class TestRequiredFieldsPopulated(unittest.TestCase):

    def test_ict_required_fields_not_none(self):
        """Core identifying fields must never be None in an ICT log entry."""
        from sovereign.intelligence.decision_logger import log_ict_decision

        signal = _SignalStub(grade="A")
        with tempfile.TemporaryDirectory() as tmp:
            with _patch_log_dir(Path(tmp)):
                rec = log_ict_decision(
                    signal=signal,
                    commitment_score=0.90,
                    bars_since_signal=2,
                )

        required = ["pair", "direction", "entry_level", "grade",
                    "why_this_trade", "why_this_size", "entry_timestamp",
                    "commitment_score", "system"]
        from dataclasses import asdict
        d = asdict(rec)
        for field in required:
            self.assertIsNotNone(d.get(field), f"Required field {field!r} is None")


class TestLoggerFailureNeverBlocksTrade(unittest.TestCase):

    def test_ict_logging_crash_is_silent(self):
        """If _append raises, the decision logger must not propagate the exception."""
        from sovereign.intelligence import decision_logger as dl

        signal = _SignalStub()
        with tempfile.TemporaryDirectory() as tmp:
            with _patch_log_dir(Path(tmp)):
                with patch.object(dl, "_append", side_effect=IOError("disk full")):
                    # Should NOT raise — _safe_append catches and logs
                    try:
                        result = dl.log_ict_decision(signal=signal, commitment_score=0.85)
                        self.assertIsNotNone(result, "Record should be returned even if write failed")
                    except Exception as e:
                        self.fail(f"log_ict_decision raised {e!r} — logger failure must be silent")


class TestOracleReadsClosedTrades(unittest.TestCase):

    def test_closed_trade_appears_in_oracle_summary(self):
        """A closed decision record must appear in _load_decision_log_summary output."""
        from sovereign.intelligence.decision_logger import log_forex_decision, update_outcome
        from sovereign.oracle.reflect_cycle import _load_decision_log_summary

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with _patch_log_dir(tmp_path):
                rec = log_forex_decision(
                    pair="AUDNZD", direction="SHORT", entry_level=1.0850,
                    stop_loss=1.0900, hold_days=10, risk_pct=0.0075,
                    signal_layers=["rate div", "VIX gate passed"],
                    rate_diff_z=-1.8,
                )
                update_outcome(
                    pair="AUDNZD",
                    entry_timestamp=rec.entry_timestamp[:10],
                    outcome="WIN",
                    r_realized=1.5,
                    system="FOREX",
                )

                # Patch reflect_cycle to read from our temp dir
                import sovereign.oracle.reflect_cycle as rc
                with patch.object(rc, "DECISION_LOG_DIR", tmp_path):
                    summary = _load_decision_log_summary(days=30, max_entries=5)

        self.assertIn("AUDNZD", summary, "Closed trade pair not in Oracle summary")
        self.assertIn("WIN", summary, "Closed trade outcome not in Oracle summary")


class TestReasoningAnalyzerMinSample(unittest.TestCase):

    def test_below_min_sample_returns_empty_chains(self):
        """With < MIN_SAMPLE closed trades, best_chains and worst_chains must be empty."""
        from sovereign.intelligence.decision_logger import log_forex_decision, update_outcome
        from sovereign.forensics.reasoning_analyzer import run_analysis, load_all_closed_records

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with _patch_log_dir(tmp_path):
                # Write 9 closed records (below MIN_SAMPLE=10)
                for i in range(9):
                    rec = log_forex_decision(
                        pair="EURUSD", direction="LONG", entry_level=1.08,
                        stop_loss=1.075, hold_days=10, risk_pct=0.0075,
                        signal_layers=["rate div"], rate_diff_z=1.5,
                    )
                    update_outcome(
                        pair="EURUSD",
                        entry_timestamp=rec.entry_timestamp[:10],
                        outcome="WIN",
                        r_realized=0.9,
                        system="FOREX",
                    )

                import sovereign.forensics.reasoning_analyzer as ra
                analysis_dir = tmp_path / "analysis"
                analysis_dir.mkdir(exist_ok=True)
                with patch.object(ra, "DECISION_LOG_DIR", tmp_path):
                    with patch.object(ra, "ANALYSIS_DIR", analysis_dir):
                        records = ra.load_all_closed_records()
                        self.assertEqual(len(records), 9)
                        report = ra.run_analysis(month="2026_05")

        self.assertEqual(report.n_trades_analyzed, 9)
        self.assertEqual(report.best_chains, [])
        self.assertEqual(report.worst_chains, [])


if __name__ == "__main__":
    unittest.main()
