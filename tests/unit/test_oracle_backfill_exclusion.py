"""
tests/unit/test_oracle_backfill_exclusion.py

RED-1 fix regression tests.

The Oracle's reflection read path (`reflect_cycle._load_decision_log_summary`)
and the rule-based hypothesis generator (`hypothesis_generator._load_reps`) must
reason ONLY over strategy-authored decisions. Records reconstructed from broker
fills (source="fills_backfill") carry forbidden pairs (USD_CAD, AUD_NZD) the
strategy never authored and have no entry reasoning to learn from. Synthetic test
fills (test_fill=True) are likewise not real decisions. Both must be excluded from
the reflection input and the reps population.

Every test writes raw JSONL to a temp decision-log dir so real paper-trade data is
never touched, and asserts directly on the exclusion behavior.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


def _now_iso() -> str:
    """Timestamp inside the Oracle's 7-day window."""
    return datetime.now(timezone.utc).isoformat()


def _record(pair: str, *, source=None, test_fill=None, outcome="LOSS", r=-1.0) -> dict:
    """A minimal but schema-faithful decision record."""
    rec = {
        "pair": pair,
        "system": "FOREX",
        "direction": "SHORT",
        "grade": "A",
        "session": "LONDON",
        "outcome": outcome,
        "r_realized": r,
        "entry_timestamp": _now_iso(),
    }
    if source is not None:
        rec["source"] = source
    if test_fill is not None:
        rec["test_fill"] = test_fill
    return rec


def _write_log(tmp_path: Path, records: list[dict]) -> Path:
    # Filename matches both globs: reflect_cycle's "decisions_*.jsonl" and
    # hypothesis_generator's "decisions_2026_*.jsonl".
    log_file = tmp_path / "decisions_2026_07.jsonl"
    log_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return log_file


# Records shared by the read-path tests. The backfilled USD_CAD record carries a
# REAL closed outcome ("LOSS"), so the pre-existing outcome filter alone would let
# it through — this is exactly the RED-1 leak the source guard closes.
_GENUINE = _record("EURUSD", source=None, test_fill=None, outcome="WIN", r=1.5)
_BACKFILLED_USD_CAD = _record("USD_CAD", source="fills_backfill", test_fill=False,
                              outcome="LOSS", r=-1.0)
_TEST_FILL = _record("GBPJPY", source=None, test_fill=True, outcome="WIN", r=2.0)


class TestReflectCycleExcludesBackfill(unittest.TestCase):
    """reflect_cycle._load_decision_log_summary — the Oracle's read path."""

    def _summary_for(self, records):
        import sovereign.oracle.reflect_cycle as rc
        from sovereign.oracle.reflect_cycle import _load_decision_log_summary
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_log(tmp_path, records)
            with patch.object(rc, "DECISION_LOG_DIR", tmp_path):
                return _load_decision_log_summary(days=7, max_entries=50)

    def test_backfilled_usd_cad_absent_from_reflection(self):
        summary = self._summary_for([_GENUINE, _BACKFILLED_USD_CAD])
        self.assertNotIn(
            "USD_CAD", summary,
            "fills_backfill USD_CAD record leaked into Oracle reflection input",
        )

    def test_genuine_forex_decision_present(self):
        summary = self._summary_for([_GENUINE, _BACKFILLED_USD_CAD])
        self.assertIn(
            "EURUSD", summary,
            "Legitimate strategy-authored decision was wrongly dropped",
        )

    def test_test_fill_records_excluded(self):
        summary = self._summary_for([_GENUINE, _TEST_FILL])
        self.assertNotIn(
            "GBPJPY", summary,
            "test_fill=True record leaked into Oracle reflection input",
        )

    def test_only_genuine_survives_full_mix(self):
        summary = self._summary_for([_GENUINE, _BACKFILLED_USD_CAD, _TEST_FILL])
        self.assertIn("EURUSD", summary)
        self.assertNotIn("USD_CAD", summary)
        self.assertNotIn("GBPJPY", summary)


class TestHypothesisGeneratorExcludesBackfill(unittest.TestCase):
    """hypothesis_generator._load_reps — the reps population miners read."""

    def _reps_for(self, records):
        import sovereign.autonomous.hypothesis_generator as hg
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_log(tmp_path, records)
            glob_pat = str(tmp_path / "decisions_2026_*.jsonl")
            with patch.object(hg, "DECISION_GLOB", glob_pat):
                return hg._load_reps()

    def test_backfill_and_test_fill_excluded_from_reps(self):
        reps = self._reps_for([_GENUINE, _BACKFILLED_USD_CAD, _TEST_FILL])
        pairs = {r.get("pair") for r in reps}
        self.assertIn("EURUSD", pairs, "Genuine decision missing from reps")
        self.assertNotIn("USD_CAD", pairs, "fills_backfill record entered reps population")
        self.assertNotIn("GBPJPY", pairs, "test_fill record entered reps population")

    def test_no_forbidden_source_in_reps(self):
        reps = self._reps_for([_GENUINE, _BACKFILLED_USD_CAD, _TEST_FILL])
        self.assertEqual(len(reps), 1, "Exactly one genuine rep expected")
        for r in reps:
            self.assertNotEqual(r.get("source"), "fills_backfill")
            self.assertIsNot(r.get("test_fill"), True)


if __name__ == "__main__":
    unittest.main()
