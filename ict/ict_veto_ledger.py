"""
ict/ict_veto_ledger.py
======================
ICT Veto Ledger — the hidden dataset.

Every signal that the ICT pipeline generates but does NOT execute is a labeled
data point that would otherwise be wasted.  This module records those vetoed
signals with their full feature context so that:

  1. Retroactive outcome labeling — after N bars have elapsed, we can look up
     what would have happened (did price reach TP1/TP2, or stop out?).  A veto
     that would have hit TP2 is a *false negative*.  A veto on a trade that
     would have stopped out is a *true positive*.

  2. Meta-labeling training data — the labeled veto records (combined with
     executed trade outcomes) are the training set for the second-layer
     XGBoost classifier.

  3. Veto-quality audit — which gates are generating the most false negatives?
     This drives targeted experiment proposals from the LLM lab director.

Storage: ``data/ledger/ict_veto_ledger_YYYY_MM.jsonl`` (monthly shards).

Record schema (all fields):
    timestamp:         ISO-8601 UTC string at veto time
    pair:              e.g. "GBPUSD"
    session:           "NY_PM" | "London" | "OFF-HOURS"
    signal:            "LONG" | "SHORT"
    grade:             "B" | "C" | "VETO"
    score:             pipeline score (float)
    veto_reason:       top-level veto reason string from the pipeline
    veto_stage:        "gate" | "grade" | "session" | "bias" | "memory" | "heatmap"
    entry_level:       hypothetical entry price (float | null)
    stop:              hypothetical stop-loss price (float | null)
    tp1:               hypothetical TP1 price (float | null)
    tp2:               hypothetical TP2 price (float | null)
    intended_direction: pipeline's evaluated LONG/SHORT, captured even when the
                       top-level ``signal`` reads "VETO" — the field retrospective
                       labeling keys off (str | null)
    intended_entry:    intended entry/limit price the gate prevented (float | null)
    structural_stop:   intended structural stop the gate prevented (float | null)
    adr_pct:           ADR consumed at signal time (float)
    risk_pct:          risk fraction of account (float)
    confirmations:     list of confirmed ICT factors
    missing:           list of missing/failed ICT factors
    component_scores:  dict of per-stage scores
    outcome:           null until retroactively labeled ("TP1"|"TP2"|"SL"|"OPEN")
    outcome_r:         R-multiple if outcome is known (float | null)
    outcome_labeled_at: ISO-8601 UTC string when outcome was filled in (null until labeled)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_LEDGER_ROOT = Path("data/ledger")


class ICTVetoLedger:
    """Records every ICT-vetoed signal for retroactive labeling and ML training.

    Parameters
    ----------
    ledger_root:
        Directory for JSONL shards.  Defaults to ``data/ledger/``.
    """

    def __init__(self, ledger_root: Optional[Path] = None) -> None:
        self._root = Path(ledger_root) if ledger_root else _DEFAULT_LEDGER_ROOT
        self._root.mkdir(parents=True, exist_ok=True)

    # ── Write path ──────────────────────────────────────────────────────────── #

    def record_veto(
        self,
        *,
        pair: str,
        session: str,
        signal: str,
        grade: str,
        score: float,
        veto_reason: str,
        veto_stage: str,
        entry_level: Optional[float] = None,
        stop: Optional[float] = None,
        tp1: Optional[float] = None,
        tp2: Optional[float] = None,
        intended_direction: Optional[str] = None,
        intended_entry: Optional[float] = None,
        structural_stop: Optional[float] = None,
        adr_pct: float = 0.0,
        risk_pct: float = 0.0,
        confirmations: Optional[List[str]] = None,
        missing: Optional[List[str]] = None,
        component_scores: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Append one veto event to the current monthly shard."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        record: Dict[str, Any] = {
            "timestamp":            ts,
            "pair":                 pair,
            "session":              session,
            "signal":               signal,
            "grade":                grade,
            "score":                round(float(score), 3),
            "veto_reason":          veto_reason,
            "veto_stage":           veto_stage,
            "entry_level":          entry_level,
            "stop":                 stop,
            "tp1":                  tp1,
            "tp2":                  tp2,
            # Retrospective-labeling fields — the trade the gate prevented.
            # intended_direction is the pipeline's evaluated LONG/SHORT (present
            # even when the top-level `signal` reads "VETO"); intended_entry and
            # structural_stop are the limit/stop that would have defined the
            # trade, or None when the veto fired before they were computed
            # (e.g. ADR / displacement gates that veto pre-entry).
            "intended_direction":   intended_direction,
            "intended_entry":       intended_entry,
            "structural_stop":      structural_stop,
            "adr_pct":              round(float(adr_pct), 4),
            "risk_pct":             round(float(risk_pct), 4),
            "confirmations":        confirmations or [],
            "missing":              missing or [],
            "component_scores":     component_scores or {},
            # Outcome fields — filled in later via label_outcome()
            "outcome":              None,
            "outcome_r":            None,
            "outcome_labeled_at":   None,
        }
        shard = self._shard_path(ts)
        try:
            with shard.open("a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            logger.error("ICTVetoLedger: failed to write veto record: %s", exc)

    # ── Read / label path ────────────────────────────────────────────────────── #

    def label_outcome(
        self,
        pair: str,
        timestamp: str,
        outcome: str,
        outcome_r: Optional[float] = None,
    ) -> bool:
        """Retroactively label the outcome of a vetoed signal.

        Searches the monthly shard containing ``timestamp`` for the matching
        ``(pair, timestamp)`` record, updates it in-place, and rewrites the
        shard.

        Parameters
        ----------
        pair:       Pair symbol, e.g. "GBPUSD".
        timestamp:  The ``timestamp`` field from the original ``record_veto`` call.
        outcome:    One of "TP1", "TP2", "SL", "OPEN" (still open when checked).
        outcome_r:  R-multiple achieved (positive for TP, negative for SL).

        Returns ``True`` if a matching record was found and updated.
        """
        shard = self._shard_path(timestamp)
        if not shard.exists():
            return False

        records = self._read_shard(shard)
        updated = False
        for rec in records:
            if rec.get("pair") == pair and rec.get("timestamp") == timestamp:
                rec["outcome"] = outcome
                rec["outcome_r"] = outcome_r
                rec["outcome_labeled_at"] = datetime.now(timezone.utc).isoformat()
                updated = True
                break

        if updated:
            self._write_shard(shard, records)
        return updated

    def get_unlabeled(self, pair: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all veto records that have not yet been outcome-labeled."""
        result = []
        for shard in sorted(self._root.glob("ict_veto_ledger_*.jsonl")):
            for rec in self._read_shard(shard):
                if rec.get("outcome") is None:
                    if pair is None or rec.get("pair") == pair:
                        result.append(rec)
        return result

    def get_labeled(self, pair: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all outcome-labeled veto records (training-ready)."""
        result = []
        for shard in sorted(self._root.glob("ict_veto_ledger_*.jsonl")):
            for rec in self._read_shard(shard):
                if rec.get("outcome") is not None:
                    if pair is None or rec.get("pair") == pair:
                        result.append(rec)
        return result

    def false_negative_rate(
        self,
        veto_stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fraction of labeled vetoes that would have been profitable (outcome != SL).

        A high false-negative rate on a specific ``veto_stage`` is the trigger
        for the LLM lab director to propose a targeted experiment relaxing that
        gate.

        Returns a dict with keys: ``total``, ``profitable``, ``false_negative_rate``,
        ``by_stage`` (breakdown per veto_stage).
        """
        labeled = self.get_labeled()
        if not labeled:
            return {"total": 0, "profitable": 0, "false_negative_rate": 0.0, "by_stage": {}}

        by_stage: Dict[str, Dict[str, int]] = {}
        total = 0
        profitable = 0
        for rec in labeled:
            stage = rec.get("veto_stage", "unknown")
            outcome = rec.get("outcome", "")
            if veto_stage and stage != veto_stage:
                continue
            total += 1
            is_profitable = outcome in ("TP1", "TP2")
            if is_profitable:
                profitable += 1
            entry = by_stage.setdefault(stage, {"total": 0, "profitable": 0})
            entry["total"] += 1
            if is_profitable:
                entry["profitable"] += 1

        fnr = round(profitable / total, 3) if total else 0.0
        by_stage_rates = {
            s: {
                **v,
                "false_negative_rate": round(v["profitable"] / v["total"], 3) if v["total"] else 0.0,
            }
            for s, v in by_stage.items()
        }
        return {
            "total":               total,
            "profitable":          profitable,
            "false_negative_rate": fnr,
            "by_stage":            by_stage_rates,
        }

    # ── Internal helpers ─────────────────────────────────────────────────────── #

    def _shard_path(self, timestamp: str) -> Path:
        """Monthly shard, e.g. ``ict_veto_ledger_2026_05.jsonl``."""
        try:
            month = timestamp[:7].replace("-", "_")
        except (TypeError, IndexError):
            month = datetime.now(timezone.utc).strftime("%Y_%m")
        return self._root / f"ict_veto_ledger_{month}.jsonl"

    @staticmethod
    def _read_shard(path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        try:
            with path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass
        return records

    @staticmethod
    def _write_shard(path: Path, records: List[Dict[str, Any]]) -> None:
        with path.open("w") as fh:
            for rec in records:
                fh.write(json.dumps(rec, default=str) + "\n")
