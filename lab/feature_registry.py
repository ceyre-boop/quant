"""
Feature Registry — append-only ledger of every feature ever tested.

Every feature must EARN ITS EXISTENCE.

Design principles:
- Append-only: records are never deleted. Graveyarded features stay in the log
  so future researchers see the decision trail.
- Strict promotion gate: a feature cannot enter LIVE status without passing
  three criteria (IC_OOS > 0.15, positive walk-forward marginal contribution,
  no holdout degradation).
- 90-day re-validation requirement: LIVE features that haven't been re-measured
  in 90 days are flagged as STALE by the audit script.

Verdicts:
  TESTING    Feature is being evaluated. No deployment.
  LIVE       Feature cleared all promotion gates and is active in the system.
  GRAVEYARD  Feature failed one or more gates, or was removed. Never re-deploy
             without new evidence.

Usage
-----
from lab.feature_registry import FeatureRegistry, Verdict

reg = FeatureRegistry()

# Add a new feature under evaluation:
reg.add("pd_alignment", hypothesis_id="HYP-024", verdict=Verdict.TESTING,
        standalone_expectancy=-0.12, sample_size=156,
        note="anti-edge confirmed: pd>0 → 20% WR, pd=0 → 35% WR")

# Graveyard an existing feature:
reg.graveyard("pd_alignment",
              graveyard_reason="Anti-edge: inclusion drops WR from 35% to 20%.")

# Promote to LIVE (after passing all gates):
reg.promote("rate_divergence_60d", ic_oos=0.17, marginal_contribution=0.08,
            holdout_degradation=False)

# Query:
live   = reg.get_live()
stale  = reg.get_stale(days=90)
buried = reg.get_graveyard()
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_ROOT    = Path(__file__).resolve().parents[1]
_LEDGER  = _ROOT / "data" / "lab" / "feature_registry.jsonl"

# Promotion gate thresholds — do NOT lower without human sign-off.
IC_OOS_THRESHOLD          = 0.15   # minimum out-of-sample information coefficient
MARGINAL_CONTRIB_MIN      = 0.0    # must be strictly positive
STALE_DAYS                = 90     # LIVE features must be re-validated within this window


class Verdict(str, Enum):
    TESTING   = "TESTING"
    LIVE      = "LIVE"
    GRAVEYARD = "GRAVEYARD"


class PromotionGateError(ValueError):
    """Raised when a feature fails one or more promotion gates."""


class FeatureRegistry:
    """
    Append-only feature ledger.

    Each operation (add, update, graveyard, promote) appends a new record.
    The current state of a feature is the LAST record for that feature_name.
    """

    def __init__(self, ledger_path: Optional[Path] = None) -> None:
        self._path = Path(ledger_path) if ledger_path else _LEDGER
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Internal I/O ───────────────────────────────────────────────── #

    def _append(self, record: dict) -> None:
        record.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        with self._path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _load_all(self) -> List[dict]:
        if not self._path.exists():
            return []
        records = []
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _latest_per_feature(self) -> Dict[str, dict]:
        """Return the most-recent record per feature_name."""
        all_records = self._load_all()
        latest: Dict[str, dict] = {}
        for rec in all_records:
            name = rec.get("feature_name", "")
            if name:
                latest[name] = rec
        return latest

    # ── Write API ──────────────────────────────────────────────────── #

    def add(
        self,
        feature_name: str,
        hypothesis_id: str = "",
        verdict: Verdict = Verdict.TESTING,
        standalone_expectancy: Optional[float] = None,
        marginal_contribution: Optional[float] = None,
        ic_oos: Optional[float] = None,
        holdout_degradation: Optional[bool] = None,
        sample_size: int = 0,
        system: str = "",
        note: str = "",
        graveyard_reason: str = "",
    ) -> dict:
        """
        Add or update a feature record.  Appends — never overwrites.

        Args:
            feature_name:          Unique identifier (e.g. "pd_alignment").
            hypothesis_id:         Linked hypothesis (e.g. "HYP-024").
            verdict:               TESTING | LIVE | GRAVEYARD
            standalone_expectancy: Expectancy without other features (R units).
            marginal_contribution: Improvement over base model (R or Sharpe delta).
            ic_oos:                Information coefficient measured out-of-sample.
            holdout_degradation:   True if the feature degrades on holdout window.
            sample_size:           Trade count used in measurement.
            system:                Which system (ICT / FOREX / EQUITY / ALL).
            note:                  Free-text rationale.
            graveyard_reason:      Required if verdict=GRAVEYARD.

        Returns:
            The record dict that was appended.
        """
        record = {
            "feature_name":          feature_name,
            "hypothesis_id":         hypothesis_id,
            "test_date":             datetime.now(timezone.utc).date().isoformat(),
            "verdict":               Verdict(verdict).value,
            "standalone_expectancy": standalone_expectancy,
            "marginal_contribution": marginal_contribution,
            "ic_oos":                ic_oos,
            "holdout_degradation":   holdout_degradation,
            "sample_size":           sample_size,
            "system":                system,
            "note":                  note,
            "graveyard_reason":      graveyard_reason,
            "last_validated_date":   datetime.now(timezone.utc).date().isoformat(),
        }
        self._append(record)
        logger.info(f"[FeatureRegistry] Added {feature_name!r} → {verdict}")
        return record

    def graveyard(
        self,
        feature_name: str,
        graveyard_reason: str,
        note: str = "",
    ) -> dict:
        """
        Mark a feature as GRAVEYARD.  Appends a new record; prior records remain.

        Args:
            feature_name:     Feature to bury.
            graveyard_reason: Must explain WHY the feature was removed.
            note:             Optional additional context.
        """
        existing = self._latest_per_feature().get(feature_name, {})
        record = {
            **existing,
            "feature_name":    feature_name,
            "verdict":         Verdict.GRAVEYARD.value,
            "graveyard_reason": graveyard_reason,
            "note":            note or existing.get("note", ""),
        }
        self._append(record)
        logger.info(f"[FeatureRegistry] Graveyarded {feature_name!r}: {graveyard_reason}")
        return record

    def promote(
        self,
        feature_name: str,
        ic_oos: float,
        marginal_contribution: float,
        holdout_degradation: bool,
        note: str = "",
    ) -> dict:
        """
        Promote a feature to LIVE after checking all three gates.

        Raises PromotionGateError if any gate fails.

        Gate 1: ic_oos > IC_OOS_THRESHOLD (0.15)
        Gate 2: marginal_contribution > 0
        Gate 3: holdout_degradation is False
        """
        failures = []
        if ic_oos <= IC_OOS_THRESHOLD:
            failures.append(
                f"IC_OOS {ic_oos:.3f} ≤ threshold {IC_OOS_THRESHOLD}"
            )
        if marginal_contribution <= MARGINAL_CONTRIB_MIN:
            failures.append(
                f"marginal_contribution {marginal_contribution:.4f} ≤ 0"
            )
        if holdout_degradation:
            failures.append("holdout_degradation=True — feature hurts on held-out data")

        if failures:
            msg = f"Feature {feature_name!r} failed promotion gates: " + "; ".join(failures)
            logger.warning(f"[FeatureRegistry] {msg}")
            raise PromotionGateError(msg)

        existing = self._latest_per_feature().get(feature_name, {})
        record = {
            **existing,
            "feature_name":          feature_name,
            "verdict":               Verdict.LIVE.value,
            "ic_oos":                ic_oos,
            "marginal_contribution": marginal_contribution,
            "holdout_degradation":   holdout_degradation,
            "note":                  note or existing.get("note", ""),
            "last_validated_date":   datetime.now(timezone.utc).date().isoformat(),
        }
        self._append(record)
        logger.info(f"[FeatureRegistry] Promoted {feature_name!r} → LIVE")
        return record

    def update_validation(
        self,
        feature_name: str,
        ic_oos: float,
        sample_size: int = 0,
        note: str = "",
    ) -> dict:
        """
        Record a fresh re-validation of a LIVE feature (resets the 90-day clock).
        """
        existing = self._latest_per_feature().get(feature_name, {})
        record = {
            **existing,
            "feature_name":        feature_name,
            "ic_oos":              ic_oos,
            "sample_size":         sample_size or existing.get("sample_size", 0),
            "note":                note or existing.get("note", ""),
            "last_validated_date": datetime.now(timezone.utc).date().isoformat(),
        }
        self._append(record)
        logger.info(f"[FeatureRegistry] Re-validated {feature_name!r} ic_oos={ic_oos:.3f}")
        return record

    # ── Query API ──────────────────────────────────────────────────── #

    def get_all(self) -> Dict[str, dict]:
        """Latest record per feature (all verdicts)."""
        return self._latest_per_feature()

    def get_live(self) -> Dict[str, dict]:
        """All LIVE features."""
        return {k: v for k, v in self._latest_per_feature().items()
                if v.get("verdict") == Verdict.LIVE.value}

    def get_graveyard(self) -> Dict[str, dict]:
        """All GRAVEYARD features."""
        return {k: v for k, v in self._latest_per_feature().items()
                if v.get("verdict") == Verdict.GRAVEYARD.value}

    def get_testing(self) -> Dict[str, dict]:
        """All TESTING features."""
        return {k: v for k, v in self._latest_per_feature().items()
                if v.get("verdict") == Verdict.TESTING.value}

    def get_stale(self, days: int = STALE_DAYS) -> Dict[str, dict]:
        """
        LIVE features that have NOT been re-validated within `days` days.
        These should be re-measured or graveyarded.
        """
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
        stale = {}
        for name, rec in self.get_live().items():
            last = rec.get("last_validated_date", "")
            if not last:
                stale[name] = rec
                continue
            try:
                last_date = datetime.fromisoformat(last).date()
            except ValueError:
                stale[name] = rec
                continue
            if last_date < cutoff:
                stale[name] = rec
        return stale

    def history(self, feature_name: str) -> List[dict]:
        """All records for a specific feature, oldest first."""
        return [r for r in self._load_all() if r.get("feature_name") == feature_name]

    # ── Seed ───────────────────────────────────────────────────────── #

    def seed_pd_alignment(self) -> None:
        """
        Seed pd_alignment as a GRAVEYARD entry.

        Result from HYP-024 (2026-05-19): pd_alignment>0 = 20% WR,
        pd_alignment=0 = 35% WR.  Anti-edge: inclusion degrades performance.
        Weight was set to 0.0 in config/ict_params.yml on 2026-05-19.
        """
        if "pd_alignment" in self._latest_per_feature():
            return  # already seeded

        self.add(
            feature_name="pd_alignment",
            hypothesis_id="HYP-024",
            verdict=Verdict.TESTING,
            standalone_expectancy=-0.15,
            marginal_contribution=-0.15,
            ic_oos=None,
            holdout_degradation=True,
            sample_size=156,
            system="ICT",
            note=(
                "Narrative logic: trades should align with premium/discount zones. "
                "Statistical reality: pd_alignment>0 → WR 20%, pd_alignment=0 → WR 35%. "
                "Removing narrative coherence improved performance. "
                "Lesson: statistical utility > narrative coherence."
            ),
        )
        self.graveyard(
            "pd_alignment",
            graveyard_reason=(
                "Anti-edge confirmed (HYP-024, 2026-05-19). "
                "pd_alignment weight set 1.5 → 0.0 in config/ict_params.yml. "
                "WR improved from 20% to 35% on removal. "
                "Do NOT re-introduce without new evidence from N≥200 trades."
            ),
            note="See CLAUDE.md: HYP-024 DEPLOYED 2026-05-19",
        )


# ── Module-level singleton ─────────────────────────────────────────────── #

_registry: Optional[FeatureRegistry] = None


def get_registry() -> FeatureRegistry:
    global _registry
    if _registry is None:
        _registry = FeatureRegistry()
    return _registry
