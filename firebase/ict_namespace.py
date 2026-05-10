"""
firebase/ict_namespace.py
=========================
Isolated Firebase publisher for the ICT micro-structure engine.

Writes exclusively to the  live_state/ICT_FOREX/<pair>  namespace in the
Realtime Database, keeping ICT state completely separate from the Sovereign
macro engine namespace at  live_state/SOVEREIGN_FOREX/<pair>.

Schema written per pair
-----------------------
live_state/
  ICT_FOREX/
    <GBPUSD>/
      updated_at:      ISO-8601 string
      signal:          "LONG" | "SHORT" | "NONE"
      grade:           "A+" | "A" | "B" | "C" | "VETOED" | "—"
      score:           float
      long_score:      float
      short_score:     float
      confirmations:   list[str]
      missing:         list[str]
      kill_zone:       str | null
      in_kill_zone:    bool
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # ict-engine uses a hyphen so it is not a regular importable package;
    # PairScanResult is referenced only in type hints (never at runtime).
    from ict.pipeline import ICTSignal  # noqa: F401

logger = logging.getLogger(__name__)

_RTDB_ROOT = "live_state/ICT_FOREX"


class ICTFirebasePublisher:
    """
    Thin wrapper around firebase_admin Realtime Database.

    Instantiate once per process (Firebase SDK is a singleton internally).
    Gracefully degrades to no-op logging when Firebase is not configured.
    """

    def __init__(self) -> None:
        self._rtdb = self._connect()

    # ── Connection ────────────────────────────────────────────────────── #

    @staticmethod
    def _connect():
        try:
            import firebase_admin
            from firebase_admin import db

            # Re-use an existing app if the sovereign engine already initialised one.
            try:
                firebase_admin.get_app()
            except ValueError:
                import os
                from pathlib import Path
                from firebase_admin import credentials

                sa_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "")
                rtdb_url = os.getenv("FIREBASE_RTDB_URL", "")
                if sa_path and Path(sa_path).exists():
                    cred = credentials.Certificate(sa_path)
                    firebase_admin.initialize_app(cred, {"databaseURL": rtdb_url})
                else:
                    project_id = os.getenv("FIREBASE_PROJECT_ID", "")
                    firebase_admin.initialize_app(
                        {"projectId": project_id, "databaseURL": rtdb_url}
                    )

            logger.info("ICTFirebasePublisher connected to Realtime Database")
            return db
        except ImportError:
            logger.warning(
                "firebase_admin not installed — ICT Firebase publishing disabled"
            )
            return None
        except Exception as exc:
            logger.warning("ICT Firebase connection failed: %s", exc)
            return None

    # ── Public API ────────────────────────────────────────────────────── #

    def publish_scan_result(self, result: "PairScanResult") -> None:
        """Write one pair's scan result to live_state/ICT_FOREX/<pair>."""
        if self._rtdb is None:
            logger.debug("Firebase unavailable — skipping publish for %s", result.pair)
            return

        summary = result.summary
        best: Optional["ICTSignal"] = result.best_signal

        payload = {
            "updated_at":    datetime.now(tz=timezone.utc).isoformat(),
            "signal":        summary["signal"],
            "grade":         summary["grade"],
            "score":         summary["score"],
            "long_score":    summary["long_score"],
            "short_score":   summary["short_score"],
            "confirmations": summary["confirmations"],
            "missing":       summary["missing"],
            "kill_zone":     (
                getattr(best.session_status, "kill_zone_name", None)
                if best and best.session_status else None
            ),
            "in_kill_zone": (
                bool(getattr(best.session_status, "should_trade", False))
                if best and best.session_status else False
            ),
        }

        try:
            ref = self._rtdb.reference(f"{_RTDB_ROOT}/{result.pair}")
            ref.update(payload)
            logger.debug("ICT Firebase: published %s → %s", result.pair, summary["signal"])
        except Exception as exc:
            logger.warning("ICT Firebase write failed for %s: %s", result.pair, exc)

    def publish_system_status(
        self,
        pairs_scanned: int,
        signals_found: int,
        scan_duration_s: float,
    ) -> None:
        """Write scan-level metadata to live_state/ICT_FOREX/_system."""
        if self._rtdb is None:
            return
        try:
            ref = self._rtdb.reference(f"{_RTDB_ROOT}/_system")
            ref.update({
                "updated_at":       datetime.now(tz=timezone.utc).isoformat(),
                "pairs_scanned":    pairs_scanned,
                "signals_found":    signals_found,
                "scan_duration_s":  round(scan_duration_s, 2),
                "engine":           "ICT_FOREX_v1",
            })
        except Exception as exc:
            logger.warning("ICT Firebase system status write failed: %s", exc)

    def get_pair_state(self, pair: str) -> dict:
        """Read current live state for a pair (dashboard polling helper)."""
        if self._rtdb is None:
            return {}
        try:
            ref = self._rtdb.reference(f"{_RTDB_ROOT}/{pair}")
            return ref.get() or {}
        except Exception as exc:
            logger.warning("ICT Firebase read failed for %s: %s", pair, exc)
            return {}
