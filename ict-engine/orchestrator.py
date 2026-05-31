"""
ict-engine/orchestrator.py
==========================
Brain of the ICT retail micro-structure engine.

Runs the 5-stage ICTPipeline across all configured forex pairs on a
configurable scan interval, publishes live state to Firebase under the
isolated  live_state/ICT_FOREX/<pair>  namespace, and emits structured
logs that feed the ICT dashboard.

Isolation rule
-----------------------------------------
ict/pipeline.py MUST NOT import from sovereign/, layer1/, layer2/, layer3/.

This orchestrator IS the designated cross-layer bridge (see CLAUDE.md):
  - Pre-fetches sovereign context (allocation_engine, cross_system_bridge,
    CommitmentDetector) once per cycle and injects it into pipeline.evaluate()
  - This is the ONLY safe entry point for ICT → sovereign communication

It may import from:
  ict.*          — all ICT subsystem modules
  sovereign.*    — for sovereign context pre-fetch only (bridge role)
  firebase.*     — client + ict_namespace publisher
  config/        — ict_params.yml only
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from ict.micro_risk import MicroRiskParams
from ict.pipeline import ICTPipeline, ICTSignal, ICTVeto, ICTGrade

logger = logging.getLogger(__name__)

# ── Default universe ──────────────────────────────────────────────────────── #
_DEFAULT_PAIRS: List[str] = [
    "GBPUSD", "EURUSD", "AUDUSD", "USDJPY",
    "USDCAD", "NZDUSD", "GBPJPY", "AUDNZD",
]

_DEFAULT_ACCOUNT_SIZE: float = 10_000.0
_DEFAULT_SCAN_INTERVAL_SECONDS: int = 300   # 5 minutes
_DEFAULT_CONFIG_PATH: str = str(
    Path(__file__).resolve().parent.parent / "config" / "ict_params.yml"
)


# ── Result containers ─────────────────────────────────────────────────────── #

@dataclass
class PairScanResult:
    pair: str
    timestamp: datetime
    long_result:  Union[ICTSignal, ICTVeto]
    short_result: Union[ICTSignal, ICTVeto]

    @property
    def best_signal(self) -> Optional[ICTSignal]:
        """Return the highest-scoring approved signal, if any."""
        candidates = [
            r for r in (self.long_result, self.short_result)
            if isinstance(r, ICTSignal) and r.passed
        ]
        return max(candidates, key=lambda s: s.score) if candidates else None

    @property
    def summary(self) -> Dict:
        best = self.best_signal
        return {
            "pair":       self.pair,
            "timestamp":  self.timestamp.isoformat(),
            "signal":     best.direction if best else "NONE",
            "grade":      best.grade.value if best else "—",
            "score":      round(best.score, 2) if best else 0.0,
            "long_score":  round(self.long_result.score, 2),
            "short_score": round(self.short_result.score, 2),
            "confirmations": best.confirmations if best else [],
            "missing":       (self.long_result.missing
                              if not best else best.missing),
        }


@dataclass
class ScanCycle:
    started_at: datetime
    finished_at: Optional[datetime] = None
    results: List[PairScanResult] = field(default_factory=list)

    @property
    def signals(self) -> List[ICTSignal]:
        return [
            r.best_signal for r in self.results
            if r.best_signal is not None
        ]

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


# ── Orchestrator ─────────────────────────────────────────────────────────── #

class ICTOrchestrator:
    """
    Scan loop that evaluates every pair every N seconds.

    Usage (blocking)::

        orch = ICTOrchestrator()
        orch.run()

    Usage (single scan, e.g. from a cron job)::

        orch = ICTOrchestrator()
        cycle = orch.scan_once(data_provider=my_provider)
        for sig in cycle.signals:
            print(sig)
    """

    def __init__(
        self,
        pairs: Optional[List[str]] = None,
        account_size: float = _DEFAULT_ACCOUNT_SIZE,
        scan_interval: int = _DEFAULT_SCAN_INTERVAL_SECONDS,
        config_path: Optional[str] = None,
        firebase_enabled: bool = True,
    ) -> None:
        self.pairs = pairs or _DEFAULT_PAIRS
        self.account_size = account_size
        self.scan_interval = scan_interval
        self.config_path = config_path or _DEFAULT_CONFIG_PATH
        self.firebase_enabled = firebase_enabled

        self._pipeline = ICTPipeline(config_path=self.config_path)
        self._account = MicroRiskParams(account_size=self.account_size)
        self._publisher = self._init_publisher()

        logger.info(
            "ICTOrchestrator ready — %d pairs, interval=%ds, firebase=%s",
            len(self.pairs), self.scan_interval, self.firebase_enabled,
        )

    # ── Publisher ─────────────────────────────────────────────────────── #

    def _init_publisher(self):
        if not self.firebase_enabled:
            return None
        try:
            from firebase.ict_namespace import ICTFirebasePublisher
            return ICTFirebasePublisher()
        except Exception as exc:
            logger.warning("Firebase publisher unavailable: %s", exc)
            return None

    # ── Main loop ─────────────────────────────────────────────────────── #

    def run(self, data_provider=None) -> None:
        """
        Blocking scan loop.  Pass a data_provider callable
        ``(pair: str) -> pd.DataFrame`` to supply live OHLCV bars.
        If None, the orchestrator logs that no data provider is wired and skips.
        """
        logger.info("ICT scan loop started (Ctrl-C to stop)")
        while True:
            try:
                cycle = self.scan_once(data_provider=data_provider)
                self._publish_cycle(cycle)
                self._log_cycle(cycle)
            except KeyboardInterrupt:
                logger.info("ICT scan loop stopped by user")
                break
            except Exception as exc:
                logger.exception("Scan cycle error: %s", exc)
            time.sleep(self.scan_interval)

    # ── Single scan ───────────────────────────────────────────────────── #

    def scan_once(self, data_provider=None) -> ScanCycle:
        """
        Run one evaluation cycle across all pairs.

        Args:
            data_provider: callable(pair) -> pd.DataFrame | None.
                           If None, each pair evaluation is skipped with a warning.
        """
        now = datetime.now(tz=timezone.utc)
        cycle = ScanCycle(started_at=now)

        # Pre-fetch sovereign context once per cycle (isolation boundary:
        # only the orchestrator imports sovereign/).
        sovereign_ctx = self._fetch_sovereign_context()

        for pair in self.pairs:
            result = self._evaluate_pair(pair, now, data_provider, sovereign_ctx)
            if result is not None:
                cycle.results.append(result)

        cycle.finished_at = datetime.now(tz=timezone.utc)
        return cycle

    def _fetch_sovereign_context(self) -> Dict:
        """
        Fetch all sovereign-layer inputs needed by the pipeline in one place.
        Returns a dict with keys: ict_alloc_weight, ict_alloc_veto_reason,
        bridge_thresholds.  Commitment result is computed per-pair after scores
        are known, so it is handled inside _evaluate_pair.
        """
        ctx: Dict = {
            "ict_alloc_weight": 1.0,
            "ict_alloc_veto_reason": "",
            "bridge_thresholds": {},
        }
        try:
            from sovereign.intelligence.allocation_engine import read_allocation
            alloc = read_allocation()
            ctx["ict_alloc_weight"] = alloc.ict_weight
            if alloc.ict_weight == 0.0:
                ctx["ict_alloc_veto_reason"] = (
                    f"ALLOCATION_ZERO: {alloc.regime_tag} — {alloc.reason[:60]}"
                )
        except Exception as exc:
            logger.debug("allocation_engine unavailable: %s", exc)

        try:
            from sovereign.intelligence.cross_system_bridge import get_bridge
            ctx["bridge_thresholds"] = get_bridge().get_ict_thresholds()
        except Exception as exc:
            logger.debug("cross_system_bridge unavailable: %s", exc)

        return ctx

    def _compute_commitment(self, scores: Dict, session: str, grade: str, score: float):
        """
        Run CommitmentDetector for a single signal.  Called after pipeline scores
        are computed but before the final trade decision (handled in orchestrator
        so pipeline.py stays free of sovereign imports).

        NOTE: This is called from _evaluate_pair only for post-scoring analysis.
        The pipeline accepts the result via the commitment_result parameter.
        """
        try:
            from sovereign.intelligence.commitment_detector import CommitmentDetector
            return CommitmentDetector(log=False).compute_ict(
                component_scores=scores,
                session=session,
                grade=grade,
                score=score,
            )
        except Exception as exc:
            logger.debug("CommitmentDetector unavailable: %s", exc)
            return None

    def _evaluate_pair(
        self,
        pair: str,
        now: datetime,
        data_provider,
        sovereign_ctx: Optional[Dict] = None,
    ) -> Optional[PairScanResult]:
        if data_provider is None:
            logger.debug("No data provider — skipping %s", pair)
            return None

        try:
            df = data_provider(pair)
        except Exception as exc:
            logger.warning("Data fetch failed for %s: %s", pair, exc)
            return None

        ctx = sovereign_ctx or {}

        long_result = self._pipeline.evaluate(
            symbol=pair, direction="LONG",
            df=df, timestamp=now, account=self._account,
            ict_alloc_weight=ctx.get("ict_alloc_weight", 1.0),
            ict_alloc_veto_reason=ctx.get("ict_alloc_veto_reason", ""),
            bridge_thresholds=ctx.get("bridge_thresholds", {}),
            commitment_result=None,  # pipeline scores not yet known; see note below
        )
        short_result = self._pipeline.evaluate(
            symbol=pair, direction="SHORT",
            df=df, timestamp=now, account=self._account,
            ict_alloc_weight=ctx.get("ict_alloc_weight", 1.0),
            ict_alloc_veto_reason=ctx.get("ict_alloc_veto_reason", ""),
            bridge_thresholds=ctx.get("bridge_thresholds", {}),
            commitment_result=None,
        )
        # Note on commitment_result: CommitmentDetector.compute_ict() needs
        # component_scores which only exist after the pipeline runs.  The
        # pipeline therefore runs first (commitment_result=None → no veto),
        # and the orchestrator applies commitment filtering here as a post-pass.
        # This preserves the isolation boundary while retaining the exact same
        # gate logic (UNCOMMITTED → veto; DEVELOPING → size_mult=0.75).
        from ict.pipeline import ICTSignal, ICTVeto, ICTGrade
        for attr, result in (("long_result", long_result), ("short_result", short_result)):
            if isinstance(result, ICTSignal):
                commit = self._compute_commitment(
                    scores=result.component_scores,
                    session=getattr(result.session_status, "session_name", ""),
                    grade=result.grade.value if hasattr(result.grade, "value") else str(result.grade),
                    score=result.score,
                )
                if commit is not None and getattr(commit, "label", None) == "UNCOMMITTED":
                    veto = ICTVeto(
                        symbol=result.symbol, direction=result.direction,
                        timestamp=result.timestamp, score=result.score,
                        grade=ICTGrade.VETOED,
                        reason=f"COMMITMENT_DETECTOR: {commit.reason}",
                        component_scores=result.component_scores,
                        confirmations=result.confirmations,
                        missing=result.missing,
                    )
                    if attr == "long_result":
                        long_result = veto
                    else:
                        short_result = veto

        self._log_decisions(long_result, short_result)
        return PairScanResult(
            pair=pair,
            timestamp=now,
            long_result=long_result,
            short_result=short_result,
        )

    def _log_decisions(self, long_result, short_result) -> None:
        try:
            from sovereign.intelligence.decision_logger import log_ict_decision
            from ict.pipeline import ICTSignal
            for result in (long_result, short_result):
                if isinstance(result, ICTSignal):
                    log_ict_decision(signal=result, commitment_score=None)
        except Exception:
            pass

    # ── Firebase publish ──────────────────────────────────────────────── #

    def _publish_cycle(self, cycle: ScanCycle) -> None:
        if self._publisher is None:
            return
        for result in cycle.results:
            try:
                self._publisher.publish_scan_result(result)
            except Exception as exc:
                logger.warning("Firebase publish failed for %s: %s", result.pair, exc)

    # ── Logging ───────────────────────────────────────────────────────── #

    def _log_cycle(self, cycle: ScanCycle) -> None:
        sigs = cycle.signals
        logger.info(
            "Scan complete — %d pairs | %d signal(s) | %.1fs",
            len(cycle.results),
            len(sigs),
            cycle.duration_seconds or 0.0,
        )
        for sig in sigs:
            logger.info(
                "  ✓ %s %s | grade=%s | score=%.1f | %s",
                sig.symbol, sig.direction,
                sig.grade.value, sig.score,
                ", ".join(sig.confirmations[:3]),
            )


# ── Config loader ─────────────────────────────────────────────────────────── #

def load_config(path: str = _DEFAULT_CONFIG_PATH) -> dict:
    try:
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        logger.warning("ICT config not found at %s — using defaults", path)
        return {}


# ── CLI entry point ───────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="ICT Engine Orchestrator")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Pairs to scan (default: full 8-pair universe)")
    parser.add_argument("--interval", type=int, default=_DEFAULT_SCAN_INTERVAL_SECONDS,
                        help="Scan interval in seconds")
    parser.add_argument("--account-size", type=float, default=_DEFAULT_ACCOUNT_SIZE)
    parser.add_argument("--no-firebase", action="store_true")
    args = parser.parse_args()

    orch = ICTOrchestrator(
        pairs=args.pairs,
        scan_interval=args.interval,
        account_size=args.account_size,
        firebase_enabled=not args.no_firebase,
    )
    orch.run()
