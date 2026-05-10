"""
ict/orchestrator.py
===================
ICT Micro-Edge Orchestrator.

Scans the forex pair universe on a configurable interval, runs
the ICTPipeline on each pair, and publishes results to Firebase
under signals/ICT_ENGINE/<pair>.

ISOLATION RULE
--------------
Does NOT import from: sovereign/, layer1/, layer2/, layer3/
Does NOT use: Sovereign risk engine, Kelly, Alexandrian Library, PTJ gates
Own Firebase namespace: signals/ICT_ENGINE/*

Usage:
    from ict.orchestrator import ICTOrchestrator
    from ict.micro_risk import MicroRiskParams

    orch = ICTOrchestrator()
    orch.run(interval=300)                            # blocking loop
    result = orch.scan_once(data_provider=provider)   # single pass (cron/serverless)
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default pair universe for ICT engine (majors with tight spreads)
_DEFAULT_PAIRS = [
    "GBPUSD=X", "EURUSD=X", "AUDUSD=X",
    "USDJPY=X", "GBPJPY=X", "NZDUSD=X",
]

_FIREBASE_ROOT = "signals/ICT_ENGINE"


# ── Result container ──────────────────────────────────────────────────────── #

@dataclass
class PairScanResult:
    pair: str
    timestamp: str
    signal: str        # LONG | SHORT | FLAT | VETO
    grade: str
    score: float
    session: str
    confirmations: List[str]
    missing: List[str]
    risk_pct: float = 0.0
    entry: float = 0.0
    stop: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    component_scores: Dict[str, float] = field(default_factory=dict)
    veto_reason: str = ""


# ── Firebase publisher ─────────────────────────────────────────────────────── #

class ICTFirebasePublisher:
    """
    Writes ONLY to signals/ICT_ENGINE/* — never touches Sovereign paths.
    Reuses an existing firebase_admin app if Sovereign already initialised one.
    Degrades gracefully if firebase_admin is unavailable.
    """

    def __init__(self, db_url: Optional[str] = None):
        self._db = None
        self._db_url = db_url or os.environ.get(
            "FIREBASE_DB_URL",
            "https://clawd-trading-7b8de-default-rtdb.firebaseio.com"
        )
        self._init()

    def _init(self):
        try:
            import firebase_admin
            from firebase_admin import db as rtdb

            if not firebase_admin._apps:
                sa_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT", "")
                if not sa_path:
                    sa_path = "config/firebase_service_account.json"
                if os.path.exists(sa_path):
                    from firebase_admin import credentials
                    cred = credentials.Certificate(sa_path)
                    firebase_admin.initialize_app(cred, {"databaseURL": self._db_url})
                else:
                    logger.warning("ICT Firebase: no service account found — running offline")
                    return

            self._db = rtdb.reference("/")
            logger.info("ICT Firebase: connected to %s", self._db_url)
        except Exception as e:
            logger.warning("ICT Firebase: unavailable (%s) — running offline", e)

    def publish_pair(self, result: PairScanResult) -> bool:
        if self._db is None:
            return False
        try:
            path = f"{_FIREBASE_ROOT}/{result.pair.replace('=X', '').replace('/', '')}"
            self._db.child(path).set(asdict(result))
            return True
        except Exception as e:
            logger.error("ICT Firebase write failed for %s: %s", result.pair, e)
            return False

    def publish_system(self, data: dict) -> bool:
        if self._db is None:
            return False
        try:
            self._db.child(f"{_FIREBASE_ROOT}/_system").set(data)
            return True
        except Exception as e:
            logger.error("ICT Firebase system write failed: %s", e)
            return False


# ── Data provider interface ────────────────────────────────────────────────── #

class DataProvider:
    """
    Minimal interface for price data. Override for live/paper use.
    Default implementation uses yfinance.
    """

    def get_ohlcv(self, symbol: str, period: str = "5d", interval: str = "5m"):
        import yfinance as yf
        tk = yf.Ticker(symbol)
        df = tk.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data for {symbol}")
        return df


# ── Orchestrator ───────────────────────────────────────────────────────────── #

class ICTOrchestrator:
    """
    Scans ICT pair universe and publishes signals to Firebase.

    Example::

        orch = ICTOrchestrator(account_size=10_000)
        orch.run(interval=300)
    """

    def __init__(
        self,
        pairs: Optional[List[str]] = None,
        account_size: float = 10_000.0,
        db_url: Optional[str] = None,
    ):
        self.pairs = pairs or _DEFAULT_PAIRS
        self.account_size = account_size
        self.publisher = ICTFirebasePublisher(db_url=db_url)
        self._data_provider = DataProvider()

        # Lazy import to avoid heavy deps at module load
        from ict.pipeline import ICTPipeline
        from ict.micro_risk import MicroRiskParams
        self._pipeline = ICTPipeline()
        self._MicroRiskParams = MicroRiskParams

    # ── Public API ─────────────────────────────────────────────────────── #

    def run(self, interval: int = 300):
        """Blocking scan loop. Ctrl-C to stop."""
        logger.info("ICT Orchestrator starting — %d pairs, interval=%ds", len(self.pairs), interval)
        while True:
            try:
                results = self.scan_once()
                logger.info("Scan complete: %d results", len(results))
            except KeyboardInterrupt:
                logger.info("ICT Orchestrator stopped.")
                break
            except Exception as e:
                logger.error("Scan error: %s", e)
            time.sleep(interval)

    def scan_once(self, data_provider: Optional[DataProvider] = None) -> List[PairScanResult]:
        """Single scan pass. Returns list of PairScanResult."""
        dp = data_provider or self._data_provider
        now = datetime.now(timezone.utc)
        results = []

        for pair in self.pairs:
            try:
                result = self._scan_pair(pair, now, dp)
                results.append(result)
                self.publisher.publish_pair(result)
            except Exception as e:
                logger.warning("Pair scan failed for %s: %s", pair, e)

        self.publisher.publish_system({
            "updated_at": now.isoformat(),
            "pairs_scanned": len(self.pairs),
            "signals": sum(1 for r in results if r.signal in ("LONG", "SHORT")),
            "version": "v1",
            "engine": "ICT_MICRO_EDGE",
        })

        return results

    # ── Internal ───────────────────────────────────────────────────────── #

    def _scan_pair(self, pair: str, now: datetime, dp: DataProvider) -> PairScanResult:
        from ict.pipeline import ICTSignal, ICTVeto

        df = dp.get_ohlcv(pair)
        account = self._MicroRiskParams(account_size=self.account_size)

        # Try both directions, take the higher-scoring one
        best = None
        for direction in ("LONG", "SHORT"):
            result = self._pipeline.evaluate(
                symbol=pair,
                direction=direction,
                df=df,
                timestamp=now,
                account=account,
            )
            score = result.score if hasattr(result, "score") else 0.0
            if best is None or score > best.score:
                best = result

        ts = now.isoformat()

        if isinstance(best, ICTSignal) and best.passed:
            sz = best.sizing
            return PairScanResult(
                pair=pair, timestamp=ts,
                signal=best.direction, grade=best.grade.value,
                score=best.score,
                session=best.session_status.kill_zone_name or "OFF-HOURS",
                confirmations=best.confirmations,
                missing=best.missing,
                risk_pct=getattr(sz, "risk_pct", 0.0),
                entry=getattr(sz, "entry", 0.0),
                stop=getattr(sz, "stop_loss", 0.0),
                tp1=getattr(sz, "tp1", 0.0),
                tp2=getattr(sz, "tp2", 0.0),
                component_scores=best.component_scores,
            )
        elif isinstance(best, ICTVeto):
            return PairScanResult(
                pair=pair, timestamp=ts,
                signal="VETO", grade=best.grade.value,
                score=best.score,
                session=getattr(best, "session_label", "—"),
                confirmations=best.confirmations,
                missing=best.missing,
                veto_reason=best.reason,
                component_scores=best.component_scores,
            )
        else:
            return PairScanResult(
                pair=pair, timestamp=ts,
                signal="FLAT", grade="—", score=0.0,
                session="—", confirmations=[], missing=[],
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--account", type=float, default=10_000.0)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    orch = ICTOrchestrator(account_size=args.account)
    if args.once:
        results = orch.scan_once()
        for r in results:
            print(f"{r.pair:12} {r.signal:6} grade={r.grade} score={r.score:.1f} session={r.session}")
    else:
        orch.run(interval=args.interval)
