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

# ── Direction-quality constants (Phase 2) ─────────────────────────────────── #

# Minimum absolute score for the winning direction to emit a signal.
# Scores the pipeline — must be at-or-above this to avoid emitting noisy trades.
_MIN_SCORE_TO_TRADE: float = 6.5

# Winning direction must beat the opposing direction by at least this margin.
# Prevents trading when both sides look similar (ambiguous market).
_MIN_SCORE_GAP: float = 1.5

# Pairs where USD is the quote currency (LONG = weak USD, SHORT = strong USD)
_USD_QUOTED_PAIRS = ["GBPUSD", "EURUSD", "AUDUSD", "NZDUSD"]
# Pairs where USD is the base currency (SHORT = weak USD, LONG = strong USD)
_USD_BASE_PAIRS = ["USDJPY", "USDCAD"]


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
    Default implementation uses yfinance at 1m resolution for ICT-grade intraday fidelity.
    5m data is too coarse for accurate sweep / FVG detection.
    """

    def get_ohlcv(self, symbol: str, period: str = "5d", interval: str = "1m"):
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

        # Multi-pair USD confluence boost
        results = self._apply_confluence(results)
        for r in results:
            self.publisher.publish_pair(r)

        self.publisher.publish_system({
            "updated_at": now.isoformat(),
            "pairs_scanned": len(self.pairs),
            "signals": sum(1 for r in results if r.signal in ("LONG", "SHORT")),
            "version": "v1",
            "engine": "ICT_MICRO_EDGE",
        })

        return results

    # ── Internal ───────────────────────────────────────────────────────── #

    def _apply_confluence(self, results: List[PairScanResult]) -> List[PairScanResult]:
        """
        Multi-pair USD confluence: if ≥3 USD pairs all point the SAME USD direction,
        boost the score of each aligned signal by 0.5 (capped at 10.0).

        USD direction:
          Weak USD  — LONG  on GBPUSD/EURUSD/AUDUSD/NZDUSD
                    — SHORT on USDJPY/USDCAD
          Strong USD — SHORT on GBPUSD/EURUSD/AUDUSD/NZDUSD
                    — LONG  on USDJPY/USDCAD

        Only signals that ALIGN with the dominant direction receive the boost.
        Non-USD pairs (GBPJPY, AUDNZD, …) are never counted or boosted here.
        """
        def _is_usd_weak(r: PairScanResult) -> bool:
            return (
                (r.signal == "LONG"  and any(p in r.pair for p in _USD_QUOTED_PAIRS)) or
                (r.signal == "SHORT" and any(p in r.pair for p in _USD_BASE_PAIRS))
            )

        def _is_usd_strong(r: PairScanResult) -> bool:
            return (
                (r.signal == "SHORT" and any(p in r.pair for p in _USD_QUOTED_PAIRS)) or
                (r.signal == "LONG"  and any(p in r.pair for p in _USD_BASE_PAIRS))
            )

        usd_weak_count   = sum(1 for r in results if _is_usd_weak(r))
        usd_strong_count = sum(1 for r in results if _is_usd_strong(r))

        boosted = []
        for r in results:
            if r.signal not in ("LONG", "SHORT"):
                boosted.append(r)
                continue

            aligned_weak   = usd_weak_count   >= 3 and _is_usd_weak(r)
            aligned_strong = usd_strong_count >= 3 and _is_usd_strong(r)

            if aligned_weak or aligned_strong:
                dominant_count = usd_weak_count if aligned_weak else usd_strong_count
                direction_label = "weak" if aligned_weak else "strong"
                new_score = min(r.score + 0.5, 10.0)
                confs = r.confirmations + [
                    f"USD {direction_label} confluence boost (+0.5): "
                    f"{dominant_count} pairs aligned"
                ]
                from dataclasses import replace
                r = replace(r, score=new_score, confirmations=confs)
            boosted.append(r)
        return boosted

    def _scan_pair(self, pair: str, now: datetime, dp: DataProvider) -> PairScanResult:
        from ict.pipeline import ICTSignal, ICTVeto

        df = dp.get_ohlcv(pair)
        account = self._MicroRiskParams(account_size=self.account_size)

        # Evaluate both directions so we can apply the score-gap quality filter.
        results_by_dir: dict = {}
        for direction in ("LONG", "SHORT"):
            results_by_dir[direction] = self._pipeline.evaluate(
                symbol=pair,
                direction=direction,
                df=df,
                timestamp=now,
                account=account,
            )

        long_score  = results_by_dir["LONG"].score
        short_score = results_by_dir["SHORT"].score
        best_dir    = "LONG" if long_score >= short_score else "SHORT"
        other_dir   = "SHORT" if best_dir == "LONG" else "LONG"
        best        = results_by_dir[best_dir]
        best_score  = getattr(best, "score", 0.0)
        other_score = getattr(results_by_dir[other_dir], "score", 0.0)

        ts = now.isoformat()

        # ── Phase 2: direction quality gates ─────────────────────────────
        # Gate A: winning side score must meet the minimum tradeable threshold.
        if best_score < _MIN_SCORE_TO_TRADE:
            return PairScanResult(
                pair=pair, timestamp=ts,
                signal="FLAT", grade="—", score=best_score,
                session="—", confirmations=[], missing=[],
                veto_reason=f"SCORE_TOO_LOW: best={best_score:.1f} < {_MIN_SCORE_TO_TRADE}",
            )

        # Gate B: winning side must have a clear advantage over the opposite side.
        score_gap = best_score - other_score
        if score_gap < _MIN_SCORE_GAP:
            return PairScanResult(
                pair=pair, timestamp=ts,
                signal="FLAT", grade="—", score=best_score,
                session="—", confirmations=[], missing=[],
                veto_reason=(
                    f"AMBIGUOUS_DIRECTION: gap={score_gap:.1f} < {_MIN_SCORE_GAP} "
                    f"(LONG={long_score:.1f} SHORT={short_score:.1f})"
                ),
            )

        if isinstance(best, ICTSignal) and best.passed:
            sz = best.sizing
            return PairScanResult(
                pair=pair, timestamp=ts,
                signal=best.direction, grade=best.grade.value,
                score=best.score,
                session=best.session_status.kill_zone_name or "OFF-HOURS",
                confirmations=best.confirmations,
                missing=best.missing,
                risk_pct=sz.risk_pct,
                entry=sz.entry_price,          # PositionSizing stores entry as entry_price, not entry
                stop=sz.stop_loss,
                tp1=sz.tp1,
                tp2=sz.tp2,
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
