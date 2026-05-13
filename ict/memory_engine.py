"""
ict/memory_engine.py
====================
ICT Microstructure Pattern Memory Engine.

Stores every scan result as a feature vector. After enough history
accumulates, clusters similar days and returns:
  - Which cluster today matches
  - Historical win rate of that cluster
  - Expected outcome (CONTINUATION / REVERSAL / FLAT)
  - Most similar past trade

State persists to data/ledger/ict_memory.json
Requires >= 20 closed trades to activate clustering.

Output pushed to Firebase: signals/ICT_ENGINE/memory/{pair}
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml
logger = logging.getLogger(__name__)

MEMORY_PATH  = Path('data/ledger/ict_memory.json')
MIN_PATTERNS = 20   # minimum closed trades before clustering activates


@dataclass
class MemoryMatch:
    pair:             str
    cluster:          int         # -1 = unclustered
    similarity:       float       # 0–1 cosine similarity to cluster centroid
    expected_outcome: str         # CONTINUATION | REVERSAL | FLAT | INSUFFICIENT_DATA
    historical_wr:    float       # win rate of this cluster
    n_samples:        int         # trades in this cluster
    analog_date:      str         # closest matching past trade date
    analog_outcome:   str         # what that trade did
    available:        bool        # False if < MIN_PATTERNS

    def to_dict(self):
        return asdict(self)


class ICTMemoryEngine:
    """
    Stores scan results as feature vectors and matches today against history.
    Lightweight — no sklearn dependency, uses pure cosine similarity + k-means.
    """

    def __init__(self, memory_path: Optional[Path] = None, config_path: Optional[str] = None):
        self._memory_path = memory_path or MEMORY_PATH
        self._cfg = self._load_config(config_path)
        self._cluster_k = int(self._cfg.get("cluster_k", 3))
        self._min_patterns = int(self._cfg.get("min_trades_before_cluster_gate", MIN_PATTERNS))
        self._soft_veto_wr = float(self._cfg.get("cluster_soft_veto_wr", 0.40))
        self._hard_veto_wr = float(self._cfg.get("cluster_hard_veto_wr", 0.30))
        self._similarity_floor = float(self._cfg.get("cluster_similarity_floor", 0.50))
        self._soft_veto_penalty = float(self._cfg.get("cluster_soft_veto_penalty", 0.75))
        self._soft_veto_score_floor = float(self._cfg.get("cluster_soft_veto_score_floor", 7.5))

        self._memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

    # ── Public API ─────────────────────────────────────────────────────────── #

    def record_scan(self, scan_result) -> None:
        """Store a scan result feature vector. Call after every scan."""
        vec = self._vectorize(scan_result)
        if vec is None:
            return
        entry = {
            'pair':      scan_result.pair,
            'timestamp': scan_result.timestamp,
            'signal':    scan_result.signal,
            'grade':     scan_result.grade,
            'score':     scan_result.score,
            'session':   scan_result.session,
            'vec':       vec,
            'outcome':   None,   # filled when trade closes
        }
        self._state.setdefault('scans', []).append(entry)
        # Keep only last 2000 scans
        self._state['scans'] = self._state['scans'][-2000:]
        self._save()

    def record_outcome(self, trade_id: str, pair: str, outcome: str, pnl_r: float) -> None:
        """Update the outcome field on the most recent scan for this pair."""
        scans = self._state.get('scans', [])
        for s in reversed(scans):
            if s['pair'] == pair and s.get('outcome') is None:
                s['outcome'] = outcome
                s['pnl_r']   = pnl_r
                break
        self._save()

    def match(self, scan_result) -> MemoryMatch:
        """Find the best cluster match for the current scan result."""
        closed = [s for s in self._state.get('scans', []) if s.get('outcome')]
        if len(closed) < self._min_patterns:
            return MemoryMatch(
                pair=scan_result.pair, cluster=-1, similarity=0.0,
                expected_outcome='INSUFFICIENT_DATA',
                historical_wr=0.0, n_samples=len(closed),
                analog_date='', analog_outcome='', available=False,
            )

        vec = self._vectorize(scan_result)
        if vec is None:
            return MemoryMatch(
                pair=scan_result.pair, cluster=-1, similarity=0.0,
                expected_outcome='INSUFFICIENT_DATA',
                historical_wr=0.0, n_samples=0,
                analog_date='', analog_outcome='', available=False,
            )

        pair_closed = [s for s in closed if s['pair'] == scan_result.pair]
        if len(pair_closed) < 10:
            pair_closed = closed  # fall back to all pairs if not enough pair-specific

        # Find k-means clusters (default k=3: win / loss / flat)
        clusters = self._cluster(pair_closed, k=max(2, self._cluster_k))

        # Find which cluster today's vector is closest to
        best_cluster, best_sim = 0, -1.0
        for cid, centroid in enumerate(clusters['centroids']):
            sim = _cosine(vec, centroid)
            if sim > best_sim:
                best_sim, best_cluster = sim, cid

        cluster_trades = [s for s in pair_closed if s.get('_cluster') == best_cluster]
        wins = [s for s in cluster_trades if (s.get('pnl_r') or 0) > 0]
        wr   = len(wins) / len(cluster_trades) if cluster_trades else 0.0

        # Expected outcome
        avg_r = sum(s.get('pnl_r', 0) for s in cluster_trades) / max(len(cluster_trades), 1)
        expected = 'CONTINUATION' if avg_r > 0.5 else 'REVERSAL' if avg_r < -0.3 else 'FLAT'

        # Best analog (most similar individual trade)
        analog = max(pair_closed, key=lambda s: _cosine(vec, s.get('vec', [])))
        analog_date    = analog.get('timestamp', '')[:10]
        analog_outcome = analog.get('outcome', '—')

        return MemoryMatch(
            pair=scan_result.pair,
            cluster=best_cluster,
            similarity=round(best_sim, 3),
            expected_outcome=expected,
            historical_wr=round(wr, 3),
            n_samples=len(cluster_trades),
            analog_date=analog_date,
            analog_outcome=analog_outcome,
            available=True,
        )

    def assess_match(self, match: MemoryMatch) -> dict:
        """
        Returns hard/soft veto decision based on cluster quality thresholds.
        """
        if not match.available:
            return {
                "hard_veto": False,
                "soft_veto": False,
                "penalty": 0.0,
                "score_floor": self._soft_veto_score_floor,
                "reason": "insufficient_data",
            }

        hard_veto = (
            match.similarity < self._similarity_floor
            or match.historical_wr < self._hard_veto_wr
        )
        if hard_veto:
            return {
                "hard_veto": True,
                "soft_veto": False,
                "penalty": 0.0,
                "score_floor": self._soft_veto_score_floor,
                "reason": "hard_threshold",
            }

        soft_veto = match.historical_wr < self._soft_veto_wr
        return {
            "hard_veto": False,
            "soft_veto": soft_veto,
            "penalty": self._soft_veto_penalty if soft_veto else 0.0,
            "score_floor": self._soft_veto_score_floor,
            "reason": "soft_cluster_wr" if soft_veto else "ok",
        }

    # ── Internal ───────────────────────────────────────────────────────────── #

    def _vectorize(self, scan_result) -> Optional[List[float]]:
        """Convert a ScanResult into a fixed-length feature vector."""
        try:
            score   = float(scan_result.score or 0)
            session = {'NY_PM': 1.0, 'London': 0.5, 'OFF-HOURS': 0.0}.get(scan_result.session, 0.0)
            signal  = 1.0 if scan_result.signal == 'LONG' else -1.0 if scan_result.signal == 'SHORT' else 0.0
            grade   = {'A+': 1.0, 'A': 0.8, 'B': 0.5, 'C': 0.2, 'VETO': 0.0}.get(scan_result.grade, 0.0)
            adr     = float(scan_result.adr_pct or 0)
            risk    = float(scan_result.risk_pct or 0)
            n_conf  = len(scan_result.confirmations or [])
            n_miss  = len(scan_result.missing or [])
            return [score/10, session, signal, grade, adr, risk/0.02, n_conf/10, n_miss/10]
        except Exception:
            return None

    def _cluster(self, trades: List[dict], k: int = 3) -> dict:
        """Minimal k-means clustering on trade feature vectors."""
        vecs = [t['vec'] for t in trades if t.get('vec') and len(t['vec']) == 8]
        if len(vecs) < k:
            return {'centroids': [vecs[0]] if vecs else [[0]*8]}

        # Initialize centroids from spread-out samples
        step = len(vecs) // k
        centroids = [vecs[i * step] for i in range(k)]

        for _ in range(10):  # 10 iterations is enough for small datasets
            assignments = [_nearest(v, centroids) for v in vecs]
            new_centroids = []
            for cid in range(k):
                members = [vecs[i] for i, a in enumerate(assignments) if a == cid]
                if members:
                    new_centroids.append(_mean_vec(members))
                else:
                    new_centroids.append(centroids[cid])
            centroids = new_centroids

        # Tag trades with cluster assignment
        assignments = [_nearest(v, centroids) for v in vecs]
        vi = 0
        for t in trades:
            if t.get('vec') and len(t.get('vec', [])) == 8:
                t['_cluster'] = assignments[vi]
                vi += 1

        return {'centroids': centroids}

    def _load(self) -> dict:
        if self._memory_path.exists():
            try:
                return json.loads(self._memory_path.read_text())
            except Exception:
                pass
        return {'scans': [], 'created': datetime.now(timezone.utc).isoformat()}

    def _save(self):
        self._memory_path.write_text(json.dumps(self._state, indent=2, default=str))

    @staticmethod
    def _load_config(config_path: Optional[str]) -> dict:
        path = config_path or _default_config_path()
        try:
            with open(path) as f:
                full = yaml.safe_load(f) or {}
            return full.get('memory', {})
        except FileNotFoundError:
            logger.warning("ICT config not found at %s — using memory defaults", path)
            return {}


def _default_config_path() -> str:
    import os
    override = os.environ.get("ICT_CONFIG_PATH")
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "config", "ict_params.yml")


# ── Math helpers ───────────────────────────────────────────────────────────── #

def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot  = sum(x*y for x,y in zip(a,b))
    na   = math.sqrt(sum(x*x for x in a))
    nb   = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb + 1e-9)

def _nearest(vec: List[float], centroids: List[List[float]]) -> int:
    return max(range(len(centroids)), key=lambda i: _cosine(vec, centroids[i]))

def _mean_vec(vecs: List[List[float]]) -> List[float]:
    n = len(vecs)
    return [sum(v[i] for v in vecs) / n for i in range(len(vecs[0]))]
