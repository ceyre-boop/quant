"""
Regime Performance Tracker — answers: "Is this strategy appropriate right now?"

Every trade close is tagged with:
  - regime:    MOMENTUM | REVERSION | FLAT  (from Softmax + HMM)
  - vol_state: EXPANDING | COMPRESSING | NEUTRAL  (ATR slope)
  - system:    ICT | FOREX | EQUITY

Records appended to data/intelligence/regime_performance.jsonl.

Query API:
  tracker.tag_trade(...)       → write one record
  tracker.get_stats(system, regime, vol_state, min_n)
                               → {"wr": float, "avg_r": float, "sharpe": float,
                                  "n": int, "above_expectancy": bool|None}
  tracker.get_all_stats()      → dict[system][regime][vol_state] → stats
  tracker.summary()            → markdown-style text for dashboard
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ROOT   = Path(__file__).resolve().parents[2]
_OUTPUT = _ROOT / "data" / "intelligence" / "regime_performance.jsonl"

SYSTEMS   = ("ICT", "FOREX", "EQUITY")
REGIMES   = ("MOMENTUM", "REVERSION", "FLAT", "UNKNOWN")
VOL_STATES = ("EXPANDING", "COMPRESSING", "NEUTRAL")

# Minimum trades per cell before stats are considered reliable
MIN_N_REPORT = 30

# Backtest / prior expectancy per system — used for z-score baseline.
# These are conservative estimates from historical backtests.
# Update them as live data accumulates.
_PRIOR_SHARPE: Dict[str, float] = {
    "ICT":    1.225,   # London-only, from CLAUDE.md
    "FOREX":  1.071,   # v007 avg Sharpe
    "EQUITY": 0.500,   # conservative prior; no confirmed figure yet
}


# ── Record schema ──────────────────────────────────────────────────────── #

def _make_record(
    system: str,
    regime: str,
    vol_state: str,
    pnl_r: float,
    trade_id: str = "",
    pair: str = "",
    session: str = "",
    extra: Optional[dict] = None,
) -> dict:
    return {
        "tagged_at": datetime.now(timezone.utc).isoformat(),
        "system":    system.upper(),
        "regime":    regime.upper(),
        "vol_state": vol_state.upper(),
        "pnl_r":     float(pnl_r),
        "win":       pnl_r > 0,
        "trade_id":  trade_id,
        "pair":      pair,
        "session":   session,
        **(extra or {}),
    }


# ── Vol-state classification ───────────────────────────────────────────── #

def classify_vol_state(atr_series: List[float], slope_window: int = 5) -> str:
    """
    Classify volatility direction from an ATR time-series.

    Args:
        atr_series: ordered list of ATR values, most-recent last.
        slope_window: number of bars used for slope calculation.

    Returns:
        'EXPANDING' | 'COMPRESSING' | 'NEUTRAL'
    """
    if len(atr_series) < slope_window + 1:
        return "NEUTRAL"

    recent = atr_series[-(slope_window + 1):]
    x = np.arange(len(recent), dtype=float)
    y = np.array(recent, dtype=float)

    if y.std() < 1e-9:
        return "NEUTRAL"

    slope = float(np.polyfit(x, y, 1)[0])
    baseline = float(np.mean(y))
    rel_slope = slope / (baseline + 1e-9)

    if rel_slope > 0.02:
        return "EXPANDING"
    if rel_slope < -0.02:
        return "COMPRESSING"
    return "NEUTRAL"


# ── Tracker class ─────────────────────────────────────────────────────── #

class RegimePerformanceTracker:
    """
    Append-only log + query layer for per-regime performance.

    Usage
    -----
    tracker = RegimePerformanceTracker()

    # On trade close:
    tracker.tag_trade(
        system="ICT",
        regime="MOMENTUM",
        vol_state="EXPANDING",
        pnl_r=-1.0,
        trade_id="ICT_gbpusd_001",
        pair="GBPUSD",
        session="London",
    )

    # Query:
    stats = tracker.get_stats("ICT", "MOMENTUM", min_n=30)
    """

    def __init__(self, output_path: Optional[Path] = None) -> None:
        self._path = Path(output_path) if output_path else _OUTPUT
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Optional[List[dict]] = None

    # ── Write ──────────────────────────────────────────────────────── #

    def tag_trade(
        self,
        system: str,
        regime: str,
        vol_state: str,
        pnl_r: float,
        trade_id: str = "",
        pair: str = "",
        session: str = "",
        extra: Optional[dict] = None,
    ) -> dict:
        """
        Append one tagged trade record.

        Args:
            system:    "ICT" | "FOREX" | "EQUITY"
            regime:    "MOMENTUM" | "REVERSION" | "FLAT" | "UNKNOWN"
            vol_state: "EXPANDING" | "COMPRESSING" | "NEUTRAL"
            pnl_r:     P&L in units of R (1.0 = 1R profit, -1.0 = 1R loss)
            trade_id:  Optional identifier
            pair:      Optional symbol/pair
            session:   Optional session label
            extra:     Optional extra fields to store

        Returns:
            The record dict that was written.
        """
        record = _make_record(
            system=system,
            regime=regime,
            vol_state=vol_state,
            pnl_r=pnl_r,
            trade_id=trade_id,
            pair=pair,
            session=session,
            extra=extra,
        )
        with self._path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self._cache = None  # invalidate cache
        return record

    # ── Read ──────────────────────────────────────────────────────── #

    def _load(self) -> List[dict]:
        if self._cache is not None:
            return self._cache
        if not self._path.exists():
            self._cache = []
            return self._cache
        records = []
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        self._cache = records
        return records

    def _filter(
        self,
        records: List[dict],
        system: str,
        regime: Optional[str] = None,
        vol_state: Optional[str] = None,
        last_n: Optional[int] = None,
    ) -> List[dict]:
        out = [r for r in records if r.get("system", "").upper() == system.upper()]
        if regime:
            out = [r for r in out if r.get("regime", "").upper() == regime.upper()]
        if vol_state:
            out = [r for r in out if r.get("vol_state", "").upper() == vol_state.upper()]
        if last_n:
            out = out[-last_n:]
        return out

    def get_stats(
        self,
        system: str,
        regime: Optional[str] = None,
        vol_state: Optional[str] = None,
        min_n: int = MIN_N_REPORT,
        last_n: Optional[int] = None,
    ) -> dict:
        """
        Compute WR, avgR, Sharpe, trade count for a (system, regime, vol_state) cell.

        Returns dict with keys: wr, avg_r, sharpe, n, above_expectancy, reliable.
        above_expectancy is None when reliable=False.
        """
        records = self._filter(
            self._load(), system, regime, vol_state, last_n
        )
        n = len(records)
        if n == 0:
            return {"wr": None, "avg_r": None, "sharpe": None,
                    "n": 0, "above_expectancy": None, "reliable": False}

        r_values = [rec["pnl_r"] for rec in records]
        wins = [r > 0 for r in r_values]
        wr       = float(sum(wins)) / n
        avg_r    = float(np.mean(r_values))
        std_r    = float(np.std(r_values, ddof=1)) if n > 1 else 0.0
        sharpe   = avg_r / (std_r + 1e-9)
        reliable = n >= min_n

        prior = _PRIOR_SHARPE.get(system.upper(), 0.5)
        above = (sharpe > prior) if reliable else None

        return {
            "wr":               round(wr,    4),
            "avg_r":            round(avg_r, 4),
            "sharpe":           round(sharpe, 4),
            "n":                n,
            "above_expectancy": above,
            "reliable":         reliable,
        }

    def get_all_stats(self, min_n: int = MIN_N_REPORT) -> dict:
        """
        Returns nested dict: stats[system][regime][vol_state] → get_stats result.
        Only populated for cells that have at least 1 record.
        """
        records = self._load()
        result: dict = {}
        for sys in SYSTEMS:
            result[sys] = {}
            for reg in REGIMES:
                result[sys][reg] = {}
                for vs in VOL_STATES:
                    cell = self._filter(records, sys, reg, vs)
                    if cell:
                        result[sys][reg][vs] = self.get_stats(
                            sys, reg, vs, min_n=min_n
                        )
        return result

    def current_regime_stats(
        self,
        system: str,
        regime: str,
        vol_state: str = "NEUTRAL",
        min_n: int = MIN_N_REPORT,
    ) -> dict:
        """
        Convenience: stats for a specific current regime.
        Falls back to regime-only (ignoring vol_state) if vol_state cell is thin.
        """
        stats = self.get_stats(system, regime, vol_state, min_n=min_n)
        if not stats["reliable"]:
            # fall back to regime-only (pool all vol_states)
            stats = self.get_stats(system, regime, vol_state=None, min_n=min_n)
            stats["vol_state_fallback"] = True
        return stats

    def rolling_sharpe_zscore(
        self,
        system: str,
        regime: str,
        window: int = 20,
    ) -> Optional[float]:
        """
        Z-score of most-recent `window` trades' Sharpe vs. historical Sharpe for
        that system/regime cell.  None if insufficient data.

        Used by the capital allocator to decide sizing.
        """
        records = self._filter(self._load(), system, regime)
        if len(records) < window + 5:
            return None

        all_r = [r["pnl_r"] for r in records]
        recent_r = all_r[-window:]

        recent_sharpe = float(np.mean(recent_r)) / (float(np.std(recent_r, ddof=1)) + 1e-9)

        # Historical distribution: compute Sharpe for rolling windows of same size
        sharpe_dist = []
        step = max(1, window // 4)
        for i in range(0, len(all_r) - window, step):
            chunk = all_r[i : i + window]
            s = float(np.mean(chunk)) / (float(np.std(chunk, ddof=1)) + 1e-9)
            sharpe_dist.append(s)

        if len(sharpe_dist) < 3:
            return None

        mu  = float(np.mean(sharpe_dist))
        std = float(np.std(sharpe_dist, ddof=1))
        if std < 1e-9:
            return None

        return (recent_sharpe - mu) / std

    # ── Reporting ─────────────────────────────────────────────────────── #

    def summary(
        self,
        min_n: int = MIN_N_REPORT,
        current_regime: Optional[str] = None,
    ) -> str:
        """
        Return a plain-text summary table of reliable cells, sorted by system.
        Highlights current_regime if provided.
        """
        records = self._load()
        total = len(records)
        lines = [f"Regime Performance Tracker  ({total} total records)"]
        lines.append("-" * 60)

        for sys in SYSTEMS:
            sys_records = self._filter(records, sys)
            if not sys_records:
                continue
            lines.append(f"\n{sys}:")
            for reg in REGIMES:
                for vs in VOL_STATES:
                    cell = self._filter(sys_records, sys, reg, vs)
                    if len(cell) < 5:
                        continue
                    stats = self.get_stats(sys, reg, vs, min_n=min_n)
                    marker = ""
                    if current_regime and reg == current_regime.upper():
                        marker = " ◄ CURRENT"
                    reliable_flag = "" if stats["reliable"] else " [thin]"
                    above = ""
                    if stats["above_expectancy"] is True:
                        above = " ✓"
                    elif stats["above_expectancy"] is False:
                        above = " ✗"
                    lines.append(
                        f"  {reg:<12} {vs:<12}  "
                        f"n={stats['n']:<5} WR={stats['wr']:.0%}  "
                        f"avgR={stats['avg_r']:+.3f}  "
                        f"Sharpe={stats['sharpe']:+.2f}{above}{reliable_flag}{marker}"
                    )

        return "\n".join(lines)


# ── Module-level singleton ─────────────────────────────────────────────── #

_tracker: Optional[RegimePerformanceTracker] = None


def get_tracker() -> RegimePerformanceTracker:
    global _tracker
    if _tracker is None:
        _tracker = RegimePerformanceTracker()
    return _tracker
