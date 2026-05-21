"""
System Health Monitor — failure-state intelligence.

Answers: "Is this system currently reliable?"

Per-system, per-day metrics:
  1. WR entropy of last 20 outcomes (binary Shannon entropy)
  2. Consecutive loss streak
  3. Variance of R multiples (last 20 trades)
  4. Structural break: rolling 20-trade WR vs. historical WR —
     flag if outside 95% CI from binomial distribution

Output:
  health_score  : float [0.0, 1.0]
  degradation   : bool
  reliability   : "HIGH" | "MEDIUM" | "LOW" | "UNRELIABLE"

Red-flag messages are written to data/agent/messages_to_colin.json for any
system that enters UNRELIABLE state.

Usage
-----
from sovereign.intelligence.system_health import SystemHealthMonitor

monitor = SystemHealthMonitor()
snapshot = monitor.compute("ICT", r_multiples=[-1, -1, 1, -1, 1, -1, -1])
print(snapshot["reliability"])   # → "LOW"
monitor.check_and_alert()        # → writes to messages_to_colin if UNRELIABLE
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import binom

logger = logging.getLogger(__name__)

_ROOT          = Path(__file__).resolve().parents[2]
_HEALTH_LOG    = _ROOT / "data" / "intelligence" / "system_health.jsonl"
_MESSAGES_PATH = _ROOT / "data" / "agent" / "messages_to_colin.json"

SYSTEMS = ("ICT", "FOREX", "EQUITY")

# Rolling window for short-term health computation
WINDOW = 20

# Binomial confidence level for structural break detection
BINOM_CI = 0.95

# Reliability thresholds
_THRESHOLDS = {
    "HIGH":        0.75,   # health_score ≥ 0.75
    "MEDIUM":      0.50,   # health_score ≥ 0.50
    "LOW":         0.25,   # health_score ≥ 0.25
    # < 0.25 → UNRELIABLE
}


# ── Statistical helpers ───────────────────────────────────────────────── #

def _binary_entropy(wr: float) -> float:
    """Shannon binary entropy H(p) = -p log₂p - (1-p) log₂(1-p)."""
    if wr <= 0.0 or wr >= 1.0:
        return 0.0
    q = 1.0 - wr
    return -(wr * math.log2(wr) + q * math.log2(q))


def _consecutive_losses(outcomes: List[bool]) -> int:
    """Count the current streak of losses at the end of the sequence."""
    streak = 0
    for win in reversed(outcomes):
        if not win:
            streak += 1
        else:
            break
    return streak


def _structural_break(
    recent_wins: int,
    recent_n: int,
    historical_wr: float,
    ci: float = BINOM_CI,
) -> Tuple[bool, float, float]:
    """
    Test if recent_wins / recent_n is outside the binomial CI for historical_wr.

    Returns (is_break, lower_bound, upper_bound).
    """
    if recent_n < 5 or historical_wr <= 0 or historical_wr >= 1:
        return False, 0.0, 1.0

    alpha = 1.0 - ci
    lo = binom.ppf(alpha / 2,   recent_n, historical_wr) / recent_n
    hi = binom.ppf(1 - alpha / 2, recent_n, historical_wr) / recent_n

    current_wr = recent_wins / recent_n
    is_break = bool(current_wr < lo or current_wr > hi)
    return is_break, float(lo), float(hi)


# ── Health computation ────────────────────────────────────────────────── #

def _compute_health_score(
    r_multiples: List[float],
    historical_wr: Optional[float] = None,
) -> dict:
    """
    Compute a health snapshot from a list of R multiples.

    Args:
        r_multiples:   Most-recent last. Positive = win, negative = loss.
        historical_wr: Long-run win rate baseline.  If None, uses the full
                       r_multiples history as baseline.

    Returns dict with all health components.
    """
    if not r_multiples:
        return {
            "health_score": 0.5,
            "degradation": False,
            "reliability": "MEDIUM",
            "n": 0,
            "note": "No data",
        }

    outcomes   = [r > 0 for r in r_multiples]
    window_r   = r_multiples[-WINDOW:]
    window_out = outcomes[-WINDOW:]
    n_win      = len(window_r)

    # --- 1. WR entropy ---
    recent_wins = sum(window_out)
    recent_wr   = recent_wins / n_win if n_win else 0.5
    entropy     = _binary_entropy(recent_wr)
    # Normalise: max entropy at WR=0.5 → H=1.0; pure win/loss → H=0.0
    entropy_score = entropy  # already in [0, 1]

    # --- 2. Consecutive loss streak ---
    streak          = _consecutive_losses(outcomes)
    streak_penalty  = min(1.0, streak / 8.0)   # 8 losses in a row = 1.0 penalty

    # --- 3. R-variance (higher variance = less reliable, but not always bad) ---
    r_std = float(np.std(window_r, ddof=1)) if n_win > 1 else 0.0
    # Normalise relative to mean abs R; cap at 1.0
    mean_abs_r  = float(np.mean(np.abs(window_r))) if window_r else 1.0
    var_score   = min(1.0, r_std / (mean_abs_r + 1e-9) / 3.0)

    # --- 4. Structural break ---
    if historical_wr is None:
        historical_wr = sum(outcomes) / len(outcomes) if outcomes else 0.5
    is_break, lo, hi = _structural_break(recent_wins, n_win, historical_wr)

    # --- Composite health score ---
    # Components: entropy_ok (1=50-50, bad), streak (1=many losses, bad),
    #             var (1=very volatile, warning), structural_break (1=bad)
    break_penalty = 0.3 if is_break else 0.0
    raw_penalty   = (
        0.30 * streak_penalty
        + 0.20 * var_score
        + 0.30 * break_penalty
        + 0.20 * (1.0 - entropy_score)  # low entropy (all wins or all losses) = low uncertainty signal
    )
    health_score = max(0.0, min(1.0, 1.0 - raw_penalty))

    # --- Reliability label ---
    if health_score >= _THRESHOLDS["HIGH"]:
        reliability = "HIGH"
    elif health_score >= _THRESHOLDS["MEDIUM"]:
        reliability = "MEDIUM"
    elif health_score >= _THRESHOLDS["LOW"]:
        reliability = "LOW"
    else:
        reliability = "UNRELIABLE"

    degradation = reliability in ("LOW", "UNRELIABLE")

    return {
        "health_score":       round(health_score, 4),
        "degradation":        degradation,
        "reliability":        reliability,
        "n":                  n_win,
        "recent_wr":          round(recent_wr, 4),
        "consecutive_losses": streak,
        "r_std":              round(r_std, 4),
        "structural_break":   is_break,
        "break_ci_lo":        round(lo, 4),
        "break_ci_hi":        round(hi, 4),
        "historical_wr":      round(historical_wr, 4),
        "entropy":            round(entropy, 4),
    }


# ── Monitor class ─────────────────────────────────────────────────────── #

class SystemHealthMonitor:
    """
    Computes, logs, and alerts on per-system health state.

    Usage
    -----
    monitor = SystemHealthMonitor()
    snap = monitor.compute("ICT", r_multiples=[...])
    monitor.check_and_alert()   # writes to messages_to_colin if UNRELIABLE
    """

    def __init__(
        self,
        health_log: Optional[Path] = None,
        messages_path: Optional[Path] = None,
    ) -> None:
        self._log_path = Path(health_log) if health_log else _HEALTH_LOG
        self._msg_path = Path(messages_path) if messages_path else _MESSAGES_PATH
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Compute + log ──────────────────────────────────────────────── #

    def compute(
        self,
        system: str,
        r_multiples: List[float],
        historical_wr: Optional[float] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        """
        Compute health snapshot for a system and append to log.

        Args:
            system:        "ICT" | "FOREX" | "EQUITY"
            r_multiples:   All closed-trade R values, oldest first.
            historical_wr: Long-run baseline WR.  None = use full r_multiples.
            extra:         Optional extra fields to store.

        Returns:
            dict with health_score, degradation, reliability, and all components.
        """
        snap = _compute_health_score(r_multiples, historical_wr)
        snap.update({
            "system":    system.upper(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(extra or {}),
        })

        with self._log_path.open("a") as f:
            f.write(json.dumps(snap) + "\n")

        if snap["reliability"] == "UNRELIABLE":
            logger.warning(
                f"[SystemHealth] {system} is UNRELIABLE "
                f"(score={snap['health_score']:.2f}, "
                f"streak={snap['consecutive_losses']}, "
                f"break={snap['structural_break']})"
            )
        return snap

    def latest_per_system(self) -> Dict[str, dict]:
        """Return the most-recent health snapshot per system."""
        if not self._log_path.exists():
            return {}
        snaps: Dict[str, dict] = {}
        with self._log_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    sys = rec.get("system", "")
                    if sys:
                        snaps[sys] = rec
                except json.JSONDecodeError:
                    pass
        return snaps

    # ── Alert ─────────────────────────────────────────────────────── #

    def check_and_alert(self) -> List[str]:
        """
        Check the most-recent snapshot for each system.
        Write a red-flag message to messages_to_colin for any UNRELIABLE system.

        Returns list of system names that triggered an alert.
        """
        latest = self.latest_per_system()
        alerted = []
        for sys, snap in latest.items():
            if snap.get("reliability") == "UNRELIABLE":
                self._write_message(
                    text=(
                        f"🔴 {sys} system is UNRELIABLE — "
                        f"health_score={snap['health_score']:.2f}, "
                        f"consecutive_losses={snap['consecutive_losses']}, "
                        f"structural_break={snap['structural_break']}, "
                        f"recent_wr={snap.get('recent_wr', '?'):.0%}. "
                        f"Do NOT deploy to prop challenge until health recovers."
                    ),
                    priority="URGENT",
                )
                alerted.append(sys)
            elif snap.get("reliability") == "LOW":
                self._write_message(
                    text=(
                        f"🟡 {sys} system health is LOW — "
                        f"health_score={snap['health_score']:.2f}, "
                        f"consecutive_losses={snap['consecutive_losses']}. "
                        f"Consider reducing size until trend reverses."
                    ),
                    priority="IMPORTANT",
                )
                alerted.append(sys)
        return alerted

    def _write_message(self, text: str, priority: str) -> None:
        """Append to messages_to_colin.json (same format as oracle_agent)."""
        emoji_map = {"URGENT": "🔴", "IMPORTANT": "🟡", "FYI": "🟢"}
        emoji = emoji_map.get(priority, "🟢")
        now = datetime.now(timezone.utc)

        try:
            if self._msg_path.exists():
                data = json.loads(self._msg_path.read_text())
            else:
                data = {"messages": []}
        except (json.JSONDecodeError, OSError):
            data = {"messages": []}

        messages = data.get("messages", [])
        # Avoid duplicate alerts within the same hour
        for msg in messages:
            if msg.get("text") == text and not msg.get("read", True):
                return

        messages.append({
            "id":        f"health-{now.strftime('%Y%m%d%H%M%S')}",
            "priority":  priority,
            "emoji":     emoji,
            "text":      text,
            "timestamp": now.isoformat(),
            "read":      False,
        })
        # Keep last 100 messages
        data["messages"] = messages[-100:]

        self._msg_path.parent.mkdir(parents=True, exist_ok=True)
        self._msg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # ── Summary ───────────────────────────────────────────────────── #

    def summary(self) -> str:
        """Plain-text health snapshot for all systems."""
        latest = self.latest_per_system()
        if not latest:
            return "No system health data yet."

        lines = ["System Health Summary"]
        lines.append("-" * 50)
        for sys in SYSTEMS:
            snap = latest.get(sys)
            if not snap:
                lines.append(f"  {sys:<10}  no data")
                continue
            r = snap["reliability"]
            icon = {"HIGH": "✓", "MEDIUM": "~", "LOW": "↓", "UNRELIABLE": "✗"}.get(r, "?")
            lines.append(
                f"  {sys:<10}  {icon} {r:<12}  "
                f"score={snap['health_score']:.2f}  "
                f"streak={snap['consecutive_losses']}  "
                f"break={snap['structural_break']}"
            )
        return "\n".join(lines)


# ── Module-level singleton ─────────────────────────────────────────────── #

_monitor: Optional[SystemHealthMonitor] = None


def get_monitor() -> SystemHealthMonitor:
    global _monitor
    if _monitor is None:
        _monitor = SystemHealthMonitor()
    return _monitor
