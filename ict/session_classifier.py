"""
ict/session_classifier.py
=========================
Time-of-day / session classifier for the ICT micro-edge engine.

Returns which Kill Zone (if any) is currently active, whether it is
a high-probability window, and whether the NY lunch consolidation period
should block new entries.

ISOLATION: No imports from sovereign/, layer1/, layer2/, layer3/.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Optional, List

import yaml

logger = logging.getLogger(__name__)

# ── Defaults (overridden by ict_params.yml) ──────────────────────────────── #

_DEFAULT_KILL_ZONES: dict = {
    "London":  ("02:00", "05:00"),
    "NY_Open": ("07:00", "10:00"),
    "NY_PM":   ("13:30", "16:00"),
    "Asia":    ("20:00", "23:59"),
}
_DEFAULT_HP_ZONES = {"London", "NY_Open", "NY_PM"}
_DEFAULT_LUNCH_START = "12:00"
_DEFAULT_LUNCH_END = "13:30"


# ── Data classes ─────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class SessionWindow:
    """Describes a named Kill Zone window."""
    name: str
    start_utc: time
    end_utc: time
    is_high_probability: bool

    def contains(self, t: time) -> bool:
        """Return True if *t* (UTC) falls inside [start_utc, end_utc]."""
        if self.start_utc <= self.end_utc:
            return self.start_utc <= t <= self.end_utc
        # Overnight window (e.g. 23:00 → 01:00) — not currently used but safe
        return t >= self.start_utc or t <= self.end_utc


@dataclass(frozen=True)
class KillZoneStatus:
    """Result of classifying a single timestamp."""
    timestamp: datetime                # original tz-aware or naive UTC input
    utc_time: time                     # resolved UTC time component
    in_kill_zone: bool
    kill_zone_name: Optional[str]
    is_high_probability: bool
    in_ny_lunch: bool
    should_trade: bool                 # True iff in HP kill zone AND NOT lunch


# ── Classifier ───────────────────────────────────────────────────────────── #

class SessionClassifier:
    """
    Classifies any timestamp into ICT session windows.

    Usage::

        clf = SessionClassifier()
        status = clf.classify(datetime.utcnow())
        if status.should_trade:
            ...
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        cfg = self._load_config(config_path)
        self._windows: List[SessionWindow] = self._build_windows(cfg)
        self._lunch_start: time = self._parse_time(
            cfg.get("ny_lunch_start_utc", _DEFAULT_LUNCH_START)
        )
        self._lunch_end: time = self._parse_time(
            cfg.get("ny_lunch_end_utc", _DEFAULT_LUNCH_END)
        )

    # ── Public API ─────────────────────────────────────────────────────── #

    def classify(self, ts: datetime) -> KillZoneStatus:
        """
        Classify *ts* into session/kill-zone membership.

        *ts* may be timezone-aware or naive (assumed UTC if naive).
        Kill zone constants are defined in US/Eastern time, so comparison
        uses ET. The output struct still records UTC for auditability.
        """
        forced = os.getenv('ICT_FORCE_SESSION', '').strip()
        if forced:
            for w in self._windows:
                if w.name == forced:
                    return KillZoneStatus(
                        timestamp=ts,
                        utc_time=self._to_utc_time(ts),
                        in_kill_zone=True,
                        kill_zone_name=forced,
                        is_high_probability=w.is_high_probability,
                        in_ny_lunch=False,
                        should_trade=True,
                    )
            return KillZoneStatus(
                timestamp=ts,
                utc_time=self._to_utc_time(ts),
                in_kill_zone=True,
                kill_zone_name=forced,
                is_high_probability=True,
                in_ny_lunch=False,
                should_trade=True,
            )

        utc_t = self._to_utc_time(ts)
        et_t = self._to_et_time(ts)   # kill zone windows are in ET
        active_window: Optional[SessionWindow] = None
        for w in self._windows:
            if w.contains(et_t):      # compare ET time against ET constants
                active_window = w
                break

        in_lunch = self._lunch_start <= et_t <= self._lunch_end
        in_kz = active_window is not None
        is_hp = in_kz and active_window.is_high_probability  # type: ignore[union-attr]
        should_trade = is_hp and not in_lunch

        return KillZoneStatus(
            timestamp=ts,
            utc_time=utc_t,
            in_kill_zone=in_kz,
            kill_zone_name=active_window.name if active_window else None,
            is_high_probability=is_hp,
            in_ny_lunch=in_lunch,
            should_trade=should_trade,
        )

    @property
    def windows(self) -> List[SessionWindow]:
        return list(self._windows)

    # ── Private helpers ────────────────────────────────────────────────── #

    @staticmethod
    def _load_config(config_path: Optional[str]) -> dict:
        path = config_path or _default_config_path()
        try:
            with open(path) as f:
                full = yaml.safe_load(f)
            return full.get("session", {})
        except FileNotFoundError:
            logger.warning("ICT config not found at %s — using defaults", path)
            return {}

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def _build_windows(self, cfg: dict) -> List[SessionWindow]:
        raw_kz = cfg.get("kill_zones", _DEFAULT_KILL_ZONES)
        hp_set = set(cfg.get("high_probability_zones", list(_DEFAULT_HP_ZONES)))
        windows: List[SessionWindow] = []
        for name, spec in raw_kz.items():
            if isinstance(spec, dict):
                start_s = spec["start_utc"]
                end_s = spec["end_utc"]
            else:
                # Fallback: tuple / list ("HH:MM", "HH:MM")
                start_s, end_s = spec
            windows.append(SessionWindow(
                name=name,
                start_utc=self._parse_time(start_s),
                end_utc=self._parse_time(end_s),
                is_high_probability=(name in hp_set),
            ))
        return windows

    @staticmethod
    def _to_utc_time(ts: datetime) -> time:
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts.time()

    @staticmethod
    def _to_et_time(ts: datetime) -> time:
        from zoneinfo import ZoneInfo
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(ZoneInfo("America/New_York")).time().replace(tzinfo=None)


# ── Module-level helper ───────────────────────────────────────────────────── #

def _default_config_path() -> str:
    import os
    override = os.environ.get("ICT_CONFIG_PATH")
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "config", "ict_params.yml")
