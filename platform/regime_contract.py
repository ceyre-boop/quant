"""regime_contract — the neutral data-model + per-strategy classifiers that
build data/agent/system_regime_state.json.

Imports ONLY the standard library + PyYAML (config). Imports neither ict/ nor
sovereign/ — it reads their JSON *outputs*, never their code.

Discipline (mirrors scripts/obsidian_sync.py):
  * Every section carries a status: OK | STALE | UNAVAILABLE.
  * UNAVAILABLE / STALE always carry a reason.
  * A stale/missing source downgrades that strategy's verdict to STAND_ASIDE.
    We NEVER emit a favorable verdict from missing or stale data.
  * No number is hardcoded: thresholds/limits come from config/parameters.yml
    (or config/ict_params.yml). New keys are logged to
    data/agent/param_change_log.jsonl with a rationale before use (done by the
    build script, not here).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

OK = "OK"
STALE = "STALE"
UNAVAILABLE = "UNAVAILABLE"

GO = "GO"
CAUTION = "CAUTION"
STAND_ASIDE = "STAND_ASIDE"
INFO = "INFO"  # research-only sections that never size

NOW = datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Config access (never hardcode thresholds — CLAUDE.md NON-NEGOTIABLE #3).
# --------------------------------------------------------------------------
def load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # local import keeps module import cheap

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Return (data, error). error is a human string when the read failed."""
    if not path.exists():
        return None, f"source missing: {path}"
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh), None
    except (OSError, ValueError) as exc:
        return None, f"source unreadable: {path} ({exc})"


def _age_hours(ts: datetime | None, fallback_path: Path | None = None) -> float | None:
    """Age in hours from an embedded timestamp; fall back to file mtime."""
    ref: datetime | None = ts
    if ref is None and fallback_path is not None and fallback_path.exists():
        ref = datetime.fromtimestamp(fallback_path.stat().st_mtime, tz=timezone.utc)
    if ref is None:
        return None
    return (NOW - ref).total_seconds() / 3600.0


# --------------------------------------------------------------------------
# Section: a status-carrying unit of the contract.
# --------------------------------------------------------------------------
@dataclass
class Section:
    name: str
    status: str = UNAVAILABLE
    verdict: str = STAND_ASIDE
    favorable: bool = False
    size_multiplier: float = 0.0
    reason: str = "not yet populated"
    source: str | None = None
    source_age_hours: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "verdict": self.verdict,
            "favorable": self.favorable,
            "size_multiplier": round(self.size_multiplier, 4),
            "reason": self.reason,
            "source": self.source,
            "source_age_hours": (
                round(self.source_age_hours, 3) if self.source_age_hours is not None else None
            ),
            "detail": self.detail,
        }

    def stand_aside(self, status: str, reason: str) -> "Section":
        """Force the safe verdict. Used for every stale/missing path."""
        self.status = status
        self.verdict = STAND_ASIDE
        self.favorable = False
        self.size_multiplier = 0.0
        self.reason = reason
        return self


# --------------------------------------------------------------------------
# Per-strategy classifiers. Each is small, documented, config-driven.
# Each takes the loaded config dict + resolved data-dir and returns a Section.
# None of them raise; they return a STAND_ASIDE/UNAVAILABLE section on failure.
# --------------------------------------------------------------------------
def _vix_is_extreme(config: dict[str, Any], data_dir: Path) -> tuple[bool, float | None]:
    """Read VIX from macro_snapshot.json; compare to config vix_extreme_threshold.
    Returns (is_extreme, vix_value). If VIX unreadable, treat as NOT extreme but
    the caller should note it — we never fabricate a value."""
    threshold = (
        config.get("hard_constraints", {}).get("vix_extreme_threshold")
    )
    data, err = _read_json(data_dir / "macro" / "macro_snapshot.json")
    if data is None or threshold is None:
        return False, None
    try:
        vix = float(data["series"]["vix"]["value"])
    except (KeyError, TypeError, ValueError):
        return False, None
    return vix >= float(threshold), vix


def classify_carry(config: dict[str, Any], data_dir: Path, freshness_h: float) -> Section:
    """carry <- data/agent/forex_proximity.json (per-pair differential_trend).

    Rule (Connected Edge x Regime — trend of the driver, not its level):
      * STAND_ASIDE if VIX EXTREME, or if ALL pairs NARROWING.
      * GO / favorable iff >=1 pair WIDENING and VIX not EXTREME;
        size_multiplier = (# widening) / (# pairs).
      * else CAUTION (mixed, none widening) — favorable False, size 0.
    """
    src = data_dir / "agent" / "forex_proximity.json"
    sec = Section("carry", source=str(_rel(src)))
    data, err = _read_json(src)
    if data is None:
        return sec.stand_aside(UNAVAILABLE, err or "forex_proximity.json unreadable")

    ts = _parse_ts(data.get("last_scan"))
    age = _age_hours(ts, src)
    sec.source_age_hours = age
    if age is not None and age > freshness_h:
        return sec.stand_aside(
            STALE, f"forex_proximity.json {age:.1f}h old > {freshness_h}h freshness limit"
        )

    pairs = data.get("pairs") or []
    if not pairs:
        return sec.stand_aside(UNAVAILABLE, "forex_proximity.json has no pairs")

    per_pair: dict[str, dict[str, Any]] = {}
    trends: list[str] = []
    for p in pairs:
        name = str(p.get("pair", "?"))
        trend = str(p.get("differential_trend", "UNKNOWN")).upper()
        trends.append(trend)
        per_pair[name] = {
            "differential_trend": trend,
            "rate_differential": p.get("rate_differential"),
            "regime": p.get("regime"),
        }
    sec.detail["per_pair"] = per_pair

    is_extreme, vix = _vix_is_extreme(config, data_dir)
    sec.detail["vix"] = vix
    sec.detail["vix_extreme"] = is_extreme

    n = len(trends)
    n_wide = sum(1 for t in trends if t == "WIDENING")
    n_narrow = sum(1 for t in trends if t == "NARROWING")

    if is_extreme:
        sec.status = OK
        return sec.stand_aside(OK, f"VIX EXTREME ({vix}) — carry stands aside")

    if n_wide >= 1:
        sec.status = OK
        sec.verdict = GO
        sec.favorable = True
        sec.size_multiplier = n_wide / n
        sec.reason = f"{n_wide}/{n} pairs WIDENING; VIX {vix} not extreme"
        return sec

    if n_narrow == n:
        # today's live truth
        sec.status = OK
        sec.verdict = STAND_ASIDE
        sec.favorable = False
        sec.size_multiplier = 0.0
        sec.reason = f"rate differentials NARROWING on all {n} pairs; none widening"
        return sec

    # mixed: none widening, not all narrowing
    sec.status = OK
    sec.verdict = CAUTION
    sec.favorable = False
    sec.size_multiplier = 0.0
    sec.reason = f"no pair WIDENING ({n_narrow}/{n} narrowing); differentials not improving"
    return sec


def classify_es_nq(config: dict[str, Any], data_dir: Path, freshness_h: float) -> Section:
    """es_nq <- data/research/nqes_regime.json. Research-only: verdict INFO,
    NEVER sizes (no execution path — CME futures, OANDA is forex-only)."""
    src = data_dir / "research" / "nqes_regime.json"
    sec = Section("es_nq", source=str(_rel(src)))
    data, err = _read_json(src)
    if data is None:
        return sec.stand_aside(UNAVAILABLE, err or "nqes_regime.json unreadable")

    ts = _parse_ts(data.get("as_of"))
    age = _age_hours(ts, src)
    sec.source_age_hours = age
    if age is not None and age > freshness_h:
        return sec.stand_aside(
            STALE, f"nqes_regime.json {age:.1f}h old > {freshness_h}h freshness limit"
        )

    sec.status = OK
    sec.verdict = INFO
    sec.favorable = False
    sec.size_multiplier = 0.0  # research-only never sizes
    sec.detail = {
        "regime": data.get("regime"),
        "read": data.get("read"),
        "divergence": data.get("divergence"),
        "contemporaneous_corr": data.get("contemporaneous_corr"),
    }
    sec.reason = f"research INFO only — regime {data.get('regime')!r}"
    return sec


def classify_macro(config: dict[str, Any], data_dir: Path, freshness_h: float) -> Section:
    """macro backdrop <- data/macro/macro_snapshot.json (VIX, curve, credit).
    A shared INFO input (never sizes on its own)."""
    src = data_dir / "macro" / "macro_snapshot.json"
    sec = Section("macro", source=str(_rel(src)))
    data, err = _read_json(src)
    if data is None:
        return sec.stand_aside(UNAVAILABLE, err or "macro_snapshot.json unreadable")

    ts = _parse_ts(data.get("fetched_at"))
    age = _age_hours(ts, src)
    sec.source_age_hours = age
    if age is not None and age > freshness_h:
        return sec.stand_aside(
            STALE, f"macro_snapshot.json {age:.1f}h old > {freshness_h}h freshness limit"
        )

    series = data.get("series") or {}

    def _val(key: str) -> Any:
        node = series.get(key)
        return node.get("value") if isinstance(node, dict) else None

    sec.status = OK
    sec.verdict = INFO
    sec.favorable = False
    sec.size_multiplier = 0.0
    sec.detail = {
        "vix": _val("vix"),
        "curve_10y2y": _val("t10y2y"),
        "curve_10y3m": _val("t10y3m"),
        "credit_hyg": _val("hyg"),
        "dgs10": _val("dgs10"),
    }
    sec.reason = "macro backdrop INFO (VIX / curve / credit) — shared input"
    return sec


def _rel(path: Path) -> Path:
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path
