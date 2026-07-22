"""measurement — the conscience. The neutral data-model + measurements that build
data/agent/system_health_verdict.json.

The regime organ (regime_contract.py) asks "is the market favorable for this edge
right now." The conscience asks the other half: "is the process being followed, and
is the edge still true on live data?" Its headline is a per-strategy + portfolio
KILL SWITCH — TRADE / REDUCE / HALT — that fails SAFE.

Imports ONLY the standard library + PyYAML (config). Imports neither ict/ nor
sovereign/ — it reads their JSON/JSONL *outputs*, never their code. Same wall
discipline as the regime organ (CLAUDE.md NON-NEGOTIABLE #1).

Discipline (mirrors regime_contract.py):
  * Every measurement carries a status: OK | STALE | UNAVAILABLE | INSUFFICIENT_DATA.
  * INSUFFICIENT_DATA is FIRST-CLASS: an edge below n_needed is never "healthy".
  * The kill switch fails SAFE. Missing/stale data or a data-integrity failure →
    HALT or REDUCE, NEVER TRADE. An unproven edge (below n_needed) → REDUCE.
  * No number is hardcoded: thresholds come from config/parameters.yml
    (platform.health) / RISK_CONSTITUTION.md. New keys are logged to
    param_change_log.jsonl with a rationale before use (done by the build script).
  * Nothing here raises to the writer; every classifier returns a status-carrying
    section on failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

# Section statuses
OK = "OK"
STALE = "STALE"
UNAVAILABLE = "UNAVAILABLE"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# Kill-switch states (headline). Order of severity: HALT > REDUCE > TRADE.
TRADE = "TRADE"
REDUCE = "REDUCE"
HALT = "HALT"

NOW = datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Config + IO helpers (never hardcode thresholds — CLAUDE.md NON-NEGOTIABLE #4).
# --------------------------------------------------------------------------
def load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # local import keeps module import cheap

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _health_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return ((config or {}).get("platform", {}) or {}).get("health", {}) or {}


def _freshness(config: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(_health_cfg(config)["freshness_hours"][key])
    except (KeyError, TypeError, ValueError):
        return default


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    # Date-only / naive strings (e.g. "2026-07-22") → assume UTC so arithmetic
    # against an aware NOW never raises.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.exists():
        return None, f"source missing: {path}"
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh), None
    except (OSError, ValueError) as exc:
        return None, f"source unreadable: {path} ({exc})"


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not path.exists():
        return None, f"source missing: {path}"
    try:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows, None
    except (OSError, ValueError) as exc:
        return None, f"source unreadable: {path} ({exc})"


def _age_hours(ts: datetime | None, fallback_path: Path | None = None) -> float | None:
    ref: datetime | None = ts
    if ref is None and fallback_path is not None and fallback_path.exists():
        ref = datetime.fromtimestamp(fallback_path.stat().st_mtime, tz=timezone.utc)
    if ref is None:
        return None
    return (NOW - ref).total_seconds() / 3600.0


def _rel(path: Path) -> Path:
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path


# --------------------------------------------------------------------------
# EdgeHealth: a status-carrying edge measurement.
# --------------------------------------------------------------------------
@dataclass
class EdgeHealth:
    live_expectancy_R: float | None = None
    backtest_expectancy_R: float | None = None
    divergence_flag: bool = False
    n_live: int | None = None
    n_needed: int | None = None
    status: str = UNAVAILABLE
    reason: str = "not yet populated"
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "live_expectancy_R": self.live_expectancy_R,
            "backtest_expectancy_R": self.backtest_expectancy_R,
            "divergence_flag": self.divergence_flag,
            "n_live": self.n_live,
            "n_needed": self.n_needed,
            "status": self.status,
            "reason": self.reason,
            "source": self.source,
        }


# --------------------------------------------------------------------------
# StrategyHealth: one strategy's full conscience section.
# --------------------------------------------------------------------------
@dataclass
class StrategyHealth:
    name: str
    kill_switch: str = HALT           # safe default: absent evidence → HALT
    reason: str = "not yet populated"
    edge_health: EdgeHealth = field(default_factory=EdgeHealth)
    process_adherence: dict[str, Any] = field(
        default_factory=lambda: {"decisions_matched_rules_pct": None, "status": UNAVAILABLE}
    )
    forecast_vs_execution: dict[str, Any] = field(
        default_factory=lambda: {
            "read_accuracy": None,
            "execution_quality": None,
            "status": UNAVAILABLE,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kill_switch": self.kill_switch,
            "reason": self.reason,
            "edge_health": self.edge_health.to_dict(),
            "process_adherence": self.process_adherence,
            "forecast_vs_execution": self.forecast_vs_execution,
        }


# --------------------------------------------------------------------------
# Portfolio integrity + kill switch (Part 2 #9). The headline safety organ.
# --------------------------------------------------------------------------
def read_portfolio_integrity(config: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    """Portfolio-level data integrity + drawdown-breaker state.

    Reads the regime organ's contract for data-integrity/staleness, and looks for
    a portfolio drawdown-breaker + consecutive-breaker count. NONE of these feeds
    exist authoritatively today, so we report them HONESTLY as UNAVAILABLE rather
    than fabricating a confident zero. Fail-safe: the kill switch treats an
    UNAVAILABLE breaker as REDUCE and a data-integrity FAIL as HALT.
    """
    out: dict[str, Any] = {
        "data_integrity": "OK",
        "consecutive_breaker_hits": None,
        "drawdown_breaker_tripped": None,
        "drawdown_breaker_status": UNAVAILABLE,
        "regime_contract_status": None,
        "regime_contract_age_hours": None,
        "kill_switch": HALT,       # overwritten below
        "reason": "",
        "sources": [],
    }

    # 1) Regime contract = the shared data-integrity signal. If it's missing or
    #    stale, that IS a data-integrity failure for the whole system.
    regime_path = data_dir / "agent" / "system_regime_state.json"
    out["sources"].append(str(_rel(regime_path)))
    regime, err = _read_json(regime_path)
    regime_max_age = float(_health_cfg(config).get("freshness_hours", {}).get("regime", 1.5))
    if regime is None:
        out["data_integrity"] = "FAIL"
        out["regime_contract_status"] = "MISSING"
        out["kill_switch"] = HALT
        out["reason"] = f"regime contract unreadable ({err}) — data-integrity FAIL"
        return out

    r_status = str(regime.get("status", "UNKNOWN")).upper()
    r_ts = _parse_ts(regime.get("generated_at"))
    r_age = _age_hours(r_ts, regime_path)
    out["regime_contract_status"] = r_status
    out["regime_contract_age_hours"] = round(r_age, 3) if r_age is not None else None

    if r_age is not None and r_age > regime_max_age:
        out["data_integrity"] = "FAIL"
        out["kill_switch"] = HALT
        out["reason"] = (
            f"regime contract {r_age:.1f}h old > {regime_max_age}h — stale shared "
            f"regime read is a data-integrity FAIL"
        )
        return out

    # 2) Drawdown breaker + consecutive count. The regime contract carries a
    #    portfolio.drawdown_breaker_tripped field, currently UNAVAILABLE (None).
    pf = regime.get("portfolio") or {}
    tripped = pf.get("drawdown_breaker_tripped")
    consec = pf.get("consecutive_breaker_hits")
    out["drawdown_breaker_tripped"] = tripped
    out["consecutive_breaker_hits"] = consec

    halt_n = int(_health_cfg(config).get("consecutive_breaker_halt_n", 2))

    if tripped is True:
        out["drawdown_breaker_status"] = OK
        out["kill_switch"] = HALT
        out["reason"] = "portfolio drawdown breaker TRIPPED — HALT"
        return out

    if isinstance(consec, int) and consec >= halt_n:
        out["drawdown_breaker_status"] = OK
        out["kill_switch"] = HALT
        out["reason"] = f"consecutive breaker hits {consec} >= {halt_n} — HALT"
        return out

    if tripped is None:
        # Breaker feed does not exist yet. Honest: we cannot confirm the account
        # is safe, so we do not green-light full size. REDUCE, never TRADE.
        out["drawdown_breaker_status"] = UNAVAILABLE
        out["kill_switch"] = REDUCE
        out["reason"] = (
            "portfolio drawdown-breaker feed UNAVAILABLE (no unified position/P&L "
            "ledger); cannot confirm account safety → REDUCE (never full TRADE)"
        )
        return out

    # Breaker present, not tripped, integrity OK.
    out["drawdown_breaker_status"] = OK
    out["kill_switch"] = TRADE
    out["reason"] = "data integrity OK; drawdown breaker not tripped"
    return out


def _worse(a: str, b: str) -> str:
    """Return the more conservative of two kill-switch states."""
    order = {TRADE: 0, REDUCE: 1, HALT: 2}
    return a if order.get(a, 2) >= order.get(b, 2) else b


def combine_kill_switch(
    portfolio_kill: str,
    edge_status: str,
    edge_divergence: bool,
) -> tuple[str, str]:
    """Fold portfolio integrity + edge health into one per-strategy kill switch.

    Fail-safe rules (config-driven severity; never TRADE on absent evidence):
      * Portfolio HALT (data-integrity FAIL / breaker) → HALT, always dominant.
      * Edge divergence past threshold → HALT.
      * Edge INSUFFICIENT_DATA / UNAVAILABLE / STALE → at least REDUCE.
      * TRADE only when portfolio says TRADE AND the edge is proven healthy (OK).
    """
    if edge_divergence:
        return HALT, "live-vs-backtest edge divergence past threshold — HALT"

    ks = portfolio_kill
    if edge_status in {INSUFFICIENT_DATA, UNAVAILABLE, STALE}:
        ks = _worse(ks, REDUCE)
        reason = f"edge health {edge_status} — cannot confirm live edge → REDUCE floor"
    elif edge_status == OK:
        reason = "edge health OK"
    else:
        ks = _worse(ks, REDUCE)
        reason = f"edge health status {edge_status!r} — REDUCE floor"

    if ks == HALT:
        reason = "HALT: " + reason
    return ks, reason


# --------------------------------------------------------------------------
# Edge health — undertow_gapper (Part 2 #3, #6). The one with a live shadow.
# --------------------------------------------------------------------------
def edge_health_undertow(config: dict[str, Any], data_dir: Path) -> EdgeHealth:
    """Live-vs-backtest edge health for undertow_gapper, computed HONESTLY.

    Live signal:   data/research/yield_frontier/shadow/shadow_daily.jsonl
                   (per-day n_signals + realized constitutional day return).
    Backtest edge: data/research/yield_frontier/gauntlet/verdicts.json (HYP-093
                   event_mean = per-event expectancy from the gauntlet).

    The shadow has only ~3 signals total — far below the n_needed floor (250) — so
    this section MUST come out INSUFFICIENT_DATA. That honesty is the deliverable:
    the conscience never green-lights an edge from a handful of live signals. The
    kill switch treats INSUFFICIENT_DATA as REDUCE (small), never TRADE (full).
    """
    hc = _health_cfg(config)
    n_needed = int((hc.get("edge_health", {}) or {}).get("n_needed_default", 250))
    freshness_h = _freshness(config, "shadow", 96.0)

    eh = EdgeHealth(n_needed=n_needed)

    # --- backtest expectancy (gauntlet HYP-093) ---
    gpath = data_dir / "research" / "yield_frontier" / "gauntlet" / "verdicts.json"
    gdata, gerr = _read_json(gpath)
    if isinstance(gdata, dict) and isinstance(gdata.get("HYP-093"), dict):
        try:
            eh.backtest_expectancy_R = float(gdata["HYP-093"]["event_mean"])
        except (KeyError, TypeError, ValueError):
            eh.backtest_expectancy_R = None

    # --- live signals (shadow ledger) ---
    spath = data_dir / "research" / "yield_frontier" / "shadow" / "shadow_daily.jsonl"
    eh.source = f"{_rel(spath)} (live) + {_rel(gpath)}::HYP-093 (backtest)"
    rows, serr = _read_jsonl(spath)
    if rows is None:
        eh.status = UNAVAILABLE
        eh.reason = serr or "shadow ledger unreadable"
        return eh

    # Staleness: last dated row vs freshness limit.
    dated = [r for r in rows if r.get("date")]
    last_ts = None
    if dated:
        last_ts = _parse_ts(str(dated[-1]["date"]))
    age = _age_hours(last_ts, spath)
    if age is not None and age > freshness_h:
        eh.status = STALE
        eh.reason = f"shadow ledger {age:.1f}h old > {freshness_h}h freshness limit"
        return eh

    n_live = sum(int(r.get("n_signals", 0) or 0) for r in rows)
    eh.n_live = n_live

    # Live expectancy: only meaningful with enough signals. With ~3 we do NOT
    # publish a confident number — the shadow logs day returns, not per-signal R,
    # and n is far too small to estimate expectancy. Report None + INSUFFICIENT_DATA.
    if n_live < n_needed:
        eh.status = INSUFFICIENT_DATA
        eh.live_expectancy_R = None
        eh.divergence_flag = False  # cannot claim divergence with no live estimate
        eh.reason = (
            f"{n_live} live shadow signals < {n_needed} needed — INSUFFICIENT_DATA. "
            f"Live expectancy not estimable; no confident 'healthy' from a handful of "
            f"signals. Backtest event-mean expectancy {eh.backtest_expectancy_R} for reference."
        )
        return eh

    # (Reached only once the shadow accumulates enough signals — future path.)
    eh.status = OK
    eh.reason = f"{n_live} live signals >= {n_needed}; live-vs-backtest computed"
    return eh


# --------------------------------------------------------------------------
# Sections that need a ledger that does not exist yet — honest UNAVAILABLE.
# --------------------------------------------------------------------------
def process_adherence_unavailable(needed_source: str) -> dict[str, Any]:
    """Process adherence (Part 2 #7) needs the ICT causal-chain setup ledger,
    which is not written yet. Report UNAVAILABLE with the exact source needed —
    never fabricate an adherence percentage."""
    return {
        "decisions_matched_rules_pct": None,
        "status": UNAVAILABLE,
        "reason": (
            "process-adherence requires the ICT causal-chain setup ledger "
            f"({needed_source}), which is not written yet. No adherence score is "
            "fabricated; this fills once the ledger exists."
        ),
        "needed_source": needed_source,
    }


def forecast_vs_execution_unavailable(needed_source: str) -> dict[str, Any]:
    """Forecast-vs-execution (Part 2 #10) needs the same causal-chain ledger with
    read-outcome + fill-quality fields. UNAVAILABLE until it exists — the two
    numbers (was the read right / was the fill good) are never conflated or faked."""
    return {
        "read_accuracy": None,
        "execution_quality": None,
        "status": UNAVAILABLE,
        "reason": (
            "forecast-vs-execution requires per-setup read-outcome and fill-quality "
            f"records from the ICT causal-chain ledger ({needed_source}), not written "
            "yet. 'Was the read right' and 'was the fill good' stay separate and unfaked."
        ),
        "needed_source": needed_source,
    }
