"""
CostCalibrator — sovereign/execution/cost_calibrator.py  (Gap 1: estimate → measure → update)

Backtest costs are hardcoded guesses; live fills are truth. This connects them: it
reads real fills, computes median realized slippage per pair, compares to the
backtester's assumption, messages drift, and writes calibrated_costs.json that the
backtester overlays — so the reported Sharpe is finally verified against execution.

SCOPE (honest): execution_tracker logs SLIPPAGE (|fill - signal|), not spread (which
needs bid/ask at fill). So this calibrates slippage only; spread stays modeled and is
flagged unverified. Inert until ≥20 real fills (fills.jsonl is currently empty) — it
returns INSUFFICIENT_DATA and writes nothing until then.

Gating: calibrated costs are MEASUREMENT (backtest accuracy), not live config/risk —
auto-apply + logged drift, consistent with non-negotiable #4. Touches no config/.
"""
from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILLS_LOG = ROOT / "data" / "execution" / "fills.jsonl"
CALIBRATED = ROOT / "data" / "execution" / "calibrated_costs.json"
MESSAGES = ROOT / "data" / "agent" / "messages_to_colin.json"

MIN_FILLS_TOTAL = 20      # below this → INSUFFICIENT_DATA
MIN_FILLS_PER_PAIR = 5
DRIFT_PIPS_THRESHOLD = 0.3
ASSUMED_SLIPPAGE_PIPS = 0.5   # mirrors forex_backtester.SLIPPAGE_PER_SIDE (0.00005 = 0.5 pips)

log = logging.getLogger("execution.cost_calibrator")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_pair(p: str) -> str:
    return str(p).replace("=X", "").replace("_", "").upper()


def _pip_size(pair: str) -> float:
    return 0.01 if "JPY" in _norm_pair(pair) else 0.0001


def _message(subject: str, body: str, priority: str = "IMPORTANT") -> None:
    try:
        data = json.loads(MESSAGES.read_text()) if MESSAGES.exists() else {"messages": []}
        data.setdefault("messages", []).insert(0, {
            "id": f"costcal-{_now()[:19].replace(':', '').replace('-', '')}",
            "timestamp": _now(), "priority": priority, "source": "COST_CALIBRATOR",
            "subject": subject, "message": body, "action_required": False,
        })
        data["messages"] = data["messages"][:80]
        MESSAGES.write_text(json.dumps(data, indent=2))
    except Exception as exc:
        log.warning("message write failed: %s", exc)


def _load_fills() -> list[dict]:
    if not FILLS_LOG.exists():
        return []
    out = []
    for line in FILLS_LOG.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def calibrate_from_fills() -> dict:
    fills = _load_fills()
    if len(fills) < MIN_FILLS_TOTAL:
        return {"status": "INSUFFICIENT_DATA", "n_fills": len(fills),
                "need": MIN_FILLS_TOTAL,
                "note": "Calibrator inert until enough real fills accumulate."}

    by_pair: dict[str, list[float]] = {}
    for f in fills:
        sp = f.get("slippage_pips")
        if sp is None:
            continue
        by_pair.setdefault(_norm_pair(f.get("pair", "")), []).append(float(sp))

    calibrated, drifts = {}, []
    for pair, slips in by_pair.items():
        if len(slips) < MIN_FILLS_PER_PAIR:
            continue
        med = round(statistics.median(slips), 3)
        drift = med - ASSUMED_SLIPPAGE_PIPS
        calibrated[pair] = {
            "slippage_pips_median": med,
            "slippage_price_per_side": round(med * _pip_size(pair), 8),
            "n": len(slips),
            "drift_pips_vs_assumed": round(drift, 3),
        }
        if abs(drift) > DRIFT_PIPS_THRESHOLD:
            drifts.append((pair, med, drift))
            _message(
                f"COST DRIFT {pair}: live slippage {med:.1f} pips vs assumed {ASSUMED_SLIPPAGE_PIPS:.1f}",
                f"{pair} realized median slippage is {med:.2f} pips (n={len(slips)}) vs the "
                f"backtest assumption of {ASSUMED_SLIPPAGE_PIPS:.1f} pips — drift {drift:+.2f} pips. "
                f"The reported Sharpe is {'OVER' if drift > 0 else 'UNDER'}stated; "
                f"calibrated_costs.json updated so the backtest now uses live truth.",
                priority="IMPORTANT")

    payload = {
        "calibrated_at": _now(),
        "n_fills_total": len(fills),
        "assumed_slippage_pips": ASSUMED_SLIPPAGE_PIPS,
        "slippage_by_pair": calibrated,
        "drifts_flagged": [{"pair": p, "median_pips": m, "drift": d} for p, m, d in drifts],
        "spread_calibration": "NOT_MEASURED — execution_tracker logs slippage only; spread "
                              "stays modeled and unverified (needs bid/ask at fill).",
    }
    CALIBRATED.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATED.write_text(json.dumps(payload, indent=2))
    return {"status": "CALIBRATED", "pairs": len(calibrated),
            "drifts": len(drifts), "path": str(CALIBRATED.relative_to(ROOT))}


if __name__ == "__main__":
    print(json.dumps(calibrate_from_fills(), indent=2))
