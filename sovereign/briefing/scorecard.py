#!/usr/bin/env python3
"""C8 — Briefing scorecard (the thing that makes the briefing honest, not just impressive).

Every morning the briefing states a bias + a confidence. We LOG it, then score it against
what actually happened — the same log-now / score-against-reality-later discipline as the
decision_logger. Over a real sample this answers the only question that matters: is the
briefing's confidence CALIBRATED, or is the narrative just narrative?

For each past briefing (dated, pre-open) we score the regular-session outcome of that day:
  - direction_correct : did price move the stated bias direction (by close vs prior close)?
  - regime_correct    : did NQ/ES behave as the regime call implied?
  - key_level_held     : did the stated invalidation level hold (or break as implied)?

report() (after ~30-60 samples) computes directional hit-rate, confidence CALIBRATION
(when it says 70%, is it ~70% right?), and regime-classification accuracy.

Appends data/briefing/scorecard.jsonl. Idempotent — never double-scores a date.

Usage:  python3 -m sovereign.briefing.scorecard [--report]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
for _lib in ("yfinance", "urllib3", "requests"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

BRIEF_DIR = ROOT / "data" / "oracle" / "market_briefings"
SCORE_FILE = ROOT / "data" / "briefing" / "scorecard.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _day_metrics(sym: str, d: date):
    """Regular-session return for date d (close vs prior close) + the day's range."""
    import yfinance as yf
    h = yf.Ticker(sym).history(
        start=(d - timedelta(days=8)).isoformat(),
        end=(d + timedelta(days=3)).isoformat(),
        interval="1d", auto_adjust=True,
    )
    if h is None or len(h) < 2:
        return None
    rows = [(ts.date(), r) for ts, r in h.iterrows()]
    for i, (dd, r) in enumerate(rows):
        if dd == d and i > 0:
            prev_close = float(rows[i - 1][1]["Close"])
            close = float(r["Close"])
            return {"close": close, "high": float(r["High"]), "low": float(r["Low"]),
                    "ret_pct": (close / prev_close - 1) * 100 if prev_close else None}
    return None


def _scored_dates() -> set:
    if not SCORE_FILE.exists():
        return set()
    out = set()
    for line in SCORE_FILE.read_text().splitlines():
        if line.strip():
            try:
                out.add(json.loads(line)["briefing_date"])
            except Exception:
                continue
    return out


def _briefing_dates() -> list[str]:
    if not BRIEF_DIR.exists():
        return []
    out = []
    for f in BRIEF_DIR.glob("*.json"):
        if f.stem == "latest":
            continue
        try:
            date.fromisoformat(f.stem)
            out.append(f.stem)
        except Exception:
            continue
    return sorted(out)


def _score_one(briefing_date: str) -> dict | None:
    try:
        b = json.loads((BRIEF_DIR / f"{briefing_date}.json").read_text())
    except Exception:
        return None
    d = date.fromisoformat(briefing_date)

    bias = (b.get("directional_bias") or "").upper() or None
    conf = b.get("confidence")
    regime = b.get("regime_call") or b.get("meta_regime")
    key_level = b.get("key_level")

    nq = _day_metrics("NQ=F", d)
    es = _day_metrics("ES=F", d)
    if nq is None:
        return None  # no settled bar yet — retry next run

    nq_ret = nq["ret_pct"]
    es_ret = es["ret_pct"] if es else None

    # Directional correctness (NQ is the primary instrument for the bias).
    direction_correct = None
    if bias in ("LONG", "SHORT") and nq_ret is not None:
        direction_correct = (nq_ret > 0) if bias == "LONG" else (nq_ret < 0)

    # Regime correctness (best-effort behavioral check).
    regime_correct = None
    if regime and nq_ret is not None and es_ret is not None:
        spread = nq_ret - es_ret
        same_sign = (nq_ret >= 0) == (es_ret >= 0)
        if regime == "CONCENTRATION":
            regime_correct = spread > 0            # NQ outperformed ES
        elif regime == "BREADTH":
            regime_correct = same_sign and abs(spread) < 0.5
        elif regime in ("ROTATION_WARN", "ROTATION_DIVERGENCE"):
            regime_correct = (not same_sign) or spread < 0

    # Key-level (invalidation) — held as implied by the bias?
    key_level_held = None
    try:
        kl = float(key_level) if key_level is not None else None
    except Exception:
        kl = None
    if kl is not None:
        if bias == "LONG":
            key_level_held = nq["low"] >= kl       # invalidation below held
        elif bias == "SHORT":
            key_level_held = nq["high"] <= kl      # invalidation above held

    return {
        "briefing_date": briefing_date,
        "scored_at": _now(),
        "directional_bias": bias,
        "confidence": conf,
        "regime_call": regime,
        "key_level": kl,
        "nq_return_pct": round(nq_ret, 2) if nq_ret is not None else None,
        "es_return_pct": round(es_ret, 2) if es_ret is not None else None,
        "direction_correct": direction_correct,
        "regime_correct": regime_correct,
        "key_level_held": key_level_held,
    }


def score_yesterday(max_backfill: int = 10) -> int:
    """Score any past, unscored briefings that now have a settled session. Returns count."""
    scored = _scored_dates()
    today = date.today().isoformat()
    pending = [d for d in _briefing_dates() if d < today and d not in scored][-max_backfill:]
    n = 0
    SCORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SCORE_FILE.open("a") as fh:
        for bd in pending:
            rec = _score_one(bd)
            if rec is not None:
                fh.write(json.dumps(rec) + "\n")
                n += 1
    return n


def report() -> dict:
    if not SCORE_FILE.exists():
        return {"n": 0, "note": "No scored briefings yet — scorecard starts after the first session settles."}
    recs = []
    for line in SCORE_FILE.read_text().splitlines():
        if line.strip():
            try:
                recs.append(json.loads(line))
            except Exception:
                continue

    dirs = [r for r in recs if r.get("direction_correct") is not None]
    regimes = [r for r in recs if r.get("regime_correct") is not None]
    n = len(dirs)
    hit = round(sum(1 for r in dirs if r["direction_correct"]) / n, 3) if n else None
    regime_acc = (round(sum(1 for r in regimes if r["regime_correct"]) / len(regimes), 3)
                  if regimes else None)

    # Confidence calibration buckets: stated confidence vs realized hit-rate.
    buckets = {"<50": [], "50-69": [], "70-84": [], "85+": []}
    for r in dirs:
        c = r.get("confidence")
        if c is None:
            continue
        b = "<50" if c < 50 else "50-69" if c < 70 else "70-84" if c < 85 else "85+"
        buckets[b].append(r["direction_correct"])
    calibration = {b: {"n": len(v), "hit_rate": round(sum(v) / len(v), 3)} for b, v in buckets.items() if v}

    return {
        "n_scored_directional": n,
        "directional_hit_rate": hit,
        "regime_accuracy": regime_acc,
        "n_regime_scored": len(regimes),
        "calibration": calibration,
        "maturity": ("CALIBRATING — need ~30-60 samples before the hit-rate/calibration mean anything"
                     if n < 30 else "MATURE — calibration is now interpretable"),
    }


def summary_line() -> str:
    r = report()
    if not r.get("n_scored_directional"):
        return "Briefing scorecard: no scored calls yet (starts once a session settles)."
    return (f"Briefing scorecard: n={r['n_scored_directional']} dir-calls, hit {r['directional_hit_rate']}, "
            f"regime acc {r['regime_accuracy']} — {r['maturity']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        print(json.dumps(report(), indent=2))
    else:
        n = score_yesterday()
        print(f"Scored {n} briefing(s). {summary_line()}")


if __name__ == "__main__":
    main()
