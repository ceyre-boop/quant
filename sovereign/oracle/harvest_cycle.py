"""
Oracle Learning Cycle — Phase 1: HARVEST
sovereign/oracle/harvest_cycle.py

Runs at 2:00 AM ET daily. No LLM call. Pure computation.
Reads closed trades from the last 24h, runs forensic classification,
computes daily summary. Cost: $0.00.

Output: data/oracle/daily_harvest_YYYY_MM_DD.json
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
LEDGER_DIR     = ROOT / "data" / "ledger"
FORENSICS_FILE = ROOT / "data" / "forensics" / "trade_forensics.jsonl"
HARVEST_DIR    = ROOT / "data" / "oracle"
HARVEST_DIR.mkdir(parents=True, exist_ok=True)

FAILURE_LABELS = ["TIMING_FAILURE", "THESIS_FAILURE", "REGIME_FAILURE",
                  "EXECUTION_FAILURE", "SIZING_FAILURE", "COMMITMENT_FAILURE"]


def _load_recent_ledger_trades(hours: int = 24) -> list[dict]:
    """Load closed trades from trade ledger files within rolling window."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    trades = []
    for path in sorted(LEDGER_DIR.glob("trade_ledger_*.jsonl")):
        try:
            with open(path) as f:
                for line in f:
                    try:
                        t = json.loads(line)
                        ts_str = t.get("closed_at") or t.get("timestamp", "")
                        if not ts_str:
                            continue
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if ts > cutoff:
                            trades.append(t)
                    except Exception:
                        pass
        except Exception:
            pass
    return trades


def _load_recent_forensics(hours: int = 24) -> list[dict]:
    """Load enriched forensic records within rolling window."""
    if not FORENSICS_FILE.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    records = []
    with open(FORENSICS_FILE) as f:
        for line in f:
            try:
                r = json.loads(line)
                ts_str = r.get("entry_time", "")
                if not ts_str:
                    continue
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > cutoff:
                    records.append(r)
            except Exception:
                pass
    return records


def _compute_failure_distribution(records: list[dict]) -> dict:
    dist = {label: 0 for label in FAILURE_LABELS}
    for r in records:
        label = r.get("failure_label", "")
        if label in dist:
            dist[label] += 1
    return dist


def _dominant_failure(dist: dict) -> Optional[str]:
    if not any(dist.values()):
        return None
    return max(dist, key=lambda k: dist[k])


def _detect_anomalies(records: list[dict], trades: list[dict]) -> list[str]:
    anomalies = []

    # 3+ consecutive TIMING_FAILURE
    labels = [r.get("failure_label", "") for r in records if r.get("outcome") == "LOSS"]
    run = 0
    for label in labels:
        if label == "TIMING_FAILURE":
            run += 1
            if run >= 3:
                anomalies.append(f"{run} consecutive TIMING_FAILURE trades — entry may be too early this week")
                break
        else:
            run = 0

    # Commitment score divergence
    commit_wins  = [r.get("commitment_score", 0.5) for r in records if r.get("outcome") == "WIN"]
    commit_loss  = [r.get("commitment_score", 0.5) for r in records if r.get("outcome") == "LOSS"]
    if commit_wins and commit_loss:
        delta = statistics.mean(commit_wins) - statistics.mean(commit_loss)
        if delta < 0.05:
            anomalies.append(f"Commitment score not separating wins/losses (delta={delta:.3f}) — review gate threshold")

    # Win rate below 30%
    if trades:
        wins = sum(1 for t in trades if t.get("pnl_r", 0) > 0)
        wr = wins / len(trades)
        if wr < 0.30 and len(trades) >= 5:
            anomalies.append(f"Win rate {wr*100:.0f}% below 30% threshold ({len(trades)} trades) — possible regime shift")

    return anomalies


def run_harvest(date: Optional[str] = None, hours: int = 24) -> dict:
    """
    Run the full harvest cycle. Returns daily summary dict.
    Saves to data/oracle/daily_harvest_YYYY_MM_DD.json.
    """
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    trades    = _load_recent_ledger_trades(hours)
    forensics = _load_recent_forensics(hours)

    wins   = [t for t in trades if t.get("pnl_r", 0) > 0]
    losses = [t for t in trades if t.get("pnl_r", 0) <= 0]

    loss_forensics = [r for r in forensics if r.get("outcome") == "LOSS"]
    win_forensics  = [r for r in forensics if r.get("outcome") == "WIN"]

    commit_wins  = [r.get("commitment_score", 0.5) for r in win_forensics]
    commit_loss  = [r.get("commitment_score", 0.5) for r in loss_forensics]

    r_multiples = [t.get("pnl_r", 0) for t in trades]

    summary = {
        "date":             date,
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "trades_closed":    len(trades),
        "wins":             len(wins),
        "losses":           len(losses),
        "win_rate":         round(len(wins) / len(trades), 4) if trades else None,
        "avg_r":            round(statistics.mean(r_multiples), 4) if r_multiples else None,
        "failure_distribution": _compute_failure_distribution(loss_forensics),
        "dominant_failure_mode": _dominant_failure(_compute_failure_distribution(loss_forensics)),
        "avg_commitment_score_wins":   round(statistics.mean(commit_wins), 4) if commit_wins else None,
        "avg_commitment_score_losses": round(statistics.mean(commit_loss), 4) if commit_loss else None,
        "anomalies":        _detect_anomalies(forensics, trades),
        "forensic_records_processed": len(forensics),
        "data_window_hours": hours,
    }

    out_path = HARVEST_DIR / f"daily_harvest_{date}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def load_recent_harvests(days: int = 7) -> list[dict]:
    """Load last N days of harvest summaries for Oracle reflection input."""
    summaries = []
    for i in range(days):
        date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
        path = HARVEST_DIR / f"daily_harvest_{date}.json"
        if path.exists():
            try:
                summaries.append(json.loads(path.read_text()))
            except Exception:
                pass
    return summaries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Oracle harvest cycle")
    parser.add_argument("--date", help="Date to harvest (YYYY-MM-DD), default: today")
    parser.add_argument("--hours", type=int, default=24, help="Rolling window in hours")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    result = run_harvest(date=args.date, hours=args.hours)

    if args.verbose:
        print(json.dumps(result, indent=2))
    else:
        print(f"Harvest {result['date']}: {result['trades_closed']} trades "
              f"| WR={result['win_rate']*100:.0f}% " if result['win_rate'] else
              f"Harvest {result['date']}: no trades in window")
        if result["anomalies"]:
            print("Anomalies:")
            for a in result["anomalies"]:
                print(f"  ⚠ {a}")
