"""P1: the A0 reconcile ABORT gate (HYP-090).

Runs the canonical ForexBacktester(2015-01-01, 2024-12-31).backtest_all() and
requires weighted_portfolio_sharpe within 0.6886 +/- 0.01 (exit_policy_evolution
convention). Out of band -> SystemExit, study halts, escalate to operator — the
band is NEVER re-tuned after data.

Side effects handled: backtest_all overwrites logs/forex_backtest_{results,trades}.json.
We back both up first, snapshot OUR run's trades for the P2 config-385 parity test,
then restore the originals (HYP-067 pattern).

Run: python3 -m research.modern.reconcile
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.modern._lib import OUT_DIR, RECON_TARGET, RECON_TOL, ROOT, gate_zero, write_json

RESULTS_PATH = ROOT / "logs" / "forex_backtest_results.json"
TRADES_PATH = ROOT / "logs" / "forex_backtest_trades.json"
SNAPSHOT = OUT_DIR / "reconcile_snapshot_trades.json"
REPORT = OUT_DIR / "reconcile_report.json"


def run_reconcile() -> float:
    gate_zero()
    backups = {}
    for p in (RESULTS_PATH, TRADES_PATH):
        if p.exists():
            backups[p] = p.with_suffix(".hyp090-backup")
            shutil.copy2(p, backups[p])
    try:
        from sovereign.forex.forex_backtester import ForexBacktester
        from sovereign.reporting.equity_curve import weighted_portfolio_sharpe

        bt = ForexBacktester(start="2015-01-01", end="2024-12-31")
        results = bt.backtest_all()
        ws = weighted_portfolio_sharpe([(r.sharpe, r.total_trades) for r in results])

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        if TRADES_PATH.exists():
            shutil.copy2(TRADES_PATH, SNAPSHOT)      # config-385 parity target (P2)
    finally:
        for orig, bak in backups.items():
            shutil.move(str(bak), str(orig))          # restore originals

    write_json(REPORT, {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weighted_portfolio_sharpe": round(ws, 6),
        "target": RECON_TARGET, "tol": RECON_TOL,
        "window": "2015-01-01..2024-12-31",
        "per_pair": [{"pair": r.pair, "sharpe": round(r.sharpe, 4),
                      "trades": r.total_trades} for r in results],
        "in_band": bool(abs(ws - RECON_TARGET) <= RECON_TOL),
    })

    if abs(ws - RECON_TARGET) > RECON_TOL:
        raise SystemExit(
            f"RECONCILE ABORT: weighted portfolio Sharpe {ws:.4f} outside "
            f"{RECON_TARGET}±{RECON_TOL}. Canonical data has drifted — the study HALTS "
            f"here by design. Escalate to operator; NEVER re-tune the band.")
    print(f"RECONCILE OK: {ws:.4f} within {RECON_TARGET}±{RECON_TOL}; "
          f"snapshot -> {SNAPSHOT}")
    return ws


if __name__ == "__main__":
    run_reconcile()
