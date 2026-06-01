"""
Rolling walk-forward — forex macro system.
==========================================

Replaces the single 2015-2022 / 2023-2024 split with 4 expanding windows whose
test years are 2021, 2022, 2023, 2024. The honest, costed, √n-weighted Sharpe must
hold across all four.

HONEST FRAMING: this system has NO per-window parameter *fitting* step — params are
hardcoded constants. So "train" here is just the expanding prior window shown for a
decay reference; this is fundamentally a multi-window OUT-OF-SAMPLE STABILITY test,
not a true train/test walk-forward. The value is: does the edge survive in every
disjoint test year, or only in the favorable ones?

Reuses ForexBacktester + the Phase-1 √n weighting and CI helpers from
holdout_validation_v014.

Usage:  python3 scripts/rolling_walkforward.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
for lib in ("yfinance", "peewee", "urllib3", "requests"):
    logging.getLogger(lib).setLevel(logging.ERROR)

from sovereign.forex.forex_backtester import ForexBacktester
from scripts.holdout_validation_v014 import _sharpe_from_results, _total_trades, sharpe_ci

OUT_PATH = ROOT / "data" / "research" / "rolling_walkforward_results.json"

WINDOWS = [
    {"label": "2021", "train": ("2015-01-01", "2020-12-31"), "test": ("2021-01-01", "2021-12-31")},
    {"label": "2022", "train": ("2015-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")},
    {"label": "2023", "train": ("2015-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    {"label": "2024", "train": ("2015-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
]


def _run(start: str, end: str, pair_vix_gates: dict | None = None):
    bt = ForexBacktester(start=start, end=end)
    if pair_vix_gates is not None:
        gates = dict(bt.PAIR_VIX_GATES)
        gates.update(pair_vix_gates)
        bt.PAIR_VIX_GATES = gates
    results = bt.backtest_all()
    return _sharpe_from_results(results), _total_trades(results)


def run_walkforward(pair_vix_gates: dict | None = None, save: bool = True, verbose: bool = True) -> dict:
    """Rolling walk-forward. Optional pair_vix_gates injects a parameter delta
    (e.g. {'USDJPY=X': 13.0}) so EdgePipeline can test a candidate before committing.
    Returns the result dict (also saved to data/research/ when save=True and no override)."""
    if verbose:
        print("\n" + "=" * 60)
        print("ROLLING WALK-FORWARD — forex macro (costed, √n-weighted)")
        print("Multi-window OOS stability test (params are static, not fit per window)")
        print("=" * 60)
    rows = []
    for w in WINDOWS:
        if verbose:
            print(f"\n  [{w['label']}] test {w['test'][0]}→{w['test'][1]}")
        train_sharpe, train_n = _run(*w["train"], pair_vix_gates=pair_vix_gates)
        test_sharpe, test_n = _run(*w["test"], pair_vix_gates=pair_vix_gates)
        ci_low, ci_high, se = sharpe_ci(test_sharpe, test_n)
        decay = round(test_sharpe / train_sharpe, 3) if train_sharpe > 0 else None
        rows.append({
            "test_year": w["label"],
            "train_sharpe": round(train_sharpe, 3), "train_n": train_n,
            "test_sharpe": round(test_sharpe, 3), "test_n": test_n,
            "test_ci_95": [ci_low, ci_high], "decay_ratio": decay,
        })
        if verbose:
            print(f"    train={train_sharpe:.3f} (n={train_n})  test={test_sharpe:.3f} "
                  f"(n={test_n}, CI [{ci_low:+.3f},{ci_high:+.3f}])  decay={decay}")

    test_sharpes = [r["test_sharpe"] for r in rows]
    avg_test = round(sum(test_sharpes) / len(test_sharpes), 3)
    min_test = min(test_sharpes)
    all_positive = all(s > 0 for s in test_sharpes)
    verdict = "ROBUST" if (all_positive and min_test > 0.3) else "FRAGILE"

    if verbose:
        print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
        print(f"  Test Sharpes by year: {test_sharpes}")
        print(f"  Avg OOS Sharpe: {avg_test}  Min: {min_test}  All positive: {all_positive}")
        print(f"  VERDICT: {verdict}")

    result = {
        "test_date": datetime.now(timezone.utc).isoformat(),
        "framing": "multi-window OOS stability (no per-window parameter fitting)",
        "pair_vix_gates_override": pair_vix_gates,
        "windows": rows,
        "avg_oos_sharpe": avg_test,
        "min_oos_sharpe": min_test,
        "all_positive": all_positive,
        "verdict": verdict,
    }
    if save and pair_vix_gates is None:   # don't overwrite the canonical file with an override run
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(result, indent=2))
        if verbose:
            print(f"  Saved: {OUT_PATH.relative_to(ROOT)}")
    return result


def main():
    run_walkforward()


if __name__ == "__main__":
    main()
