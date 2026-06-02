#!/usr/bin/env python3
"""Canonical FUTURES hypothesis runner — costed IS/OOS, same discipline as run_hypothesis.py.

run_hypothesis.py is forex-only (ForexBacktester + ALL_PAIRS). This is its ES/NQ futures twin:
every futures hypothesis MUST clear the same gates before anything is trusted —
  1. OOS Sharpe > 0
  2. OOS permutation p < 0.05   (entries shuffled to random days at the SAME frequency)
  3. Decay ratio (OOS / IS) >= 0.50
  4. 2025 holdout Sharpe > 0     (survives a third, untouched window)
  5. Benjamini-Hochberg FDR=5% survives across all stored futures p-values

A SURVIVOR is an edge-#2 candidate (uncorrelated with forex macro). A failure is logged
REJECTED — and the NQ/ES lead-lag stays a regime INPUT only, never a trade signal.

Usage:
  PYTHONPATH=. python3 scripts/run_futures_hypothesis.py \\
      --id ESNQ-LL-01 --name "NQ-leads-ES entry" --signal nq_leads_es \\
      [--threshold 0.003] [--traded ES=F] [--perms 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sovereign.futures.futures_backtester import FuturesBacktester, SIGNALS, sharpe_ci

IS_START, IS_END   = "2015-01-01", "2022-12-31"
OOS_START, OOS_END = "2023-01-01", "2024-12-31"
HOLD_START, HOLD_END = "2025-01-01", "2025-12-31"

OUT_DIR = ROOT / "data" / "agent" / "canonical"
LEDGER  = ROOT / "data" / "research" / "futures_ledger.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LEDGER.parent.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _position(signal_fn, bt, start, end, traded, lead="NQ=F"):
    """Materialize the position series the signal produces over [start,end]."""
    df_t = bt.load(traded, start, end)
    df_l = bt.load(lead, start, end)
    if df_t is None or len(df_t) < 30:
        return None
    return signal_fn({traded: df_t, lead: df_l}).reindex(df_t.index).fillna(0.0)


def _sharpe_from_position(bt, pos, start, end, traded):
    """Backtest a fixed position series (used for the permutation null)."""
    fn = lambda frames: pos
    r = bt.run_signal(fn, start, end, traded=traded)
    return r.sharpe if r else 0.0


def _permutation_p(bt, signal_fn, traded, perms, seed=7):
    """Shuffle entries to random days at the SAME frequency; p = P(null OOS Sharpe >= real)."""
    real = bt.run_signal(signal_fn, OOS_START, OOS_END, traded=traded)
    if real is None or real.total_trades < 2:
        return None, None, []
    pos = _position(signal_fn, bt, OOS_START, OOS_END, traded)
    arr = pos.to_numpy().copy()
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(perms):
        shuffled = arr.copy()
        rng.shuffle(shuffled)
        p2 = pos.copy()
        p2.iloc[:] = shuffled
        null.append(_sharpe_from_position(bt, p2, OOS_START, OOS_END, traded))
    null = np.array(null)
    p_value = float((np.sum(null >= real.sharpe) + 1) / (perms + 1))
    return real.sharpe, p_value, null.tolist()


def _walk_forward(bt, signal_fn, traded):
    out = {}
    for yr in range(2021, 2026):
        r = bt.run_signal(signal_fn, f"{yr}-01-01", f"{yr}-12-31", traded=traded)
        out[str(yr)] = round(r.sharpe, 3) if r else None
    return out


def _benjamini_hochberg(pvals, q=0.05):
    if not pvals:
        return True
    ps = sorted(pvals)
    m = len(ps)
    survive = any(p <= (i + 1) / m * q for i, p in enumerate(ps))
    return survive


def _update_ledger(record: dict):
    led = {}
    if LEDGER.exists():
        try:
            led = json.loads(LEDGER.read_text())
        except Exception:
            led = {}
    led.setdefault("hypotheses", {})
    led["hypotheses"][record["id"]] = record
    # BH across all stored OOS p-values.
    pvals = [h["oos_p_value"] for h in led["hypotheses"].values()
             if isinstance(h.get("oos_p_value"), (int, float))]
    led["bh_survives_family"] = _benjamini_hochberg(pvals)
    led["updated_at"] = _now()
    LEDGER.write_text(json.dumps(led, indent=2))
    return led["bh_survives_family"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--signal", required=True, choices=list(SIGNALS.keys()))
    ap.add_argument("--threshold", type=float, default=0.003)
    ap.add_argument("--traded", default="ES=F")
    ap.add_argument("--perms", type=int, default=1000)
    args = ap.parse_args()

    bt = FuturesBacktester()
    signal_fn = SIGNALS[args.signal](args.threshold)

    print(f"\n{'='*64}\n  FUTURES HYPOTHESIS: {args.id} — {args.name}\n"
          f"  signal={args.signal} thr={args.threshold} traded={args.traded}\n{'='*64}")

    is_r  = bt.run_signal(signal_fn, IS_START, IS_END, traded=args.traded)
    oos_r = bt.run_signal(signal_fn, OOS_START, OOS_END, traded=args.traded)
    hold_r = bt.run_signal(signal_fn, HOLD_START, HOLD_END, traded=args.traded)
    if is_r is None or oos_r is None:
        print("  INSUFFICIENT DATA — aborting."); raise SystemExit(1)

    print(f"  IS  ({IS_START}..{IS_END}):  Sharpe {is_r.sharpe:+.3f}  n={is_r.total_trades}  "
          f"WR={is_r.win_rate:.0%}  PF={is_r.profit_factor}")
    print(f"  OOS ({OOS_START}..{OOS_END}): Sharpe {oos_r.sharpe:+.3f}  CI{oos_r.sharpe_ci}  "
          f"n={oos_r.total_trades}  WR={oos_r.win_rate:.0%}")
    print(f"  HOLDOUT 2025: Sharpe {hold_r.sharpe:+.3f} n={hold_r.total_trades}" if hold_r
          else "  HOLDOUT 2025: no data")

    print(f"  Permutation ({args.perms}) on OOS …")
    real_sr, p_value, null = _permutation_p(bt, signal_fn, args.traded, args.perms)
    wf = _walk_forward(bt, signal_fn, args.traded)
    print(f"    OOS Sharpe {real_sr} | p={p_value} | null mean {round(float(np.mean(null)),3) if null else None}")
    print(f"  Walk-forward (yearly Sharpe): {wf}")

    decay = round(oos_r.sharpe / is_r.sharpe, 3) if is_r.sharpe > 0 else None
    hold_sr = hold_r.sharpe if hold_r else None

    gates = {
        "oos_sharpe_positive": oos_r.sharpe > 0,
        "oos_p_lt_005": (p_value is not None and p_value < 0.05),
        "decay_ok": (decay is not None and decay >= 0.50),
        "holdout_positive": (hold_sr is not None and hold_sr > 0),
    }
    record = {
        "id": args.id, "name": args.name, "signal": args.signal, "threshold": args.threshold,
        "traded": args.traded, "date_tested": _now(),
        "is_sharpe": is_r.sharpe, "oos_sharpe": oos_r.sharpe, "oos_ci": oos_r.sharpe_ci,
        "oos_p_value": p_value, "decay_ratio": decay, "holdout_2025_sharpe": hold_sr,
        "walk_forward": wf, "oos_trades": oos_r.total_trades, "gates": gates,
    }
    bh = _update_ledger(record)
    gates["bh_survives"] = bh

    confirmed = all(gates.values())
    verdict = "CONFIRMED" if confirmed else (
        "SUGGESTIVE" if gates["oos_sharpe_positive"] and gates["oos_p_lt_005"] else "REJECTED")
    record["verdict"] = verdict
    _update_ledger(record)
    (OUT_DIR / f"{args.id.lower().replace('-', '_')}.json").write_text(json.dumps(record, indent=2))

    print(f"\n  Gate checks:")
    for k, v in gates.items():
        print(f"    {'✓' if v else '✗'} {k}")
    print(f"\n  VERDICT: {verdict}")
    if verdict != "CONFIRMED":
        print("  → Not a tradeable edge. The NQ/ES lead-lag stays a regime INPUT only.")
    print(f"{'='*64}\n  Saved: data/agent/canonical/{args.id.lower().replace('-','_')}.json + futures_ledger.json\n")


if __name__ == "__main__":
    main()
