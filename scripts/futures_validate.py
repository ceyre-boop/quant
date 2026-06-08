#!/usr/bin/env python3
"""
Validation gate for the ES/NQ scalp — does the edge beat random AND clear costs?

Mirrors scripts/permutation_test_ict.py's discipline for the futures sandbox:
the scalp's REAL entries are compared against random entries drawn from the same
eligible bar pool (same bias direction), each simulated with the same exit logic
and the REAL round-turn cost. If the scalp's mean net P&L is not better than random
at p < threshold, the "edge" is noise — do NOT trust --auto and do NOT fund.

Gate (config/futures_params.yml::validation): p < p_value_threshold AND mean net > 0.

Usage:
    python3.13 scripts/futures_validate.py --instrument MES --source yf --lookback 5d
    python3.13 scripts/futures_validate.py --instrument MNQ --source ib
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sovereign.futures import scalp_strategy as strat                      # noqa: E402
from sovereign.futures import bar_feed as bf                               # noqa: E402
from sovereign.futures.config import futures_params, contract_spec, round_turn_cost_usd  # noqa: E402
import futures_replay as fr                                                # noqa: E402

OUT = ROOT / "data" / "research" / "futures_validation.json"


def _sim_trade(day_df, i: int, direction: str, instrument: str) -> float:
    """Forward-simulate a trade entered at bar i; return net $ after one round-turn cost."""
    dpp = contract_spec(instrument)["dollars_per_point"]
    entry = float(day_df["Close"].iloc[i])
    stop = strat.compute_stop(direction, entry, None, None)        # fallback-% stop (fair to all)
    target = strat.target_from_rr(direction, entry, stop)
    exit_price = float(day_df["Close"].iloc[-1])                   # default EOD
    for j in range(i + 1, len(day_df)):
        hi, lo = float(day_df["High"].iloc[j]), float(day_df["Low"].iloc[j])
        if direction == "LONG":
            if lo <= stop:
                exit_price = stop; break
            if hi >= target:
                exit_price = target; break
        else:
            if hi >= stop:
                exit_price = stop; break
            if lo <= target:
                exit_price = target; break
    pts = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
    return pts * dpp - round_turn_cost_usd(instrument)


def main() -> None:
    ap = argparse.ArgumentParser(description="Permutation validation gate for the ES/NQ scalp")
    ap.add_argument("--instrument", default="MES", choices=["MES", "MNQ"])
    ap.add_argument("--source", default="yf", choices=["yf", "ib"])
    ap.add_argument("--lookback", default="5d")
    ap.add_argument("--bias", default="auto", choices=["auto", "long", "short", "neutral"])
    args = ap.parse_args()

    v = futures_params()["validation"]
    rng = np.random.default_rng(v["permutation_seed"])

    print(f"Loading {args.instrument} history ({args.source})...", end=" ", flush=True)
    df = bf.load_history(args.instrument, source=args.source, lookback=args.lookback)
    if df is None or len(df) == 0:
        print("\n  No data. Try --source ib for deeper history.")
        sys.exit(1)
    print(f"done. {len(df)} bars.")

    days = bf.session_days(df)
    real_nets: list[float] = []
    eligible: list[tuple] = []      # (day_df, i, bias_dir)
    prior_close = None

    for day in days:
        day_df = df[df.index.tz_convert(bf.ET).strftime("%Y-%m-%d") == day]
        if len(day_df) < 5:
            prior_close = float(day_df["Close"].iloc[-1]) if len(day_df) else prior_close
            continue
        bias_dir, key_levels = fr._day_bias(day_df, day, prior_close, args.instrument, args.bias)
        prior_close = float(day_df["Close"].iloc[-1])
        if bias_dir not in ("LONG", "SHORT"):
            continue
        # real scalp trades for this session (net $)
        sess = fr.simulate_session(day_df, day, bias_dir, key_levels, args.instrument, "safe")
        real_nets.extend(t["net_usd"] for t in sess["trades"] if t["setup"] == "MICRO")
        # eligible entry pool: any bar that leaves room to resolve
        for i in range(2, len(day_df) - 2):
            eligible.append((day_df, i, bias_dir))

    n_real = len(real_nets)
    if n_real == 0 or len(eligible) < n_real:
        print(f"\n  Not enough data: {n_real} real trades, {len(eligible)} eligible bars.")
        print("  Run again after accumulating IB/Databento history or paper sessions.")
        sys.exit(1)

    real_mean = float(np.mean(real_nets))
    iters = v["permutation_iterations"]
    null_means = np.empty(iters)
    idx = np.arange(len(eligible))
    for k in range(iters):
        pick = rng.choice(idx, size=n_real, replace=False)
        null_means[k] = np.mean([_sim_trade(*eligible[j]) for j in pick])
    p_value = float(np.mean(null_means >= real_mean))

    passed = (p_value < v["p_value_threshold"]) and (real_mean > 0)
    adequate = n_real >= 30

    G, R, Y, BD, RS = "\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m"
    print(f"\n{BD}{'═'*60}{RS}")
    print(f"  {BD}SCALP VALIDATION — {args.instrument}{RS}")
    print(f"{BD}{'═'*60}{RS}")
    print(f"  Real trades:        {n_real}" + ("" if adequate else f"  {Y}(< 30 — underpowered){RS}"))
    print(f"  Real mean net/trade: ${real_mean:+.2f}  (cost ${round_turn_cost_usd(args.instrument):.2f}/RT)")
    print(f"  Random mean net:     ${float(np.mean(null_means)):+.2f}")
    print(f"  p-value:             {p_value:.4f}  (threshold {v['p_value_threshold']})")
    g = (G if passed else R)
    print(f"\n  {BD}Gate: {g}{'PASS — edge beats random & clears costs' if passed else 'FAIL — do not fund / trust --auto'}{RS}")
    if not adequate:
        print(f"  {Y}Note: result is provisional until n_real >= 30.{RS}")
    print(f"{BD}{'═'*60}{RS}\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "instrument": args.instrument, "source": args.source,
        "n_real": n_real, "real_mean_net_usd": round(real_mean, 4),
        "random_mean_net_usd": round(float(np.mean(null_means)), 4),
        "p_value": round(p_value, 4), "threshold": v["p_value_threshold"],
        "sample_adequate": adequate, "passed": passed,
    }, indent=2))
    print(f"  → {OUT.relative_to(ROOT)}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
