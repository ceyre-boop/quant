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
from sovereign.futures.config import futures_params, contract_spec, round_turn_cost_usd, tick_value_usd  # noqa: E402
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


SETUP_LABEL = {"orb": "ORB", "micro": "MICRO", "vwap_mr": "VWAP_MR"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-setup expectancy + permutation gate for the ES/NQ scalp")
    ap.add_argument("--instrument", default="MES", choices=["MES", "MNQ"])
    ap.add_argument("--setup", default="orb", choices=["orb", "micro", "vwap_mr"],
                    help="which strategy to validate IN ISOLATION (never blended)")
    ap.add_argument("--source", default="yf", choices=["yf", "ib"])
    ap.add_argument("--lookback", default="5d")
    ap.add_argument("--bias", default="auto", choices=["auto", "long", "short", "neutral"])
    ap.add_argument("--cvd-gate", action="store_true", help="validate only CVD-confirmed entries")
    ap.add_argument("--min-confluence", type=int, default=0,
                    help="validate only entries with confluence >= this (0–3)")
    args = ap.parse_args()

    v = futures_params()["validation"]
    min_trades = v["min_trades"]
    margin = v["expectancy_margin_ticks"] * tick_value_usd(args.instrument)
    rt_cost = round_turn_cost_usd(args.instrument)
    rng = np.random.default_rng(v["permutation_seed"])

    print(f"Loading {args.instrument} history ({args.source}) for setup '{args.setup}'...", end=" ", flush=True)
    df = bf.load_history(args.instrument, source=args.source, day=None, lookback=args.lookback)
    if df is None or len(df) == 0:
        print("\n  No data. Try --source ib for deeper history.")
        sys.exit(1)
    print(f"done. {len(df)} bars.")

    setups = {args.setup}
    real_nets: list[float] = []
    eligible: list[tuple] = []      # (day_df, i, dir) for the permutation null
    prior_close = None
    prior_day_df = None
    for day in bf.session_days(df):
        day_df = df[df.index.tz_convert(bf.ET).strftime("%Y-%m-%d") == day]
        if len(day_df) < 5:
            prior_close = float(day_df["Close"].iloc[-1]) if len(day_df) else prior_close
            continue
        bias_dir, key_levels = fr._day_bias(day_df, day, prior_close, args.instrument, args.bias)
        prior_close = float(day_df["Close"].iloc[-1])
        from sovereign.futures import volume_profile as _vp
        prior_profile = _vp.compute_profile(prior_day_df) if prior_day_df is not None else None
        sess = fr.simulate_session(day_df, day, bias_dir, key_levels, args.instrument,
                                   "safe", setups=setups, cvd_gate=args.cvd_gate,
                                   prior_profile=prior_profile)
        for t in sess["trades"]:
            if t.get("confluence", 0) >= args.min_confluence:
                real_nets.append(t["net_usd"])
        d = bias_dir if bias_dir in ("LONG", "SHORT") else "LONG"
        for i in range(2, len(day_df) - 2):
            eligible.append((day_df, i, d))
        prior_day_df = day_df

    n_real = len(real_nets)
    arr = np.array(real_nets) if n_real else np.array([0.0])
    wins = arr[arr > 0]; losses = arr[arr <= 0]
    win_rate = float(len(wins) / n_real) if n_real else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0   # <= 0
    real_mean = float(arr.mean()) if n_real else 0.0

    # permutation p-value (generic random-entry baseline)
    p_value = 1.0
    null_mean = 0.0
    if n_real and len(eligible) >= n_real:
        iters = v["permutation_iterations"]
        null_means = np.empty(iters)
        idx = np.arange(len(eligible))
        for k in range(iters):
            pick = rng.choice(idx, size=n_real, replace=False)
            null_means[k] = np.mean([_sim_trade(eligible[j][0], eligible[j][1], eligible[j][2],
                                                args.instrument) for j in pick])
        p_value = float(np.mean(null_means >= real_mean))
        null_mean = float(null_means.mean())

    # ── the gate: ALL must hold ──
    adequate = n_real >= min_trades
    expectancy_ok = real_mean > margin
    winrate_ok = win_rate * avg_win > (1 - win_rate) * abs(avg_loss)
    perm_ok = p_value < v["p_value_threshold"]
    passed = adequate and expectancy_ok and winrate_ok and perm_ok

    G, R, Y, BD, RS = "\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m"
    ok = lambda b: f"{G}✓{RS}" if b else f"{R}✗{RS}"
    print(f"\n{BD}{'═'*62}{RS}")
    print(f"  {BD}VALIDATION — {args.instrument} · setup '{args.setup}'{RS}")
    print(f"{BD}{'═'*62}{RS}")
    print(f"  Trades: {n_real}    Win rate: {win_rate:.0%}    "
          f"avg win ${avg_win:+.2f}  avg loss ${avg_loss:+.2f}")
    print(f"  Mean net/trade: ${real_mean:+.2f}   (RT cost ${rt_cost:.2f}; margin bar ${margin:.2f})")
    print(f"  Random-entry mean: ${null_mean:+.2f}    permutation p: {p_value:.4f}")
    print(f"\n  Gate:")
    print(f"    {ok(adequate)} n ≥ {min_trades} (have {n_real})")
    print(f"    {ok(expectancy_ok)} mean net > {v['expectancy_margin_ticks']} tick (${margin:.2f})")
    print(f"    {ok(winrate_ok)} win%·avg_win > (1−win%)·avg_loss")
    print(f"    {ok(perm_ok)} beats random at p < {v['p_value_threshold']}")
    g = G if passed else R
    msg = ("PASS — edge clears costs; safe to trust" if passed
           else "FAIL — do NOT fund / do NOT trust --auto")
    print(f"\n  {BD}{g}{msg}{RS}")
    if not adequate:
        print(f"  {Y}Underpowered: a {n_real}-trade result is noise. Accumulate IB/paper history "
              f"to n≥{min_trades} before believing any verdict.{RS}")
    print(f"{BD}{'═'*62}{RS}\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "instrument": args.instrument, "setup": args.setup, "source": args.source,
        "n_real": n_real, "win_rate": round(win_rate, 3),
        "avg_win_usd": round(avg_win, 2), "avg_loss_usd": round(avg_loss, 2),
        "real_mean_net_usd": round(real_mean, 4), "margin_usd": round(margin, 2),
        "random_mean_net_usd": round(null_mean, 4), "p_value": round(p_value, 4),
        "adequate": adequate, "expectancy_ok": expectancy_ok, "winrate_ok": winrate_ok,
        "perm_ok": perm_ok, "passed": passed,
    }, indent=2))
    print(f"  → {OUT.relative_to(ROOT)}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
