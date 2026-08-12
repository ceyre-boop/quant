"""P2: the 1,540 (config x pair) decade kernel runs (HYP-090).

Grid: ids 0..383 = product(theta x hold x stop x trail x gate) in canonical
nesting order; id 384 = v015-exact incumbent (per-pair trailing overrides).

Open-tail recovery (kernel drops the final still-open position): every run is
PADDED with 200 synthetic flat bars (price frozen at last close, signal 0) so an
open position time-exits inside the padding and is emitted as a trade whose
exit_idx lands beyond the real span -> flagged is_open, M2M rows truncated to
the real span. Padding provably cannot change closed trades (flat prices trigger
no stop/trail; test_padding_invariance verifies).

Run: python3 -m research.modern.precompute [--parity]
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from research.modern._lib import OUT_DIR, gate_zero, write_json
from research.modern.signal_arrays import HOLDS, PAIRS, THETAS

STOPS = [1.5, 2.0, 2.5]
TRAILS = [1.0, 1.25, 1.5, 2.0]
GATES = ["on", "off"]
PAD = 200
V015_TRAILING = {"GBPUSD=X": 2.0, "AUDUSD=X": 1.0, "EURUSD=X": 1.25, "USDJPY=X": 1.25}
V015_ID = 384


def config_manifest() -> list[dict]:
    configs = []
    for ti, hi, stop, trail, gate in itertools.product(
            range(len(THETAS)), range(len(HOLDS)), STOPS, TRAILS, GATES):
        configs.append({"config_id": len(configs), "theta": THETAS[ti], "theta_idx": ti,
                        "hold": HOLDS[hi], "hold_idx": hi, "stop_atr_mult": stop,
                        "trailing": trail, "gate": gate})
    configs.append({"config_id": V015_ID, "theta": 0.15, "theta_idx": 1, "hold": 60,
                    "hold_idx": 2, "stop_atr_mult": 2.0, "trailing": "per_pair",
                    "gate": "on", "note": "v015-exact incumbent"})
    assert len(configs) == 385
    return configs


def run_config_pair(npz, cfg: dict, pair: str) -> list[dict]:
    from sovereign.forex.fast_backtester import simulate_forex_trades_arrays

    key = pair.replace("=X", "")
    closes = npz[f"{key}__close"]
    opens = npz[f"{key}__open"]
    atr = npz[f"{key}__atr_pct"]
    sig = npz[f"{key}__sig_t{cfg['theta_idx']}_h{cfg['hold_idx']}"].copy()
    hold = npz[f"{key}__hold_t{cfg['theta_idx']}_h{cfg['hold_idx']}"]
    if cfg["gate"] == "on":
        sig[npz[f"{key}__gate_mask"]] = 0

    n_real = len(closes)
    pad_price = np.full(PAD, closes[-1])
    closes_p = np.concatenate([closes, pad_price])
    opens_p = np.concatenate([opens, pad_price])
    sig_p = np.concatenate([sig, np.zeros(PAD, dtype=sig.dtype)])
    hold_p = np.concatenate([hold, np.full(PAD, 60, dtype=hold.dtype)])
    atr_p = np.concatenate([atr, np.full(PAD, max(float(atr[-1]), 1e-6))])

    trailing = (V015_TRAILING.get(pair, 1.25) if cfg["trailing"] == "per_pair"
                else float(cfg["trailing"]))
    trades = simulate_forex_trades_arrays(
        opens=opens_p, closes=closes_p, signals=sig_p, hold_days=hold_p,
        stop_pct=0.04, index=None, atr_pcts=atr_p,
        stop_atr_mult=float(cfg["stop_atr_mult"]), trailing_atr_mult=trailing,
        strict_mode=False, donchian_exit_days=10, allow_pyramiding=False,
        max_pyramid_units=1, risk_pct=0.01, max_risk_pct=0.01, enable_cb_refresh=True,
    )
    out = []
    for t in trades:
        e, x = int(t["entry_date"]), int(t["exit_date"])   # index=None -> integer positions
        if e >= n_real:
            continue                                        # phantom entry inside padding
        is_open = x >= n_real
        out.append({"config_id": cfg["config_id"], "pair": pair, "entry_idx": e,
                    "exit_idx": min(x, n_real - 1), "is_open": is_open,
                    "direction": t["direction"], "entry": t["entry"],
                    "pnl_pct_raw": (np.nan if is_open else t["pnl_pct"]),
                    "hold_days": t["hold_days"], "exit_reason": ("open_tail" if is_open else t["exit_reason"])})
    return out


def main(signals_file: str = "signals.npz", out_prefix: str = "") -> None:
    gate_zero()
    npz = np.load(OUT_DIR / signals_file)
    configs = config_manifest()
    t0 = time.time()
    rows = []
    for i, cfg in enumerate(configs):
        for pair in PAIRS:
            rows.extend(run_config_pair(npz, cfg, pair))
        if i % 50 == 0:
            print(f"  config {i}/385 ({time.time()-t0:.0f}s, {len(rows)} trades)")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / f"{out_prefix}trades.parquet")
    write_json(OUT_DIR / f"{out_prefix}config_manifest.json",
               {"configs": configs, "n_trades": len(df),
                "signals_file": signals_file, "pad": PAD})
    print(f"done: {len(df)} trades ({int(df.is_open.sum())} open-tail) in "
          f"{time.time()-t0:.0f}s -> {out_prefix}trades.parquet")


if __name__ == "__main__":
    if "--parity" in sys.argv:
        main("signals_parity_2015_2024.npz", "parity_")
    else:
        main()
