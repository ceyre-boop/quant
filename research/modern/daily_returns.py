"""P2: trade -> daily mark-to-market decomposition with CAUSAL costs (HYP-090).

Locks (prereg): raw daily M2M for trade (entry e at open[e], exit x at close[x]):
  day e:        d * (close[e] - open[e]) / open[e]
  day s in (e,x]: d * (close[s] - close[s-1]) / open[e]
Sum telescopes EXACTLY to the kernel's pnl_pct = d*(close[x]/open[e] - 1) (1e-12
test). Every exit prices at the decision bar's CLOSE (verified in
_simulate_forex_core:108), so no stop-fill residual exists.

Causal costs: spread+slippage fraction on the ENTRY day (known at entry); swap
accrued DAILY at 1.4 x annual/365 (signed) per held day after entry — exit-day
allocation would leak final hold length into trailing windows. Costed totals
match ForexBacktester._apply_costs within the swap-rounding tolerance
(hold%5 != 0 -> <= 0.4 swap-days). Calibrated slippage is resolved ONCE at build
and recorded (frozen), matching what the canonical run used.

Output: daily_returns.npz — costed float32 (385 x 4 x n_days) + raw + open mask
+ the shared union calendar. NOT committed to git (manifest-pinned sha256;
reproducible from frozen inputs).

Run: python3 -m research.modern.daily_returns
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from research.modern._lib import OUT_DIR, gate_zero, sha256_file, write_json
from research.modern.signal_arrays import PAIRS

SWAP_UPLIFT = 1.4          # daily accrual of the hold + hold//5*2 weekend approximation


def _cost_constants(pair: str) -> tuple[float, float]:
    from sovereign.forex.forex_backtester import (
        SPREAD_COST, SLIPPAGE_PER_SIDE, SWAP_RATES_ANNUAL,
        _DEFAULT_SPREAD, _DEFAULT_SWAP, _calibrated_slippage,
    )
    spread = SPREAD_COST.get(pair, _DEFAULT_SPREAD)
    slip = _calibrated_slippage(pair)
    if slip is None:
        slip = SLIPPAGE_PER_SIDE
    cost_price = spread + 2 * slip
    return cost_price, SWAP_RATES_ANNUAL.get(pair, _DEFAULT_SWAP)


def build(trades_file: str = "trades.parquet", signals_file: str = "signals.npz",
          out_file: str = "daily_returns.npz") -> Path:
    gate_zero()
    trades = pd.read_parquet(OUT_DIR / trades_file)
    npz = np.load(OUT_DIR / signals_file)

    closes, opens, indices = {}, {}, {}
    for pair in PAIRS:
        key = pair.replace("=X", "")
        closes[pair] = npz[f"{key}__close"]
        opens[pair] = npz[f"{key}__open"]
        indices[pair] = pd.DatetimeIndex(npz[f"{key}__index"].astype("datetime64[ns]"))

    union = indices[PAIRS[0]]
    for pair in PAIRS[1:]:
        union = union.union(indices[pair])
    n_days = len(union)
    pos_in_union = {pair: union.get_indexer(indices[pair]) for pair in PAIRS}

    n_cfg = int(trades["config_id"].max()) + 1
    raw = np.zeros((n_cfg, len(PAIRS), n_days), dtype=np.float64)
    costed = np.zeros_like(raw)
    open_mask = np.zeros((n_cfg, len(PAIRS)), dtype=bool)
    cost_meta = {}

    for pi, pair in enumerate(PAIRS):
        cost_price, swap_tbl = _cost_constants(pair)
        cost_meta[pair] = {"cost_price": cost_price, "swap_annual": swap_tbl}
        c, o, u = closes[pair], opens[pair], pos_in_union[pair]
        sub = trades[trades["pair"] == pair]
        for t in sub.itertuples(index=False):
            e, x, d = int(t.entry_idx), int(t.exit_idx), int(t.direction)
            entry = max(float(t.entry), 1e-9)
            cid = int(t.config_id)
            if t.is_open:
                open_mask[cid, pi] = True
            # raw daily rows
            raw[cid, pi, u[e]] += d * (c[e] - o[e]) / entry
            costed[cid, pi, u[e]] += d * (c[e] - o[e]) / entry - cost_price / entry
            side = "LONG" if d >= 0 else "SHORT"
            daily_swap = (swap_tbl[side] / 365.0) * SWAP_UPLIFT
            for s in range(e + 1, x + 1):
                r = d * (c[s] - c[s - 1]) / entry
                raw[cid, pi, u[s]] += r
                costed[cid, pi, u[s]] += r + daily_swap

    out = OUT_DIR / out_file
    np.savez_compressed(
        out,
        costed=costed.astype(np.float32), raw=raw.astype(np.float32),
        open_mask=open_mask, union_index=union.astype("int64").to_numpy(),
        pairs=np.array([p.encode() for p in PAIRS]),
    )
    write_json(OUT_DIR / (out_file + ".meta.json"), {
        "sha256": sha256_file(out), "shape": list(costed.shape),
        "cost_constants": cost_meta, "swap_uplift": SWAP_UPLIFT,
        "trades_file": trades_file, "n_open_tail": int(open_mask.sum()),
    })
    print(f"daily returns: {costed.shape} -> {out} "
          f"({out.stat().st_size/1e6:.1f} MB, {int(open_mask.sum())} open tails)")
    return out


if __name__ == "__main__":
    build()
