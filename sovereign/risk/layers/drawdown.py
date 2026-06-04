"""Layer 4 — DRAWDOWN MODULATION. Smooth, MONOTONIC taper in [0,1] on drawdown depth.

Deeper drawdown must NEVER produce a larger factor. Uses the deeper of static/trailing drawdown,
interpolated through the config table. Modulator (compounds onto base).
"""
from __future__ import annotations


def factor(signal, state, cfg) -> float:
    d = cfg["drawdown"]
    table = d["table"]                      # [[dd, factor], ...] dd ascending, factor non-increasing
    min_factor = d["min_factor"]
    dd = max(state.drawdown_static, state.drawdown_trailing)

    xs = [row[0] for row in table]
    ys = [row[1] for row in table]
    if dd <= xs[0]:
        return ys[0]
    if dd >= xs[-1]:
        return max(min_factor, ys[-1])
    for i in range(1, len(xs)):
        if dd <= xs[i]:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            f = y0 + (y1 - y0) * (dd - x0) / (x1 - x0)   # linear interp; non-increasing
            return max(min_factor, f)
    return max(min_factor, ys[-1])
