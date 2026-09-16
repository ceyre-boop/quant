"""TimesFM 3.0 (MLX) forecasts, cached. Sigma from the model's own quantiles: (q0.9 − q0.1) / (2·1.2816)."""
from __future__ import annotations

import numpy as np

_F = None
Z90 = 1.2815515655


def model():
    global _F
    if _F is None:
        from timesfm3.mlx import TimesFM3Forecaster
        _F = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")
    return _F


def forecast(ctxs: list[np.ndarray], horizon: int, batch: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Returns (point[n, h], sigma[n, h]) with sigma from the 10–90 quantile spread."""
    f = model(); pts, sig = [], []
    for i in range(0, len(ctxs), batch):
        outs = list(f.predict_batch(ctxs[i: i + batch], horizon=horizon, return_quantiles=True, use_symmetric_averaging=False))
        for o in outs:
            q = np.asarray(o.quantiles); pts.append(np.asarray(o.forecast)); sig.append((q[:, 8] - q[:, 0]) / (2 * Z90))
    return np.array(pts), np.array(sig)
