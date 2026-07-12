"""Shared plumbing for research/tsmom_hyp091 (TICK-027, HYP-091).

Path anchors, locked constants (mirrors the hash-locked prereg
data/research/preregister/HYP-091_tsmom_carry_diversification.json), seeded RNG,
and canonical JSON. Everything this study writes stays under
data/research/tsmom_hyp091/ — write-safety is a ticket acceptance criterion.

Reads sovereign READ-ONLY (Sharpe utils, price loader, rate differentials, v015
CSV). Does NOT touch the parallel session's research/tsmom/ or any execution-path
file. Financing uses the CORRECT rate-differential-derived model (operator
decision 2026-07-12), NOT the Colin-gated SWAP_RATES_ANNUAL (TICK-024).
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "research" / "tsmom_hyp091"

# ── Locked spec constants (must match the prereg) ────────────────────────────
PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
# base/quote country codes for sovereign.forex.data_fetcher.get_pair_differentials
PAIR_COUNTRIES = {
    "EURUSD=X": ("EU", "US"),
    "GBPUSD=X": ("UK", "US"),
    "USDJPY=X": ("US", "JP"),
    "AUDUSD=X": ("AU", "US"),
}
LOOKBACK_DAYS = 252         # 12-month momentum
VOL_COM = 60                # ex-ante vol EWMA center-of-mass
TARGET_VOL_ANN = 0.10
LEV_CAP = 2.0
TRADING_YEAR = 252

WARMUP_START = "2013-06-01"   # so the 252d signal exists across the full eval window
EVAL_START = "2015-01-01"
EVAL_END = "2024-12-31"
OOS_START = "2023-01-01"      # adjudicated holdout (favorable rate-trending regime)

SEED = 42
N_PERM = 10_000

SWAP_CALIB_PATH = ROOT / "data" / "research" / "swap_calibration.json"  # TICK-024 output
V015_DECADE_CSV = ROOT / "data" / "proof" / "backtest_trades_v015_2015_2024.csv"

VOLATILE_KEYS = {"generated_at", "run_utc", "env", "runtime_seconds"}


def get_rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def env_record() -> dict:
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "platform": platform.platform()}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def canonical(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: canonical(v) for k, v in sorted(obj.items()) if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [canonical(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 10)
    return obj
