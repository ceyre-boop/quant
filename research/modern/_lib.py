"""Shared plumbing for research/modern (HYP-090 / TICK-023).

Paths, cross-process-stable seeds (crc32 — never builtin hash(); TICK-022 salt
bug), canonical JSON, the gate-zero prereg guard, daily-Sharpe helpers, and the
stationary block bootstrap (L=5) used for arm-vs-baseline p-values —
sovereign.discovery.gate.bootstrap_sharpe_diff_pvalue resamples i.i.d., which is
wrong for autocorrelated daily mark-to-market series.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = ROOT / "data" / "research" / "modern"
CACHE_DIR = OUT_DIR / "spot_cache"
CHARTS_DIR = OUT_DIR / "charts"
PREREG_PATH = ROOT / "data" / "research" / "preregister" / "HYP-090_modern_adaptive_params.json"
LEDGER_PATH = ROOT / "data" / "agent" / "hypothesis_ledger.json"

HYP_ID = "HYP-090"

# Locked constants (mirrors of the prereg — the prereg is the law; these are the code's copy)
RECON_TARGET, RECON_TOL = 0.6886, 0.01          # exit_policy_evolution.py:104 convention
BLOCK_L = 5                                      # block_length.compute_atr_block_length locked floor
N_BOOT = 10_000
BOOT_SEED = 42
TRADING_DAYS = 252.0

VOLATILE_KEYS = {"generated_at", "env", "runtime_seconds"}


def seed_from(*parts) -> np.random.Generator:
    """Deterministic, cross-process-stable RNG from string/number parts."""
    ints = [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(ints))


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
        return round(obj, 12)
    return obj


def canonical_dumps(obj: Any) -> str:
    return json.dumps(canonical(obj), sort_keys=True)


def canonical_hash(doc: dict) -> str:
    """Prereg hash convention — copied from preregister_positioning.py:221 (NOT imported)."""
    body = {k: v for k, v in doc.items() if k != "hash_lock"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_dtindex(int_arr) -> "pd.DatetimeIndex":
    """Rebuild a DatetimeIndex from stored int64 epochs, unit-safe: parquet
    round-trips can yield ms-resolution indices (astype int64 -> ms), while
    native pandas gives ns. Detect by magnitude (ns since 1970 > 1e16)."""
    import pandas as pd
    a = np.asarray(int_arr, dtype="int64")
    unit = "ns" if (a.size and int(a.max()) > int(1e16)) else "ms"
    return pd.DatetimeIndex(a.astype(f"datetime64[{unit}]").astype("datetime64[ns]"))


def gate_zero() -> dict:
    """Verify the prereg hash-lock AND the ledger PREREGISTERED entry BEFORE any
    data is read (run_positioning_family.gate_zero pattern). SystemExit on any
    mismatch — the V2 ledger-gap failure class dies here."""
    if not PREREG_PATH.exists():
        raise SystemExit(f"GATE-ZERO FAIL: prereg missing: {PREREG_PATH} — run "
                         f"preregister_hyp090.py BEFORE any data touch")
    doc = json.loads(PREREG_PATH.read_text())
    stored = doc.get("hash_lock")
    if stored != canonical_hash(doc):
        raise SystemExit("GATE-ZERO FAIL: prereg hash mismatch — the locked document changed")
    ledger = json.loads(LEDGER_PATH.read_text())
    entry = next((e for e in ledger if e.get("id") == HYP_ID), None)
    if entry is None:
        raise SystemExit(f"GATE-ZERO FAIL: {HYP_ID} missing from hypothesis ledger")
    if entry.get("status") != "PREREGISTERED":
        raise SystemExit(f"GATE-ZERO FAIL: ledger status {entry.get('status')!r} != PREREGISTERED")
    if entry.get("hash_lock") != stored:
        raise SystemExit("GATE-ZERO FAIL: ledger hash_lock != prereg hash_lock")
    return doc


# ── Sharpe + block bootstrap ────────────────────────────────────────────────

def daily_sharpe(returns: np.ndarray) -> float:
    """Annualized (√252) Sharpe of a daily return series; 0 for degenerate."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    s = r.std(ddof=1)
    if s == 0 or not np.isfinite(s):
        return 0.0
    return float(r.mean() / s * np.sqrt(TRADING_DAYS))


def _stationary_block_indices(rng: np.random.Generator, n: int, mean_len: int) -> np.ndarray:
    """Politis–Romano stationary bootstrap indices (wrap-around, geometric lengths)."""
    idx = np.empty(n, dtype=np.int64)
    p = 1.0 / mean_len
    pos = 0
    while pos < n:
        start = int(rng.integers(0, n))
        length = int(rng.geometric(p))
        length = min(length, n - pos)
        block = (start + np.arange(length)) % n
        idx[pos:pos + length] = block
        pos += length
    return idx


def block_bootstrap_sharpe_diff_p(a: np.ndarray, b: np.ndarray, *,
                                  mean_block: int = BLOCK_L, n_boot: int = N_BOOT,
                                  seed: int = BOOT_SEED) -> dict:
    """One-sided p for H1: Sharpe(a) > Sharpe(b), paired daily series.

    Resamples PAIRED days with stationary blocks (preserving cross-series
    correlation and short-range autocorrelation), builds the bootstrap
    distribution of the Sharpe difference, p = P(diff* <= 0) with the
    (n_le + 1)/(N + 1) convention.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired series required"
    n = a.size
    rng = np.random.default_rng(seed)
    d_obs = daily_sharpe(a) - daily_sharpe(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = _stationary_block_indices(rng, n, mean_block)
        diffs[i] = daily_sharpe(a[idx]) - daily_sharpe(b[idx])
    n_le = int(np.sum(diffs <= 0.0))
    return {"d_obs": round(d_obs, 6), "p_one_sided": (n_le + 1) / (n_boot + 1),
            "boot_mean": round(float(diffs.mean()), 6),
            "boot_p5": round(float(np.percentile(diffs, 5)), 6),
            "n_boot": n_boot, "mean_block": mean_block, "seed": seed}
