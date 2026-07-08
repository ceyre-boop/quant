#!/usr/bin/env python3
"""scripts/bench_throughput.py — BENCH-THROUGHPUT.

Measure how many backtests/second the system ACTUALLY does — replacing the
never-measured "148,193 backtests/sec" claim with a documented, re-runnable number.

It measures, on this machine, a matrix of:
  - KERNELS:  the FastBacktestEngine (backtest/fast_engine.py — Numba @njit IF numba
              is installed, else a transparent pure-Python fallback) vs the pure-Python
              forex kernel (sovereign/forex/fast_backtester.py).
  - DATA TIERS ("better data"): a 90-bar window (the legacy regime), daily, 5-min,
              and 1-min NQ (data/es_nq/*.parquet).
  - CORES:    single-core vs all physical cores (ParameterSweep fork pool).

Each cell is TIME-BOXED (run for ~T seconds, count completions) so it adapts to kernel
speed and data size. Headlines:
  - backtests/sec       — the challenge number (vs the legacy 148,193 claim).
  - bar-evaluations/sec = backtests/sec x bars/backtest — the honest "faster on better
    data" metric (heavier data does fewer backtests/sec but ~the same bar-evals/sec).

HONEST: if numba is NOT installed for this Python, the "jit" kernel is really the
pure-Python fallback — the matrix labels it `nojit_fallback` and the findings say so.

Documented + automated (read->filter-by-id->append->write, the repo's ledger pattern):
  data/research/backtest_benchmark_<date>.json   full matrix + environment
  data/agent/hypothesis_ledger.json              BENCH-THROUGHPUT entry (idempotent)
  data/research/bench_history.jsonl              one row/run — regression time-series
  data/research/bench_findings.md                human-readable leaderboard

    python3 scripts/bench_throughput.py
    python3 scripts/bench_throughput.py --tiers 90bar,daily --seconds 2
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.fast_engine import FastBacktestEngine, SweepParams, _sma_crossover_signals
from backtest.sweep import ParameterSweep, _worker_run

RESEARCH = ROOT / "data" / "research"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
HISTORY = RESEARCH / "bench_history.jsonl"
FINDINGS = RESEARCH / "bench_findings.md"
LEGACY_CLAIM = 148_193  # the never-measured GOALS.md number we are replacing

ES_NQ = ROOT / "data" / "es_nq"
TIER_SOURCES = {
    "daily": ES_NQ / "nq_daily.parquet",
    "5min": ES_NQ / "nq_historical_5min.parquet",
    "1min": ES_NQ / "nq_globex_1min.parquet",
}


def _numba_version() -> str | None:
    try:
        import numba
        return numba.__version__
    except Exception:
        return None


NUMBA = _numba_version()
JIT_LABEL = "jit" if NUMBA else "nojit_fallback"


# ── environment ─────────────────────────────────────────────────────────────

def _physical_cores() -> int:
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip())
    except Exception:
        import os
        return os.cpu_count() or 4


def environment() -> dict:
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        cpu = platform.processor() or "unknown"
    return {
        "cpu": cpu,
        "physical_cores": _physical_cores(),
        "numba": NUMBA or "NOT INSTALLED — @njit kernels run as pure-Python fallback",
        "numba_active": bool(NUMBA),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


# ── data loading ────────────────────────────────────────────────────────────

def _coerce_ohlc(df: pd.DataFrame) -> pd.DataFrame | None:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    alias = {
        "rth_open": "open", "rth_high": "high", "rth_low": "low", "rth_close": "close",
        "o": "open", "h": "high", "l": "low", "c": "close",
        "adj close": "close", "last": "close", "price": "close",
    }
    df = df.rename(columns={k: v for k, v in alias.items() if k in df.columns})
    need = {"open", "high", "low", "close"}
    if "close" in df.columns and not need.issubset(df.columns):
        for col in ("open", "high", "low"):
            df[col] = df.get(col, df["close"])
    if not need.issubset(df.columns):
        return None
    return df[["open", "high", "low", "close"]].astype(np.float64).dropna()


def load_tier(name: str) -> tuple[pd.DataFrame, int] | None:
    src = TIER_SOURCES.get(name)
    if src is None or not src.exists():
        print(f"  ! {name}: source missing ({src})")
        return None
    try:
        df = _coerce_ohlc(pd.read_parquet(src))
    except Exception as e:  # noqa: BLE001
        print(f"  ! {name}: load failed ({type(e).__name__}: {e})")
        return None
    if df is None or len(df) < 60:
        print(f"  ! {name}: unusable columns or <60 bars")
        return None
    return df, len(df)


# ── time-boxed kernels (run ~`seconds`, count completions) ──────────────────

def bench_single(df: pd.DataFrame, seconds: float) -> tuple[float, int]:
    engine = FastBacktestEngine.from_dataframe(df)
    engine.warmup()
    p = SweepParams()
    engine.run_single(p)  # extra warm
    count, t0 = 0, time.perf_counter()
    while True:
        engine.run_single(p)
        count += 1
        if time.perf_counter() - t0 >= seconds and count >= 3:
            break
    return count / (time.perf_counter() - t0), count


def bench_parallel(df: pd.DataFrame, cores: int, seconds: float) -> tuple[float, int]:
    engine = FastBacktestEngine.from_dataframe(df)
    sweep = ParameterSweep(engine, n_cores=cores)  # warms JIT in __init__
    pool = sweep._ensure_pool()
    p = SweepParams()
    engine.run_single(p)
    t = time.perf_counter()
    for _ in range(3):
        engine.run_single(p)
    single_s = max((time.perf_counter() - t) / 3, 1e-7)
    n = max(cores * 4, min(500_000, int(cores * seconds / single_s)))
    stops, rrs = [1.0, 1.5, 2.0, 2.5, 3.0], [1.5, 2.0, 2.5, 3.0, 4.0]
    tuples = [SweepParams(stop_atr_mult=stops[i % 5], tp_rr=rrs[(i // 5) % 5]).as_tuple()
              for i in range(n)]
    chunk = max(1, n // (cores * 8))
    t0 = time.perf_counter()
    pool.map(_worker_run, tuples, chunksize=chunk)
    elapsed = time.perf_counter() - t0
    sweep.close()
    return n / elapsed, n


def bench_pure_python_forex(df: pd.DataFrame, seconds: float) -> tuple[float, int]:
    from sovereign.forex.fast_backtester import simulate_forex_trades_arrays
    closes = df["close"].to_numpy(dtype=np.float64)
    opens = df["open"].to_numpy(dtype=np.float64)
    sig, _ = _sma_crossover_signals(closes.astype(np.float32))
    signals = sig.astype(np.int8)
    hold_days = np.full(len(closes), 60, dtype=np.int32)
    count, t0 = 0, time.perf_counter()
    while True:
        simulate_forex_trades_arrays(opens, closes, signals, hold_days, 0.04,
                                     stop_atr_mult=2.0, trailing_atr_mult=1.25)
        count += 1
        if time.perf_counter() - t0 >= seconds and count >= 1:
            break
    return count / (time.perf_counter() - t0), count


# ── run ─────────────────────────────────────────────────────────────────────

def _fmt(x) -> str:
    return f"{x:,.0f}" if isinstance(x, (int, float)) else "n/a"


def run(seconds: float = 1.5, tiers: list[str] | None = None, cores: int | None = None) -> dict:
    env = environment()
    cores = cores or env["physical_cores"]
    tiers = tiers or ["90bar", "daily", "5min", "1min"]
    print(f"BENCH-THROUGHPUT — {env['cpu']} · {cores} cores · numba {env['numba']}")
    print(f"legacy claim (never measured): {LEGACY_CLAIM:,} backtests/sec")
    if not env["numba_active"]:
        print("  ⚠ numba NOT active — 'jit' kernels are the pure-Python fallback (labeled nojit_fallback)\n")
    else:
        print()

    daily = load_tier("daily")
    frames: dict[str, tuple[pd.DataFrame, int]] = {}
    if "90bar" in tiers and daily is not None:
        frames["90bar"] = (daily[0].iloc[:90].reset_index(drop=True), 90)
    for t in ("daily", "5min", "1min"):
        if t in tiers:
            ld = daily if t == "daily" else load_tier(t)
            if ld is not None:
                frames[t] = ld

    rows = []
    for tier, (df, nbars) in frames.items():
        print(f"[{tier}] {nbars:,} bars")
        for kernel, fn in (
            (f"{JIT_LABEL}_1core", lambda d: bench_single(d, seconds)),
            (f"{JIT_LABEL}_{cores}core", lambda d: bench_parallel(d, cores, seconds)),
            ("pure_python_forex", lambda d: bench_pure_python_forex(d, seconds)),
        ):
            try:
                rate, ncalls = fn(df)
                rows.append({
                    "tier": tier, "kernel": kernel, "n_bars": nbars, "n_backtests": ncalls,
                    "cores": cores if kernel.endswith(f"{cores}core") else 1,
                    "backtests_per_sec": round(rate, 1),
                    "ms_per_backtest": round(1000.0 / rate, 6) if rate else None,
                    "bar_evals_per_sec": round(rate * nbars, 0),
                })
                print(f"    {kernel:22s} {rate:14,.0f} bt/s   {rate*nbars:18,.0f} bar-evals/s  (n={ncalls})")
            except Exception as e:  # noqa: BLE001
                print(f"    {kernel:22s} FAILED ({type(e).__name__}: {e})")

    head_single = next((r["backtests_per_sec"] for r in rows
                        if r["tier"] == "90bar" and r["kernel"].endswith("_1core")), None)
    parallel_rates = [r["backtests_per_sec"] for r in rows if r["kernel"].endswith(f"{cores}core")]
    head_parallel = max(parallel_rates) if parallel_rates else None
    best_barevals = max((r["bar_evals_per_sec"] for r in rows), default=0)

    now = datetime.now(timezone.utc)
    result = {
        "id": "BENCH-THROUGHPUT",
        "generated_at": now.isoformat(),
        "environment": env,
        "legacy_claim_backtests_per_sec": LEGACY_CLAIM,
        "headline": {
            "single_core_90bar": head_single,
            "parallel_ceiling": head_parallel,
            "best_bar_evals_per_sec": best_barevals,
            "numba_active": env["numba_active"],
            "beats_legacy_claim": bool(head_parallel and head_parallel > LEGACY_CLAIM),
        },
        "matrix": rows,
    }

    RESEARCH.mkdir(parents=True, exist_ok=True)
    out_json = RESEARCH / f"backtest_benchmark_{now.strftime('%Y-%m-%d')}.json"
    out_json.write_text(json.dumps(result, indent=2))

    raw = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    led_list = raw.get("hypotheses", []) if isinstance(raw, dict) else raw
    led_list = [e for e in led_list if e.get("id") != "BENCH-THROUGHPUT"]
    nb_note = "Numba ACTIVE" if env["numba_active"] else f"numba INACTIVE on py{env['python']} → pure-Python fallback"
    led_list.append({
        "id": "BENCH-THROUGHPUT",
        "name": "Backtest throughput benchmark (measured, replaces the 148k claim)",
        "status": "MEASURED",
        "date_tested": now.strftime("%Y-%m-%d"),
        "result": (f"single-core(90bar)={_fmt(head_single)}/s · {cores}-core ceiling={_fmt(head_parallel)}/s · "
                   f"best bar-evals={_fmt(best_barevals)}/s · legacy claim {LEGACY_CLAIM:,} "
                   f"({'BEATEN' if result['headline']['beats_legacy_claim'] else 'NOT beaten'}; {nb_note})"),
        "methodology_note": (f"{env['cpu']}, {cores} physical cores, {nb_note}. Time-boxed. FastBacktestEngine "
                             "vs pure-Python forex kernel across 90bar/daily/5min/1min, single vs all-cores. "
                             "See backtest_benchmark JSON."),
    })
    LEDGER.write_text(json.dumps({**raw, "hypotheses": led_list} if isinstance(raw, dict) else led_list, indent=2))

    with HISTORY.open("a") as f:
        f.write(json.dumps({
            "t": now.isoformat(), "cpu": env["cpu"], "cores": cores, "numba_active": env["numba_active"],
            "single_core_90bar": head_single, "parallel_ceiling": head_parallel,
            "best_bar_evals_per_sec": best_barevals,
        }) + "\n")

    _write_findings(result)

    print("\n" + "=" * 72)
    print("  THE NUMBER")
    print("=" * 72)
    print(f"  legacy claim (never measured) : {LEGACY_CLAIM:>18,} backtests/sec")
    print(f"  measured single-core (90bar)  : {_fmt(head_single):>18} backtests/sec")
    verdict = "BEATEN ✓" if result["headline"]["beats_legacy_claim"] else "below the claim"
    print(f"  measured {cores}-core ceiling      : {_fmt(head_parallel):>18} backtests/sec   ({verdict})")
    print(f"  best bar-evaluations/sec      : {_fmt(best_barevals):>18} bar-evals/sec")
    if not env["numba_active"]:
        print(f"\n  ⚠ numba is INACTIVE on Python {env['python']} → the JIT engine is dead weight right now.")
        print("    The 148k 'Numba JIT' path is currently impossible. Unlock = a numba-compatible Python.")
    print("=" * 72)
    print(f"  → {out_json}\n  → ledger BENCH-THROUGHPUT · {HISTORY.name} · {FINDINGS.name}")
    return result


def _write_findings(r: dict) -> None:
    h, env = r["headline"], r["environment"]
    lines = [
        "# Backtest Throughput — Measured Leaderboard",
        "",
        f"> {r['generated_at'][:19]}Z · {env['cpu']} · {env['physical_cores']} cores · numba {env['numba']}",
        "",
        f"**Legacy claim (never measured):** {r['legacy_claim_backtests_per_sec']:,} backtests/sec  ",
        f"**Measured single-core (90-bar):** {_fmt(h['single_core_90bar'])} backtests/sec  ",
        f"**Measured parallel ceiling:** {_fmt(h['parallel_ceiling'])} backtests/sec "
        f"({'**beats** the legacy claim ✓' if h['beats_legacy_claim'] else 'below the legacy claim'})  ",
        f"**Best bar-evaluations/sec:** {_fmt(h['best_bar_evals_per_sec'])}",
        "",
    ]
    if not env["numba_active"]:
        lines += [f"> ⚠ **numba is INACTIVE on Python {env['python']}** — the `@njit` kernels run as a pure-Python "
                  "fallback, so the 148k 'Numba JIT' figure is currently unreachable. The unlock is a "
                  "numba-compatible Python (≤3.13), not new code.", ""]
    lines += ["| tier | bars | kernel | cores | backtests/sec | bar-evals/sec |",
              "|---|---:|---|---:|---:|---:|"]
    for row in r["matrix"]:
        lines.append(f"| {row['tier']} | {row['n_bars']:,} | {row['kernel']} | {row['cores']} | "
                     f"{_fmt(row['backtests_per_sec'])} | {_fmt(row['bar_evals_per_sec'])} |")
    lines += ["", "_bar-evals/sec = backtests/sec × bars — the honest 'faster on better data' metric: "
              "heavier data does fewer backtests/sec but ~the same total bar-evaluations._", ""]
    FINDINGS.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure backtest throughput (backtests/sec).")
    ap.add_argument("--seconds", type=float, default=1.5, help="time budget per measurement cell")
    ap.add_argument("--tiers", default="90bar,daily,5min,1min", help="comma list: 90bar,daily,5min,1min")
    ap.add_argument("--cores", type=int, default=None, help="parallel core count (default: physical cores)")
    args = ap.parse_args()
    run(seconds=args.seconds, tiers=[t.strip() for t in args.tiers.split(",") if t.strip()], cores=args.cores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
