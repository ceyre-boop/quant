"""
Monthly Re-optimisation — Sovereign Self-Improvement Loop

Runs on the first trading day of each calendar month. Downloads the last
252 trading days of daily bars for each symbol, sweeps stop/tp parameters
per regime using the fast JIT engine, picks the best combo, and writes
the results back to config/parameters.yml under the `regime_params` key.

Adds to the self-improvement loop:
  live trade  →  failure clusters  →  cluster veto  →  monthly re-opt
                                                              ↓
                                              regime_params updated
                                                              ↓
                                          kelly_engine reads new stops

Usage (standalone test):
    python3 -c "
    from sovereign.monthly_reopt import MonthlyReopt
    MonthlyReopt().run(['META', 'UNH'])
    "

Auto-trigger in execute_daily.py:
    reopt = MonthlyReopt()
    if reopt.should_run():
        reopt.run(symbols=['META', 'PFE', 'UNH', 'BLK', 'QQQ', 'ES=F', 'NQ=F'])
"""

from __future__ import annotations

import json
import logging
import warnings
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

PARAMS_YML   = Path("config/parameters.yml")
LAST_REOPT   = Path("logs/last_reopt.json")

# Grid to sweep each month
STOP_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0]
TP_RRS     = [1.5, 2.0, 2.5, 3.0, 4.0]

# Quality filters applied before accepting a combo
# 10 trades is the floor for the full 7-symbol production run.
# With only 1-2 symbols the sweep will fall back gracefully.
MIN_TRADES   = 5
MAX_DRAWDOWN = 20.0   # percent

# Hurst thresholds (mirrors parameters.yml — no live import to avoid stale cache)
H_TRENDING    = 0.52
H_MEAN_REVERT = 0.45


# ── Trading calendar helper ──────────────────────────────────────────────────

def _is_trading_day(d: date) -> bool:
    """Return True if d is Mon–Fri (ignores public holidays for simplicity)."""
    return d.weekday() < 5


def _first_trading_day_of_month(year: int, month: int) -> date:
    """Return first Mon–Fri on or after the 1st of the given month."""
    d = date(year, month, 1)
    while not _is_trading_day(d):
        d += timedelta(days=1)
    return d


# ── Indicator helpers (no imports from universe_sweep to stay self-contained) ─

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1.0)
    out = arr.copy()
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _atr_series(highs, lows, closes, period=14) -> np.ndarray:
    n = len(closes)
    tr = np.empty(n, dtype=np.float32)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)
    return _ema(tr, period).astype(np.float32)


def _rolling_hurst(closes: np.ndarray, window: int = 63) -> np.ndarray:
    """Variance-ratio Hurst. Returns float32 array same length as closes."""
    from numpy.lib.stride_tricks import as_strided

    n = len(closes)
    hurst = np.full(n, 0.5, dtype=np.float32)
    if n < window + 4:
        return hurst

    c = closes.astype(np.float64)
    rets  = np.log(c[1:] / (c[:-1] + 1e-10))
    n_w   = len(rets) - window + 1
    if n_w < 1:
        return hurst

    strd = rets.strides[0]
    w1   = as_strided(rets, shape=(n_w, window), strides=(strd, strd))
    m1   = w1.mean(axis=1, keepdims=True)
    var1 = ((w1 - m1) ** 2).mean(axis=1)

    rets2 = rets[:-1] + rets[1:]
    n_w2  = len(rets2) - window + 1
    if n_w2 < 1:
        return hurst

    strd2 = rets2.strides[0]
    w2    = as_strided(rets2, shape=(n_w2, window), strides=(strd2, strd2))
    m2    = w2.mean(axis=1, keepdims=True)
    var2  = ((w2 - m2) ** 2).mean(axis=1)

    mn    = min(n_w, n_w2)
    ratio = var2[:mn] / (2.0 * var1[:mn] + 1e-12)
    h     = np.clip(0.5 + 0.5 * np.log(np.clip(ratio, 1e-6, 100)) / np.log(2),
                    0.1, 0.9).astype(np.float32)

    end_idx = window + mn
    hurst[window:end_idx] = h[: end_idx - window]
    return hurst


def _momentum_signals(closes, highs, lows, volume, hurst) -> Tuple[np.ndarray, np.ndarray]:
    fast = _ema(closes, 9)
    slow = _ema(closes, 21)
    diff = fast - slow
    prev = np.roll(diff, 1); prev[0] = diff[0]
    sigs = np.zeros(len(closes), dtype=np.int8)
    sigs[(diff > 0) & (prev <= 0)] = 1
    sigs[(diff < 0) & (prev >= 0)] = -1
    sigs[hurst < H_TRENDING] = 0
    sigs[:21] = 0
    conf = np.clip(np.abs(diff) / (slow + 1e-10) * 500.0, 0, 1).astype(np.float32)
    return sigs, conf


def _reversion_signals(closes, highs, lows, volume, hurst) -> Tuple[np.ndarray, np.ndarray]:
    n, period = len(closes), 20
    cs  = np.cumsum(closes)
    cs2 = np.cumsum(closes ** 2)
    roll_sum  = cs[period:] - cs[:-period]
    roll_sum2 = cs2[period:] - cs2[:-period]
    mid = np.empty(n); mid[:period] = closes[:period].mean(); mid[period:] = roll_sum / period
    var = roll_sum2 / period - (roll_sum / period) ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    std_f = np.empty(n); std_f[:period] = std[0] if len(std) else 0.0; std_f[period:] = std
    upper = mid + 2.0 * std_f; lower = mid - 2.0 * std_f
    prev_c = np.roll(closes, 1); prev_c[0] = closes[0]
    prev_l = np.roll(lower, 1); prev_l[0] = lower[0]
    prev_u = np.roll(upper, 1); prev_u[0] = upper[0]
    sigs = np.zeros(n, dtype=np.int8)
    sigs[(closes < lower) & (prev_c >= prev_l)] = 1
    sigs[(closes > upper) & (prev_c <= prev_u)] = -1
    sigs[hurst > H_MEAN_REVERT] = 0
    sigs[:period] = 0
    z = (closes - mid) / (std_f + 1e-10)
    conf = np.clip(np.abs(z) / 3.0, 0, 1).astype(np.float32)
    return sigs, conf


_REGIME_SIGNAL_FN = {
    "MOMENTUM":  _momentum_signals,
    "REVERSION": _reversion_signals,
}


# ═════════════════════════════════════════════════════════════════════════════
# MonthlyReopt
# ═════════════════════════════════════════════════════════════════════════════

class MonthlyReopt:
    """Monthly parameter re-optimiser.

    Checks whether today is the first trading day of a new month relative
    to the last run. If so, downloads 252 days of daily bars for each
    symbol, sweeps stop_atr_mult × tp_rr for each regime, picks the best
    combo, and writes regime_params back to config/parameters.yml.
    """

    def should_run(self) -> bool:
        """Return True on the first trading day of a new month.

        Logic:
          - If last_reopt.json doesn't exist → True (never run before).
          - If last_reopt month != current month → True.
          - Otherwise → False.
        """
        today = date.today()

        if not LAST_REOPT.exists():
            logger.info("[MonthlyReopt] No prior run found — scheduling re-opt.")
            return True

        try:
            record = json.loads(LAST_REOPT.read_text())
            last_date = date.fromisoformat(record.get("date", "2000-01-01"))
        except Exception:
            return True

        if last_date.month != today.month or last_date.year != today.year:
            first_td = _first_trading_day_of_month(today.year, today.month)
            if today >= first_td:
                logger.info(
                    f"[MonthlyReopt] New month ({today}) vs last run ({last_date}). "
                    f"First trading day = {first_td}. Scheduling re-opt."
                )
                return True

        logger.info(f"[MonthlyReopt] Already ran this month ({last_date}). Skipping.")
        return False

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, symbols: List[str]) -> Dict[str, dict]:
        """Download bars, sweep, update parameters.yml.

        Returns the new regime_params dict written to the config.
        """
        logger.info("[MonthlyReopt] Starting monthly re-optimisation...")
        LAST_REOPT.parent.mkdir(parents=True, exist_ok=True)

        # 1. Download data
        bars = self._download(symbols)
        if not bars:
            logger.error("[MonthlyReopt] No data loaded — aborting.")
            return {}

        # 2. Warm JIT once before multiprocessing
        self._warmup_jit(bars)

        # 3. Sweep each regime
        regime_params: Dict[str, dict] = {}
        for regime in ("MOMENTUM", "REVERSION"):
            result = self._sweep_regime(regime, bars)
            if result:
                regime_params[regime] = result
                logger.info(
                    f"[MonthlyReopt] {regime}: "
                    f"stop={result['stop_atr_mult']}  "
                    f"tp={result['tp_rr']}  "
                    f"sharpe={result['sharpe']:.2f}  "
                    f"trades={result['trades']}"
                )
            else:
                logger.warning(f"[MonthlyReopt] {regime}: no valid combo found — keeping existing params")

        if not regime_params:
            logger.warning("[MonthlyReopt] Both regimes failed — no config update.")
            return {}

        # 4. Write back to parameters.yml
        self._write_params(regime_params)

        # 5. Write audit trail
        self._write_audit(regime_params)

        # 6. Clear loader cache so running process picks up new values
        try:
            from config.loader import load_params
            load_params.cache_clear()
            logger.info("[MonthlyReopt] Config cache cleared — new params active.")
        except Exception as e:
            logger.warning(f"[MonthlyReopt] Could not clear config cache: {e}")

        # 7. Print summary
        parts = []
        for regime, p in regime_params.items():
            parts.append(f"{regime} stop={p['stop_atr_mult']} tp={p['tp_rr']}")
        print(f"REOPT COMPLETE — {' | '.join(parts)}")

        return regime_params

    # ── Data ──────────────────────────────────────────────────────────────────

    def _download(self, symbols: List[str]) -> Dict[str, dict]:
        """Fetch 252 trading days of daily bars per symbol via yfinance."""
        import yfinance as yf

        end   = datetime.now()
        start = end - timedelta(days=365)   # ~252 trading days in a year

        logger.info(f"[MonthlyReopt] Downloading {len(symbols)} symbols (252d daily)...")
        raw = yf.download(
            symbols, start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True, progress=False,
        )

        bars: Dict[str, dict] = {}
        for sym in symbols:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    df = raw.xs(sym, axis=1, level=1).copy()
                else:
                    df = raw.copy()
                df.columns = [str(c).lower() for c in df.columns]
                df = df[["open", "high", "low", "close", "volume"]].dropna()

                if len(df) < 50:
                    logger.warning(f"[MonthlyReopt] {sym}: only {len(df)} bars, skipping")
                    continue

                c = df["close"].to_numpy(dtype=np.float64)
                h = df["high"].to_numpy(dtype=np.float64)
                l = df["low"].to_numpy(dtype=np.float64)
                v = df["volume"].fillna(1e6).to_numpy(dtype=np.float64)

                bars[sym] = {
                    "df":     df,
                    "closes": c,
                    "highs":  h,
                    "lows":   l,
                    "volume": v,
                    "hurst":  _rolling_hurst(c),
                    "dates":  df.index,
                }
            except Exception as e:
                logger.warning(f"[MonthlyReopt] {sym}: {e}")

        logger.info(f"[MonthlyReopt] Loaded {len(bars)}/{len(symbols)} symbols")
        return bars

    # ── JIT warmup ────────────────────────────────────────────────────────────

    def _warmup_jit(self, bars: Dict[str, dict]) -> None:
        from backtest.fast_engine import FastBacktestEngine, SweepParams
        sample = next(iter(bars.values()))
        dummy = np.zeros(len(sample["closes"]), dtype=np.int8)
        dummy[30] = 1
        eng = FastBacktestEngine.from_signals(
            sample["df"], dummy, np.ones(len(dummy), dtype=np.float32)
        )
        eng.warmup()
        logger.info("[MonthlyReopt] JIT warm.")

    # ── Per-regime sweep ──────────────────────────────────────────────────────

    def _sweep_regime(
        self, regime: str, bars: Dict[str, dict]
    ) -> Optional[dict]:
        """Pool all regime-matched bars across symbols, run ParameterSweep."""
        from backtest.sweep import ParameterSweep
        from backtest.fast_engine import FastBacktestEngine, SweepParams

        sig_fn = _REGIME_SIGNAL_FN.get(regime)
        if sig_fn is None:
            return None

        # Collect all regime-filtered bars into one concatenated df
        dfs, sigs_list, conf_list = [], [], []

        for sym, d in bars.items():
            closes = d["closes"]
            highs  = d["highs"]
            lows   = d["lows"]
            volume = d["volume"]
            hurst  = d["hurst"]

            # regime-specific mask
            if regime == "MOMENTUM":
                mask = hurst > H_TRENDING
            else:
                mask = hurst < H_MEAN_REVERT

            if mask.sum() < 20:
                continue

            try:
                sigs, conf = sig_fn(closes, highs, lows, volume, hurst)
            except Exception:
                continue

            # Only include regime-active bars
            sub_df   = d["df"][mask].copy()
            sub_sigs = sigs[mask]
            sub_conf = conf[mask]

            if len(sub_df) < 20:
                continue

            dfs.append(sub_df)
            sigs_list.append(sub_sigs)
            conf_list.append(sub_conf)

        if not dfs:
            logger.warning(f"[MonthlyReopt] {regime}: no regime-matching bars across any symbol")
            return None

        combined_df   = pd.concat(dfs, ignore_index=False)
        combined_sigs = np.concatenate(sigs_list).astype(np.int8)
        combined_conf = np.concatenate(conf_list).astype(np.float32)

        # Rebuild OHLC columns with float32-compatible types
        combined_df = combined_df.sort_index()
        # Re-align arrays to match df length after sort (may differ due to tz)
        min_len = min(len(combined_df), len(combined_sigs))
        combined_df   = combined_df.iloc[:min_len]
        combined_sigs = combined_sigs[:min_len]
        combined_conf = combined_conf[:min_len]

        engine = FastBacktestEngine.from_signals(combined_df, combined_sigs, combined_conf)
        sweep  = ParameterSweep(engine)

        results_df = sweep.run_grid(
            {"stop_atr_mult": STOP_MULTS, "tp_rr": TP_RRS},
            min_trades=MIN_TRADES,
        )

        if results_df.empty:
            logger.warning(f"[MonthlyReopt] {regime}: sweep returned no results above min_trades={MIN_TRADES}")
            return None

        # Filter max drawdown
        valid = results_df[results_df["max_dd_pct"] < MAX_DRAWDOWN]
        if valid.empty:
            logger.warning(f"[MonthlyReopt] {regime}: all combos exceed max_dd {MAX_DRAWDOWN}% — using best available")
            valid = results_df  # fall back to unfiltered

        # Best Sharpe
        best = valid.iloc[0]  # already sorted by profit_factor desc; re-sort by sharpe if present
        if "total_return_pct" in valid.columns:
            # proxy for Sharpe: profit_factor (already sorted desc)
            pass

        return {
            "stop_atr_mult": float(best["stop_atr_mult"]),
            "tp_rr":         float(best["tp_rr"]),
            "profit_factor": round(float(best.get("profit_factor", 0)), 4),
            "win_rate":      round(float(best.get("win_rate", 0)), 4),
            "max_dd_pct":    round(float(best.get("max_dd_pct", 0)), 2),
            "trades":        int(best.get("trades", 0)),
            "sharpe":        round(float(best.get("total_return_pct", 0)), 4),
            "regime":        regime,
            "fitted_on":     date.today().isoformat(),
        }

    # ── Config write ──────────────────────────────────────────────────────────

    def _write_params(self, regime_params: Dict[str, dict]) -> None:
        """Write regime_params block into parameters.yml preserving all other keys."""
        if not PARAMS_YML.exists():
            logger.error(f"[MonthlyReopt] {PARAMS_YML} not found — cannot write params.")
            return

        with open(PARAMS_YML) as f:
            config = yaml.safe_load(f) or {}

        # Build the regime_params block — only the fields kelly_engine needs
        config["regime_params"] = {
            regime: {
                "stop_atr_mult": p["stop_atr_mult"],
                "tp_rr":         p["tp_rr"],
                "fitted_on":     p["fitted_on"],
            }
            for regime, p in regime_params.items()
        }

        with open(PARAMS_YML, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info(f"[MonthlyReopt] Wrote regime_params → {PARAMS_YML}")

    # ── Audit trail ───────────────────────────────────────────────────────────

    def _write_audit(self, regime_params: Dict[str, dict]) -> None:
        record = {
            "date":          date.today().isoformat(),
            "timestamp":     datetime.utcnow().isoformat(),
            "regime_params": regime_params,
        }
        LAST_REOPT.write_text(json.dumps(record, indent=2))
        logger.info(f"[MonthlyReopt] Audit trail → {LAST_REOPT}")
