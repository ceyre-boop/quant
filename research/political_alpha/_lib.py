"""Shared library for the political-alpha event study (HYP-085, TICK-020).

SELF-CONTAINED BY MANDATE: this module imports NOTHING from sovereign/, ict/,
ict-engine/, config/, audit/, or scripts/ (spec §4; AST-enforced by
tests/test_isolation.py). Where the live system already solved a problem, the
code is COPIED here with a provenance comment, never imported:

  - env parse            <- sovereign/sentiment/store.py::env_key (:279)
  - ThetaData transport  <- sovereign/research/vrp/data_loader.py::ThetaDataLoader (:114)
  - smile / rr25 math    <- sovereign/sentiment/options_surface_feed.py (:62-181)
                            + sovereign/sentiment/vrp_feed.py (:69-105)

Data caches live under research/political_alpha/data/{raw,cache}/ (gitignored).
Governing spec: ~/Obsidian/Obsidian/Trading/Research/Political-Alpha-Claude-Code-Spec.md
Pre-registration: data/research/preregister/HYP-085_political_alpha_trump_events.json
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ── paths & constants ────────────────────────────────────────────────────────────────

MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT.parents[1]
DATA_DIR = MODULE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = MODULE_ROOT / "output"

STUDY_START = "2025-01-20"          # inauguration — locked spec §0
ET = ZoneInfo("America/New_York")

# Locked instrument universe (spec §3) + asset-class mapping for T0 rules (spec §6).
UNIVERSE = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "DX-Y.NYB",
    "XLE", "SLX", "XLF", "KWEB", "GLD",
]
ASSET_CLASS = {
    "EURUSD=X": "fx", "GBPUSD=X": "fx", "USDJPY=X": "fx", "AUDUSD=X": "fx",
    "DX-Y.NYB": "fx",                   # ICE session is near-24h — fx mapping fits
    "XLE": "us_etf", "SLX": "us_etf", "XLF": "us_etf", "KWEB": "us_etf", "GLD": "us_etf",
}
# ThetaData chain symbol per ETF (probed at Phase 3; unavailable -> recorded gap).
ETF_CHAIN = {"XLE": "XLE", "SLX": "SLX", "XLF": "XLF", "KWEB": "KWEB", "GLD": "GLD"}
FXE_SYMBOL = "FXE"                     # positioning proxy for ALL forex rows (spec §3/§7-P3)

# USD leg per fx instrument: +1 means "instrument UP is USD-bullish".
# Used to translate a resolved move into FXE (EUR) bullish/bearish for formula (d).
USD_SIGN = {"EURUSD=X": -1, "GBPUSD=X": -1, "AUDUSD=X": -1, "USDJPY=X": +1, "DX-Y.NYB": +1}

# smile constants — mirror sovereign/sentiment/options_surface_feed.py::_DEFAULTS (:43)
NEAR_BAND = {"target": 30, "min": 20, "max": 45}
R_FLAT = 0.04
MIN_STRIKES = 5


# ── env (copied from sovereign/sentiment/store.py::env_key :279 — NOT imported) ─────

def env_key(name: str, default: str | None = None) -> str | None:
    """os.environ first, else manual parse of REPO_ROOT/.env (no python-dotenv)."""
    val = os.environ.get(name)
    if val:
        return val
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == name:
                return v.strip().strip('"').strip("'")
    return default


# ── jsonl IO ─────────────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=False) + "\n" for r in rows))
    tmp.replace(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── daily OHLCV via yfinance (repo conventions; CSV cache, refresh daily) ────────────

def fetch_daily(ticker: str, start: str = "2023-06-01", refresh: bool = False) -> pd.DataFrame:
    """Daily auto-adjusted OHLCV, tz-naive normalized index. Cached to CSV; a cache
    file written before today refreshes automatically. Empty frame on total failure
    (callers must treat as data_ok:false — never fabricate)."""
    cache = CACHE_DIR / "yf" / f"{ticker.replace('=', '_').replace('^', '_')}.csv"
    if cache.exists() and not refresh:
        mtime = datetime.fromtimestamp(cache.stat().st_mtime, tz=timezone.utc)
        if mtime.date() >= datetime.now(timezone.utc).date():
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            return df
    try:
        import yfinance as yf                        # lazy import (repo convention)
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):    # single-symbol MultiIndex gotcha
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
        return df
    except Exception as exc:
        print(f"  [yf] {ticker}: FETCH FAILED ({type(exc).__name__}: {exc})")
        if cache.exists():                            # stale cache beats nothing — flagged
            print(f"  [yf] {ticker}: using stale cache {cache.name}")
            return pd.read_csv(cache, index_col=0, parse_dates=True)
        return pd.DataFrame()


# ── event-window mechanics (pinned formulas (a); spec §6) ────────────────────────────

def parse_ts(ts_utc: str) -> datetime:
    """ISO-8601 (Z or offset) -> aware UTC datetime."""
    dt = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc)


def map_t0(ts_utc: str, index: pd.DatetimeIndex, asset_class: str) -> pd.Timestamp | None:
    """Statement timestamp -> T0 bar date on the instrument's own calendar.

    fx:     first bar date >= the statement's UTC date (24x5 market).
    us_etf: statement's ET date if it is a session day AND the ET time is before
            16:00 (the close); otherwise the next session date (spec §6).
    """
    if index is None or len(index) == 0:
        return None
    dt = parse_ts(ts_utc)
    if asset_class == "fx":
        day = pd.Timestamp(dt.date())
        pos = index.searchsorted(day)
    else:
        local = dt.astimezone(ET)
        day = pd.Timestamp(local.date())
        if day in index and local.hour < 16:
            pos = index.get_loc(day)
        else:
            pos = index.searchsorted(day + pd.Timedelta(days=1))
    if pos >= len(index):
        return None
    return index[pos]


def log_returns(close: pd.Series) -> pd.Series:
    close = close.astype(float)
    return np.log(close / close.shift(1))


def trailing_sigma60(returns: pd.Series) -> pd.Series:
    """Trailing 60-day rolling SD, shifted one day — NEVER includes the tested day."""
    return returns.rolling(60).std(ddof=1).shift(1)


# ── ThetaData V3 client (copied from sovereign/research/vrp/data_loader.py :114) ─────

class ThetaClient:
    """Two-endpoint ThetaTerminal v3 client. Localhost needs no auth header (the
    terminal authenticates itself); non-local base adds a Bearer token. 403 is
    ambiguous (rate limit vs entitlement depth wall) -> one retry then raise so the
    caller skip-and-counts; HTTP 472 = NO_DATA -> empty. CSV cache per
    (symbol, date, expiration); consecutive-failure circuit breaker at 10."""

    def __init__(self, base_url: str | None = None, cache_dir: Path | None = None) -> None:
        self.base_url = (base_url or env_key("THETADATA_BASE_URL", "http://127.0.0.1:25503")).rstrip("/")
        self.api_key = env_key("THETADATA_API_KEY")
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR / "theta"
        self._exp_cache: dict[str, list[str]] = {}
        self.consec_failures = 0

    def _is_local(self) -> bool:
        return "127.0.0.1" in self.base_url or "localhost" in self.base_url

    def _get(self, path: str, timeout: int = 60) -> str:
        import urllib.error
        import urllib.request
        headers = {} if self._is_local() else {"Authorization": f"Bearer {self.api_key}"}
        req = urllib.request.Request(self.base_url + path, headers=headers)
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read().decode()
            except urllib.error.HTTPError as exc:
                if exc.code == 403 and attempt < 1:
                    time.sleep(5)
                    continue
                raise
        raise RuntimeError("unreachable")

    @staticmethod
    def _csv(text: str) -> pd.DataFrame:
        from io import StringIO
        if not text or not text.strip():
            return pd.DataFrame()
        return pd.read_csv(StringIO(text))

    def alive(self) -> bool:
        """Liveness probe — list FXE expirations; False on any failure."""
        try:
            return len(self.expirations(FXE_SYMBOL)) > 0
        except Exception:
            return False

    def expirations(self, symbol: str) -> list[str]:
        if symbol not in self._exp_cache:
            df = self._csv(self._get(f"/v3/option/list/expirations?symbol={symbol}"))
            exps = (sorted(df["expiration"].astype(str).str[:10].unique().tolist())
                    if not df.empty else [])
            self._exp_cache[symbol] = exps
        return self._exp_cache[symbol]

    def eod_chain(self, symbol: str, date: str, expiration: str) -> pd.DataFrame:
        """Pivoted chain for one (date, expiration): one row per strike with call/put
        mids AND call/put volume kept SEPARATE (the study needs the put/call split).
        Cached as CSV (pyarrow undeclared in this repo — no parquet). Empty on 472."""
        import urllib.error
        d, e = str(date)[:10], str(expiration)[:10]
        cache = self.cache_dir / symbol / f"{d}_{e}.csv"
        if cache.exists():
            df = pd.read_csv(cache)
            return df

        try:
            raw = self._csv(self._get(
                f"/v3/option/history/eod?symbol={symbol}&expiration={e}&start_date={d}&end_date={d}"))
            self.consec_failures = 0
        except urllib.error.HTTPError as ex:
            if ex.code == 472:                        # NO_DATA — cache the emptiness
                raw = pd.DataFrame()
                self.consec_failures = 0
            else:
                self.consec_failures += 1
                if self.consec_failures >= 10:
                    raise RuntimeError(
                        "ThetaData circuit breaker: 10 consecutive failures — aborting "
                        "rather than hammering the terminal") from ex
                raise
        except Exception:
            self.consec_failures += 1
            if self.consec_failures >= 10:
                raise RuntimeError(
                    "ThetaData circuit breaker: 10 consecutive failures — aborting") from None
            raise

        df = self._pivot(raw)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        return df

    @staticmethod
    def _pivot(raw: pd.DataFrame) -> pd.DataFrame:
        """v3 EOD rows (strike x right) -> one row per strike. Adapted from
        ThetaDataLoader._pivot_chain (:184) with call/put volume kept separate."""
        cols = ["strike", "call_mid", "put_mid", "call_volume", "put_volume"]
        if raw.empty or "strike" not in raw.columns:
            return pd.DataFrame(columns=cols)
        raw = raw.copy()
        raw["side"] = raw["right"].astype(str).str.upper().str[0]     # 'C' / 'P'
        rows = []
        for strike, g in raw.groupby("strike"):
            row = {"strike": float(strike), "call_mid": float("nan"), "put_mid": float("nan"),
                   "call_volume": 0.0, "put_volume": 0.0}
            for _, r in g.iterrows():
                side = "call" if r["side"] == "C" else "put"
                bid = float(r.get("bid", float("nan")))
                ask = float(r.get("ask", float("nan")))
                row[f"{side}_mid"] = (bid + ask) / 2.0
                row[f"{side}_volume"] += float(r.get("volume", 0) or 0)
            rows.append(row)
        return pd.DataFrame(rows)[cols].sort_values("strike").reset_index(drop=True)


# ── smile math (verbatim copies — provenance in each docstring) ──────────────────────

def _phi(x: float) -> float:
    """Standard-normal CDF via erf. Copied: sovereign/sentiment/vrp_feed.py:69."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs76_call(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-76 European call PV on a forward. Copied: vrp_feed.py:74."""
    if sigma <= 0 or T <= 0 or F <= 0 or K <= 0:
        return math.exp(-r * T) * max(F - K, 0.0)
    srt = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * srt * srt) / srt
    d2 = d1 - srt
    return math.exp(-r * T) * (F * _phi(d1) - K * _phi(d2))


def implied_vol_atm(call_mid: float, put_mid: float, K: float, T: float, r: float) -> float:
    """ATM IV via Black-76 bisection, forward from put-call parity. Copied: vrp_feed.py:84."""
    if not (np.isfinite(call_mid) and np.isfinite(put_mid) and call_mid > 0 and K > 0 and T > 0):
        return float("nan")
    F = K + (call_mid - put_mid) * math.exp(r * T)
    if F <= 0:
        return float("nan")
    intrinsic = math.exp(-r * T) * max(F - K, 0.0)
    if call_mid <= intrinsic:
        return float("nan")
    lo, hi = 1e-4, 5.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _bs76_call(F, K, T, r, mid) < call_mid:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _strike_iv(mid: float, F: float, K: float, T: float, r: float, is_call: bool) -> float:
    """Black-76 IV for one OTM quote. Copied: options_surface_feed.py:62."""
    if not (np.isfinite(mid) and mid > 0 and F > 0 and K > 0 and T > 0):
        return float("nan")
    if is_call:
        call_mid = mid
    else:
        call_mid = mid + math.exp(-r * T) * (F - K)   # put-call parity -> equivalent call PV
    intrinsic = math.exp(-r * T) * max(F - K, 0.0)
    if call_mid <= intrinsic:
        return float("nan")
    lo, hi = 1e-4, 5.0
    for _ in range(100):
        m = 0.5 * (lo + hi)
        if _bs76_call(F, K, T, r, m) < call_mid:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def _call_delta(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-76 call delta. Copied: options_surface_feed.py:83."""
    if sigma <= 0 or T <= 0:
        return float("nan")
    srt = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * srt * srt) / srt
    return math.exp(-r * T) * _phi(d1)


def _interp_delta_iv(points: list[tuple[float, float]], target: float) -> float:
    """Linear IV interpolation at |delta|=target. Copied: options_surface_feed.py:92."""
    pts = sorted((d, iv) for d, iv in points if np.isfinite(d) and np.isfinite(iv))
    for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
        if d0 <= target <= d1:
            if d1 == d0:
                return v0
            w = (target - d0) / (d1 - d0)
            return v0 + w * (v1 - v0)
    return float("nan")


def smile_read(chain: pd.DataFrame, spot: float, dte: int, r: float, min_strikes: int) -> dict | None:
    """One expiration's (atm_iv, rr25, bf25, n_strikes). Copied: options_surface_feed.py:104."""
    if chain is None or chain.empty or not np.isfinite(spot):
        return None
    T = dte / 365.0
    atm_row = chain.iloc[(chain["strike"] - spot).abs().argmin()]
    K0 = float(atm_row["strike"])
    atm_iv = implied_vol_atm(float(atm_row.get("call_mid", np.nan)),
                             float(atm_row.get("put_mid", np.nan)), K0, T, r)
    if not (np.isfinite(atm_iv) and atm_iv > 0):
        return None
    F = K0 + (float(atm_row["call_mid"]) - float(atm_row["put_mid"])) * math.exp(r * T)
    calls, puts = [], []
    for _, row in chain.iterrows():
        K = float(row["strike"])
        if K >= F:                                    # OTM call side
            iv = _strike_iv(float(row.get("call_mid", np.nan)), F, K, T, r, is_call=True)
            if np.isfinite(iv):
                d = _call_delta(F, K, T, r, iv)
                if np.isfinite(d) and 0.02 <= d <= 0.60:
                    calls.append((d, iv))
        if K <= F:                                    # OTM put side
            iv = _strike_iv(float(row.get("put_mid", np.nan)), F, K, T, r, is_call=False)
            if np.isfinite(iv):
                dc = _call_delta(F, K, T, r, iv)
                dp = abs(dc - math.exp(-r * T))
                if np.isfinite(dp) and 0.02 <= dp <= 0.60:
                    puts.append((dp, iv))
    if len(calls) + len(puts) < min_strikes:
        return None
    iv_c25 = _interp_delta_iv(calls, 0.25)
    iv_p25 = _interp_delta_iv(puts, 0.25)
    rr25 = iv_c25 - iv_p25 if np.isfinite(iv_c25) and np.isfinite(iv_p25) else float("nan")
    bf25 = 0.5 * (iv_c25 + iv_p25) - atm_iv if np.isfinite(rr25) else float("nan")
    return {"atm_iv": float(atm_iv), "rr25": float(rr25) if np.isfinite(rr25) else None,
            "bf25": float(bf25) if np.isfinite(bf25) else None,
            "n_strikes": int(len(calls) + len(puts))}


def pick_expiry(expirations: list[str], d: pd.Timestamp, band: dict = NEAR_BAND) -> tuple[str, int]:
    """Listed expiration with DTE closest to band target within [min,max].
    Copied: options_surface_feed.py::_pick_expiry (:173)."""
    best, best_gap = ("", 0), None
    for exp in expirations:
        dte = (pd.Timestamp(exp) - d).days
        if band["min"] <= dte <= band["max"]:
            gap = abs(dte - band["target"])
            if best_gap is None or gap < best_gap:
                best, best_gap = (exp, dte), gap
    return best


def rr25_for_day(client: ThetaClient, symbol: str, day: pd.Timestamp, spot: float) -> dict | None:
    """EOD rr25 + put/call volume for `day` on the ~30d expiry. None when the smile
    is unreadable (thin chain, no expiry in band, missing quotes) — never synthesized."""
    exp, dte = pick_expiry(client.expirations(symbol), pd.Timestamp(day))
    if not exp:
        return None
    chain = client.eod_chain(symbol, str(pd.Timestamp(day).date()), exp)
    if chain.empty:
        return None
    read = smile_read(chain, spot, dte, R_FLAT, MIN_STRIKES)
    if read is None or read.get("rr25") is None:
        return None
    return {
        "rr25": read["rr25"], "n_strikes": read["n_strikes"],
        "expiration": exp, "dte": dte,
        "call_volume": float(chain["call_volume"].sum()),
        "put_volume": float(chain["put_volume"].sum()),
    }
