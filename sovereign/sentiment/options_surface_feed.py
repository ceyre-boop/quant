"""sovereign/sentiment/options_surface_feed.py — FX-ETF options SURFACE feeder.

ONE table serving BOTH VRP-001 (ATM IV term structure) and the positioning board
(25Δ risk reversals = skew crowd, 25Δ butterflies = tail demand — HYP-074/075/078/079
substrate). Per (pair, weekly Friday obs):

    atm_iv_1m / atm_iv_3m   ATM implied vol at the ~30d and ~90d expirations
    term_slope              atm_iv_1m − atm_iv_3m  (>0 = inverted, stress pricing)
    rr25                    iv(25Δ call) − iv(25Δ put)   (FX risk-reversal convention)
    bf25                    0.5·(iv25c + iv25p) − atm_iv_1m  (smile curvature / wings)

Method: per-strike Black-76 IV inversion from EOD mids (Options Value tier serves quotes,
not greeks); forward from ATM put-call parity; per-strike delta from Black-76 d1; 25Δ IVs
linearly interpolated in delta space across OTM strikes. Same discipline as vrp_feed:
`iv_obs_date == date` provenance, forward-look-audited, idempotent upsert, and a LOUD skip
(board carries NULLs) when ThetaTerminal is unreachable — never a silent mock.

FIXTURE MODE (explicit, never silent): `update(fixture=True)` (CLI `--fixture` in the runner)
reads recorded/synthetic chains from data/fixtures/thetadata/ via FixtureLoader and stamps
every coverage line and every row's `iv_source` with the FIXTURE prefix. Fixture data exists
so the schema/tests/wiring land before the subscription question is resolved — fixture rows
are for TESTS ONLY and must never back a hypothesis run (the prereg protocol requires real
provenance; `iv_source LIKE 'FIXTURE%'` rows are excluded by the runner's coverage report).

Isolation (NN#1): config + sentiment store + vrp helpers only; nothing from ict/ or layer1/2.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.loader import params
from sovereign.sentiment.store import connect, env_key, upsert
from sovereign.sentiment.vrp_feed import _bs76_call, _phi, _yf_close, implied_vol_atm

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "thetadata"

_DEFAULTS = {
    "etf": {"EURUSD": "FXE", "GBPUSD": "FXB", "USDJPY": "FXY", "AUDUSD": "FXA"},
    "near": {"target": 30, "min": 20, "max": 45},
    "far": {"target": 90, "min": 60, "max": 130},
    "risk_free_rate": 0.04,
    "sample_freq": "W-FRI",
    "backfill_start": "2016-01-01",
    "min_strikes": 5,          # need at least this many usable OTM IVs to trust a smile read
}


def _cfg() -> dict:
    c = dict(_DEFAULTS)
    c.update(params.get("sentiment", {}).get("options_surface", {}) or {})
    return c


# ── smile math (pure; unit-tested offline) ──────────────────────────────────────────────────────

def _strike_iv(mid: float, F: float, K: float, T: float, r: float, is_call: bool) -> float:
    """Black-76 IV for one OTM quote (calls above F, puts below F via parity-equivalent call)."""
    if not (np.isfinite(mid) and mid > 0 and F > 0 and K > 0 and T > 0):
        return float("nan")
    if is_call:
        call_mid = mid
    else:
        call_mid = mid + math.exp(-r * T) * (F - K)   # put-call parity → equivalent call PV
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
    """Black-76 call delta (forward measure, discounted)."""
    if sigma <= 0 or T <= 0:
        return float("nan")
    srt = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * srt * srt) / srt
    return math.exp(-r * T) * _phi(d1)


def _interp_delta_iv(points: list[tuple[float, float]], target: float) -> float:
    """Linear interpolation of IV at |delta|=target from (|delta|, iv) points (nan if unbracketed)."""
    pts = sorted((d, iv) for d, iv in points if np.isfinite(d) and np.isfinite(iv))
    for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
        if d0 <= target <= d1:
            if d1 == d0:
                return v0
            w = (target - d0) / (d1 - d0)
            return v0 + w * (v1 - v0)
    return float("nan")


def smile_read(chain: pd.DataFrame, spot: float, dte: int, r: float, min_strikes: int) -> dict | None:
    """One expiration's (atm_iv, rr25, bf25, n_strikes) from an EOD chain with call/put mids."""
    if chain is None or chain.empty or not np.isfinite(spot):
        return None
    T = dte / 365.0
    atm_row = chain.iloc[(chain["strike"] - spot).abs().argmin()]
    K0 = float(atm_row["strike"])
    atm_iv = implied_vol_atm(float(atm_row.get("call_mid", np.nan)), float(atm_row.get("put_mid", np.nan)),
                             K0, T, r)
    if not (np.isfinite(atm_iv) and atm_iv > 0):
        return None
    F = K0 + (float(atm_row["call_mid"]) - float(atm_row["put_mid"])) * math.exp(r * T)
    calls, puts = [], []
    for _, row in chain.iterrows():
        K = float(row["strike"])
        if K >= F:                                   # OTM call side
            iv = _strike_iv(float(row.get("call_mid", np.nan)), F, K, T, r, is_call=True)
            if np.isfinite(iv):
                d = _call_delta(F, K, T, r, iv)
                if np.isfinite(d) and 0.02 <= d <= 0.60:
                    calls.append((d, iv))
        if K <= F:                                   # OTM put side (|put delta| = e^{-rT} − call delta)
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


# ── loaders ─────────────────────────────────────────────────────────────────────────────────────

class FixtureLoader:
    """Recorded/synthetic chains from data/fixtures/thetadata/ — EXPLICIT fixture mode only.

    File layout: {SYMBOL}.json = {"expirations": [...], "chains": {"{date}|{exp}":
    [{"strike":..., "call_mid":..., "put_mid":...}, ...]}, "spot": {"{date}": ...}}.
    """
    is_fixture = True

    def __init__(self, root: Path = FIXTURE_DIR):
        self.root = Path(root)

    def _doc(self, symbol: str) -> dict:
        p = self.root / f"{symbol}.json"
        if not p.exists():
            raise FileNotFoundError(f"no fixture for {symbol} at {p}")
        return json.loads(p.read_text())

    def list_expirations(self, symbol: str) -> list[str]:
        return list(self._doc(symbol)["expirations"])

    def get_option_chain(self, symbol: str, date: str, expiration: str) -> pd.DataFrame:
        rows = self._doc(symbol)["chains"].get(f"{date}|{expiration}", [])
        return pd.DataFrame(rows)

    def spot(self, symbol: str, date: str) -> float:
        return float(self._doc(symbol).get("spot", {}).get(date, float("nan")))


def _pick_expiry(expirations: list[str], d: pd.Timestamp, band: dict) -> tuple[str, int]:
    best, best_gap = ("", 0), None
    for exp in expirations:
        dte = (pd.Timestamp(exp) - d).days
        if band["min"] <= dte <= band["max"]:
            gap = abs(dte - band["target"])
            if best_gap is None or gap < best_gap:
                best, best_gap = (exp, dte), gap
    return best


# ── feeder entry point ──────────────────────────────────────────────────────────────────────────

def update(con=None, start: str | None = None, fixture: bool = False) -> dict:
    """Build the options surface per (pair, weekly obs) → sentiment_options_surface. Returns coverage.

    Real mode: ThetaData via the shared loader; LOUD skip when unreachable.
    Fixture mode: FixtureLoader; every coverage entry and every row is FIXTURE-stamped.
    """
    cfg = _cfg()
    start = start or cfg["backfill_start"]
    src_prefix = ""
    if fixture:
        loader = FixtureLoader()
        src_prefix = "FIXTURE:"
        print("  [options_surface] ⚠️ FIXTURE MODE — NOT REAL DATA (tests/wiring only; "
              "never back a hypothesis run with these rows)")
    else:
        from sovereign.research.vrp.data_loader import ThetaDataLoader
        try:
            key = env_key("THETADATA_API_KEY")
        except RuntimeError:
            key = None
        try:
            base = env_key("THETADATA_BASE_URL")
        except RuntimeError:
            base = "http://127.0.0.1:25503"
        loader = ThetaDataLoader(api_key=key, base_url=base)
        probe = next(iter(cfg["etf"].values()), "FXE")
        try:
            loader.list_expirations(probe)
        except Exception:
            print(f"  [options_surface] ThetaData not reachable — start ThetaTerminal ({base}); skipping "
                  "(board carries NULL rr25/bf25/atm_term_slope)")
            return {}

    own = con is None
    con = con or connect()
    now = datetime.now(timezone.utc)
    frames, coverage = [], {}
    for pair, symbol in cfg["etf"].items():
        try:
            expirations = loader.list_expirations(symbol)
        except Exception as exc:
            coverage[f"{src_prefix}{pair}"] = {"symbol": symbol, "rows": 0,
                                               "note": f"expirations failed: {type(exc).__name__}"}
            continue
        if fixture:
            doc_spots = loader._doc(symbol).get("spot", {})
            sample_dates = [pd.Timestamp(d) for d in sorted(doc_spots)]
        else:
            closes = _yf_close(symbol)
            if closes.empty:
                coverage[f"{src_prefix}{pair}"] = {"symbol": symbol, "rows": 0, "note": "no yfinance spot"}
                continue
            idx = closes.index[closes.index >= pd.Timestamp(start)]
            sample_dates = list(pd.Series(idx, index=idx).resample(cfg["sample_freq"]).last().dropna())
        recs = []
        for d in sample_dates:
            d = pd.Timestamp(d)
            spot = loader.spot(symbol, str(d.date())) if fixture else float(closes.asof(d))
            if not np.isfinite(spot):
                continue
            row = {"date": d.date(), "pair": pair, "symbol": symbol,
                   "iv_source": f"{src_prefix}bs_invert", "iv_obs_date": d.date(), "fetched_at": now}
            got = False
            for label, band in (("1m", cfg["near"]), ("3m", cfg["far"])):
                exp, dte = _pick_expiry(expirations, d, band)
                if not exp:
                    continue
                try:
                    chain = loader.get_option_chain(symbol, str(d.date()), exp)
                except Exception:
                    continue
                read = smile_read(chain, spot, dte, cfg["risk_free_rate"], cfg["min_strikes"])
                if read is None:
                    continue
                got = True
                row[f"expiry_{label}"] = pd.Timestamp(exp).date()
                row[f"dte_{label}"] = dte
                row[f"atm_iv_{label}"] = read["atm_iv"]
                if label == "1m":
                    row["rr25"] = read["rr25"]
                    row["bf25"] = read["bf25"]
                    row["n_strikes"] = read["n_strikes"]
            if got:
                if np.isfinite(row.get("atm_iv_1m", np.nan)) and np.isfinite(row.get("atm_iv_3m", np.nan)):
                    row["term_slope"] = row["atm_iv_1m"] - row["atm_iv_3m"]
                recs.append(row)
        if not recs:
            coverage[f"{src_prefix}{pair}"] = {"symbol": symbol, "rows": 0, "note": "no usable chains"}
            continue
        df = pd.DataFrame(recs)
        for col in ("expiry_1m", "dte_1m", "atm_iv_1m", "rr25", "bf25", "n_strikes",
                    "expiry_3m", "dte_3m", "atm_iv_3m", "term_slope"):
            if col not in df.columns:
                df[col] = None
        out = df[["date", "pair", "symbol", "expiry_1m", "dte_1m", "atm_iv_1m", "rr25", "bf25",
                  "n_strikes", "expiry_3m", "dte_3m", "atm_iv_3m", "term_slope", "iv_source",
                  "iv_obs_date", "fetched_at"]]
        frames.append(out)
        coverage[f"{src_prefix}{pair}"] = {"symbol": symbol, "rows": int(len(out)),
                                           "start": str(out["date"].min()), "end": str(out["date"].max()),
                                           "rr25_nonnull": int(out["rr25"].notna().sum())}
    if frames:
        upsert(con, "sentiment_options_surface", pd.concat(frames, ignore_index=True), ["date", "pair"])
    if own:
        con.close()
    return coverage
