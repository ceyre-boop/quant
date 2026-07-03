"""sovereign/sentiment/vrp_feed.py — Volatility Risk Premium (VRP) feature feeder.

VRP-001 is the standing TRUE_DIVERSIFIER candidate: the premium for selling vol insurance to
HEDGERS is driven by a DIFFERENT crowd than the yield-chasing carry trade, so it could be the one
edge uncorrelated with carry. This feeder builds the FEATURE only (no model, no diagnostic) and
fuses it into the board:

    vrp_signal = iv_atm − rv_trailing            (annualized vol units, e.g. 0.085 = 8.5%)
    vrp_pct    = trailing percentile of vrp_signal over its own history (rich vs cheap)

⚠️ LOOK-AHEAD GUARD (load-bearing). VRP = IV − RV is inherently forward-looking and is the #1 way VRP
research fabricates a backtest. This FEATURE uses ONLY past+present:
  • iv_atm      — implied vol read from the option chain dated to its OBSERVABLE EOD close (the chain
                  is fetched start_date=end_date=D, i.e. day-D end-of-day). `iv_obs_date == D`.
  • rv_trailing — realized vol over the PAST rv_window days of pair spot (forward=False, causal).
                  `rv_last_date == D`.
The FORWARD realized vol (what a VRP trade bets on) NEVER appears here — it may only be a later
diagnostic's OUTCOME/label. Both provenance dates are persisted and the forward-look test asserts
`iv_obs_date <= date AND rv_last_date <= date` (0 violations). Feature = past+present; outcome = future.

IV source (decided 2026-07-01): prefer ThetaData NATIVE IV when the chain carries it; otherwise
Black-Scholes-INVERT ATM IV from the call+put mids. The forward is taken from put-call parity
(F = K + (C−P)·e^{rT}), which encodes the pair's rate-differential carry directly — so no separate
dividend/foreign-rate assumption is needed. Assumptions when inverting: flat short rate `risk_free_rate`
(ATM IV is second-order in r), and American≈European for near-dated ATM. `iv_source` records which path.

RV underlying (decided 2026-07-01): the FX PAIR spot (yfinance `EURUSD=X` …) — longer/cleaner history,
the exact series carry trades. IV comes from the currency-ETF options (FXE/FXB/FXY/FXA), RV from the
pair; they track near-identically (FXE≈EURUSD) and vol is sign-agnostic (FXY↔USDJPY reciprocal is fine).

Isolation (NN#1): imports only config, the sentiment store, external data libs, and the isolation-clean
sovereign.research.vrp helpers (loader + pure calculator) — NOTHING from ict/ or layer1/2. Idempotent
(delete-then-insert upsert). If ThetaTerminal is unreachable and no key is set, reports and returns {}.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config.loader import params
from sovereign.research.vrp.vrp_calculator import realized_variance
from sovereign.sentiment.store import connect, env_key, upsert

# ── config (in-code defaults; overridable via config/parameters.yml :: sentiment.vrp) ──────────── #
_DEFAULTS = {
    # FX pair → currency-ETF option underlying (the tradeable IV proxy; ThetaData has no spot-FX options)
    "etf": {"EURUSD": "FXE", "GBPUSD": "FXB", "USDJPY": "FXY", "AUDUSD": "FXA"},
    # FX pair → yfinance spot ticker (the RV underlying)
    "spot": {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", "AUDUSD": "AUDUSD=X"},
    "dte_target": 30, "dte_min": 25, "dte_max": 45,   # ~1-month IV
    "rv_window": 20,                                    # trailing realized-vol lookback (days)
    "percentile_window": 252, "percentile_min_periods": 40,   # trailing-1yr rich/cheap percentile
    "risk_free_rate": 0.04,                            # flat short rate for BS inversion (documented proxy)
    "sample_freq": "W-FRI",                            # weekly obs; the board ASOF-fills to daily
    "backfill_start": "2016-01-01",                    # first run bound; extend + re-run (cached/idempotent)
}


def _cfg() -> dict:
    c = dict(_DEFAULTS)
    c.update(params.get("sentiment", {}).get("vrp", {}) or {})
    return c


# ── Black-76 ATM IV inversion (forward from put-call parity; minimal assumptions) ─────────────── #
def _phi(x: float) -> float:
    """Standard-normal CDF via erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs76_call(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-76 European call PV on a forward F."""
    if sigma <= 0 or T <= 0 or F <= 0 or K <= 0:
        return math.exp(-r * T) * max(F - K, 0.0)
    srt = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * srt * srt) / srt
    d2 = d1 - srt
    return math.exp(-r * T) * (F * _phi(d1) - K * _phi(d2))


def implied_vol_atm(call_mid: float, put_mid: float, K: float, T: float, r: float) -> float:
    """ATM IV by inverting Black-76 on the call, with the forward from put-call parity.

    F = K + (C − P)·e^{rT}  (parity encodes the rate-differential carry, so no q assumption).
    Returns annualized IV, or nan if inputs are unusable.
    """
    if not (np.isfinite(call_mid) and np.isfinite(put_mid) and call_mid > 0 and K > 0 and T > 0):
        return float("nan")
    F = K + (call_mid - put_mid) * math.exp(r * T)
    if F <= 0:
        return float("nan")
    intrinsic = math.exp(-r * T) * max(F - K, 0.0)
    if call_mid <= intrinsic:                     # no time value → cannot invert
        return float("nan")
    lo, hi = 1e-4, 5.0
    for _ in range(100):                          # bisection (BS76 call is monotincreasing in sigma)
        mid = 0.5 * (lo + hi)
        if _bs76_call(F, K, T, r, mid) < call_mid:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── data helpers ──────────────────────────────────────────────────────────────────────────────── #
def _yf_close(ticker: str) -> pd.Series:
    """Daily close for a yfinance ticker as a tz-naive, date-indexed Series ([] on failure)."""
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
        if h is None or len(h) == 0:
            return pd.Series(dtype=float)
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        return h["Close"].astype(float)
    except Exception as exc:
        print(f"  [vrp] yfinance {ticker}: FETCH FAILED ({type(exc).__name__}: {exc})")
        return pd.Series(dtype=float)


def _best_expiry(loader, symbol: str, d: pd.Timestamp, cfg: dict) -> tuple[str, int]:
    """The listed expiration whose DTE from `d` is closest to dte_target within [dte_min, dte_max]."""
    best, best_gap = None, None
    for exp in loader.list_expirations(symbol):
        dte = (pd.Timestamp(exp) - d).days
        if cfg["dte_min"] <= dte <= cfg["dte_max"]:
            gap = abs(dte - cfg["dte_target"])
            if best_gap is None or gap < best_gap:
                best, best_gap = (exp, dte), gap
    return best if best else ("", 0)


def _atm_iv_for_date(loader, symbol: str, d: pd.Timestamp, spot: float, cfg: dict) -> dict | None:
    """One (iv_atm, source, strike, expiry, dte) reading for `symbol` on trading day `d`, or None."""
    exp, dte = _best_expiry(loader, symbol, d, cfg)
    if not exp:
        return None
    chain = loader.get_option_chain(symbol, str(d.date()), exp)
    if chain is None or chain.empty or not np.isfinite(spot):
        return None
    row = chain.iloc[(chain["strike"] - spot).abs().argmin()]      # ATM strike (nearest to spot)
    K = float(row["strike"])
    T = dte / 365.0
    # native IV first (if the tier serves greeks/iv on the chain), else BS-invert from mids
    civ, piv = float(row.get("call_iv", float("nan"))), float(row.get("put_iv", float("nan")))
    native = np.nanmean([v for v in (civ, piv) if np.isfinite(v)]) if (np.isfinite(civ) or np.isfinite(piv)) else float("nan")
    if np.isfinite(native) and native > 0:
        return {"iv_atm": float(native), "iv_source": "native", "atm_strike": K, "expiry": exp, "dte": int(dte)}
    iv = implied_vol_atm(float(row.get("call_mid", float("nan"))), float(row.get("put_mid", float("nan"))),
                         K, T, cfg["risk_free_rate"])
    if not (np.isfinite(iv) and iv > 0):
        return None
    return {"iv_atm": float(iv), "iv_source": "bs_invert", "atm_strike": K, "expiry": exp, "dte": int(dte)}


def _reachable(loader, symbol: str) -> bool:
    try:
        loader.list_expirations(symbol)
        return True
    except Exception:
        return False


# ── feeder entry point ──────────────────────────────────────────────────────────────────────── #
def update(con=None, start: str | None = None) -> dict:
    """Fetch FX-ETF option chains, compute the VRP feature per (pair, weekly obs), upsert. Returns coverage.

    Graceful: if ThetaTerminal is unreachable and no key is set, prints a named message and returns {}.
    """
    from sovereign.research.vrp.data_loader import ThetaDataLoader   # local import: keeps module network-free

    cfg = _cfg()
    start = start or cfg["backfill_start"]
    try:
        key = env_key("THETADATA_API_KEY")
    except RuntimeError:
        key = None
    try:
        base = env_key("THETADATA_BASE_URL")
    except RuntimeError:
        base = "http://127.0.0.1:25503"

    loader = ThetaDataLoader(api_key=key, base_url=base)
    probe_sym = next(iter(cfg["etf"].values()), "FXE")
    if not _reachable(loader, probe_sym):
        print(f"  [vrp] ThetaData not reachable — start ThetaTerminal ({base}) or set THETADATA_API_KEY; skipping")
        return {}

    own = con is None
    con = con or connect()
    now = datetime.now(timezone.utc)
    frames, coverage = [], {}
    for pair, symbol in cfg["etf"].items():
        etf_close = _yf_close(symbol)
        pair_close = _yf_close(cfg["spot"].get(pair, ""))
        if etf_close.empty or pair_close.empty:
            coverage[pair] = {"symbol": symbol, "rows": 0, "note": "no yfinance spot"}
            continue
        rv = np.sqrt(realized_variance(pair_close, window=cfg["rv_window"], forward=False))   # annualized vol, causal
        earliest_exp = loader.earliest_available_date(symbol)

        # weekly sample dates = last real ETF trading day per week, on/after `start`
        idx = etf_close.index[etf_close.index >= pd.Timestamp(start)]
        weekly = pd.Series(idx, index=idx).resample(cfg["sample_freq"]).last().dropna()
        recs, skipped_forbidden = [], 0
        for d in weekly:
            d = pd.Timestamp(d)
            spot = etf_close.asof(d)                      # ETF spot at/just-before D (the option underlying)
            if not np.isfinite(spot):
                continue
            try:
                iv = _atm_iv_for_date(loader, symbol, d, float(spot), cfg)
            except Exception as exc:                      # Value-tier depth wall 403s pre-~2020 chains
                if "403" in str(exc):
                    skipped_forbidden += 1
                    continue
                raise
            if iv is None:
                continue
            rv_d = rv.asof(d)                             # trailing RV, window ends at D (<= D)
            if not np.isfinite(rv_d):
                continue
            recs.append({"date": d.date(), "pair": pair, "symbol": symbol,
                         "expiry": pd.Timestamp(iv["expiry"]).date(), "dte": iv["dte"],
                         "atm_strike": iv["atm_strike"], "iv_atm": iv["iv_atm"],
                         "rv_trailing": float(rv_d), "vrp_signal": iv["iv_atm"] - float(rv_d),
                         "iv_source": iv["iv_source"], "iv_obs_date": d.date(), "rv_last_date": d.date(),
                         "fetched_at": now})
        if not recs:
            coverage[pair] = {"symbol": symbol, "rows": 0, "earliest_expiry": earliest_exp,
                              "skipped_forbidden": skipped_forbidden,
                              "note": "no option data in window"}
            continue
        df = pd.DataFrame(recs).sort_values("date").reset_index(drop=True)
        # trailing percentile of vrp_signal over its OWN history (rich vs cheap; no look-ahead)
        df["vrp_pct"] = df["vrp_signal"].rolling(
            cfg["percentile_window"], min_periods=cfg["percentile_min_periods"]).apply(
            lambda x: float((x <= x[-1]).mean()), raw=True)
        out = df[["date", "pair", "symbol", "expiry", "dte", "atm_strike", "iv_atm", "rv_trailing",
                  "vrp_signal", "vrp_pct", "iv_source", "iv_obs_date", "rv_last_date", "fetched_at"]]
        frames.append(out)
        src = out["iv_source"].value_counts().to_dict()
        coverage[pair] = {"symbol": symbol, "rows": int(len(out)), "start": str(out["date"].min()),
                          "end": str(out["date"].max()), "earliest_expiry": earliest_exp,
                          "skipped_forbidden": skipped_forbidden,
                          "iv_source": {k: int(v) for k, v in src.items()}}
    if frames:
        upsert(con, "sentiment_vrp_daily", pd.concat(frames, ignore_index=True), ["date", "pair"])
    if own:
        con.close()
    return coverage
