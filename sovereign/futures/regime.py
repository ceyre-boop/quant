"""Regime classification + per-setup routing for the futures sandbox (Increment 3).

Sandbox-local: no forex/ICT/intelligence imports. The single highest-leverage idea —
ORB and VWAP mean-reversion are opposite-regime strategies, so each must fire ONLY in
its favorable regime or they bleed in each other's. This is the same move that lifted
the forex USDJPY book.

  - classify_session(): label a day OSCILLATING (chop) vs TRENDING (one-sided around VWAP),
    with ADR-used% and VIX context.
  - setup_allowed(): given a regime, may this setup fire? (config-driven router.)
"""
from __future__ import annotations

from typing import Optional

from sovereign.futures.config import futures_params


def _frac_above_vwap(day_df) -> float:
    """Fraction of bars whose close sits above the running session VWAP."""
    typical = (day_df["High"] + day_df["Low"] + day_df["Close"]) / 3.0
    vol = day_df["Volume"]
    cum_vol = vol.cumsum()
    if float(cum_vol.iloc[-1]) <= 0:
        # no volume — fall back to a simple mean line
        mean_line = day_df["Close"].expanding().mean()
        return float((day_df["Close"] > mean_line).mean())
    vwap_run = (typical * vol).cumsum() / cum_vol.replace(0, float("nan"))
    return float((day_df["Close"] > vwap_run).mean())


def adr_used_from_ranges(day_range: float, prior_ranges: list[float]) -> Optional[float]:
    """Session range as a fraction of ADR (mean of up to 14 prior daily ranges)."""
    rs = [r for r in prior_ranges if r and r > 0][-14:]
    if not rs:
        return None
    adr = sum(rs) / len(rs)
    return round(day_range / adr, 3) if adr > 0 else None


def classify_session(day_df, vix: Optional[float] = None,
                     adr_used_pct: Optional[float] = None) -> dict:
    """Label a session. trend_state OSCILLATING|TRENDING from how one-sided price is
    around VWAP; carries ADR-used% and VIX for the router."""
    r = futures_params()["regime"]
    frac = _frac_above_vwap(day_df)
    trend_state = "OSCILLATING" if abs(frac - 0.5) <= r["oscillating_band"] else "TRENDING"
    return {
        "trend_state": trend_state,
        "frac_above_vwap": round(frac, 3),
        "adr_used_pct": adr_used_pct,
        "vix": vix,
    }


def setup_allowed(setup: str, regime: dict) -> tuple[bool, str]:
    """May `setup` ('orb'|'vwap_mr') fire in this regime? Returns (allowed, reason)."""
    r = futures_params()["regime"]
    ts = regime.get("trend_state")
    if setup == "vwap_mr":
        if ts not in r["vwap_mr_regimes"]:
            return False, f"VWAP-MR needs {r['vwap_mr_regimes']}, regime is {ts}"
        adr = regime.get("adr_used_pct")
        if adr is not None and adr > r["adr_used_low"]:
            return False, f"ADR-used {adr:.0%} > {r['adr_used_low']:.0%} (too directional)"
        vix = regime.get("vix")
        if vix is not None and vix > r["vix_high"]:
            return False, f"VIX {vix:.1f} > {r['vix_high']} (event/blowout risk)"
        return True, "oscillating, low-ADR, calm"
    if setup == "orb":
        if ts not in r["orb_regimes"]:
            return False, f"ORB needs {r['orb_regimes']}, regime is {ts}"
        return True, "directional open"
    # unknown setup (e.g. legacy 'micro') — no regime restriction
    return True, "ungated"
