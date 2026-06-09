"""Pure ES/NQ scalp + ORB strategy logic — the single source of truth.

Sandbox-local: no I/O, no IB, no forex/ICT/intelligence imports. Both the live
terminal (scripts/futures_monitor.py) and the nightly replay (scripts/futures_replay.py)
import THIS module, so what you backtest is exactly what you trade.

Time is always passed in (a wall-clock `now` live, a bar timestamp in replay) — never
read from the clock here — so the cooldown logic replays deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sovereign.futures.config import futures_params


@dataclass(frozen=True)
class Indicators:
    last_price: float
    vwap: float
    rsi: float
    curr_volume: float
    avg_volume: float
    ema_fast: float
    ema_slow: float


def compute_indicators(bars) -> Indicators:
    """Session VWAP, RSI, EMAs and volume from a 1-min OHLCV DataFrame (RTH-filtered).

    `bars` must have columns High, Low, Close, Volume. Raises ValueError if empty.
    """
    if bars is None or len(bars) == 0:
        raise ValueError("empty bars")
    p = futures_params()["micro"]
    bars = bars.copy()

    last_price = float(bars["Close"].iloc[-1])

    typical = (bars["High"] + bars["Low"] + bars["Close"]) / 3.0
    cum_tp_vol = (typical * bars["Volume"]).cumsum()
    cum_vol = bars["Volume"].cumsum()
    vwap = float((cum_tp_vol / cum_vol).iloc[-1])

    delta = bars["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    alpha = 1.0 / p["rsi_period"]
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])

    curr_volume = float(bars["Volume"].iloc[-1])
    avg_volume = float(bars["Volume"].tail(p["vol_lookback"]).mean())
    ema_fast = float(bars["Close"].ewm(span=p["ema_fast"], adjust=False).mean().iloc[-1])
    ema_slow = float(bars["Close"].ewm(span=p["ema_slow"], adjust=False).mean().iloc[-1])

    return Indicators(last_price, vwap, rsi, curr_volume, avg_volume, ema_fast, ema_slow)


def micro_signal(
    bias_dir: str,
    curr: Indicators,
    prev: Indicators,
    *,
    now: datetime,
    last_entry_time: Optional[datetime],
    trades_taken: int,
) -> Optional[str]:
    """VWAP reclaim + RSI cross, structurally gated. Returns 'LONG' | 'SHORT' | None.

    Gates (all from config/futures_params.yml::micro):
      - bias_dir must be LONG/SHORT and match the signal direction
      - per-session trade cap and cooldown (uses passed `now`, not the clock)
      - volume confirmation: signal-bar volume >= mult * avg volume
      - EMA position filter: price on the correct side of BOTH emas
      - VWAP cross with an RSI cross through the level
    """
    if bias_dir not in ("LONG", "SHORT"):
        return None
    m = futures_params()["micro"]
    if trades_taken >= m["max_trades_per_session"]:
        return None
    if last_entry_time is not None:
        if (now - last_entry_time).total_seconds() < m["cooldown_seconds"]:
            return None
    if curr.avg_volume > 0 and curr.curr_volume < m["volume_gate_mult"] * curr.avg_volume:
        return None

    above_both = curr.last_price > curr.ema_fast and curr.last_price > curr.ema_slow
    below_both = curr.last_price < curr.ema_fast and curr.last_price < curr.ema_slow
    lvl = m["rsi_cross_level"]
    long_signal = (prev.last_price < prev.vwap and curr.last_price >= curr.vwap
                   and prev.rsi < lvl and curr.rsi >= lvl)
    short_signal = (prev.last_price > prev.vwap and curr.last_price <= curr.vwap
                    and prev.rsi > lvl and curr.rsi <= lvl)
    if long_signal and bias_dir == "LONG" and above_both:
        return "LONG"
    if short_signal and bias_dir == "SHORT" and below_both:
        return "SHORT"
    return None


# ── VWAP mean-reversion setup (Increment 3) ───────────────────────────────────

def vwap_bands(bars, n_sigma: Optional[float] = None) -> Optional[tuple[float, float, float, float]]:
    """Session VWAP ± n_sigma · (volume-weighted σ of typical price around VWAP).
    Returns (lower, upper, vwap, sigma) or None if not enough volume."""
    if bars is None or len(bars) == 0:
        return None
    if n_sigma is None:
        n_sigma = futures_params()["vwap_mr"]["n_sigma"]
    typical = (bars["High"] + bars["Low"] + bars["Close"]) / 3.0
    vol = bars["Volume"]
    vsum = float(vol.sum())
    if vsum <= 0:
        return None
    vwap = float((typical * vol).cumsum().iloc[-1] / vol.cumsum().iloc[-1])
    var = float((vol * (typical - vwap) ** 2).sum() / vsum)
    sigma = var ** 0.5
    return (vwap - n_sigma * sigma, vwap + n_sigma * sigma, vwap, sigma)


def vwap_mr_signal(bars, curr: Indicators, *, now: datetime,
                   last_entry_time: Optional[datetime], trades_taken: int) -> Optional[str]:
    """Fade a stretch from VWAP: LONG at/below −nσ, SHORT at/above +nσ. Returns dir|None.

    A mean-reversion setup — does NOT take a directional bias (the regime router decides
    WHEN it may fire). Cooldown / cap from config/futures_params.yml::vwap_mr."""
    m = futures_params()["vwap_mr"]
    if trades_taken >= m["max_trades_per_session"]:
        return None
    if last_entry_time is not None and (now - last_entry_time).total_seconds() < m["cooldown_seconds"]:
        return None
    bands = vwap_bands(bars, m["n_sigma"])
    if bands is None:
        return None
    lower, upper, _vwap, sigma = bands
    if sigma <= 0:
        return None
    if curr.last_price <= lower:
        return "LONG"
    if curr.last_price >= upper:
        return "SHORT"
    return None


def vwap_mr_levels(direction: str, entry: float, bands: tuple[float, float, float, float],
                   instrument: str) -> tuple[float, float]:
    """(stop, target) for a VWAP-MR entry: target = VWAP (the mean); stop = a buffer
    beyond BOTH the entry and the band, so a LONG's stop is always < entry and a SHORT's
    stop is always > entry (guards the degenerate case where price has already pierced the
    band past the buffer — which otherwise produced a stop on the wrong side / absurd R)."""
    lower, upper, vwap, _sigma = bands
    p = futures_params()["vwap_mr"]
    spec = futures_params()["contracts"][{"ES": "MES", "NQ": "MNQ"}.get(instrument.upper(), instrument.upper())]
    buf = p["stop_buffer_ticks"] * spec["tick"]
    if direction == "LONG":
        return min(entry, lower) - buf, vwap     # always strictly below entry
    return max(entry, upper) + buf, vwap         # always strictly above entry


# ── Time-of-day gate (Increment 3) ────────────────────────────────────────────

def in_trade_window(ts: datetime) -> bool:
    """True if `ts` (tz-aware) falls in an allowed ET trading window. Midday is blocked.
    Returns True (no gating) if session_windows.enabled is false."""
    from zoneinfo import ZoneInfo
    sw = futures_params()["session_windows"]
    if not sw.get("enabled", True):
        return True
    et = ts.astimezone(ZoneInfo("America/New_York"))
    hm = et.hour * 60 + et.minute

    def _win(pair):
        (a, b) = pair
        ah, am = map(int, a.split(":")); bh, bm = map(int, b.split(":"))
        return ah * 60 + am <= hm < bh * 60 + bm
    return _win(sw["open"]) or _win(sw["close"])


def orb_range(bars, minutes: Optional[int] = None) -> Optional[tuple[float, float]]:
    """High/low of the first `minutes` 1-min RTH bars. None if not enough bars yet."""
    if minutes is None:
        minutes = futures_params()["orb"]["minutes"]
    if bars is None or len(bars) < minutes:
        return None
    window = bars.iloc[:minutes]
    return float(window["High"].max()), float(window["Low"].min())


def orb_break(bias_dir: str, price: float, orb_high: float, orb_low: float,
              curr: Indicators) -> Optional[str]:
    """ORB breakout in the bias direction with volume confirmation. Returns dir or None."""
    if bias_dir not in ("LONG", "SHORT"):
        return None
    o = futures_params()["orb"]
    if not (curr.avg_volume > 0 and curr.curr_volume >= o["volume_gate_mult"] * curr.avg_volume):
        return None
    direction = "LONG" if price > orb_high else ("SHORT" if price < orb_low else None)
    return direction if direction == bias_dir else None


def compute_stop(direction: str, entry: float,
                 overnight_low: Optional[float], overnight_high: Optional[float]) -> float:
    """Stop at the overnight level if available, else a fixed fallback % from entry."""
    s = futures_params()["stops"]
    if s["use_overnight_levels"]:
        if direction == "LONG" and overnight_low:
            return float(overnight_low)
        if direction == "SHORT" and overnight_high:
            return float(overnight_high)
    fb = s["fallback_pct"]
    return entry * (1 - fb) if direction == "LONG" else entry * (1 + fb)


def target_from_rr(direction: str, entry: float, stop: float,
                   rr: Optional[float] = None) -> float:
    """T1 at `rr` * (entry-stop distance). Defaults to micro.target_rr."""
    if rr is None:
        rr = futures_params()["micro"]["target_rr"]
    risk = abs(entry - stop)
    return entry + rr * risk if direction == "LONG" else entry - rr * risk


def kill_level(bias_dir: str, key_levels: dict,
               oracle_invalidation: Optional[float] = None) -> Optional[float]:
    """Price beyond which the bias is DEAD — the soonest of the rules invalidation
    (overnight high/low) and the oracle's falsifier. SHORT dies ABOVE, LONG dies BELOW."""
    if bias_dir not in ("LONG", "SHORT"):
        return None
    levels: list[float] = []
    kl = key_levels or {}
    rules = kl.get("overnight_high") if bias_dir == "SHORT" else kl.get("overnight_low")
    if isinstance(rules, (int, float)):
        levels.append(float(rules))
    if isinstance(oracle_invalidation, (int, float)):
        levels.append(float(oracle_invalidation))
    if not levels:
        return None
    return min(levels) if bias_dir == "SHORT" else max(levels)


def bias_invalidated(bias_dir: str, price: float, kill: Optional[float]) -> bool:
    """True once price crosses the kill level against the bias."""
    if kill is None or bias_dir not in ("LONG", "SHORT"):
        return False
    return (price > kill) if bias_dir == "SHORT" else (price < kill)


def sizing_rationale(trades_taken: int, session_r: float) -> str:
    """probe (first trade) / press (winning) / reduce (losing)."""
    if trades_taken == 0:
        return "probe"
    return "press" if session_r > 0 else "reduce"


def compute_r(entry: float, stop: float, exit_price: float, direction: str) -> float:
    """R-multiple: 1R = entry→stop distance. Mirrors scripts/futures_log.py::_compute_r."""
    risk = abs(entry - stop)
    if risk == 0:
        return 0.0
    if direction == "LONG":
        return round((exit_price - entry) / risk, 3)
    return round((entry - exit_price) / risk, 3)
