"""
Shared entry-decision engine for the MES/MNQ learning agent.

ONE engine, called by BOTH the live monitor (scripts/futures_monitor.py) and the replay
(scripts/futures_replay.py) so live == replay. It runs the setup ladder (ORB -> VWAP-MR -> micro),
applies gates, and returns a fully-populated EntryDecision carrying everything the reasoning layer
needs: setup, direction, stop/target, expected R, confluence (POC/VAH/VAL), CVD slope+quality+confirm,
regime, ADR-used, key levels, falsifier, time-gate, confidence — and `would_have_blocked`, the list of
STRICT gates a learning-mode entry bypassed.

Learning mode (the paper-month default): bypasses the session-window + regime gates and loosens the
volume gate to maximize reps, while recording what strict mode would have blocked. Strict mode is
validation-only and fires exactly as the legacy scalp_strategy functions do (equivalence-tested).

Trigger math is kept identical to sovereign.futures.scalp_strategy (asserted by
tests/unit/test_futures_decision_engine.py) so there is no drift between this and the source of truth.

Sandbox-local: no forex/ICT/intelligence imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from sovereign.futures import scalp_strategy as strat
from sovereign.futures import cvd as cvd_mod
from sovereign.futures import regime as regime_mod
from sovereign.futures import volume_profile as vp
from sovereign.futures.config import futures_params, contract_spec


@dataclass
class EntryDecision:
    setup_type: str               # ORB | VWAP_MR | MICRO
    direction: str                # LONG | SHORT
    entry: float
    stop: float
    target: float
    expected_r: float
    contracts: int
    # telemetry (null-safe — None where the input is unavailable)
    confluence: int = 0
    cvd_slope: Optional[float] = None
    cvd_quality: Optional[str] = None       # HIGH | LOW | None(unknown)
    cvd_confirmed: Optional[bool] = None
    regime_state: Optional[str] = None      # TRENDING | OSCILLATING | None
    adr_used_pct: Optional[float] = None
    vwap: Optional[float] = None
    rsi: Optional[float] = None
    key_levels: dict = field(default_factory=dict)
    falsifier: Optional[float] = None
    falsifier_text: Optional[str] = None
    time_gate: str = "UNKNOWN"              # OPEN | CLOSE | MIDDAY
    confidence: str = "MEDIUM"             # HIGH | MEDIUM | LOW
    would_have_blocked: list = field(default_factory=list)
    learning_mode: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ── helpers ──────────────────────────────────────────────────────────────────

def _cvd_quality(state: Optional[dict]) -> Optional[str]:
    if state is None:
        return None
    return "HIGH" if cvd_mod.is_strong(state) else "LOW"


def _time_gate_label(ts: datetime) -> str:
    """OPEN / CLOSE / MIDDAY from the configured ET session windows."""
    from zoneinfo import ZoneInfo
    sw = futures_params()["session_windows"]
    et = ts.astimezone(ZoneInfo("America/New_York"))
    hm = et.strftime("%H:%M")
    for label, key in (("OPEN", "open"), ("CLOSE", "close")):
        lo, hi = sw[key]
        if lo <= hm <= hi:
            return label
    return "MIDDAY"


def _volume_ok(curr: strat.Indicators, mult: float) -> bool:
    return not (curr.avg_volume > 0 and curr.curr_volume < mult * curr.avg_volume)


def _confidence(confluence: int, cvd_confirmed: Optional[bool], conviction: int,
                would_have_blocked: list) -> str:
    if confluence >= 2 and cvd_confirmed is True and conviction >= 2:
        return "HIGH"
    if cvd_confirmed is False or would_have_blocked:
        return "LOW"
    return "MEDIUM"


# ── the one decision function ────────────────────────────────────────────────

def evaluate_entry(
    window,                       # RTH 1-min OHLCV DataFrame up to `now`
    *,
    bias: dict,                   # {bias, conviction, key_levels:{overnight_high/low,...}}
    ts: datetime,                 # tz-aware current bar timestamp
    instrument: str,
    prev_ind: Optional[strat.Indicators] = None,
    orb_levels: Optional[tuple] = None,   # (orb_high, orb_low) or None
    orb_taken: bool = False,
    regime: Optional[dict] = None,        # classify_session(...) or None
    prior_profile: Optional[dict] = None, # POC/VAH/VAL or None
    last_entry_time: Optional[datetime] = None,
    trades_taken: int = 0,
    learning_mode: bool = False,
    oracle_invalidation: Optional[float] = None,
) -> Optional[EntryDecision]:
    """Evaluate the setup ladder; return an EntryDecision or None. Pure (no I/O)."""
    p = futures_params()
    try:
        ind = strat.compute_indicators(window)
    except Exception:
        return None

    bias_dir = bias.get("bias", "NEUTRAL")
    conviction = int(bias.get("conviction", 0) or 0)
    key_levels = dict(bias.get("key_levels", {}) or {})
    on_low = key_levels.get("overnight_low")
    on_high = key_levels.get("overnight_high")
    tick = contract_spec(instrument)["tick"]
    tol_price = p["volume_profile"]["confluence_tol_ticks"] * tick
    price = ind.last_price

    window_ok = strat.in_trade_window(ts)
    lm = p.get("learning_mode", {})
    strict_mult = p["micro"]["volume_gate_mult"]
    loose_mult = float(lm.get("volume_gate_mult", 0.0)) if learning_mode else strict_mult
    cstate = cvd_mod.cvd_state(window)

    def _cap_cooldown_ok(setup_cfg_key: str) -> bool:
        # ORB has no cap/cooldown (it's gated once-per-session by orb_taken) — default safe.
        m = p.get(setup_cfg_key, {})
        if trades_taken >= m.get("max_trades_per_session", 10 ** 9):
            return False
        cd = m.get("cooldown_seconds", 0)
        if last_entry_time is not None and (ts - last_entry_time).total_seconds() < cd:
            return False
        return True

    # candidate = (setup, direction, stop, target, contracts, expected_r, strict_gate_fails)
    candidate = None

    # ── 1) ORB (bias-aligned, once/session) ───────────────────────────────────
    if candidate is None and orb_levels is not None and not orb_taken and bias_dir in ("LONG", "SHORT"):
        orb_hi, orb_lo = orb_levels
        raw = "LONG" if price > orb_hi else ("SHORT" if price < orb_lo else None)
        if raw == bias_dir and _cap_cooldown_ok("orb"):
            blocked = []
            if not window_ok:
                blocked.append("session_window")
            if not _volume_ok(ind, p["orb"]["volume_gate_mult"]):
                blocked.append("volume")
            if regime is not None:
                ok, why = regime_mod.setup_allowed("orb", regime)
                if not ok:
                    blocked.append(f"regime:{why}")
            fires = learning_mode or not blocked
            if fires and (learning_mode or _volume_ok(ind, p["orb"]["volume_gate_mult"])):
                stop = orb_lo if raw == "LONG" else orb_hi
                tgt_pts = p["orb"]["safe_target_points"]
                target = price + tgt_pts if raw == "LONG" else price - tgt_pts
                candidate = ("ORB", raw, stop, target, p["orb"]["safe_contracts"], blocked)

    # ── 2) VWAP mean-reversion (no bias; regime-gated) ────────────────────────
    if (candidate is None and _cap_cooldown_ok("vwap_mr")
            and len(window) >= p["vwap_mr"]["band_lookback"]):   # need real structure for a band
        bands = strat.vwap_bands(window, p["vwap_mr"]["n_sigma"])
        if bands is not None and bands[3] > 0:
            lower, upper, _v, _s = bands
            raw = "LONG" if price <= lower else ("SHORT" if price >= upper else None)
            if raw:
                blocked = []
                if not window_ok:
                    blocked.append("session_window")
                if regime is not None:
                    ok, why = regime_mod.setup_allowed("vwap_mr", regime)
                    if not ok:
                        blocked.append(f"regime:{why}")
                fires = learning_mode or not blocked
                if fires:
                    stop, target = strat.vwap_mr_levels(raw, price, bands, instrument)
                    candidate = ("VWAP_MR", raw, stop, target, 1, blocked)

    # ── 3) MICRO (VWAP reclaim + RSI cross, bias-aligned) ─────────────────────
    if candidate is None and prev_ind is not None and bias_dir in ("LONG", "SHORT") and _cap_cooldown_ok("micro"):
        lvl = p["micro"]["rsi_cross_level"]
        above_both = price > ind.ema_fast and price > ind.ema_slow
        below_both = price < ind.ema_fast and price < ind.ema_slow
        long_x = (prev_ind.last_price < prev_ind.vwap and price >= ind.vwap
                  and prev_ind.rsi < lvl and ind.rsi >= lvl and above_both)
        short_x = (prev_ind.last_price > prev_ind.vwap and price <= ind.vwap
                   and prev_ind.rsi > lvl and ind.rsi <= lvl and below_both)
        raw = "LONG" if (long_x and bias_dir == "LONG") else ("SHORT" if (short_x and bias_dir == "SHORT") else None)
        if raw:
            blocked = []
            if not window_ok:
                blocked.append("session_window")
            if not _volume_ok(ind, strict_mult):
                blocked.append("volume")
            fires = (learning_mode and _volume_ok(ind, loose_mult)) or (not blocked)
            if fires:
                stop = strat.compute_stop(raw, price, on_low, on_high)
                target = strat.target_from_rr(raw, price, stop)
                candidate = ("MICRO", raw, stop, target, 1, blocked)

    if candidate is None:
        return None

    setup, direction, stop, target, contracts, blocked = candidate
    risk = abs(price - stop) or 1e-9
    expected_r = round(abs(target - price) / risk, 2)
    cvd_confirmed = cvd_mod.cvd_confirms(setup, direction, cstate)
    confluence = vp.confluence_score(price, prior_profile, tol_price)
    falsifier = strat.kill_level(bias_dir, key_levels, oracle_invalidation)
    kl = dict(key_levels)
    if orb_levels is not None:
        kl["orb_high"], kl["orb_low"] = round(orb_levels[0], 2), round(orb_levels[1], 2)
    kl["vwap"] = round(ind.vwap, 2)
    if prior_profile:
        for k in ("poc", "vah", "val"):
            if prior_profile.get(k) is not None:
                kl[k] = prior_profile[k]

    return EntryDecision(
        setup_type=setup, direction=direction, entry=round(price, 2), stop=round(stop, 2),
        target=round(target, 2), expected_r=expected_r, contracts=contracts,
        confluence=confluence,
        cvd_slope=(round(cstate["slope"], 2) if cstate else None),
        cvd_quality=_cvd_quality(cstate), cvd_confirmed=cvd_confirmed,
        regime_state=(regime or {}).get("trend_state"),
        adr_used_pct=(regime or {}).get("adr_used_pct"),
        vwap=round(ind.vwap, 2), rsi=round(ind.rsi, 1),
        key_levels=kl,
        falsifier=(round(falsifier, 2) if falsifier is not None else None),
        falsifier_text=(f"{'Below' if bias_dir == 'LONG' else 'Above'} {falsifier:.2f} invalidates"
                        if falsifier is not None else None),
        time_gate=_time_gate_label(ts),
        confidence=_confidence(confluence, cvd_confirmed, conviction, blocked),
        would_have_blocked=blocked,
        learning_mode=learning_mode,
    )
