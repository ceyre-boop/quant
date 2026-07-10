"""Strategy distribution feeds (TICK-022 P3).

Every feed returns a TradePool: normalized R-multiples (or a parametric sampler),
the sizing that reproduces the source strategy, a trade-frequency clock, and an
EVIDENCE STAMP. The stamp is the honesty layer — the funnel simulates UNPROVEN
strategies identically to PROVEN ones, so the stamp (and its caveat) must travel
with every downstream number.

Return-scale convention (Open Question #1 for Colin, non-blocking): R-multiples
via R = pnl_pct / risk_pct, sized at base_risk_pct — the monte_carlo_prop
convention (%-equity per trade), NOT the equity_curve.py display convention
(risk_adjusted_pnl_pct, ~100x smaller). Sharpe is scale-invariant; dollars are not.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from research.prop_funnel._lib import ROOT, EvidenceStamp

OOS_POOL = ROOT / "data" / "risk" / "oos_trades_2023_2024.json"
DECADE_CSV = ROOT / "data" / "proof" / "backtest_trades_v015_2015_2024.csv"
DECISION_LOG_DIR = ROOT / "data" / "decision_logs"
ICT_WINDOW_FILES = {
    "london_a": ROOT / "logs" / "ict_backtest_london_a.json",
    "london_all": ROOT / "logs" / "ict_backtest_london_all.json",
    "window_A": ROOT / "logs" / "ict_backtest_window_A.json",
    "window_B": ROOT / "logs" / "ict_backtest_window_B.json",
    "results": ROOT / "logs" / "ict_backtest_results.json",
}
FUTURES_REPLAY_GLOB = "replay_report_*.json"

MIN_POOL_N = 30

# Copied VERBATIM from sovereign/risk/monte_carlo_prop.py's regime_caveat (2026-07-10)
# so it travels with every carry verdict row even when that module isn't loaded.
CARRY_REGIME_CAVEAT = (
    "CRITICAL: this bootstraps the 2023-2024 OOS window only — a FAVORABLE, rate-trending "
    "regime. The forex edge is REGIME-FRAGILE: rolling walk-forward was 2021 -0.13 / 2022 +0.51 "
    "/ 2023 +1.26 / 2024 -0.09. In a flat/adverse regime these numbers would be materially "
    "worse. Fresh 2025-26 measured ≈ FLAT (v015 remeasurement, 2026-06-27)."
)
ICT_CAVEAT = "ICT edge UNPROVEN: permutation p=0.52, fails BH (2026-06-30 audit). Backtest pools only."
LIVE_CAVEAT = ("n=27 closed live outcomes (3 WIN / 24 LOSS, WR 11%) vs backtest WR ~63.6% — "
               "selection-biased (most signals EXPIRED unexecuted) and too small for MC; sanity anchor only.")


@dataclass
class TradePool:
    name: str
    stamp: EvidenceStamp
    base_risk_pct: float
    trades_per_day: float                    # trading-day clock (252/yr)
    r_values: Optional[np.ndarray] = None    # empirical pool (bootstrap sampler)
    param_mu_r: Optional[float] = None       # parametric t sampler: per-trade mean R
    param_t_df: int = 4
    caveat: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return 0 if self.r_values is None else int(len(self.r_values))

    @property
    def sufficient(self) -> bool:
        return self.param_mu_r is not None or self.n >= MIN_POOL_N

    @property
    def kind(self) -> str:
        return "parametric_t" if self.param_mu_r is not None else "bootstrap"

    def draw(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Per-trade R draws: iid bootstrap of the empirical pool, or standardized
        student-t(df) around the parametric mean (unit per-trade variance)."""
        if self.param_mu_r is not None:
            t = rng.standard_t(self.param_t_df, size=size)
            t /= np.sqrt(self.param_t_df / (self.param_t_df - 2.0))
            return self.param_mu_r + t
        if not self.sufficient:
            raise ValueError(f"pool {self.name!r} has n={self.n} < {MIN_POOL_N} — refusing to MC "
                             f"an insufficient pool; report it as INSUFFICIENT_DATA instead")
        return rng.choice(self.r_values, size=size, replace=True)

    def sharpe_ann(self) -> Optional[float]:
        """Trade-clock annualized Sharpe of the pool itself."""
        if self.param_mu_r is not None:
            return self.param_mu_r * np.sqrt(252.0 * self.trades_per_day)
        if self.n < 2:
            return None
        m, s = float(self.r_values.mean()), float(self.r_values.std())
        if s == 0:
            return None
        return m / s * float(np.sqrt(252.0 * self.trades_per_day))


def _years_span(dates: list[str]) -> float:
    ds = sorted(d[:10] for d in dates if d)
    d0 = datetime.fromisoformat(ds[0])
    d1 = datetime.fromisoformat(ds[-1])
    return max((d1 - d0).days / 365.25, 1e-9)


# ── Carry (PROVEN, regime-fragile) ──────────────────────────────────────────

def load_carry_oos() -> TradePool:
    if not OOS_POOL.exists():
        raise SystemExit(f"FATAL: frozen OOS pool missing: {OOS_POOL} "
                         f"(regen: python3 scripts/generate_oos_pool.py)")
    data = json.loads(OOS_POOL.read_text())
    rs, dates, per_pair = [], [], {}
    for pair, trades in data.items():
        vals = [float(t["pnl_pct"]) / float(t["risk_pct"]) for t in trades
                if t.get("risk_pct") and "pnl_pct" in t]
        per_pair[pair] = len(vals)
        rs.extend(vals)
        dates.extend(str(t.get("entry_date", "")) for t in trades)
    if not rs:
        raise SystemExit("FATAL: carry OOS pool is empty — refusing to fabricate")
    years = _years_span(dates)
    risk_pcts = {float(t["risk_pct"]) for tr in data.values() for t in tr if t.get("risk_pct")}
    base_risk = sorted(risk_pcts)[0] if len(risk_pcts) == 1 else 0.0075
    return TradePool(
        name="carry_oos", stamp=EvidenceStamp.PROVEN_REGIME_FRAGILE,
        base_risk_pct=base_risk, trades_per_day=len(rs) / years / 252.0,
        r_values=np.asarray(rs, dtype=float), caveat=CARRY_REGIME_CAVEAT,
        meta={"source": str(OOS_POOL.relative_to(ROOT)), "per_pair": per_pair,
              "years_span": round(years, 2), "risk_pcts_seen": sorted(risk_pcts)},
    )


def load_carry_decade() -> TradePool:
    if not DECADE_CSV.exists():
        raise SystemExit(f"FATAL: decade trade CSV missing: {DECADE_CSV}")
    rs, dates = [], []
    with DECADE_CSV.open() as fh:
        for row in csv.DictReader(fh):
            risk = float(row.get("risk_pct") or 0)
            if risk <= 0:
                continue
            rs.append(float(row["pnl_pct"]) / risk)
            dates.append(row["entry_date"])
    if not rs:
        raise SystemExit("FATAL: decade CSV parsed to zero usable trades")
    years = _years_span(dates)
    return TradePool(
        name="carry_decade", stamp=EvidenceStamp.PROVEN_REGIME_FRAGILE,
        base_risk_pct=0.0075, trades_per_day=len(rs) / years / 252.0,
        r_values=np.asarray(rs, dtype=float),
        caveat="Full-decade pool (Sharpe ~0.69): includes both paying and flat regimes.",
        meta={"source": str(DECADE_CSV.relative_to(ROOT)), "years_span": round(years, 2)},
    )


def carry_scenario(pool: TradePool, target_sharpe_ann: float) -> TradePool:
    """Mean-shift a carry pool to a target annualized Sharpe (forward-band SCENARIO).
    shift = (S_target - S_pool) * std_R / sqrt(252*tpd), applied per trade."""
    s_pool = pool.sharpe_ann()
    std = float(pool.r_values.std())
    shift = (target_sharpe_ann - s_pool) * std / float(np.sqrt(252.0 * pool.trades_per_day))
    return TradePool(
        name=f"carry_fwd_S{target_sharpe_ann:g}", stamp=EvidenceStamp.SCENARIO,
        base_risk_pct=pool.base_risk_pct, trades_per_day=pool.trades_per_day,
        r_values=pool.r_values + shift,
        caveat=(f"SCENARIO: {pool.name} mean-shifted from Sharpe {s_pool:.2f} to "
                f"{target_sharpe_ann:g} ('if carry forward-Sharpe were {target_sharpe_ann:g}'). "
                + CARRY_REGIME_CAVEAT),
        meta={"parent": pool.name, "shift_per_trade_R": round(shift, 6),
              "parent_sharpe": round(s_pool, 4)},
    )


# ── ICT (UNPROVEN) ──────────────────────────────────────────────────────────

def load_ict_window(window: str, trades_per_day: float = 0.8) -> TradePool:
    path = ICT_WINDOW_FILES[window]
    if not path.exists():
        raise SystemExit(f"FATAL: ICT window file missing: {path}")
    raw = json.loads(path.read_text())
    rs = np.asarray([float(t["pnl_r"]) for t in raw.get("trades", [])], dtype=float)
    if rs.size == 0:
        raise SystemExit(f"FATAL: no trades in {path.name}")
    return TradePool(
        name=f"ict_{window}", stamp=EvidenceStamp.UNPROVEN,
        base_risk_pct=0.0075, trades_per_day=trades_per_day,
        r_values=rs, caveat=ICT_CAVEAT,
        meta={"source": str(path.relative_to(ROOT)), "stats": raw.get("stats", {}),
              "tpd_sensitivity": [0.4, 0.8, 1.6]},
    )


# ── Live closed outcomes (LOW_N sanity anchor, NOT an MC feed) ──────────────

def load_live_closed_outcomes() -> TradePool:
    rows = []
    for f in sorted(DECISION_LOG_DIR.glob("decisions_*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("outcome") in ("WIN", "LOSS") and r.get("r_realized") is not None:
                rows.append(r)
    rs = np.asarray([float(r["r_realized"]) for r in rows], dtype=float)
    by_system: dict = {}
    by_source: dict = {}
    for r in rows:
        by_system[r.get("system")] = by_system.get(r.get("system"), 0) + 1
        src = r.get("source") or "live"
        by_source[src] = by_source.get(src, 0) + 1
    wins = sum(1 for r in rows if r["outcome"] == "WIN")
    return TradePool(
        name="live_closed_outcomes", stamp=EvidenceStamp.LOW_N_SANITY_ONLY,
        base_risk_pct=0.0075, trades_per_day=0.0,
        r_values=rs, caveat=LIVE_CAVEAT,
        meta={"n": int(rs.size), "wins": wins, "losses": int(rs.size) - wins,
              "by_system": by_system, "by_source": by_source},
    )


# ── Futures ORB (UNVALIDATED) ───────────────────────────────────────────────

def load_futures_orb() -> TradePool:
    rs, files = [], []
    for f in sorted((ROOT / "data" / "futures").glob(FUTURES_REPLAY_GLOB)):
        files.append(f.name)
        raw = json.loads(f.read_text())
        sessions = raw.get("sessions", [raw])
        for s in sessions:
            for t in s.get("trades", []):
                if t.get("r_realized") is not None:
                    rs.append(float(t["r_realized"]))
    return TradePool(
        name="futures_orb", stamp=EvidenceStamp.UNVALIDATED,
        base_risk_pct=0.0075, trades_per_day=1.0,
        r_values=np.asarray(rs, dtype=float),
        caveat=("UNVALIDATED: n_real=0 live (futures_validation.json passed:false); replay pool "
                "is in-sample and tiny. Regenerate via Phase R (operator-gated) before trusting."),
        meta={"files": files},
    )


# ── Synthetic frontier ──────────────────────────────────────────────────────

def synthetic_pool(sharpe_ann: float, trades_per_day: float, t_df: int = 4) -> TradePool:
    """Parametric per-trade sampler: R = s_trade + standardized t(df); unit per-trade
    variance so daily $ vol ≈ risk_pct * sqrt(trades_per_day) * equity."""
    s_trade = sharpe_ann / float(np.sqrt(252.0 * trades_per_day))
    return TradePool(
        name=f"synthetic_S{sharpe_ann:g}_f{trades_per_day:g}", stamp=EvidenceStamp.SYNTHETIC,
        base_risk_pct=0.01, trades_per_day=trades_per_day,
        param_mu_r=s_trade, param_t_df=t_df,
        caveat="SYNTHETIC: existence not claimed — requirements frontier only.",
        meta={"sharpe_ann": sharpe_ann, "per_trade_mean_R": round(s_trade, 6), "t_df": t_df},
    )


def synthetic_risk_pct(vol_daily: float, trades_per_day: float) -> float:
    """Sizing that maps a synthetic pool onto a target DAILY equity vol."""
    return vol_daily / float(np.sqrt(trades_per_day))
