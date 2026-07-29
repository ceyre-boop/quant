#!/usr/bin/env python3
"""
gate_scorer.py — rule-based conviction rubric for the Petrules Gate (Phase 0).

Pure functions. No network, no ML. Given a per-instrument feature dict, produce
factor scores, a weighted conviction score, and a tier — all thresholds pulled
from config/gate_params.yml (ticket non-negotiable: no hardcoded thresholds).

    conviction = insider*0.30 + revision*0.25 + options*0.25 + activist*0.20

Tiers: 4 (>=0.85), 3 (>=0.70), 2 (>=0.50), 1 (<0.50).

The `sizing` block produced here is INFORMATIONAL ONLY and carries an explicit
calibration_status — this scanner never auto-trades.

Standalone research module — no sovereign/ or ict/ imports.
"""
from __future__ import annotations


# ── per-factor scorers ───────────────────────────────────────────────────────
def score_insider_cluster(feat: dict, cfg: dict) -> float:
    """feat: {n_buyers, n_sellers, total_buy_usd, includes_csuite,
    cluster_within_days}. Returns [0,1]."""
    s = cfg["scoring"]["insider_cluster"]
    n_buyers = int(feat.get("n_buyers", 0))
    n_sellers = int(feat.get("n_sellers", 0))
    total = float(feat.get("total_buy_usd", 0.0))
    csuite = bool(feat.get("includes_csuite", False))
    cluster_days = feat.get("cluster_within_days", 0)
    window = s["cluster_window_days"]

    # Top band: 4+ insiders incl C-suite, > $1M total => 1.0
    if n_buyers >= 4 and csuite and total > s["csuite_bonus_min_usd"]:
        score = 1.0
    else:
        score = 0.0
        for band in s["bands"]:
            mb = band.get("min_buyers", 0)
            need_cluster = band.get("cluster_within_days")
            min_usd = band.get("min_total_usd", 0)
            if n_buyers >= mb and total >= min_usd:
                if need_cluster is not None:
                    # cluster band requires the buys to fall within the window
                    if not (n_buyers >= mb and cluster_days <= need_cluster):
                        continue
                score = band["score"]
                break
    # Informed selling in the same window is a negative.
    if n_sellers > 0:
        score -= s["sell_penalty"]
    return max(0.0, min(1.0, score))


def score_revision_velocity(feat: dict, cfg: dict) -> float:
    """feat: {n_upgrades, n_downgrades}. Returns [0,1]."""
    s = cfg["scoring"]["revision_velocity"]
    up = int(feat.get("n_upgrades", 0))
    down = int(feat.get("n_downgrades", 0))
    score = 0.0
    for band in s["bands"]:
        if up >= band["min_upgrades"] and down <= band["max_downgrades"]:
            score = band["score"]
            break
    if down > 0:
        score = min(score, s["downgrade_cap"])
    return max(0.0, min(1.0, score))


def score_options_flow(feat: dict, cfg: dict) -> float:
    """feat: {best_vol_oi, best_premium_usd, multiple_strikes_same_side}.
    Returns [0,1]."""
    s = cfg["scoring"]["options_flow"]
    vol_oi = float(feat.get("best_vol_oi", 0.0))
    premium = float(feat.get("best_premium_usd", 0.0))
    multi = bool(feat.get("multiple_strikes_same_side", False))
    for band in s["bands"]:
        if band.get("multiple_strikes_same_side"):
            if multi:
                return band["score"]
            continue
        if vol_oi >= band.get("min_vol_oi", 0.0) and premium >= band.get(
            "min_premium_usd", 0
        ):
            return band["score"]
    return 0.0


def score_activist(feat: dict, cfg: dict) -> float:
    """feat: {filing_type}. Returns [0,1]."""
    bands = cfg["scoring"]["activist"]["bands"]
    ftype = feat.get("filing_type", "none")
    return float(bands.get(ftype, 0.0))


# ── aggregate ────────────────────────────────────────────────────────────────
def conviction_score(factors: dict, cfg: dict) -> float:
    w = cfg["weights"]
    score = (
        factors["insider_cluster"] * w["insider_cluster"]
        + factors["revision_velocity"] * w["revision_velocity"]
        + factors["options_flow"] * w["options_flow"]
        + factors["activist"] * w["activist"]
    )
    return round(max(0.0, min(1.0, score)), 4)


def assign_tier(score: float, cfg: dict) -> int:
    t = cfg["tiers"]
    if score >= t["tier4_min"]:
        return 4
    if score >= t["tier3_min"]:
        return 3
    if score >= t["tier2_min"]:
        return 2
    return 1


def score_instrument(symbol: str, features: dict, cfg: dict) -> dict:
    """Full score for one instrument.

    `features` bundles the per-source feature dicts:
      {"insider": {...}, "revision": {...}, "options": {...}, "activist": {...}}
    Missing sources score 0.0 (never fabricated).
    """
    factors = {
        "insider_cluster": score_insider_cluster(features.get("insider") or {}, cfg),
        "revision_velocity": score_revision_velocity(
            features.get("revision") or {}, cfg
        ),
        "options_flow": score_options_flow(features.get("options") or {}, cfg),
        "activist": score_activist(features.get("activist") or {}, cfg),
    }
    score = conviction_score(factors, cfg)
    tier = assign_tier(score, cfg)
    return {
        "symbol": symbol,
        "conviction_score": score,
        "tier": tier,
        "factors": factors,
    }


# ── informational sizing (NOT a trade directive) ─────────────────────────────
def sizing_block(conviction: float, cfg: dict) -> dict:
    """Compute an informational quarter-Kelly size with a hard f_max ceiling.

    This is BACKTEST-ONLY. The scanner does not trade. The calibration_status
    string is mandatory and must reach the dashboard verbatim.
    """
    s = cfg["sizing"]
    f_max = float(s["f_max"])
    budget = float(s["budget_per_trade"])
    kelly_frac = float(s["kelly_fraction"])
    edge_ratio = float(s["edge_ratio_default"])
    worst_case = float(s["worst_case_loss_default"])

    kelly_fraction = (conviction * edge_ratio - (1 - conviction)) / edge_ratio
    raw = min(kelly_fraction * kelly_frac, budget / worst_case)
    f = max(0.0, min(raw, f_max))
    f_max_hit = raw >= f_max
    # size_multiplier is relative to the 1% base budget, for the dashboard.
    size_multiplier = round(f / budget, 2) if budget else 1.0
    return {
        "size_multiplier": size_multiplier,
        "f": round(f, 4),
        "f_max": f_max,
        "f_max_hit": bool(f_max_hit),
        "calibration_status": s["calibration_status"],
    }
