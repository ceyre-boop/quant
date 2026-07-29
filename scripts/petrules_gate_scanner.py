#!/usr/bin/env python3
"""
petrules_gate_scanner.py — Petrules Gate daily scanner (Phase 0, rule-based).

Wakes up every weekday morning, scores the whole universe (S&P 500 + Russell 2000
+ 50 ETFs + 4 carry pairs) on four rule-based factors — Form 4 insider clusters,
analyst revision velocity, options vol/OI flow, and 13D/G activist filings —
tiers each instrument 1–4, and writes the result to a single JSON file the
dashboard reads.

    data/agent/petrules_gate_scan.json     ← dashboard reads this
    data/agent/gate_scan_history.jsonl     ← append-only, tier 2+
    data/agent/gate_calibration.jsonl      ← append-only outcome log (tier 3+)

DISCIPLINE (ticket non-negotiables, enforced here):
  * It NEVER auto-trades. It writes a JSON file, nothing else. The `sizing` block
    is informational and carries calibration_status "BACKTEST ONLY — uncalibrated".
  * No sovereign/ or ict/ imports. Standalone research script.
  * Free data only (SEC EDGAR + yfinance). If a source is unreachable, write the
    honest error JSON {"error": "...", "scanned_at": "..."} — never fabricate.
  * All thresholds live in config/gate_params.yml.
  * gate_calibration.jsonl is append-only.

Usage:
    python3 scripts/petrules_gate_scanner.py            # full live scan
    python3 scripts/petrules_gate_scanner.py --limit 50 # cap universe (debug)
    python3 scripts/petrules_gate_scanner.py --self-test # offline fixture scan
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gate_scorer  # noqa: E402
import gate_universe  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "gate_params.yml"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as fh:
        return yaml.safe_load(fh)


def _path(cfg: dict, key: str) -> Path:
    return REPO_ROOT / cfg["paths"][key]


def write_error_scan(cfg: dict, message: str) -> Path:
    """Honest failure record — dashboard renders the timestamp, never crashes."""
    out = _path(cfg, "scan_output")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"error": message, "scanned_at": _utc_now()}, indent=2))
    return out


def ensure_universe(cfg: dict) -> list[str]:
    upath = _path(cfg, "universe")
    if gate_universe.universe_is_stale(upath, cfg["universe"]["refresh_days"]):
        gate_universe.write_universe(cfg)  # raises on source failure
    return json.loads(upath.read_text())["symbols"]


# ── feature assembly ─────────────────────────────────────────────────────────
def gather_features(symbols: list[str], cfg: dict, limit: int | None) -> dict:
    """Assemble per-symbol feature bundles from the free data sources.

    EDGAR filings are fetched once for the whole window and indexed by ticker
    (cheap — a handful of requests). Options + revisions are per-symbol yfinance
    calls (the expensive part on a real run). Any hard source failure raises so
    the scanner writes the error JSON rather than a half-fabricated scan.
    """
    from gate_edgar_client import EdgarClient
    import gate_options_screen as opt

    if limit:
        symbols = symbols[:limit]

    edgar = EdgarClient(cfg)
    form4 = edgar.fetch_form4()          # {ticker: {...}}  raises EdgarUnavailable
    activist = edgar.fetch_activist()    # {ticker: {...}}

    features: dict[str, dict] = {}
    for sym in symbols:
        insider = form4.get(sym)
        act = activist.get(sym)
        # Options + revisions only for equities/ETFs, not FX pairs.
        options = revision = None
        if not sym.endswith("=X"):
            options = opt.screen_options(sym, cfg)
            revision = opt.revision_velocity(sym, cfg)
        features[sym] = {
            "insider": insider,
            "activist": act,
            "options": options,
            "revision": revision,
        }
    return features


# ── build the output JSON ────────────────────────────────────────────────────
def _direction(factors: dict) -> str:
    return "bullish"  # Phase-0 rubric only scores accumulation/positive setups


def build_signal_labels(sym: str, feats: dict, factors: dict) -> list[dict]:
    labels: list[dict] = []
    ins = feats.get("insider") or {}
    if factors["insider_cluster"] > 0 and ins.get("n_buyers"):
        labels.append({
            "label": (
                f"Form 4: {ins.get('n_buyers')} insiders bought "
                f"${ins.get('total_buy_usd', 0):,.0f} open market"
                + (" incl C-suite" if ins.get("includes_csuite") else "")
                + (f"; {ins.get('n_sellers')} sold" if ins.get("n_sellers") else "")
            ),
            "direction": "bullish",
        })
    opts = feats.get("options") or {}
    if factors["options_flow"] > 0 and opts.get("best_vol_oi"):
        labels.append({
            "label": (
                f"Options: {opts.get('best_volume', 0):,} contracts @ "
                f"{opts.get('best_strike')} strike, vol/OI="
                f"{opts.get('best_vol_oi', 0):.1f}, "
                f"${opts.get('best_premium_usd', 0):,.0f} premium"
            ),
            "direction": "bullish",
        })
    rev = feats.get("revision") or {}
    if factors["revision_velocity"] > 0 and rev.get("n_upgrades"):
        labels.append({
            "label": (
                f"Revisions: {rev.get('n_upgrades')} upgrades vs "
                f"{rev.get('n_downgrades', 0)} cuts in window"
            ),
            "direction": "bullish",
        })
    act = feats.get("activist") or {}
    if factors["activist"] > 0 and act.get("filing_type"):
        labels.append({
            "label": f"Activist: {act.get('filing_type').replace('_', ' ')}",
            "direction": "bullish",
        })
    return labels


def build_top_signal(scored: dict, feats: dict, cfg: dict) -> dict:
    factors = scored["factors"]
    labels = build_signal_labels(scored["symbol"], feats, factors)
    return {
        "symbol": scored["symbol"],
        "tier": scored["tier"],
        "conviction_score": scored["conviction_score"],
        "hypothesis": (
            "Rule-based Phase-0 signal: disclosed-flow / revision / options "
            "footprints diverge from priced-in consensus. Uncalibrated — "
            "surfaced for Colin's review, not a trade directive."
        ),
        "consensus": {
            "summary": "Phase-0 scanner does not yet compute a priced-in baseline",
            "narrative": "consensus baseline TBD (ML phase)",
        },
        "divergence_signals": labels,
        "sizing": gate_scorer.sizing_block(scored["conviction_score"], cfg),
        "move_up": ["Insider cluster expands", "Second options block same strike"],
        "move_down": ["Any analyst downgrade", "Macro shock"],
    }


def run_scan(cfg: dict, limit: int | None = None,
             injected_features: dict | None = None) -> dict:
    """Score the universe and return the scan dict. Pure given features."""
    if injected_features is not None:
        features = injected_features
        symbols = list(features.keys())
    else:
        symbols = ensure_universe(cfg)
        features = gather_features(symbols, cfg, limit)
        symbols = list(features.keys())

    all_signals: list[dict] = []
    tier2_plus: list[tuple[dict, dict]] = []
    for sym in symbols:
        scored = gate_scorer.score_instrument(sym, features[sym], cfg)
        all_signals.append({
            "symbol": sym,
            "tier": scored["tier"],
            "conviction_score": scored["conviction_score"],
        })
        if scored["tier"] >= 2:
            tier2_plus.append((scored, features[sym]))

    all_signals.sort(key=lambda s: s["conviction_score"], reverse=True)
    tier2_plus.sort(key=lambda x: x[0]["conviction_score"], reverse=True)

    tier3_plus = sum(1 for s, _ in tier2_plus if s["tier"] >= 3)
    top_signal = None
    if tier2_plus:
        top_scored, top_feats = tier2_plus[0]
        top_signal = build_top_signal(top_scored, top_feats, cfg)

    return {
        "scanned_at": _utc_now(),
        "instruments_scanned": len(symbols),
        "tier3_plus": tier3_plus,
        "top_signal": top_signal,
        "all_signals": [s for s in all_signals if s["tier"] >= 2][:50],
        "_tier2_plus_internal": tier2_plus,  # stripped before write
    }


# ── persistence ──────────────────────────────────────────────────────────────
def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def persist(cfg: dict, scan: dict) -> Path:
    tier2_plus = scan.pop("_tier2_plus_internal", [])

    # 1) main scan output (dashboard reads this)
    out = _path(cfg, "scan_output")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scan, indent=2))

    # 2) scan history — one line per scan, tier 2+ summary, append-only
    append_jsonl(_path(cfg, "scan_history"), {
        "scanned_at": scan["scanned_at"],
        "instruments_scanned": scan["instruments_scanned"],
        "tier3_plus": scan["tier3_plus"],
        "signals": [
            {"symbol": s["symbol"], "tier": s["tier"],
             "conviction_score": s["conviction_score"]}
            for s, _ in tier2_plus
        ],
    })

    # 3) calibration log — append a stub per tier-3+ signal (outcome filled later)
    cal_path = _path(cfg, "calibration_log")
    date = scan["scanned_at"][:10]
    for scored, _feats in tier2_plus:
        if scored["tier"] < 3:
            continue
        append_jsonl(cal_path, {
            "date": date,
            "symbol": scored["symbol"],
            "tier": scored["tier"],
            "conviction_score": scored["conviction_score"],
            "hypothesis": "Phase-0 rule-based surface",
            "actual_outcome_pct": None,      # filled when the trade/setup resolves
            "outcome_direction": None,
            "consensus_implied_move": None,
            "beat_implied": None,
            "calibration_status": "BACKTEST ONLY — uncalibrated, not a trade directive",
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Petrules Gate daily scanner")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap universe size (debug)")
    ap.add_argument("--self-test", action="store_true",
                    help="run offline fixture scan, no network")
    args = ap.parse_args()

    cfg = load_config()

    if args.self_test:
        return _self_test(cfg)

    try:
        scan = run_scan(cfg, limit=args.limit)
    except Exception as exc:
        out = write_error_scan(cfg, f"{type(exc).__name__}: {exc}")
        print(f"[petrules_gate] source failure — wrote error JSON to {out}: {exc}",
              file=sys.stderr)
        return 1

    out = persist(cfg, scan)
    top = scan.get("top_signal")
    hd = (f"TOP {top['symbol']} T{top['tier']} {top['conviction_score']}"
          if top else "QUIET — nothing cleared Tier 2")
    print(f"[petrules_gate] scanned {scan['instruments_scanned']} · "
          f"tier3+={scan['tier3_plus']} · {hd} → {out}")
    return 0


def _self_test(cfg: dict) -> int:
    """Offline structural proof: hand-built features → tiering + schema + write."""
    fixtures = {
        "NVDA": {  # engineered Tier-4 screamer
            "insider": {"n_buyers": 4, "n_sellers": 0, "total_buy_usd": 4_200_000,
                        "includes_csuite": True, "cluster_within_days": 3},
            "revision": {"n_upgrades": 6, "n_downgrades": 0},
            "options": {"best_vol_oi": 4.8, "best_premium_usd": 2_800_000,
                        "best_volume": 14000, "best_strike": 140.0,
                        "multiple_strikes_same_side": True,
                        "n_accumulating_strikes": 3},
            "activist": {"filing_type": "new_13d"},
        },
        "AAPL": {  # engineered Tier-3 strong
            "insider": {"n_buyers": 3, "n_sellers": 0, "total_buy_usd": 800_000,
                        "includes_csuite": False, "cluster_within_days": 4},
            "revision": {"n_upgrades": 4, "n_downgrades": 0},
            "options": {"best_vol_oi": 3.5, "best_premium_usd": 600_000,
                        "best_volume": 6000, "best_strike": 210.0,
                        "multiple_strikes_same_side": False,
                        "n_accumulating_strikes": 1},
            "activist": {"filing_type": "new_13d"},
        },
        "KO": {  # noise, Tier 1 — should be discarded from surfaced list
            "insider": {"n_buyers": 0, "n_sellers": 2, "total_buy_usd": 0,
                        "includes_csuite": False, "cluster_within_days": 0},
            "revision": {"n_upgrades": 0, "n_downgrades": 0},
            "options": None,
            "activist": None,
        },
    }
    scan = run_scan(cfg, injected_features=fixtures)
    tier2 = scan.pop("_tier2_plus_internal", [])
    print("=== SELF-TEST: fixture scan ===")
    print(json.dumps(scan, indent=2))
    assert scan["instruments_scanned"] == 3
    top = scan["top_signal"]
    assert top and top["symbol"] == "NVDA", "NVDA should be top signal"
    assert top["tier"] == 4, f"expected Tier 4, got {top['tier']}"
    assert top["sizing"]["calibration_status"].startswith("BACKTEST ONLY"), \
        "sizing block must be marked BACKTEST ONLY"
    assert top["sizing"]["f"] <= top["sizing"]["f_max"], "f must respect f_max ceiling"
    assert scan["tier3_plus"] == 2, f"expected 2 tier3+, got {scan['tier3_plus']}"
    assert all(s["tier"] >= 2 for s in scan["all_signals"]), "KO noise must be dropped"
    print("\nSELF-TEST PASSED: tiering, top-signal, f_max ceiling, "
          "sizing honesty, noise-drop all correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
