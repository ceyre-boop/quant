"""
Re-derive hypothesis p-values + Benjamini-Hochberg correction.
==============================================================

The ledger stores ZERO p-values across 70 hypotheses — so the 20 "CONFIRMED"
findings were never multiple-testing-controlled (expected false discoveries at
alpha=0.05: ~1). This script re-derives p-values where they can be honestly
generated, applies Benjamini-Hochberg (FDR=5%), and writes the result back to the
ledger. Hypotheses with no reproducible on/off toggle are marked UNRECONSTRUCTABLE
rather than assigned a fabricated p-value.

Sources of derived p-values:
  • Permutation tests (run first): forex macro edge + ICT pattern edge — these ARE
    the re-derivation for the foundational edge hypotheses.
  • Toggle bootstrap: HYP-044 (VIX 13 vs 15) — run the forex backtest both ways,
    bootstrap the mean costed-return difference.

Reconstructable-but-deferred: the ICT sub-weight hypotheses (pd_alignment,
market_structure, displacement, fvg_tap, score-inversion, DOW veto) are config-
toggleable, BUT their parent ICT entry edge is already insignificant (permutation
p≈0.52) — individual sub-weight significance is moot, so they are flagged rather
than re-scanned. Foundational hypotheses (carry, RMT, term-structure gates) have no
generic toggle and are marked UNRECONSTRUCTABLE.

Usage:  python3 scripts/derive_hypothesis_pvalues.py [--nboot 10000]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
for lib in ("yfinance", "peewee", "urllib3", "requests"):
    logging.getLogger(lib).setLevel(logging.ERROR)

LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
TRADES = ROOT / "logs" / "forex_backtest_trades.json"
PERM_FOREX = ROOT / "data" / "research" / "permutation_test_forex.json"
PERM_ICT = ROOT / "data" / "research" / "permutation_test_ict.json"
REPORT = ROOT / "data" / "research" / "hypothesis_pvalues.json"
ALPHA = 0.05


def bootstrap_diff_pvalue(a: list, b: list, n_boot: int, rng) -> float:
    """One-sided bootstrap: p = P(resampled mean(a) - mean(b) <= 0).

    Small p ⇒ config A's mean return is reliably above B's (the feature helps)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = (rng.choice(a, len(a), replace=True).mean()
                    - rng.choice(b, len(b), replace=True).mean())
    return float(np.mean(diffs <= 0))


def _forex_trade_returns(vix_jpy_audnzd: float, only_pairs=("USDJPY=X", "AUDNZD=X")) -> list:
    """Run the forex backtest with USDJPY/AUDNZD VIX gate at the given threshold;
    return the costed per-trade returns for ONLY the affected pairs (others are
    identical across configs and would just dilute the comparison)."""
    from sovereign.forex.forex_backtester import ForexBacktester
    bt = ForexBacktester(start="2015-01-01", end="2024-12-31")
    gates = dict(bt.PAIR_VIX_GATES)
    gates["USDJPY=X"] = vix_jpy_audnzd
    gates["AUDNZD=X"] = vix_jpy_audnzd
    bt.PAIR_VIX_GATES = gates
    bt.backtest_all()  # writes logs/forex_backtest_trades.json (costed)
    data = json.loads(TRADES.read_text())
    return [t["pnl_pct"] for pair, trades in data.items()
            if pair in only_pairs for t in trades]


def benjamini_hochberg(items: list, alpha: float) -> None:
    """items: list of dicts with 'p_value'. Mutates in place adding bh_* fields.
    Only ranks entries with a numeric p_value."""
    scored = sorted([it for it in items if isinstance(it.get("p_value"), float)
                     and not np.isnan(it["p_value"])], key=lambda x: x["p_value"])
    m = len(scored)
    survive_rank = 0
    for rank, it in enumerate(scored, 1):
        thr = (rank / m) * alpha
        it["bh_rank"] = rank
        it["bh_threshold"] = round(thr, 5)
        if it["p_value"] <= thr:
            survive_rank = rank
    for rank, it in enumerate(scored, 1):
        it["bh_status"] = "SURVIVES_BH" if rank <= survive_rank else "FAILS_BH"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=10000)
    args = ap.parse_args()
    rng = np.random.default_rng(7)

    derived = []  # {id, label, p_value, source}

    # ── 1. Permutation-derived edge p-values ────────────────────────────────
    if PERM_FOREX.exists():
        pf = json.loads(PERM_FOREX.read_text())
        derived.append({"id": "HYP-003", "label": "Forex macro edge (rate-diff signal timing)",
                        "p_value": float(pf["p_value"]), "source": "permutation_test_forex",
                        "detail": f"real Sharpe {pf['real_portfolio_sharpe']} vs null; costed"})
    if PERM_ICT.exists():
        pi = json.loads(PERM_ICT.read_text())
        derived.append({"id": "HYP-004", "label": "ICT pattern edge (entry selection)",
                        "p_value": float(pi["p_value"]), "source": "permutation_test_ict",
                        "detail": f"real meanR {pi['real_mean_R']} vs null {pi['null_mean_R']}"})

    # ── 2. HYP-044 toggle bootstrap (VIX 13 vs 15) ──────────────────────────
    print("  Re-deriving HYP-044 (VIX 13 vs 15) via toggle bootstrap...")
    print("    running forex backtest @ VIX>15 ...")
    r15 = _forex_trade_returns(15.0)
    print("    running forex backtest @ VIX>13 ...")
    r13 = _forex_trade_returns(13.0)
    # H1: tightening to 13 IMPROVES returns → p = P(mean(r13) - mean(r15) <= 0)
    p044 = bootstrap_diff_pvalue(r13, r15, args.nboot, rng)
    derived.append({"id": "HYP-044", "label": "VIX gate 15→13 (USDJPY/AUDNZD)",
                    "p_value": p044, "source": "toggle_bootstrap",
                    "detail": f"meanR@13={np.mean(r13):.4f} (n={len(r13)}) vs "
                              f"meanR@15={np.mean(r15):.4f} (n={len(r15)})"})

    # ── 3. Benjamini-Hochberg across derived p-values ───────────────────────
    benjamini_hochberg(derived, ALPHA)

    # ── 4. Hypotheses flagged but not individually re-derived ───────────────
    deferred = {
        "HYP-024": "pd_alignment weight — config-toggleable, but parent ICT entry edge insignificant (perm p≈0.52); moot",
        "HYP-034": "market_structure weight — same: parent ICT edge insignificant; moot",
        "HYP-046": "displacement gate — same: parent ICT edge insignificant; moot",
        "HYP-047": "ICT score inversion — same: parent ICT edge insignificant; moot",
        "HYP-037": "fvg_tap timing — same: parent ICT edge insignificant; moot",
        "HYP-050": "DOW veto — same: parent ICT edge insignificant; moot",
        "HYP-022": "NY_PM anti-edge — same: parent ICT edge insignificant; moot",
    }
    unreconstructable = {
        "HYP-001": "Carry base — foundational, no generic on/off toggle",
        "HYP-005": "ICT walk-forward — superseded by permutation_test_ict",
        "HYP-018": "Library regime fix — bug fix, not a return-edge hypothesis",
        "HYP-030": "Three-confirmation term-structure gate — bespoke, not generically toggleable",
        "HYP-035": "RMT covariance cleaning — bespoke, not generically toggleable",
        "HYP-033": "Null result (no latent feature) — nothing to test",
    }

    # ── 5. Write back to ledger ─────────────────────────────────────────────
    ledger = json.loads(LEDGER.read_text())
    all_hyps = ledger.get("ledger", []) + ledger.get("hypotheses", [])
    by_id = {}
    for h in all_hyps:
        if isinstance(h, dict):
            by_id.setdefault(h.get("id"), []).append(h)

    def _annotate(hid, **fields):
        for h in by_id.get(hid, []):
            h.update(fields)

    for d in derived:
        _annotate(d["id"], p_value=round(d["p_value"], 4) if d["p_value"] == d["p_value"] else None,
                  pvalue_source=d["source"], bh_status=d.get("bh_status", "NO_P_VALUE"),
                  bh_threshold=d.get("bh_threshold"))
    for hid, reason in deferred.items():
        _annotate(hid, pvalue_status="RECONSTRUCTABLE_DEFERRED", pvalue_note=reason)
    for hid, reason in unreconstructable.items():
        _annotate(hid, pvalue_status="UNRECONSTRUCTABLE", pvalue_note=reason)

    LEDGER.write_text(json.dumps(ledger, indent=2))

    # ── 6. Report ───────────────────────────────────────────────────────────
    n_confirmed = sum(1 for h in all_hyps if isinstance(h, dict)
                      and h.get("status") in ("CONFIRMED", "DEPLOYED", "PARTIAL_CONFIRMED",
                                               "CONFIRMED_DIVERSIFICATION"))
    report = {
        "test_date": datetime.now(timezone.utc).isoformat(),
        "alpha": ALPHA,
        "n_boot": args.nboot,
        "confirmed_hypotheses": n_confirmed,
        "stored_pvalues_before": 0,
        "expected_false_discoveries_uncontrolled": round(n_confirmed * ALPHA, 1),
        "derived": derived,
        "deferred": deferred,
        "unreconstructable": unreconstructable,
        "note": ("Multiple-testing exposure remains PARTIALLY uncontrolled: only the "
                 "decision-relevant edges + HYP-044 were re-derived. The remaining confirmed "
                 "hypotheses never recorded p-values; going forward every test must log one."),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print("HYPOTHESIS P-VALUE RE-DERIVATION + BENJAMINI-HOCHBERG")
    print("=" * 60)
    print(f"  Confirmed hypotheses: {n_confirmed}  |  p-values stored before: 0")
    print(f"  Expected false discoveries (uncontrolled, α={ALPHA}): ~{n_confirmed*ALPHA:.1f}\n")
    for d in sorted(derived, key=lambda x: (x["p_value"] if x["p_value"]==x["p_value"] else 9)):
        print(f"  {d['id']:9s} p={d['p_value']:.4f}  {d.get('bh_status','?'):12s} {d['label']}")
        print(f"            └ {d['detail']}")
    print(f"\n  Deferred (parent edge insignificant): {', '.join(deferred)}")
    print(f"  Unreconstructable (no generic toggle): {', '.join(unreconstructable)}")
    print(f"\n  Saved report: {REPORT.relative_to(ROOT)}")
    print(f"  Ledger updated with p_value / bh_status fields.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
