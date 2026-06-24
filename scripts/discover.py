#!/usr/bin/env python3
"""
scripts/discover.py — Edge-Discovery pipeline CLI.

Discovery GENERATES candidate setups; the existing permutation + Deflated-Sharpe +
Benjamini-Hochberg gate DECIDES which are real. Research bench only — never touches
live config or trading (Phase 5 = human review).

    python3 scripts/discover.py --track forex-daily                 # the deliverable
    python3 scripts/discover.py --track forex-daily --selfcheck      # gate sanity assertions
    python3 scripts/discover.py --track nq-intraday                  # scaffolded status
    python3 scripts/discover.py --track intraday-fx                  # runbook pointer

Outputs to data/discovery/<track>/: strategy_table.csv, discover.html, preregister/*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DISCOVERY_DIR = ROOT / "data" / "discovery"


def _preregister(cands, out_dir: Path, train_window, holdout_window, stamp: str):
    pre = out_dir / "preregister"
    pre.mkdir(parents=True, exist_ok=True)
    for c in cands:
        (pre / f"{c.id}.json").write_text(json.dumps({
            "id": c.id, "name": c.name, "description": c.description,
            "family": c.meta.get("family"), "spec": c.meta.get("spec"),
            "train_window": train_window, "holdout_window": holdout_window,
            "preregistered_at": stamp,
        }, indent=2))


def _print_table(verdicts):
    order = {"VALID_EDGE": 0, "NOT_SIGNIFICANT": 1, "SCREENED_OUT": 2}
    gated = [v for v in verdicts if v.verdict != "SCREENED_OUT"]
    print(f"\n{'name':22s} {'verdict':16s} {'train':>6s} {'full':>6s} {'hold':>6s} {'perm_p':>7s} {'dsr':>6s} bh")
    for v in sorted(gated, key=lambda v: (order.get(v.verdict, 9), -v.full_sharpe)):
        print(f"{v.name:22s} {v.verdict:16s} {v.train_sharpe:+6.2f} {v.full_sharpe:+6.2f} "
              f"{v.holdout_sharpe:+6.2f} {str(v.perm_p):>7s} {str(v.dsr_prob):>6s} {v.bh_survives}")
    sc = [v for v in verdicts if v.verdict == "SCREENED_OUT"]
    if sc:
        print(f"  (+ {len(sc)} screened out below the in-sample floor)")


def run_forex_daily(args) -> int:
    from sovereign.discovery.data_adapter import ForexDailyAdapter
    from sovereign.discovery.features import compute_features
    from sovereign.discovery import candidates as C
    from sovereign.discovery.gate import Gate
    from sovereign.discovery import visuals
    from datetime import datetime, timezone

    train_window = (args.start, args.train_end)
    holdout_window = (args.holdout_start, args.end)
    stamp = datetime.now(timezone.utc).isoformat()

    print(f"[discover] forex-daily | train {train_window} | holdout {holdout_window} | perms={args.perms}")
    adapter = ForexDailyAdapter(start=args.start, end=args.end, pairs=args.pairs).preload()
    print(f"[discover] pairs: {[p.replace('=X','') for p in adapter.pairs]}")
    feats = {p: compute_features(adapter.price_df(p)) for p in adapter.pairs}
    cands = C.generate(adapter, feats, train_window, include_clusters=not args.no_clusters)
    print(f"[discover] {len(cands)} candidates "
          f"({sum(1 for c in cands if c.meta['family']=='rule')} rule + "
          f"{sum(1 for c in cands if c.meta['family']=='cluster')} cluster) — pre-registering")

    out_dir = Path(args.out) / "forex-daily"
    _preregister(cands, out_dir, train_window, holdout_window, stamp)

    gate = Gate(adapter, train_window=train_window, holdout_window=holdout_window,
                n_perms=args.perms, seed=args.seed)
    verdicts = gate.evaluate(cands, feats, max_finalists=args.max_finalists,
                             progress=lambda m: print(f"   {m}"))

    n_valid = sum(1 for v in verdicts if v.verdict == "VALID_EDGE")
    n_finalists = sum(1 for v in verdicts if v.verdict != "SCREENED_OUT")
    summary = {"window": f"{args.start}..{args.end}", "n_candidates": len(cands),
               "n_finalists": n_finalists, "n_valid": n_valid}
    paths = visuals.render("forex-daily", verdicts, adapter, feats, train_window, out_dir, summary)

    _print_table(verdicts)
    print(f"\n[discover] VALID_EDGE survivors: {[v.name for v in verdicts if v.verdict=='VALID_EDGE'] or 'none'}")
    if not n_valid:
        print("[discover] 0 survivors is the CORRECT, protective output — the gate refuses to")
        print("           hand you a false positive. Promote a near-miss only via a ≥10k-perm")
        print("           research_factory run with a logged rationale.")
    print(f"[discover] artifacts → {out_dir}/{{strategy_table.csv, discover.html, preregister/}}")
    return 0


def run_selfcheck(args) -> int:
    """Gate sanity: the real macro signal must pass; random must fail."""
    import numpy as np
    from sovereign.discovery.data_adapter import ForexDailyAdapter
    from sovereign.discovery.gate import Gate, Candidate

    print("[selfcheck] gate sanity — real edge must pass, random must fail")
    ad = ForexDailyAdapter(start=args.start, end=args.end).preload()
    rng = np.random.default_rng(1)

    def real_sig(pair, pdf, fdf):
        return ad.dataset(pair).signals

    def rand_sig(pair, pdf, fdf):
        n = len(pdf); s = np.zeros(n, np.int8)
        k = max(1, int(np.count_nonzero(ad.dataset(pair).signals)))
        pos = rng.choice(n, k, replace=False); s[pos] = rng.choice([-1, 1], k); return s

    cands = [Candidate("real", "macro-replay", "proven carry/rate signal", real_sig),
             Candidate("rand", "random", "random entries", rand_sig)]
    feats = {p: ad.price_df(p) for p in ad.pairs}
    g = Gate(ad, train_window=(args.start, args.train_end), holdout_window=(args.holdout_start, args.end),
             n_perms=max(80, args.perms), seed=7)
    res = {r.name: r for r in g.evaluate(cands, feats, progress=lambda m: None)}
    real, rand = res["macro-replay"], res["random"]
    print(f"  real:   perm_p={real.perm_p}  full_sharpe={real.full_sharpe:+.3f}  verdict={real.verdict}")
    print(f"  random: perm_p={rand.perm_p}  full_sharpe={rand.full_sharpe:+.3f}  verdict={rand.verdict}")
    ok = (real.perm_p is not None and real.perm_p < 0.05) and (rand.perm_p is not None and rand.perm_p > 0.5)
    print(f"[selfcheck] {'PASS ✓' if ok else 'FAIL ✗'} — real significant, random not")
    return 0 if ok else 1


def run_nq_intraday(args) -> int:
    from sovereign.discovery.data_adapter import NQIntradayAdapter
    from sovereign.discovery.features import compute_features
    from sovereign.discovery import candidates as C
    print(f"[discover] nq-intraday ({args.nq_tf}) — scaffolded track")
    try:
        ad = NQIntradayAdapter(timeframe=args.nq_tf).preload()
    except FileNotFoundError as e:
        print(f"[discover] {e}")
        return 1
    df = ad.price_df()
    print(f"[discover] loaded {len(df):,} bars  ({df.index[0]} → {df.index[-1]})")
    feats = compute_features(df)
    cands = C._rule_candidates()  # rule candidates work on any OHLCV
    print(f"[discover] features {feats.shape} computed; {len(cands)} rule candidates ready.")
    print("[discover] STATUS: data + features + candidates work. The remaining completion")
    print("           point is routing signals through the NQ futures simulator")
    print("           (sovereign/es_nq/backtest.py) — NQ uses futures costs + the adaptive")
    print("           ladder, not the forex kernel, so the gate's eval_signals is NQ-specific.")
    return 0


def run_intraday_fx(args) -> int:
    from sovereign.discovery.data_adapter import IntradayFXAdapter
    print("[discover] intraday-fx — blocked on vendor data")
    try:
        IntradayFXAdapter().preload()
    except NotImplementedError as e:
        print(f"[discover] {e}")
    runbook = ROOT / "docs" / "intraday_fx_acquisition.md"
    print(f"[discover] runbook: {runbook}")
    return 0


def _print_regime_table(verdicts):
    order = {"VALID_EDGE": 0, "NOT_SIGNIFICANT": 1, "SCREENED_OUT": 2}
    gated = [v for v in verdicts if v.verdict != "SCREENED_OUT"]
    print(f"\n{'name':20s} {'verdict':16s} {'full':>6s} {'hold':>6s} {'perm_p':>7s} {'delta_p':>8s} {'wf_ok':>6s} bh")
    for v in sorted(gated, key=lambda v: (order.get(v.verdict, 9), v.delta_p if v.delta_p is not None else 1)):
        print(f"{v.name:20s} {v.verdict:16s} {v.full_sharpe:+6.2f} {v.holdout_sharpe:+6.2f} "
              f"{str(v.perm_p):>7s} {str(v.delta_p):>8s} {str(v.wf_robust):>6s} {v.bh_survives}")


def run_regime(args) -> int:
    from sovereign.discovery.data_adapter import ForexDailyAdapter
    from sovereign.discovery import regime as R
    from sovereign.discovery import regime_candidates as RC
    from sovereign.discovery.gate import Gate
    from sovereign.discovery import visuals
    from datetime import datetime, timezone

    train_window = (args.start, args.train_end)
    holdout_window = (args.holdout_start, args.end)
    wf_years = [2021, 2022, 2023, 2024]
    stamp = datetime.now(timezone.utc).isoformat()

    print(f"[discover] regime-router | train {train_window} | holdout {holdout_window} | perms={args.perms}")
    adapter = ForexDailyAdapter(start=args.start, end=args.end, pairs=args.pairs).preload()
    print(f"[discover] pairs: {[p.replace('=X','') for p in adapter.pairs]}")
    feats = R.regime_features(adapter)
    cands = RC.generate(adapter)
    print(f"[discover] {len(cands)} regime candidates (ungated base × macro-state filter) — pre-registering")

    out_dir = Path(args.out) / "regime"
    _preregister(cands, out_dir, train_window, holdout_window, stamp)

    base_signals = {p: adapter.dataset(p).signals for p in adapter.pairs}
    gate = Gate(adapter, train_window=train_window, holdout_window=holdout_window,
                n_perms=args.perms, seed=args.seed)
    verdicts = gate.evaluate(cands, feats, max_finalists=args.max_finalists,
                             progress=lambda m: print(f"   {m}"),
                             base_signals_by_pair=base_signals, wf_years=wf_years)

    n_valid = sum(1 for v in verdicts if v.verdict == "VALID_EDGE")
    n_finalists = sum(1 for v in verdicts if v.verdict != "SCREENED_OUT")
    summary = {"window": f"{args.start}..{args.end}", "n_candidates": len(cands),
               "n_finalists": n_finalists, "n_valid": n_valid}
    visuals.render("regime", verdicts, adapter, feats, train_window, out_dir, summary)

    _print_regime_table(verdicts)
    dep = next((v for v in verdicts if v.id == "regime_vixgate_deployed"), None)
    if dep and dep.delta_p is not None:
        word = ("still beats the ungated base" if dep.delta_p < 0.10
                else "NO LONGER beats the ungated base on current data")
        print(f"\n[discover] HYP-027 deployed VIX gate re-validation: delta_p={dep.delta_p}, "
              f"wf_robust={dep.wf_robust} → {word}")
    print(f"[discover] VALID_EDGE survivors: {[v.name for v in verdicts if v.verdict=='VALID_EDGE'] or 'none'}")
    if not n_valid:
        print("[discover] 0 survivors = the gate found no regime filter that beats the ungated base")
        print("           robustly across every OOS year. Promote a near-miss only via a ≥10k-perm")
        print("           edge_pipeline run + approve_edge.py with a logged rationale.")
    print(f"[discover] artifacts → {out_dir}/")
    return 0


def run_validate(args) -> int:
    """Resolve a single named candidate at full perm depth + delta-vs-base (the kill-test)."""
    from sovereign.discovery.data_adapter import ForexDailyAdapter
    from sovereign.discovery.features import compute_features
    from sovereign.discovery import candidates as C
    from sovereign.discovery import regime as R, regime_candidates as RC
    from sovereign.discovery.gate import Gate

    target = args.validate
    train_window = (args.start, args.train_end)
    holdout_window = (args.holdout_start, args.end)
    wf_years = [2021, 2022, 2023, 2024]
    print(f"[validate] resolving {target!r} at {args.perms} perms + delta-vs-base + walk-forward")
    adapter = ForexDailyAdapter(start=args.start, end=args.end, pairs=args.pairs).preload()

    gen_feats = {p: compute_features(adapter.price_df(p)) for p in adapter.pairs}
    gen_cands = C.generate(adapter, gen_feats, train_window, include_clusters=True)
    reg_feats = R.regime_features(adapter)
    reg_cands = RC.generate(adapter)
    pool = {}
    for c in gen_cands:
        pool[c.id] = ("generic", c); pool[c.name] = ("generic", c)
    for c in reg_cands:
        pool[c.id] = ("regime", c); pool[c.name] = ("regime", c)
    if target not in pool:
        print(f"[validate] {target!r} not found. Available names: "
              f"{sorted({c.name for c in gen_cands + reg_cands})}")
        print("[validate] (a cluster candidate may not form on current data — itself informative.)")
        return 1

    kind, cand = pool[target]
    feats = reg_feats if kind == "regime" else gen_feats
    base_signals = {p: adapter.dataset(p).signals for p in adapter.pairs}
    gate = Gate(adapter, train_window=train_window, holdout_window=holdout_window,
                n_perms=args.perms, seed=args.seed)
    res = gate.evaluate([cand], feats, max_finalists=1, progress=lambda m: print(f"   {m}"),
                        base_signals_by_pair=base_signals, wf_years=wf_years)
    if not res:
        print("[validate] candidate produced no signals/trades — dead.")
        return 0
    r = res[0]
    print(f"\n[validate] {r.name} ({kind})")
    print(f"  full_sharpe   : {r.full_sharpe:+.3f}   holdout: {r.holdout_sharpe:+.3f}")
    print(f"  permutation p : {r.perm_p}   (n_perms={r.n_perms})")
    print(f"  delta vs base : {r.delta_p}   (<0.10 ⇒ beats the ungated carry base)")
    print(f"  per-year OOS  : {r.wf_by_year}   robust={r.wf_robust}")
    print(f"  deflated prob : {r.dsr_prob}")
    real = (r.perm_p is not None and r.perm_p < 0.05 and r.delta_p is not None
            and r.delta_p < 0.10 and bool(r.wf_robust) and r.holdout_sharpe > 0)
    print(f"\n[validate] VERDICT: {'REAL — flag for the official ledger (≥10k via edge_pipeline)' if real else 'DEAD — does not beat the base robustly; close the question'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Alta edge-discovery pipeline (discovery feeds the gate).")
    ap.add_argument("--track", default="forex-daily",
                    choices=["forex-daily", "regime", "nq-intraday", "intraday-fx"])
    ap.add_argument("--validate", default=None, metavar="CANDIDATE",
                    help="resolve a single named candidate (e.g. cluster4_short) at full perm depth")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--train-end", default="2022-12-31")
    ap.add_argument("--holdout-start", default="2023-01-01")
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--max-finalists", type=int, default=6)
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-clusters", action="store_true")
    ap.add_argument("--nq-tf", default="5m", choices=["1m", "5m", "1d"])
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", default=str(DISCOVERY_DIR))
    args = ap.parse_args()

    if args.selfcheck:
        return run_selfcheck(args)
    if args.validate:
        return run_validate(args)
    if args.track == "forex-daily":
        return run_forex_daily(args)
    if args.track == "regime":
        return run_regime(args)
    if args.track == "nq-intraday":
        return run_nq_intraday(args)
    return run_intraday_fx(args)


if __name__ == "__main__":
    raise SystemExit(main())
