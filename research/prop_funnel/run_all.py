"""Orchestrator (TICK-022 P6): parity gate -> verdicts -> sizing -> frontier -> report.

Run:  python3 research/prop_funnel/run_all.py [--seed 7] [--fast]
Everything written under data/research/prop_funnel/. Aborts if parity is red.
"""
from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from research.prop_funnel import feeds, parity, report, sizing_opt
from research.prop_funnel._lib import OUT_DIR, env_record, write_json
from research.prop_funnel.funnel import run_funnel
from research.prop_funnel.rulesets import FirmSpec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fast", action="store_true", help="2k trials instead of 10k")
    args = ap.parse_args()
    n = 2_000 if args.fast else 10_000
    n_sizing = 1_000 if args.fast else 3_000
    n_frontier = 1_000 if args.fast else 4_000
    t0 = time.time()

    print("── parity gate ──")
    par = parity.run_all_parity()
    for r in par["results"]:
        print(f"  {r['name']:30s} {'EXACT' if r.get('exact') else ('OK' if r['ok'] else 'FAIL')}")
    if not par["all_ok"]:
        print("PARITY RED — aborting (engine not trusted)")
        return 1

    print("── feeds ──")
    carry = feeds.load_carry_oos()
    pools = [carry, feeds.load_carry_decade(),
             feeds.carry_scenario(carry, 0.0), feeds.carry_scenario(carry, 0.69),
             feeds.carry_scenario(carry, 1.25),
             feeds.load_ict_window("london_a"), feeds.load_ict_window("window_B"),
             feeds.load_futures_orb()]
    live = feeds.load_live_closed_outcomes()
    for p in pools:
        print(f"  {p.name:20s} n={p.n:<5d} stamp={p.stamp.value:22s} tpd={p.trades_per_day:.3f}")

    specs = FirmSpec.load_all()
    forex_firms = [specs["FTMO_100K_SWING"], specs["MFF_100K"]]
    futures_firms = [specs["TOPSTEP_50K"], specs["APEX_50K"]]

    print("── verdict rows ──")
    rows = []
    for pool in pools:
        firms = futures_firms if pool.name.startswith("futures") else forex_firms
        for spec in firms:
            # zlib.crc32 is stable across processes (builtin hash() is salt-randomized)
            rng = np.random.default_rng(np.random.SeedSequence(
                [args.seed, zlib.crc32(pool.name.encode()), zlib.crc32(spec.name.encode())]))
            row = run_funnel(rng, pool, spec, n_attempts=n, n_funded_sims=n)
            rows.append(row)
            pf = row.get("p_funded", "—")
            ev = row.get("program_ev_per_month_usd", "—")
            print(f"  {pool.name:20s} × {spec.name:16s} P(funded)={pf}  EV/mo=${ev}")

    print("── sizing grids ──")
    opts = []
    for pool, firm in ((carry, "MFF_100K"), (carry, "FTMO_100K_SWING"),
                       (feeds.load_ict_window("window_B"), "MFF_100K")):
        o = sizing_opt.optimize_sizing(pool, specs[firm], base_seed=args.seed,
                                       n_attempts=n_sizing, n_funded_sims=n_sizing)
        opts.append(o)
        b = o.get("best") or {}
        print(f"  {pool.name} × {firm}: best {b.get('challenge_mult')}x/{b.get('funded_mult')}x "
              f"→ ${b.get('ev_per_month')}/mo")

    print("── synthetic frontier ──")
    fronts = []
    for firm in ("FTMO_100K_SWING", "TOPSTEP_50K"):
        f = sizing_opt.frontier(specs[firm], base_seed=args.seed,
                                n_attempts=n_frontier, n_funded_sims=n_frontier)
        fronts.append(f)
        print(f"  {firm}: {len(f['cells'])} cells")

    print("── charts + report ──")
    report.write_verdict_table(rows)
    for f in fronts:
        tag = f["firm"].lower()
        report._heatmap(f, "p_funded", "P(funded) — full challenge", f"frontier_pass_{tag}.png")
        report._heatmap(f, "p_month_ge_target", "P(funded month ≥ $10k)",
                        f"frontier_income_{tag}.png")
        report.tension_chart(f, f"tension_{tag}.png")
    for o in opts:
        report.sizing_heatmap(o, f"sizing_{o['strategy']}_{o['firm'].lower()}.png")
    report.days_to_pass_chart(rows, "days_to_pass.png")
    report.write_summary(rows, fronts, opts, par, live.meta, args.seed, env_record())

    write_json(OUT_DIR / "results.json", {
        "ticket": "TICK-022", "seed": args.seed, "fast": args.fast, "env": env_record(),
        "runtime_seconds": round(time.time() - t0, 1),
        "parity": par, "verdicts": rows, "sizing": opts, "frontiers": fronts,
        "live_sanity_pool": live.meta,
    })
    print(f"DONE in {time.time() - t0:.0f}s → {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
