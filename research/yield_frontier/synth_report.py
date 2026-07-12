#!/usr/bin/env python3
"""M4 — yield board synthesis. Run: python3 -m research.yield_frontier.synth_report"""
import csv
import json

from ._lib import OUT, STAMP, mined_total
from .yield_board import CONTEXT_ROWS, MIN_N_RANK

ARITHMETIC = (
    "2%/day compounds to ~147x/year; over 10-15 years the number exceeds all money on "
    "Earth. It is a falsifiability ceiling, not a target. Scale: Medallion, history's "
    "best, ran ~0.15%/day net on capped capital; this shop's proven carry edge is "
    "~0.02%/day. Every number below is MINED (look-back-selected across "
    "{n} configurations) — the true out-of-sample yield of any row is LOWER, and only "
    "the G-phase gauntlet can say by how much.")

CAVEATS = [
    "All rows are MINING output: selected in-sample across ~{n} trials; nothing here is evidence.",
    "Equities shorts assume locates at 50-75% fill and no overnight borrow when intraday; halt gap-through modeled from cached bars; SSR not modeled.",
    "Equities capacity = 1% of median event dollar-volume — most equity rows cap out at $0.1-1.5M deployed.",
    "NQ rows are % of 1x notional; leverage scales returns and drawdowns identically.",
    "Options: VRP-era cache serves ~30-DTE monthlies only (7/14-DTE grids unmineable); call marks spotty; fills at mid±0.5×half-spread — at k=1.0 every cell worsens; OP3 strangles skipped (margin model too coarse); OP4 VIX overlay skipped (all base cells net-negative — gating losers mines noise); OP5 lottery unmineable (no deep-OTM marks).",
    "Sealed context: 13 pre-registered daily-resolution hypotheses adjudicated null in this shop; every survivor is a structural premium.",
]


def main():
    rows = []
    for sess in ("m1_equities", "m2_nq", "m3_options"):
        fp = OUT / sess / "board_rows.json"
        if fp.exists():
            rows += json.loads(fp.read_text())["rows"]
    ranked = sorted([r for r in rows if r.get("n", 0) >= MIN_N_RANK],
                    key=lambda r: -r["net_pct_day"])
    n_mined = mined_total()

    cols = ["universe", "family", "config", "n", "mean_pct", "median_pct",
            "events_per_day", "gross_pct_day", "net_pct_day", "tail_p5",
            "tail_p1", "ruin_frac", "capacity_usd"]
    with open(OUT / "yield_board.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["stamp"], extrasaction="ignore")
        w.writeheader()
        for r in ranked:
            w.writerow(r)

    md = [f"# Yield Board — MINING RESULTS ({len(ranked)} rankable rows, "
          f"{n_mined} configurations mined)\n",
          f"**{ARITHMETIC.format(n=n_mined)}**\n",
          f"`{STAMP}`\n",
          "## Top 20 by net %/day at stated capacity\n",
          "| # | universe | family | config | n | med/event | net %/day | p5 | ruin | capacity |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ranked[:20], 1):
        md.append(f"| {i} | {r['universe']} | {r['family']} | {r['config']} | {r['n']} "
                  f"| {r['median_pct']:+.3f} | {r['net_pct_day']:+.4f} | {r['tail_p5']:+.3f} "
                  f"| {r['ruin_frac']:.3f} | ${r['capacity_usd']:,} |")
    md.append("\n## Per-universe honest ceilings (best mined net %/day)\n")
    for u in ("equities", "nq", "options"):
        best = next((r for r in ranked if r["universe"] == u), None)
        md.append(f"- **{u}**: " + (f"{best['net_pct_day']:+.4f}/day — {best['family']} "
                  f"{best['config']} (n={best['n']}, p5={best['tail_p5']:+.3f}, "
                  f"cap=${best['capacity_usd']:,})" if best else "no rankable rows"))
    md.append("\n## Settled families (context, from the ledger — NOT mined here)\n")
    for c in CONTEXT_ROWS:
        md.append(f"- {c['universe']} / {c['family']}: {c['status']}")
    md.append("\n## Caveats (binding)\n")
    for c in CAVEATS:
        md.append(f"- {c.format(n=n_mined)}")
    md.append("\nNext: operator picks ≤3 rows → G0 preregs (HYP-093/094/095) → G1 "
              "holdout fetch → G2 gauntlet. Plan: Plans/immutable-wondering-alpaca.md")
    (OUT / "yield_board.md").write_text("\n".join(md))
    print(f"[M4] board: {len(ranked)} ranked rows, {n_mined} mined configs")
    for r in ranked[:12]:
        print(f"  {r['universe']:<9} {r['family']:<22} {r['config']:<42} "
              f"n={r['n']:>4} net/day={r['net_pct_day']:+.4f} p5={r['tail_p5']:+.3f} "
              f"cap=${r['capacity_usd']:,}")


if __name__ == "__main__":
    main()
