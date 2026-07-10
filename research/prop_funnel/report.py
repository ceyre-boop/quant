"""Verdict table, charts, and the readable report (TICK-022 P6).

Everything lands under data/research/prop_funnel/. Every row carries its evidence
stamp; carry rows carry the regime caveat verbatim; the i.i.d.-attempts caveat is
printed on the table itself, not buried in a footnote.
"""
from __future__ import annotations

import csv
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.prop_funnel._lib import CHARTS_DIR, OUT_DIR

IID_CAVEAT = ("Assumes i.i.d. attempts/months — FALSE under regime shift. p^100 in particular "
              "treats 100 attempts as independent draws of the same edge; a regime that kills "
              "the edge kills all remaining attempts at once.")

TABLE_COLS = [
    ("strategy", "Strategy"), ("stamp", "Evidence"), ("firm", "Firm"),
    ("p_funded", "P(funded)"), ("p_pass_100_consecutive", "P(pass 100/100)"),
    ("med_cal_days", "Med cal-days P1"), ("e_attempts_to_funded", "E[attempts]"),
    ("fees_to_funded_usd", "Fees→funded $"), ("p_survive_12mo", "P(surv 12mo)"),
    ("e_payout_per_month_usd", "E[payout/mo] $"), ("p_month_ge_target", "P(mo ≥$10k)"),
    ("p_target_every_month_12", "P($10k every mo ×12)"),
    ("program_ev_per_month_usd", "Program EV/mo $"), ("pricing_flag", "Pricing"),
]


def _flat(row: dict) -> dict:
    if row.get("verdict") == "INSUFFICIENT_DATA":
        return {"strategy": row["strategy"], "stamp": row["stamp"], "firm": row["firm"],
                "p_funded": "INSUFFICIENT_DATA", **{k: "" for k, _ in TABLE_COLS[4:]},
                "note": row.get("note", "")}
    f = row.get("funded", {})
    p1 = row["phases"][0] if row.get("phases") else {}
    return {
        "strategy": row["strategy"], "stamp": row["stamp"], "firm": row["firm"],
        "p_funded": row.get("p_funded"),
        "p_pass_100_consecutive": row.get("p_pass_100_consecutive"),
        "med_cal_days": p1.get("cal_days_to_pass_med"),
        "e_attempts_to_funded": row.get("e_attempts_to_funded"),
        "fees_to_funded_usd": row.get("fees_to_funded_usd"),
        "p_survive_12mo": f.get("p_survive_12mo"),
        "e_payout_per_month_usd": f.get("e_payout_per_month_usd"),
        "p_month_ge_target": f.get("p_month_ge_target"),
        "p_target_every_month_12": f.get("p_target_every_month_12"),
        "program_ev_per_month_usd": row.get("program_ev_per_month_usd"),
        "pricing_flag": row.get("pricing_flag", ""),
        "note": (row.get("caveat") or "")[:140],
    }


def write_verdict_table(rows: list) -> None:
    flats = sorted((_flat(r) for r in rows),
                   key=lambda x: (x["program_ev_per_month_usd"]
                                  if isinstance(x.get("program_ev_per_month_usd"), (int, float))
                                  else -1e18),
                   reverse=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "verdict_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k, _ in TABLE_COLS] + ["note"])
        w.writeheader()
        w.writerows(flats)

    lines = ["# Prop-Funnel Verdict Table (TICK-022)", "",
             f"> **CAVEAT (applies to every row):** {IID_CAVEAT}", "",
             "| " + " | ".join(h for _, h in TABLE_COLS) + " |",
             "|" + "|".join("---" for _ in TABLE_COLS) + "|"]
    for x in flats:
        lines.append("| " + " | ".join(str(x.get(k, "")) for k, _ in TABLE_COLS) + " |")
    lines += ["", "## Row caveats", ""]
    for x in flats:
        if x.get("note"):
            lines.append(f"- **{x['strategy']} × {x['firm']}** — {x['note']}")
    (OUT_DIR / "verdict_table.md").write_text("\n".join(lines))


# ── charts ──────────────────────────────────────────────────────────────────

def _grid(front: dict, key: str, freq: float) -> np.ndarray:
    sharpes, vols = front["sharpes"], front["vols"]
    g = np.full((len(vols), len(sharpes)), np.nan)
    for c in front["cells"]:
        if c["trades_per_day"] == freq and c.get(key) is not None:
            g[vols.index(c["vol_daily"]), sharpes.index(c["sharpe"])] = c[key]
    return g


def _heatmap(front: dict, key: str, title: str, fname: str, freq: float = 1.0,
             fmt: str = "{:.0%}") -> None:
    g = _grid(front, key, freq)
    sharpes, vols = front["sharpes"], front["vols"]
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(g, aspect="auto", origin="lower", cmap="RdYlGn",
                   vmin=np.nanmin(g), vmax=np.nanmax(g))
    ax.set_xticks(range(len(sharpes)), [f"{s:g}" for s in sharpes])
    ax.set_yticks(range(len(vols)), [f"{v:.2%}" for v in vols])
    ax.set_xlabel("TRUE annualized Sharpe")
    ax.set_ylabel("daily equity vol")
    ax.set_title(f"{title}\n{front['firm']} · {freq:g} trades/day · SYNTHETIC frontier")
    for i in range(len(vols)):
        for j in range(len(sharpes)):
            if not np.isnan(g[i, j]):
                ax.text(j, i, fmt.format(g[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / fname, dpi=150)
    plt.close(fig)


def tension_chart(front: dict, fname: str, freq: float = 1.0) -> None:
    """The one chart that answers Colin's question: pass-rate, income and ruin
    cannot all be had at once. Contours on the same Sharpe x vol plane."""
    sharpes, vols = front["sharpes"], front["vols"]
    X, Y = np.meshgrid(sharpes, vols)
    p_pass = _grid(front, "p_funded", freq)
    p_10k = _grid(front, "p_month_ge_target", freq)
    p_bust = _grid(front, "p_bust_funded_month", freq)

    fig, ax = plt.subplots(figsize=(9, 6))
    cs1 = ax.contour(X, Y, p_pass, levels=[0.90, 0.99], colors="tab:green",
                     linestyles=["--", "-"])
    ax.clabel(cs1, fmt={0.90: "pass 90%", 0.99: "pass 99%"}, fontsize=8)
    cs2 = ax.contour(X, Y, p_10k, levels=[0.25, 0.50], colors="tab:blue",
                     linestyles=["--", "-"])
    ax.clabel(cs2, fmt={0.25: "$10k/mo 25%", 0.50: "$10k/mo 50%"}, fontsize=8)
    cs3 = ax.contour(X, Y, p_bust, levels=[0.05], colors="tab:red")
    ax.clabel(cs3, fmt={0.05: "funded-month bust 5%"}, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("TRUE annualized Sharpe")
    ax.set_ylabel("daily equity vol (log)")
    ax.set_title(f"The pass-vs-income tension — {front['firm']} ({freq:g} trades/day)\n"
                 "green = pass the challenge · blue = make $10k months · red = ruin risk\n"
                 "No region satisfies all three without Sharpe ≳ 2-3 — SYNTHETIC map, existence not claimed")
    fig.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / fname, dpi=150)
    plt.close(fig)


def sizing_heatmap(opt: dict, fname: str) -> None:
    mults = opt["mults"]
    g = np.full((len(mults), len(mults)), np.nan)
    for c in opt["cells"]:
        if c["ev_per_month"] is not None:
            g[mults.index(c["funded_mult"]), mults.index(c["challenge_mult"])] = c["ev_per_month"]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(g, aspect="auto", origin="lower", cmap="RdYlGn")
    ax.set_xticks(range(len(mults)), [f"{m:g}x" for m in mults])
    ax.set_yticks(range(len(mults)), [f"{m:g}x" for m in mults])
    ax.set_xlabel("challenge risk multiplier")
    ax.set_ylabel("funded risk multiplier")
    best = opt.get("best") or {}
    ax.set_title(f"Program EV/month — {opt['strategy']} × {opt['firm']} [{opt['stamp']}]\n"
                 f"best: challenge {best.get('challenge_mult')}x / funded {best.get('funded_mult')}x "
                 f"→ ${best.get('ev_per_month')}/mo")
    for i in range(len(mults)):
        for j in range(len(mults)):
            if not np.isnan(g[i, j]):
                ax.text(j, i, f"{g[i, j]:,.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / fname, dpi=150)
    plt.close(fig)


def days_to_pass_chart(rows: list, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels, meds, p90s = [], [], []
    for r in rows:
        if r.get("verdict") == "INSUFFICIENT_DATA" or not r.get("phases"):
            continue
        p1 = r["phases"][0]
        if p1.get("cal_days_to_pass_med") is None:
            continue
        labels.append(f"{r['strategy']}\n× {r['firm']}")
        meds.append(p1["cal_days_to_pass_med"])
        p90s.append((p1["tdays_to_pass_p90"] or 0) * 365 / 252)
    y = np.arange(len(labels))
    ax.barh(y, p90s, color="lightsteelblue", label="p90")
    ax.barh(y, meds, color="tab:blue", label="median")
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("calendar days to pass Phase 1")
    ax.set_title("Time-to-pass is a first-class cost (no-time-limit ≠ fast)")
    ax.legend()
    fig.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / fname, dpi=150)
    plt.close(fig)


# ── the readable report ─────────────────────────────────────────────────────

def write_summary(rows: list, fronts: list, opts: list, parity: dict,
                  live_pool_meta: dict, seed: int, env: dict) -> None:
    def best_real() -> Optional[dict]:
        real = [r for r in rows
                if r.get("stamp") == "PROVEN_REGIME_FRAGILE"
                and isinstance(r.get("program_ev_per_month_usd"), (int, float))]
        return max(real, key=lambda r: r["program_ev_per_month_usd"]) if real else None

    br = best_real()
    lines = [
        "# Prop-Funnel EV Simulator — Summary Report (TICK-022)",
        "",
        f"Seed {seed} · env {env} · parity: "
        f"{'ALL EXACT' if parity.get('all_exact') else ('GREEN' if parity.get('all_ok') else 'RED')}",
        "",
        "## The two questions this tool was built to answer",
        "",
        "**1. \"Can a strategy pass a sim prop test 100 times?\"** — Only with a TRUE Sharpe well "
        "above anything this firm has proven. See `charts/frontier_pass_*.png`: P(pass 100/100) "
        "requires per-attempt pass ≥ 99.99%%… realistically Sharpe ≳ 2 at low vol — and at low vol "
        "the same configuration cannot produce meaningful monthly income. The verdict table's "
        "`P(pass 100/100)` column shows every real strategy's number.",
        "",
        "**2. \"Can it make $10k/month consistently for years?\"** — See "
        "`P($10k every mo ×12)` in the table and `charts/tension_*.png`: the pass contour and the "
        "income contour do not overlap at survivable ruin risk until TRUE Sharpe ≳ 2-3, or until "
        "capital is much larger (income scales with account size; a $10k month on $100k = 10%/mo).",
        "",
        f"> **{IID_CAVEAT}**",
        "",
        "## Honest input inventory",
        "",
        "- CARRY (PROVEN, regime-fragile): decade Sharpe 0.69, OOS-window 1.25, fresh 2025-26 ≈ flat. "
        "The forward-band SCENARIO rows {0, 0.69, 1.25} bracket what carry might be going forward.",
        "- ICT (UNPROVEN p=0.52): backtest pools only. The live sample is its own finding: "
        f"**{live_pool_meta.get('n')} closed outcomes, {live_pool_meta.get('wins')}W/"
        f"{live_pool_meta.get('losses')}L** vs backtest WR ~63.6%% (selection-biased, LOW_N).",
        "- FUTURES ORB (UNVALIDATED): n=2 replay trades → INSUFFICIENT_DATA row; Phase R "
        "(operator-gated) can regenerate a larger replay pool.",
        "- SYNTHETIC frontier: requirements map, existence not claimed.",
        "",
        "## Program EV reality",
        "",
    ]
    if br:
        lines.append(
            f"Best PROVEN-strategy cell: **{br['strategy']} × {br['firm']}** → program EV "
            f"≈ ${br['program_ev_per_month_usd']}/mo at P(funded) {br['p_funded']} "
            f"(pricing {br.get('pricing_flag')}). Set against the caveat that fresh-window carry "
            f"measured ≈ flat — the SCENARIO S0 row is the pessimistic bracket.")
    lines += [
        "",
        "## Open items for Colin",
        "",
        "1. **Pricing**: every fee/payout number is UNVERIFIED_PRICING — verify against live firm "
        "pricing before acting on EV rankings.",
        "2. **Return-scale convention**: carry rows use the monte_carlo_prop convention "
        "(R = pnl_pct/risk_pct). The equity-curve display convention is ~100x smaller in dollars.",
        "3. **rules_engine.py divergence (documented, not fixed)**: its `dd_trail_stops_at_starting` "
        "caps the floor at initial−dd from day one, making its 'trailing' effectively static. Parity "
        "presets mirror it; TOPSTEP/APEX rows use real trailing semantics.",
        "4. **Intraday-trailing bracket**: APEX-style rows use the pessimistic κ-stressed bound "
        "(no MFE data on daily pools).",
        "5. **Zero-edge rows show positive EV/mo — read carefully**: the funded account is "
        "effectively a call option (payouts keep positive months, the firm eats drawdowns), and "
        "the reset-to-initial payout policy plus unverified pricing make that option value "
        "model-OPTIMISTIC. Real firms' consistency rules, payout minimums and pricing exist "
        "precisely to close this; do NOT read S0-positive-EV as free money.",
        "",
        "## Sizing policy results",
        "",
    ]
    for o in opts:
        b = o.get("best") or {}
        lines.append(f"- {o['strategy']} × {o['firm']} [{o['stamp']}]: best challenge "
                     f"{b.get('challenge_mult')}x / funded {b.get('funded_mult')}x → "
                     f"${b.get('ev_per_month')}/mo (see charts/sizing_*.png). {o['note']}")
    lines += ["", "## Artifacts", "",
              "- `verdict_table.md` / `.csv` — every strategy × firm, sorted by program EV/mo",
              "- `charts/frontier_pass_*.png`, `charts/frontier_income_*.png` — requirements maps",
              "- `charts/tension_*.png` — the pass-vs-income-vs-ruin contours",
              "- `charts/sizing_*.png` — policy grids · `charts/days_to_pass.png` — time cost",
              "- `results.json` — full machine-readable output · `parity/parity_report.json`"]
    (OUT_DIR / "summary_report.md").write_text("\n".join(lines))
