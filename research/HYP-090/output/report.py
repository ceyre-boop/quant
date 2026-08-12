"""P4: charts, summary report, and the ledger verdict seal (HYP-090)."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.modern._lib import CHARTS_DIR, LEDGER_PATH, OUT_DIR


def equity_chart(a0, runs, placebo_sharpes, eval_dates, fname="equity_arms.png"):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(eval_dates, np.cumsum(a0), color="black", lw=2, label="A0 static v015")
    for run in runs:
        ax.plot(eval_dates, np.cumsum(run["returns"]), lw=1,
                label=f"{run['arm']} W{run['window']}")
    ax.set_title("HYP-090 MODERN — adaptive arms vs static v015 (costed daily M2M, cumulative)\n"
                 f"placebo (A3) Sharpe p95 by window: "
                 + ", ".join(f"W{w}={s:.2f}" for w, s in placebo_sharpes.items()))
    ax.legend(fontsize=8)
    ax.set_ylabel("cumulative return (fraction)")
    fig.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / fname, dpi=150)
    plt.close(fig)


def selection_timeline(run, uni, eval_dates, fname):
    cfg_ids = np.array([uni.variants[v][0] for v in run["selections"]])
    n_pairs = np.array([len(uni.variants[v][1]) for v in run["selections"]])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.scatter(eval_dates, cfg_ids, s=1)
    ax1.set_ylabel("selected config id")
    ax1.set_title(f"{run['arm']} W{run['window']}: what the machine chose "
                  f"({run['n_switches']} switches)")
    ax2.plot(eval_dates, n_pairs, lw=0.7)
    ax2.set_ylabel("# pairs in subset")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / fname, dpi=150)
    plt.close(fig)


def per_year_chart(adjud, fname="per_year.png"):
    years = sorted(adjud["a0_per_year"])
    x = np.arange(len(years))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.35, [adjud["a0_per_year"][y] for y in years], 0.2, label="A0", color="black")
    for i, row in enumerate(adjud["rows"]):
        ax.bar(x - 0.25 + (i + 1) * 0.09, [row["per_year"][y] for y in years], 0.08,
               label=row["run"])
    ax.set_xticks(x, years)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title("Per-year daily-M2M Sharpe — criterion 4 (non-degrade, tol 0.05)")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / fname, dpi=150)
    plt.close(fig)


def write_summary(adjud, prereg, seed, env, runtime_s) -> None:
    v = adjud["verdict"]
    rows = adjud["rows"]
    lines = [
        "# HYP-090 MODERN — Adjudication Report (TICK-023)",
        "",
        f"**VERDICT: {v}** · prereg hash {prereg['hash_lock'][:12]}… verified pre/post · "
        f"seed {seed} · env {env} · runtime {runtime_s:.0f}s",
        "",
        f"Registered prior: **{prereg['prior_expectation']}**. "
        f"A0 static v015 daily-M2M Sharpe on the replay span: **{adjud['a0_sharpe']}**.",
        "",
        "| run | Sharpe | costed | p vs A0 | BH | > placebo p95 | DSR@5775 | per-year | all |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['sharpe']} | {r['sharpe_costed']} | {r['p_vs_a0']:.4f} | "
            f"{'✓' if r['c1_bh_survives'] else '✗'} | "
            f"{'✓' if r['c2_beats_placebo'] else '✗'} ({r['placebo_p95']}) | "
            f"{r['dsr']} {'✓' if r['c3_dsr_positive'] else '✗'} | "
            f"{'✓' if r['c4_per_year_nondegrade'] else '✗'} | "
            f"{'PASS' if r['all_pass'] else '—'} |")
    lines += [
        "",
        "Criteria are the prereg's locked wording (verdict_criteria). The A3 placebo "
        "envelope is the selection-noise floor: beating A0 while not beating A3 is the "
        "in-sample-inflation signature, not an edge.",
        "",
        "Prior family kills (disclosed, not double-counted): HYP-065, HYP-066, HYP-067, "
        "the 180-config exit sweep, the regime router.",
        "",
        "Charts: `charts/equity_arms.png`, `charts/selection_timeline_*.png`, "
        "`charts/per_year.png`. Full numbers: `results.json`.",
    ]
    (OUT_DIR / "summary_report.md").write_text("\n".join(lines))


def seal_ledger(adjud: dict) -> None:
    """Write the verdict into the HYP-090 ledger entry (backup + atomic replace;
    status stays PREREGISTERED per repo convention, verdict + annotation added)."""
    ledger = json.loads(LEDGER_PATH.read_text())
    entry = next(e for e in ledger if e.get("id") == "HYP-090")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = LEDGER_PATH.with_suffix(f".bak-{stamp}.json")
    shutil.copy2(LEDGER_PATH, backup)

    entry["verdict"] = adjud["verdict"]
    entry["result"] = adjud["verdict"]
    entry["p_value"] = adjud["primary_p_min"]
    entry["bh_survives"] = any(r["c1_bh_survives"] and r["c1_p_lt_05"] for r in adjud["rows"])
    entry.setdefault("annotations", []).append({
        "date": stamp[:8], "note": (
            f"HYP-090 adjudicated {adjud['verdict']}: best run {adjud['best_run']} "
            f"Sharpe {max(r['sharpe'] for r in adjud['rows'])} vs A0 {adjud['a0_sharpe']}; "
            f"min p={adjud['primary_p_min']:.4f}; criteria per locked prereg 6dd9cc85"),
    })
    with tempfile.NamedTemporaryFile("w", dir=LEDGER_PATH.parent, suffix=".tmp",
                                     delete=False) as tmp:
        tmp.write(json.dumps(ledger, indent=2) + "\n")
    Path(tmp.name).replace(LEDGER_PATH)
    check = json.loads(LEDGER_PATH.read_text())
    assert any(e.get("id") == "HYP-090" and e.get("verdict") == adjud["verdict"]
               for e in check), "ledger seal failed post-write check"
    print(f"ledger sealed: HYP-090 verdict={adjud['verdict']} (backup {backup.name})")
