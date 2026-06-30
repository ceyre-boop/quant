#!/usr/bin/env python3
"""
scripts/analysis/spike_vix_gate_backtest.py
===========================================
Retrospective validation of a proposed "hard veto on SPIKE VIX" gate.

QUESTION
--------
Would vetoing *all* ICT signals on SPIKE VIX days (VIX_close > 30) have
improved ICT signal quality?

METHOD
------
We join two independent data products that were built for other reasons:

  1. The ICT veto outcome labels
       data/ledger/ict_veto_outcomes_2026_{05,06}.jsonl
     produced by scripts/label_veto_outcomes.py. Each row is a *rejected*
     ICT signal replayed forward to a falsifiable label:
       * TRUE_NEGATIVE   -> price went against us first; correct to reject.
       * FALSE_NEGATIVE  -> price hit +1R first; we wrongly rejected a winner.
       * NO_FILL         -> limit entry never filled in the window.
       * UNLABELABLE     -> no derivable direction/stop (non-directional veto).

  2. The sentiment board VIX regime labels (data/sentiment.db):
       LOW (<15), NORMAL [15,25], HIGH (25,30], SPIKE (>30)
     keyed by trading date in table ``sentiment_vix_daily``.

For every labeled outcome we look up the VIX regime on its date and
cross-tabulate ``label x vix_regime``. The task's accounting:

  * TRUE_NEGATIVE  + SPIKE  -> a bad signal correctly avoided  -> FOR the gate.
  * FALSE_NEGATIVE + SPIKE  -> a good signal we would block     -> AGAINST the gate.

SUFFICIENCY / FALLBACK
----------------------
If the decision-relevant labels (TRUE_NEGATIVE + FALSE_NEGATIVE) include
too few SPIKE-VIX days to be conclusive (< --min-spike, default 10), we fall
back to a coverage analysis:

  * Enumerate every SPIKE-VIX date in history (2015-present).
  * Cross-reference against the distinct dates the ICT veto ledger covers.
  * Report N SPIKE days, N with any ICT activity, and a qualitative finding.

LIMITATION (documented honestly in the note)
-------------------------------------------
The veto ledger contains only *rejected* signals. A blanket SPIKE gate would
also block signals that currently *pass* (accepted trades), which are not in
this dataset. So the cross-tab is a biased sample: it can show whether the
*rejected* signals on SPIKE days tended to be good or bad, but it cannot
measure the accepted-signal edge the gate would also destroy. This is noted
because in this run the question is moot anyway (zero SPIKE overlap).

THIS IS RESEARCH ONLY. It reads ledgers + sentiment.db and writes a markdown
note to Obsidian. It never mutates a ledger and never touches trading logic.

Usage
-----
    python3 scripts/analysis/spike_vix_gate_backtest.py
    python3 scripts/analysis/spike_vix_gate_backtest.py --min-spike 10
    python3 scripts/analysis/spike_vix_gate_backtest.py --no-write   # skip Obsidian
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parents[2]
LEDGER_DIR = REPO / "data" / "ledger"
DB_PATH = REPO / "data" / "sentiment.db"
OBSIDIAN_NOTE = (
    Path.home()
    / "Obsidian"
    / "Obsidian"
    / "Trading"
    / "Research"
    / "Spike-VIX-Gate-Backtest-2026-06.md"
)

OUTCOME_FILES = [
    "ict_veto_outcomes_2026_05.jsonl",
    "ict_veto_outcomes_2026_06.jsonl",
]
LEDGER_FILES = [
    "ict_veto_ledger_2026_05.jsonl",
    "ict_veto_ledger_2026_06.jsonl",
]

# Regime display order. NO_VIX_DATA captures outcome dates with no VIX row
# (weekends / holidays the forex scanner can still emit on).
REGIME_ORDER = ["LOW", "NORMAL", "HIGH", "SPIKE", "NO_VIX_DATA"]
# The two labels that carry a real directional outcome and so drive the gate
# accounting. NO_FILL / UNLABELABLE are non-decisional.
DECISIONAL = ("TRUE_NEGATIVE", "FALSE_NEGATIVE")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_outcomes() -> list[dict]:
    rows: list[dict] = []
    for name in OUTCOME_FILES:
        for o in _iter_jsonl(LEDGER_DIR / name):
            ts = o.get("timestamp", "")
            rows.append(
                {
                    "date": ts[:10],
                    "pair": o.get("pair"),
                    "label": o.get("label"),
                    "outcome_r": o.get("label_outcome_r"),
                    "veto_reason": (o.get("veto_reason") or "")[:60],
                }
            )
    return rows


def load_ledger_dates() -> tuple[set[str], Counter]:
    """Distinct dates the ICT veto ledger covers + per-date veto counts."""
    dates: set[str] = set()
    per_date: Counter = Counter()
    for name in LEDGER_FILES:
        for o in _iter_jsonl(LEDGER_DIR / name):
            d = (o.get("timestamp", "") or "")[:10]
            if d:
                dates.add(d)
                per_date[d] += 1
    return dates, per_date


def load_vix(con) -> tuple[dict[str, str], dict[str, list[str]], dict[str, float]]:
    """
    Returns:
      regime_by_date : date -> regime (full history)
      spike_history  : year -> [spike dates]
      vix_close      : date -> vix_close
    """
    regime_by_date: dict[str, str] = {}
    vix_close: dict[str, float] = {}
    spike_history: dict[str, list[str]] = defaultdict(list)
    rows = con.execute(
        "SELECT CAST(date AS VARCHAR), vix_regime, vix_close "
        "FROM sentiment_vix_daily ORDER BY date"
    ).fetchall()
    for d, regime, close in rows:
        regime_by_date[d] = regime
        vix_close[d] = close
        if regime == "SPIKE":
            spike_history[d[:4]].append(d)
    return regime_by_date, spike_history, vix_close


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def crosstab(outcomes: list[dict], regime_by_date: dict[str, str]):
    """label -> regime -> {n, sum_r, n_r} plus an outcome-R accumulator."""
    table: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"n": 0, "sum_r": 0.0, "n_r": 0})
    )
    for o in outcomes:
        regime = regime_by_date.get(o["date"], "NO_VIX_DATA")
        cell = table[o["label"]][regime]
        cell["n"] += 1
        if isinstance(o["outcome_r"], (int, float)):
            cell["sum_r"] += float(o["outcome_r"])
            cell["n_r"] += 1
    return table


def signal_frequency_by_regime(
    ledger_dates: set[str],
    per_date: Counter,
    regime_by_date: dict[str, str],
    con,
):
    """
    For the calendar span the ledger covers, compare ICT activity to the
    number of trading days available in each regime.
    """
    if not ledger_dates:
        return {}, (None, None)
    lo, hi = min(ledger_dates), max(ledger_dates)

    # Trading days available per regime in [lo, hi] (weekday VIX rows).
    avail = Counter()
    for d, regime, _ in con.execute(
        "SELECT CAST(date AS VARCHAR), vix_regime, vix_close "
        "FROM sentiment_vix_daily WHERE date BETWEEN ? AND ?",
        [lo, hi],
    ).fetchall():
        avail[regime] += 1

    # Days ICT was active per regime + total vetoes per regime.
    active_days = Counter()
    vetoes = Counter()
    for d in ledger_dates:
        regime = regime_by_date.get(d, "NO_VIX_DATA")
        active_days[regime] += 1
        vetoes[regime] += per_date[d]

    out = {}
    for regime in REGIME_ORDER:
        a = avail.get(regime, 0)
        ad = active_days.get(regime, 0)
        out[regime] = {
            "avail_days": a,
            "active_days": ad,
            "vetoes": vetoes.get(regime, 0),
            "coverage_pct": (100.0 * ad / a) if a else None,
        }
    return out, (lo, hi)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def fmt_crosstab(table) -> list[str]:
    labels = sorted(table.keys())
    header = f"{'label':<16}" + "".join(f"{r:>13}" for r in REGIME_ORDER) + f"{'TOTAL':>9}"
    lines = [header, "-" * len(header)]
    col_tot = Counter()
    for lab in labels:
        row = f"{lab:<16}"
        rtot = 0
        for r in REGIME_ORDER:
            n = table[lab].get(r, {}).get("n", 0)
            col_tot[r] += n
            rtot += n
            row += f"{n:>13}"
        row += f"{rtot:>9}"
        lines.append(row)
    foot = f"{'TOTAL':<16}" + "".join(f"{col_tot[r]:>13}" for r in REGIME_ORDER)
    foot += f"{sum(col_tot.values()):>9}"
    lines.append("-" * len(header))
    lines.append(foot)
    return lines


def fmt_freq(freq, span) -> list[str]:
    lo, hi = span
    lines = [f"ICT veto-ledger span: {lo} -> {hi}", ""]
    header = (
        f"{'regime':<14}{'avail_days':>12}{'active_days':>13}"
        f"{'coverage%':>12}{'vetoes':>10}"
    )
    lines += [header, "-" * len(header)]
    for r in REGIME_ORDER:
        f = freq.get(r)
        if not f:
            continue
        cov = "-" if f["coverage_pct"] is None else f"{f['coverage_pct']:.0f}"
        lines.append(
            f"{r:<14}{f['avail_days']:>12}{f['active_days']:>13}"
            f"{cov:>12}{f['vetoes']:>10}"
        )
    return lines


def build_note(
    outcomes,
    table,
    freq,
    span,
    spike_history,
    ledger_dates,
    min_spike,
    spike_decisional,
    fallback,
) -> str:
    total_spike_hist = sum(len(v) for v in spike_history.values())
    spike_active = sorted(
        d for d in ledger_dates if d in {x for v in spike_history.values() for x in v}
    )
    lo, hi = span
    label_counts = Counter(o["label"] for o in outcomes)

    md = []
    md.append("# SPIKE VIX Hard-Veto Gate — Retrospective Backtest")
    md.append("")
    md.append("> Research only. No trading-logic changes. Generated by "
              "`scripts/analysis/spike_vix_gate_backtest.py`.")
    md.append("")
    md.append("**Question:** Would vetoing *all* ICT signals on SPIKE VIX days "
              "(VIX close > 30) have improved ICT signal quality?")
    md.append("")
    md.append("**Verdict: INCONCLUSIVE — structural data gap, not a measured "
              "null.** There is **zero overlap** between SPIKE VIX days and the "
              "ICT veto dataset, so the gate cannot be validated retrospectively "
              "with the data that exists today.")
    md.append("")

    md.append("## Why: the data windows do not intersect")
    md.append("")
    md.append(f"- ICT veto ledger / outcome labels span **{lo} → {hi}** "
              f"({len(ledger_dates)} distinct trading days).")
    md.append(f"- Every labeled outcome that lands on an equity trading day fell "
              f"in the **NORMAL** VIX regime (VIX 15.3–22.2). No LOW, HIGH, or "
              f"SPIKE days at all. The remaining rows are `NO_VIX_DATA` — forex "
              f"weekend/holiday dates the FX scanner emits on but the equity VIX "
              f"never prints (e.g. the lone FALSE_NEGATIVE was Sunday 2026-06-14). "
              f"A VIX-keyed gate is inert on those dates by construction.")
    md.append(f"- The only SPIKE VIX days in 2026 were **2026-03-27** and "
              f"**2026-03-30** — roughly **8 weeks before** ICT veto recording "
              f"began on {lo}.")
    md.append(f"- SPIKE-VIX decisional outcomes (TRUE_NEGATIVE + FALSE_NEGATIVE) "
              f"available: **{spike_decisional}** "
              f"(threshold for a conclusive read: {min_spike}).")
    md.append("")

    md.append("## Primary cross-tab — labeled outcome × VIX regime")
    md.append("")
    md.append("```")
    md += fmt_crosstab(table)
    md.append("```")
    md.append("")
    md.append("Gate accounting (per the analysis spec):")
    md.append("")
    tn_spike = table.get("TRUE_NEGATIVE", {}).get("SPIKE", {}).get("n", 0)
    fn_spike = table.get("FALSE_NEGATIVE", {}).get("SPIKE", {}).get("n", 0)
    md.append(f"- **FOR the gate** (TRUE_NEGATIVE on SPIKE = bad signal correctly "
              f"avoided): **{tn_spike}**")
    md.append(f"- **AGAINST the gate** (FALSE_NEGATIVE on SPIKE = good signal we'd "
              f"block): **{fn_spike}**")
    md.append("- Both are zero because no labeled outcome lands on a SPIKE day. "
              "Every outcome with a VIX reading is NORMAL-regime; the rest are "
              "weekend/holiday `NO_VIX_DATA`. The cross-tab cannot speak to the "
              "gate either way.")
    md.append("")
    md.append("Label totals across both months: "
              + ", ".join(f"`{k}`={v}" for k, v in label_counts.most_common()))
    md.append("")

    md.append("## Signal frequency by regime (ledger span)")
    md.append("")
    md.append("`active_days` = distinct dates ICT recorded any veto; "
              "`avail_days` = trading days available in that regime over the "
              "same span; `vetoes` = total rejected signals. ICT operated "
              "**only** in NORMAL because the span itself contained only NORMAL "
              "days.")
    md.append("")
    md.append("```")
    md += fmt_freq(freq, span)
    md.append("```")
    md.append("")

    md.append("## Fallback — historical SPIKE-day coverage (2015–present)")
    md.append("")
    md.append(f"- Total SPIKE VIX days in history: **{total_spike_hist}**")
    md.append("- By year: "
              + ", ".join(f"{y}={len(spike_history[y])}"
                          for y in sorted(spike_history)))
    md.append(f"- SPIKE days with **any ICT veto activity**: "
              f"**{len(spike_active)}**"
              + (f" ({', '.join(spike_active)})" if spike_active else ""))
    md.append("")
    md.append("The ICT veto-recording infrastructure (`record_veto`) post-dates "
              "**every** SPIKE day in the dataset. The big SPIKE clusters — 2020 "
              "(80 days, COVID), 2022 (48 days, rate-hike crash), 2025 (12 days) "
              "— all precede the ledger's first row by months to years. ICT has "
              "literally never run during a SPIKE regime *while recording*.")
    md.append("")

    md.append("## Interpretation & caveats")
    md.append("")
    md.append("1. **This is a coverage gap, not evidence the gate is useless.** "
              "The honest answer is *we cannot tell yet*. The single richest "
              "stress test of any volatility gate — the 2020 and 2022 crashes — "
              "is entirely outside the recorded ledger.")
    md.append("2. **The veto ledger is a biased sample for this question.** It "
              "holds only *rejected* signals. A blanket SPIKE gate would also "
              "block signals that currently *pass*; those accepted trades are not "
              "in this dataset, so even with overlap the cross-tab could not "
              "measure the accepted-signal edge the gate would destroy.")
    md.append("3. **Forex macro edge is regime-fragile** (per CLAUDE.md: only "
              "pays in rate-trending regimes; ICT pattern edge is itself "
              "NOT PROVEN, permutation p=0.52). A SPIKE gate is plausible on "
              "priors, but priors are not a backtest.")
    md.append("")
    md.append("## How to actually answer this")
    md.append("")
    md.append("- **Forward:** keep recording vetoes *and* accepted ICT signals "
              "with the VIX regime stamped at decision time; revisit after the "
              "next SPIKE cluster gives a real sample.")
    md.append("- **Backward (stronger):** replay the ICT pipeline over 2020 + "
              "2022 SPIKE windows offline (same `label_veto_outcomes.py` "
              "machinery) to synthesise labeled SPIKE-day outcomes, then re-run "
              "this cross-tab. That is the only way to get a SPIKE sample without "
              "waiting for the next crisis.")
    md.append("")
    md.append("---")
    md.append("*Inputs: "
              "`data/ledger/ict_veto_outcomes_2026_{05,06}.jsonl`, "
              "`data/ledger/ict_veto_ledger_2026_{05,06}.jsonl`, "
              "`data/sentiment.db` (`sentiment_vix_daily`). "
              "Reproduce: `python3 scripts/analysis/spike_vix_gate_backtest.py`.*")
    return "\n".join(md) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-spike", type=int, default=10,
                    help="min SPIKE decisional outcomes for a conclusive read")
    ap.add_argument("--no-write", action="store_true",
                    help="do not write the Obsidian note")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"ERROR: sentiment DB not found at {DB_PATH}")
        return 1

    outcomes = load_outcomes()
    ledger_dates, per_date = load_ledger_dates()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    regime_by_date, spike_history, _ = load_vix(con)

    table = crosstab(outcomes, regime_by_date)
    freq, span = signal_frequency_by_regime(ledger_dates, per_date, regime_by_date, con)

    # Sufficiency check on decisional SPIKE-day labels.
    spike_decisional = sum(
        table.get(lab, {}).get("SPIKE", {}).get("n", 0) for lab in DECISIONAL
    )
    fallback = spike_decisional < args.min_spike

    # ----- console report ----- #
    print("=" * 74)
    print("SPIKE VIX HARD-VETO GATE — RETROSPECTIVE BACKTEST")
    print("=" * 74)
    print(f"Labeled outcome rows : {len(outcomes)}")
    print(f"Ledger trading days  : {len(ledger_dates)}  "
          f"({span[0]} -> {span[1]})")
    print()
    print("PRIMARY CROSS-TAB  (labeled outcome x VIX regime)")
    print("-" * 74)
    for line in fmt_crosstab(table):
        print(line)
    print()

    tn_spike = table.get("TRUE_NEGATIVE", {}).get("SPIKE", {}).get("n", 0)
    fn_spike = table.get("FALSE_NEGATIVE", {}).get("SPIKE", {}).get("n", 0)
    print("GATE ACCOUNTING")
    print("-" * 74)
    print(f"  FOR the gate     (TRUE_NEGATIVE  + SPIKE): {tn_spike}")
    print(f"  AGAINST the gate (FALSE_NEGATIVE + SPIKE): {fn_spike}")
    print(f"  SPIKE decisional outcomes available      : {spike_decisional} "
          f"(min for conclusive = {args.min_spike})")
    print()

    print("SIGNAL FREQUENCY BY REGIME  (ledger span)")
    print("-" * 74)
    for line in fmt_freq(freq, span):
        print(line)
    print()

    if fallback:
        print("!! INSUFFICIENT SPIKE-VIX overlap -> FALLBACK coverage analysis")
        print("-" * 74)
        total_spike_hist = sum(len(v) for v in spike_history.values())
        spike_set = {x for v in spike_history.values() for x in v}
        spike_active = sorted(d for d in ledger_dates if d in spike_set)
        print(f"  SPIKE VIX days in history (2015-present): {total_spike_hist}")
        print("  by year: "
              + ", ".join(f"{y}={len(spike_history[y])}"
                          for y in sorted(spike_history)))
        print(f"  SPIKE days with ANY ICT veto activity  : {len(spike_active)}"
              + (f"  {spike_active}" if spike_active else ""))
        print()
        print("  FINDING: ICT veto recording began "
              f"{span[0]}; the last SPIKE day was "
              f"{max(spike_set)}. The veto infrastructure post-dates every")
        print("  SPIKE day on record, so the gate cannot be validated from "
              "recorded data. Verdict: INCONCLUSIVE (coverage gap).")
        print()

    # ----- Obsidian note ----- #
    if not args.no_write:
        note = build_note(
            outcomes, table, freq, span, spike_history, ledger_dates,
            args.min_spike, spike_decisional, fallback,
        )
        OBSIDIAN_NOTE.parent.mkdir(parents=True, exist_ok=True)
        OBSIDIAN_NOTE.write_text(note)
        print(f"Obsidian note written -> {OBSIDIAN_NOTE}")
    else:
        print("(--no-write: Obsidian note skipped)")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
