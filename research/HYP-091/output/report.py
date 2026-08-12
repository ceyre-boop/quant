"""Phase 4 — human-readable report for HYP-091. Verdict first, then the numbers
and honest limitations. Writes data/research/tsmom_hyp091/REPORT.md."""
from __future__ import annotations

from research.tsmom_hyp091._lib import OUT_DIR


def write_report(results: dict) -> None:
    b = results["backtest"]; c = results["correlation"]; g = results["gauntlet"]
    prim = c["primary"]; brk = c["robustness_broken_model"]
    by_mode = b["sharpe_by_mode"]

    lines = []
    lines.append(f"# HYP-091 — TSMOM diversification of the v015 carry book\n")
    lines.append(f"**VERDICT: {results['verdict']}**  ·  prior_expectation: {results['prior_expectation']}  ·  "
                 f"prereg `{results['prereg']}`  ·  TICK-027\n")
    lines.append("Corrected pre-registration vs the parallel HYP-089 quick-look "
                 "(proxy correlation + no financing + daily rebalance). Here: **monthly** rebalance "
                 "(Moskowitz), correlation vs the **actual v015 returns**, and **correct "
                 "rate-differential-derived financing** (NOT the Colin-gated SWAP_RATES_ANNUAL; TICK-024).\n")

    lines.append("## Why the null triggered\n")
    for why in g["verdict_reasons"]:
        lines.append(f"- {why}")
    lines.append("")

    lines.append("## 1. Standalone Sharpe by financing model\n")
    lines.append("| Financing | Full | IS (…2022) | OOS (2023-24) | mean/mo | n |")
    lines.append("|-----------|------|-----------|---------------|---------|---|")
    for m, label in (("ratediff", "ratediff (PRIMARY, correct)"), ("broken", "broken SWAP_RATES (robust.)"),
                     ("none", "price-only (robust.)")):
        s = by_mode[m]
        lines.append(f"| {label} | {s['full']:+.3f} | {s['is']:+.3f} | {s['oos']:+.3f} | {s['mean_monthly']:+.5f} | {s['n']} |")
    lines.append("")
    lines.append("Correct financing makes the strategy **worse** than price-only (it pays the real carry "
                 "costs the broken model understated ~10x) — the OOS Sharpe is negative either way.\n")

    lines.append("## 2. Per-calendar-year Sharpe (ratediff primary — DESCRIPTIVE, ~12 obs/yr)\n")
    lines.append("| Year | Sharpe | Positive? | n |")
    lines.append("|------|--------|-----------|---|")
    for r in b["per_year_ratediff"]:
        lines.append(f"| {r['year']} | {r['sharpe']:+.3f} | {'✅' if r['sharpe'] > 0 else '❌'} | {r['n']} |")
    lines.append(f"\n**Positive years: {b['positive_years']}/{b['total_years']}.** 2022 (the rate-trending "
                 "regime) dominates; the OOS years 2023/2024 are ~flat/negative — the concentration the "
                 "per-year table was designed to expose.\n")

    lines.append("## 3. Correlation vs the ACTUAL v015 carry (monthly)\n")
    lines.append(f"- Primary (correct-financing TSMOM vs v015): **ρ = {prim['corr_full']}** over "
                 f"{prim['n_overlap_months']} months (SE≈{prim['corr_SE_approx']}); 2022-window ρ = {prim['corr_2022']}.")
    lines.append(f"- Robustness (broken-model TSMOM vs v015, the mismatch-free apples-to-apples): "
                 f"ρ = {brk['corr_full']}.")
    lines.append(f"- Correlation is LOW (well below the 0.5 null bar) — so TSMOM would diversify IF it had "
                 f"positive OOS Sharpe. It does not.")
    lines.append(f"- Confirmatory 50/50 equal-vol blend Sharpe = {prim['sharpe_5050_blend']} vs "
                 f"max(tsmom {prim['sharpe_tsmom_overlap']}, v015 {prim['sharpe_v015_overlap']}); "
                 f"diversification lift = {prim['diversification_lift']}.\n")

    lines.append("## 4. Gauntlet\n")
    lines.append("| Gate | Result | Pass |")
    lines.append("|------|--------|------|")
    lines.append(f"| pre-reg null (OOS Sharpe>0 AND |corr|≤0.5) | OOS {g['oos_sharpe']:+.3f}, "
                 f"null_triggered={g['null_triggered']} | {'✅' if not g['null_triggered'] else '❌'} |")
    lines.append(f"| directional permutation p<0.05 | p={g['permutation_p']} (N={g['n_perm']}) | "
                 f"{'✅' if g['permutation_p'] < 0.05 else '❌'} |")
    lines.append(f"| deflated-Sharpe prob>0.95 | {g['deflated_sharpe_prob']} | "
                 f"{'✅' if g['deflated_sharpe_prob'] > 0.95 else '❌'} |")
    lines.append(f"| BH survives | {g['bh_survives']} | {'✅' if g['bh_survives'] else '❌'} |")
    lines.append(f"| holdout OOS Sharpe>0 | {g['oos_sharpe']:+.3f} | {'✅' if g['oos_sharpe'] > 0 else '❌'} |")
    lines.append("\nFamily of one → DSR/BH are near-vacuous; the real guards are the permutation timing "
                 "test and the Phase-0 hash-lock.\n")

    lines.append("## 5. Honest limitations\n")
    lines.append(f"- ~{b['window']['n_months']} monthly obs / ~{g['n_oos_months']} OOS / ~12 per year → LOW "
                 "power; NOT_SIGNIFICANT is about as consistent with low power as with no edge (hence the prior).")
    lines.append("- Per-year Sharpes are descriptive (SE ≈ ±0.5), not inferential.")
    lines.append("- yfinance FX spot ≠ tradable forward; financing is the rate-differential-anchored model "
                 "(reproduces the 2026 OANDA snapshot + trade-227), not live broker fills.")
    lines.append("- Primary ρ correlates correct-financing TSMOM against the v015 CSV (costed with the broken "
                 "swap) — the broken-model robustness leg is the mismatch-free cross-check and agrees.")
    lines.append("- **Research pass only.** VALID_EDGE/NOT_SIGNIFICANT — no deployment, no live capital "
                 "(RISK_CONSTITUTION Art. 6; promotion is a separate human step).\n")

    (OUT_DIR / "REPORT.md").parent.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")
