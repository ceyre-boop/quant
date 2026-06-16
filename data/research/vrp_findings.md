# VRP Findings — Volatility Risk Premium (SPY/QQQ)

*Generated 2026-06-16T17:18:47.724050+00:00 · verdict **DATA_INSUFFICIENT** · research only, validate-don't-deploy*

## Pre-registered hypothesis
The SPY/QQQ volatility risk premium (implied vol > realized vol) is a real systematic premium. The decisive question is whether it is an **orthogonal** second edge to the live carry edge, or correlated **return-stacking** like overnight-QQQ.

Mechanism (academic, 30+ yrs): Coval & Shumway (2001); Bakshi & Kapadia (2003); Bollerslev, Tauchen & Zhou (2009). Option buyers are net hedgers/speculators who pay premium for defined outcomes; sellers earn it for warehousing volatility risk. The premium is real and large in calm regimes and collapses/inverts in vol shocks (2008/2020/2022) — that is the cost of harvesting it.

## Why this is the inverted gauntlet (not the brief's 4-stage order)
The system has **no historical SPY/QQQ option chains** (yfinance = current only; `data/polygon_client.py` has no options endpoints). The brief forbids synthesizing option prices, so the iron-condor backtest is genuinely blocked. Per the operation's own lesson (`sovereign_core_verdict`: run the cheapest falsifying test first), we run the free orthogonality kill-gate **before** spending money on option data. If VRP re-couples with carry in crisis, it is return-stacking — same fate as overnight-QQQ — and no option data is worth procuring.

## Stage 1 — does VRP exist? (BTZ forward IV−RV gap)
- **SPY/^VIX**: mean gap 0.00985 (ann. var), 0.84 of days positive, t=16.674, permutation p=9.999000099990002e-05 → exists=True
- **QQQ/^VXN**: mean gap 0.01853, 0.814 positive, t=22.005, p=9.999000099990002e-05 → exists=True
- **Both-sides (NN#2), SPY**: calm VIX≤30 mean 0.00854 vs stressed VIX>30 mean 0.02408. Both sides positive — the FORWARD gap is even larger right after spikes (implied overshoots and mean-reverts). The crisis *cost* of harvesting does not show up in this forward existence gap; it shows up in the Stage-2 daily harvest mark — which is exactly why Stage 2, not Stage 1, is the kill-gate.

## Stage 2 — orthogonality kill-gate (causal harvest return vs carry)
- harvest standalone: Sharpe(rf0)=1.412, 0.807 positive days (n=8401)
- **vs DBV carry**: full ρ=0.057 (LOW), max crisis |ρ|=0.216, VIX>30 ρ=0.065 → **TRUE_DIVERSIFIER**
- vs v015 carry (recent secondary): TRUE_DIVERSIFIER
- vs overnight-QQQ: CORRELATED_IN_CRISIS

## Stage 3 — iron-condor strategy backtest
**DATA_INSUFFICIENT.** No historical SPY/QQQ option chains available; brief forbids synthesizing option prices. The pre-registered strategy + cost spec are frozen in `strategy_simulator.py`, ready for the day real chains exist. Candidate providers and required fields are listed in the JSON.

## Verdict
**DATA_INSUFFICIENT** — gates: {"vrp_exists": true, "carry_relationship": "TRUE_DIVERSIFIER", "recouples_with_carry": false, "orthogonal_to_carry": true}

## Caveats
- No historical SPY/QQQ option chains in this system — the iron-condor backtest is BLOCKED, not faked.
- Stage-1 BTZ gap uses forward realized variance (look-ahead) and is quarantined to the existence stat.
- Stage-2 harvest return is strictly causal: (IV_{t-1}/100)^2/252 - r_t^2.
- Carry crisis coverage comes from DBV (2006->2023-03); the v015 forex-log carry is recent/noisy secondary.
- Crisis/stress correlation decides orthogonality, NOT the benign full-sample average.
- The Stage-2 harvest proxy is a daily LINEAR variance P&L; a real short iron condor's crisis loss is CONVEX and clusters in vol spikes, so this proxy likely UNDERSTATES true crisis coupling. A TRUE_DIVERSIFIER read vs carry is encouraging but must be confirmed on real chains, not banked.
- Harvest standalone Sharpe is UNCOSTED and on a proxy series, not the tradeable strategy — do not treat it as a deployable number (the operation's prior over-annualized-Sharpe trap).
