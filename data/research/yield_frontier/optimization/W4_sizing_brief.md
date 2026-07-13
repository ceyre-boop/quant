# W4 — Sizing Under Jump Risk (specialist agent brief, 2026-07-13)
Stamp: research input to TICK-033; informs the W6 simulator spec; NOT evidence.

[Verbatim agent deliverable follows]

## Framing
Per-event distribution (median +4.9%, mean +1.6%, p5 -32.6%, worst -60% observed) = positive-drift
process with left-tail jumps. Key results: (1) Kelly under jumps != Merton-Kelly — hard constraint is
1 + f*R_min > 0 with a STRESSED worst jump, not the observed one (Bermin & Holm 2026). (2) Full Kelly
fragile, first-order sensitive to mu estimation error (MacLean/Thorp/Ziemba). (3) Current rule
(1.25% = 0.75%/60%) is a Vince-style worst-case identity; critique: worst case not knowable ex ante —
observed -60% is observational, NOT structural (shorts unbounded; buy-ins forced). Current size is
plausibly 1/30-1/60 Kelly; grid should span 1-15% notional with drawdown/CVaR constraints binding.

## Policy families for the W6 grid
- F0 fixed-fractional worst-case (control): f = R/|L_wc|; R in {0.375,0.75,1.5,3}%, L_wc in {60,100,150}%
- F1 fractional Kelly on bootstrap events + stressed no-ruin floor: c in {0.1..0.5}, R_stress in {-100,-150}%;
  MUST include estimation-error arm (mu +/-50%, score rank stability)
- F2 Risk-Constrained Kelly (Busseti-Ryu-Boyd, arXiv:1603.06183) ** best theoretical fit **:
  max E[ln(1+fR)] s.t. E[(1+fR)^-lambda] <= 1, lambda = ln(beta)/ln(alpha) — whole-path drawdown
  certificate P(W ever < alpha*W0) <= beta; grid alpha {0.90,0.85,0.80,0.70} x beta {0.01,0.05,0.10};
  shown to DOMINATE fractional Kelly.
- F3 drawdown-modulated (Grossman-Zhou 1993 / Hsieh-Barmish 1710.01503): f_t scales down with drawdown
  vs HWM; d_max {10,15,20,25}%; CAVEAT jumps pierce the floor (Klass-Nowicki 2005) — compose with F1 cap.
- F4 per-day CVaR heat + correlation penalty ** handles 2.3 events/day **: normal regime haircut
  1/(1+(k-1)rho); TAIL regime rho=1 (worst cases ADD) -> per-DAY worst-case budget, not per-event;
  current implicit day heat 1.5-2.25% at rho=1 — must be chosen explicitly.
Recommended composition: F2 (base size) x F4 (day heat) x optional F3 governor; F0/F1 as controls.

## Loss-model requirements for the simulator
- Disaster mixture: P ~ 0.1-0.5%/event of -100%..-200% (halt+gap+buy-in; Engelberg/Reed/Ringgenberg
  short-selling risk; JFQA short squeezes). Policies differing only in this cell can't be ranked by history.
- Stops don't truncate tails under jumps: loss-given-stop = mixture (prob q fill near s; else
  s + GPD exceedance) — size on E[L] and tail of L, not stop distance (QF 2019 stop-loss/gaps).

## Scoring metrics (all policies, same paths)
G = E[ln W]/T (median+mean) · CVaR 95/99 daily+monthly · MaxDD dist p50/p95/p99 (1y,3y) · CDaR(0.9) ·
P(DD>=10/20/30%/yr) · time-under-water + Triple-Penance benchmark · ruin prob vs kill level ·
estimation-error robustness (mu x{0.5,0.75,1.25} rank stability).
Selection rule: DON'T scalarize — maximize median growth s.t. hard constraints (e.g. P(DD>=20%/yr)<=5%,
CVaR99(day)<=H, G>0 under mu x0.5 stress).

Full citations: Bermin&Holm 2026 (s10203-025-00561-6) · Busseti/Ryu/Boyd 1603.06183 · MacLean/Thorp/
Ziemba (Good & Bad Kelly) · Grossman-Zhou 1993 · Klass-Nowicki 2005 · Hsieh-Barmish 1710.01503 ·
Chekhlov/Uryasev/Zabarankin CDaR (ssrn 544742) · Bailey/Lopez de Prado Triple Penance (ssrn 2201302) ·
Engelberg/Reed/Ringgenberg Short Selling Risk · JFQA Short Squeezes · QF 2019 stop-loss overnight gaps ·
vegapit multivariate Kelly · QuantPedia optimal-f critique · Cassini PB margin models.
