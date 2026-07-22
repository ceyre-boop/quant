# Template B — Minimum Viable Income System
**Colin Eyre · Alta Investments · scaffolded 2026-07-22 · STATUS: DRAFT FOR COLIN**
*Machine-computable numbers pre-filled (sources cited). Every `⬜ COLIN:` is a decision only you can make. Fill those and the monthly number falls out.*

---

## 1. The Monthly Number

**Target: ⬜ COLIN: $______ /month net**

Reference math (`FINANCIAL_FREEDOM_STEPS.md`): $10k/mo = $120k/yr ⇒ needs **$400k @ 30%** or **$800k @ 15%**. The bottleneck is capital, not edge quality. A smaller honest interim target makes the chain below start paying sooner.

## 2. The Shortest Chain — Signal → Payout, per proven edge

### Edge 1 — Carry v015 (PROVEN, live paper · OOS Sharpe 1.25, ~5%/yr net after regime haircuts)
```
FOMC/rate-differential signal → conviction sizing → OANDA position → 60d hold / exit manager → P&L
```
Payout paths:
- **Own capital:** $ capital × ~5%/yr net ÷ 12 = $/mo (see worksheet).
- **Funded account:** FTMO 2-Step Swing / FundedNext Stellar are the only structural fits (static DD, no time limit — `project_carry_propfirm_fit`). ⚠ Headroom is coin-flip: levering to the 10% target implies ~9.2% DD vs a 10% cap. 80% split on $100k funded @ carry's honest pace ≈ **$300–$1,600/mo** (Lever 3 range, `FINANCIAL_FREEDOM_STEPS.md`) — but P(pass) analysis (`project_prop_funnel_simulator`) says favorable-window entries only.

### Edge 2 — The Undertow HYP-093 (VALID, sized, in shadow · ~15%/yr gross pre-friction, W6 F2+F3)
```
gap ≥100% by 10:30 ET → locate check → short @ F2+F3 size (4.0%/3.4% notional) → close same day → P&L
```
Payout path: **own IBKR margin account only** (no prop firm funds micro-cap shorts — RESULTS_REPORT Gate 2). Base case: **$10k → +$1.5k/yr · $25k → +$3.75k/yr** ($ math only interesting above $25k). Friction may cut 20–30%. First live trade: **90–180 days** (gates: W7 forward shadow ~250 events, IBKR+HTB setup, TICK-024 cost cascade).

### Edge 3 — Petrules Gate (SPEC'd, prereg v1.1 locked · target 15–25%/yr, calibration TBD)
Not income yet — first possible live trade ≥10 months out. Feeds the number in year 2+, not the MVI.

## 3. Requirements With Your Name On Them

| # | Decision | Context | Your call |
|---|---|---|---|
| 1 | Undertow account size + open IBKR margin + HTB locate subscription | Only realistic locate broker at retail scale | ⬜ COLIN: $______ at ______ |
| 2 | Prop-firm go/no-go (carry) | Coin-flip headroom; decision was explicitly deferred to you | ⬜ COLIN: GO / NO-GO / WAIT |
| 3 | Petrules ~$130 data buy (ThetaData STANDARD, 2012+ history) | Regime gate currently says STAND_ASIDE — wait is aligned | ⬜ COLIN: buy now / wait |
| 4 | Lever-2 injection rate | $/mo from income into the trading base — the timeline compressor | ⬜ COLIN: $______ /mo |
| 5 | First-live-trade authorization (per edge) | The machine never touches real money without your explicit go | ⬜ COLIN: standing rule or per-trade |

## 4. The Worksheet — fill capital, read $/month

| Capital deployed | Carry ~5%/yr net | Undertow ~15%/yr gross* | Combined $/mo (approx) |
|---|---|---|---|
| $10k | $42/mo | $125/mo | ~$165 |
| $25k | $104/mo | $313/mo | ~$415 |
| $50k | $208/mo | $625/mo | ~$830 |
| $100k | $417/mo | $1,250/mo | ~$1,665 |
| $400k | $1,667/mo | $5,000/mo | ~$6,665 |
| **⬜ COLIN: $______** | | | **= the number** |

\* Undertow gross, pre-friction (−20–30%), capacity untested above ~$50k (locate availability unmodeled — RESULTS_REPORT borrow caveat). Funded-account income (Lever 3, $300–$1,600/mo) stacks on top if #2 is GO.

**The honest reading:** at today's proven edges, the MVI is capital-bound: ~$830/mo at $50k, ~$1,665/mo at $100k. The $10k/mo number requires $400k+ (Lever 2 + compounding + Lever 3), or a confirmed Edge 3. There is no configuration of the current edges that gets $10k/mo from $50k — anything claiming otherwise is ruin-math.

---
*Sources: RESULTS_REPORT_2026-07-21.md · FINANCIAL_FREEDOM_STEPS.md · CLAUDE.md live-state (carry v015) · memory: carry-propfirm-fit, prop-funnel-simulator, W6 build. No invented figures; all rates are the conservative documented ones.*
