# Quantitative Finance 📐

#quant #quantitative-finance #mathematics #statistics #derivatives #risk #portfolio #algo-trading

> **Core Idea:** Apply mathematical, statistical, and computational methods to model, price, and manage financial instruments and portfolios. The language of modern finance is math.

---

## 🗺️ Map of Contents

- [[#Foundations — Mathematics]]
- [[#Foundations — Statistics & Probability]]
- [[#Foundations — Stochastic Calculus]]
- [[#Financial Mathematics]]
- [[#Asset Pricing Theory]]
- [[#Derivatives & Options]]
- [[#Options Greeks]]
- [[#Volatility]]
- [[#Fixed Income & Interest Rate Models]]
- [[#Portfolio Theory]]
- [[#Risk Management]]
- [[#Factor Models]]
- [[#Algorithmic & Systematic Trading]]
- [[#Market Microstructure]]
- [[#Time Series Analysis]]
- [[#Machine Learning in Finance]]
- [[#Alternative Data]]
- [[#Execution & Transaction Costs]]
- [[#Key Quant Roles]]
- [[#Essential Quant Stack]]
- [[#Core Textbooks & Resources]]

---

## Foundations — Mathematics

### Linear Algebra (Essential)
- **Vectors & Matrices** → portfolio returns, factor exposures, covariance matrices
- **Eigenvalues/Eigenvectors** → PCA on covariance matrix; risk decomposition
- **Matrix decomposition:**
  - Cholesky → generate correlated random variables (Monte Carlo)
  - SVD → dimensionality reduction, noise filtering
  - LU → solve linear systems efficiently
- **Dot product** → portfolio return: `r_p = w · r`
- **Quadratic form** → portfolio variance: `σ²_p = wᵀ Σ w`

### Calculus
- **Partial derivatives** → sensitivity analysis (Greeks, factor exposures)
- **Chain rule** → critical for Itô's lemma derivations
- **Taylor expansion** → option approximation; duration/convexity; delta-gamma
  - `f(x+Δx) ≈ f(x) + f'(x)Δx + ½f''(x)Δx²`
- **Lagrange multipliers** → constrained optimization (mean-variance)
- **Integration** → pricing via risk-neutral expectation: `V = e^{-rT} E^Q[payoff]`

### Optimization
| Method | Use Case |
|---|---|
| Quadratic Programming (QP) | Mean-variance portfolio optimization |
| Linear Programming (LP) | Index tracking, linear constraints |
| Convex Optimization | Risk parity, robust optimization |
| Gradient Descent / SGD | ML model training |
| Genetic Algorithms | Non-convex strategy parameter optimization |
| Sequential QP | Black-Litterman, general constrained optimization |

### Key Mathematical Results
- **Law of Large Numbers** → sample mean converges to population mean
- **Central Limit Theorem** → sum of iid variables → Normal distribution (underlies much of classical finance)
- **Jensen's Inequality** → `E[f(X)] ≥ f(E[X])` for convex f — critical for options pricing
- **Cauchy-Schwarz** → correlation bounded [-1, 1]

---

## Foundations — Statistics & Probability

### Probability Distributions in Finance
| Distribution | Use |
|---|---|
| Normal (Gaussian) | Returns (assumption), Z-scores, VaR |
| Log-Normal | Stock prices (GBM), option pricing |
| Student's t | Fat-tailed returns; better VaR estimation |
| Poisson | Jump processes; event arrival rates |
| Gamma / Inverse Gamma | Volatility modeling (variance processes) |
| Beta | Bounded variables; Bayesian priors |
| Pareto / Power Law | Extreme value theory; tail risk |
| Stable Distribution | Heavy tails; Mandelbrot's critique of normality |

### Moments & Statistics
- **Mean (μ)** → expected return
- **Variance (σ²)** → total risk
- **Skewness** → asymmetry; negative skew = left tail risk (options sellers love positive skew)
- **Kurtosis** → tail fatness; excess kurtosis >0 = leptokurtic = fat tails (real markets)
- **Covariance** → `Cov(X,Y) = E[(X-μ_X)(Y-μ_Y)]`
- **Correlation** → `ρ = Cov(X,Y) / (σ_X σ_Y)` bounded [-1,1]

### Hypothesis Testing
- **t-test** → is a strategy's mean return significantly different from zero?
- **p-value** → probability of observing result under null hypothesis; threshold typically 0.05
- **t-statistic** → `t = mean / (std / √n)`; need |t| > 2 for significance
- **Multiple testing problem** → running 100 backtests inflates false discovery rate; use Bonferroni / BH correction
- **Sharpe ratio t-stat** → `t = SR × √T`; need SR > 0.5 annualized for significance with 5 years data

### Regression
| Type | Use |
|---|---|
| OLS | Factor model estimation, beta calculation |
| WLS | Heteroskedastic errors |
| Ridge / Lasso | Regularized factor models; feature selection |
| Logistic | Classification (trade signal, default prediction) |
| Panel Data (FE/RE) | Cross-sectional + time series combined |

- **OLS assumptions (GAUSS-MARKOV):** Linearity, exogeneity, homoskedasticity, no autocorrelation, no perfect multicollinearity
- **R²** → fraction of variance explained; in finance often low (<10%) but that's fine
- **Heteroskedasticity** → variance of errors non-constant; use robust standard errors (White/Newey-West)

### Bayesian Statistics
- **Bayes' Theorem:** `P(θ|data) ∝ P(data|θ) × P(θ)`
- Prior → belief before data; Likelihood → data model; Posterior → updated belief
- **Black-Litterman model** is fundamentally Bayesian (blend market equilibrium with investor views)
- **Bayesian updating** → key for adaptive trading systems

---

## Foundations — Stochastic Calculus

### Brownian Motion (Wiener Process)
- `W(t)` is a Brownian motion if:
  1. `W(0) = 0`
  2. Independent increments
  3. `W(t) - W(s) ~ N(0, t-s)`
  4. Continuous paths
- **Key property:** `dW ~ N(0, dt)` → increments scale as `√dt` not `dt`

### Itô's Lemma
> The chain rule of stochastic calculus. **The most important formula in quant finance.**

If `dX = μ dt + σ dW`, then for `f(X,t)`:

```
df = (∂f/∂t + μ ∂f/∂X + ½σ² ∂²f/∂X²) dt + σ ∂f/∂X dW
```

The extra `½σ² ∂²f/∂X²` term is the **Itô correction** — arises because `(dW)² = dt` (not zero as in ordinary calculus)

### Geometric Brownian Motion (GBM)
```
dS = μS dt + σS dW
```
Solution: `S(t) = S(0) exp[(μ - σ²/2)t + σW(t)]`

- `μ` = drift (expected return)
- `σ` = volatility
- `σ²/2` correction → Itô term; without it prices are biased upward
- **Limitation:** Constant volatility, no jumps, log-normal prices — unrealistic but tractable

### Stochastic Differential Equations (SDEs)
| SDE | Model | Key Feature |
|---|---|---|
| `dS = μS dt + σS dW` | GBM | Stock prices |
| `dr = κ(θ-r)dt + σdW` | Vasicek | Mean-reverting rates |
| `dr = κ(θ-r)dt + σ√r dW` | CIR | Mean-reverting; non-negative rates |
| `dS = μS dt + σS dW + S dJ` | Jump-Diffusion (Merton) | Jumps + diffusion |
| `dv = κ(θ-v)dt + ξ√v dW_v` | Heston (variance) | Stochastic volatility |

### Martingales & Risk-Neutral Pricing
- **Martingale:** `E[X(t)|F_s] = X(s)` — no predictable drift; fair game
- **Girsanov's Theorem:** Change of measure from real-world `P` to risk-neutral `Q`
  - Under `Q`, all assets grow at risk-free rate `r`
  - Drift changes: `μ → r`; Brownian motion gets a drift term removed
- **Fundamental Theorem of Asset Pricing:**
  1. No arbitrage ↔ existence of equivalent martingale measure `Q`
  2. Completeness ↔ uniqueness of `Q`
- **Risk-Neutral Pricing:** `V(0) = e^{-rT} E^Q[V(T)]`

---

## Financial Mathematics

### Time Value of Money
- **Future Value:** `FV = PV(1+r)^n` (discrete) or `FV = PV·e^{rT}` (continuous)
- **Present Value:** `PV = FV / (1+r)^n`
- **NPV:** `Σ CF_t / (1+r)^t` — sum of discounted cash flows
- **IRR:** Rate that makes NPV = 0

### Compounding Conventions
| Convention | Formula |
|---|---|
| Annual | `(1+r)^n` |
| Semi-annual | `(1 + r/2)^{2n}` |
| Continuous | `e^{rT}` |
| Conversion | `r_c = ln(1 + r_d)` |

> Quant finance almost always uses **continuous compounding**

### Returns
- **Simple return:** `R = (P_t - P_{t-1}) / P_{t-1}`
- **Log return:** `r = ln(P_t / P_{t-1})`
- Log returns are additive over time: `r_{0→T} = Σ r_t`
- Log returns ≈ simple returns for small moves
- **Annualizing:** `μ_annual = μ_daily × 252`; `σ_annual = σ_daily × √252`

### Discounting & Yield Curves
- **Spot rate** `z(T)` → yield on zero-coupon bond maturing at T
- **Forward rate** `f(t,T)` → rate locked in today for borrowing from t to T
  - `f(t,T) = [z(T)·T - z(t)·t] / (T-t)`
- **Par rate** → coupon rate that prices bond at par
- **Bootstrapping** → derive spot curve from market instruments (swaps, bonds)
- **Discount factor** `D(T) = e^{-z(T)·T}`

---

## Asset Pricing Theory

### CAPM (Capital Asset Pricing Model)
```
E[R_i] = R_f + β_i (E[R_m] - R_f)
```
- `β_i = Cov(R_i, R_m) / Var(R_m)` — systematic risk exposure
- **Market premium** `E[R_m] - R_f` ≈ 5–7% historically
- **SML (Security Market Line):** Linear relationship between β and expected return
- **Alpha (α):** Excess return above CAPM prediction; what quants try to capture
- **Assumptions:** Mean-variance investors, frictionless markets, homogeneous expectations
- **Critique:** Empirically rejected; single factor insufficient; many anomalies

### Arbitrage Pricing Theory (APT)
```
E[R_i] = R_f + Σ β_{i,k} λ_k
```
- `λ_k` = risk premium for factor k; `β_{i,k}` = exposure to factor k
- No assumption on investor utility — just no-arbitrage
- Basis for multi-factor models (Fama-French, etc.)

### Efficient Market Hypothesis (EMH)
| Form | What's Priced In | Implication |
|---|---|---|
| Weak | All historical prices | Technical analysis useless |
| Semi-strong | All public information | Fundamental analysis useless |
| Strong | All information (incl. private) | Nothing works |

> Most evidence: markets are *mostly* weak-to-semi-strong efficient with anomalies at the edges — where quants operate

### No-Arbitrage Principle
- If two portfolios have identical future payoffs → they must have the same price today
- Violation = **arbitrage** = free money → instantly eliminated in theory
- Basis for: put-call parity, forward pricing, risk-neutral pricing

### Law of One Price
- Identical assets must trade at same price across all markets
- Violations = **statistical arbitrage** opportunities (pairs trading, etc.)

---

## Derivatives & Options

### Forwards & Futures
- **Forward price:** `F = S·e^{(r-q)T}` where `q` = dividend yield / convenience yield
- **Futures vs Forward:** Futures marked-to-market daily; forward settled at maturity
- **Basis:** `Basis = Spot - Futures` → converges to zero at expiry
- **Cost of carry model:** `F = S·e^{(r+u-y)T}` where `u` = storage cost, `y` = convenience yield

### Options Fundamentals
| Term | Definition |
|---|---|
| Call | Right to BUY at strike K |
| Put | Right to SELL at strike K |
| European | Exercise only at expiry |
| American | Exercise any time |
| ITM | In-the-money (call: S>K; put: S<K) |
| ATM | At-the-money (S≈K) |
| OTM | Out-of-the-money |
| Intrinsic Value | Max(S-K, 0) for call |
| Time Value | Option price − intrinsic value |

### Put-Call Parity (European, no dividends)
```
C - P = S - K·e^{-rT}
```
- Violation = arbitrage
- Implies: knowing call price → derive put price

### Black-Scholes-Merton (BSM) Model
**Assumptions:** GBM price process, constant vol, no dividends, continuous trading, no transaction costs, risk-free rate constant

**Call price:**
```
C = S·N(d₁) - K·e^{-rT}·N(d₂)

d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

**Put price:**
```
P = K·e^{-rT}·N(-d₂) - S·N(-d₁)
```

- `N(·)` = cumulative standard normal CDF
- `N(d₂)` = risk-neutral probability of expiring ITM
- `N(d₁)` = delta of the call

**BSM PDE:**
```
∂V/∂t + ½σ²S² ∂²V/∂S² + rS ∂V/∂S - rV = 0
```

### BSM Limitations & Extensions
| Issue | Extension |
|---|---|
| Constant volatility | Stochastic vol (Heston, SABR) |
| No jumps | Jump-diffusion (Merton, Kou) |
| European only | American options (binomial tree, PDE) |
| No dividends | Merton dividend adjustment |
| Single underlying | Multi-asset, correlation models |
| Implied vol ≠ constant | Local vol (Dupire), SVI parametrization |

### Binomial Tree (CRR Model)
- Discretize time into N steps; at each step: up `u = e^{σ√Δt}` or down `d = 1/u`
- Risk-neutral probability: `p = (e^{rΔt} - d) / (u - d)`
- Price by backward induction from terminal payoffs
- Converges to BSM as N → ∞
- Handles **American options** (check early exercise at each node)

### Monte Carlo for Options
```python
# European call via MC
S0, K, r, sigma, T, N = 100, 100, 0.05, 0.2, 1, 100000
Z = np.random.standard_normal(N)
ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
payoff = np.maximum(ST - K, 0)
price = np.exp(-r*T) * np.mean(payoff)
```
- **Variance reduction:** Antithetic variates, control variates, importance sampling
- Scales well to path-dependent and multi-asset options

### Exotic Options
| Type | Description |
|---|---|
| Asian | Payoff based on average price |
| Barrier | Knocked in/out if price hits barrier |
| Lookback | Payoff based on min/max over life |
| Binary/Digital | Fixed payoff if condition met |
| Compound | Option on an option |
| Basket | On portfolio of assets |
| Variance Swap | Pay realized var, receive fixed var |
| Volatility Swap | Pay realized vol, receive fixed vol |

---

## Options Greeks

> Sensitivities of option price to inputs. Core to hedging and risk management.

| Greek | Symbol | Measures | Formula (Call) |
|---|---|---|---|
| Delta | Δ | Price sensitivity to S | `N(d₁)` |
| Gamma | Γ | Delta sensitivity to S | `N'(d₁) / (Sσ√T)` |
| Vega | ν | Price sensitivity to σ | `S·N'(d₁)·√T` |
| Theta | Θ | Price sensitivity to time | Complex; negative for long options |
| Rho | ρ | Price sensitivity to r | `KT·e^{-rT}·N(d₂)` |
| Vanna | — | Delta sensitivity to σ | `∂Δ/∂σ` |
| Volga/Vomma | — | Vega sensitivity to σ | `∂ν/∂σ` |
| Charm | — | Delta sensitivity to time | `∂Δ/∂t` |

### Delta Hedging
- Hold Δ shares per option to create **delta-neutral** portfolio
- Hedge must be rebalanced continuously (in practice: discretely)
- **P&L of delta-hedged option** depends on realized vol vs implied vol:
  - Realized > implied vol → long gamma profits
  - Realized < implied vol → long gamma loses

### Gamma & the Gamma P&L
```
P&L ≈ ½ Γ (ΔS)² - Θ Δt
```
- Long gamma (long options) → profits from large moves; bleeds theta daily
- Short gamma (short options) → collects theta; hurt by large moves
- **Gamma scalping:** Delta hedge + collect from gamma; works when realized vol > implied vol

### Vega Risk
- Vega highest for ATM options; declines for deep ITM/OTM
- Long vega = benefits from vol increase; short vega = benefits from vol decrease
- **Vega hedging:** Trade variance swaps or other options to neutralize vega

---

## Volatility

### Historical vs Implied Volatility
- **Historical (realized) vol:** `σ = std(log returns) × √252`
- **Implied vol (IV):** Vol that makes BSM price = market price; extracted by inversion
- **Vol risk premium:** Implied vol typically > realized vol on average → options sellers collect premium

### Volatility Surface
- **Smile:** IV varies by strike — higher IV for OTM puts and calls vs ATM
- **Skew:** Negative skew — OTM puts more expensive than OTM calls (crash protection)
- **Term structure:** IV varies by expiry
- Together: **volatility surface** `σ(K,T)`

### Volatility Models
| Model | Key Feature | Use |
|---|---|---|
| BSM | Constant vol | Benchmark, simple |
| Local Vol (Dupire) | `σ(S,t)` — deterministic function | Fits surface exactly; path-dependent |
| Heston | Stochastic vol with mean reversion | Semi-analytic pricing; realistic dynamics |
| SABR | Stochastic vol; analytic approximation | Interest rate options; widely used |
| Rough Volatility (Bergomi) | Fractional Brownian motion vol | Best empirical fit to VIX dynamics |
| GARCH | Conditional heteroskedasticity from returns | Risk modeling, not pricing |

### GARCH(1,1)
```
σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}
```
- `ω` = long-run variance × `(1-α-β)`
- `α` = reaction to shocks (ARCH term)
- `β` = persistence (GARCH term)
- `α + β < 1` for stationarity; typical values α≈0.05, β≈0.90

### Realized Variance & VIX
- **Realized variance:** `RV_T = Σ r²_t` (sum of squared returns)
- **VIX:** Model-free implied vol from S&P 500 options for next 30 days (annualized)
  - `VIX ≈ √(E^Q[RV_{30d}]) × 100`
- **Variance swap payoff:** `N × (RV - K_var)` where `K_var` = strike set at fair variance

---

## Fixed Income & Interest Rate Models

### Bond Basics
- **Price:** `P = Σ C_t·e^{-z(t)·t} + F·e^{-z(T)·T}` (continuous; C = coupon, F = face)
- **YTM (Yield to Maturity):** Single rate that discounts all cash flows to market price
- **Duration (Macaulay):** `D = Σ [t × PV(CF_t)] / P` — weighted avg time to cash flows
- **Modified Duration:** `MD = D / (1 + y/m)` — % price change per 1% yield change
- **Convexity:** Second-order price sensitivity; accounts for curvature
- **Price change approx:** `ΔP/P ≈ -MD·Δy + ½·Convexity·(Δy)²`

### DV01 (Dollar Value of 1bp)
```
DV01 = -∂P/∂y × 0.0001 ≈ MD × P × 0.0001
```
Key risk metric for fixed income desks.

### Yield Curve Shapes & Theories
| Shape | Meaning |
|---|---|
| Normal (upward) | Higher rates for longer maturities |
| Inverted | Short rates > long rates; recession predictor |
| Flat | Transition or uncertainty |
| Humped | Short and long rates lower than medium |

**Theories:**
- **Expectations:** Long rates = expected future short rates
- **Liquidity Premium:** Long rates = expected short rates + liquidity premium
- **Market Segmentation:** Supply/demand at each maturity segment independently

### Short Rate Models
| Model | SDE | Properties |
|---|---|---|
| Vasicek | `dr = κ(θ-r)dt + σdW` | Analytic; can go negative |
| CIR | `dr = κ(θ-r)dt + σ√r dW` | Non-negative; analytic |
| Hull-White | `dr = (θ(t)-ar)dt + σdW` | Fits initial curve exactly |
| Black-Karasinski | `d ln r = (θ-a ln r)dt + σdW` | Log-normal; non-negative |

### HJM Framework
- Models entire **forward rate curve** evolution
- `df(t,T) = α(t,T)dt + σ(t,T)dW(t)`
- **No-arbitrage condition (HJM drift):** `α(t,T) = σ(t,T) ∫_t^T σ(t,u)du`
- LIBOR Market Model (LMM/BGM) is a special case — industry standard for interest rate derivatives

### Credit Risk
- **Default probability:** `PD` — likelihood of default in given horizon
- **Loss given default:** `LGD = 1 - Recovery Rate` (typically 40–60% recovery for senior)
- **Exposure at default:** `EAD`
- **Expected Loss:** `EL = PD × LGD × EAD`
- **Credit spread:** `s ≈ PD × LGD` (rough approximation)
- **CDS (Credit Default Swap):** Pays LGD on default in exchange for periodic premium (spread)
- **Structural models (Merton):** Default when firm value < debt; firm equity = call on assets
- **Reduced-form models:** Default as Poisson process with intensity `λ(t)`

---

## Portfolio Theory

### Mean-Variance Optimization (Markowitz)
**Objective:** Minimize portfolio variance for a given expected return

```
min  wᵀ Σ w
s.t. wᵀ μ = μ_target
     wᵀ 1 = 1
```

- **Efficient Frontier:** Set of portfolios with maximum return per unit risk
- **Global Minimum Variance (GMV):** Leftmost point on frontier
- **Tangency Portfolio:** Point where Sharpe ratio is maximized; the "optimal" risky portfolio
- **Capital Market Line (CML):** Line from risk-free rate tangent to efficient frontier
- **Two-Fund Separation:** All efficient portfolios = combination of risk-free + tangency portfolio

**Problems in practice:**
- Requires estimating μ and Σ → estimation error is huge
- Weights are extremely sensitive to inputs (error maximization)
- Tends to produce extreme, concentrated weights

### Covariance Matrix Estimation
| Method | Notes |
|---|---|
| Sample covariance | Noisy; singular if N > T |
| Shrinkage (Ledoit-Wolf) | Blend sample + structured estimator; robust |
| Factor model | `Σ = B F Bᵀ + D`; low-rank + diagonal |
| Exponential weighting | Recent data weighted more |
| Random Matrix Theory | Filter noise eigenvalues; keep signal |

### Black-Litterman Model
- Starts from **market equilibrium weights** (reverse optimization from CAPM)
- Blends with **investor views** using Bayesian updating
- `μ_BL = [(τΣ)^{-1} + PᵀΩ^{-1}P]^{-1} [(τΣ)^{-1}Π + PᵀΩ^{-1}Q]`
- `Π` = implied equilibrium returns; `P` = view matrix; `Q` = view returns; `Ω` = view uncertainty
- Produces more stable, diversified weights than pure Markowitz

### Risk Parity
- Allocate so each asset contributes **equally to total portfolio risk**
- Asset `i` risk contribution: `RC_i = w_i × (Σw)_i / σ_p`
- Equal risk contribution: `RC_i = RC_j ∀ i,j`
- Typically overweights bonds vs equities → often levered to match return target
- **Bridgewater's All Weather** is a famous risk parity fund

### Performance Metrics
| Metric | Formula | Good Value |
|---|---|---|
| Sharpe Ratio | `(R_p - R_f) / σ_p` | >1.0 (annualized) |
| Sortino Ratio | `(R_p - R_f) / σ_downside` | >2.0 |
| Calmar Ratio | `CAGR / Max Drawdown` | >1.0 |
| Information Ratio | `α / σ(α)` (vs benchmark) | >0.5 |
| Treynor Ratio | `(R_p - R_f) / β` | Higher = better |
| Max Drawdown | Peak-to-trough % loss | Context-dependent |
| Hit Rate | % of positive periods | >50% for trend; lower ok for high R:R |
| Profit Factor | Gross profit / Gross loss | >1.5 |

---

## Risk Management

### Value at Risk (VaR)
> **VaR(α, T):** Maximum loss not exceeded with probability α over horizon T

**Methods:**
| Method | Approach | Pros/Cons |
|---|---|---|
| Historical Simulation | Sort past P&L; take α percentile | Non-parametric; slow to adapt |
| Parametric (Normal) | `VaR = μ - z_α × σ` | Fast; assumes normality (bad tails) |
| Monte Carlo | Simulate P&L distribution | Flexible; computationally heavy |

**Limitations of VaR:**
- Not sub-additive (can violate diversification)
- Tells you nothing about **severity** beyond threshold
- Assumptions often wrong in crisis

### Expected Shortfall (CVaR / ES)
```
ES_α = E[Loss | Loss > VaR_α]
```
- Average loss in the worst α% of scenarios
- **Sub-additive** → proper coherent risk measure
- Preferred by Basel III/IV for regulatory capital
- More sensitive to tail shape than VaR

### Coherent Risk Measures
A risk measure `ρ` is coherent if it satisfies:
1. **Monotonicity:** If X ≤ Y always, ρ(X) ≥ ρ(Y)
2. **Sub-additivity:** `ρ(X+Y) ≤ ρ(X) + ρ(Y)` (diversification reduces risk)
3. **Positive homogeneity:** `ρ(λX) = λρ(X)`
4. **Translation invariance:** `ρ(X+m) = ρ(X) - m`

VaR fails sub-additivity. ES satisfies all four.

### Greeks-Based Risk (Options Books)
- **Delta risk:** `P&L ≈ Δ × ΔS`
- **Gamma risk:** `P&L ≈ ½Γ × (ΔS)²`
- **Vega risk:** `P&L ≈ ν × Δσ`
- **Theta decay:** Daily time decay of options book
- **DV01:** Interest rate sensitivity

### Stress Testing & Scenario Analysis
- **Historical scenarios:** 1987 crash, 2008 GFC, 2020 COVID
- **Hypothetical scenarios:** Rates +200bp, equity -30%, vol ×3
- **Reverse stress test:** What scenario would cause firm failure?
- Regulatory requirement: Basel, Solvency II, FRTB

### Liquidity Risk
- **Bid-ask spread** → transaction cost; wider in stressed markets
- **Market impact** → your own trades move prices
- **Funding liquidity** → ability to roll short-term funding; key in 2008
- **Liquidity-adjusted VaR (LVaR):** Add cost of unwinding position over liquidation horizon

---

## Factor Models

### Fama-French 3-Factor Model
```
R_i - R_f = α + β_m(R_m - R_f) + β_SMB·SMB + β_HML·HML + ε
```
- **SMB (Small Minus Big):** Small-cap premium
- **HML (High Minus Low):** Value premium (high book-to-market)
- Explains ~90% of portfolio return variation vs ~70% for CAPM

### Fama-French 5-Factor Model
Adds to 3-factor:
- **RMW (Robust Minus Weak):** Profitability premium
- **CMA (Conservative Minus Aggressive):** Investment premium

### Carhart 4-Factor Model
3-Factor + **MOM (Momentum):** Past 12-month winners minus losers

### Main Factor Premia
| Factor | Signal | Premium Source |
|---|---|---|
| Market (Beta) | Market exposure | Equity risk premium |
| Size | Small cap | Liquidity, neglect |
| Value | Low P/B, P/E | Distress risk, behavioral |
| Momentum | 12-1 month return | Underreaction, trend |
| Low Volatility | Low beta/vol | Leverage aversion, behavioral |
| Profitability | High ROE/gross profit | Quality premium |
| Investment | Low asset growth | Overinvestment behavioral |
| Carry | High yield | Risk premium |
| Reversal | Short-term mean reversion | Microstructure, liquidity |

### Statistical Factor Models (PCA)
1. Compute return covariance matrix `Σ`
2. Eigendecompose: `Σ = V Λ Vᵀ`
3. Top k eigenvectors = statistical factors
4. Factor returns: `F = V_k · R`
5. Factor loadings: `B = Cov(R, F) / Var(F)`

- First PC typically = market factor
- Subsequent PCs = sector/style rotations
- Used for **risk decomposition** and **hedging**

### Barra Risk Models
- Commercial factor models (MSCI Barra, Axioma, Northfield)
- Industry standard for equity risk: world equity factors + country + industry + style factors
- `Σ = B F Bᵀ + Δ` (factor covariance + specific risk)

---

## Algorithmic & Systematic Trading

### Strategy Types
| Type | Holding Period | Signal | Edge Source |
|---|---|---|---|
| Statistical Arbitrage | Minutes–days | Mean reversion, cointegration | Market inefficiency |
| Momentum/Trend Following | Days–months | Price trend | Behavioral underreaction |
| Mean Reversion | Minutes–days | Deviation from equilibrium | Overreaction, liquidity |
| Market Making | Seconds–minutes | Bid-ask spread | Liquidity provision |
| Pairs Trading | Days–weeks | Cointegrated spread | Structural relationship |
| Event-Driven | Hours–days | News, earnings, M&A | Information processing |
| High-Frequency | Microseconds | Order flow, microstructure | Latency, co-location |
| Factor Investing | Monthly rebalance | Multi-factor | Risk premia harvesting |

### Signal Construction
- **Z-score:** `z = (x - μ) / σ` → standardize any signal for comparison
- **Information Coefficient (IC):** Correlation between signal and forward return; IC > 0.05 is useful
- **IC_IR (ICIR):** `IC / std(IC)` → consistency of signal; want ICIR > 0.5
- **Signal decay:** How fast predictive power fades — critical for execution timing
- **Orthogonalization:** Remove factor exposures from signal to isolate alpha

### Pairs Trading
1. Find two cointegrated stocks (Engle-Granger or Johansen test)
2. Estimate spread: `s_t = P_A - β·P_B`
3. Z-score the spread: `z_t = (s_t - μ_s) / σ_s`
4. Entry: `|z| > 2`; Exit: `|z| < 0.5`; Stop: `|z| > 3`
5. Hedge ratio `β` from OLS regression

### Cointegration Tests
- **Augmented Dickey-Fuller (ADF):** Test if spread is stationary (unit root test)
  - H0: unit root (non-stationary) → reject for cointegration
- **Engle-Granger:** Two-step; OLS then ADF on residuals
- **Johansen:** Multivariate; tests rank of cointegration space; handles >2 assets

### Backtesting
**Core principle:** Simulate strategy on historical data to estimate future performance

**Critical pitfalls:**
| Pitfall | Description | Solution |
|---|---|---|
| Look-ahead bias | Using future data | Strict time indexing; point-in-time data |
| Survivorship bias | Only using surviving stocks | Include delisted stocks |
| Overfitting | Too many parameters | Out-of-sample test; walk-forward |
| Transaction costs | Ignoring slippage | Model realistic costs (see execution) |
| Data snooping | Multiple testing on same data | Hold-out set; Bonferroni correction |
| Regime change | Past ≠ future | Stress test across regimes |

**Walk-forward validation:**
- Train on T years, test on 1 year, roll forward
- More realistic than single train/test split
- Reduces overfitting

**Key backtest metrics:**
- Sharpe ratio (annualized > 1.0 to be interesting)
- Max drawdown (context-dependent; typically < 20%)
- Calmar ratio > 1.0
- Trade count (need enough trades for statistical significance)
- T-stat of returns > 2.0

### Kelly Criterion
```
f* = (bp - q) / b
```
- `f*` = fraction of capital to bet
- `b` = net odds (profit per $1 risked)
- `p` = probability of win
- `q = 1 - p`
- **Full Kelly** maximizes long-run geometric growth but volatile
- **Half Kelly** (f*/2) preferred in practice — much less volatile, ~75% of geometric growth

### Alpha Decay
- `IR = IC × √BR` (Grinold's Fundamental Law of Active Management)
- `IC` = Information Coefficient per bet
- `BR` = Breadth (number of independent bets per year)
- To double IR: quadruple breadth OR double IC (IC harder to improve)

---

## Market Microstructure

### Order Types
| Order | Description |
|---|---|
| Market Order | Immediate fill at best available price |
| Limit Order | Fill only at specified price or better |
| Stop Order | Becomes market order when price hit |
| Stop-Limit | Becomes limit order when price hit |
| Iceberg | Show only portion of size |
| TWAP | Execute evenly over time |
| VWAP | Execute proportional to volume profile |
| Implementation Shortfall | Minimize decision price vs fill price |

### Bid-Ask Spread Decomposition
```
Spread = Adverse Selection Cost + Inventory Cost + Order Processing Cost
```
- **Adverse selection:** Market maker loses to informed traders
- **Inventory:** Cost of holding unwanted position
- **Processing:** Fixed costs of running market making operation

### Market Impact
- **Temporary impact:** Price moves against you during execution; reverts after
- **Permanent impact:** Reveals information; prices don't revert
- **Square-root law:** `ΔP ≈ σ × √(Q/ADV)` where Q = order size, ADV = avg daily volume
- **Almgren-Chriss model:** Optimal execution balancing market impact vs timing risk

### Order Flow & Informed Trading
- **PIN (Probability of Informed Trading):** Estimates fraction of order flow from informed traders
- **VPIN (Volume-Synchronized PIN):** Real-time measure of informed trading
- **Order flow imbalance:** `OFI = Buy volume - Sell volume`; predicts short-term price moves
- **Toxic flow:** Order flow from informed traders; adverse for market makers

### Limit Order Book (LOB)
- Best bid (highest buy) and best ask (lowest sell) form the **inside spread**
- **Depth:** Volume available at each price level
- **Order book imbalance:** `(bid_vol - ask_vol) / (bid_vol + ask_vol)` → short-term price predictor
- **Queue position:** Priority matters in HFT; first-in-first-out at each price level

---

## Time Series Analysis

### Stationarity
- **Stationary:** Mean, variance, autocovariance don't change over time
- **Required for:** Most statistical tests and models
- **Unit root:** Non-stationary series with stochastic trend
- **ADF test:** Tests H0: unit root; reject → stationary
- **KPSS test:** Tests H0: stationary; reject → unit root
- **Differencing:** `ΔX_t = X_t - X_{t-1}` makes many series stationary

### ARIMA Models
- **AR(p):** `X_t = Σ φ_i X_{t-i} + ε_t` — current value depends on past values
- **MA(q):** `X_t = Σ θ_i ε_{t-i} + ε_t` — current value depends on past errors
- **ARMA(p,q):** Combination
- **ARIMA(p,d,q):** Difference d times first, then ARMA
- **ACF/PACF plots:** Identify p and q from autocorrelation structure

### Volatility Models
- **ARCH(q):** `σ²_t = ω + Σ α_i ε²_{t-i}`
- **GARCH(p,q):** Adds lagged variance terms; GARCH(1,1) most common
- **EGARCH:** Asymmetric; captures leverage effect (vol spikes more on down moves)
- **GJR-GARCH:** Also asymmetric; simpler than EGARCH
- **HAR-RV:** Heterogeneous AR on realized variance; excellent forecasting

### Vector Autoregression (VAR)
```
X_t = A₁X_{t-1} + A₂X_{t-2} + ... + AₚX_{t-p} + ε_t
```
- Multivariate; captures lead-lag relationships between multiple time series
- **Granger causality:** X Granger-causes Y if X lags improve forecast of Y
- **Impulse Response Functions (IRF):** How system responds to shock in one variable
- **VECM (Vector Error Correction Model):** VAR for cointegrated systems

### Kalman Filter
- Recursive Bayesian filter for state estimation in linear dynamical systems
- **Applications in finance:** Time-varying beta, dynamic factor models, pairs trading hedge ratio estimation
- **State equation:** `X_t = F·X_{t-1} + w_t`
- **Observation equation:** `Y_t = H·X_t + v_t`
- Two steps: Predict → Update (using new observation)

---

## Machine Learning in Finance

### Supervised Learning Applications
| Task | Algorithm | Target |
|---|---|---|
| Return prediction | Ridge, Lasso, Random Forest, XGBoost | Forward returns |
| Classification | Logistic, SVM, Neural Nets | Up/Down signal |
| Volatility forecast | LSTM, Transformer | Realized vol |
| Default prediction | Gradient Boosting | Default probability |
| NLP: sentiment | BERT, FinBERT | News/earnings sentiment |

### Unsupervised Learning
- **PCA** → factor extraction, noise reduction, regime identification
- **K-means / Hierarchical clustering** → asset grouping, regime detection
- **Autoencoders** → anomaly detection, feature compression
- **HMM (Hidden Markov Model)** → regime switching (bull/bear/volatile)

### Feature Engineering for Finance
- **Momentum features:** 1M, 3M, 6M, 12M returns; MACD; RSI
- **Reversal features:** 1W, 1M returns (short-term mean reversion)
- **Fundamental:** P/E, P/B, ROE, EPS growth
- **Technical:** Volume, ATR, Bollinger Band position
- **Macro:** VIX, yield curve slope, credit spreads
- **Cross-sectional ranks:** Always rank features cross-sectionally; prevents look-ahead from normalization

### Pitfalls Specific to Finance ML
- **Non-stationarity:** Features and labels change distribution over time
- **Low SNR:** Signal-to-noise ratio extremely low; R² of 1–3% can be very valuable
- **Rare events:** Crashes, defaults; class imbalance severe
- **Leakage:** Subtle future information bleeding into features
- **Regime change:** Model trained in bull market fails in bear
- **Short history:** 20 years of daily data = 5,000 samples; tiny for ML

### Deep Learning in Finance
- **LSTM/GRU:** Sequence models for time series; prone to overfitting
- **Transformer:** Attention mechanism; strong for NLP (earnings calls, news)
- **CNN:** Pattern recognition in price charts; microstructure data
- **Reinforcement Learning:** Portfolio allocation, execution optimization
- **Graph Neural Networks:** Interbank networks, supply chain, correlation networks

### Cross-Validation for Finance
- **Never use random CV** → data leakage from future to past
- **Purged K-Fold:** Remove overlapping samples around split points
- **Embargo:** Additional gap after purge to prevent leakage from forward-looking labels
- **Walk-forward:** Rolling train/test; most realistic

---

## Alternative Data

### Categories
| Type | Examples | Edge |
|---|---|---|
| Web/App data | Web traffic, app downloads, search trends | Consumer demand signals |
| Satellite/Geo | Parking lot counts, ship tracking, crop yield | Physical world activity |
| Credit card | Transaction data, consumer spend by merchant | Revenue estimation |
| Social media | Twitter/Reddit sentiment, mentions | Retail sentiment |
| Job postings | LinkedIn, Indeed data | Hiring signals, R&D focus |
| Patent filings | New IP activity | Innovation pipeline |
| Earnings calls | NLP on transcripts | Management tone, guidance |
| Supply chain | Supplier relationships, logistics | Industry interconnection |
| Weather | Climate data | Commodity, retail, energy |

### Processing Pipeline
1. **Acquisition** → API, scrape, data vendor
2. **Cleaning** → missing data, outliers, formatting
3. **Normalization** → cross-sectional standardization
4. **Signal extraction** → ML model or rule-based
5. **Decay analysis** → how long signal lasts
6. **Combination** → blend with other signals
7. **Backtest** → proper out-of-sample with costs

---

## Execution & Transaction Costs

### Components of Transaction Costs
| Component | Description |
|---|---|
| Commission | Broker fee; near zero for retail; basis points for institutions |
| Bid-ask spread | Half-spread per trade |
| Market impact | Temporary + permanent price move from your order |
| Timing risk | Price moves while executing large order |
| Opportunity cost | Not executing at target price |

### Market Impact Models
- **Linear:** `ΔP = η × (Q/ADV)`
- **Square-root:** `ΔP = σ × γ × √(Q/ADV)` — most empirically supported
- **Power law:** `ΔP = α × (Q/ADV)^δ` where δ ≈ 0.5–0.6

### Execution Algorithms
| Algo | Strategy | Best For |
|---|---|---|
| TWAP | Equal size over time | Low info decay |
| VWAP | Match volume profile | Minimize market impact |
| IS (Implementation Shortfall) | Balance impact vs risk | High alpha decay |
| POV (Participation of Volume) | % of market volume | Flexible, size-adaptive |
| Sniper | Aggressive opportunistic | Urgent, small orders |

### Slippage Estimation
```
Total cost ≈ 0.5 × spread + market_impact + commission
```
For backtesting: model as `cost_bps = a + b × (turnover/ADV)^0.5`

---

## Key Quant Roles

| Role | Primary Focus | Key Skills |
|---|---|---|
| Quant Researcher | Alpha generation, factor research | Statistics, ML, Python, finance domain |
| Quant Developer (QD) | Infrastructure, execution systems | C++, Python, low latency, distributed systems |
| Quant Trader | Strategy execution, P&L management | Markets intuition, statistics, fast thinking |
| Strats (Goldman style) | Desk support, pricing models | Derivatives math, C++/Python, client facing |
| Risk Quant | VaR, stress testing, capital models | Risk measures, regulation, statistics |
| Structurer | Design/price exotic products | Deep derivatives math, client needs |
| Data Scientist | Alternative data, ML signal research | ML, Python, data engineering |
| Portfolio Manager (Quant PM) | Capital allocation, portfolio construction | Portfolio theory, statistics, leadership |

---

## Essential Quant Stack

### Programming Languages
| Language | Use |
|---|---|
| Python | Research, ML, backtesting (dominant) |
| C++ | HFT, low-latency execution, pricing libraries |
| R | Statistics, econometrics (legacy; less common now) |
| Julia | Numerical computing; growing in quant research |
| SQL | Data querying; every quant needs this |
| MATLAB | Legacy finance; still used in some banks |

### Python Libraries
| Library | Purpose |
|---|---|
| NumPy | Numerical arrays, linear algebra |
| Pandas | Time series, dataframes |
| SciPy | Statistics, optimization, integration |
| Statsmodels | Econometrics, time series (ARIMA, OLS) |
| Scikit-learn | ML models, cross-validation |
| PyTorch / TensorFlow | Deep learning |
| QuantLib | Derivatives pricing, yield curves |
| Zipline / Backtrader | Backtesting frameworks |
| CVXPY | Convex optimization (portfolio construction) |
| Matplotlib / Plotly | Visualization |

### Data Sources
| Source | Data Type | Cost |
|---|---|---|
| Yahoo Finance / yfinance | OHLCV, fundamentals | Free |
| Quandl / Nasdaq Data Link | Macro, fundamentals, alt data | Freemium |
| Bloomberg Terminal | Everything | ~$25K/year |
| Refinitiv (LSEG) | Everything | ~$20K/year |
| CRSP | Historical US equity | Academic |
| Compustat | Fundamentals | Academic/paid |
| IEX Cloud | Real-time + historical | Affordable |
| Alpha Vantage | OHLCV, indicators | Freemium |
| Polygon.io | Real-time, options | Moderate |

---

## Core Textbooks & Resources

### Mathematics & Foundations
- **Stochastic Calculus for Finance I & II** — Shreve *(the bible for stochastic calc)*
- **Options, Futures & Other Derivatives** — Hull *(industry standard derivatives reference)*
- **Mathematics for Finance** — Capiński & Zastawniak *(gentler intro)*

### Derivatives & Volatility
- **The Concepts and Practice of Mathematical Finance** — Joshi
- **Volatility Surface** — Gatheral *(vol modeling bible)*
- **Dynamic Hedging** — Taleb *(practitioner's Greeks)*

### Portfolio & Factor
- **Active Portfolio Management** — Grinold & Kahn *(fundamental law, alpha theory)*
- **Expected Returns** — Ilmanen *(comprehensive factor premia survey)*
- **Quantitative Equity Portfolio Management** — Chincarini & Kim

### Statistical & ML
- **Advances in Financial Machine Learning** — López de Prado *(essential ML for finance)*
- **Algorithmic Trading** — Chan *(practical strategy development)*
- **Quantitative Trading** — Chan *(pairs trading, backtesting)*
- **Time Series Analysis** — Hamilton *(econometrics reference)*

### Risk
- **Risk Management and Financial Institutions** — Hull
- **Quantitative Risk Management** — McNeil, Frey & Embrechts *(heavy; comprehensive)*

### Culture & Mindset
- **The Man Who Solved the Market** — Zuckerman *(Renaissance Technologies)*
- **My Life as a Quant** — Derman *(career memoir; excellent)*
- **When Genius Failed** — Lowenstein *(LTCM collapse; risk lessons)*
- **A Man for All Markets** — Thorp *(Kelly criterion, first quant hedge fund)*

---

## Related Notes

- [[Swing Trading MOC]]
- [[Options Strategies]]
- [[Python for Finance]]
- [[Statistical Arbitrage]]
- [[Portfolio Construction]]
- [[Derivatives Pricing]]
- [[Risk Models]]
- [[Backtesting Framework]]
- [[Factor Research]]
- [[Market Microstructure Notes]]
- [[Stochastic Calculus Notes]]
- [[Time Series Models]]
- [[ML Feature Engineering Finance]]

---

*For educational and research purposes. All models are wrong; some are useful.*
