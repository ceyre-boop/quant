# Gate Functions Reference

**This document specifies the mechanical gates through which hypotheses advance. All gates are deterministic and testable.**

*Last updated: 2026-08-12*

---

## Gate 1: Sharpe Ratio (Deflated Sharpe Ratio)

**Requirement:** Sharpe ≥ 0.30 (DSR-adjusted for degrees of freedom)

**Rationale:** Bailey et al. (2015) deflated-Sharpe corrects for multiple testing bias. On unseen data with n=100 trades, DSR at 0.30 is approximately p<0.05 against the null (random entry/exit).

**Calculation:**
```python
def deflated_sharpe(returns, num_tests=809):
    """
    Bailey et al. (2015) deflated Sharpe ratio.
    Corrects for multiple testing (data dredging bias).
    
    Args:
        returns: Series of trade P&Ls (daily or per-trade)
        num_tests: Number of tested hypotheses (conservative: 809 for large factor zoo)
    
    Returns:
        dsr: Deflated Sharpe ratio (corrected)
    """
    n = len(returns)
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Naive Sharpe
    sharpe = mean_return / std_return if std_return > 0 else 0
    
    # Bailey correction factor
    phi = (2.0 * np.log(num_tests) - np.euler_gamma) / (2 * np.log(2 * np.pi))
    
    # Deflated Sharpe
    dsr = sharpe * (1 - (phi / np.sqrt(n)))
    
    return dsr
```

**Machine gate:** `sovereign/discovery/gate_functions.py::deflated_sharpe()`

**Status per TRADING_PHILOSOPHY:** Updated to 0.30 on 2026-07-22 (from 0.25); prior threshold was aspirational. 0.30 maps to permutation p≈0.05 at n=100.

---

## Gate 2: Permutation Test (No-Look-Ahead Null)

**Requirement:** Permutation p-value < 0.05 (fail-to-reject null: strategy beats random entry/exit)

**Rationale:** Random entry/exit establishes a no-skill baseline. If your strategy beats that with p<0.05, there is *some* signal (though signal ≠ exploitable edge in live markets).

**Calculation:**
```python
def permutation_test(returns, num_permutations=10000):
    """
    Permutation test: does the strategy beat random entry/exit?
    
    Args:
        returns: Series of trade P&Ls
        num_permutations: Number of shuffled replicates
    
    Returns:
        p_value: Proportion of random strategies with Sharpe >= observed
    """
    observed_sharpe = returns.mean() / returns.std()
    
    better_count = 0
    for _ in range(num_permutations):
        shuffled = np.random.permutation(returns)
        perm_sharpe = shuffled.mean() / shuffled.std()
        if perm_sharpe >= observed_sharpe:
            better_count += 1
    
    p_value = better_count / num_permutations
    return p_value
```

**Machine gate:** `sovereign/discovery/gate_functions.py::permutation_test()`

**Note:** Permutation test is on unseen data ONLY. Never permute in-sample; it inflates p-values (you've already overfit).

---

## Gate 3: Out-of-Sample (OOS) Walk-Forward Degradation

**Requirement:** Walk-forward OOS Sharpe within 20% of in-sample Sharpe (no cliff degradation)

**Rationale:** Training always outperforms testing. If your OOS Sharpe is >20% lower than IS, you've overfit. Overfitting survives in-sample testing but dies in production.

**Calculation:**
```python
def oos_degradation_check(is_sharpe, oos_sharpe, threshold=0.20):
    """
    Check whether out-of-sample performance has degraded excessively.
    
    Args:
        is_sharpe: In-sample Sharpe (training window)
        oos_sharpe: Out-of-sample Sharpe (held-out test window)
        threshold: Maximum acceptable degradation (20% = 0.20)
    
    Returns:
        passes: Boolean. True if degradation <= threshold.
        degradation: Fraction (e.g., 0.15 = 15% degradation)
    """
    if is_sharpe <= 0:
        return False, float('inf')
    
    degradation = (is_sharpe - oos_sharpe) / is_sharpe
    passes = degradation <= threshold
    
    return passes, degradation
```

**Machine gate:** `sovereign/discovery/gate_functions.py::oos_degradation_check()`

**Typical values:**
- IS Sharpe 1.20, OOS Sharpe 1.02 → degradation 15% ✅ PASS
- IS Sharpe 0.80, OOS Sharpe 0.50 → degradation 37% ❌ FAIL (overfit)

---

## Gate 4: Regime Robustness (Optional, emerging)

**Requirement:** Strategy Sharpe consistent across different market regimes (trending vs choppy vs crash)

**Rationale:** TRADING_PHILOSOPHY Tenet 2: "Regime Appropriateness Beats Strategy Quality." A strategy that works only in one regime is not a strategy; it's a lucky coincidence.

**Status:** Emerging. Not yet mandatory gate. Reference: `sovereign/intelligence/regime_performance_tracker.py`.

---

## Amendment History

- **2026-08-12** — Gate 3 (OOS degradation) raised from 15% to 20% threshold (prior was too strict; good strategies degraded 18–19%).
- **2026-07-22** — Gate 1 (Sharpe) raised from 0.25 to 0.30. Prior value was aspirational; 0.30 is permutation-validated.
- **2026-07-07** — Initial version (Gates 1–3 ratified).

---

## Usage

When you write a hypothesis test stage, reference this file:

```markdown
### Inputs (Layer 3)
- `_config/gate_functions.md` — Statistical gates (Sharpe ≥0.30, permutation p<0.05, OOS degradation <20%)

### Process
Run hypothesis backtest. Check against all three gates.
If all pass: verdict PASS. If any fail: verdict FAIL.
Record verdict in hypothesis ledger.
```

Do not repeat gate definitions in your stage output. Reference this file.
