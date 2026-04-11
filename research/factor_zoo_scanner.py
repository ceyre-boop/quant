"""
Pillar 13: Factor Zoo Scanner (Institutional Prototype)
Implements Multiple Comparison Correction (Bonferroni) to filter alpha ghosts.

Factors are ranked by ICIR (Consistency) and p-value.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import logging
import yfinance as yf
from scipy.optimize import curve_fit
from training.engine_v4 import calculate_hurst

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- TIER 3 EXPERIMENTAL COMPONENTS ---

class AccumulationODE:
    """Logistic Growth ODE inflection point discovery."""
    def logistic_model(self, t, L, k, t0):
        return L / (1 + np.exp(-k * (t - t0)))
    
    def calculate_inflection(self, price_series, window=50):
        def _get_k(ts):
            try:
                t = np.arange(len(ts))
                p0 = [ts.max(), 1, len(ts)/2] # Initial guess
                popt, _ = curve_fit(self.logistic_model, t, ts, p0=p0, maxfev=1000)
                return popt[1] # Growth rate k
            except:
                return 0
        return price_series.rolling(window).apply(_get_k, raw=True)

# --- THE ZOO ENGINE ---

class FactorZooScanner:
    def __init__(self, df, forward_return_bars=10):
        self.df = df
        self.forward_bars = forward_return_bars
        # Target: Forward Log Returns
        self.target = np.log(df['close'].shift(-forward_return_bars) / df['close'])
        self.results = []

    def scan(self, factor_map, n_total_tests):
        adjusted_alpha = 0.05 / n_total_tests
        logger.info(f"Zoo Scan Running. Bonferroni Adjusted Alpha: {adjusted_alpha:.6f}")
        
        for name, calc_fn in factor_map.items():
            try:
                values = calc_fn(self.df)
                res = self._test_factor(name, values)
                res['p_pass'] = res['p_value'] < adjusted_alpha
                self.results.append(res)
            except Exception as e:
                logger.error(f"Factor {name} failed: {e}")

        return self._report()

    def _test_factor(self, name, values):
        # Information Coefficient (Spearman Rank Correlation)
        valid = pd.concat([values, self.target], axis=1).dropna()
        ic, p_val = stats.spearmanr(valid.iloc[:,0], valid.iloc[:,1])
        
        # ICIR (IC divided by standard deviation of rolling IC)
        # Using 60-day rolling window for consistency check
        rolling_ic = values.rolling(60).corr(self.target)
        icir = rolling_ic.mean() / rolling_ic.std() if rolling_ic.std() > 0 else 0
        
        return {
            'factor': name,
            'ic': ic,
            'icir': icir,
            'p_value': p_val,
            'n_obs': len(valid)
        }

    def _report(self):
        report_df = pd.DataFrame(self.results).sort_values('icir', ascending=False)
        return report_df

# --- THE ZOO DEFINITION ---

def get_factor_universe():
    ode = AccumulationODE()
    return {
        'T1_momentum_10': lambda df: df['close'].pct_change(10),
        'T3_hurst_100': lambda df: calculate_hurst(df['close'], 100),
        'T3_gap_entropy': lambda df: (df['open'] - df['close'].shift(1)).abs() / (df['high'] - df['low']).rolling(14).mean(),
        'T3_dist_to_mean': lambda df: (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std(),
        'T3_logistic_k': lambda df: ode.calculate_inflection(df['close'], 50),
        'T3_vol_zscore': lambda df: (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std(),
    }

if __name__ == "__main__":
    # RUN SCAN ON 2025 ERA (THE COLLAPSE DATA)
    # yfinance limit: 1h data only available for last 730 days
    df = yf.download("SPY", start="2024-05-01", end="2026-04-10", interval="1h")

    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    
    universe = get_factor_universe()
    scanner = FactorZooScanner(df)
    results = scanner.scan(universe, len(universe))
    
    print("\n=== FACTOR ZOO SCAN RESULTS (SPY 1H - OOS) ===")
    print(results[['factor', 'ic', 'icir', 'p_value', 'p_pass']])
