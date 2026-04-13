"""
HYP-008 Statistical Audit - T-Test and Effect Size Calculation
Tests the significance of return differences between 10-year trend resonant vs dissonant momentum signals.
Constraint: p < 0.05 and Cohen's d > 0.2 required for architectural integration.
"""

import pandas as pd
import numpy as np
import os
import scipy.stats as stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_hyp_008_validation():
    # 1. DATA ALIGNMENT (REPRODUCING RESONANCE DFS)
    labels_path = "data/router_labels_v5.csv"
    planetary_path = "data/planetary/whole_chart_corridors.csv"
    
    if not os.path.exists(labels_path) or not os.path.exists(planetary_path):
        logger.error("Data missing.")
        return

    pulse_df = pd.read_csv(labels_path)
    pulse_df['timestamp'] = pd.to_datetime(pulse_df['timestamp'])
    planetary_df = pd.read_csv(planetary_path)
    planetary_df['Date'] = pd.to_datetime(planetary_df['Date'])
    pulse_df['date_only'] = pd.to_datetime(pulse_df['timestamp'].dt.date)

    all_aligned = []
    for s in pulse_df['ticker'].unique():
        s_pulse = pulse_df[pulse_df['ticker'] == s].copy()
        s_plane = planetary_df[planetary_df['ticker'] == s].copy()
        s_plane['era_slope'] = s_plane['era_slope'].shift(1)
        merged = pd.merge(s_pulse, s_plane[['Date', 'era_slope']], 
                          left_on='date_only', right_on='Date', how='left')
        all_aligned.append(merged)
        
    df = pd.concat(all_aligned).dropna()
    df['resonance'] = np.sign(df['zscore']) == np.sign(df['era_slope'])
    
    # 2. EXTRACT RETURN DISTRIBUTIONS
    mom_mask = df['label'] == 1
    resonant_returns = df[mom_mask & df['resonance']]['mom_ev'].values
    dissonant_returns = df[mom_mask & ~df['resonance']]['mom_ev'].values
    
    # 3. STATISTICAL TESTS
    # Welch's T-test (does not assume equal variance)
    t_stat, p_value = stats.ttest_ind(resonant_returns, dissonant_returns, equal_var=False)
    
    # Cohen's d (Effect size)
    n1, n2 = len(resonant_returns), len(dissonant_returns)
    var1, var2 = np.var(resonant_returns, ddof=1), np.var(dissonant_returns, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (np.mean(dissonant_returns) - np.mean(resonant_returns)) / pooled_std
    
    # 4. RESULTS
    print("\n" + "="*40)
    print("HYP-008 STATISTICAL AUDIT RESULTS")
    print("="*40)
    print(f"Resonant Mean: {np.mean(resonant_returns):.4%}")
    print(f"Dissonant Mean: {np.mean(dissonant_returns):.4%}")
    print(f"Difference: {np.mean(dissonant_returns) - np.mean(resonant_returns):.4%}")
    print("-" * 40)
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Significant at 95%: {p_value < 0.05}")
    print(f"Cohen's d: {cohens_d:.3f}")
    
    if p_value < 0.05 and abs(cohens_d) > 0.2:
        print("\nCONCLUSION: HYP-008 IS STATISTICALLY SIGNIFICANT.")
        print("ACTION: Register HYP-008 and proceed to OOS testing.")
    else:
        print("\nCONCLUSION: HYP-008 IS INSIGNIFICANT (NOISE).")
        print("ACTION: Archive finding. Do not build.")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_hyp_008_validation()
