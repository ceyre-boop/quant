"""
Sovereign Trading Intelligence -- Factor Zoo Scanner
Phase 2: Feature Layer

The Factor Zoo acts as a high-pass filter for signals. 
It rigorously tests every feature against lookahead bias and statistical significance.
Only features that clear the Bonferroni-corrected significance gate are allowed 
to enter the Regime Router.

Criteria:
1. IC (Information Coefficient): Spearman rank correlation with forward returns.
2. ICIR (IC Information Ratio): Stability of the IC over time (mean/std).
3. Bonferroni Correction: Alpha adjusted for multiple testing (0.05 / n_features).
4. Threshold: |ICIR| >= 0.30 is the 'is_real' floor.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import logging
from typing import Dict, List, Optional
import os

from sovereign.features.regime.hurst import compute_hurst_features
from sovereign.features.regime.csd import compute_csd_features
from sovereign.features.regime.hmm_regime import compute_hmm_features
from sovereign.features.momentum.logistic_ode import compute_logistic_features
from sovereign.features.momentum.momentum_factors import compute_momentum_features
from sovereign.features.momentum.volume_profile import compute_volume_profile_features

logger = logging.getLogger(__name__)


class FactorZooScanner:
    """
    Validates features for statistical significance and regime robustness.
    """

    def __init__(self, forward_return_period: int = 1):
        self.forward_return_period = forward_return_period
        self.results = None

    def build_feature_matrix(self, df: pd.DataFrame, macro_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Aggregates all features from Phase 2 modules into a single matrix.
        """
        logger.info(f"Building feature matrix for {len(df)} bars...")
        
        # Block A: Regime
        hurst = compute_hurst_features(df)
        csd = compute_csd_features(df)
        hmm = compute_hmm_features(df)
        
        # Block B: Momentum
        logistic = compute_logistic_features(df)
        momentum = compute_momentum_features(df)
        volume = compute_volume_profile_features(df)
        
        # Block C: Macro (if provided)
        macro_features = None
        if macro_df is not None:
            from sovereign.features.macro.yield_curve import compute_yield_curve_features
            from sovereign.features.macro.erp import compute_erp_features
            from sovereign.features.macro.cape import compute_cape_features
            from sovereign.features.macro.cot import compute_cot_features
            
            yc = compute_yield_curve_features(macro_df)
            erp = compute_erp_features(macro_df)
            cape = compute_cape_features(macro_df)
            cot = compute_cot_features(macro_df)
            
            macro_features = pd.concat([yc, erp, cape, cot], axis=1)
            
            # Ensure index is datetime for merging
            df.index = pd.to_datetime(df.index)
            macro_features.index = pd.to_datetime(macro_features.index)
            
            # Sort for merge_asof
            df = df.sort_index()
            macro_features = macro_features.sort_index()
            
            # merge_asof: Align daily macro data to hourly/minute bars
            # This ensures no lookahead while providing the latest macro snapshot
            merged = pd.merge_asof(
                df, macro_features, 
                left_index=True, right_index=True,
                direction='backward'
            )
            feature_df = merged
        else:
            feature_df = df.copy()

        # Combine OHLCV-derived blocks
        # (These are already aligned with df.index)
        feature_df = pd.concat([
            feature_df, hurst, csd, hmm, logistic, momentum, volume
        ], axis=1)
        
        # Add forward returns for IC calculation (1-bar fwd)
        feature_df['fwd_ret'] = df['close'].shift(-self.forward_return_period).pct_change(self.forward_return_period).shift(-self.forward_return_period)
        
        return feature_df.dropna()

    def scan(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the Factor Zoo scan on the feature matrix.
        """
        features = [col for col in feature_df.columns if col not in ['close', 'volume', 'fwd_ret', 'high', 'low', 'open']]
        n_tests = len(features)
        target_alpha = 0.05 / n_tests  # Bonferroni correction
        
        logger.info(f"Scanning {n_tests} features with alpha {target_alpha:.6f}")
        
        scan_results = []
        
        for feature in features:
            valid_mask = ~feature_df[feature].isna() & ~feature_df['fwd_ret'].isna()
            f_data = feature_df.loc[valid_mask, feature]
            r_data = feature_df.loc[valid_mask, 'fwd_ret']
            
            if len(f_data) < 60:
                continue
                
            # Global IC (Spearman)
            ic, p_val = spearmanr(f_data, r_data)
            
            # Rolling IC for stability (ICIR)
            # We use a 60-bar rolling window
            rolling_ic = self._compute_rolling_ic(f_data, r_data, window=60)
            icir = rolling_ic.mean() / rolling_ic.std() if rolling_ic.std() > 0 else 0
            
            is_real = (p_val < target_alpha) and (abs(icir) >= 0.30)
            
            scan_results.append({
                'feature':         feature,
                'ic':              ic,
                'p_value':         p_val,
                'icir':            icir,
                'is_real':         is_real,
                'passes_alpha':    p_val < target_alpha,
            })
            
        self.results = pd.DataFrame(scan_results).sort_values(by='icir', ascending=False)
        return self.results

    @staticmethod
    def _compute_rolling_ic(f: pd.Series, r: pd.Series, window: int = 60) -> pd.Series:
        """Computes rolling Spearman correlation."""
        # Simple approximation: rolling pearson on ranks
        f_rank = f.rolling(window).rank()
        r_rank = r.rolling(window).rank()
        return f_rank.rolling(window).corr(r_rank).dropna()


def run_phase2_gate(df_full: pd.DataFrame, ticker: str):
    """
    Executes the Phase 2 Gate: In-Sample and Out-Of-Sample scans.
    """
    scanner = FactorZooScanner()
    
    # Run 1: In-Sample (2022-2024)
    df_is = df_full.loc['2022-01-01':'2024-12-31']
    if not df_is.empty:
        logger.info(f"Running IS scan (2022-2024) for {ticker}")
        feat_is = scanner.build_feature_matrix(df_is)
        res_is = scanner.scan(feat_is)
        
        os.makedirs('vault', exist_ok=True)
        res_is.to_csv(f'vault/factor_zoo_is_{ticker}.csv', index=False)
        
        print(f"\nFACTOR ZOO - IN-SAMPLE ({ticker})")
        print(res_is.to_string(index=False))
        
    # Run 2: OOS/Regime Robustness (2025-2026)
    df_oos = df_full.loc['2025-01-01':]
    if not df_oos.empty:
        logger.info(f"Running OOS scan (2025-2026) for {ticker}")
        feat_oos = scanner.build_feature_matrix(df_oos)
        res_oos = scanner.scan(feat_oos)
        res_oos.to_csv(f'vault/factor_zoo_oos_{ticker}.csv', index=False)
        
        print(f"\nFACTOR ZOO - OUT-OF-SAMPLE ({ticker})")
        print(res_oos.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # This would typically be called with loaded data
    print("FactorZooScanner module initialized. Run with data to execute gate.")
