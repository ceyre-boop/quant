"""
Sovereign Trading Intelligence — Factor Zoo Scanner
Phase 0 Diagnostic (corrected)

Three fixes applied per operator review:

1. Multi-horizon forward returns (fast/slow/macro natural timescales)
2. Benjamini-Hochberg FDR correction (Bonferroni over-penalizes correlated
   financial features — hurst_short/long, hmm_state/csd_score are all correlated)
3. Feature-horizon mapping: a feature only needs to pass at ONE of its natural
   horizons, not arbitrary 1-bar. Tide gauges don't predict the next wave.

Feature groups and their natural forward-return windows:
  FAST:   volume_zscore, rsi, logistic_k, adx_zscore, roc_5   → 1, 3, 5 bars
  SLOW:   hurst_short, hurst_long, csd_score, hmm_state, etc  → 10, 20, 40 bars
  MACRO:  yield_curve_slope, erp, cape_zscore, cot_zscore      → 20, 60 bars
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

# ─── Feature → horizon mapping ───────────────────────────────────────────────
# Grouped by the timescale each feature actually measures.
# A feature passes if it achieves ICIR ≥ threshold at ANY of its natural horizons.

FEATURE_HORIZONS = {
    "fast":  [1, 3, 5],    # 1–5 bars  → Specialists (daily entry)
    "slow":  [10, 20, 40], # 2–8 weeks → Regime Router
    "macro": [20, 60],     # 1–3 months → Petroulas Gate
}

FEATURE_GROUPS = {
    # fast
    "volume_zscore":           "fast",
    "ofi_velocity":            "fast",
    "rsi":                     "fast",
    "roc_5":                   "fast",
    "logistic_k":              "fast",
    "logistic_acceleration":   "fast",
    "adx_zscore":              "fast",
    # slow — regime
    "hurst_short":             "slow",
    "hurst_long":              "slow",
    "hurst_velocity":          "slow",
    "csd_score":               "slow",
    "csd_ar1":                 "slow",
    "csd_variance_vel":        "slow",
    "csd_recovery":            "slow",
    "hmm_state":               "slow",
    "hmm_transition_prob":     "slow",
    "bars_since_regime_change":"slow",
    "adx":                     "slow",
    "jt_momentum":             "slow",
    "atr":                     "fast",
    # macro
    "volume_entropy":          "fast",  # actually a fast information measure
    "yield_curve_slope":       "macro",
    "yield_curve_velocity":    "macro",
    "erp":                     "macro",
    "cape_zscore":             "macro",
    "cot_zscore":              "macro",
}

ICIR_THRESHOLD = 0.30   # minimum ICIR to be considered "real"
BH_ALPHA       = 0.10   # Benjamini-Hochberg FDR level (more lenient than 0.05 for finance)


class FactorZooScanner:
    """
    Validates features for statistical significance and horizon-adjusted
    regime robustness. Uses Benjamini-Hochberg FDR correction.
    """

    def __init__(self, forward_return_period: int = 1):
        # kept for backward compat; multi-horizon scanning replaces the single period
        self.forward_return_period = forward_return_period
        self.results = None

    # ─── Public API ──────────────────────────────────────────────────────────

    def build_feature_matrix(self, df: pd.DataFrame,
                             macro_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Aggregates all features from Phase 2 modules into a single matrix.
        Does NOT add forward returns here — those are added per-horizon in scan().
        """
        logger.info(f"Building feature matrix for {len(df)} bars...")

        # Block A: Regime (slow timescale)
        hurst   = compute_hurst_features(df)
        csd     = compute_csd_features(df)
        hmm     = compute_hmm_features(df)

        # Block B: Momentum (fast timescale)
        logistic = compute_logistic_features(df)
        momentum = compute_momentum_features(df)
        volume   = compute_volume_profile_features(df)

        # Block C: Macro (optional)
        if macro_df is not None:
            from sovereign.features.macro.yield_curve import compute_yield_curve_features
            from sovereign.features.macro.erp import compute_erp_features
            from sovereign.features.macro.cape import compute_cape_features
            from sovereign.features.macro.cot import compute_cot_features

            yc   = compute_yield_curve_features(macro_df)
            erp  = compute_erp_features(macro_df)
            cape = compute_cape_features(macro_df)
            cot  = compute_cot_features(macro_df)

            macro_features = pd.concat([yc, erp, cape, cot], axis=1)
            df              = df.sort_index()
            macro_features  = macro_features.sort_index()
            merged = pd.merge_asof(df, macro_features,
                                   left_index=True, right_index=True,
                                   direction='backward')
            feature_df = merged
        else:
            feature_df = df.copy()

        feature_df = pd.concat(
            [feature_df, hurst, csd, hmm, logistic, momentum, volume], axis=1
        )

        # Drop only rows that are NaN across ALL feature columns (not forward return)
        feature_cols = [c for c in feature_df.columns
                        if c not in ('open', 'high', 'low', 'close', 'volume',
                                     'vwap', 'trade_count', 'fwd_ret')]
        n_before = len(feature_df)
        # Keep row if at least one feature is non-NaN (dropna will be per-horizon)
        feature_df = feature_df.dropna(subset=feature_cols, how='all')
        n_after = len(feature_df)
        logger.info(f"Feature matrix: {n_before} → {n_after} rows "
                    f"(removed {n_before - n_after} all-NaN rows)")

        return feature_df

    def scan(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-horizon scan with Benjamini-Hochberg correction.

        For each feature:
          1. Determine its group (fast/slow/macro) → get its natural horizons
          2. Run Spearman IC + rolling ICIR at each horizon
          3. Collect (feature, best_icir, best_horizon, bh_adjusted_p, is_real)
        
        Returns a DataFrame sorted by best_icir descending.
        """
        close = feature_df['close'] if 'close' in feature_df.columns else None
        if close is None:
            logger.error("feature_df must contain 'close' column for forward return calculation")
            return pd.DataFrame(columns=['feature', 'group', 'best_horizon',
                                         'ic', 'p_value', 'icir', 'bh_p_value',
                                         'is_real', 'passes_bh'])

        feature_cols = [c for c in feature_df.columns
                        if c not in ('open', 'high', 'low', 'close', 'volume',
                                     'vwap', 'trade_count')]
        logger.info(f"Scanning {len(feature_cols)} features across horizons")

        # ─── Step 1: Collect per-feature, per-horizon IC / ICIR ─────────────
        def _scan_feature(feat):
            group = FEATURE_GROUPS.get(feat, "fast")
            horizons = FEATURE_HORIZONS[group]
            local_results = []
            
            for h in horizons:
                fwd = close.shift(-h).pct_change(h)
                valid = feature_df[feat].notna() & fwd.notna()
                if valid.sum() < 60: continue
                
                f_data = feature_df.loc[valid, feat]
                r_data = fwd[valid]
                
                ic, p_val = spearmanr(f_data, r_data)
                rolling_ic = self._compute_rolling_ic(f_data, r_data, window=60)
                if len(rolling_ic) < 10: continue
                
                icir = float(rolling_ic.mean() / rolling_ic.std()) if rolling_ic.std() > 0 else 0.0
                local_results.append({
                    "feature": feat, "group": group, "horizon": h,
                    "ic": float(ic), "p_value": float(p_val), "icir": float(icir)
                })
            return local_results

        from joblib import Parallel, delayed
        results_list = Parallel(n_jobs=-1)(delayed(_scan_feature)(feat) for feat in feature_cols)
        raw_results = [item for sublist in results_list for item in sublist]

        if not raw_results:

            logger.warning("No feature/horizon combinations produced valid IC estimates")
            return pd.DataFrame(columns=['feature', 'group', 'best_horizon',
                                         'ic', 'p_value', 'icir', 'bh_p_value',
                                         'is_real', 'passes_bh'])

        raw_df = pd.DataFrame(raw_results)

        # ─── Step 2: Per feature, take the BEST horizon (max |ICIR|) ─────────
        # Note: apply(lambda x: x.abs().idxmax()) returns the index of the max absolute value in each group
        best_idx_series = raw_df.groupby("feature")["icir"].apply(lambda x: x.abs().idxmax())
        best_rows = raw_df.loc[best_idx_series.values].copy()
        best_rows = best_rows.rename(columns={"horizon": "best_horizon"})

        # ─── Step 3: Benjamini-Hochberg FDR correction on best p-values ──────
        try:
            from statsmodels.stats.multitest import multipletests
            p_vals = best_rows["p_value"].values
            rejected, p_adj, _, _ = multipletests(p_vals, alpha=BH_ALPHA, method="fdr_bh")
            best_rows = best_rows.copy()
            best_rows["bh_p_value"] = p_adj
            best_rows["passes_bh"]  = rejected
        except ImportError:
            logger.warning("statsmodels not available — falling back to Bonferroni for BH step")
            n = len(best_rows)
            threshold = BH_ALPHA / max(n, 1)
            best_rows = best_rows.copy()
            best_rows["bh_p_value"] = best_rows["p_value"] * n
            best_rows["passes_bh"]  = best_rows["p_value"] < threshold

        # ─── Step 4: Apply ICIR gate ─────────────────────────────────────────
        best_rows["is_real"] = (
            best_rows["passes_bh"] &
            (best_rows["icir"].abs() >= ICIR_THRESHOLD)
        )

        self.results = best_rows.sort_values("icir", key=abs, ascending=False).reset_index(drop=True)

        # Log summary
        n_real = int(self.results["is_real"].sum())
        n_bh   = int(self.results["passes_bh"].sum())
        logger.info(f"Scan complete: {len(self.results)} features | "
                    f"{n_bh} pass BH | {n_real} robust (BH + ICIR≥{ICIR_THRESHOLD})")
        if n_real > 0:
            logger.info("Robust features:")
            for _, row in self.results[self.results["is_real"]].iterrows():
                logger.info(f"  {row['feature']:30s} "
                            f"group={row['group']:6s} "
                            f"horizon={row['best_horizon']:3d}  "
                            f"ICIR={row['icir']:+.3f}  "
                            f"BH_p={row['bh_p_value']:.4f}")
        else:
            # Still show the top-5 near-misses for diagnosis
            logger.info("Zero robust. Top-5 by |ICIR| (near-misses):")
            for _, row in self.results.head(5).iterrows():
                logger.info(f"  {row['feature']:30s} "
                            f"group={row['group']:6s} "
                            f"horizon={row['best_horizon']:3d}  "
                            f"ICIR={row['icir']:+.3f}  "
                            f"p={row['p_value']:.4f}  "
                            f"BH_p={row['bh_p_value']:.4f}")

        return self.results

    # ─── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_rolling_ic(f: pd.Series, r: pd.Series, window: int = 60) -> pd.Series:
        """Rolling Spearman via rank-then-pearson approximation."""
        f_rank = f.rolling(window).rank()
        r_rank = r.rolling(window).rank()
        return f_rank.rolling(window).corr(r_rank).dropna()


# ─── Standalone test function ─────────────────────────────────────────────────

def run_phase0_gate(df_full: pd.DataFrame, ticker: str):
    """
    Executes the Phase 0 Gate with multi-horizon scan.
    Saves IS and OOS results to vault/{ticker}_*.csv.
    """
    scanner = FactorZooScanner()

    for period, start, end, label in [
        ("IS",  "2022-01-01", "2024-12-31", "is"),
        ("OOS", "2025-01-01", None,          "oos"),
    ]:
        df_slice = df_full.loc[start:end] if end else df_full.loc[start:]
        if df_slice.empty:
            logger.warning(f"No data for {period} period on {ticker}")
            continue

        logger.info(f"Running {period} scan ({start}–{end or 'present'}) for {ticker}")
        feat   = scanner.build_feature_matrix(df_slice)
        result = scanner.scan(feat)

        os.makedirs("vault", exist_ok=True)
        result.to_csv(f"vault/factor_zoo_{label}_{ticker}.csv", index=False)
        print(f"\nFACTOR ZOO {period} — {ticker}")
        print(result.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("FactorZooScanner initialized. Run with data to execute gate.")
