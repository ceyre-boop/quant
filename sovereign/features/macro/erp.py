"""
Sovereign Trading Intelligence -- Equity Risk Premium (ERP)
Phase 2: Feature Layer

ERP measures the 'extra' return expected from equities over risk-free bonds.
Structural shifts in ERP are primary drivers of regime changes.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def equity_risk_premium(macro_data: pd.DataFrame) -> pd.Series:
    """
    ERP = Earnings Yield - Real 10Yr Yield.
    Earnings Yield approximated as 1/PE (target: S&P 500).
    Real Yield = T10Y - 10Y Inflation Breakeven.
    """
    # Real Yield
    if 'dgs10' in macro_data.columns and 't10yie' in macro_data.columns:
        real_yield = macro_data['dgs10'] - macro_data['t10yie']
    else:
        real_yield = pd.Series(np.nan, index=macro_data.index)
        
    # Earnings Yield (S&P 500)
    # If sp500_pe is not in FRED, we use a proxy or a rolling estimate.
    # For the purpose of this implementation, we look for 'sp500_pe' column.
    if 'sp500_pe' in macro_data.columns:
        earnings_yield = 100.0 / macro_data['sp500_pe']
    else:
        # Fallback: Historical consensus value (approx 5% for 20x PE)
        earnings_yield = pd.Series(5.0, index=macro_data.index)
        
    erp = earnings_yield - real_yield
    return erp


def erp_zscore(erp: pd.Series, window: int = 252 * 5) -> pd.Series:
    """
    Z-score vs 30-year history. 
    (Using rolling window as proxy if full history not available).
    """
    mean = erp.rolling(window, min_periods=252).mean()
    std = erp.rolling(window, min_periods=252).std()
    return (erp - mean) / std.replace(0, np.nan)


def compute_erp_features(macro_data: pd.DataFrame) -> pd.DataFrame:
    """Compute ERP and its z-score."""
    erp = equity_risk_premium(macro_data)
    z = erp_zscore(erp)
    
    result = pd.DataFrame({
        'equity_risk_premium': erp,
        'erp_zscore':          z,
    }, index=macro_data.index)
    
    return result
