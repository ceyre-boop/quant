"""
V6.0 Fractal Corridor Validation Engine
Architecture: Tri-Cycle Analysis (24h Pulse, Whole-Chart Corridor, Biological Sentinel)
Goal: Identify geometric self-similarity that maps 'Comet Cycles' (Long-term structural shifts).
"""

import pandas as pd
import numpy as np
import logging
import os
import scipy.stats as stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_corridor_geometry(price_series, window=500):
    """
    Measures the 'Fractal Geometry' of a price segment.
    Calculates the R-squared against a Log-Linear trendline (The Corridor).
    High R-squared = The market is 'locked' in a structural corridor.
    """
    def _get_geometry(ts):
        if len(ts) < 50: return 0
        t = np.arange(len(ts))
        log_p = np.log(ts)
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, log_p)
        return r_value**2 # The 'Linearity' or 'Corridor Integrity'
    
    return price_series.rolling(window).apply(_get_geometry, raw=True)

def run_fractal_validation():
    dataset_path = "data/router_labels_v5.csv"
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return

    logger.info("============================================")
    logger.info("FRACTAL CORRIDORS VALIDATION (V6.0)")
    logger.info("============================================")
    
    # 1. LOAD THE INTEGRATED REALITY DATA
    df = pd.read_csv(dataset_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. CALCULATE GEOMETRIC CORRIDORS (WHOLE-CHART PERSPECTIVE)
    # We look for periods where price is 'locked' in a geometric fractal.
    df['corridor_integrity'] = calculate_corridor_geometry(df['hurst'], window=100) # Hurst-Corridor
    
    # 3. IDENTIFY 'ONE-OFF' BIOLOGICAL ANOMALIES
    # An anomaly is any movement that stays 4+ standard deviations from the 100-year corridor.
    df['anomaly_score'] = (df['zscore'] - df['zscore'].rolling(500).mean()) / df['zscore'].rolling(500).std()
    
    # 4. CROSS-CORRELATION: Does Geometry predict our Specialists (Label 1/2)?
    # We want to know: Are momentum windows (Label 1) occurring within structural corridors?
    mom_mask = df['label'] == 1
    normal_integrity = df[~mom_mask]['corridor_integrity'].mean()
    specialist_integrity = df[mom_mask]['corridor_integrity'].mean()
    
    logger.info(f"Avg Corridor Integrity (Normal): {normal_integrity:.4f}")
    logger.info(f"Avg Corridor Integrity (Momentum Windows): {specialist_integrity:.4f}")
    
    # LIFT CALCULATION
    if specialist_integrity > normal_integrity:
        lift = (specialist_integrity / normal_integrity) - 1
        logger.info(f"GEOMETRIC LIFT: {lift:.2%} (Structural Confluence Found)")
    else:
        logger.info("No Geometric Confluence identified in the 24h noise.")

    # 5. DETECT 'ASTROPHAGE' EVENTS (One-offs that reset the regime)
    astrophage_events = df[abs(df['anomaly_score']) > 4.5]
    logger.info(f"Biological Anomalies (One-offs) Detected: {len(astrophage_events)}")
    
    if not astrophage_events.empty:
        logger.info("Latest Anomaly Timestamp: " + str(astrophage_events['timestamp'].iloc[-1]))

    logger.info("============================================")

if __name__ == "__main__":
    run_fractal_validation()
