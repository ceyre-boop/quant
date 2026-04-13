"""
V6.3 Resonance Engine - The Tri-Cycle Confluence
Goal: Correlate the 24h Pulse (Hourly) with the Planetary Corridors (10y Daily).
Hypothesis: Momentum IC increases by 20%+ when in Resonance with the Planetary Slope.
"""

import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_resonance_audit():
    # 1. LOAD THE TWO SCALES
    labels_path = "data/router_labels_v5.csv" # Hourly Pulse
    planetary_path = "data/planetary/whole_chart_corridors.csv" # 10y Daily
    
    if not os.path.exists(labels_path) or not os.path.exists(planetary_path):
        logger.error("Missing scale data. Need both Pulse and Planetary files.")
        return

    logger.info("============================================")
    logger.info("V6.3 RESONANCE AUDIT (Scales Alignment)")
    logger.info("============================================")
    
    pulse_df = pd.read_csv(labels_path)
    pulse_df['timestamp'] = pd.to_datetime(pulse_df['timestamp'])
    
    planetary_df = pd.read_csv(planetary_path)
    planetary_df['Date'] = pd.to_datetime(planetary_df['Date'])
    
    # 2. ALIGNMENT: Merge Planetary Era Data into the Hourly Pulse
    # Convert pulse timestamps to naïve dates for alignment with yfinance daily data
    pulse_df['date_only'] = pd.to_datetime(pulse_df['timestamp'].dt.date)

    
    # Merge for each ticker separately to avoid cross-contamination
    all_aligned = []
    for s in pulse_df['ticker'].unique():
        s_pulse = pulse_df[pulse_df['ticker'] == s].copy()
        s_plane = planetary_df[planetary_df['ticker'] == s].copy()
        
        # Shift planetary era data by 1 day to ensure NO LOOKAHEAD
        # We only know yesterday's planetary corridor integrity today.
        s_plane['era_slope'] = s_plane['era_slope'].shift(1)
        s_plane['era_integrity'] = s_plane['era_integrity'].shift(1)
        
        merged = pd.merge(s_pulse, s_plane[['Date', 'era_slope', 'era_integrity', 'vol_zscore']], 
                          left_on='date_only', right_on='Date', how='left')
        all_aligned.append(merged)
        
    df = pd.concat(all_aligned).dropna()
    
    # 3. CALCULATE RESONANCE
    # Resonance = 1 when the Hourly Z-score direction matches the 10-year Era Slope direction
    df['resonance'] = np.sign(df['zscore']) == np.sign(df['era_slope'])
    
    # 4. MEASURE THE SPECIALIST ACCURACY UNDER RESONANCE
    # Strategy: Momentum (Label 1)
    mom_mask = df['label'] == 1
    total_mom = mom_mask.sum()
    resonant_mom = (mom_mask & df['resonance']).sum()
    
    # Outcome Check: Did resonance lead to better specialist outcomes? (Approximated by Label density)
    logger.info(f"Total Momentum Windows Identified: {total_mom}")
    logger.info(f"Momentum Windows in Resonance: {resonant_mom} ({resonant_mom/total_mom:.2%})")
    
    # 5. THE ULTIMATE TRUTH: Momentum EV Lift
    res_ev = df[mom_mask & df['resonance']]['mom_ev'].mean()
    non_res_ev = df[mom_mask & ~df['resonance']]['mom_ev'].mean()
    
    logger.info("-" * 40)
    logger.info(f"Avg Momentum EV (Resonant): {res_ev:.4%}")
    logger.info(f"Avg Momentum EV (Dissonant): {non_res_ev:.4%}")
    
    if res_ev > non_res_ev:
        lift = (res_ev / non_res_ev) - 1 if non_res_ev != 0 else 0
        logger.info(f"PLANETARY RESONANCE LIFT: {lift:.2%}")
        logger.info("Verdict: The Comet Cycle fuels the Intraday Pulse.")
    else:
        logger.info("Verdict: No clear resonance found at these scales.")
    
    logger.info("============================================")

if __name__ == "__main__":
    run_resonance_audit()
