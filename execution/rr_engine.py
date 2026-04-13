"""
V5.2 Dynamic Risk-Reward (RR) Engine
Objective: Automated management of stops, targets, and trailing logic.
Formula: Stop = ATR(20) * 1.5. Target = ATR(20) * 3.0 (2:1 Initial).
Trailing: Activate after 1.5R.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RREngine:
    def __init__(self, atr_multiplier=1.5, target_ratio=2.0):
        self.atr_multiplier = atr_multiplier
        self.target_ratio = target_ratio
        
    def calculate_brackets(self, entry_price, current_atr, direction):
        """
        Calculates hard Stop-Loss and Take-Profit based on ATR.
        direction: 1 for Long, -1 for Short.
        """
        stop_dist = current_atr * self.atr_multiplier
        target_dist = stop_dist * self.target_ratio
        
        if direction == 1:
            sl = entry_price - stop_dist
            tp = entry_price + target_dist
        else:
            sl = entry_price + stop_dist
            tp = entry_price - target_dist
            
        return {'sl': sl, 'tp': tp}

    def update_trailing_stop(self, current_price, entry_price, current_sl, direction):
        """
        Implements Trailing Logic:
        If Profit > 1.5 * Unit Risk, move stop to Entry + 0.5R (Lock in profit).
        """
        unit_risk = abs(entry_price - current_sl)
        current_profit = (current_price - entry_price) * direction
        
        # 1.5R Threshold for Trailing Activation
        if current_profit > (unit_risk * 1.5):
            new_sl = entry_price + (unit_risk * 0.5) * direction
            # Only move stop in one direction (tightening)
            if direction == 1:
                return max(current_sl, new_sl)
            else:
                return min(current_sl, new_sl)
        
        return current_sl

if __name__ == "__main__":
    # Unit Test
    engine = RREngine()
    entry = 100.0
    atr = 2.0
    brackets = engine.calculate_brackets(entry, atr, 1)
    
    # Simulate a move to 1.5R
    # 1.5R = 1.5 * (2*1.5) = 4.5. New price = 104.5
    new_sl = engine.update_trailing_stop(104.5, entry, brackets['sl'], 1)
    
    print(f"Brackets: {brackets}")
    print(f"New SL at 1.5R: {new_sl} (Expected: 101.5)")
