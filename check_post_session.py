"""
MONDAY POST-SESSION REVIEW (04:30 PM ET)
Performs P&L Audit, Bayesian Win-Rate Update, and Log Archival.
Objective: Refine system expectations based on Phase 1 performance.
"""

import json
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# THE BAYESIAN PRIOR (Phase 1 Target: 70% Win Rate)
PRIOR_ALPHA = 70.0
PRIOR_BETA = 30.0

def run_post_session_review():
    logger.info("Initializing Post-Session Review (04:30 PM)...")
    
    # 1. LOAD PAPER TRADES LOG
    log_path = "paper_trades.jsonl" # Path from meta_evaluator
    if not os.path.exists(log_path):
        logger.error("❌ NO TRADES FOUND IN paper_trades.jsonl. NO-GO.")
        return

    # 2. AUDIT TODAY'S TRADES (Last N trades)
    todays_trades = []
    with open(log_path, 'r') as f:
        for line in f:
            trade = json.loads(line)
            # Filter for today's date
            trade_date = trade.get('timestamp', '').split('T')[0]
            if trade_date == datetime.now().date().isoformat():
                todays_trades.append(trade)

    if not todays_trades:
        logger.info("✅ No trades executed today. System stayed flat. (Protocol Followed)")
        return

    # 3. CALCULATE REALIZED P&L AND BAYESIAN UPDATE
    wins = 0
    total_pnl = 0.0
    
    for trade in todays_trades:
        pnl = trade.get('pnl', 0.0)
        status = trade.get('status', 'OPEN')
        
        if status == 'CLOSED':
            total_pnl += pnl
            if pnl > 0: wins += 1
            
    # 4. BAYESIAN WIN-RATE UPDATE
    # Update the Alpha/Beta distribution
    new_alpha = PRIOR_ALPHA + wins
    new_beta = PRIOR_BETA + (len(todays_trades) - wins)
    expected_win_rate = new_alpha / (new_alpha + new_beta)

    logger.info("============================================")
    logger.info("📈 POST-SESSION PERFORMANCE AUDIT")
    logger.info("============================================")
    logger.info(f"Total Trades Executed: {len(todays_trades)}")
    logger.info(f"Closed P&L: ${total_pnl:,.2f}")
    logger.info(f"Win Rate Impact: {wins}/{len(todays_trades)}")
    logger.info("-" * 40)
    logger.info(f"NEW BAYESIAN EXPECTED WIN RATE: {expected_win_rate:.2%}")
    logger.info("============================================")

if __name__ == "__main__":
    run_post_session_review()
