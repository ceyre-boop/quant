# sovereign/simulation.py
"""
The correction engine.
Runs simulated trades on historical data.
Identifies exactly WHERE and WHY the model fails.
Feeds failure data back into retraining.

Simulate → Measure → Find leak → Retrain → Repeat
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SimulationLoop:
    
    def __init__(self, signal_engine, trainer, db):
        self.engine  = signal_engine
        self.trainer = trainer
        self.db      = db
        self.trades  = []
        self.vetoes  = []
        self.iteration = 0
    
    def run(self, tickers: list, start: str, end: str) -> dict:
        """
        Run one full simulation pass.
        Returns metrics dict — the single source of truth.
        """
        self.iteration += 1
        self.trades = []
        self.vetoes = []
        
        dates = pd.date_range(start=start, end=end, freq='B')  # business days
        
        for date in dates:
            d_str = str(date.date())
            for ticker in tickers:
                signal = self.engine.generate_signal(ticker, d_str)
                
                if signal.direction == 'none':
                    # Log the veto with reason
                    self.vetoes.append({
                        'date':     d_str,
                        'ticker':   ticker,
                        'regime':   signal.regime,
                        'prob':     signal.probability,
                        'ev':       signal.ev,
                        'reason':   self._veto_reason(signal),
                    })
                    continue
                
                # Simulate the trade outcome
                outcome = self._simulate_outcome(ticker, date, signal)
                self.trades.append(outcome)
        
        return self._calculate_metrics()
    
    def _simulate_outcome(self, ticker, signal_date, signal) -> dict:
        """
        Use actual historical prices to determine outcome.
        This is the moment of truth.
        """
        try:
            # Get bars after signal date
            # We need to get enough bars into the future.
            # Using data_provider.get_historical_data
            future = self.db.get_historical_data(ticker, period="1mo", interval="1d")
            if future is None or future.empty:
                return self._build_trade_record(signal, 'NO_FUTURE_DATA', 0, 0)
            
            # Slice future to start from signal_date + 1 day
            future = future[future.index > str(signal_date.date())]
            
            if len(future) < 3:
                return self._build_trade_record(signal, 'INCOMPLETE', 0, 0)
            
            # Entry at next bar open (T+1)
            entry = future['open'].iloc[0]
            atr   = signal.features.get('atr_14', abs(signal.entry_price - signal.stop))
            if atr == 0: atr = entry * 0.01 # Fallback
            
            # Direction multiplier
            d = 1 if signal.direction == 'long' else -1
            
            # Recalculate stop and target from actual entry
            stop_dist   = 1.5 * atr
            stop_price  = entry - d * stop_dist
            
            # Target: depends on regime
            if signal.regime == 'reversion':
                # For MVP, target is mean_50 from features or current mean
                target_price = signal.features.get('zscore_50_mean', entry) 
                # If not available, use a 2.0x ATR target for reversion as placeholder
                if target_price == entry: target_price = entry + d * (2.0 * atr)
            else:
                target_price = entry + d * (3.0 * atr)
            
            target_dist = abs(target_price - entry)
            
            # Walk forward bar by bar
            for i in range(1, min(20, len(future))):
                bar_low = future['low'].iloc[i]
                bar_high = future['high'].iloc[i]
                bar_close = future['close'].iloc[i]
                
                # Check stop hit
                if signal.direction == 'long':
                    if bar_low <= stop_price:
                        return self._build_trade_record(signal, 'STOPPED', -1.0, i)
                    if bar_high >= target_price:
                        return self._build_trade_record(signal, 'TARGET_HIT', target_dist / stop_dist, i)
                else: # short
                    if bar_high >= stop_price:
                        return self._build_trade_record(signal, 'STOPPED', -1.0, i)
                    if bar_low <= target_price:
                        return self._build_trade_record(signal, 'TARGET_HIT', target_dist / stop_dist, i)
            
            # Neither hit within window
            final_move = (future['close'].iloc[min(19, len(future)-1)] - entry) * d
            rr_achieved = final_move / stop_dist
            outcome = 'EXPIRED'
            bars    = min(19, len(future)-1)
            
            return self._build_trade_record(signal, outcome, rr_achieved, bars)
        
        except Exception as e:
            logger.error(f"Simulation error for {ticker} on {signal_date}: {e}")
            return self._build_trade_record(signal, 'ERROR', 0, 0)
    
    def _build_trade_record(self, signal, outcome, rr, bars) -> dict:
        return {
            'date':        signal.date,
            'ticker':      signal.ticker,
            'direction':   signal.direction,
            'regime':      signal.regime,
            'probability': signal.probability,
            'ev_at_signal': signal.ev,
            'outcome':     outcome,
            'rr_achieved': round(rr, 3),
            'bars_held':   bars,
            'winner':      1 if outcome == 'TARGET_HIT' else 0,
            # All features for debugging
            **{f'feat_{k}': v for k,v in signal.features.items()
               if k in ['hurst','adx_14','rsi_14','zscore_50','volume_zscore']},
        }
    
    def _calculate_metrics(self) -> dict:
        if not self.trades:
            return {'error': 'no trades generated'}
        
        df = pd.DataFrame(self.trades)
        
        # Core metrics
        total       = len(df)
        winners     = df['winner'].sum()
        win_rate    = winners / total if total > 0 else 0
        avg_rr      = df['rr_achieved'].mean()
        
        winning_trades = df[df['winner']==1]
        avg_win_rr = winning_trades['rr_achieved'].mean() if len(winning_trades) > 0 else 0
        expectancy  = (win_rate * avg_win_rr - (1-win_rate) * 1.0) if total > 0 else 0
        
        # High conviction subset
        hc = df[df['probability'] >= 0.60]
        hc_win_rate = hc['winner'].mean() if len(hc) > 0 else 0
        
        # Regime breakdown — WHERE is it failing?
        regime_stats = df.groupby('regime').agg(
            count=('winner','count'),
            win_rate=('winner','mean'),
            avg_rr=('rr_achieved','mean')
        ).round(3)
        
        # Veto stats
        veto_df = pd.DataFrame(self.vetoes) if self.vetoes else pd.DataFrame()
        veto_by_reason = veto_df['reason'].value_counts() if len(veto_df) > 0 else {}
        
        # Failure indices for retraining
        failure_idx = df[df['winner']==0].index.tolist()
        
        metrics = {
            'iteration':       self.iteration,
            'total_trades':    total,
            'win_rate':        round(win_rate, 4),
            'win_rate_hc':     round(hc_win_rate, 4),
            'hc_trade_count':  len(hc),
            'avg_rr':          round(avg_rr, 3),
            'expectancy':      round(expectancy, 3),
            'total_vetoes':    len(self.vetoes),
            'regime_stats':    regime_stats.to_dict(),
            'veto_breakdown':  dict(veto_by_reason),
            'failure_indices': failure_idx,
            'raw_trades':      df,
            'pass':            hc_win_rate >= 0.63 and len(hc) >= 30,
        }
        
        self._print_metrics(metrics)
        return metrics
    
    def correct_and_retrain(self, metrics: dict, features_df: dict, labels: dict):
        """
        The correction loop.
        Takes simulation results → retrains on failures → better model.
        """
        failures = metrics['failure_indices']
        
        print(f"\n🔧 CORRECTION ITERATION {self.iteration}")
        print(f"   Failures to learn from: {len(failures)}")
        
        # Retrain momentum model on failures
        if 'momentum' in features_df:
            self.engine.momentum_model = self.trainer.retrain_on_failures(
                features_df['momentum'],
                labels['momentum'],
                self.trainer.MOMENTUM_FEATURES,
                'momentum',
                failures
            )
        
        # Retrain reversion model on failures
        if 'reversion' in features_df:
            self.engine.reversion_model = self.trainer.retrain_on_failures(
                features_df['reversion'],
                labels['reversion'],
                self.trainer.REVERSION_FEATURES,
                'reversion',
                failures
            )
        
        print(f"   Models retrained. Run simulation again to measure improvement.")
    
    def _veto_reason(self, signal) -> str:
        if signal.regime == 'dead_zone':    return 'HURST_DEAD_ZONE'
        if signal.regime == 'gap_rejected': return 'GAP_ENTROPY'
        if signal.regime == 'no_model':     return 'NO_MODEL'
        if signal.regime == 'no_data':      return 'NO_DATA'
        if signal.regime == 'insufficient_data': return 'INSUFFICIENT_DATA'
        if signal.probability < 0.50:       return 'PROB_TOO_LOW'
        if signal.ev <= 0:                  return 'EV_NEGATIVE'
        return 'RR_TOO_LOW'
    
    def _print_metrics(self, m: dict):
        status = '✅ PASS' if m.get('pass') else '❌ NOT YET'
        print(f"""
{'='*55}
  SIMULATION RESULTS — ITERATION {m['iteration']}
{'='*55}
  Status:           {status}
  Total trades:     {m['total_trades']}
  Win rate (all):   {m['win_rate']:.1%}
  Win rate (0.60+): {m['win_rate_hc']:.1%}  ← target: 63%
  HC trade count:   {m['hc_trade_count']}   ← need 30+
  Avg R:R:          {m['avg_rr']:.2f}
  Expectancy:       {m['expectancy']:.3f}R per trade
  Total vetoes:     {m['total_vetoes']}

  REGIME BREAKDOWN:
{pd.DataFrame(m['regime_stats']).to_string() if m.get('regime_stats') else 'N/A'}

  VETO BREAKDOWN:
{chr(10).join(f"    {k}: {v}" for k,v in m.get('veto_breakdown', {}).items())}
{'='*55}
        """)
