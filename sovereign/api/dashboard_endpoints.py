# sovereign/api/dashboard_endpoints.py
"""
Flask Blueprint for Sovereign MVP.
Plug into existing Clawdbot dashboard_api.py.
The dashboard shows you exactly what to fix next.
"""

from flask import Blueprint, jsonify, request
import logging
from datetime import datetime, timedelta
import pandas as pd

from sovereign.signal_engine import SignalEngine
from sovereign.ml_trainer import MLTrainer
from sovereign.simulation import SimulationLoop
from data.providers import get_provider

logger = logging.getLogger(__name__)
sovereign_bp = Blueprint('sovereign', __name__)

# Singletons for MVP
data_provider = get_provider()
signal_engine = SignalEngine(data_provider)
trainer = MLTrainer()
sim_loop = SimulationLoop(signal_engine, trainer, data_provider)

# Cache for latest results
last_metrics = {}

@sovereign_bp.route('/simulation/run', methods=['GET'])
def run_simulation():
    """Trigger a simulation pass. Returns metrics."""
    global last_metrics
    start = request.args.get('start', (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'))
    end = request.args.get('end', datetime.now().strftime('%Y-%m-%d'))
    tickers = request.args.get('tickers', 'SPY,QQQ,AAPL,MSFT,TSLA').split(',')
    
    # In a real MVP, you'd want to train the model first if it's None
    if signal_engine.momentum_model is None:
        # Mini training session on historical data
        try:
            df = data_provider.get_historical_data(tickers[0], period="2y", interval="1d")
            if df is not None:
                feat_df = signal_engine.compute_features(df)
                mom_labels = trainer.build_momentum_labels(df)
                rev_labels = trainer.build_reversion_labels(df)
                
                signal_engine.momentum_model, _ = trainer.train(feat_df, mom_labels, trainer.MOMENTUM_FEATURES, 'momentum')
                signal_engine.reversion_model, _ = trainer.train(feat_df, rev_labels, trainer.REVERSION_FEATURES, 'reversion')
        except Exception as e:
            logger.error(f"Training error: {e}")
            return jsonify({"error": f"Failed to train initial models: {str(e)}"}), 500

    metrics = sim_loop.run(tickers, start, end)
    
    # Handle non-serializable objects (like DataFrame) for JSON
    serializable_metrics = metrics.copy()
    if 'raw_trades' in serializable_metrics:
        serializable_metrics['raw_trades'] = serializable_metrics['raw_trades'].tail(50).to_dict(orient='records')
    
    last_metrics = serializable_metrics
    return jsonify(serializable_metrics)

@sovereign_bp.route('/simulation/metrics', methods=['GET'])
def get_metrics():
    """Current iteration metrics."""
    return jsonify(last_metrics)

@sovereign_bp.route('/simulation/failures', methods=['GET'])
def get_failures():
    """
    Returns the losing trades with full feature breakdown.
    This is the most useful view — tells you WHY it failed.
    """
    if not last_metrics or 'raw_trades' not in last_metrics:
        return jsonify({"error": "No simulation run yet"}), 404
        
    # last_metrics['raw_trades'] in memory is a list of dicts if already serialized, 
    # but in sim_loop it's a list of dicts.
    trades = last_metrics.get('raw_trades', [])
    failures = [t for t in trades if t.get('winner') == 0]
    
    return jsonify({
        "count": len(failures),
        "failures": failures[-20:] # Return last 20 failures
    })

@sovereign_bp.route('/signals/today', methods=['GET'])
def get_todays_signals():
    """Live signals for next open."""
    tickers = request.args.get('tickers', 'SPY,QQQ,AAPL,MSFT,TSLA').split(',')
    today = datetime.now().strftime('%Y-%m-%d')
    
    signals = []
    for ticker in tickers:
        sig = signal_engine.generate_signal(ticker, today)
        signals.append(sig.to_dict())
        
    return jsonify({
        "date": today,
        "signals": signals
    })

@sovereign_bp.route('/regime/current', methods=['GET'])
def get_current_regime():
    """
    What regime are we in right now?
    What is the Hurst reading across the universe?
    """
    tickers = request.args.get('tickers', 'SPY,QQQ,AAPL,MSFT,TSLA').split(',')
    today = datetime.now().strftime('%Y-%m-%d')
    
    regimes = {}
    for ticker in tickers:
        # Get enough data for features
        df = data_provider.get_historical_data(ticker, period="1y", interval="1d")
        if df is not None:
            feat_df = signal_engine.compute_features(df)
            latest = feat_df.iloc[-1].to_dict()
            regimes[ticker] = {
                "regime": signal_engine.classify_regime(latest),
                "hurst": latest.get('hurst', 0.5),
                "adx": latest.get('adx_14', 0),
                "price": latest.get('close', 0)
            }
            
    return jsonify(regimes)
