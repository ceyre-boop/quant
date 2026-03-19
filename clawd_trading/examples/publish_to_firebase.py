"""
Firebase Demo Publisher
Pushes sample trading data to Firebase Realtime DB
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from firebase_admin import credentials, initialize_app, db
except ImportError:
    print("Installing firebase-admin...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "firebase-admin", "-q"])
    from firebase_admin import credentials, initialize_app, db

# Initialize Firebase (uses default credentials or env vars)
def init_firebase():
    try:
        # Try to initialize with default (works in Cloud Functions)
        initialize_app(options={
            'databaseURL': 'https://clawd-trading-7b8de-default-rtdb.firebaseio.com'
        })
        print("✅ Firebase initialized")
    except Exception as e:
        print(f"Firebase init error: {e}")
        print("Make sure you have FIREBASE_SERVICE_ACCOUNT env var set")
        return False
    return True

def push_demo_data():
    """Push demo trading state to Firebase"""
    
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Three-layer state
    state = {
        "bias": {
            "direction": 1,
            "magnitude": 2,
            "confidence": 0.78,
            "regime_override": False,
            "rationale": ["LIQUIDITY_SWEEP_CONFIRMED", "MOMENTUM_ACCELERATION", "BREAKOUT_CONFIRMED"],
            "model_version": "v1.0"
        },
        "risk": {
            "position_size": 1.2,
            "kelly_fraction": 0.42,
            "stop_price": 21840.0,
            "tp1_price": 21975.0,
            "tp2_price": 22050.0,
            "expected_value": 1.84,
            "ev_positive": True
        },
        "game": {
            "game_state_aligned": True,
            "adversarial_risk": "LOW",
            "forced_move_probability": 0.61,
            "game_state_summary": "SHORTS_TRAPPED_SQUEEZE_RISK",
            "kyle_lambda": 0.0034
        },
        "regime": {
            "volatility": "NORMAL",
            "trend": "STRONG_TREND",
            "risk_appetite": "RISK_ON",
            "momentum": "ACCELERATING",
            "event_risk": "CLEAR",
            "composite_score": 0.72
        },
        "session": {
            "pnl": 312,
            "position": "LONG",
            "entry": 21905.0
        },
        "controls": {
            "trading_enabled": True,
            "daily_loss_pct": 0.4,
            "open_positions": 1,
            "hard_logic_status": "CLEAR",
            "max_daily_loss": 500
        },
        "liquidity_pools": [
            {"price": 18600, "type": "highs", "prob": 0.91, "strength": 4, "swept": False},
            {"price": 18540, "type": "highs", "prob": 0.67, "strength": 3, "swept": False},
            {"price": 18500, "type": "highs", "prob": 0.44, "strength": 2, "swept": False},
            {"price": 18450, "type": "highs", "prob": 0.29, "strength": 1, "swept": False},
            {"price": 21905, "type": "price", "prob": 0, "strength": 0, "swept": False},
            {"price": 21860, "type": "lows", "prob": 0.35, "strength": 2, "swept": False},
            {"price": 21820, "type": "lows", "prob": 0.58, "strength": 3, "swept": False},
            {"price": 21750, "type": "lows", "prob": 0.82, "strength": 4, "swept": False},
            {"price": 21680, "type": "lows", "prob": 0.91, "strength": 4, "swept": True}
        ],
        "signals": [
            {"dt": "2026-03-18 09:42", "dir": 1, "conf": 0.78, "entry": 21905, "stop": 21840, "tp1": 21975, "ev": 1.84, "outcome": "open"},
            {"dt": "2026-03-17 10:15", "dir": 1, "conf": 0.68, "entry": 21810, "stop": 21745, "tp1": 21895, "ev": 2.11, "outcome": "win"},
            {"dt": "2026-03-14 09:55", "dir": -1, "conf": 0.62, "entry": 22005, "stop": 22075, "tp1": 21930, "ev": 1.67, "outcome": "loss"},
            {"dt": "2026-03-13 13:42", "dir": 1, "conf": 0.74, "entry": 21750, "stop": 21675, "tp1": 21845, "ev": 2.33, "outcome": "win"}
        ],
        "explainability": {
            "LIQUIDITY_SWEEP_CONFIRMED": 0.31,
            "MOMENTUM_ACCELERATION": 0.24,
            "BREAKOUT_CONFIRMED": 0.18,
            "VOLATILITY_SPIKE": 0.12,
            "SENTIMENT_FLOW": 0.08,
            "BREADTH_DIVERGENCE": 0.05,
            "CALENDAR_RISK": 0.02
        },
        "model": {
            "version": "v1.0",
            "oos_accuracy": 0.587,
            "trained_at": "2026-03-10T02:14:22",
            "n_features": 26,
            "refit_flagged": False
        },
        "weekly": {
            "sharpe": 1.42,
            "win_rate": 0.587,
            "max_dd": -0.038
        },
        "pnl_series": [0, 85, 165, 210, 175, 248, 312],
        "timestamp": timestamp
    }
    
    # Push to Firebase
    ref = db.reference('trading_state')
    ref.set(state)
    print(f"✅ Data pushed at {timestamp}")
    
    # Also push session_controls
    controls_ref = db.reference('session_controls')
    controls_ref.set({
        "trading_enabled": True,
        "max_daily_loss": 500,
        "daily_loss_pct": 0.4,
        "open_positions": 1,
        "hard_logic_status": "CLEAR"
    })
    print("✅ Controls pushed")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("CLAWD TRADING - FIREBASE DEMO PUBLISHER")
    print("="*60)
    
    if init_firebase():
        if push_demo_data():
            print("\n✅ All data published to Firebase!")
            print("Dashboard should now show live data")
        else:
            print("\n❌ Failed to push data")
    else:
        print("\n❌ Firebase not initialized")
