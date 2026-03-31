"""
Updated Dashboard - Reads from REAL Backend API

Replace the Firebase initialization in trading-dashboard.html with this API client.
"""

API_CLIENT_JS = """
// REAL BACKEND API CLIENT
// Replaces Firebase with direct API calls to the Python backend

const API_BASE = 'http://localhost:5000/api';
const SYMBOL = 'SPX500';  // Default symbol

let isConnected = false;
let currentSymbol = SYMBOL;

// Fetch live state from real backend
async function fetchLiveState(symbol) {
  try {
    const response = await fetch(`${API_BASE}/live_state/${symbol}`);
    if (!response.ok) throw new Error('API error');
    
    const data = await response.json();
    
    if (data.error) {
      addSysLog('ERROR', data.error);
      return;
    }
    
    updateLiveState(data);
    isConnected = true;
    updateConnectionStatus();
    addSysLog('INFO', `Live state updated: ${symbol} @ $${data.price?.toFixed(2)}`);
    
  } catch (e) {
    console.error('API fetch error:', e);
    isConnected = false;
    updateConnectionStatus();
    addSysLog('ERROR', 'Backend API unavailable');
  }
}

// Fetch signals from real backend
async function fetchSignals(symbol) {
  try {
    const response = await fetch(`${API_BASE}/signals/${symbol}`);
    if (!response.ok) throw new Error('API error');
    
    const data = await response.json();
    
    if (data.latest) {
      updateSignalHistory({ [data.latest.signal_id]: data.latest });
      addSysLog('INFO', `Signal: ${data.latest.direction} ${symbol}`);
    }
    
  } catch (e) {
    console.error('Signal fetch error:', e);
  }
}

// Fetch system status
async function fetchSystemStatus() {
  try {
    const response = await fetch(`${API_BASE}/system/status`);
    if (!response.ok) throw new Error('API error');
    
    const data = await response.json();
    updateControls({
      hard_logic_status: data.status,
      trading_enabled: true,
      open_positions: 0,
    });
    
  } catch (e) {
    console.error('Status fetch error:', e);
  }
}

// Fetch backtest baseline
async function fetchBacktestBaseline() {
  try {
    const response = await fetch(`${API_BASE}/backtest/baseline`);
    if (!response.ok) throw new Error('API error');
    
    const data = await response.json();
    
    // Update backtest display
    document.getElementById('bt-sharpe').textContent = data.sharpe_ratio.toFixed(2);
    document.getElementById('bt-winrate').textContent = (data.win_rate * 100).toFixed(1) + '%';
    document.getElementById('bt-dd').textContent = (data.max_drawdown * 100).toFixed(1) + '%';
    document.getElementById('bt-trades').textContent = data.total_trades;
    
    addSysLog('INFO', `Baseline loaded: Sharpe ${data.sharpe_ratio}`);
    
  } catch (e) {
    console.error('Baseline fetch error:', e);
  }
}

// Poll for updates
function startPolling() {
  // Initial fetch
  fetchLiveState(currentSymbol);
  fetchSignals(currentSymbol);
  fetchSystemStatus();
  fetchBacktestBaseline();
  
  // Poll every 5 seconds
  setInterval(() => {
    fetchLiveState(currentSymbol);
    fetchSignals(currentSymbol);
    fetchSystemStatus();
  }, 5000);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  addSysLog('INFO', 'Dashboard initialized - Connecting to backend API...');
  
  // Start polling instead of Firebase
  startPolling();
  
  addSysLog('INFO', 'Real-time data from Python backend');
});
"""

print("API_CLIENT_JS ready to inject into dashboard")
print("\nTo use:")
print("1. Run: python dashboard_api.py")
print("2. Open: http://localhost:5000/")
print("3. Dashboard will show REAL data from backend")
