"""
CLAWD Quant Trading - Centralized Configuration
Loads from .env file, provides fallback defaults
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from current directory or parent
def _load_env():
    env_paths = [
        Path(__file__).parent / '.env',
        Path(__file__).parent.parent / '.env',
        Path.cwd() / '.env',
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            return env_path
    return None

_env_file = _load_env()

# === PATHS ===
def get_working_dir():
    """Get the working directory - uses env var or defaults to 02-Trading location"""
    from_env = os.getenv('WORKING_DIR')
    if from_env:
        return Path(from_env)
    # Try to find 02-Trading in parent paths
    current = Path.cwd()
    if current.name == '02-Trading':
        return current
    if (current / '02-Trading').exists():
        return current / '02-Trading'
    # Fallback to hardcoded path (for backwards compatibility)
    return Path(r'C:\Users\Admin\clawd\02-Trading')

WORKING_DIR = get_working_dir()
DATA_DIR = Path(os.getenv('DATA_DIR', WORKING_DIR / 'data'))
LOG_DIR = Path(os.getenv('LOG_DIR', WORKING_DIR / 'logs'))
CSV_EXPORT_DIR = Path(os.getenv('CSV_EXPORT_DIR', WORKING_DIR / 'csv_exports'))

# === TRADING PARAMETERS ===
STARTING_EQUITY = float(os.getenv('STARTING_EQUITY', 100000))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.02))
MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', 5))

# === SERVER SETTINGS ===
BACKEND_PORT = int(os.getenv('BACKEND_PORT', 8081))
HEALTH_PORT = int(os.getenv('HEALTH_PORT', 8082))
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', 8080))

# === SYMBOLS ===
DEFAULT_SYMBOL = os.getenv('DEFAULT_SYMBOL', 'SPY')
WATCHLIST_SYMBOLS = os.getenv('WATCHLIST_SYMBOLS', 
    'SPY,QQQ,MSFT,AAPL,TSLA,NVDA,AMZN,GOOGL,META,NFLX,AMD,CRM,ES=F,NQ=F,YM=F,CL=F,GC=F'
).split(',')

# === ALPACA (Primary Data Feed) ===
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_PAPER = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'

# === POLYGON (Optional) ===
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')

# === DATA SETTINGS ===
DEFAULT_TIMEFRAME = os.getenv('DEFAULT_TIMEFRAME', '1H')
DEFAULT_HISTORY_DAYS = int(os.getenv('DEFAULT_HISTORY_DAYS', 90))
MAX_HISTORY_DAYS = int(os.getenv('MAX_HISTORY_DAYS', 1825))  # 5 years

# === FEATURE FLAGS ===
USE_POLYGON = os.getenv('USE_POLYGON', 'false').lower() == 'true'
USE_ALPACA = os.getenv('USE_ALPACA', 'true').lower() == 'true'
USE_YFINANCE = os.getenv('USE_YFINANCE', 'true').lower() == 'true'
PAPER_TRADING = os.getenv('PAPER_TRADING', 'true').lower() == 'true'

# === LOGGING ===
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_RETENTION_DAYS = int(os.getenv('LOG_RETENTION_DAYS', 30))

# === COT/GEX DATA ===
COT_DATA_PATH = Path(os.getenv('COT_DATA_PATH', WORKING_DIR / 'data' / 'cot'))
GEX_DATA_PATH = Path(os.getenv('GEX_DATA_PATH', WORKING_DIR / 'data' / 'gex'))
USE_COMMERCIAL_COT = os.getenv('USE_COMMERCIAL_COT', 'false').lower() == 'true'
USE_COMMERCIAL_GEX = os.getenv('USE_COMMERCIAL_GEX', 'false').lower() == 'true'

# Ensure directories exist
def ensure_dirs():
    """Create necessary directories if they don't exist"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    COT_DATA_PATH.mkdir(parents=True, exist_ok=True)
    GEX_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Auto-create on import
ensure_dirs()

if __name__ == '__main__':
    print("CLAWD Quant Trading Configuration")
    print(f"  Working Dir: {WORKING_DIR}")
    print(f"  Data Dir: {DATA_DIR}")
    print(f"  Log Dir: {LOG_DIR}")
    print(f"  Starting Equity: ${STARTING_EQUITY:,.2f}")
    print(f"  Default Symbol: {DEFAULT_SYMBOL}")
    print(f"  Backend Port: {BACKEND_PORT}")
    print(f"  .env loaded from: {_env_file or 'NOT FOUND - using defaults'}")
