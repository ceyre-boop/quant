import os
import logging
from data.alpaca_client import AlpacaDataClient
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_alpaca_connection():
    # Load .env
    load_dotenv()
    
    logger.info("Initializing Alpaca Data Client...")
    client = AlpacaDataClient()
    
    symbol = "SPY"
    logger.info(f"Testing bar fetching for {symbol}...")
    
    try:
        # Fetch last 5 hourly bars
        from datetime import datetime, timedelta
        end = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        start = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        bars = client.get_bars(symbol, '1Hour', start, end, limit=5)
        
        if bars:
            logger.info(f"✅ Success! Received {len(bars)} bars for {symbol}.")
            latest_bar = bars[-1]
            logger.info(f"Latest Close: ${latest_bar['c']} at {latest_bar['t']}")
            return True
        else:
            logger.error("❌ Failed: Received empty bar list. Check if your API key has access to historical data (Market Data v2).")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during connection test: {e}")
        return False

if __name__ == "__main__":
    test_alpaca_connection()
