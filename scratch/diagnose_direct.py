import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

def diagnose():
    load_dotenv()
    key = os.getenv('ALPACA_API_KEY')
    secret = os.getenv('ALPACA_SECRET_KEY')
    
    headers = {
        'APCA-API-KEY-ID': key,
        'APCA-API-SECRET-KEY': secret
    }
    
    symbol = "SPY"
    base_url = "https://data.alpaca.markets/v2/stocks"
    
    test_days = [1, 7, 30, 90, 365]
    
    for days in test_days:
        end = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        start = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        url = f"{base_url}/{symbol}/bars"
        params = {
            'timeframe': '1Hour',
            'start': start,
            'end': end,
            'limit': 5,
            'adjustment': 'raw',
            'feed': 'iex'
        }
        
        print(f"Testing {days} days history...")
        try:
            resp = requests.get(url, headers=headers, params=params)
            if resp.status_code == 200:
                print(f"  OK: Received {len(resp.json().get('bars', []))} bars")
            else:
                print(f"  FAILED: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    diagnose()
