"""
Sovereign Universe — full asset list for backtest and live trading.
Edit this file to add/remove assets; all sweep scripts read from here.
"""

UNIVERSE = [
    # TECH MEGA CAP
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'TSLA', 'AMD', 'AVGO', 'ASML', 'TSM',
    # AI PURE PLAY (lifetime holds)
    'PLTR', 'CRWD', 'NET', 'SNOW', 'AI', 'PATH', 'SOUN', 'IONQ', 'BBAI',
    # FINANCE
    'JPM', 'GS', 'BAC', 'MS', 'BLK', 'V', 'MA',
    # HEALTH
    'UNH', 'JNJ', 'PFE', 'ABBV', 'LLY', 'MRK',
    # ENERGY
    'XOM', 'CVX', 'OXY', 'SLB',
    # MACRO ETF
    'SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'SLV', 'USO', 'UUP',
    # FUTURES PROXIES
    'ES=F', 'NQ=F', 'GC=F', 'SI=F',
    # VOLATILITY
    'VXX',
    # INTERNATIONAL
    'EEM', 'EFA', 'FXI',
    # SECTOR
    'XLF', 'XLK', 'XLE', 'XLV', 'XLU',
]

AI_BUCKET = ['PLTR', 'CRWD', 'NET', 'SNOW', 'AI', 'PATH', 'SOUN', 'IONQ', 'BBAI']
