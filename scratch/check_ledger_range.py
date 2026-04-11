import pandas as pd
df = pd.read_csv('data/backtest_results/signals_raw_20260409_222335.csv')
df['timestamp'] = pd.to_datetime(df.iloc[:, 0], utc=True)
for symbol in ['SPY', 'NVDA', 'ARM']:
    subset = df[df['symbol'] == symbol]
    if not subset.empty:
        print(f"{symbol} Range: {subset['timestamp'].min()} to {subset['timestamp'].max()}")
    else:
        print(f"{symbol} NOT FOUND")
