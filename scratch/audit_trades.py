import pandas as pd
df = pd.read_csv('data/backtest_results/trades_raw_20260410_133259.csv')
print(f"Trades Count: {len(df)}")
if not df.empty and 'confidence' in df.columns:
    print(f"Min Conf: {df['confidence'].min()}")
    print(f"Max Conf: {df['confidence'].max()}")
    print(f"Avg Conf: {df['confidence'].mean()}")
else:
    print(f"Columns: {df.columns}")
