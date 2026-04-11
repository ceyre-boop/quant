import pandas as pd
import os

def analyze():
    file_path = 'data/backtest_results/trades_raw_current_harvest.csv'
    if not os.path.exists(file_path):
        print("File not found.")
        return
        
    df = pd.read_csv(file_path)
    print(f"Total Trades: {len(df)}")
    print(f"Win Rate: {df['win'].mean():.2%}")
    print(f"Total PNL: ${df['pnl'].sum():,.2f}")
    
    # Range
    print(f"Start Date: {df['entry_date'].min()}")
    print(f"End Date: {df['entry_date'].max()}")
    
    # Per symbol performance
    symbol_perf = df.groupby('symbol').agg({
        'pnl': 'sum',
        'win': 'mean',
        'entry_date': 'count'
    }).rename(columns={'entry_date': 'trades'}).sort_values('pnl', ascending=False)
    
    print("\n[TOP SYMBOLS]")
    print(symbol_perf.head(5))
    
    print("\n[WORST SYMBOLS]")
    print(symbol_perf.tail(5))

if __name__ == "__main__":
    analyze()
