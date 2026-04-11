
import pandas as pd
import glob
import os
from datetime import datetime

def run_audit():
    # Priority 1: Current Harvest (Incremental)
    # Priority 2: Latest Timestamped Harvest
    partial_file = 'data/backtest_results/signals_raw_current_harvest.csv'
    if os.path.exists(partial_file):
        latest_file = partial_file
    else:
        list_of_files = glob.glob('data/backtest_results/signals_raw_*.csv')
        if not list_of_files:
            print("No signal logs found in data/backtest_results/")
            return
        latest_file = max(list_of_files, key=os.path.getctime)
    print(f"--- POST-SESSION AUDIT: {os.path.basename(latest_file)} ---")
    
    df = pd.read_csv(latest_file)
    
    # Check if 'grade' exists (new V2 logic)
    if 'grade' not in df.columns:
        print("Log file uses legacy scoring. Run V2 to see grades.")
        return

    # 1. Conviction Distribution
    distribution = df['grade'].value_counts()
    print("\n[CONVICTION DISTRIBUTION]")
    for grade, count in distribution.items():
        print(f" - Grade {grade}: {count} signals")

    # 2. Performance of Grades (if Win/PNL data exists)
    if 'win' in df.columns:
        print("\n[THEORETICAL PERFORMANCE BY GRADE]")
        perf = df.groupby('grade').agg({
            'win': 'mean',
            'pnl': 'sum'
        }).rename(columns={'win': 'WinRate', 'pnl': 'TotalPNL'})
        print(perf)

    # 3. The "Near Miss" Gallery (High Score B-Grades)
    print("\n[THE NEAR MISS GALLERY (Top B-Grades)]")
    near_misses = df[df['grade'] == 'B'].sort_values('score', ascending=False).head(5)
    for _, row in near_misses.iterrows():
        print(f" - {row['entry_date']}: {row['symbol']} scored {row['score']:.1f}/10. Missing: {row.get('missing', 'N/A')}")

    # 4. Identifying Structural Friction
    print("\n[STRUCTURAL FRICTION: Top Deal Breakers]")
    all_missing = []
    for m in df['missing'].dropna():
        # Handle string representation of lists
        items = eval(m) if m.startswith('[') else [m]
        all_missing.extend(items)
    
    if all_missing:
        friction = pd.Series(all_missing).value_counts()
        for rule, count in friction.items():
            print(f" - {rule}: Blocked {count} signals")

if __name__ == "__main__":
    run_audit()
