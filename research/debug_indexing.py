import pandas as pd
import yfinance as yf
from training.engine_v4 import build_v4_features

df = yf.download("NVDA", period="1mo", interval="1h")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [c.lower() for c in df.columns]

f_mom = build_v4_features(df)

print(f"DF Index Type: {type(df.index)}")
print(f"F_MOM Index Type: {type(f_mom.index)}")
print(f"DF Index sample: {df.index[0]}")
print(f"F_MOM Index sample: {f_mom.index[0]}")
