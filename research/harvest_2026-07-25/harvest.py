#!/usr/bin/env python3
"""Feature-data harvest 2026-07-25 — real data only, dated, no fabrication.
Outputs: policy_rates/, indicators/, cot/, volstructure/ under this dir."""
import io, json, os, sys, time, zipfile, urllib.request
from pathlib import Path
import pandas as pd, numpy as np

HERE = Path(__file__).parent
ENV = {}
for line in (Path.home()/"quant"/".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1); ENV[k.strip()] = v.strip().strip('"')

def fred(series, start="2015-01-01"):
    url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={series}"
           f"&api_key={ENV['FRED_API_KEY']}&file_type=json&observation_start={start}")
    obs = json.loads(urllib.request.urlopen(url, timeout=60).read())["observations"]
    s = pd.Series({o["date"]: float(o["value"]) for o in obs if o["value"] != "."}, name=series)
    s.index = pd.to_datetime(s.index); return s

# 1 ── policy rates + differentials
d = HERE/"policy_rates"; d.mkdir(exist_ok=True)
series = {"US":"DFF", "EU":"ECBDFR", "UK":"IUDSOIA", "AU":"IR3TIB01AUM156N", "JP":"IRSTCI01JPM156N"}
cols = {}
for cc, sid in series.items():
    try:
        cols[cc] = fred(sid); print(f"policy {cc}({sid}): {len(cols[cc])} obs, last {cols[cc].index[-1].date()} = {cols[cc].iloc[-1]}")
    except Exception as e: print(f"policy {cc}({sid}) FAILED: {e}")
pr = pd.DataFrame(cols).resample("D").last().ffill()
pr.to_csv(d/"policy_rates_daily.csv")
diffs = pd.DataFrame({"UK_US": pr["UK"]-pr["US"], "EU_US": pr["EU"]-pr["US"],
                      "AU_US": pr["AU"]-pr["US"], "US_JP": pr["US"]-pr["JP"]})
diffs.to_csv(d/"differentials_daily.csv")
print("latest differentials:", diffs.iloc[-1].round(3).to_dict())

# 2 ── indicator panel (ATR/RSI etc — derived features, not price action)
import yfinance as yf
d = HERE/"indicators"; d.mkdir(exist_ok=True); (d/"raw").mkdir(exist_ok=True)
def rsi(close, n=14):
    delta = close.diff(); up = delta.clip(lower=0).ewm(alpha=1/n).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/n).mean(); return 100 - 100/(1 + up/dn)
rows = []
for tkr in ["GBPUSD=X","EURUSD=X","AUDUSD=X","GBPJPY=X","USDJPY=X","ES=F","NQ=F","SPY","QQQ","^VIX"]:
    try:
        df = yf.download(tkr, start="2015-01-01", progress=False, auto_adjust=True)
        if df.empty: print(f"ind {tkr}: EMPTY"); continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.to_csv(d/"raw"/f"{tkr.replace('=','_').replace('^','')}.csv")
        c, h, l = df["Close"], df["High"], df["Low"]
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        out = pd.DataFrame({
            "atr14": tr.rolling(14).mean(), "atr14_pct": tr.rolling(14).mean()/c*100,
            "rsi14": rsi(c), "realized_vol_20d": c.pct_change().rolling(20).std()*np.sqrt(252)*100,
            "dist_52w_high_pct": (c/c.rolling(252).max()-1)*100,
            "dist_52w_low_pct": (c/c.rolling(252).min()-1)*100,
            "ret_5d": c.pct_change(5)*100, "ret_20d": c.pct_change(20)*100, "ret_60d": c.pct_change(60)*100,
            "zscore_20d": (c-c.rolling(20).mean())/c.rolling(20).std(),
        })
        if "Volume" in df and df["Volume"].sum() > 0:
            out["volume_ratio_20d"] = df["Volume"]/df["Volume"].rolling(20).mean()
        out["instrument"] = tkr; rows.append(out.reset_index().rename(columns={"Date":"date"}))
        print(f"ind {tkr}: {len(out)} rows, last {out.index[-1].date()}")
        time.sleep(1)
    except Exception as e: print(f"ind {tkr} FAILED: {e}")
if rows: pd.concat(rows).to_csv(d/"indicator_panel.csv", index=False)

# 3 ── CFTC COT
d = HERE/"cot"; d.mkdir(exist_ok=True)
MKTS = ["BRITISH POUND","EURO FX","AUSTRALIAN DOLLAR","JAPANESE YEN","USD INDEX",
        "E-MINI S&P 500","NASDAQ MINI","MICRO E-MINI NASDAQ"]
frames = []
for yr in range(2020, 2027):
    try:
        raw = urllib.request.urlopen(f"https://www.cftc.gov/files/dea/history/deacot{yr}.zip", timeout=120).read()
        z = zipfile.ZipFile(io.BytesIO(raw)); name = z.namelist()[0]
        df = pd.read_csv(z.open(name), low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        mcol = df.columns[0]
        sel = df[df[mcol].str.upper().str.contains("|".join(MKTS), na=False)]
        frames.append(sel); print(f"cot {yr}: {len(sel)} rows")
    except Exception as e: print(f"cot {yr} FAILED: {e}")
if frames:
    cot = pd.concat(frames)
    keep = {}
    for c in cot.columns:
        cl = c.lower()
        if "market" in cl and "name" in cl: keep[c] = "market"
        elif cl.startswith("as of date in form yymmdd") or "report_date" in cl: pass
        elif "as of date" in cl and "yyyy" in cl: keep[c] = "report_date"
        elif cl == "noncommercial positions-long (all)": keep[c] = "noncomm_long"
        elif cl == "noncommercial positions-short (all)": keep[c] = "noncomm_short"
        elif cl == "open interest (all)": keep[c] = "open_interest"
    slim = cot[list(keep)].rename(columns=keep)
    slim["net_spec"] = slim["noncomm_long"] - slim["noncomm_short"]
    slim["net_spec_pct_oi"] = slim["net_spec"]/slim["open_interest"]*100
    slim["report_date"] = pd.to_datetime(slim["report_date"])
    slim["published_date"] = slim["report_date"] + pd.Timedelta(days=3)  # Tue as-of, Fri publish
    slim.sort_values(["market","report_date"]).to_csv(d/"cot_fx_panel.csv", index=False)
    print("cot latest per market:")
    print(slim.sort_values("report_date").groupby("market").tail(1)[["market","report_date","net_spec"]].to_string(index=False))

# 4 ── vol term structure
d = HERE/"volstructure"; d.mkdir(exist_ok=True)
vols = {}
for tkr in ["^VIX","^VIX9D","^VIX3M","^VVIX","^MOVE"]:
    try:
        df = yf.download(tkr, start="2015-01-01", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if not df.empty: vols[tkr.strip('^')] = df["Close"]; print(f"vol {tkr}: {len(df)} rows")
        time.sleep(1)
    except Exception as e: print(f"vol {tkr} FAILED: {e}")
if vols:
    v = pd.DataFrame(vols)
    if {"VIX9D","VIX"} <= set(v): v["vix9d_vix"] = v["VIX9D"]/v["VIX"]
    if {"VIX","VIX3M"} <= set(v):
        v["vix_vix3m"] = v["VIX"]/v["VIX3M"]; v["backwardation"] = (v["vix_vix3m"] > 1).astype(int)
    v.to_csv(d/"vol_term_structure.csv"); print("vol latest:", v.iloc[-1].round(3).to_dict())
print("HARVEST DONE")
