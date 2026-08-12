"""Shared library for the Political-Alpha V2 event study (HYP-087 Track A + HYP-088
Track B; TICK-021).

SELF-CONTAINED BY MANDATE (spec §4, HARD): this module imports NOTHING from
sovereign/, ict/, ict-engine/, config/, audit/, or scripts/. AST-enforced by
tests/test_isolation.py. Where the V1 module already solved a problem the code is
COPIED here with a provenance comment, never imported:

  - env parse / jsonl IO / yfinance fetch / T0 mapping / SAR primitives
        <- research/political_alpha/_lib.py (V1, HYP-085)

V2 differences from V1:
  - instrument universe = 9 ETFs/index (no FX pairs): spec §2
  - primary metric = CSAR over [0,+72h] = SAR(T+0)+SAR(T+1)+SAR(T+2), where
    SAR_t = (r_t - mu)/sigma, mu/sigma from the T-252..T-10 estimation window
    (spec §2 "Abnormal return (SAR)"). Pre-window = SAR(T-2)+SAR(T-1).
  - keyword clustering on statement_text per config/cluster_rules.json (spec §2 A)

Governing spec: ~/Obsidian/Obsidian/Trading/Research/Political-Alpha-V2-Claude-Code-Spec.md
Data caches under research/political_alpha_v2/data/{raw,cache}/ (gitignored).
NO SILENT MOCKING (spec §8): missing/short data -> data_ok:false + gap_reason.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ── paths & constants ────────────────────────────────────────────────────────────────

MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT.parents[1]
CONFIG_DIR = MODULE_ROOT / "config"
DATA_DIR = MODULE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = MODULE_ROOT / "output"

# V1 catalog is the READ-ONLY event spine (spec §3, §4 — read, never import/modify).
V1_EVENTS = REPO_ROOT / "research" / "political_alpha" / "data" / "trump_events.jsonl"

STUDY_START = "2025-01-20"          # inauguration — locked spec §0
ET = ZoneInfo("America/New_York")

# Locked instrument universe (spec §2) — 9 ETFs/index, ALL via yfinance.
UNIVERSE = ["XLE", "XLF", "XLV", "XLI", "KWEB", "SLX", "GLD", "TLT", "DX-Y.NYB"]
# Asset class for the T0 mapping rule (spec §2 "Abnormal return"). DX-Y.NYB (ICE
# dollar index) trades a near-24h session -> fx-style mapping; the ETFs are us_etf.
ASSET_CLASS = {t: "us_etf" for t in UNIVERSE}
ASSET_CLASS["DX-Y.NYB"] = "fx"

# yfinance fetch start — >=1yr before the first event (2025-01-20) so the full
# 242-bar estimation window (T-252..T-10) is available for every event.
FETCH_START = "2023-06-01"

# Estimation / event windows (spec §2).
EST_LOOKBACK = 252
EST_GAP = 10                        # window is r[pos-252 : pos-10]
EST_MIN_OBS = 100                   # spec §Phase-2: estimation window < 100 -> data_ok:false
POST_BARS = (0, 1, 2)               # [0,+72h] -> T+0,T+1,T+2 daily bars
PRE_BARS = (-2, -1)                 # [-48h,0] -> T-2,T-1 daily bars
BIG_MOVE_SIGMA = 2.0                # |CSAR_72h| > 2 sigma-cumulative = "big move" (descriptive)


# ── env (copied from V1 _lib.py::env_key) ────────────────────────────────────────────

def env_key(name: str, default: str | None = None) -> str | None:
    """os.environ first, else manual parse of REPO_ROOT/.env (no python-dotenv).
    Track A/B need no secrets; kept for parity / any optional probe."""
    import os
    val = os.environ.get(name)
    if val:
        return val
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == name:
                return v.strip().strip('"').strip("'")
    return default


# ── jsonl / json IO (copied from V1 _lib.py) ─────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=False) + "\n" for r in rows))
    tmp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json_pretty(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── V1 event spine loader (read-only; dedup to unique events) ─────────────────────────

def load_v1_events() -> list[dict]:
    """Load the V1 catalog, dedup to unique event_ids (V1 has 223 rows / 168 unique
    events — the same event appears once per V1 instrument tag; V2 re-derives the
    instrument set from the cluster, so we keep one row per event_id). Returns rows
    sorted by timestamp with fields: event_id, timestamp_utc, statement_text, source."""
    rows = read_jsonl(V1_EVENTS)
    seen, out = set(), []
    for r in rows:
        if r["event_id"] in seen:
            continue
        seen.add(r["event_id"])
        out.append({
            "event_id": r["event_id"],
            "timestamp_utc": r["timestamp_utc"],
            "statement_text": r.get("statement_text", ""),
            "source": r.get("source", ""),
        })
    out.sort(key=lambda e: e["timestamp_utc"])
    return out


# ── keyword clustering (spec §2 A, config-driven) ────────────────────────────────────

def load_cluster_rules() -> dict:
    return read_json(CONFIG_DIR / "cluster_rules.json")


def _contains_any(text: str, terms: list[str]) -> list[str]:
    """Case-insensitive substring hits (config match.mode = substring)."""
    return [t for t in terms if t.lower() in text]


def classify_event(statement_text: str, rules: dict) -> dict:
    """Assign a statement to at most one cluster by the priority hierarchy. First
    cluster in priority_order whose rule matches wins. Returns cluster (or None),
    matched_keywords, instruments, and confidence (high = >=1 word-boundary match,
    low = only mid-word substring matches). Spec §2 / §Phase-1."""
    text = re.sub(r"\s+", " ", (statement_text or "")).lower()
    clusters = rules["clusters"]
    for name in rules["priority_order"]:
        c = clusters[name]
        req = _contains_any(text, c.get("required_any", []))
        if not req:
            continue
        sec_terms = c.get("required_any_secondary", [])
        if sec_terms and not _contains_any(text, sec_terms):
            continue
        if _contains_any(text, c.get("exclude_if", [])):
            continue
        matched = req + _contains_any(text, sec_terms)
        # confidence: high if any matched term appears at a word boundary
        conf = "low"
        for kw in matched:
            if re.search(r"\b" + re.escape(kw.strip().lower()) + r"\b", text):
                conf = "high"
                break
        return {"cluster": name, "matched_keywords": matched,
                "instruments": c.get("instruments", []), "confidence": conf}
    return {"cluster": None, "matched_keywords": [], "instruments": [], "confidence": "none"}


# ── daily OHLCV via yfinance (copied from V1 _lib.py::fetch_daily) ────────────────────

def fetch_daily(ticker: str, start: str = FETCH_START, refresh: bool = False) -> pd.DataFrame:
    """Daily auto-adjusted OHLCV, tz-naive normalized index. Cached to CSV; a cache
    written before today refreshes automatically. Empty frame on total failure
    (callers MUST treat empty as data_ok:false — never fabricate)."""
    cache = CACHE_DIR / "yf" / f"{ticker.replace('=', '_').replace('^', '_')}.csv"
    if cache.exists() and not refresh:
        mtime = datetime.fromtimestamp(cache.stat().st_mtime, tz=timezone.utc)
        if mtime.date() >= datetime.now(timezone.utc).date():
            return pd.read_csv(cache, index_col=0, parse_dates=True)
    try:
        import yfinance as yf                        # lazy import (repo convention)
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):    # single-symbol MultiIndex gotcha
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
        return df
    except Exception as exc:
        print(f"  [yf] {ticker}: FETCH FAILED ({type(exc).__name__}: {exc})")
        if cache.exists():
            print(f"  [yf] {ticker}: using stale cache {cache.name} (flagged)")
            return pd.read_csv(cache, index_col=0, parse_dates=True)
        return pd.DataFrame()


# ── event-window mechanics (copied/adapted from V1 _lib.py) ───────────────────────────

def parse_ts(ts_utc: str) -> datetime:
    dt = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc)


def map_t0(ts_utc: str, index: pd.DatetimeIndex, asset_class: str) -> pd.Timestamp | None:
    """Statement timestamp -> T0 bar date on the instrument's own calendar (spec §2).
    fx:     first bar date >= the statement's UTC date (near-24h market).
    us_etf: statement's ET date if it is a session day AND ET time < 16:00 (close);
            otherwise the next session date."""
    if index is None or len(index) == 0:
        return None
    dt = parse_ts(ts_utc)
    if asset_class == "fx":
        day = pd.Timestamp(dt.date())
        pos = index.searchsorted(day)
    else:
        local = dt.astimezone(ET)
        day = pd.Timestamp(local.date())
        if day in index and local.hour < 16:
            pos = index.get_loc(day)
        else:
            pos = index.searchsorted(day + pd.Timedelta(days=1))
    if pos >= len(index):
        return None
    return index[pos]


def log_returns(close: pd.Series) -> pd.Series:
    close = close.astype(float)
    return np.log(close / close.shift(1))


def sar_windows(ts_utc: str, instrument: str, px: pd.DataFrame) -> dict:
    """Compute SAR-based CSAR windows for one event x instrument (spec §2, §Phase-2).

    estimation window: r[pos-252 : pos-10] (inclusive of pos-10), mean-adjusted
      mu/sigma (ddof=1); require >= EST_MIN_OBS non-NaN else data_ok:false.
    SAR_t = (r_t - mu) / sigma.
    csar_72h  = SAR(T+0) + SAR(T+1) + SAR(T+2).
    pre_csar_48h = SAR(T-2) + SAR(T-1).
    big_move = |csar_72h| > 2 (>=2 sigma cumulative standardized; descriptive flag).
    NO SILENT MOCKING: any gap -> data_ok:false + gap_reason; row emitted, never filled.
    """
    out = {
        "instrument": instrument, "t0": None,
        "est_mu": None, "est_sigma": None, "n_est_days": 0,
        "csar_72h": None, "pre_csar_48h": None,
        "big_move": None, "direction": None,
        "data_ok": False, "gap_reason": "",
    }
    if px is None or px.empty:
        out["gap_reason"] = "no_price_data"
        return out

    idx = px.index
    r = log_returns(px["Close"])
    t0 = map_t0(ts_utc, idx, ASSET_CLASS[instrument])
    if t0 is None:
        out["gap_reason"] = "t0_beyond_data"
        return out
    pos = idx.get_loc(t0)
    out["t0"] = str(pd.Timestamp(t0).date())

    if pos + max(POST_BARS) >= len(idx):
        out["gap_reason"] = "post_window_beyond_data"
        return out
    if pos + min(PRE_BARS) < 1:                     # r[0] is NaN by construction
        out["gap_reason"] = "pre_window_before_data"
        return out
    if pos - EST_LOOKBACK < 1:
        out["gap_reason"] = "estimation_window_short"
        return out

    est = r.iloc[pos - EST_LOOKBACK: pos - EST_GAP + 1].dropna()
    out["n_est_days"] = int(len(est))
    if len(est) < EST_MIN_OBS:
        out["gap_reason"] = "estimation_window_short"
        return out
    mu, sigma = float(est.mean()), float(est.std(ddof=1))
    if not (np.isfinite(sigma) and sigma > 0):
        out["gap_reason"] = "estimation_sigma_degenerate"
        return out

    def sar(offset: int) -> float:
        return (float(r.iloc[pos + offset]) - mu) / sigma

    post_sars = [sar(o) for o in POST_BARS]
    pre_sars = [sar(o) for o in PRE_BARS]
    if not all(np.isfinite(x) for x in post_sars + pre_sars):
        out["gap_reason"] = "window_return_nan"
        return out

    csar = float(np.sum(post_sars))
    pre_csar = float(np.sum(pre_sars))
    out.update({
        "est_mu": round(mu, 8), "est_sigma": round(sigma, 8),
        "csar_72h": round(csar, 6), "pre_csar_48h": round(pre_csar, 6),
        "big_move": bool(abs(csar) > BIG_MOVE_SIGMA),
        "direction": "up" if csar > 0 else ("down" if csar < 0 else "flat"),
        "data_ok": True,
    })
    return out
