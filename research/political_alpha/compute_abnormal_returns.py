"""Phase 2 — abnormal-return event-study calculator (HYP-085, spec §7-P2).

Pinned formulas (locked in the HYP-085 prereg; plan §Pinned formulas (a)/(b)):
  - daily LOG returns
  - estimation window = return positions [pos(T0)-252, pos(T0)-10] on the
    instrument's own calendar; mean-adjusted model: est_mu, est_sigma (ddof=1);
    >=200 non-NaN required else data_ok:false
  - post window (T+0 -> T+24h, daily default): post_return = r_T0 + r_T1
    (= ln(Close_T1 / Close_P0)); abnormal_return = post_return - 2*est_mu;
    std_abnormal_return = abnormal_return / (est_sigma * sqrt(2))
  - big_move = |r_T0| > 2*sigma60(T0) OR |r_T1| > 2*sigma60(T1), where sigma60 is
    the trailing 60-day rolling SD shifted one day (never includes the tested day)

NO SILENT MOCKING (spec §8): missing/short data -> data_ok:false + gap_reason,
the row is emitted and skipped downstream — never backfilled or fabricated.

Run:  python3 research/political_alpha/compute_abnormal_returns.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402


def study_row(event: dict, px: pd.DataFrame) -> dict:
    out = {
        "event_id": event["event_id"],
        "instrument_tagged": event["instrument_tagged"],
        "timestamp_utc": event["timestamp_utc"],
        "t0": None,
        "est_mu": None, "est_sigma": None, "n_est_days": 0,
        "post_return": None, "abnormal_return": None, "std_abnormal_return": None,
        "big_move": None, "direction": None,
        "data_ok": False, "gap_reason": "",
    }
    if px is None or px.empty:
        out["gap_reason"] = "no_price_data"
        return out

    idx = px.index
    r = _lib.log_returns(px["Close"])
    sig60 = _lib.trailing_sigma60(r)

    t0 = _lib.map_t0(event["timestamp_utc"], idx, _lib.ASSET_CLASS[event["instrument_tagged"]])
    if t0 is None:
        out["gap_reason"] = "post_window_beyond_data"
        return out
    pos = idx.get_loc(t0)
    out["t0"] = str(pd.Timestamp(t0).date())
    if pos + 1 >= len(idx):
        out["gap_reason"] = "post_window_beyond_data"
        return out
    if pos - 252 < 1:                       # r[0] is NaN by construction
        out["gap_reason"] = "estimation_window_short"
        return out

    est = r.iloc[pos - 252: pos - 10 + 1].dropna()
    out["n_est_days"] = int(len(est))
    if len(est) < 200:
        out["gap_reason"] = "estimation_window_short"
        return out
    est_mu, est_sigma = float(est.mean()), float(est.std(ddof=1))
    if not (np.isfinite(est_sigma) and est_sigma > 0):
        out["gap_reason"] = "estimation_sigma_degenerate"
        return out

    r_t0, r_t1 = float(r.iloc[pos]), float(r.iloc[pos + 1])
    if not (np.isfinite(r_t0) and np.isfinite(r_t1)):
        out["gap_reason"] = "post_window_return_nan"
        return out
    post = r_t0 + r_t1
    abnormal = post - 2.0 * est_mu
    std_abn = abnormal / (est_sigma * math.sqrt(2.0))

    s0, s1 = float(sig60.iloc[pos]), float(sig60.iloc[pos + 1])
    if np.isfinite(s0) and np.isfinite(s1) and s0 > 0 and s1 > 0:
        big = bool(abs(r_t0) > 2.0 * s0 or abs(r_t1) > 2.0 * s1)
    else:
        big = None
        out["gap_reason"] = "sigma60_warmup"

    out.update({
        "est_mu": round(est_mu, 8), "est_sigma": round(est_sigma, 8),
        "post_return": round(post, 8), "abnormal_return": round(abnormal, 8),
        "std_abnormal_return": round(std_abn, 6),
        "big_move": big,
        "direction": "up" if post > 0 else ("down" if post < 0 else "flat"),
        "data_ok": big is not None,
    })
    return out


def main() -> int:
    events = _lib.read_jsonl(_lib.DATA_DIR / "trump_events.jsonl")
    if not events:
        print("STOP: data/trump_events.jsonl is empty — run Phase 1 first.")
        return 2
    print(f"Phase 2 — abnormal returns for {len(events)} event rows")

    px = {t: _lib.fetch_daily(t) for t in _lib.UNIVERSE}
    rows = [study_row(e, px.get(e["instrument_tagged"])) for e in events]
    _lib.write_jsonl(_lib.DATA_DIR / "event_study_results.jsonl", rows)

    ok = [r for r in rows if r["data_ok"]]
    big = [r for r in ok if r["big_move"]]
    gaps = pd.Series([r["gap_reason"] for r in rows if r["gap_reason"]]).value_counts().to_dict()
    print(f"  rows: {len(rows)}   evaluable: {len(ok)}   big_move: {len(big)} "
          f"({(len(big) / len(ok) * 100) if ok else 0:.1f}% of evaluable)")
    print(f"  gaps: {gaps if gaps else 'none'}")
    if ok:
        sar = pd.Series([r["std_abnormal_return"] for r in ok])
        print(f"  std_abnormal_return: mean {sar.mean():+.3f}  median {sar.median():+.3f}  "
              f"|>2|: {(sar.abs() > 2).sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
