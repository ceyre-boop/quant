"""sovereign/layer1/meta_label_builder.py — meta-labels for the Layer-1 secondary model (HYP-064).

Meta-labeling (Lopez de Prado): the carry edge (primary) fixes direction; the secondary model
predicts whether each carry signal becomes a WINNING trade under the exit machine. The label is the
exit machine's actual NET verdict — `meta_win = 1 if realized net pnl_pct > 0 else 0` — produced by
the CANONICAL backtester (`ForexBacktester`, strict_mode=False, the proven Sharpe-1.25 config that
`prove.py` uses). Each trade's features come from the existing 41-feature panel at its ENTRY bar.

Holdout-safe by construction: the backtester is bounded to end=2023-12-31, so no trade can enter or
exit in the 2024+ holdout. Fail-loud: trades whose entry bar has no feature row are reported, never
silently dropped or zero-filled.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FEATURES_PARQUET = ROOT / "data" / "layer1" / "features_v1.parquet"
OUT = ROOT / "data" / "layer1" / "meta_dataset_v1.parquet"
REPORT = ROOT / "data" / "layer1" / "meta_label_report.json"
SPEC = ROOT / "docs" / "layer1" / "feature_windows.json"

START, END = "2015-01-01", "2023-12-31"      # holdout 2024+ never run
PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]
CCY_TO_COUNTRY = {"EUR": "EU", "GBP": "UK", "JPY": "JP", "USD": "US",
                  "AUD": "AU", "CAD": "CA", "NZD": "NZ", "CHF": "CH"}


def _collect_trades() -> tuple[pd.DataFrame, dict]:
    from sovereign.forex.forex_backtester import ForexBacktester
    from sovereign.forex.pair_universe import PAIR_CONFIG

    bt = ForexBacktester(start=START, end=END)       # strict_mode=False (canonical/proven)
    rows, per_pair = [], {}
    for tk in PAIRS:
        cfg = PAIR_CONFIG[tk]
        base_c = CCY_TO_COUNTRY[cfg.base_currency]
        quote_c = CCY_TO_COUNTRY[cfg.quote_currency]
        try:
            _result, trades = bt.run_pair_with_trades(tk, base_c, quote_c)
        except Exception as e:  # noqa: BLE001 — fail loud
            per_pair[tk] = f"ERROR: {type(e).__name__}: {e}"
            continue
        trades = trades or []
        per_pair[tk] = len(trades)
        for t in trades:
            entry = pd.Timestamp(t["entry_date"]).normalize()
            exit_ = pd.Timestamp(t["exit_date"]).normalize()
            pnl = float(t.get("pnl_pct", 0.0))           # NET of costs (_apply_costs already ran)
            rows.append({
                "entry_date": entry, "exit_date": exit_,
                "pair": tk.replace("=X", ""),
                "direction": int(t.get("direction", 0)),
                "realized_r": pnl,
                "meta_win": 1 if pnl > 0 else 0,
                "exit_reason": t.get("exit_reason"),
                "hold_days": int(t.get("hold_days", 0)),
            })
    return pd.DataFrame(rows), per_pair


def _declared_features() -> list[str]:
    return [f["name"] for f in json.loads(SPEC.read_text())["features"]]


def build() -> dict:
    trades_df, per_pair = _collect_trades()
    if trades_df.empty:
        REPORT.write_text(json.dumps({"error": "no trades produced", "per_pair": per_pair}, indent=2, default=str))
        raise SystemExit("FATAL: backtester produced no trades — see meta_label_report.json")

    feats = pd.read_parquet(FEATURES_PARQUET).rename_axis(["date", "pair"]).reset_index()
    declared = [c for c in _declared_features() if c in feats.columns]

    merged = trades_df.merge(
        feats, left_on=["entry_date", "pair"], right_on=["date", "pair"], how="left",
    )
    # Fail-loud: a trade whose entry bar has no feature row (all declared features NaN).
    unmatched_mask = merged[declared].isna().all(axis=1)
    n_unmatched = int(unmatched_mask.sum())
    matched = merged.loc[~unmatched_mask].copy()

    # Holdout guard (belt + suspenders).
    max_date = pd.concat([matched["entry_date"], matched["exit_date"]]).max()
    assert max_date <= pd.Timestamp(END), f"HOLDOUT LEAK: max date {max_date} > {END}"

    label_aux = ["meta_win", "realized_r", "exit_reason", "direction", "exit_date", "hold_days"]
    keep_cols = declared + label_aux
    out = matched.set_index(["entry_date", "pair"])[keep_cols].sort_index()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)

    exit_mix = matched["exit_reason"].value_counts().to_dict()
    report = {
        "n_events": int(len(out)),
        "win_rate": round(float(out["meta_win"].mean()), 4),
        "per_pair_trade_count": per_pair,
        "exit_reason_mix": exit_mix,
        "mean_realized_r": round(float(out["realized_r"].mean()), 5),
        "n_unmatched_entries_dropped_loud": n_unmatched,
        "ablation_target_features_present": [
            f for f in ["day_of_week", "pair_real_rate_diff", "pair_rate_diff_mom_1m",
                        "pair_rate_diff_mom_3m", "pair_rate_diff_mom_6m"] if f in declared],
        "n_features": len(declared),
        "date_range": [str(out.index.get_level_values(0).min().date()),
                       str(out.index.get_level_values(0).max().date())],
        "holdout_fetched": False,
        "strict_mode": False,
        "label": "meta_win = net pnl_pct > 0 under ForexBacktester (canonical, strict_mode=False)",
    }
    REPORT.write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    r = build()
    print("=" * 70)
    print("META-DATASET v1 BUILT")
    print(f"  events (carry trades): {r['n_events']}")
    print(f"  win rate             : {r['win_rate']}")
    print(f"  mean realized R      : {r['mean_realized_r']}")
    print(f"  per-pair counts      : {r['per_pair_trade_count']}")
    print(f"  exit-reason mix      : {r['exit_reason_mix']}")
    print(f"  features             : {r['n_features']} | ablation targets present: {r['ablation_target_features_present']}")
    print(f"  unmatched entries    : {r['n_unmatched_entries_dropped_loud']} (reported, not zero-filled)")
    print(f"  date range           : {r['date_range'][0]} .. {r['date_range'][1]} (holdout untouched)")
    print("=" * 70)
