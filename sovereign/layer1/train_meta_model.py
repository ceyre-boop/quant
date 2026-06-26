"""sovereign/layer1/train_meta_model.py — Phase 3: train + validate the meta-model (HYP-064).

Meta-labeling secondary model: XGBoost predicts P(carry trade wins) from the 41-feature panel at
entry. Runs the FULL pre-registered pipeline (data/research/preregister/HYP-064_..., hash d22ae88f):
12-config grid → val ROC-AUC → permutation null → Benjamini-Hochberg → UNCONDITIONAL ablation →
economic bar (R-multiple lift, retention, bootstrap Sharpe-diff) → verdict vs BOTH bars.

HARD RULES: holdout (2024+) is NEVER loaded. No commit. No live integration. Reports + ledger only.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score, brier_score_loss          # noqa: E402
from sklearn.calibration import CalibratedClassifierCV               # noqa: E402
import xgboost as xgb                                                # noqa: E402

from sovereign.discovery.gate import benjamini_hochberg, deflated_sharpe_ratio, bootstrap_sharpe_diff_pvalue  # noqa: E402

DATA = ROOT / "data" / "layer1" / "meta_dataset_v1.parquet"
REPORT_MD = ROOT / "docs" / "layer1" / "VALIDATION_HYP-064.md"
REPORT_JSON = ROOT / "data" / "layer1" / "validation_HYP-064.json"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
SPEC = ROOT / "docs" / "layer1" / "feature_windows.json"

AUX = {"meta_win", "realized_r", "exit_reason", "direction", "exit_date", "hold_days"}
ABLATION_DROP = ["day_of_week", "pair_real_rate_diff", "pair_rate_diff_mom_1m",
                 "pair_rate_diff_mom_3m", "pair_rate_diff_mom_6m"]
TRAIN_END = pd.Timestamp("2021-12-31")
VAL_END = pd.Timestamp("2023-12-31")
GRID = list(itertools.product([2, 3, 4], [200, 400], [0.03, 0.1]))   # 12 configs
STOP_ATR_MULT = 2.0
PAIR_YF = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
           "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X"}


# ─── R-multiple reconstruction (R = pnl_pct / (2 * ATR%@entry)) ──────────────────

def _atr_pct_by_pair() -> dict:
    import yfinance as yf
    out = {}
    for pair, tk in PAIR_YF.items():
        df = yf.download(tk, start="2015-01-01", end="2024-01-01", auto_adjust=True, progress=False)
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        high, low, close = df["High"], df["Low"], df["Close"]
        prev = close.shift(1)
        tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        out[pair] = (atr / close).rename("atr_pct")
    return out


def _r_multiples(idx, realized_r: pd.Series, atr_by_pair: dict) -> pd.Series:
    r = []
    for (entry_date, pair), pnl in zip(idx, realized_r):
        s = atr_by_pair.get(pair)
        a = float(s.reindex([pd.Timestamp(entry_date)]).iloc[0]) if s is not None else np.nan
        r.append(pnl / (STOP_ATR_MULT * a) if a and not np.isnan(a) and a > 0 else np.nan)
    return pd.Series(r, index=idx)


# ─── stats helpers ──────────────────────────────────────────────────────────────

def _perm_p_auc(y_true: np.ndarray, score: np.ndarray, n: int = 1000, seed: int = 11) -> float:
    obs = roc_auc_score(y_true, score)
    rng = np.random.default_rng(seed)
    ge = sum(roc_auc_score(rng.permutation(y_true), score) >= obs for _ in range(n))
    return (ge + 1) / (n + 1)


def _sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1)) if len(x) > 1 and x.std(ddof=1) > 0 else 0.0


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(e)


def _fit_predict(Xtr, ytr, Xva, depth, n_est, lr):
    m = xgb.XGBClassifier(max_depth=depth, n_estimators=n_est, learning_rate=lr,
                          subsample=0.8, eval_metric="logloss", verbosity=0)
    m.fit(Xtr, ytr)
    return m, m.predict_proba(Xva)[:, 1]


def _gate_threshold(p_tr, r_tr):
    """Tune τ on TRAIN only: maximize gated mean-R subject to retention ≥ 50%."""
    best_tau, best_r = 0.0, -np.inf
    for q in np.quantile(p_tr, np.linspace(0.0, 0.5, 26)):   # never gate out > 50% on train
        keep = p_tr >= q
        if keep.mean() >= 0.50 and np.isfinite(r_tr[keep]).sum() > 5:
            mr = np.nanmean(r_tr[keep])
            if mr > best_r:
                best_r, best_tau = mr, q
    return best_tau


def main() -> dict:
    df = pd.read_parquet(DATA)
    assert df.index.get_level_values(0).max() <= VAL_END, "HOLDOUT LEAK: dataset has 2024+ rows"
    declared = [f["name"] for f in json.loads(SPEC.read_text())["features"] if f["name"] in df.columns]

    ent = df.index.get_level_values(0)
    tr_mask, va_mask = ent <= TRAIN_END, (ent > TRAIN_END) & (ent <= VAL_END)
    ytr, yva = df.loc[tr_mask, "meta_win"].to_numpy(int), df.loc[va_mask, "meta_win"].to_numpy(int)

    atr_by_pair = _atr_pct_by_pair()
    R = _r_multiples(df.index, df["realized_r"], atr_by_pair)
    r_tr, r_va = R[tr_mask].to_numpy(float), R[va_mask].to_numpy(float)

    def run_grid(features):
        Xtr, Xva = df.loc[tr_mask, features], df.loc[va_mask, features]
        rows, best = [], None
        for depth, n_est, lr in GRID:
            m, pva = _fit_predict(Xtr, ytr, Xva, depth, n_est, lr)
            auc = roc_auc_score(yva, pva)
            pp = _perm_p_auc(yva, pva)
            rows.append({"max_depth": depth, "n_estimators": n_est, "learning_rate": lr,
                         "val_auc": round(float(auc), 4), "perm_p": round(float(pp), 4)})
            if best is None or auc > best["val_auc_raw"]:
                best = {"model": m, "pva": pva, "ptr": m.predict_proba(Xtr)[:, 1],
                        "val_auc_raw": auc, "cfg": (depth, n_est, lr)}
        bh = benjamini_hochberg([r["perm_p"] for r in rows], alpha=0.05)
        for r, surv in zip(rows, bh):
            r["bh_survives"] = bool(surv)
        return rows, best

    # ── main run (all features) ──
    rows, best = run_grid(declared)
    tau = _gate_threshold(best["ptr"], r_tr)
    keep_va = best["pva"] >= tau
    gated_r = r_va[keep_va & np.isfinite(r_va)]
    ungated_r = r_va[np.isfinite(r_va)]
    retention = float(keep_va.mean())
    mean_r_delta = float(np.nanmean(gated_r) - np.nanmean(ungated_r)) if len(gated_r) else float("nan")
    boot_p = bootstrap_sharpe_diff_pvalue(gated_r, ungated_r) if len(gated_r) > 5 else 1.0
    gated_sharpe_full = _sharpe(gated_r)

    # ── unconditional ablation ──
    abl_features = [f for f in declared if f not in ABLATION_DROP]
    _abl_rows, abl_best = run_grid(abl_features)
    abl_keep = abl_best["pva"] >= _gate_threshold(abl_best["ptr"], r_tr)
    abl_gated_sharpe = _sharpe(r_va[abl_keep & np.isfinite(r_va)])
    abl_drop_pct = (round((gated_sharpe_full - abl_gated_sharpe) / abs(gated_sharpe_full) * 100, 1)
                    if gated_sharpe_full else None)
    if abl_drop_pct is None:
        abl_band = "UNDEFINED (full gated Sharpe ~0)"
    elif abl_drop_pct < 5:
        abl_band = "REDUNDANT with carry (<5% drop) — carry-overlap features add ~nothing independent"
    elif abl_drop_pct > 20:
        abl_band = "DEPENDENT-but-real (>20% drop) — overlap features do real work"
    else:
        abl_band = "AMBIGUOUS (5-20% drop) — FLAG for investigation, not a clean pass"

    # ── calibration + SHAP (best config) ──
    Xtr_b, Xva_b = df.loc[tr_mask, declared], df.loc[va_mask, declared]
    cal = CalibratedClassifierCV(xgb.XGBClassifier(max_depth=best["cfg"][0], n_estimators=best["cfg"][1],
                                                   learning_rate=best["cfg"][2], subsample=0.8,
                                                   eval_metric="logloss", verbosity=0),
                                 cv=3, method="isotonic")
    cal.fit(Xtr_b, ytr)
    p_cal = cal.predict_proba(Xva_b)[:, 1]
    brier = float(brier_score_loss(yva, p_cal))
    ece = _ece(yva, p_cal)
    contribs = best["model"].get_booster().predict(xgb.DMatrix(Xva_b), pred_contribs=True)
    imp = np.abs(contribs[:, :-1]).mean(axis=0)
    shap_top5 = sorted(zip(declared, imp), key=lambda t: -t[1])[:5]

    # ── verdict (both bars) ──
    best_auc = best["val_auc_raw"]
    best_perm = min(r["perm_p"] for r in rows)
    any_bh = any(r["bh_survives"] for r in rows)
    dsr, _ = deflated_sharpe_ratio(_sharpe(ungated_r) if len(ungated_r) else 0.0, n_trials=len(GRID))
    stat_pass = bool(best_auc >= 0.55 and any_bh and best_perm < 0.05 and dsr > 0)
    econ_pass = bool(mean_r_delta >= 0.10 and retention >= 0.50 and boot_p < 0.05)
    verdict = "VALID_EDGE" if (stat_pass and econ_pass) else "NOT_SIGNIFICANT"

    out = {
        "hypothesis": "HYP-064", "prereg_hash": "d22ae88f", "verdict": verdict,
        "n_train": int(tr_mask.sum()), "n_val": int(va_mask.sum()),
        "statistical_bar": {"pass": stat_pass, "best_val_auc": round(float(best_auc), 4),
                            "bar_auc": 0.55, "best_perm_p": round(float(best_perm), 4),
                            "any_bh_survives": any_bh, "deflated_sharpe": round(float(dsr), 4)},
        "economic_bar": {"pass": econ_pass, "mean_R_delta": round(mean_r_delta, 4), "bar_R_delta": 0.10,
                         "retention": round(retention, 3), "bar_retention": 0.50,
                         "bootstrap_sharpe_diff_p": round(float(boot_p), 4), "gate_threshold": round(float(tau), 4)},
        "ablation": {"sharpe_drop_pct": abl_drop_pct, "band": abl_band, "dropped_features": ABLATION_DROP},
        "calibration": {"brier": round(brier, 4), "ece": round(ece, 4)},
        "shap_top5": [{"feature": f, "mean_abs_contrib": round(float(v), 5),
                       "is_carry_overlap": f in ABLATION_DROP} for f, v in shap_top5],
        "config_table": rows,
        "best_config": {"max_depth": best["cfg"][0], "n_estimators": best["cfg"][1], "learning_rate": best["cfg"][2]},
        "caveat": "Labels from SIMULATED exits (strict_mode=False); retrain when L2 live. Holdout 2024+ untouched.",
    }
    REPORT_JSON.write_text(json.dumps(out, indent=2, default=str))
    _write_md(out)
    _update_ledger(out)
    return out


def _write_md(o: dict) -> None:
    t = o["config_table"]
    lines = [f"# VALIDATION — HYP-064 (meta-labeling secondary model)", "",
             f"**Verdict: {o['verdict']}**  ·  pre-reg hash `{o['prereg_hash']}`  ·  "
             f"train n={o['n_train']} / val n={o['n_val']}  ·  holdout 2024+ UNTOUCHED", "",
             "> Labels from the SIMULATED exit machine (strict_mode=False). Mandatory retrain when L2 is live.", "",
             "## Statistical bar", "",
             f"- val ROC-AUC (best): **{o['statistical_bar']['best_val_auc']}** vs bar 0.55 "
             f"→ {'PASS' if o['statistical_bar']['best_val_auc']>=0.55 else 'FAIL'}",
             f"- best permutation p: **{o['statistical_bar']['best_perm_p']}** · any BH-survives: "
             f"{o['statistical_bar']['any_bh_survives']} · deflated Sharpe: {o['statistical_bar']['deflated_sharpe']}",
             f"- **bar {'PASS' if o['statistical_bar']['pass'] else 'FAIL'}**", "",
             "## Economic bar", "",
             f"- mean-R delta (meta-gated − carry-alone): **{o['economic_bar']['mean_R_delta']}** vs bar +0.10",
             f"- retention: **{o['economic_bar']['retention']}** vs bar 0.50 · bootstrap Sharpe-diff p: "
             f"**{o['economic_bar']['bootstrap_sharpe_diff_p']}** · τ={o['economic_bar']['gate_threshold']}",
             f"- **bar {'PASS' if o['economic_bar']['pass'] else 'FAIL'}**", "",
             "## Ablation (UNCONDITIONAL — independence from carry)", "",
             f"- gated-Sharpe drop when carry-overlap features removed: **{o['ablation']['sharpe_drop_pct']}%**",
             f"- {o['ablation']['band']}", "",
             "## Calibration", "",
             f"- Brier {o['calibration']['brier']} · ECE {o['calibration']['ece']}", "",
             "## SHAP top-5 (XGBoost pred_contribs — CORRELATION, NOT CAUSATION)", ""]
    for s in o["shap_top5"]:
        flag = "  ⚠️carry-overlap" if s["is_carry_overlap"] else ""
        lines.append(f"- `{s['feature']}` — {s['mean_abs_contrib']}{flag}")
    lines += ["", "## 12-config metrics", "", "| depth | n_est | lr | val_auc | perm_p | BH |",
              "|---|---|---|---|---|---|"]
    for r in t:
        lines.append(f"| {r['max_depth']} | {r['n_estimators']} | {r['learning_rate']} | "
                     f"{r['val_auc']} | {r['perm_p']} | {'✓' if r['bh_survives'] else '·'} |")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n")


def _update_ledger(o: dict) -> None:
    led = json.loads(LEDGER.read_text())
    for e in led:
        if isinstance(e, dict) and e.get("id") == "HYP-064":
            e["status"] = o["verdict"]
            e["date_tested"] = "2026-06-26"
            e["result"] = (e.get("result", "") + f" | PHASE 3 VERDICT: {o['verdict']}. "
                           f"val AUC {o['statistical_bar']['best_val_auc']} (bar 0.55, stat "
                           f"{'PASS' if o['statistical_bar']['pass'] else 'FAIL'}); mean-R delta "
                           f"{o['economic_bar']['mean_R_delta']} retention {o['economic_bar']['retention']} "
                           f"(econ {'PASS' if o['economic_bar']['pass'] else 'FAIL'}); ablation drop "
                           f"{o['ablation']['sharpe_drop_pct']}%. SIMULATED-exit labels; holdout untouched.")
    json.dump(led, open(LEDGER, "w"), indent=2)


if __name__ == "__main__":
    o = main()
    s, ec = o["statistical_bar"], o["economic_bar"]
    print("=" * 72)
    print(f"HYP-064 PHASE 3 VERDICT: {o['verdict']}   (train {o['n_train']} / val {o['n_val']})")
    print(f"  STATISTICAL: {'PASS' if s['pass'] else 'FAIL'} — best val AUC {s['best_val_auc']} (bar .55), "
          f"perm p {s['best_perm_p']}, BH {s['any_bh_survives']}, deflated {s['deflated_sharpe']}")
    print(f"  ECONOMIC   : {'PASS' if ec['pass'] else 'FAIL'} — mean-R delta {ec['mean_R_delta']} (bar +.10), "
          f"retention {ec['retention']} (bar .50), bootstrap p {ec['bootstrap_sharpe_diff_p']}")
    print(f"  ABLATION   : {o['ablation']['sharpe_drop_pct']}% drop — {o['ablation']['band']}")
    print(f"  CALIBRATION: Brier {o['calibration']['brier']}, ECE {o['calibration']['ece']}")
    print(f"  SHAP top-5 : {[s2['feature'] for s2 in o['shap_top5']]}")
    print(f"  holdout 2024+ UNTOUCHED; labels SIMULATED-exit; nothing committed.")
    print("=" * 72)
