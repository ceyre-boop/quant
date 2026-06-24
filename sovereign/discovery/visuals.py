"""
sovereign/discovery/visuals.py
==============================
Phase-4 visual suite for a discovery run. Produces a self-contained HTML report
(charts embedded as base64 — open directly, no server) plus a strategy-table CSV.

Charts (each guarded — degrades gracefully if a lib/data is absent):
  • strategy table        — every gated candidate, colour-coded by verdict
  • equity overlay        — finalist equity curves (reuses reporting.equity_curve)
  • feature importance     — XGBoost over features → forward-return direction
  • win-rate by weekday    — session/time heatmap from the top candidate's trades

Reuses scripts/prove.py plotting conventions and sovereign/reporting/equity_curve.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign.reporting.equity_curve import build_from_trades


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def _equity_overlay(verdicts) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    finalists = [v for v in verdicts if v.trades]
    if not finalists:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))
    for v in sorted(finalists, key=lambda v: -v.full_sharpe)[:6]:
        curve = build_from_trades(v.trades, label=v.name)
        ys = [p["equity"] for p in curve["points"]]
        if len(ys) > 1:
            ax.plot(range(len(ys)), ys, lw=1.4, label=f"{v.name} (S={v.full_sharpe:+.2f})")
    ax.axhline(100000, color="#888", ls="--", lw=0.8)
    ax.set_title("Finalist equity curves (risk-sized) — full window")
    ax.set_xlabel("closed trades")
    ax.set_ylabel("equity ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    return _fig_to_b64(fig)


def _feature_importance(adapter, features_by_pair, train_window, fwd=5) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from xgboost import XGBClassifier
    except Exception:
        return None
    # Use whatever feature columns the frames actually carry (generic OR regime features).
    first = next((f for f in features_by_pair.values() if f is not None), None)
    if first is None:
        return None
    cols = [c for c in first.columns if first[c].dtype.kind in "fi"]
    if not cols:
        return None
    X_rows, y_rows = [], []
    for pair in adapter.pairs:
        fdf, pdf = features_by_pair.get(pair), adapter.price_df(pair)
        if fdf is None or pdf is None:
            continue
        use = [c for c in cols if c in fdf.columns]
        close = pdf["Close"] if "Close" in pdf.columns else pdf.iloc[:, 0]
        fret = (close.shift(-fwd) / close - 1.0)
        mask = (fdf.index >= pd.Timestamp(train_window[0])) & (fdf.index <= pd.Timestamp(train_window[1]))
        X_rows.append(fdf.loc[mask, use].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy())
        y_rows.append((fret.loc[mask].fillna(0.0) > 0).astype(int).to_numpy())
    if not X_rows:
        return None
    FEATURE_COLUMNS = use
    X, y = np.vstack(X_rows), np.concatenate(y_rows)
    if len(np.unique(y)) < 2:
        return None
    clf = XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.1,
                        subsample=0.8, verbosity=0, eval_metric="logloss")
    clf.fit(X, y)
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([FEATURE_COLUMNS[i] for i in order][::-1], imp[order][::-1], color="#2E86DE")
    ax.set_title("Feature importance — predicting 5-bar forward direction (train window, XGBoost)")
    ax.grid(alpha=0.2, axis="x")
    return _fig_to_b64(fig)


def _winrate_by_dow(verdicts) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    top = next((v for v in sorted(verdicts, key=lambda v: -v.full_sharpe) if v.trades), None)
    if top is None:
        return None
    rows = []
    for t in top.trades:
        ed = t.get("exit_date") or t.get("entry_date")
        try:
            dow = pd.Timestamp(str(ed)).dayofweek
        except Exception:
            continue
        rows.append((dow, 1 if t.get("pnl_pct", 0) > 0 else 0))
    if not rows:
        return None
    dfr = pd.DataFrame(rows, columns=["dow", "win"])
    wr = dfr.groupby("dow")["win"].mean().reindex(range(5)).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(["Mon", "Tue", "Wed", "Thu", "Fri"], [wr.get(i, 0) for i in range(5)], color="#16A085")
    ax.axhline(0.5, color="#888", ls="--", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_title(f"Win rate by weekday — {top.name}")
    ax.set_ylabel("win rate")
    return _fig_to_b64(fig)


def strategy_table_rows(verdicts) -> list[dict]:
    rows = []
    for v in sorted(verdicts, key=lambda v: ({"VALID_EDGE": 0, "NOT_SIGNIFICANT": 1, "SCREENED_OUT": 2}.get(v.verdict, 3), -v.full_sharpe)):
        rows.append({
            "name": v.name, "verdict": v.verdict, "family": v.meta.get("family", ""),
            "train_sharpe": v.train_sharpe, "full_sharpe": v.full_sharpe,
            "holdout_sharpe": v.holdout_sharpe, "n_trades": v.n_trades,
            "perm_p": v.perm_p, "dsr_prob": v.dsr_prob, "bh_survives": v.bh_survives,
            "description": v.description,
        })
    return rows


def render(track: str, verdicts, adapter, features_by_pair, train_window, out_dir: Path,
           summary: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = strategy_table_rows(verdicts)
    pd.DataFrame(rows).to_csv(out_dir / "strategy_table.csv", index=False)

    imgs = {
        "Equity curves (finalists)": _equity_overlay(verdicts),
        "Feature importance": _feature_importance(adapter, features_by_pair, train_window),
        "Win rate by weekday (top candidate)": _winrate_by_dow(verdicts),
    }

    def _color(v):
        return {"VALID_EDGE": "#123d2b", "NOT_SIGNIFICANT": "#2a2118", "SCREENED_OUT": "#1a1d24"}.get(v, "#1a1d24")

    def _tcolor(v):
        return {"VALID_EDGE": "#37d39b", "NOT_SIGNIFICANT": "#e8b85a", "SCREENED_OUT": "#8a93a0"}.get(v, "#ccc")

    trows = "".join(
        f'<tr style="background:{_color(r["verdict"])}"><td>{r["name"]}</td>'
        f'<td style="color:{_tcolor(r["verdict"])};font-weight:700">{r["verdict"]}</td>'
        f'<td>{r["family"]}</td><td>{r["train_sharpe"]}</td><td>{r["full_sharpe"]}</td>'
        f'<td>{r["holdout_sharpe"]}</td><td>{r["n_trades"]}</td><td>{r["perm_p"]}</td>'
        f'<td>{r["dsr_prob"]}</td><td>{r["bh_survives"]}</td></tr>'
        for r in rows
    )
    img_html = "".join(
        f'<h3>{title}</h3><img src="data:image/png;base64,{b64}" style="width:100%;max-width:1100px;border:1px solid #242832;border-radius:8px;margin:6px 0"/>'
        for title, b64 in imgs.items() if b64
    )
    n_valid = summary.get("n_valid", 0)
    banner = ("#123d2b" if n_valid else "#2a2118")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Alta Discovery — {track}</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:#0f1115;color:#e6e6e6;margin:0}}
 .wrap{{max-width:1180px;margin:0 auto;padding:26px}} h1{{margin:0 0 4px}} .muted{{color:#8a93a0}}
 .banner{{background:{banner};border-radius:8px;padding:12px 16px;margin:12px 0;font-weight:600}}
 table{{border-collapse:collapse;width:100%;font-size:12.5px;margin:10px 0}}
 th,td{{padding:6px 8px;text-align:left;border-bottom:1px solid #242832}} th{{color:#8a93a0}}
</style></head><body><div class="wrap">
 <h1>Edge Discovery — {track}</h1>
 <div class="muted">{summary.get('window','')} · {summary.get('n_candidates',0)} candidates · {summary.get('n_finalists',0)} gated · n_trials(DSR)={summary.get('n_candidates',0)}</div>
 <div class="banner">{'✓ '+str(n_valid)+' VALID_EDGE survivor(s)' if n_valid else '0 VALID_EDGE survivors — the gate found no real new edge. This is the correct, protective output: it refuses to hand you a false positive.'}</div>
 <table><thead><tr><th>name</th><th>verdict</th><th>family</th><th>train</th><th>full</th><th>holdout</th><th>n</th><th>perm_p</th><th>dsr_prob</th><th>bh</th></tr></thead><tbody>{trows}</tbody></table>
 {img_html}
 <p class="muted">Discovery proposes; the permutation + Deflated-Sharpe + Benjamini-Hochberg gate decides.
 A survivor here is a screening-grade candidate — promote to the official ledger by re-running at ≥10k permutations via research_factory. No live config touched.</p>
</div></body></html>"""
    (out_dir / "discover.html").write_text(html)
    return {"table_csv": str(out_dir / "strategy_table.csv"), "html": str(out_dir / "discover.html")}
