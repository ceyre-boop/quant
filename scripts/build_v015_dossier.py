#!/usr/bin/env python3
"""
scripts/build_v015_dossier.py — the v015 verification dossier (charts + verdict).

Renders the forward-walk result of FROZEN v015 (manifest data/proof/v015_manifest.json)
across IS / OOS / FRESH windows, cross-checked on yfinance AND OANDA. Writes PNGs +
a self-contained HTML, copies the gallery to ~/Downloads/alta_v015_dossier/, and opens it.

The numbers below are the MEASURED results from this session's runs (one-shot, frozen):
  holdout_validation_v014.py (IS/OOS) + prove.py --start 2025-01-01 --end 2026-06-27 (yfinance)
  + frozen v015 re-run on OANDA candles (clean cross-source).
"""
from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GALLERY = Path.home() / "Downloads" / "alta_v015_dossier"

# ── measured results (frozen-v015 forward-walk, this session) ──────────────────
WINDOWS = {
    "IS\n2015-22":        {"sharpe": 0.577, "lo": 0.458, "hi": 0.695, "n": 318, "color": "#8a93a0"},
    "OOS\n2023-24":       {"sharpe": 1.250, "lo": 1.001, "hi": 1.500, "n": 110, "color": "#2E86DE"},
    "FRESH\n2025-26 (yf)": {"sharpe": -0.015, "lo": -0.277, "hi": 0.247, "n": 56, "color": "#C0392B"},
    "FRESH\n2025-26 (OANDA)": {"sharpe": -0.085, "lo": -0.341, "hi": 0.171, "n": 59, "color": "#7d3c98"},
}
PER_PAIR = {
    "OOS 2023-24":        {"EURUSD": 1.28, "GBPUSD": 1.44, "USDJPY": 1.50, "AUDUSD": 0.78},
    "FRESH yfinance":     {"EURUSD": -0.63, "GBPUSD": 0.78, "USDJPY": -1.12, "AUDUSD": 0.95},
    "FRESH OANDA":        {"EURUSD": -0.43, "GBPUSD": 0.93, "USDJPY": -0.98, "AUDUSD": 0.14},
}
PERM_FULL = {"p": 0.001, "real": 0.353, "null_mean": -0.145, "null_std": 0.156, "null_pct95": 0.117}


def _save(fig, name):
    GALLERY.mkdir(parents=True, exist_ok=True)
    p = GALLERY / name
    fig.savefig(p, dpi=120, bbox_inches="tight")
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); buf.seek(0)
    plt.close(fig)
    return p, base64.b64encode(buf.read()).decode()


def chart_decay():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    labels = list(WINDOWS.keys())[:3]  # IS, OOS, FRESH(yf) as the trajectory
    xs = range(len(labels))
    sh = [WINDOWS[l]["sharpe"] for l in labels]
    lo = [WINDOWS[l]["sharpe"] - WINDOWS[l]["lo"] for l in labels]
    hi = [WINDOWS[l]["hi"] - WINDOWS[l]["sharpe"] for l in labels]
    ax.errorbar(xs, sh, yerr=[lo, hi], fmt="o-", lw=2.5, ms=11, color="#2E86DE",
                ecolor="#888", capsize=6, capthick=1.5)
    # OANDA fresh as a second point
    ax.plot(2, -0.085, "s", ms=11, color="#7d3c98", label="FRESH OANDA (clean)")
    ax.axhline(1.5, color="#37d39b", ls="--", lw=0.9, label="institutional 1.5")
    ax.axhline(0.0, color="#C0392B", ls="--", lw=0.9, label="zero edge")
    for i, l in enumerate(labels):
        ax.annotate(f"{sh[i]:+.2f}", (i, sh[i]), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=12, fontweight="bold")
    ax.annotate("the 1.25 collapses\nto ~0 on unseen data", (2, -0.015), textcoords="offset points",
                xytext=(-10, -42), ha="center", fontsize=10, color="#C0392B")
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels)
    ax.set_ylabel("√n-weighted portfolio Sharpe (costed)")
    ax.set_title("v015 DECAY TRAJECTORY — IS → OOS → FRESH (95% CI)\nfrozen params, one-shot forward-walk", fontweight="bold")
    ax.grid(alpha=0.25); ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(-1.0, 2.0)
    return _save(fig, "1_decay_trajectory.png")


def chart_per_pair():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    series = list(PER_PAIR.keys())
    x = np.arange(len(pairs)); w = 0.26
    colors = ["#2E86DE", "#C0392B", "#7d3c98"]
    for i, s in enumerate(series):
        vals = [PER_PAIR[s][p] for p in pairs]
        ax.bar(x + (i - 1) * w, vals, w, label=s, color=colors[i])
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(pairs)
    ax.set_ylabel("per-pair Sharpe")
    ax.set_title("Per-pair: the OOS winners (USDJPY +1.50, EURUSD +1.28) INVERT on fresh data\n"
                 "— consistent across yfinance AND OANDA", fontweight="bold")
    ax.grid(alpha=0.25, axis="y"); ax.legend(fontsize=9)
    return _save(fig, "2_per_pair_windows.png")


def chart_cross_source():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    x = np.arange(len(pairs)); w = 0.38
    ax.bar(x - w / 2, [PER_PAIR["FRESH yfinance"][p] for p in pairs], w, label="yfinance", color="#C0392B")
    ax.bar(x + w / 2, [PER_PAIR["FRESH OANDA"][p] for p in pairs], w, label="OANDA (clean broker)", color="#7d3c98")
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(pairs); ax.set_ylabel("FRESH per-pair Sharpe")
    ax.set_title("Cross-source check (fresh 2025-26): two data sources AGREE on the collapse\n"
                 "(0.5% price divergence ≠ a different conclusion)", fontweight="bold")
    ax.grid(alpha=0.25, axis="y"); ax.legend(fontsize=9)
    return _save(fig, "3_cross_source_fresh.png")


def chart_perm():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    rng = np.random.default_rng(7)
    null = rng.normal(PERM_FULL["null_mean"], PERM_FULL["null_std"], 4000)
    ax.hist(null, bins=50, color="#8a93a0", alpha=0.8, label="null (random entries)")
    ax.axvline(PERM_FULL["real"], color="#2E86DE", lw=2.5, label=f"real edge {PERM_FULL['real']} (p={PERM_FULL['p']})")
    ax.axvline(PERM_FULL["null_pct95"], color="#37d39b", ls="--", lw=1, label="null 95th pct")
    ax.set_title("Is the EDGE real? Permutation test (2015-2024, costed)\n"
                 "real Sharpe beats random entries at p<0.001 — the edge is REAL... in-sample/used-OOS", fontweight="bold")
    ax.set_xlabel("portfolio Sharpe"); ax.legend(fontsize=9); ax.grid(alpha=0.25)
    return _save(fig, "4_permutation_real.png")


def main() -> int:
    if GALLERY.exists():
        for f in GALLERY.glob("*.png"):
            f.unlink()
    charts = [chart_decay(), chart_per_pair(), chart_cross_source(), chart_perm()]
    # copy the actual fresh equity curve from prove.py
    fresh_png = ROOT / "data" / "proof" / "backtest_equity_v015_fresh.png"
    if fresh_png.exists():
        shutil.copy(fresh_png, GALLERY / "5_fresh_equity_curve.png")

    imgs = "".join(
        f'<h3>{p.name}</h3><img src="data:image/png;base64,{b64}" style="width:100%;max-width:1100px;'
        f'border:1px solid #242832;border-radius:8px;margin:6px 0"/>' for p, b64 in charts)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>v015 Verification Dossier</title>
<style>body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:#0f1115;color:#e6e6e6;margin:0}}
.wrap{{max-width:1180px;margin:0 auto;padding:28px}} h1{{margin:0 0 6px}} .muted{{color:#8a93a0}}
.verdict{{background:#3d1f12;color:#e8845a;border-radius:8px;padding:14px 18px;margin:14px 0;font-weight:600;font-size:16px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}} th,td{{padding:6px 10px;border-bottom:1px solid #242832;text-align:left}} th{{color:#8a93a0}}</style></head><body><div class="wrap">
<h1>v015 Verification Dossier</h1>
<div class="muted">Frozen v015 (manifest commit 897534ac) · one-shot forward-walk · {len(charts)+1} charts</div>
<div class="verdict">VERDICT: the 1.25 reproduces EXACTLY on the used 2023-24 window — but FROZEN v015 on
GENUINELY UNSEEN 2025-2026 data scored Sharpe ≈ 0 (−0.015 yfinance / −0.085 OANDA, CI includes 0,
excludes 1.25). The rate-differential majors (USDJPY, EURUSD) that drove the 1.25 INVERTED forward.
Decay: IS 0.58 → OOS 1.25 → FRESH ~0. The edge did NOT hold out-of-sample-forward. Cross-source robust.</div>
<table><tr><th>window</th><th>Sharpe</th><th>95% CI</th><th>n</th><th>status</th></tr>
<tr><td>IS 2015-22</td><td>0.58</td><td>[0.46, 0.70]</td><td>318</td><td>baseline</td></tr>
<tr><td>OOS 2023-24 (used)</td><td>1.25</td><td>[1.00, 1.50]</td><td>110</td><td>reproduces ✓</td></tr>
<tr><td>FRESH 2025-26 yfinance</td><td>-0.015</td><td>[-0.28, 0.25]</td><td>56</td><td>≈ 0 (not 1.25)</td></tr>
<tr><td>FRESH 2025-26 OANDA</td><td>-0.085</td><td>[-0.34, 0.17]</td><td>59</td><td>≈ 0 (confirms)</td></tr></table>
{imgs}</div></body></html>"""
    out = ROOT / "data" / "proof" / "v015_dossier.html"
    out.write_text(html)
    shutil.copy(out, GALLERY / "v015_dossier.html")

    print(f"[dossier] {len(charts)+1} charts → {GALLERY}")
    for f in sorted(GALLERY.glob("*.png")):
        print("  ", f.name)
    # open on screen (Colin can't see inline images)
    subprocess.run(["open", str(GALLERY)], check=False)
    subprocess.run(["open", "-a", "Preview", *[str(p) for p in sorted(GALLERY.glob("*.png"))]], check=False)
    subprocess.run(["open", str(GALLERY / "v015_dossier.html")], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
