#!/usr/bin/env python3
"""
scripts/prove.py — THE proof command.

Runs the authoritative v015 forex backtester over full history, draws the
money-over-time curve, and prints a one-screen falsifiable verdict.

Backtests are the proof engine: fast, reproducible, falsifiable — unlike the
live paper loop which takes years to reach significance at 4-14 trades/yr.
Prove an edge here BEFORE anything touches live.

    python3 scripts/prove.py                              # v015 universe, 2015-2024
    python3 scripts/prove.py --start 2023-01-01 --end 2024-12-31   # OOS holdout only
    python3 scripts/prove.py --pairs EURUSD=X GBPUSD=X

Outputs to data/proof/:
    backtest_equity_<tag>.json   (equity_curve.v1 — shared schema with the live curve)
    backtest_equity_<tag>.csv
    backtest_trades_<tag>.csv
    backtest_equity_<tag>.png
    backtest_equity_latest.json  (copy the dashboard / `alta money` read)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.reporting.equity_curve import (
    build_from_trades, weighted_portfolio_sharpe, points_to_csv,
)

PROOF_DIR = ROOT / "data" / "proof"

_TRADE_COLS = ["pair", "entry_date", "exit_date", "direction", "entry", "exit",
               "pnl_pct", "risk_pct", "risk_adjusted_pnl_pct", "hold_days",
               "exit_reason", "units"]


def _build(pairs, start, end, equity):
    """Run the v015 backtester and assemble (curve, per-pair results, trades)."""
    from sovereign.forex.forex_backtester import ForexBacktester, RESULTS_PATH
    import sovereign.forex.forex_backtester as fb

    bt = ForexBacktester(start=start, end=end)

    if pairs:
        orig = fb.ALL_PAIRS
        fb.ALL_PAIRS = list(pairs)
        try:
            results = bt.backtest_all()
        finally:
            fb.ALL_PAIRS = orig
    else:
        results = bt.backtest_all()

    trades_path = RESULTS_PATH.parent / "forex_backtest_trades.json"
    trades_by_pair = json.loads(trades_path.read_text()) if trades_path.exists() else {}
    all_trades = []
    for pair, tl in trades_by_pair.items():
        for t in tl:
            row = dict(t)
            row["pair"] = pair
            all_trades.append(row)

    label = f"Forex v015 {start[:4]}-{end[:4]}"
    curve = build_from_trades(all_trades, initial_equity=equity, label=label, source="backtest")
    curve["stats"]["portfolio_sharpe_weighted"] = weighted_portfolio_sharpe(
        [(r.sharpe, r.total_trades) for r in results]
    )
    curve["window"] = {"start": start, "end": end}
    return curve, results, all_trades


def _trades_csv(all_trades: list[dict]) -> str:
    lines = [",".join(_TRADE_COLS)]
    for t in all_trades:
        lines.append(",".join(str(t.get(c, "")) for c in _TRADE_COLS))
    return "\n".join(lines) + "\n"


def _plot(curve: dict, png_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  (matplotlib unavailable — skipping PNG: {type(e).__name__})")
        return False
    pts = curve.get("points", [])
    if not pts:
        return False
    ys = [p["equity"] for p in pts]
    x = list(range(len(ys)))
    init = curve["initial_equity"]
    s = curve["stats"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, ys, lw=1.6, color="#2E86DE")
    ax.axhline(init, color="#888", ls="--", lw=0.8)
    ax.fill_between(x, init, ys, where=[y >= init for y in ys], alpha=0.12, color="#2E86DE")
    ax.fill_between(x, init, ys, where=[y < init for y in ys], alpha=0.12, color="#C0392B")
    ax.set_title(
        f'{curve["label"]} — equity curve   '
        f'(Sharpe {s.get("portfolio_sharpe_weighted")}, '
        f'ret {s.get("total_return_pct")}%, maxDD {s.get("max_drawdown_pct")}%)'
    )
    ax.set_xlabel(f'closed trades (n={len(ys)})   {pts[0]["t"]} → {pts[-1]["t"]}')
    ax.set_ylabel("equity ($)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    return True


def _verdict(curve: dict, results) -> None:
    s = curve["stats"]
    pts = curve["points"]
    wsharpe = s.get("portfolio_sharpe_weighted") or 0.0
    bar = "INSTITUTIONAL ✓" if wsharpe >= 1.5 else ("VIABLE EDGE" if wsharpe >= 0.8 else "WEAK / REVIEW")
    print()
    print("=" * 66)
    print(f"  ALTA PROOF — {curve['label']}")
    print("=" * 66)
    print(f"  Trades closed:    {s.get('n_trades')}")
    if pts:
        print(f"  Window:           {pts[0]['t']} → {pts[-1]['t']}  ({s.get('years')}y)")
    print(f"  Start equity:     ${curve['initial_equity']:,.0f}")
    print(f"  Final equity:     ${curve['final_equity']:,.0f}")
    print()
    print(f"  PORTFOLIO SHARPE: {wsharpe}   (√n-weighted per-pair — the v015 number)")
    print(f"  VERDICT:          {bar}   (target ≥1.5; viable ≥0.8)")
    print(f"  Total return:     {s.get('total_return_pct'):+}%   (risk-sized, conviction model)")
    print(f"  CAGR:             {s.get('cagr_pct'):+}%")
    print(f"  Max drawdown:     {s.get('max_drawdown_pct')}%")
    print(f"  Win rate:         {(s.get('win_rate') or 0) * 100:.1f}%")
    print(f"  Profit factor:    {s.get('profit_factor')}")
    print()
    print("  NOTE: Sharpe is leverage-invariant = the proven EDGE. The dollar return")
    print("        is at the system's conservative risk-sized basis (pnl × risk_pct);")
    print("        a flat $ curve with a high Sharpe means SIZING is the lever, not the edge.")
    print()
    print("  Per-pair (the falsifiable detail):")
    for r in sorted(results, key=lambda r: -r.sharpe):
        print(f"    {r.pair:10s}  sharpe={r.sharpe:+.2f}  win={r.win_rate:.0%}  "
              f"pf={r.profit_factor:.2f}  n={r.total_trades:>3}  dd={r.max_drawdown:.1%}")
    print("=" * 66)


def _svg_polyline(values, w, h, pad, color, baseline=None):
    """Return (svg_path_d, baseline_y) for a value series scaled into a w×h box."""
    if len(values) < 2:
        return "", None
    lo, hi = min(values), max(values)
    if baseline is not None:
        lo, hi = min(lo, baseline), max(hi, baseline)
    rng = (hi - lo) or 1.0
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = pad + (w - 2 * pad) * i / (n - 1)
        y = pad + (h - 2 * pad) * (1 - (v - lo) / rng)
        pts.append(f"{x:.1f},{y:.1f}")
    d = "M " + " L ".join(pts)
    base_y = None
    if baseline is not None:
        base_y = pad + (h - 2 * pad) * (1 - (baseline - lo) / rng)
    return d, base_y


def _write_html(curve: dict, live_rows: list, path: Path) -> None:
    """Self-contained 'Am I Making Money?' page — data inlined, no server, no deps."""
    W, H, PAD = 1100, 420, 36
    s = curve.get("stats", {})
    pts = curve.get("points", [])
    eq = [p["equity"] for p in pts]
    init = curve.get("initial_equity", 0)
    d, base_y = _svg_polyline(eq, W, H, PAD, "#2E86DE", baseline=init)
    up = curve.get("final_equity", init) >= init
    line_color = "#2E86DE" if up else "#C0392B"
    wsharpe = s.get("portfolio_sharpe_weighted")
    verdict = ("INSTITUTIONAL" if (wsharpe or 0) >= 1.5
               else "VIABLE EDGE" if (wsharpe or 0) >= 0.8 else "WEAK / REVIEW")

    # live curve (optional)
    live_block = '<p class="muted">No live snapshots yet — the 2h pulse populates this.</p>'
    if live_rows:
        lv = [r.get("nav") for r in live_rows if r.get("nav") is not None]
        ld, lbase = _svg_polyline(lv, W, 160, PAD, "#16A085", baseline=lv[0] if lv else None)
        cur = live_rows[-1]
        pct = round((lv[-1] / lv[0] - 1) * 100, 3) if len(lv) > 1 and lv[0] else 0.0
        live_block = (
            f'<div class="stat">NAV now <b>${cur.get("nav",0):,.0f}</b> · '
            f'since {live_rows[0]["t"][:10]} <b>{pct:+}%</b> · {len(lv)} snapshots</div>'
            f'<svg viewBox="0 0 {W} 160" class="chart">'
            f'<path d="{ld}" fill="none" stroke="#16A085" stroke-width="2"/></svg>'
        )

    base_line = f'<line x1="{PAD}" y1="{base_y:.1f}" x2="{W-PAD}" y2="{base_y:.1f}" stroke="#888" stroke-dasharray="4" stroke-width="1"/>' if base_y is not None else ""
    x_label = f'{pts[0]["t"]} → {pts[-1]["t"]}  ·  n={len(eq)} closed trades' if pts else ""

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alta — Am I Making Money?</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}}
 .wrap{{max-width:1180px;margin:0 auto;padding:28px}}
 h1{{font-size:22px;margin:0 0 4px}} .muted{{color:#8a93a0}}
 .verdict{{display:inline-block;padding:4px 12px;border-radius:6px;font-weight:700;margin:8px 0;
   background:{('#123d2b' if (wsharpe or 0)>=0.8 else '#3d1f12')};color:{('#37d39b' if (wsharpe or 0)>=0.8 else '#e8845a')}}}
 .grid{{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}}
 .stat{{background:#181b22;border:1px solid #242832;border-radius:8px;padding:10px 14px}}
 .stat b{{color:#fff;font-size:17px}}
 .chart{{width:100%;height:auto;background:#181b22;border:1px solid #242832;border-radius:10px;margin:10px 0}}
 .section{{margin-top:26px}} .note{{color:#8a93a0;font-size:13px;margin-top:8px}}
</style></head><body><div class="wrap">
 <h1>Are we making money?</h1>
 <div class="muted">{curve.get("label","")} · generated by <code>alta prove</code></div>
 <div class="verdict">{verdict} — Sharpe {wsharpe}</div>
 <div class="grid">
   <div class="stat">Total return <b>{s.get("total_return_pct")}%</b><div class="muted">{s.get("years")}y</div></div>
   <div class="stat">CAGR <b>{s.get("cagr_pct")}%</b></div>
   <div class="stat">Max DD <b>{s.get("max_drawdown_pct")}%</b></div>
   <div class="stat">Win rate <b>{round((s.get("win_rate") or 0)*100,1)}%</b></div>
   <div class="stat">Profit factor <b>{s.get("profit_factor")}</b></div>
   <div class="stat">Trades <b>{s.get("n_trades")}</b></div>
 </div>
 <svg viewBox="0 0 {W} {H}" class="chart">
   {base_line}
   <path d="{d}" fill="none" stroke="{line_color}" stroke-width="2"/>
 </svg>
 <div class="muted">{x_label}</div>
 <div class="note">Sharpe is the leverage-invariant EDGE (the proven part). A flat $ curve with a
   high Sharpe means SIZING is the lever, not the edge. Backtests are the fast falsifiable proof;
   the live curve below is slow forward confirmation (~4-14 trades/yr).</div>
 <div class="section"><h1 style="font-size:18px">Live paper (OANDA practice) — forward</h1>{live_block}</div>
</div></body></html>"""
    path.write_text(html)


def main() -> int:
    ap = argparse.ArgumentParser(description="Draw the v015 backtest equity curve + falsifiable verdict.")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--equity", type=float, default=100_000.0)
    ap.add_argument("--pairs", nargs="*", default=None, help="subset (default: v015 ALL_PAIRS)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=str(PROOF_DIR))
    args = ap.parse_args()

    curve, results, all_trades = _build(args.pairs, args.start, args.end, args.equity)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"v015_{args.start[:4]}_{args.end[:4]}"
    (out / f"backtest_equity_{tag}.json").write_text(json.dumps(curve, indent=2, default=str))
    (out / f"backtest_equity_{tag}.csv").write_text(points_to_csv(curve))
    (out / f"backtest_trades_{tag}.csv").write_text(_trades_csv(all_trades))
    drew = _plot(curve, out / f"backtest_equity_{tag}.png")
    # 'latest' copy for `alta money`; mirror into data/agent (served by live_signals_server)
    (out / "backtest_equity_latest.json").write_text(json.dumps(curve, indent=2, default=str))
    agent_dir = ROOT / "data" / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "backtest_equity_latest.json").write_text(json.dumps(curve, indent=2, default=str))

    # self-contained 'Am I Making Money?' page (data inlined — open directly, no server)
    live_rows = []
    live_path = agent_dir / "equity_curve_live.jsonl"
    if live_path.exists():
        live_rows = [json.loads(l) for l in live_path.read_text().splitlines() if l.strip()]
    _write_html(curve, live_rows, out / "proof.html")

    _verdict(curve, results)
    print(f"\n  Artifacts → {out}/backtest_equity_{tag}.{{json,csv,png}}  +  {out}/proof.html"
          f"{'' if drew else '  (png skipped)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
