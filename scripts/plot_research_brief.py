#!/usr/bin/env python3
"""
Alta Investments — Forex Macro Strategy Research Brief
Generates a single professional multi-panel figure from real backtest data.

Usage:
    python3 scripts/plot_research_brief.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────── #
ROOT        = Path(__file__).parents[1]
RESULTS     = ROOT / 'logs' / 'forex_backtest_results.json'
TRADES      = ROOT / 'logs' / 'forex_backtest_trades.json'
OUT_DIR     = ROOT / 'logs' / 'charts'
OUT_FILE    = OUT_DIR / 'alta_research_brief.png'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────── #
BG          = '#07071a'
PANEL_BG    = '#0d0d2b'
BORDER      = '#1a1a3e'
WHITE       = '#f0f0f0'
MUTED       = '#8888aa'
GREEN       = '#00d4aa'
RED         = '#ff4757'
BLUE        = '#3d9bff'
GOLD        = '#ffd700'
ACCENT      = '#7b68ee'

PAIR_COLORS = {
    'EURUSD=X': '#3d9bff', 'GBPUSD=X': '#00d4aa', 'USDJPY=X': '#ffd700',
    'USDCHF=X': '#ff9f43', 'AUDUSD=X': '#ee5a24', 'USDCAD=X': '#c44569',
    'NZDUSD=X': '#f8a5c2', 'EURGBP=X': '#778ca3', 'EURJPY=X': '#e056fd',
    'GBPJPY=X': '#67e480',  'AUDNZD=X': '#45aaf2',
}


def _label(ticker: str) -> str:
    return ticker.replace('=X', '')


def _load() -> tuple[list[dict], dict[str, list]]:
    with open(RESULTS) as f:
        results = json.load(f)
    trades: dict[str, list] = {}
    if TRADES.exists():
        with open(TRADES) as f:
            trades = json.load(f)
    return results, trades


def _equity_curve(trades: list[dict]) -> pd.Series:
    """Compound PnL curve from trade list."""
    if not trades:
        return pd.Series([1.0])
    pnls = [t['pnl_pct'] for t in trades]
    curve = np.cumprod([1 + p for p in pnls])
    return pd.Series(curve)


def _drawdown_series(curve: pd.Series) -> pd.Series:
    roll_max = curve.expanding().max()
    return (curve - roll_max) / roll_max * 100


def build(results: list[dict], all_trades: dict[str, list]) -> None:
    sorted_r = sorted(results, key=lambda r: r['sharpe'], reverse=True)

    fig = plt.figure(figsize=(22, 28), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # ── Header ────────────────────────────────────────────────────────── #
    header_ax = fig.add_axes([0.0, 0.955, 1.0, 0.045])
    header_ax.set_facecolor('#0b0b22')
    header_ax.axis('off')
    header_ax.text(0.03, 0.55, 'ALTA INVESTMENTS',
                   color=WHITE, fontsize=18, fontweight='bold',
                   transform=header_ax.transAxes, va='center',
                   fontfamily='monospace')
    header_ax.text(0.03, 0.18, 'Forex Macro Strategy  ·  Systematic Research Brief  ·  Backtest 2015–2024',
                   color=MUTED, fontsize=9, transform=header_ax.transAxes, va='center',
                   fontfamily='monospace')
    header_ax.text(0.97, 0.55, f'Generated {datetime.now().strftime("%d %b %Y")}',
                   color=MUTED, fontsize=9, transform=header_ax.transAxes, va='center',
                   ha='right', fontfamily='monospace')
    header_ax.text(0.97, 0.18, 'CONFIDENTIAL  ·  INTERNAL USE ONLY',
                   color='#ff4757', fontsize=8, transform=header_ax.transAxes, va='center',
                   ha='right', fontfamily='monospace', alpha=0.7)
    # Thin gold separator line
    header_ax.axhline(0, color=GOLD, linewidth=1.5, alpha=0.6)

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        top=0.945, bottom=0.04,
        left=0.06, right=0.97,
        hspace=0.42, wspace=0.30,
    )

    # ── 1. Equity curves (spans full width, top row) ──────────────────── #
    ax_eq = fig.add_subplot(gs[0, :])
    _style(ax_eq)
    ax_eq.set_title('Cumulative Return by Pair  (equal weight, compounded per trade)',
                    color=WHITE, fontsize=11, pad=8, loc='left', fontfamily='monospace')

    for r in sorted_r:
        pair   = r['pair']
        trades = all_trades.get(pair, [])
        if not trades:
            continue
        curve = _equity_curve(trades)
        pct   = (curve - 1) * 100
        x     = np.linspace(0, 1, len(pct))
        lw    = 2.0 if r['sharpe'] > 0 else 0.8
        alpha = 0.9 if r['sharpe'] > 0 else 0.35
        ax_eq.plot(x, pct, color=PAIR_COLORS[pair], linewidth=lw,
                   alpha=alpha, label=_label(pair))

    ax_eq.axhline(0, color=MUTED, linewidth=0.6, linestyle='--', alpha=0.4)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:+.0f}%'))
    ax_eq.set_xlim(0, 1)
    ax_eq.set_xlabel('Backtest Progress (2015 → 2024)', color=MUTED, fontsize=8)

    legend = ax_eq.legend(
        ncol=11, loc='upper left', fontsize=7.5,
        facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=WHITE,
        framealpha=0.9, handlelength=1.2, columnspacing=0.8,
    )

    # ── 2. Sharpe bar chart ────────────────────────────────────────────── #
    ax_sh = fig.add_subplot(gs[1, :2])
    _style(ax_sh)
    ax_sh.set_title('Annualised Sharpe Ratio', color=WHITE, fontsize=10,
                    pad=6, loc='left', fontfamily='monospace')

    pairs_sh  = [_label(r['pair']) for r in sorted_r]
    sharpes   = [r['sharpe'] for r in sorted_r]
    colors_sh = [GREEN if s > 0 else RED for s in sharpes]

    bars = ax_sh.barh(pairs_sh[::-1], sharpes[::-1], color=colors_sh[::-1],
                      height=0.6, edgecolor='none')
    ax_sh.axvline(0, color=MUTED, linewidth=0.8, linestyle='--', alpha=0.5)

    for bar, val in zip(bars, sharpes[::-1]):
        x_pos = val + (0.02 if val >= 0 else -0.02)
        ha    = 'left' if val >= 0 else 'right'
        ax_sh.text(x_pos, bar.get_y() + bar.get_height() / 2,
                   f'{val:+.2f}', va='center', ha=ha,
                   color=WHITE, fontsize=8, fontfamily='monospace')

    ax_sh.set_xlabel('Sharpe Ratio', color=MUTED, fontsize=8)
    ax_sh.tick_params(axis='y', labelsize=9, labelcolor=WHITE)

    # ── 3. Signal layer attribution ────────────────────────────────────── #
    ax_att = fig.add_subplot(gs[1, 2])
    _style(ax_att)
    ax_att.set_title('Signal Architecture', color=WHITE, fontsize=10,
                     pad=6, loc='left', fontfamily='monospace')
    ax_att.axis('off')

    layers = [
        ('Layer 1', 'CB Event Trigger',       'Post-decision drift, 10–20d hold', GREEN),
        ('Layer 1.5', 'CPI Surprise Fade',    'Edge 2: fade CPI overshoot, 5d',   BLUE),
        ('Layer 2', 'Calendar Signals',       'QE rebalance + March JPY flow',    GOLD),
        ('Layer 3', 'Macro (IRP + RRD)',       'Monthly, 60d hold',               ACCENT),
        ('Gate A', 'Buffett Conviction',       'Min 0.35 → 0.85 size tiers',      MUTED),
        ('Gate B', 'COT Positioning',          'Dalio Q3: crowded = 0.5× size',   '#ff9f43'),
        ('Gate C', 'DXY Dollar Smile',         'Growth vs safety regime modifier', '#c44569'),
        ('Gate D', 'VIX Carry Unwind',         'VIX>25 + backwardation override',  RED),
    ]

    for i, (tag, name, desc, col) in enumerate(layers):
        y = 0.92 - i * 0.115
        ax_att.add_patch(mpatches.FancyBboxPatch(
            (0.0, y - 0.04), 0.17, 0.075,
            boxstyle='round,pad=0.01',
            facecolor=col, alpha=0.25, transform=ax_att.transAxes))
        ax_att.text(0.085, y, tag, color=col, fontsize=7.5, fontweight='bold',
                    ha='center', va='center', transform=ax_att.transAxes,
                    fontfamily='monospace')
        ax_att.text(0.21, y + 0.015, name, color=WHITE, fontsize=8,
                    transform=ax_att.transAxes, fontfamily='monospace', fontweight='bold')
        ax_att.text(0.21, y - 0.018, desc, color=MUTED, fontsize=7,
                    transform=ax_att.transAxes, fontfamily='monospace')

    # ── 4. Win rate vs Profit Factor scatter ──────────────────────────── #
    ax_sc = fig.add_subplot(gs[2, 0])
    _style(ax_sc)
    ax_sc.set_title('Win Rate vs Profit Factor', color=WHITE, fontsize=10,
                    pad=6, loc='left', fontfamily='monospace')

    for r in sorted_r:
        col  = GREEN if r['sharpe'] > 0 else RED
        size = max(r['total_trades'] * 0.8, 30)
        ax_sc.scatter(r['win_rate'] * 100, min(r['profit_factor'], 2.5),
                      c=col, s=size, alpha=0.85, edgecolors=PANEL_BG, linewidth=0.8, zorder=3)
        ax_sc.annotate(
            _label(r['pair']),
            (r['win_rate'] * 100, min(r['profit_factor'], 2.5)),
            xytext=(5, 3), textcoords='offset points',
            color=WHITE, fontsize=7, fontfamily='monospace',
        )

    ax_sc.axhline(1.0, color=MUTED, linewidth=0.7, linestyle='--', alpha=0.5)
    ax_sc.axvline(50,  color=MUTED, linewidth=0.7, linestyle='--', alpha=0.5)
    ax_sc.set_xlabel('Win Rate (%)', color=MUTED, fontsize=8)
    ax_sc.set_ylabel('Profit Factor', color=MUTED, fontsize=8)
    ax_sc.text(51, 0.25, 'Profitable zone →', color=GREEN, fontsize=7,
               alpha=0.6, fontfamily='monospace')

    # ── 5. Drawdown bars ──────────────────────────────────────────────── #
    ax_dd = fig.add_subplot(gs[2, 1])
    _style(ax_dd)
    ax_dd.set_title('Max Drawdown', color=WHITE, fontsize=10,
                    pad=6, loc='left', fontfamily='monospace')

    dds   = [r['max_drawdown'] * 100 for r in sorted_r]
    p_lbl = [_label(r['pair']) for r in sorted_r]
    dd_colors = [PAIR_COLORS[r['pair']] for r in sorted_r]

    ax_dd.barh(p_lbl[::-1], dds[::-1], color=dd_colors[::-1],
               height=0.6, alpha=0.7, edgecolor='none')
    for i, (val, pair) in enumerate(zip(dds[::-1], p_lbl[::-1])):
        ax_dd.text(val - 0.3, i, f'{val:.1f}%', va='center', ha='right',
                   color=WHITE, fontsize=8, fontfamily='monospace')

    ax_dd.set_xlabel('Max Drawdown (%)', color=MUTED, fontsize=8)
    ax_dd.tick_params(axis='y', labelsize=9, labelcolor=WHITE)
    ax_dd.invert_xaxis()

    # ── 6. Trade count + TPY ──────────────────────────────────────────── #
    ax_tr = fig.add_subplot(gs[2, 2])
    _style(ax_tr)
    ax_tr.set_title('Trade Density', color=WHITE, fontsize=10,
                    pad=6, loc='left', fontfamily='monospace')

    total_t = [r['total_trades'] for r in sorted_r]
    tpy_v   = [r['trades_per_year'] for r in sorted_r]
    x_pos   = np.arange(len(sorted_r))

    bars1 = ax_tr.bar(x_pos - 0.2, total_t, width=0.35, color=BLUE,
                      alpha=0.7, label='Total trades', edgecolor='none')
    ax_tr2 = ax_tr.twinx()
    ax_tr2.set_facecolor('none')
    ax_tr2.tick_params(colors=GOLD)
    ax_tr2.spines[:].set_visible(False)
    ax_tr2.plot(x_pos, tpy_v, 'o-', color=GOLD, linewidth=1.5,
                markersize=5, label='Trades/year')
    ax_tr2.set_ylabel('Trades / Year', color=GOLD, fontsize=8)
    ax_tr2.yaxis.label.set_color(GOLD)

    ax_tr.set_xticks(x_pos)
    ax_tr.set_xticklabels([_label(r['pair']) for r in sorted_r],
                          rotation=45, ha='right', fontsize=7.5, color=WHITE)
    ax_tr.set_ylabel('Total Trades (2015–2024)', color=MUTED, fontsize=8)
    ax_tr.tick_params(axis='y', labelsize=8, labelcolor=MUTED)

    # ── 7. Summary table ──────────────────────────────────────────────── #
    ax_tbl = fig.add_subplot(gs[3, :])
    ax_tbl.set_facecolor(PANEL_BG)
    ax_tbl.axis('off')

    col_labels = ['Pair', 'Trades', 'Trades/Yr', 'Win Rate',
                  'Sharpe', 'Profit Factor', 'Max Drawdown', 'Avg Hold (d)', 'Edge']
    rows = []
    for r in sorted_r:
        edge = ('✓ POSITIVE' if r['sharpe'] > 0.5 else
                '~ MARGINAL' if r['sharpe'] > 0 else '✗ NEGATIVE')
        rows.append([
            _label(r['pair']),
            str(r['total_trades']),
            f"{r['trades_per_year']:.1f}",
            f"{r['win_rate']:.1%}",
            f"{r['sharpe']:+.2f}",
            f"{r['profit_factor']:.2f}",
            f"{r['max_drawdown']:.1%}",
            f"{r['avg_hold_days']:.0f}",
            edge,
        ])

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor('#14143a')
        cell.set_text_props(color=GOLD, fontweight='bold', fontfamily='monospace')
        cell.set_edgecolor(BORDER)

    for i, r in enumerate(sorted_r):
        bg = '#0d0d2b' if i % 2 == 0 else '#10102e'
        for j in range(len(col_labels)):
            cell = tbl[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor(BORDER)
            if j == 4:  # Sharpe column
                col = GREEN if r['sharpe'] > 0 else RED
                cell.set_text_props(color=col, fontweight='bold', fontfamily='monospace')
            elif j == 8:  # Edge column
                col = GREEN if r['sharpe'] > 0.5 else (GOLD if r['sharpe'] > 0 else RED)
                cell.set_text_props(color=col, fontfamily='monospace')
            else:
                cell.set_text_props(color=WHITE, fontfamily='monospace')

    # ── Footer ────────────────────────────────────────────────────────── #
    footer = fig.add_axes([0.0, 0.0, 1.0, 0.022])
    footer.set_facecolor('#0b0b22')
    footer.axis('off')
    footer.text(
        0.5, 0.5,
        'This document is for internal research purposes only. '
        'Past performance is not indicative of future results. '
        'Alta Investments — Systematic Forex Research Program',
        color=MUTED, fontsize=7.5, ha='center', va='center',
        transform=footer.transAxes, fontfamily='monospace',
    )
    footer.axhline(1, color=GOLD, linewidth=0.8, alpha=0.4)

    plt.savefig(OUT_FILE, dpi=180, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f'Research brief saved → {OUT_FILE}')


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.title.set_fontfamily('monospace')


if __name__ == '__main__':
    results, trades = _load()
    build(results, trades)
