#!/usr/bin/env python3
"""
Forex backtest results visualiser.
Reads logs/forex_backtest_results.json → saves 4 PNGs to logs/charts/.

Usage:
    python3 scripts/plot_forex_results.py
    python3 scripts/plot_forex_results.py --json path/to/results.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DEFAULT = Path(__file__).parents[1] / 'logs' / 'forex_backtest_results.json'
CHARTS_DIR      = Path(__file__).parents[1] / 'logs' / 'charts'


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _pair_label(ticker: str) -> str:
    return ticker.replace('=X', '')


def _color(val: float, good_high: bool = True) -> str:
    if good_high:
        return '#2ecc71' if val >= 0 else '#e74c3c'
    return '#e74c3c' if val >= 0 else '#2ecc71'


def plot_sharpe(results: list[dict], out: Path) -> None:
    pairs   = [_pair_label(r['pair']) for r in results]
    sharpes = [r['sharpe'] for r in results]
    colors  = [_color(s) for s in sharpes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(pairs, sharpes, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.6)

    for bar, val in zip(bars, sharpes):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                y + (0.02 if y >= 0 else -0.05),
                f'{val:+.2f}', ha='center', va='bottom' if y >= 0 else 'top',
                fontsize=8, color='white')

    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#cccccc')
    ax.spines[:].set_color('#333355')
    ax.set_title('Sharpe Ratio by Pair — 2015–2024', color='white', fontsize=13, pad=12)
    ax.set_ylabel('Annualised Sharpe', color='#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.tick_params(axis='x', colors='#cccccc', rotation=30)

    pos = mpatches.Patch(color='#2ecc71', label='Positive Sharpe')
    neg = mpatches.Patch(color='#e74c3c', label='Negative Sharpe')
    ax.legend(handles=[pos, neg], facecolor='#1a1a2e', labelcolor='white', fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor='#1a1a2e')
    plt.close()
    print(f'Saved: {out}')


def plot_winrate_vs_pf(results: list[dict], out: Path) -> None:
    win_rates = [r['win_rate'] * 100 for r in results]
    pfs       = [min(r['profit_factor'], 4.0) for r in results]
    trades    = [r['total_trades'] for r in results]
    sharpes   = [r['sharpe'] for r in results]
    pairs     = [_pair_label(r['pair']) for r in results]
    colors    = [_color(s) for s in sharpes]
    sizes     = [max(t * 3, 40) for t in trades]

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(win_rates, pfs, c=colors, s=sizes,
                         alpha=0.85, edgecolors='white', linewidth=0.6)

    for i, pair in enumerate(pairs):
        ax.annotate(pair, (win_rates[i], pfs[i]),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=7.5, color='#dddddd')

    ax.axhline(1.0, color='#cccccc', linewidth=0.7, linestyle='--', alpha=0.5)
    ax.axvline(50,  color='#cccccc', linewidth=0.7, linestyle='--', alpha=0.5)

    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#cccccc')
    ax.spines[:].set_color('#333355')
    ax.set_title('Win Rate vs Profit Factor\n(bubble size = total trades)', color='white', fontsize=12, pad=10)
    ax.set_xlabel('Win Rate (%)', color='#cccccc')
    ax.set_ylabel('Profit Factor (capped 4×)', color='#cccccc')

    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor='#1a1a2e')
    plt.close()
    print(f'Saved: {out}')


def plot_drawdown(results: list[dict], out: Path) -> None:
    pairs = [_pair_label(r['pair']) for r in results]
    dds   = [r['max_drawdown'] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(pairs, dds, color='#e74c3c', edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, dds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val - 0.5,
                f'{val:.1f}%', ha='center', va='top',
                fontsize=8, color='white')

    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#cccccc')
    ax.spines[:].set_color('#333355')
    ax.set_title('Max Drawdown by Pair — 2015–2024', color='white', fontsize=13, pad=12)
    ax.set_ylabel('Max Drawdown (%)', color='#cccccc')
    ax.tick_params(axis='x', colors='#cccccc', rotation=30)

    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor='#1a1a2e')
    plt.close()
    print(f'Saved: {out}')


def plot_summary_table(results: list[dict], out: Path) -> None:
    sorted_r = sorted(results, key=lambda r: r['sharpe'], reverse=True)

    cols  = ['Pair', 'Trades', 'TPY', 'Win%', 'Sharpe', 'PF', 'MaxDD']
    rows  = []
    for r in sorted_r:
        rows.append([
            _pair_label(r['pair']),
            str(r['total_trades']),
            f'{r["trades_per_year"]:.1f}',
            f'{r["win_rate"]:.1%}',
            f'{r["sharpe"]:+.2f}',
            f'{r["profit_factor"]:.2f}',
            f'{r["max_drawdown"]:.1%}',
        ])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis('off')

    tbl = ax.table(
        cellText=rows,
        colLabels=cols,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)

    # Colour header
    for j in range(len(cols)):
        tbl[0, j].set_facecolor('#2c2c54')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Colour Sharpe cells
    for i, r in enumerate(sorted_r):
        cell = tbl[i + 1, 4]
        cell.set_facecolor('#1a3a1a' if r['sharpe'] >= 0 else '#3a1a1a')
        cell.set_text_props(color='#2ecc71' if r['sharpe'] >= 0 else '#e74c3c',
                            fontweight='bold')
        # Alternate row bg
        bg = '#1e1e2e' if i % 2 == 0 else '#16213e'
        for j in range(len(cols)):
            if j != 4:
                tbl[i + 1, j].set_facecolor(bg)
                tbl[i + 1, j].set_text_props(color='#dddddd')

    fig.patch.set_facecolor('#0f0f23')
    ax.set_title('Forex Backtest Summary — 2015–2024 (sorted by Sharpe)',
                 color='white', fontsize=12, pad=15, y=0.97)

    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor='#0f0f23', bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=Path, default=RESULTS_DEFAULT)
    args = parser.parse_args()

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    results = load(args.json)

    plot_sharpe(results,          CHARTS_DIR / '01_sharpe.png')
    plot_winrate_vs_pf(results,   CHARTS_DIR / '02_winrate_vs_pf.png')
    plot_drawdown(results,        CHARTS_DIR / '03_drawdown.png')
    plot_summary_table(results,   CHARTS_DIR / '04_summary_table.png')

    print(f'\nAll charts written to {CHARTS_DIR}/')


if __name__ == '__main__':
    main()
