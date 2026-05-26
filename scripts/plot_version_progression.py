#!/usr/bin/env python3
"""
Sovereign Forex — Version Progression Chart
Generates logs/charts/sovereign_progression.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Version history ──────────────────────────────────────────────────────────
VERSIONS = [
    ('v001',  0.1785, 'Baseline momentum'),
    ('v002',  0.3551, 'Macro signals'),
    ('v002.5',0.38,   'Micro-tuning'),
    ('v003',  0.4523, 'SNB / EURGBP gates'),
    ('v004',  0.6260, 'Macro confirmed'),
    ('v005',  1.0237, 'Trailing stop + forensics'),
    ('v006',  1.0547, 'GBPUSD 6d hold'),
    ('v007',  1.0713, 'All-pair hold sweep'),
    ('v008',  1.1955, 'USDCAD retired'),
    ('v009',  1.2864, 'GBPJPY retired'),
    ('v010',  1.4396, 'USDJPY VIX gate'),
    ('v011',  1.6476, 'Universal VIX gate'),
    ('v012',  1.7176, 'EUR/GBP VIX 20→18'),
    ('v013',  1.8552, 'Current live'),
]

# ── Theme ────────────────────────────────────────────────────────────────────
BG      = '#07071a'
PANEL   = '#0d0d2b'
GREEN   = '#00d4aa'
GOLD    = '#ffd700'
BLUE    = '#4a9eff'
MUTED   = '#5a6a7a'
WHITE   = '#e8eaf6'
RED     = '#ff4444'

fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

labels = [v[0] for v in VERSIONS]
sharpes = [v[1] for v in VERSIONS]
annots  = [v[2] for v in VERSIONS]
x = np.arange(len(VERSIONS))

# ── Draw step function ───────────────────────────────────────────────────────
for i in range(len(VERSIONS) - 1):
    delta = sharpes[i+1] - sharpes[i]
    if delta > 0.15:
        color = GOLD
    elif delta > 0.05:
        color = BLUE
    else:
        color = MUTED

    # Horizontal segment at current level
    ax.plot([x[i], x[i+1]], [sharpes[i], sharpes[i]], color=color, lw=2.5, solid_capstyle='round')
    # Vertical rise
    ax.plot([x[i+1], x[i+1]], [sharpes[i], sharpes[i+1]], color=color, lw=2.5, solid_capstyle='round')

    # Delta label on the rise
    if abs(delta) > 0.01:
        mid_y = (sharpes[i] + sharpes[i+1]) / 2
        sign = '+' if delta >= 0 else ''
        ax.text(x[i+1] + 0.08, mid_y, f'{sign}{delta:.3f}',
                color=color, fontsize=7.5, va='center', fontweight='bold')

# Final horizontal tail
ax.plot([x[-1], x[-1] + 0.4], [sharpes[-1], sharpes[-1]], color=GREEN, lw=2.5,
        solid_capstyle='round', linestyle='--')

# Dots at each version
for i, (xi, yi) in enumerate(zip(x, sharpes)):
    is_last = (i == len(VERSIONS) - 1)
    ax.scatter(xi, yi, s=55, color=GREEN if is_last else WHITE, zorder=5,
               edgecolors=BG, linewidths=1)

# ── Reference lines ──────────────────────────────────────────────────────────
ref_lines = [
    (1.0,    MUTED,  'First edge'),
    (1.5,    BLUE,   'Institutional grade'),
    (1.9052, GOLD,   'v014 target (+0.05)'),
]
for yval, color, label in ref_lines:
    ax.axhline(yval, color=color, lw=1, linestyle='--', alpha=0.55)
    ax.text(len(VERSIONS) - 0.4, yval + 0.02, label, color=color,
            fontsize=8, ha='right', alpha=0.85)

# ── Annotations below each version ──────────────────────────────────────────
for i, (xi, yi, ann) in enumerate(zip(x, sharpes, annots)):
    ax.text(xi, -0.12, labels[i], color=WHITE, fontsize=8, ha='center',
            rotation=35, va='top')
    ax.text(xi, yi + 0.05, ann, color=MUTED, fontsize=6.5, ha='center',
            va='bottom', rotation=25)

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xlim(-0.5, len(VERSIONS) - 0.3)
ax.set_ylim(-0.05, 2.15)
ax.set_xticks([])
ax.set_yticks([0, 0.5, 1.0, 1.5, 1.8552, 1.9052, 2.0])
ax.set_yticklabels(['0', '0.5', '1.0', '1.5', '1.8552\n(v013)', '1.9052\n(target)', '2.0'],
                   color=WHITE, fontsize=8)
ax.tick_params(axis='y', colors=MUTED, length=3)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.yaxis.set_tick_params(labelcolor=WHITE)

# ── Title + legend ────────────────────────────────────────────────────────────
fig.suptitle('SOVEREIGN FOREX — VERSION PROGRESSION', color=WHITE, fontsize=15,
             fontweight='bold', x=0.5, y=0.97)
ax.set_title(f'v001 → v013  |  Sharpe 0.178 → 1.855  |  +{1.8552-0.1785:.3f} total',
             color=MUTED, fontsize=9, pad=8)

legend_elements = [
    Line2D([0],[0], color=GOLD, lw=2, label='Large gain (>0.15 Sharpe)'),
    Line2D([0],[0], color=BLUE, lw=2, label='Medium gain (0.05–0.15)'),
    Line2D([0],[0], color=MUTED, lw=2, label='Small gain (<0.05)'),
]
ax.legend(handles=legend_elements, loc='upper left', facecolor=BG, edgecolor=MUTED,
          labelcolor=WHITE, fontsize=8, framealpha=0.8)

# ── Current version callout ──────────────────────────────────────────────────
ax.annotate('v013\n1.8552', xy=(x[-1], sharpes[-1]),
            xytext=(x[-1] - 1.2, sharpes[-1] + 0.22),
            color=GREEN, fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

out = Path('logs/charts/sovereign_progression.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'Saved → {out}')
