"""
ict/regime_execution.py
=======================
Regime-aware dynamic TP ratios for the ICT engine.

Maps Alexandrian Library regime → R:R targets.
The ICT engine uses 2R/4R by default. In trending regimes we let it run.
In stress/ranging regimes we take profits earlier.

Output:
  get_regime_targets(regime, threat) → (tp1_r, tp2_r, label)
"""
from __future__ import annotations

from typing import Tuple

# (tp1_r, tp2_r, label, description)
_REGIME_MAP = {
    # Strong trending → let it run
    'TRENDING':              (3.0, 6.0, 'TRENDING',   'Strong trend — extended targets'),
    'MELT_UP':               (3.0, 6.0, 'TRENDING',   'Melt-up — maximum extension'),
    'EARLY_RECOVERY':        (2.5, 5.0, 'TRENDING',   'Recovery phase — above average'),
    'MID_CYCLE_ACCELERATION':(2.5, 5.0, 'TRENDING',   'Mid-cycle — above average'),
    'CARRY_TRADE_BUILDUP':   (2.5, 5.0, 'TRENDING',   'Carry building — momentum trades'),
    'FED_CUTTING':           (2.5, 5.0, 'TRENDING',   'Rate cut cycle — risk-on trends'),

    # Neutral / default
    'FED_HIKING_PAUSE':      (2.0, 4.0, 'NEUTRAL',    'Standard ICT targets'),
    'LATE_CYCLE_FED_HIKING': (2.0, 4.0, 'NEUTRAL',    'Standard ICT targets'),
    'STAGFLATION':           (2.0, 4.0, 'NEUTRAL',    'Standard ICT targets'),
    'INSUFFICIENT_DATA':     (2.0, 4.0, 'NEUTRAL',    'No library data — defaults'),
    'UNKNOWN':               (2.0, 4.0, 'NEUTRAL',    'Unknown regime — defaults'),

    # Ranging / choppy — take profits faster
    'RANGING':               (1.5, 3.0, 'RANGING',    'Choppy — tight targets'),
    'COMPRESSION':           (1.5, 3.0, 'RANGING',    'Vol compression — tight targets'),
    'LOW_VOL_REGIME':        (1.5, 3.0, 'RANGING',    'Low vol — tight targets'),

    # Stress — skip or very tight
    'SHALE_SUPPLY_OIL_CRASH':(1.5, 3.0, 'STRESSED',  'Commodity stress — tight targets'),
    'ASIAN_CURRENCY_CONTAGION':(1.5, 2.5,'STRESSED',  'EM stress — very tight'),
    'REPO_MARKET_STRESS':    (1.5, 2.5, 'STRESSED',   'Liquidity stress — very tight'),
    'CREDIT_DISLOCATION':    (1.5, 2.5, 'STRESSED',   'Credit stress — very tight'),
    'CONTAGION':             (1.5, 2.5, 'STRESSED',   'Contagion — very tight'),
}

# Threat-level overrides — never skip entirely (timeout kills prop challenge)
# Instead: reduce targets and risk. Only true black swans (repo freeze) skip.
_THREAT_OVERRIDE = {
    'CRITICAL': (1.5, 2.5, 'STRESSED', 'CRITICAL — tightest targets, 0.5% risk'),
    'DANGER':   (1.5, 3.0, 'STRESSED', 'DANGER — tight targets, reduced risk'),
    'WARNING':  (2.0, 3.5, 'STRESSED', 'WARNING — conservative targets'),
}

# True skip events — only these hard-coded regime names cause a full skip
_SKIP_REGIMES = {
    'REPO_MARKET_STRESS',       # interbank freeze
    'EXCHANGE_CIRCUIT_BREAKER', # market-wide halt
    'FLASH_CRASH',              # genuine 1-in-100 day
}

DEFAULT = (2.0, 4.0, 'NEUTRAL', 'Default ICT 2R/4R')


def get_regime_targets(regime: str, threat: str = 'NORMAL') -> dict:
    """
    Returns dynamic TP ratios based on regime + threat level.

    Returns:
      {
        'tp1_r':      2.0,
        'tp2_r':      4.0,
        'risk_mult':  1.0,   # multiply base risk% by this
        'mode':       'NEUTRAL',
        'reason':     'Standard ICT 2R/4R',
        'skip':       False,
      }
    """
    # Hard skip for true black-swan regimes only
    if regime in _SKIP_REGIMES:
        return {'tp1_r': 0, 'tp2_r': 0, 'risk_mult': 0,
                'mode': 'SKIP', 'reason': f'{regime} — market halted', 'skip': True}

    # Threat override adjusts targets + risk multiplier
    risk_mult = 1.0
    if threat in _THREAT_OVERRIDE:
        tp1, tp2, mode, reason = _THREAT_OVERRIDE[threat]
        risk_mult = {'CRITICAL': 0.5, 'DANGER': 0.75, 'WARNING': 0.85}.get(threat, 1.0)
        return {
            'tp1_r': tp1, 'tp2_r': tp2, 'risk_mult': risk_mult,
            'mode': mode, 'reason': reason, 'skip': False,
        }

    # Regime lookup — try exact match then keyword scan
    entry = _REGIME_MAP.get(regime)
    if entry is None:
        r = regime.upper()
        for key, val in _REGIME_MAP.items():
            if key in r or r in key:
                entry = val
                break

    if entry is None:
        entry = DEFAULT

    tp1, tp2, mode, reason = entry
    return {
        'tp1_r':     tp1,
        'tp2_r':     tp2,
        'risk_mult': 1.0,
        'mode':      mode,
        'reason':    reason,
        'skip':      False,
    }
