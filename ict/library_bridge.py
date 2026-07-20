"""
ict/library_bridge.py
=====================
Read-only bridge from the Alexandrian Library into the ICT engine.

Calls AlexandrianLibrary.query() once per scan and returns a simplified
context dict safe for ICT use. Zero changes to master-branch sovereign/ code.

Output shape:
  {
    'regime':        'ASIAN_CURRENCY_CONTAGION',
    'threat':        'WARNING',          # NORMAL | ELEVATED | WARNING | DANGER | CRITICAL
    'threat_score':  0.73,
    'size_modifier': 0.50,
    'advisory':      'Reduce exposure...',
    'top_volumes':   ['I Dislocations', 'IV Currency Crises'],
    'available':     True,
  }
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_FALLBACK = {
    'regime': 'UNKNOWN', 'threat': 'NORMAL', 'threat_score': 0.0,
    'size_modifier': 1.0, 'advisory': 'Library unavailable — default sizing.',
    'top_volumes': [], 'available': False,
}


def query_library() -> dict:
    """
    Query the Alexandrian Library with live price arrays.
    Fetches SPY/VIX/GLD/DXY via yfinance (same as sovereign orchestrator).
    Falls back gracefully if data is unavailable.
    """
    try:
        import sys
        import numpy as np
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from sovereign.risk.alexandrian_library import AlexandrianLibrary
        lib = AlexandrianLibrary()

        if not lib.n_volumes:
            logger.debug("Library: no volumes loaded")
            return _FALLBACK

        spy_arr, vix_arr, gold_arr, dxy_arr = _fetch_price_arrays()
        if spy_arr is None or len(spy_arr) < 200:
            logger.debug("Library: insufficient SPY history (%s bars)",
                         len(spy_arr) if spy_arr is not None else 0)
            return _FALLBACK

        insight = lib.query(spy_arr, vix_prices=vix_arr,
                            gold_prices=gold_arr, dxy_prices=dxy_arr)

        # Similarity floor: cosine similarity below 0.30 is indistinguishable
        # from noise across a 23-feature normalised vector.  Firing a regime
        # label (and the G1_LIBRARY veto) on a 0.067 match produces false
        # precision.  Below the floor we return UNKNOWN with default sizing
        # so the ICT engine behaves as if the library abstained.
        # Added 2026-07-20 after forensics showed sim=0.067 at the last scan.
        SIMILARITY_FLOOR = 0.30
        if insight.primary_similarity < SIMILARITY_FLOOR:
            logger.info(
                "Library: sim=%.3f < floor %.2f — returning UNKNOWN (no regime call)",
                insight.primary_similarity, SIMILARITY_FLOOR,
            )
            return {
                **_FALLBACK,
                'available': True,   # library loaded OK, just no confident match
                'regime':   'UNKNOWN',
                'advisory': (
                    f'sim={insight.primary_similarity:.3f} below floor '
                    f'{SIMILARITY_FLOOR:.2f} — no confident regime match.'
                ),
            }

        result = {
            'regime':        insight.primary_regime,
            'threat':        insight.threat_level,
            'threat_score':  round(float(insight.threat_score), 3),
            'size_modifier': round(float(insight.size_modifier), 3),
            'advisory':      (insight.advisory or '')[:200],
            'top_volumes':   getattr(insight, 'matching_volumes', [])[:3],
            'available':     insight.primary_regime not in ('LIBRARY_EMPTY', 'INSUFFICIENT_DATA'),
        }
        logger.info("Library: %s | %s | size=%.2f×",
                    result['regime'], result['threat'], result['size_modifier'])
        return result

    except Exception as e:
        logger.warning("Library bridge failed: %s", e)
        return _FALLBACK


def _fetch_price_arrays():
    """Fetch daily close arrays for SPY, VIX, GLD, DXY."""
    try:
        import yfinance as yf
        import numpy as np

        def _arr(ticker, period='400d'):
            df = yf.download(ticker, period=period, interval='1d',
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, __import__('pandas').MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df['Close'].dropna().values if len(df) else None

        spy  = _arr('SPY')
        vix  = _arr('^VIX', '60d')
        gold = _arr('GLD',  '400d')
        dxy  = _arr('DX-Y.NYB', '60d')

        return spy, vix, gold, dxy
    except Exception as e:
        logger.debug("Price array fetch failed: %s", e)
        return None, None, None, None
