"""sovereign/sentiment/board_state.py — fuse the three feeders into the daily per-pair board.

Builds one row per (trading-day, pair) by LEFT-JOINing news + VIX onto the FRED macro date spine:
    date | pair | news_score | macro_curve | macro_spread | macro_inflation
         | vix_level | vix_momentum | vix_regime

macro_curve = T10Y2Y, macro_spread = BAMLH0A0HYM2 (HY OAS), macro_inflation = T10YIE.
No forward-look: `date` is the signal date; every value is as-of that date's close, never shifted +1.
News is per-pair (NULL where the ~30-day NewsAPI window doesn't cover the date); macro/VIX are US-wide
(broadcast across pairs). `con` is injectable so tests run on an in-memory DuckDB with no network.
"""
from __future__ import annotations

from config.loader import params
from sovereign.sentiment.store import connect

PAIRS = list(params["sentiment"]["news"]["pairs"].keys())
REQUIRED_COLUMNS = [
    "date", "pair", "news_score", "gdelt_tone", "gdelt_tone_5d", "gdelt_volume", "econ_surprise_z",
    "macro_curve", "macro_spread", "macro_inflation", "vix_level", "vix_momentum", "vix_regime",
]


def rebuild(con=None) -> int:
    """Regenerate sentiment_board_state from the three source tables. Idempotent (full rebuild).

    The date spine is VIX (trading days) — one row per (trading-day, pair). Macro is attached with an
    ASOF (as-of) join, i.e. the latest macro observation on-or-before the trading day, so a FRED publish
    lag or holiday never leaks a future value backward and never drops a trading day. News is a plain
    per-pair join (NULL where the ~30-day NewsAPI window doesn't reach). VIX is broadcast across pairs.
    """
    own = con is None
    con = con or connect()
    pairs_values = ", ".join(f"('{p}')" for p in PAIRS)
    con.execute("DELETE FROM sentiment_board_state")
    con.execute(f"""
        INSERT INTO sentiment_board_state
          (date, pair, news_score, gdelt_tone, gdelt_tone_5d, gdelt_volume, econ_surprise_z,
           macro_curve, macro_spread, macro_inflation, vix_level, vix_momentum, vix_regime, built_at)
        WITH macro_pivot AS (
            SELECT date,
                MAX(CASE WHEN series = 'T10Y2Y'       THEN value END) AS macro_curve,
                MAX(CASE WHEN series = 'BAMLH0A0HYM2' THEN value END) AS macro_spread,
                MAX(CASE WHEN series = 'T10YIE'       THEN value END) AS macro_inflation
            FROM sentiment_macro_daily
            GROUP BY date
            HAVING macro_curve IS NOT NULL OR macro_spread IS NOT NULL OR macro_inflation IS NOT NULL
        ),
        pairs(pair) AS (VALUES {pairs_values}),
        spine AS (
            SELECT v.date, v.vix_close, v.vix_momentum, v.vix_regime, p.pair
            FROM sentiment_vix_daily v
            CROSS JOIN pairs p
        )
        SELECT s.date, s.pair, n.news_score,
               g.tone_score AS gdelt_tone, g.tone_5d AS gdelt_tone_5d, g.volume AS gdelt_volume,
               sd.econ_surprise_z,
               m.macro_curve, m.macro_spread, m.macro_inflation,
               s.vix_close AS vix_level, s.vix_momentum, s.vix_regime,
               CAST(now() AS TIMESTAMP) AS built_at
        FROM spine s
        ASOF LEFT JOIN macro_pivot m ON s.date >= m.date
        LEFT JOIN sentiment_news_daily    n  ON n.date  = s.date AND n.pair = s.pair
        LEFT JOIN sentiment_gdelt_daily   g  ON g.date  = s.date AND g.pair = s.pair
        LEFT JOIN sentiment_surprise_daily sd ON sd.date = s.date
    """)
    n = con.execute("SELECT COUNT(*) FROM sentiment_board_state").fetchone()[0]
    if own:
        con.close()
    return int(n)


def get_state(date, pair: str, con=None) -> dict | None:
    """The board row for one (date, pair) as a dict, or None if absent."""
    own = con is None
    con = con or connect(read_only=True)
    try:
        df = con.execute(
            "SELECT * FROM sentiment_board_state WHERE date = ? AND pair = ?", [str(date), pair]
        ).df()
    finally:
        if own:
            con.close()
    return None if df.empty else df.iloc[0].to_dict()


def get_history(start, end, pair: str, con=None):
    """Board rows for one pair across [start, end] (inclusive), ordered by date, as a DataFrame."""
    own = con is None
    con = con or connect(read_only=True)
    try:
        df = con.execute(
            "SELECT * FROM sentiment_board_state WHERE pair = ? AND date BETWEEN ? AND ? ORDER BY date",
            [pair, str(start), str(end)],
        ).df()
    finally:
        if own:
            con.close()
    return df
