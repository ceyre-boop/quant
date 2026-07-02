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
    "cot_net_pct", "cot_net_oi", "cot_net_pct_1y", "cot_net_z", "cot_flush_1w", "tff_lev_net_pct",
    "vrp_signal", "vrp_pct", "vrp_iv_atm", "vrp_rv_trailing",
    "rr25", "bf25", "atm_term_slope",
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
           cot_net_pct, cot_net_oi, cot_net_pct_1y, cot_net_z, cot_flush_1w, tff_lev_net_pct,
           vrp_signal, vrp_pct, vrp_iv_atm, vrp_rv_trailing,
           rr25, bf25, atm_term_slope,
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
               c.net_pct AS cot_net_pct, c.net_oi AS cot_net_oi,
               c.net_pct_1y AS cot_net_pct_1y, c.net_z_1y AS cot_net_z, c.flush_1w AS cot_flush_1w,
               t.lev_net_pct AS tff_lev_net_pct,
               r.vrp_signal, r.vrp_pct, r.iv_atm AS vrp_iv_atm, r.rv_trailing AS vrp_rv_trailing,
               o.rr25, o.bf25, o.term_slope AS atm_term_slope,
               m.macro_curve, m.macro_spread, m.macro_inflation,
               s.vix_close AS vix_level, s.vix_momentum, s.vix_regime,
               CAST(now() AS TIMESTAMP) AS built_at
        FROM spine s
        ASOF LEFT JOIN macro_pivot m ON s.date >= m.date
        -- COT is weekly, dated to its FRIDAY publish_date → ASOF carries the latest published row forward
        -- per pair (board sees it only on/after publish — no Tuesday look-ahead). TFF (leveraged funds)
        -- rides the identical keying.
        ASOF LEFT JOIN sentiment_cot_weekly c ON c.pair = s.pair AND s.date >= c.publish_date
        ASOF LEFT JOIN sentiment_cot_tff_weekly t ON t.pair = s.pair AND s.date >= t.publish_date
        -- VRP + options surface are weekly obs dated to their observable EOD close → ASOF carries the
        -- latest reading forward per pair (past+present only, never forward). FIXTURE-stamped surface
        -- rows are test scaffolding and are excluded from the board fusion.
        ASOF LEFT JOIN sentiment_vrp_daily r ON r.pair = s.pair AND s.date >= r.date
        ASOF LEFT JOIN (SELECT * FROM sentiment_options_surface
                        WHERE iv_source NOT LIKE 'FIXTURE%') o
            ON o.pair = s.pair AND s.date >= o.date
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
