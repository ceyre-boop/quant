"""Deterministic classification rules for the political-alpha event catalog (HYP-085).

DATA ONLY — no functions. These tables ARE the operationalization of the locked
"clear and blatant language" definition (spec §2): a statement qualifies when it
(1) names an entity in ENTITY_MAP, (2) matches a POLICY_PATTERNS category, and
(3) matches ACTION_PATTERNS (announces/previews an action). Federal Register
executive orders satisfy (3) by construction (a signed EO is the action).

AUTHORED 2026-07-08 BEFORE the classifier first ran on any scraped text
(pre-registration discipline). Any edit made after the first classification run
must be disclosed in the Phase-1 Obsidian note. Matched literals are recorded
per event in the catalog's `notes` field for audit.

Entities outside the 10-instrument universe (Canada, Mexico, pharma, autos, ...)
do NOT qualify — they are never stretched onto DXY (spec: the instrument must be
"directly named"/mapped, and the honesty gate forbids loosening for count).
"""

# ── entity -> (sector_tagged, instrument_tagged) ─────────────────────────────────────
# Ordered list of (entity_label, regex, [(sector, instrument), ...]).
# Case-insensitive, applied to title + full text.

ENTITY_MAP = [
    ("china", r"\b(china|chinese|beijing|xi\s+jinping)\b", [("china_tech", "KWEB")]),
    ("steel_aluminum", r"\b(steel|aluminum|aluminium)\b", [("steel", "SLX")]),
    ("energy", r"\b(oil|crude|opec|drilling|petroleum|gasoline|lng|natural\s+gas|energy\s+(dominance|emergency|production))\b",
     [("energy", "XLE")]),
    ("financials", r"\b(dodd[-\s]?frank|bank(ing)?\s+(de)?regulation|capital\s+requirements|community\s+banks?)\b",
     [("financials", "XLF")]),
    # "gold card" is Trump's visa program, not the metal — excluded via lookahead.
    ("gold", r"\bgold\b(?!\s+card)", [("gold", "GLD")]),
    ("eurozone", r"\b(euro(zone)?\b|european\s+union|\be\.?u\.?\b|brussels)\b", [("eurusd", "EURUSD=X")]),
    ("japan", r"\b(japan(ese)?|yen|tokyo)\b", [("usdjpy", "USDJPY=X")]),
    ("uk", r"\b(united\s+kingdom|britain|british|u\.?k\.?\b|sterling|pound\s+sterling)\b",
     [("gbpusd", "GBPUSD=X")]),
    ("australia", r"\b(australia(n)?)\b", [("audusd", "AUDUSD=X")]),
    ("usd", r"\b(dollar|de[-\s]?dollariz\w*|brics|currency\s+manipulat\w*|reserve\s+currency)\b",
     [("usd", "DX-Y.NYB")]),
    ("fed_rates", r"\b(federal\s+reserve|the\s+fed\b|powell|interest\s+rates?|rate\s+(cut|hike)s?)\b",
     [("usd_rates", "DX-Y.NYB")]),
]

# ── policy-action category (the "policy action with direct price implications") ─────

POLICY_PATTERNS = [
    ("tariff", r"\b(tariffs?|duties|import\s+tax(es)?|reciprocal\s+(trade|tariff)|section\s+(232|301)|trade\s+deficit)\b"),
    ("sanction", r"\b(sanctions?|embargo(es)?|export\s+controls?|blacklist)\b"),
    ("trade_deal", r"\b(trade\s+(deal|agreement|talks|war|pact)|purchase\s+agreement)\b"),
    ("rate", r"\b(interest\s+rates?|rate\s+(cut|hike)s?|monetary\s+policy|federal\s+reserve|the\s+fed\b)\b"),
    ("energy", r"\b(drill(ing)?|energy\s+emergency|oil\s+production|opec|strategic\s+petroleum|lng\s+exports?|pipeline)\b"),
    ("currency", r"\b(currency\s+manipulat\w*|devalu\w*|(strong|weak)\s+dollar|de[-\s]?dollariz\w*|brics\s+currenc\w*)\b"),
    ("regulation", r"\b(deregulat\w*|regulation|regulatory\s+(relief|rollback))\b"),
]

# ── action verbs (announcing / previewing) ───────────────────────────────────────────
# One regex; Federal Register EOs auto-satisfy this leg (signed instrument).

ACTION_PATTERNS = (
    r"\b(impos(e|es|ing|ed)|announc(e|es|ing|ed)|sign(ed|ing|s)?|order(ed|ing|s)?|"
    r"proclaim\w*|direct(ed|ing|s)?|implement\w*|effective|levy(ing)?|levie[sd]|"
    r"rais(e|es|ing|ed)|increas(e|es|ing|ed)|cut(s|ting)?|lower(ed|ing|s)?|"
    r"ban(ned|ning|s)?|block(ed|ing|s)?|revok(e|es|ing|ed)|terminat(e|es|ing|ed)|"
    r"withdraw\w*|suspend\w*|hereby|will\s+(be\s+)?(impos|plac|put|charg|sign|pay)\w*|"
    r"\d{1,3}\s?(percent|%))"
)

# ── scheduled-macro exclusion calendar (placebo eligibility, Test 3) ─────────────────
# Public release dates, transcribed 2026-07-08. Sources:
#   FOMC decision days: federalreserve.gov/monetarypolicy/fomccalendars.htm
#   CPI 08:30-ET releases: bls.gov/schedule/news_release/cpi.htm
#   Employment Situation (NFP): bls.gov/schedule/news_release/empsit.htm
# Coverage: study period 2025-01-20 -> 2026-08 (placebo draws never leave this span).
# A MISSED exclusion only inflates the placebo null (harder to reject H0) — the
# conservative direction; VERIFY-AT-BUILD against the source pages where reachable.

SCHEDULED_MACRO_DATES = [
    # FOMC 2025 (decision day = second day of each meeting)
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # FOMC 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    # CPI 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13", "2025-06-11",
    "2025-07-15", "2025-08-12", "2025-09-11", "2025-10-15", "2025-11-13", "2025-12-10",
    # CPI 2026 (through study horizon)
    "2026-01-13", "2026-02-11", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-10",
    "2026-07-14", "2026-08-11",
    # NFP / Employment Situation 2025
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04", "2025-05-02", "2025-06-06",
    "2025-07-03", "2025-08-01", "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
    # NFP 2026 (through study horizon)
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03", "2026-05-08", "2026-06-05",
    "2026-07-02", "2026-08-07",
]

STUDY_START = "2025-01-20"
