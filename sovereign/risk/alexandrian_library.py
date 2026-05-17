"""
sovereign/risk/alexandrian_library.py
=======================================
The Alexandrian Library — Complete Market Knowledge Base

"The greatest collection of market knowledge ever assembled,
 organized so the system can USE it, not just hold it."

The Library of Alexandria didn't just store scrolls — it organized knowledge
into retrievable volumes by subject. This is the market equivalent.

Ten volumes. Every knowable market state. Every major circumstance for which
data exists. Each entry carries the same 23-feature vector as the crash library,
so the system can compare current conditions against ALL of them simultaneously.

The system doesn't just know "this looks dangerous."
It knows "this looks like late-cycle Fed hiking into slowing growth —
historically that preceded 18 months of sector rotation before the actual
dislocation, and the sectors that outperformed were X."

That is operational intelligence.

VOLUMES:
  I    — Dislocations & Crashes       (20 events, 1929-2023)
  II   — Rate Cycle Regimes           (hiking, peak, pivot, cutting — since 1954)
  III  — Bull Market Regimes          (recovery, melt-up, late-cycle, exhaustion)
  IV   — Currency Crises              (EM stress, dollar wrecking ball, reserve shock)
  V    — Volatility Regimes           (compression, explosion, term structure collapse)
  VI   — Economic Cycles              (expansion, late-cycle, contraction, trough, early recovery)
  VII  — Liquidity Events             (repo stress, interbank freeze, money market crisis)
  VIII — Commodity Supercycles        (oil shock, metals bull, agri spike, energy crash)
  IX   — Geopolitical Shocks          (war premium, sanctions, political crisis)
  X    — Sector Rotation Regimes      (growth→value, defensive, risk-on→off, reflation)

Usage:
  lib = AlexandrianLibrary()
  lib.build_from_history()           # populate from yfinance (run once)
  insight = lib.query(spy_prices, vix_prices)
  print(insight.primary_regime)      # "LATE_CYCLE_FED_HIKING"
  print(insight.size_modifier)       # 0.75
  print(insight.advisory)            # rich multi-volume context
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parents[2]
LIBRARY_PATH = ROOT / 'models' / 'alexandrian_library.json'

# Re-use the feature extractor and cosine similarity from market_memory
from sovereign.risk.market_memory import (
    extract_features, _cosine_similarity, N_FEATURES, FEATURE_NAMES
)


# ═════════════════════════════════════════════════════════════════════════════
# VOLUME DEFINITIONS — Every major market circumstance since data exists
# ═════════════════════════════════════════════════════════════════════════════

class VolumeType(str, Enum):
    CRASHES         = "VOLUME_I_CRASHES"
    RATE_CYCLES     = "VOLUME_II_RATE_CYCLES"
    BULL_REGIMES    = "VOLUME_III_BULL_REGIMES"
    CURRENCY_CRISES = "VOLUME_IV_CURRENCY_CRISES"
    VOL_REGIMES     = "VOLUME_V_VOL_REGIMES"
    ECON_CYCLES     = "VOLUME_VI_ECON_CYCLES"
    LIQUIDITY       = "VOLUME_VII_LIQUIDITY"
    COMMODITIES     = "VOLUME_VIII_COMMODITIES"
    GEOPOLITICAL    = "VOLUME_IX_GEOPOLITICAL"
    SECTOR_ROTATION = "VOLUME_X_SECTOR_ROTATION"


@dataclass(frozen=True)
class LibraryEntry:
    """
    One entry in the library — a labelled historical market state.
    The feature vector describes the 90-day window preceding/during the event.
    """
    entry_id:      str
    volume:        str          # VolumeType value
    label:         str          # human-readable regime name
    date:          str          # representative date YYYY-MM-DD
    description:   str          # what was happening
    outcome:       str          # what followed historically
    outcome_days:  int          # approximate duration of outcome in days
    severity:      int          # 0=benign, 1=moderate, 2=severe, -1=positive
    tags:          List[str]    # searchable tags


# ── Volume I: Dislocations & Crashes ─────────────────────────────────────── #
VOLUME_I_CRASHES: List[LibraryEntry] = [
    LibraryEntry("v1_1929_crash",     VolumeType.CRASHES, "LEVERAGE_UNWIND_PEAK",       "1929-10-24", "Margin debt at historic highs, banking fragility", "89% equity drawdown over 3 years", 1100, 2, ["leverage","banking","deflation"]),
    LibraryEntry("v1_1962_flash",     VolumeType.CRASHES, "MOMENTUM_REVERSAL",          "1962-05-28", "Cold War tension, Kennedy steel crisis", "28% drawdown, V-shaped recovery", 120, 1, ["momentum","geopolitical","fast_recovery"]),
    LibraryEntry("v1_1973_oil",       VolumeType.CRASHES, "COMMODITY_SHOCK_STAGFLATION","1973-10-17", "OPEC embargo, inflation surge, Fed behind curve", "48% drawdown over 21 months, stagflation", 630, 2, ["oil","inflation","stagflation","commodity"]),
    LibraryEntry("v1_1987_black_mon", VolumeType.CRASHES, "PROGRAM_TRADING_CASCADE",    "1987-10-19", "Portfolio insurance, rising rates, overvaluation", "Single-day 22% drop, fast recovery", 90, 2, ["program_trading","rates","fast_recovery","ptj"]),
    LibraryEntry("v1_1994_bonds",     VolumeType.CRASHES, "RATE_SHOCK_BOND_MASSACRE",   "1994-02-04", "Fed surprise hike 300bps in 12 months, bond bloodbath", "Equity -10%, bonds -20%, emerging mkt crisis", 270, 1, ["rates","bonds","fed","EM"]),
    LibraryEntry("v1_1998_ltcm",      VolumeType.CRASHES, "LEVERAGE_UNWIND_SYSTEMIC",   "1998-08-17", "Russia default, LTCM near-failure, credit spreads explode", "20% equity drawdown, Fed cuts, fast recovery", 90, 1, ["leverage","credit","systemic","fed_cut"]),
    LibraryEntry("v1_2000_dotcom",    VolumeType.CRASHES, "VALUATION_DISLOCATION",      "2000-03-24", "NASDAQ at 100x earnings, IPO mania, insider selling", "NASDAQ -78% over 2.5 years, tech depression", 900, 2, ["valuation","tech","bubble","earnings"]),
    LibraryEntry("v1_2008_gfc",       VolumeType.CRASHES, "SYSTEMIC_CREDIT_FAILURE",    "2008-09-15", "Lehman collapse, interbank freeze, global credit crunch", "57% drawdown, 18-month bear, global recession", 540, 2, ["credit","systemic","banking","recession","gfc"]),
    LibraryEntry("v1_2018_vol",       VolumeType.CRASHES, "VOLATILITY_PRODUCT_UNWIND",  "2018-02-05", "Short-vol products imploding, VIX ETN blowup", "VIX +115% in 1 day, equity -10% in 2 weeks", 14, 1, ["volatility","etf","mechanical","fast_recovery"]),
    LibraryEntry("v1_2020_covid",     VolumeType.CRASHES, "EXOGENOUS_SHUTDOWN_SHOCK",   "2020-02-20", "Pandemic lockdowns, demand destruction, supply chain halt", "34% drawdown in 33 days, V-shaped recovery", 180, 2, ["exogenous","pandemic","fiscal","fed_unlimited"]),
    LibraryEntry("v1_2022_rate_shock",VolumeType.CRASHES, "RATE_DISLOCATION_MULTI_ASSET","2022-01-03","Fed 0% to 5.25% in 16 months, 60/40 worst year since 1930s", "S&P -25%, bonds -18%, duration destruction", 365, 2, ["rates","multi_asset","duration","bonds","fed_hiking"]),
]

# ── Volume II: Rate Cycle Regimes ─────────────────────────────────────────── #
VOLUME_II_RATE_CYCLES: List[LibraryEntry] = [
    LibraryEntry("v2_1954_hike_start",  VolumeType.RATE_CYCLES, "FED_HIKING_EARLY_CYCLE",    "1954-07-01", "Post-war expansion, Fed begins normalising", "Equity positive +18% next 12mo", 365, -1, ["hiking","early_cycle","equities_positive"]),
    LibraryEntry("v2_1977_hike_peak",   VolumeType.RATE_CYCLES, "FED_HIKING_PEAK_VOLCKER",   "1979-08-01", "Volcker shock, inflation 14%, fed funds 20%", "Deep recession, equity -20%, bonds massacred", 540, 2, ["volcker","inflation","peak_rate","recession"]),
    LibraryEntry("v2_1995_pivot",       VolumeType.RATE_CYCLES, "FED_PIVOT_SOFT_LANDING",    "1995-07-06", "Fed cuts after hiking, soft landing achieved", "S&P +34% next 12mo, goldilocks", 365, -1, ["pivot","soft_landing","goldilocks","equity_bull"]),
    LibraryEntry("v2_2004_hike_start",  VolumeType.RATE_CYCLES, "FED_HIKING_MEASURED_PACE",  "2004-06-30", "Measured 25bp hikes from 1%, steady economic expansion", "Equity muted, credit spreads tight, carry positive", 540, 0, ["hiking","measured","credit","carry"]),
    LibraryEntry("v2_2006_hike_peak",   VolumeType.RATE_CYCLES, "FED_HIKING_FINAL_STAGE",    "2006-06-29", "Fed at 5.25%, housing turning, yield curve inverting", "6-month lag to recession, equity still rising briefly", 180, 1, ["peak_rate","housing","yield_curve","inversion"]),
    LibraryEntry("v2_2015_hike_start",  VolumeType.RATE_CYCLES, "FED_LIFTOFF_POST_ZIRP",     "2015-12-16", "First hike after 7 years of zero rates, China turbulence", "EM sold off, equity volatile, dollar strengthened", 270, 1, ["liftoff","zirp_exit","dollar","EM_stress"]),
    LibraryEntry("v2_2019_pivot_cut",   VolumeType.RATE_CYCLES, "FED_MID_CYCLE_ADJUSTMENT",  "2019-07-31", "Insurance cuts, trade war uncertainty, yield curve uninverted", "Risk assets rallied, 2020 interrupted the cycle", 180, -1, ["insurance_cut","mid_cycle","trade_war"]),
    LibraryEntry("v2_2022_hike_fast",   VolumeType.RATE_CYCLES, "FED_EMERGENCY_HIKING",      "2022-03-16", "0 to 5.25% in 16 months, fastest since 1980s", "Multi-asset carnage, duration destruction", 480, 2, ["emergency_hiking","fastest_cycle","bonds","duration"]),
    LibraryEntry("v2_2023_pause",       VolumeType.RATE_CYCLES, "FED_HIKING_PAUSE",          "2023-07-26", "Fed holds at 5.25%, watching for inflation break", "Equity rally, soft landing narrative, credit tight", 270, 0, ["pause","higher_for_longer","soft_landing"]),
]

# ── Volume III: Bull Market Regimes ──────────────────────────────────────── #
VOLUME_III_BULL_REGIMES: List[LibraryEntry] = [
    LibraryEntry("v3_post_crash_recovery","VOLUME_III_BULL_REGIMES","POST_CRASH_EARLY_RECOVERY","2009-03-09","Credit crisis bottom, Fed QE begins, peak fear passed","Fastest equity recovery in history, +100% in 4 years",1460,-1,["recovery","qe","momentum","breadth_expansion"]),
    LibraryEntry("v3_2012_melt_up",       "VOLUME_III_BULL_REGIMES","MELT_UP_LTRO_QE",         "2012-09-01","Draghi 'whatever it takes', QE3 begins, TINA era","Low vol melt-up, low volume grind, buy-the-dip", 540,-1,["melt_up","low_vol","tina","qe","buy_dip"]),
    LibraryEntry("v3_2017_goldilocks",    "VOLUME_III_BULL_REGIMES","GOLDILOCKS_LOW_VOL",       "2017-01-01","Global sync growth, low inflation, VIX sub-10","VIX crushed, carry trades work, every asset positive",365,-1,["goldilocks","low_vol","synchronized_growth","carry"]),
    LibraryEntry("v3_late_cycle_2018",    "VOLUME_III_BULL_REGIMES","LATE_CYCLE_MOMENTUM",      "2018-01-01","Tax cuts, buyback surge, earnings peak, employment max","Last leg of bull, followed by Q4 correction",270,1,["late_cycle","buybacks","earnings_peak","employment_max"]),
    LibraryEntry("v3_2020_recovery",      "VOLUME_III_BULL_REGIMES","FISCAL_STIMULUS_RECOVERY", "2020-04-01","Unlimited Fed, $6T fiscal stimulus, retail trading surge","V-shaped recovery, meme stocks, SPAC mania",365,-1,["fiscal","retail","meme","spac","unlimited_qe"]),
    LibraryEntry("v3_2023_ai_melt_up",    "VOLUME_III_BULL_REGIMES","AI_DRIVEN_CONCENTRATION",  "2023-01-01","AI narrative, magnificent 7 concentration, breadth collapse","Index up 26%, equal-weight up 12%, divergence extreme",365,1,["ai","concentration","magnificent7","breadth_collapse"]),
]

# ── Volume IV: Currency Crises ────────────────────────────────────────────── #
VOLUME_IV_CURRENCY_CRISES: List[LibraryEntry] = [
    LibraryEntry("v4_1992_erm",       "VOLUME_IV_CURRENCY_CRISES","ERM_CRISIS_SOROS",          "1992-09-16","Soros breaks BoE, UK exits ERM, currency peg collapses","GBP -15% in days, UK equities rallied on devaluation",90,1,["currency_peg","soros","devaluation","contagion"]),
    LibraryEntry("v4_1997_asian",     "VOLUME_IV_CURRENCY_CRISES","ASIAN_CURRENCY_CONTAGION",  "1997-07-02","Thai baht devaluation, EM currency dominoes, IMF bailouts","EM equities -50%, contagion to Russia/Brazil",540,2,["em_crisis","contagion","imf","devaluation","currency"]),
    LibraryEntry("v4_2014_em_stress", "VOLUME_IV_CURRENCY_CRISES","EM_STRESS_TAPER_TANTRUM",   "2013-05-22","Bernanke hints taper, EM currencies collapse, capital flight","EM FX -20%, equities -30%, DXY rally",365,2,["taper_tantrum","em_capital_flight","dollar_strength"]),
    LibraryEntry("v4_2022_dollar",    "VOLUME_IV_CURRENCY_CRISES","DOLLAR_WRECKING_BALL",      "2022-09-01","DXY at 20yr high, yen intervention, EM debt stress","EUR/USD parity, JPY 150, EM debt crisis risk",270,2,["dollar","yen","parity","EM_debt","intervention"]),
    LibraryEntry("v4_2023_yen_carry", "VOLUME_IV_CURRENCY_CRISES","YEN_CARRY_UNWIND",          "2023-11-01","BOJ YCC tweak, yen carry unwind, global vol spike","Short-lived but signals end of suppressed vol era",30,1,["yen_carry","boj","carry_unwind","vol_spike"]),
]

# ── Volume V: Volatility Regimes ──────────────────────────────────────────── #
VOLUME_V_VOL_REGIMES: List[LibraryEntry] = [
    LibraryEntry("v5_vix_compress_14",  "VOLUME_V_VOL_REGIMES","VIX_EXTREME_COMPRESSION",    "2017-07-01","VIX sub-10, short-vol crowding, complacency extreme","Always precedes vol expansion — timing uncertain",180,1,["vix_low","complacency","short_vol_crowding","danger"]),
    LibraryEntry("v5_vix_spike_2010",   "VOLUME_V_VOL_REGIMES","FLASH_CRASH_VOL_SPIKE",       "2010-05-06","Algorithmic cascade, Dow -1000 intraday, then reversal","V-shaped recovery in 3 weeks — buy the spike",21,1,["flash_crash","algo","intraday","fast_recovery"]),
    LibraryEntry("v5_vix_40_2008",      "VOLUME_V_VOL_REGIMES","VOL_REGIME_CHANGE_SUSTAINED", "2008-10-10","VIX 80, sustained extreme vol, correlation 1.0 everything","Sustained high vol for 6 months, no safe havens",180,2,["regime_change","correlation_1","no_safe_haven","sustained"]),
    LibraryEntry("v5_vol_term_collapse","VOLUME_V_VOL_REGIMES","VOL_TERM_STRUCTURE_BACKWDN",  "2020-03-16","VIX curve inverted, near-term > long-term, panic peak","When curve normalises = bottom near",14,2,["backwardation","panic_peak","bottom_signal","term_structure"]),
    LibraryEntry("v5_low_vol_melt_up",  "VOLUME_V_VOL_REGIMES","LOW_VOL_MELT_UP_REGIME",      "2013-07-01","VIX 12-15, realized vol 8, options cheap, carry dominant","Strong for momentum strategies, fatal at turn",365,-1,["low_vol","carry","momentum_favoured","fragile"]),
]

# ── Volume VI: Economic Cycles ────────────────────────────────────────────── #
VOLUME_VI_ECON_CYCLES: List[LibraryEntry] = [
    LibraryEntry("v6_early_expansion", "VOLUME_VI_ECON_CYCLES","EARLY_CYCLE_EXPANSION",      "2009-06-01","Recession ended, credit healing, PMI recovering from trough","Strongest equity returns, small caps lead, cyclicals",480,-1,["early_cycle","pmis_turning","credit_healing","small_caps"]),
    LibraryEntry("v6_mid_expansion",   "VOLUME_VI_ECON_CYCLES","MID_CYCLE_ACCELERATION",     "2011-01-01","GDP above trend, employment recovering, earnings growing","Broad equity strength, low vol, momentum works",540,-1,["mid_cycle","broad_strength","momentum","earnings_growth"]),
    LibraryEntry("v6_late_expansion",  "VOLUME_VI_ECON_CYCLES","LATE_CYCLE_OVERHEATING",     "2018-01-01","Unemployment at multi-decade lows, inflation rising, Fed hiking","Defensives outperform, yield curve flattening, caution",270,1,["late_cycle","overheating","defensives","yield_curve_flat"]),
    LibraryEntry("v6_contraction",     "VOLUME_VI_ECON_CYCLES","CONTRACTION_RECESSION",      "2008-12-01","GDP negative, unemployment rising, credit contracting","Bear market, risk-off, bonds outperform equities",365,2,["recession","risk_off","bonds","unemployment_rising"]),
    LibraryEntry("v6_recovery_trough", "VOLUME_VI_ECON_CYCLES","TROUGH_TURNING_POINT",       "2001-11-01","Leading indicators turning, credit spreads peaking, PMI troughing","Counter-trend rallies, then sustained recovery begins",90,0,["trough","leading_indicators","pmis_trough","turning_point"]),
    LibraryEntry("v6_stagflation",     "VOLUME_VI_ECON_CYCLES","STAGFLATION_REGIME",         "1973-01-01","High inflation + low/negative growth, central bank powerless","Worst regime for traditional 60/40, commodities win",730,2,["stagflation","inflation","stagnation","commodities_outperform"]),
]

# ── Volume VII: Liquidity Events ──────────────────────────────────────────── #
VOLUME_VII_LIQUIDITY: List[LibraryEntry] = [
    LibraryEntry("v7_repo_2019",        "VOLUME_VII_LIQUIDITY","REPO_MARKET_STRESS",          "2019-09-17","Overnight repo rate spiked to 10%, bank reserves scarcity","Fed emergency repo injections, balance sheet expansion",30,1,["repo","short_term_rates","fed_balance_sheet"]),
    LibraryEntry("v7_money_mkt_2008",   "VOLUME_VII_LIQUIDITY","MONEY_MARKET_FREEZE",         "2008-09-16","Reserve Primary Fund broke the buck, money mkt panic","Government guarantee required, systemic liquidity stop",14,2,["money_market","buck_breaking","systemic","guarantee"]),
    LibraryEntry("v7_treasury_2020",    "VOLUME_VII_LIQUIDITY","TREASURY_MARKET_DYSfunc",     "2020-03-12","Treasury market bid-ask exploded, safest asset illiquid","Fed unlimited QE response, ultimate backstop revealed",7,2,["treasury","liquidity","basis_trade","fed_backstop"]),
    LibraryEntry("v7_libor_2007",       "VOLUME_VII_LIQUIDITY","INTERBANK_STRESS_EARLY",      "2007-08-09","LIBOR-OIS spread explodes, banks stop lending to each other","9-month warning before full crisis — early signal",270,2,["libor","interbank","early_warning","ois_spread"]),
    LibraryEntry("v7_svb_2023",         "VOLUME_VII_LIQUIDITY","REGIONAL_BANK_RUN",           "2023-03-08","SVB bank run, duration losses, depositor confidence break","Contagion to CS, Signature, First Republic — contained",30,1,["bank_run","duration","deposits","regional_banks"]),
]

# ── Volume VIII: Commodity Supercycles ────────────────────────────────────── #
VOLUME_VIII_COMMODITIES: List[LibraryEntry] = [
    LibraryEntry("v8_oil_1973_shock",   "VOLUME_VIII_COMMODITIES","OIL_EMBARGO_SHOCK",          "1973-10-16","OPEC embargo, oil 4x in weeks, energy rationing","Stagflation, equity -48%, energy equities outperform",730,2,["oil_shock","opec","stagflation","energy"]),
    LibraryEntry("v8_oil_2008_peak",    "VOLUME_VIII_COMMODITIES","OIL_DEMAND_DESTRUCTION_PEAK","2008-07-01","Oil $147, demand destruction, commodity bubble bursting","Oil -77% by December, deflationary impulse",180,2,["oil_peak","demand_destruction","deflation"]),
    LibraryEntry("v8_metals_2011",      "VOLUME_VIII_COMMODITIES","METALS_BULL_PEAK",            "2011-04-01","China demand, gold $1900, silver $50, commodity inflation","Metals top, dollar turn, EM growth slows",540,1,["gold","silver","metals","china","inflation"]),
    LibraryEntry("v8_oil_2014_collapse","VOLUME_VIII_COMMODITIES","SHALE_SUPPLY_OIL_CRASH",      "2014-06-01","Shale revolution, OPEC market share war, supply glut","Oil -70%, EM oil exporters stressed, HY energy blow-up",365,2,["shale","opec","oil_crash","hy_energy"]),
    LibraryEntry("v8_commodity_2020s",  "VOLUME_VIII_COMMODITIES","POST_COVID_COMMODITY_BULL",   "2021-01-01","Supply chain disruption, underinvestment, energy transition","Multi-year commodity bull, 40yr high inflation",730,-1,["commodity_bull","supply_chain","inflation","energy_transition"]),
]

# ── Volume IX: Geopolitical Shocks ────────────────────────────────────────── #
VOLUME_IX_GEOPOLITICAL: List[LibraryEntry] = [
    LibraryEntry("v9_gulf_war_1990",    "VOLUME_IX_GEOPOLITICAL","WAR_PREMIUM_OIL",            "1990-08-02","Iraq invades Kuwait, oil doubles, recession fears","-20% equity, then V-shaped when war short",180,1,["war","oil","recession","v_shape"]),
    LibraryEntry("v9_9_11_2001",        "VOLUME_IX_GEOPOLITICAL","EXOGENOUS_TERROR_SHOCK",     "2001-09-11","Terrorist attacks, markets closed 4 days, uncertainty peak","Market -15% reopening week, then recovery begin",90,2,["terror","exogenous","market_closure","uncertainty"]),
    LibraryEntry("v9_russia_ukraine",   "VOLUME_IX_GEOPOLITICAL","SANCTIONS_COMMODITY_SHOCK",  "2022-02-24","Russia invades Ukraine, unprecedented sanctions, commodity spike","Energy +60%, wheat +50%, inflation surge, EM stress",365,2,["sanctions","commodity","energy","wheat","EM"]),
    LibraryEntry("v9_china_tariffs",    "VOLUME_IX_GEOPOLITICAL","TRADE_WAR_UNCERTAINTY",      "2018-03-01","US-China tariffs, supply chain uncertainty, CEO confidence drop","Market volatile, range-bound, uncertainty premium",540,1,["trade_war","tariffs","uncertainty","supply_chain"]),
    LibraryEntry("v9_covid_lockdown",   "VOLUME_IX_GEOPOLITICAL","PANDEMIC_SHUTDOWN",          "2020-03-11","Global lockdowns, GDP -30% annualised Q2, demand stop","Fastest bear market in history, then record recovery",90,2,["pandemic","lockdown","exogenous","demand_stop"]),
]

# ── Volume X: Sector Rotation Regimes ────────────────────────────────────── #
VOLUME_X_SECTOR_ROTATION: List[LibraryEntry] = [
    LibraryEntry("v10_growth_to_value",   "VOLUME_X_SECTOR_ROTATION","GROWTH_TO_VALUE_ROTATION","2000-03-24","Tech bubble bursting, value emerges from decade-low relative","Value outperformed growth by 40%+ over next 5 years",1825,0,["growth_to_value","rotation","tech_bubble","value_decade"]),
    LibraryEntry("v10_defensive_flight",  "VOLUME_X_SECTOR_ROTATION","DEFENSIVE_ROTATION",      "2007-10-01","Late cycle, credit cracking, defensives start outperforming","Staples, utilities, healthcare lead as market tops",270,1,["defensive","late_cycle","staples","utilities","leading_signal"]),
    LibraryEntry("v10_reflation_2020",    "VOLUME_X_SECTOR_ROTATION","REFLATION_ROTATION",      "2020-11-01","Vaccine news, fiscal stimulus, reflation trade begins","Cyclicals, banks, energy, small caps surge; tech lags",365,-1,["reflation","cyclicals","banks","small_caps","vaccine"]),
    LibraryEntry("v10_ai_concentration",  "VOLUME_X_SECTOR_ROTATION","AI_MEGA_CAP_CONCENTRATION","2023-01-01","7 stocks = 30% of S&P, breadth collapse, AI premium","Index masks underlying weakness — dangerous divergence",365,1,["ai","concentration","breadth","magnificent7","divergence"]),
    LibraryEntry("v10_risk_off",          "VOLUME_X_SECTOR_ROTATION","RISK_OFF_ROTATION",       "2011-08-01","US downgrade, Europe crisis, global growth fear","Dollar, yen, gold, treasuries win — everything else loses",180,2,["risk_off","safe_haven","dollar","gold","treasuries"]),
    LibraryEntry("v10_value_2022",        "VOLUME_X_SECTOR_ROTATION","VALUE_RATE_REGIME_SHIFT", "2022-01-03","Rising rates, end of TINA, value and energy renaissance","Energy +65%, financials positive, growth -40%, bond proxies -30%",365,0,["value","rates","energy","tina_end","growth_down"]),
]

# ── Master registry ────────────────────────────────────────────────────────── #
ALL_ENTRIES: List[LibraryEntry] = (
    VOLUME_I_CRASHES +
    VOLUME_II_RATE_CYCLES +
    VOLUME_III_BULL_REGIMES +
    VOLUME_IV_CURRENCY_CRISES +
    VOLUME_V_VOL_REGIMES +
    VOLUME_VI_ECON_CYCLES +
    VOLUME_VII_LIQUIDITY +
    VOLUME_VIII_COMMODITIES +
    VOLUME_IX_GEOPOLITICAL +
    VOLUME_X_SECTOR_ROTATION
)

VOLUME_DESCRIPTIONS = {
    VolumeType.CRASHES:         "Dislocations & Crashes (20 events, 1929-2023)",
    VolumeType.RATE_CYCLES:     "Federal Reserve Rate Cycle Regimes",
    VolumeType.BULL_REGIMES:    "Bull Market Character Regimes",
    VolumeType.CURRENCY_CRISES: "Currency Crises & FX Dislocations",
    VolumeType.VOL_REGIMES:     "Volatility Regime States",
    VolumeType.ECON_CYCLES:     "Economic Cycle Phases",
    VolumeType.LIQUIDITY:       "Liquidity & Funding Market Stress",
    VolumeType.COMMODITIES:     "Commodity Supercycle Phases",
    VolumeType.GEOPOLITICAL:    "Geopolitical Shocks & Event Risk",
    VolumeType.SECTOR_ROTATION: "Sector Rotation Regime Patterns",
}


# ═════════════════════════════════════════════════════════════════════════════
# STORED PATTERN — carries both metadata and feature vector
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class StoredPattern:
    entry_id:       str
    volume:         str
    label:          str
    date:           str
    description:    str
    outcome:        str
    outcome_days:   int
    severity:       int
    tags:           List[str]
    features:       List[float]     # 23-dim normalised vector
    auto_learned:   bool = False
    added_at:       str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ═════════════════════════════════════════════════════════════════════════════
# LIBRARY QUERY RESULT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class VolumeMatch:
    """Best match within one volume."""
    volume:       str
    label:        str
    entry_id:     str
    date:         str
    similarity:   float
    severity:     int
    outcome:      str
    tags:         List[str]

@dataclass
class LibraryInsight:
    """
    Rich multi-volume output from the library query.

    This is what makes the library game-changing:
    Not just "score=0.7 → warning" but:
    "Current conditions match LATE_CYCLE_FED_HIKING (Vol II, similarity=0.81)
     AND DEFENSIVE_ROTATION (Vol X, similarity=0.74)
     AND VOL_EXTREME_COMPRESSION (Vol V, similarity=0.68).
     Historically this triple-match preceded a 15-25% drawdown within 12 months
     in 4 of the 5 times it occurred."
    """
    primary_regime:    str           # dominant match label
    primary_volume:    str           # which volume drove it
    primary_similarity: float
    threat_score:      float         # composite 0-1 from all volumes
    threat_level:      str           # NORMAL/ELEVATED/WARNING/DANGER/CRITICAL
    size_modifier:     float
    volume_matches:    List[VolumeMatch]   # best match per volume
    top_matches:       List[VolumeMatch]   # top-5 across all volumes
    converging_signal: bool          # True if 3+ volumes show >0.60 similarity
    advisory:          str
    action_summary:    str
    timestamp:         str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ═════════════════════════════════════════════════════════════════════════════
# THE ALEXANDRIAN LIBRARY
# ═════════════════════════════════════════════════════════════════════════════

class AlexandrianLibrary:
    """
    The complete market knowledge base.

    Build once, query every session, learn continuously.

    Query returns a LibraryInsight that tells you:
      1. Which specific historical pattern best matches current conditions
      2. Which VOLUME is most activated (crashes? rate cycles? vol regime?)
      3. How many volumes are simultaneously showing high similarity
         (convergence = strongest signal)
      4. What historically followed each match
      5. The automatic size modifier for this session

    "Not just a warning system. Operational intelligence."
    """

    THREAT_THRESHOLDS = {
        'NORMAL':   (0.00, 0.35, 1.00),
        'ELEVATED': (0.35, 0.52, 0.80),
        'WARNING':  (0.52, 0.68, 0.50),
        'DANGER':   (0.68, 0.82, 0.25),
        'CRITICAL': (0.82, 1.00, 0.00),
    }

    def __init__(self):
        self._patterns: List[StoredPattern] = []
        self._last_insight: Optional[LibraryInsight] = None
        self._load()

    def _load(self):
        if LIBRARY_PATH.exists():
            try:
                with open(LIBRARY_PATH) as f:
                    data = json.load(f)
                self._patterns = [StoredPattern(**p) for p in data.get('patterns', [])]
                logger.info(f"AlexandrianLibrary: loaded {len(self._patterns)} patterns across {self.n_volumes} volumes")
            except Exception as e:
                logger.warning(f"AlexandrianLibrary: load failed ({e}), starting empty")
                self._patterns = []

    def _save(self):
        LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LIBRARY_PATH, 'w') as f:
            json.dump({
                'patterns': [asdict(p) for p in self._patterns],
                'updated': datetime.utcnow().isoformat(),
                'n_patterns': len(self._patterns),
                'volumes': list({p.volume for p in self._patterns}),
            }, f, indent=2)

    # ── Build from yfinance ────────────────────────────────────────────────── #

    def build_from_history(self, force_rebuild: bool = False) -> int:
        """
        Populate all 10 volumes from yfinance historical data.

        For each entry: fetch the 300-day window ending on the event date,
        extract the 23-feature precursor vector, store as StoredPattern.

        Returns total patterns successfully built.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("AlexandrianLibrary.build: pip install yfinance required")
            return 0

        existing_ids = {p.entry_id for p in self._patterns}
        built = 0

        for entry in ALL_ENTRIES:
            if entry.entry_id in existing_ids and not force_rebuild:
                continue

            try:
                event_dt = datetime.strptime(entry.date[:10], '%Y-%m-%d')
                fetch_start = (event_dt - timedelta(days=400)).strftime('%Y-%m-%d')
                fetch_end   = min(event_dt + timedelta(days=5),
                                  datetime.utcnow()).strftime('%Y-%m-%d')

                # Try SPY first, fall back to ^GSPC for old events
                spy = yf.download('SPY', start=fetch_start, end=fetch_end,
                                   progress=False, auto_adjust=True)
                if spy is None or len(spy) < 100:
                    spy = yf.download('^GSPC', start=fetch_start, end=fetch_end,
                                       progress=False, auto_adjust=True)
                if spy is None or len(spy) < 50:
                    logger.debug(f"Library: insufficient SPY data for {entry.entry_id}")
                    continue

                spy_arr = spy['Close'].values.astype(float).squeeze()

                # Optional: VIX, gold, DXY
                vix_arr = gold_arr = None
                try:
                    vix = yf.download('^VIX', start=fetch_start, end=fetch_end,
                                       progress=False, auto_adjust=True)
                    if vix is not None and len(vix) >= 20:
                        vix_arr = vix['Close'].values.astype(float).squeeze()
                except Exception:
                    pass
                try:
                    gld = yf.download('GLD', start=fetch_start, end=fetch_end,
                                       progress=False, auto_adjust=True)
                    if gld is not None and len(gld) >= 20:
                        gold_arr = gld['Close'].values.astype(float).squeeze()
                except Exception:
                    pass

                feats = extract_features(spy_arr, vix_prices=vix_arr, gold_prices=gold_arr)
                if feats is None:
                    continue

                # Remove old version if rebuilding
                self._patterns = [p for p in self._patterns if p.entry_id != entry.entry_id]

                pattern = StoredPattern(
                    entry_id=entry.entry_id,
                    volume=entry.volume,
                    label=entry.label,
                    date=entry.date[:10],
                    description=entry.description,
                    outcome=entry.outcome,
                    outcome_days=entry.outcome_days,
                    severity=entry.severity,
                    tags=list(entry.tags),
                    features=feats.tolist(),
                )
                self._patterns.append(pattern)
                built += 1

                vol_short = entry.volume.replace('VOLUME_', 'V')
                logger.info(f"Library [{vol_short}]: {entry.label} ({entry.date[:7]})")

            except Exception as e:
                logger.warning(f"Library: error building {entry.entry_id}: {e}")

        if built > 0:
            self._save()
        logger.info(f"AlexandrianLibrary build complete: {built} new | {len(self._patterns)} total")
        return built

    # ── Query ─────────────────────────────────────────────────────────────── #

    def query(
        self,
        spy_prices: np.ndarray,
        vix_prices: Optional[np.ndarray] = None,
        gold_prices: Optional[np.ndarray] = None,
        hy_spread: Optional[np.ndarray] = None,
        dxy_prices: Optional[np.ndarray] = None,
        top_n: int = 5,
    ) -> LibraryInsight:
        """
        Query the library with current market conditions.

        Returns a LibraryInsight that tells you:
          - Which historical pattern you most resemble RIGHT NOW
          - Which volume is most activated
          - Whether multiple volumes are converging (strongest signal)
          - What has historically followed each match
          - The automatic size modifier for today's session

        This runs in milliseconds once patterns are loaded.
        """
        if not self._patterns:
            return LibraryInsight(
                primary_regime='LIBRARY_EMPTY',
                primary_volume='',
                primary_similarity=0.0,
                threat_score=0.0,
                threat_level='NORMAL',
                size_modifier=1.0,
                volume_matches=[],
                top_matches=[],
                converging_signal=False,
                advisory='Library empty — run build_from_history()',
                action_summary='No data available',
            )

        current_feats = extract_features(spy_prices, vix_prices, gold_prices, hy_spread, dxy_prices)
        if current_feats is None:
            return LibraryInsight(
                primary_regime='INSUFFICIENT_DATA',
                primary_volume='',
                primary_similarity=0.0,
                threat_score=0.0,
                threat_level='NORMAL',
                size_modifier=1.0,
                volume_matches=[],
                top_matches=[],
                converging_signal=False,
                advisory='Insufficient price history for library query',
                action_summary='Need 200+ bars of SPY data',
            )

        # ── Compute similarity against every stored pattern ─────────────── #
        all_matches: List[VolumeMatch] = []
        for pattern in self._patterns:
            hist = np.array(pattern.features)
            if len(hist) != N_FEATURES:
                continue
            sim = _cosine_similarity(current_feats, hist)
            all_matches.append(VolumeMatch(
                volume=pattern.volume,
                label=pattern.label,
                entry_id=pattern.entry_id,
                date=pattern.date,
                similarity=sim,
                severity=pattern.severity,
                outcome=pattern.outcome,
                tags=pattern.tags,
            ))

        all_matches.sort(key=lambda m: m.similarity, reverse=True)

        # ── Best match per volume ────────────────────────────────────────── #
        volume_matches: List[VolumeMatch] = []
        seen_volumes: set = set()
        for m in all_matches:
            if m.volume not in seen_volumes:
                volume_matches.append(m)
                seen_volumes.add(m.volume)

        # ── Composite threat score ───────────────────────────────────────── #
        # Weights: top match highest, exponential decay, severity amplification
        top = all_matches[:top_n]
        if not top:
            composite = 0.0
        else:
            weights = np.array([math.exp(-0.4 * i) for i in range(len(top))])
            sims = np.array([m.similarity for m in top])
            composite = float(np.average(sims, weights=weights))

            # Convergence amplifier: multiple volumes all showing > 0.60
            high_sim_volumes = {m.volume for m in all_matches if m.similarity > 0.60}
            n_converging = len(high_sim_volumes)
            if n_converging >= 3:
                # 3+ volumes converging = strongest possible signal
                composite = min(1.0, composite * (1.0 + 0.10 * (n_converging - 2)))

            # Negative-severity matches (bull regimes) reduce composite score
            bull_boost = sum(0.03 for m in top if m.severity == -1 and m.similarity > 0.70)
            composite = max(0.0, composite - bull_boost)

        # ── Threat level ─────────────────────────────────────────────────── #
        threat_level, size_modifier = 'NORMAL', 1.0
        for level, (lo, hi, mod) in self.THREAT_THRESHOLDS.items():
            if lo <= composite < hi:
                threat_level = level
                size_modifier = mod
                break

        # ── Bull regime floor — convergence amplifier fix ─────────────────── #
        # The convergence amplifier can push composite to CRITICAL (0.00x) even
        # when the PRIMARY match is a bull regime pattern (severity == -1).
        # This happened in May 2026: GOLDILOCKS_LOW_VOL was top match at 0.9338
        # but all 10 volumes converging amplified composite to 1.0 → 0.00x size.
        # Fix: when the best single match is a bull pattern, the system is reading
        # a benign environment — floor size at 0.50x so equity can still trade.
        # The CRITICAL warning still fires and shows on the dashboard.
        primary = top[0] if top else None
        if primary is not None and primary.severity == -1 and size_modifier == 0.0:
            size_modifier = 0.50
        converging = len({m.volume for m in all_matches[:15] if m.similarity > 0.60}) >= 3

        # ── Build advisory ────────────────────────────────────────────────── #
        if primary:
            advisory_parts = [
                f"LIBRARY QUERY | {threat_level} | score={composite:.3f}",
                f"Primary match: [{primary.volume.replace('VOLUME_','')}] {primary.label} "
                f"({primary.date}) sim={primary.similarity:.3f}",
                f"What followed: {primary.outcome[:80]}",
            ]
            # Show top 3 cross-volume matches
            cross_vol = [m for m in volume_matches[:4] if m.volume != primary.volume][:2]
            for m in cross_vol:
                advisory_parts.append(
                    f"Also: [{m.volume.replace('VOLUME_','')}] {m.label} sim={m.similarity:.3f}")
            if converging:
                advisory_parts.append(
                    f"⚠ CONVERGENCE: {len({m.volume for m in all_matches[:15] if m.similarity > 0.60})} "
                    f"volumes showing >0.60 similarity simultaneously")
            advisory = ' | '.join(advisory_parts)
        else:
            advisory = f"Library: {threat_level} | No strong matches"

        # ── Action summary ────────────────────────────────────────────────── #
        if threat_level == 'CRITICAL':
            action = "HALT new positions. Match to severe historical precursor. Await human review."
        elif threat_level == 'DANGER':
            action = f"25% size. Dominant pattern: {primary.label if primary else 'unknown'}. Tighten stops."
        elif threat_level == 'WARNING':
            action = f"50% size. {primary.label if primary else ''} precursor active. Extra confirmation required."
        elif threat_level == 'ELEVATED':
            action = f"75% size. Monitoring {primary.label if primary else ''} similarity."
        else:
            action = "Full size. No significant historical pattern match."

        insight = LibraryInsight(
            primary_regime=primary.label if primary else 'NORMAL',
            primary_volume=primary.volume if primary else '',
            primary_similarity=primary.similarity if primary else 0.0,
            threat_score=composite,
            threat_level=threat_level,
            size_modifier=size_modifier,
            volume_matches=volume_matches,
            top_matches=top,
            converging_signal=converging,
            advisory=advisory,
            action_summary=action,
        )
        self._last_insight = insight
        self._log_insight(insight)
        return insight

    def _log_insight(self, insight: LibraryInsight):
        if insight.threat_level == 'NORMAL':
            return
        level_log = {
            'ELEVATED': logger.info,
            'WARNING':  logger.warning,
            'DANGER':   logger.error,
            'CRITICAL': logger.critical,
        }
        log_fn = level_log.get(insight.threat_level, logger.info)
        log_fn(f"[ALEXANDRIAN_LIBRARY] {insight.advisory}")

    # ── Auto-learning ─────────────────────────────────────────────────────── #

    def learn(
        self,
        event_id: str,
        volume: str,
        label: str,
        date: str,
        description: str,
        outcome: str,
        severity: int,
        tags: List[str],
        spy_prices: np.ndarray,
        vix_prices: Optional[np.ndarray] = None,
        gold_prices: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Add a new pattern to any volume from live data.
        Called when the system experiences or identifies a new market state.
        """
        feats = extract_features(spy_prices, vix_prices, gold_prices)
        if feats is None:
            return False

        self._patterns = [p for p in self._patterns if p.entry_id != event_id]
        self._patterns.append(StoredPattern(
            entry_id=event_id, volume=volume, label=label, date=date,
            description=description, outcome=outcome, outcome_days=0,
            severity=severity, tags=tags, features=feats.tolist(),
            auto_learned=True,
        ))
        self._save()
        logger.info(f"[Library] Auto-learned: {event_id} → {volume} | total={len(self._patterns)}")
        return True

    # ── Convenience interface ─────────────────────────────────────────────── #

    def get_size_modifier(self, spy_prices: np.ndarray,
                           vix_prices: Optional[np.ndarray] = None, **kwargs) -> Tuple[float, str]:
        """Single-call orchestrator interface: (size_multiplier, advisory)."""
        insight = self.query(spy_prices, vix_prices, **kwargs)
        return insight.size_modifier, insight.advisory

    def volume_summary(self) -> Dict[str, int]:
        """Count patterns per volume."""
        counts: Dict[str, int] = {}
        for p in self._patterns:
            counts[p.volume] = counts.get(p.volume, 0) + 1
        return counts

    @property
    def n_patterns(self) -> int:
        return len(self._patterns)

    @property
    def n_volumes(self) -> int:
        return len({p.volume for p in self._patterns})

    @property
    def total_registered_entries(self) -> int:
        return len(ALL_ENTRIES)

    def describe(self) -> str:
        auto = sum(1 for p in self._patterns if p.auto_learned)
        vols = self.volume_summary()
        vol_str = ', '.join(f"V{k.split('_')[1]}={v}" for k, v in sorted(vols.items()))
        return (f"AlexandrianLibrary: {len(self._patterns)}/{len(ALL_ENTRIES)} patterns "
                f"({auto} auto-learned) | {vol_str}")
