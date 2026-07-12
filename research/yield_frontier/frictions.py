"""Coarse friction models — applied even in mining; pessimistic scenario is the headline.

Constants per Plans/immutable-wondering-alpaca.md §Frictions. These are deliberately
crude tiers, not broker quotes; the gauntlet preregs freeze refined versions.
"""


def htb_apr(gap: float) -> float:
    """Hard-to-borrow fee APR tiered by how far the stock has gapped."""
    if gap < 0.5:
        return 0.50
    if gap < 1.0:
        return 1.50
    if gap < 1.5:
        return 3.00
    return 5.00


def short_borrow_cost(gap: float, days_held: float) -> float:
    """Fractional cost of carry for a short held overnight(s). Intraday-only shorts
    pay no borrow fee (locate still required — see locate_fill_prob)."""
    return htb_apr(gap) / 365.0 * days_held


def locate_fill_prob(gap: float) -> float:
    """Fraction of signalled events assumed actually shortable (locate found)."""
    return 0.50 if gap >= 1.0 else 0.75


EQ_SLIPPAGE = 0.0025          # 25 bps per side on smallcap fills
EQ_PARTICIPATION_CAP = 0.01   # position <= 1% of by-10:30 dollar volume

NQ_TICK = 0.25
NQ_SLIP_TICKS_PER_SIDE = 1
NQ_RT_COST_PTS = {            # slippage (2 ticks RT) + commission in index points
    "MNQ": 2 * NQ_SLIP_TICKS_PER_SIDE * NQ_TICK + 0.74 / 2.0,   # $2/pt  -> 0.87 pt
    "NQ": 2 * NQ_SLIP_TICKS_PER_SIDE * NQ_TICK + 2.50 / 20.0,   # $20/pt -> 0.625 pt
}

OPT_COMMISSION_PER_LEG_SIDE = 0.65   # dollars per contract per leg per side
OPT_K_GRID = (0.25, 0.5, 1.0)        # fill at mid -/+ k*half_spread (against us)


def option_fill(mid: float, half_spread: float, k: float, selling: bool) -> float:
    """Price received (selling) or paid (buying) with k*half-spread against us."""
    return max(mid - k * half_spread, 0.0) if selling else mid + k * half_spread
