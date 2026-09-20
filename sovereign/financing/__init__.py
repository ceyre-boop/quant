"""Financing measurement — what the broker actually takes.

At $15k with a ~5%/yr carry premium, the broker's swap is not a cost line, it is
the deciding variable (TICK-024). This package measures it instead of modelling
it. Read-only: it reads rates and charged transactions; it never opens, modifies
or closes a position.
"""
