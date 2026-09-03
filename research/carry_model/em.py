"""UNSEEN UNIVERSE for HYP-118: emerging-market currencies (MXN ZAR BRL KRW INR) against three funding
currencies (USD JPY EUR) — 15 signed positions per month, FRED only. No row of this universe was read
before HYP-118 was sealed (data guard EM_HOLDOUT_UNLOCK)."""
from __future__ import annotations

import os
from itertools import product

import numpy as np
import pandas as pd

from research.carry_model.data_fred import fred, month_end_spot, month_rates, month_cpi

EM = {"MXN": ("DEXMXUS", "IRSTCI01MXM156N", "IR3TIB01MXM156N", "MEXCPIALLMINMEI"),
      "ZAR": ("DEXSFUS", "IRSTCI01ZAM156N", "IR3TIB01ZAM156N", "ZAFCPIALLMINMEI"),
      "BRL": ("DEXBZUS", "IRSTCI01BRM156N", None, "BRACPIALLMINMEI"),
      "KRW": ("DEXKOUS", "IRSTCI01KRM156N", "IR3TIB01KRM156N", "KORCPIALLMINMEI"),
      "INR": ("DEXINUS", "IRSTCI01INM156N", None, "INDCPIALLMINMEI")}
FUND = ["USD", "JPY", "EUR"]
HAIRCUT, COST_M = 0.30, 0.0006 * 2        # EM spreads: 6 bp per leg per month (double G10), frozen


def guard_ok() -> bool:
    return os.environ.get("EM_HOLDOUT_UNLOCK") == "1"


def build_em() -> tuple[pd.DataFrame, pd.Series]:
    if not guard_ok():
        raise SystemExit("EM universe is a sealed holdout — set EM_HOLDOUT_UNLOCK=1 only from the HYP-118 test after gate zero")
    S = month_end_spot(); R = month_rates(); C = month_cpi()
    for c, (sp, rt, fb, cpi) in EM.items():
        s = fred(sp); S[c] = (1 / s).resample("ME").last()                # all DEX*US quoted as ccy per USD
        r = fred(rt); r = r.combine_first(fred(fb)) if fb else r; R[c] = r.resample("ME").last().ffill(limit=6)
        C[c] = fred(cpi).resample("ME").last().ffill(limit=4)
    idx = S.index.intersection(R.index); S, R, C = S.reindex(idx), R.reindex(idx), C.reindex(idx)
    lv = np.log(S); dlog = lv.diff(); r_prev = R.shift(1)
    usd_idx = -dlog[["EUR", "GBP", "JPY", "AUD", "CAD", "CHF"]].mean(axis=1)
    vix = fred("VIXCLS").resample("ME").last().reindex(idx)
    rows = []
    for a, b in product(EM.keys(), FUND):
        spot_ab = lv[a] - lv[b]; carry = (R[a] - R[b]).shift(1)
        target = (dlog[a] - dlog[b]) + (carry / 1200.0) * (1 - HAIRCUT) - COST_M
        rp = spot_ab.diff(); real_ab = spot_ab + np.log(C[b]) - np.log(C[a])
        rows.append(pd.DataFrame({
            "pair": f"{a}/{b}", "a": a, "b": b, "target": target, "carry": carry,
            "mom12": rp.rolling(12).sum().shift(1), "value": -(real_ab - real_ab.shift(60)).shift(1),
            "rvol3": rp.rolling(3).std().shift(1) * np.sqrt(12), "vix_prev": vix.shift(1),
            "dollar_beta": rp.rolling(36).cov(usd_idx).shift(1) / usd_idx.rolling(36).var().shift(1)}, index=idx))
    P = pd.concat(rows).reset_index().rename(columns={"index": "date"})
    P = P.dropna(subset=["target", "carry", "mom12", "rvol3"])
    P = P[P["date"] >= "1997-01-31"]                                          # BRL rate begins 1996-10; post-tequila
    return P, vix
