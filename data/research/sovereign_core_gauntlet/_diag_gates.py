"""Diagnostic: tally where the Sovereign Core pipeline rejects entries, in-sample."""
import sys, json, types
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import logging
logging.disable(logging.CRITICAL)

# reuse the frozen yf stub + helpers from the perm script
import scripts.permutation_test_sovereign as P
P._install_yf_stub()

EQ = ROOT / "data" / "cache" / "equity"
manifest = json.loads((EQ / "manifest.json").read_text())
symbols = list(manifest["symbols"].keys())
for s in set(symbols) | {"SPY"}:
    P._FROZEN[s] = P._load_frozen(s)

from train_core import _build_records_from_df
from sovereign.orchestrator import SovereignOrchestrator
from contracts.types import Direction
from config.loader import params

print("regime params:", json.dumps(params.get("regime", {}), default=str))
print("slow_passing_features:", params.get("factor_zoo", {}).get("slow_passing_features"))
print("fast_passing_features:", params.get("factor_zoo", {}).get("fast_passing_features"))

all_records = []
insample = {}
for s in symbols:
    df = P._FROZEN[s].loc[P.IN_SAMPLE_START:P.IN_SAMPLE_END]
    insample[s] = df
    all_records.extend(_build_records_from_df(s, df))

orch = SovereignOrchestrator(mode="paper")
orch.train(all_records)

# regime label + router prediction distribution on SPY
df = insample["SPY"]
recs = _build_records_from_df("SPY", df)
hurst_short = [r.regime.hurst_short for r in recs]
print(f"\nSPY hurst_short: min {min(hurst_short):.3f} max {max(hurst_short):.3f} mean {np.mean(hurst_short):.3f}")
print("label dist (rule):", Counter(orch.router._label_regime(r) for r in recs))

tally = Counter()
router_regimes = Counter()
for s in symbols:
    df = insample[s]
    recs = _build_records_from_df(s, df)
    atr = P._atr14(df)
    closes = df["close"].to_numpy(float)
    P._SIM_TS = None
    for k, rec in enumerate(recs):
        i = P.LOOKBACK + k
        P._SIM_TS = df.index[i] if i < len(df) else df.index[-1]
        ro = orch.router.classify(rec)
        router_regimes[ro.regime] += 1
        if ro.regime == "FLAT" or ro.specialist_to_run is None:
            tally["router_FLAT"] += 1; continue
        spec = orch.specialists.get(ro.specialist_to_run)
        if spec is None:
            tally["no_specialist"] += 1; continue
        bias = spec.predict(rec)
        if bias.direction == Direction.NEUTRAL:
            tally["specialist_NEUTRAL"] += 1; continue
        atr_i = atr[i] if i < len(atr) and not np.isnan(atr[i]) else closes[i]*0.02
        try:
            r = orch.risk.compute(bias=bias, router=ro, account_equity=100000.0,
                                         atr=atr_i, entry_price=closes[i])
        except Exception as e:
            tally[f"risk_err:{type(e).__name__}"] += 1; continue
        if r.position_size <= 0:
            tally["risk_size0:"+r.stop_method] += 1; continue
        if not r.ev_positive:
            tally["risk_EV_neg"] += 1; continue
        tally["WOULD_ENTER"] += 1

print("\nrouter regime dist:", dict(router_regimes))
print("gate tally:", dict(tally))
print("risk_engine attr present:", hasattr(orch, "risk_engine"))
