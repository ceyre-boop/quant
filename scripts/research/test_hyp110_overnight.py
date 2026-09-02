#!/usr/bin/env python3
"""HYP-110 — overnight partition on ten liquid ETFs. THE test. Runs ONCE.

Everything here is fixed by data/research/preregister/HYP-110.json (sealed
d981bf1d43170fe0). Gate zero asserts the hash and PREREGISTERED status before a
single partition is computed, and again after. Writes the verdict to the ledger
exactly once as ADJUDICATED. If it fails, it is dead. No re-run.

  .venv313/bin/python scripts/research/test_hyp110_overnight.py --gate-only   # wiring check, computes nothing
  .venv313/bin/python scripts/research/test_hyp110_overnight.py               # THE run
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sovereign.discovery.cpcv import combinatorial_purged_splits  # noqa: E402
from sovereign.discovery.gate import deflated_sharpe_ratio        # noqa: E402

PREREG = ROOT / "data" / "research" / "preregister" / "HYP-110.json"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
DATA = ROOT / "data" / "cache" / "daily_universe"
OUT = ROOT / "data" / "research" / "hyp110"


def _canonical_hash(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "hash_lock"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def gate_zero(label: str) -> dict:
    doc = json.loads(PREREG.read_text())
    if _canonical_hash(doc) != doc["hash_lock"]:
        raise SystemExit(f"GATE ZERO FAILED ({label}): prereg hash mismatch — do not proceed")
    entry = next((e for e in json.loads(LEDGER.read_text()) if e.get("id") == doc["id"]), None)
    if entry is None or entry.get("hash_lock") != doc["hash_lock"]:
        raise SystemExit(f"GATE ZERO FAILED ({label}): ledger does not match prereg")
    if label == "start" and entry.get("status") != "PREREGISTERED":
        raise SystemExit(f"GATE ZERO FAILED: status is {entry.get('status')!r}, not PREREGISTERED — "
                         "this hypothesis has already been adjudicated; it does not run twice")
    print(f"[gate zero {label}] {doc['id']} hash {doc['hash_lock'][:16]} OK")
    return doc


def load(sym: str, window: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"{sym}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= window[0]) & (df["date"] <= window[1])].sort_values("date").set_index("date")
    o, c = df["open"].astype(float), df["close"].astype(float)
    return pd.DataFrame({
        "on": np.log(o / c.shift(1)),       # overnight  ln(open_t / close_{t-1})
        "id": np.log(c / o),                # intraday   ln(close_t / open_t)
        "cc": np.log(c / c.shift(1)),       # close-to-close (incumbent)
        "stale": (o == c),
    }).dropna()


def sharpe(r: np.ndarray) -> float:
    s = r.std(ddof=1)
    return float(r.mean() / s * np.sqrt(252)) if s > 0 else 0.0


def max_dd(r: np.ndarray) -> float:
    eq = np.cumprod(1 + r)
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def boot_idx(n: int, L: int, draws: int, rng) -> np.ndarray:
    """Stationary block bootstrap index matrix (draws × n), Politis–Romano, p=1/L. Joint resampling."""
    out = np.empty((draws, n), dtype=int)
    for d in range(draws):
        pos = 0
        while pos < n:
            start = rng.integers(n)
            length = rng.geometric(1.0 / L)
            take = min(length, n - pos)
            out[d, pos:pos + take] = (start + np.arange(take)) % n
            pos += take
    return out


def ci(v: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def main(gate_only: bool = False) -> int:
    doc = gate_zero("start")
    if gate_only:
        print("gate-only: wiring OK, nothing computed"); return 0
    P, win, warm = doc["frozen_parameters"], doc["data"]["window"], doc["frozen_parameters"]["warmup_sessions"]
    bp = P["round_trip_bp"] / 1e4
    L, seed, draws, floor = P["block_L"], P["seed"], P["draws"], P["floor_per_day"]
    rng = np.random.default_rng(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\nHYP-110 — {doc['name']}")
    print(f"frozen: cost={P['round_trip_bp']}bp/instrument-day warm={warm} L={L} draws={draws} "
          f"seed={seed} cpcv={P['cpcv']} embargo={P['embargo_sessions']} n_trials={P['n_trials']}\n")

    # ── series ───────────────────────────────────────────────────────────────
    frames = {s: load(s, win) for s in doc["instrument_set"]}
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    idx = idx[warm:]
    ON = pd.DataFrame({s: f["on"].reindex(idx) for s, f in frames.items()})
    ID = pd.DataFrame({s: f["id"].reindex(idx) for s, f in frames.items()})
    CC = pd.DataFrame({s: f["cc"].reindex(idx) for s, f in frames.items()})
    STALE = pd.DataFrame({s: f["stale"].reindex(idx) for s, f in frames.items()})
    n = len(idx)
    years = idx.year

    # data-quality abort (pre-declared)
    stale_pct = STALE.mean()
    data_abort = bool((stale_pct > 0.01).any())
    print("stale-open (open == close) share per instrument:")
    for s, v in stale_pct.items():
        print(f"  {s:5s} {v*100:.2f}%{'  <-- ABORT' if v > 0.01 else ''}")

    bh = CC.mean(axis=1)                             # THE INCUMBENT
    on_gross = ON.mean(axis=1)
    id_gross = ID.mean(axis=1)
    on_net = on_gross - bp                           # one round trip per instrument-day, EW -> bp per day
    delta = on_net - bh
    print(f"\nINCUMBENT (EW buy-and-hold, {idx[0].date()} → {idx[-1].date()}, {n} sessions):")
    print(f"  total {float(np.expm1(bh.sum()))*100:+.1f}%   Sharpe {sharpe(bh.values):.3f}   "
          f"maxDD {max_dd(bh.values)*100:.1f}%   {bh.mean()*100:+.4f}%/day")
    print(f"OVERNIGHT net:")
    print(f"  total {float(np.expm1(on_net.sum()))*100:+.1f}%   Sharpe {sharpe(on_net.values):.3f}   "
          f"maxDD {max_dd(on_net.values)*100:.1f}%   {on_net.mean()*100:+.4f}%/day   "
          f"(gross {on_gross.mean()*100:+.4f}%/day, intraday {id_gross.mean()*100:+.4f}%/day)\n")

    I = boot_idx(n, L, draws, rng)

    # (a) partition
    part = float(on_gross.mean() - id_gross.mean())
    bpart = np.array([on_gross.values[ix].mean() - id_gross.values[ix].mean() for ix in I])
    alo, ahi = ci(bpart)
    a_pass = alo > 0
    print("── (a) PARTITION  mean(overnight) − mean(intraday), gross ──")
    print(f"  {part*100:+.4f}%/day   95% CI [{alo*100:+.4f}%, {ahi*100:+.4f}%]   {'PASS' if a_pass else 'FAIL -> KILL_STRUCTURE'}")

    # (b) delta
    dsh = sharpe(on_net.values) - sharpe(bh.values)
    bdsh = np.array([sharpe(on_net.values[ix]) - sharpe(bh.values[ix]) for ix in I])
    blo, bhi = ci(bdsh)
    b1 = blo > 0
    dsr_sr, dsr_prob = deflated_sharpe_ratio(sharpe(delta.values), n_trials=P["n_trials"], n_obs=n)
    b2 = dsr_prob >= 0.95
    print("── (b) DELTA ──")
    print(f"  dSharpe (overnight_net − incumbent) = {dsh:+.3f}   95% CI [{blo:+.3f}, {bhi:+.3f}]   {'PASS' if b1 else 'FAIL'}")
    print(f"  delta series Sharpe {sharpe(delta.values):+.3f}   DSR@{P['n_trials']} deflated {dsr_sr:+.3f} prob {dsr_prob:.3f}   {'PASS' if b2 else 'FAIL'}")

    # (c) folds
    dates = idx.values
    exit_ = np.append(dates[1:], [dates[-1]])
    folds = []
    fold_err = False
    try:
        for tr, te in combinatorial_purged_splits(dates, exit_, n_groups=6, test_groups=2,
                                                  embargo_frac=P["embargo_sessions"] / n):
            folds.append(sharpe(on_net.values[te]) - sharpe(bh.values[te]))
    except Exception as e:  # pre-declared: a fold error is INCONCLUSIVE
        fold_err = True; print("  fold error:", e)
    folds = np.asarray(folds)
    c_pos = int((folds > 0).sum())
    print(f"── (c) 15 PURGED FOLDS ──")
    for i, d in enumerate(folds, 1):
        print(f"  fold {i:2d}  dSharpe {d:+.3f}")
    print(f"  positive: {c_pos}/15   mean {folds.mean() if len(folds) else float('nan'):+.3f}")

    # (g) golden rule per instrument
    print("── (g) GOLDEN RULE — per instrument ──")
    per_inst = {}
    for s in ON.columns:
        a_i = ON[s].values - bp
        h_i = CC[s].values
        d_i = sharpe(a_i) - sharpe(h_i)
        per_inst[s] = {"hold_sharpe": sharpe(h_i), "overnight_sharpe": sharpe(a_i), "d_sharpe": d_i,
                       "on_per_day": float(ON[s].mean()), "id_per_day": float(ID[s].mean())}
        print(f"  {s:5s} hold {sharpe(h_i):+.3f}  overnight {sharpe(a_i):+.3f}  Δ {d_i:+.3f}   "
              f"on {ON[s].mean()*100:+.4f}%/d  id {ID[s].mean()*100:+.4f}%/d")
    g_pos = sum(v["d_sharpe"] > 0 for v in per_inst.values())
    print(f"  positive: {g_pos}/10")

    # (x) ex-2020
    m = years != 2020
    x = sharpe(on_net.values[m]) - sharpe(bh.values[m])
    x_pass = x > 0
    print(f"── (x) EX-2020 ──  dSharpe {x:+.3f}   {'PASS' if x_pass else 'FAIL'}")

    # (d) raw
    d_day = float(delta.mean())
    print(f"── (d) RAW ──  delta {d_day*100:+.4f}%/day   overnight_net {on_net.mean()*100:+.4f}%/day vs floor {floor*100:.2f}%/day")

    # descriptive: break-even cost, per-year
    # dSharpe(cost) is monotone decreasing in cost; find zero by bisection on gross series
    lo_c, hi_c = 0.0, 50e-4
    for _ in range(60):
        mid = (lo_c + hi_c) / 2
        if sharpe((on_gross - mid).values) - sharpe(bh.values) > 0: lo_c = mid
        else: hi_c = mid
    breakeven_bp = lo_c * 1e4
    print(f"\n[descriptive] break-even round-trip cost ≈ {breakeven_bp:.2f} bp/instrument-day")
    per_year = {}
    for y in sorted(set(years)):
        my = years == y
        per_year[int(y)] = float(sharpe(on_net.values[my]) - sharpe(bh.values[my]))
    print("[descriptive] per-year dSharpe: " + "  ".join(f"{y}:{v:+.2f}" for y, v in per_year.items()))

    # ── verdict ladder, exactly as sealed ────────────────────────────────────
    if data_abort or fold_err or len(folds) != 15:
        verdict = "INCONCLUSIVE"
    elif not a_pass:
        verdict = "KILL_STRUCTURE"
    elif (not b1) or c_pos <= 7 or g_pos <= 5:
        verdict = "NULL"
    elif b1 and b2 and c_pos >= 12 and g_pos >= 7 and x_pass:
        verdict = "CONFIRMED" if (d_day >= 0 and on_net.mean() >= floor) else "VALID_BUT_BELOW_FLOOR"
    else:
        verdict = "INCONCLUSIVE"
    ledger_verdict = {"CONFIRMED": "CONFIRMED", "VALID_BUT_BELOW_FLOOR": "VALID_BUT_BELOW_FLOOR",
                      "KILL_STRUCTURE": "NULL", "NULL": "NULL", "INCONCLUSIVE": "INCONCLUSIVE"}[verdict]
    print(f"\n=== VERDICT: {verdict} (ledger: {ledger_verdict}) ===\n")

    res = {
        "id": doc["id"], "hash_lock": doc["hash_lock"], "run_at": datetime.now(timezone.utc).isoformat(),
        "incumbent": {"total": float(np.expm1(bh.sum())), "sharpe": sharpe(bh.values), "max_dd": max_dd(bh.values), "per_day": float(bh.mean())},
        "overnight_net": {"total": float(np.expm1(on_net.sum())), "sharpe": sharpe(on_net.values), "max_dd": max_dd(on_net.values), "per_day": float(on_net.mean())},
        "a": {"partition_per_day": part, "ci": [alo, ahi], "pass": a_pass},
        "b": {"d_sharpe": dsh, "ci": [blo, bhi], "pass_1": b1, "delta_sharpe": sharpe(delta.values),
              "dsr_deflated": float(dsr_sr), "dsr_prob": float(dsr_prob), "pass_2": b2},
        "c": {"folds": folds.tolist(), "n_pos": c_pos},
        "g": {"per_instrument": per_inst, "n_pos": g_pos},
        "x": {"ex2020_d_sharpe": x, "pass": x_pass},
        "d": {"delta_per_day": d_day, "overnight_net_per_day": float(on_net.mean()), "floor": floor},
        "descriptive": {"breakeven_rt_bp": breakeven_bp, "per_year_d_sharpe": per_year,
                        "stale_open_pct": {s: float(v) for s, v in stale_pct.items()}},
        "verdict": verdict, "ledger_verdict": ledger_verdict,
    }
    (OUT / "result.json").write_text(json.dumps(res, indent=2, default=float))

    # ledger: write once
    ledger = json.loads(LEDGER.read_text())
    backup = LEDGER.with_suffix(f".bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    shutil.copy2(LEDGER, backup)
    for e in ledger:
        if e.get("id") == doc["id"]:
            if e.get("status") != "PREREGISTERED":
                raise SystemExit("refusing: ledger entry is not PREREGISTERED")
            e["status"] = "ADJUDICATED"; e["verdict"] = ledger_verdict; e["result"] = verdict
            e["date_tested"] = res["run_at"][:10]
            e["oos_sharpe"] = sharpe(on_net.values); e["is_sharpe"] = None
            e["p_value"] = float((bdsh <= 0).mean()); e["bh_survives"] = None
            e["result_file"] = str((OUT / "result.json").relative_to(ROOT))
    tmp = tempfile.NamedTemporaryFile("w", dir=LEDGER.parent, delete=False, suffix=".tmp")
    json.dump(ledger, tmp, indent=2); tmp.close()
    Path(tmp.name).replace(LEDGER)
    print(f"ledger: {doc['id']} -> ADJUDICATED {ledger_verdict} (backup {backup.name})")
    gate_zero("end")
    return 0


if __name__ == "__main__":
    sys.exit(main(gate_only="--gate-only" in sys.argv))
