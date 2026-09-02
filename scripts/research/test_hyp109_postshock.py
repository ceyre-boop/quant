#!/usr/bin/env python3
"""HYP-109 — post-shock abstention. THE ONE TEST. Run once.

Everything this computes is dictated by data/research/preregister/HYP-109.json,
which was sealed and committed (1ac598b) before this file existed. Gate zero
asserts the hash and the PREREGISTERED status before a single number is read,
and the hash is asserted again at the end so a mid-run edit cannot hide.

Order of output, per the prereg:
  0. gate zero
  1. the incumbent's absolute R -- buy-and-hold, equal-weight, same ten ETFs,
     same window: total return, annualised Sharpe, max drawdown, %/day
  2. (a) magnitude      (b) direction-null      (c) tradeability per purged fold
     and full-sample DSR     (d) raw return vs the floor
  3. the UVXY companion -- descriptive only
  4. the verdict from the frozen ladder
  5. gate zero again

No parameter here is a knob. Every constant is read from the sealed doc.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sovereign.discovery.cpcv import combinatorial_purged_splits  # noqa: E402
from sovereign.discovery.gate import deflated_sharpe_ratio  # noqa: E402

PREREG = ROOT / "data" / "research" / "preregister" / "HYP-109.json"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
DATA = ROOT / "data" / "cache" / "daily_universe"
OUT = ROOT / "data" / "research" / "hyp109"


# ── gate zero ────────────────────────────────────────────────────────────────

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


# ── data ─────────────────────────────────────────────────────────────────────

def load(sym: str, window: list[str]) -> pd.Series:
    df = pd.read_parquet(DATA / f"{sym}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= window[0]) & (df["date"] <= window[1])].sort_values("date")
    s = df.set_index("date")["close"].astype(float)
    return np.log(s).diff().dropna()      # close-to-close log return, per prereg


def shocks(r: pd.Series, pct: float, warm: int) -> pd.Series:
    """Boolean shock flag. Percentile over t-warm..t-1 ONLY — t excluded."""
    a = r.abs()
    thresh = a.shift(1).rolling(warm).quantile(pct)     # shift(1): excludes t
    flag = a >= thresh
    flag[thresh.isna()] = False                          # warm-up: no events
    return flag


def flat_mask(shock: pd.Series, k: int) -> pd.Series:
    """True on sessions t+1..t+k after any shock; overlapping windows union."""
    m = pd.Series(False, index=shock.index)
    idx = np.where(shock.values)[0]
    n = len(m)
    for i in idx:
        m.iloc[i + 1: min(i + 1 + k, n)] = True
    return m


# ── statistics ───────────────────────────────────────────────────────────────

def stationary_block_bootstrap(x: np.ndarray, L: int, draws: int, seed: int, stat, rng=None):
    """Politis-Romano stationary bootstrap of `stat` over 1-D array x."""
    rng = rng or np.random.default_rng(seed)
    n = len(x)
    p = 1.0 / L
    out = np.empty(draws)
    for d in range(draws):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for j in range(n):
            if j and rng.random() < p:
                i = rng.integers(n)
            idx[j] = i
            i = (i + 1) % n
        out[d] = stat(x[idx])
    return out


def sharpe(r: np.ndarray) -> float:
    s = r.std(ddof=1)
    return float(r.mean() / s * np.sqrt(252)) if s > 0 else 0.0


def max_dd(r: np.ndarray) -> float:
    eq = np.cumprod(1 + r)
    return float((eq / np.maximum.accumulate(eq) - 1).min())


# ── the test ─────────────────────────────────────────────────────────────────

def main() -> int:
    doc = gate_zero("start")
    P = doc["frozen_parameters"]
    S = doc["statistics"]
    win = doc["data"]["window"]
    warm = doc["data"]["warmup_sessions"]
    pct, k, bp = P["percentile"], P["k"], P["round_trip_bp"] / 1e4
    L, seed, draws = P["block_L"], P["seed"], P["draws"]
    rng = np.random.default_rng(seed)
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"\nHYP-109 — {doc['name']}")
    print(f"frozen: p{int(pct*100)} k={k} warm={warm} cost={P['round_trip_bp']}bp "
          f"L={L} draws={draws} seed={seed} cpcv={P['cpcv']} n_trials={P['n_trials']}\n")

    # Per-instrument series -----------------------------------------------
    rets, flats, ev_rv, nev_rv, ev_cr, nev_cr = {}, {}, [], [], [], []
    n_shock = {}
    for sym in doc["instrument_set"]:
        r = load(sym, win)
        sh = shocks(r, pct, warm)
        fm = flat_mask(sh, k)
        rets[sym], flats[sym] = r, fm
        n_shock[sym] = int(sh.sum())
        # forward 5-session windows for (a) and (b)
        arr = r.values
        for i in np.where(sh.values)[0]:
            w = arr[i + 1: i + 1 + k]
            if len(w) == k:
                ev_rv.append(w.std(ddof=1)); ev_cr.append(w.sum())
        # non-shock windows: every session that is NOT a shock and not warm-up
        valid = (~sh) & sh.index.isin(sh.index[warm:])
        for i in np.where(valid.values)[0]:
            w = arr[i + 1: i + 1 + k]
            if len(w) == k:
                nev_rv.append(w.std(ddof=1)); nev_cr.append(w.sum())

    print("shock events per instrument:")
    for s, n in n_shock.items():
        print(f"  {s:5s} {n:4d}")
    ev_rv, nev_rv, ev_cr, nev_cr = map(np.asarray, (ev_rv, nev_rv, ev_cr, nev_cr))
    print(f"pooled: {len(ev_rv)} shock windows, {len(nev_rv)} non-shock windows\n")

    abort = [s for s, n in n_shock.items() if n < doc["abort"]["min_shock_events_per_instrument"]]

    # Portfolio series -----------------------------------------------------
    R = pd.DataFrame(rets).dropna()
    F = pd.DataFrame(flats).reindex(R.index).fillna(False)
    R = R.iloc[warm:]; F = F.iloc[warm:]
    bh = R.mean(axis=1)                                       # equal-weight buy & hold
    exposed = (~F).astype(float)
    ab = (R * exposed).mean(axis=1)
    # cost on every flat episode start (exit) and end (re-entry), per instrument
    episodes = (F.astype(int).diff().abs() == 1)              # a transition = one leg
    cost = episodes.sum(axis=1) * (bp / 2) / R.shape[1]       # half round-trip per leg, EW
    ab_net = ab - cost

    # 1. incumbent's absolute R ---------------------------------------------
    print("── INCUMBENT: buy-and-hold, equal-weight, same ten ETFs, same window ──")
    print(f"  total return   {float(np.expm1(bh.sum()))*100:+.1f}%")
    print(f"  ann. Sharpe    {sharpe(bh.values):.3f}")
    print(f"  max drawdown   {max_dd(bh.values)*100:.1f}%")
    print(f"  return/day     {bh.mean()*100:+.4f}%/day\n")

    # (a) magnitude -----------------------------------------------------------
    ratio = float(np.median(ev_rv) / np.median(nev_rv))
    # One-sided p under H0 (no difference): label permutation of the pooled RVs.
    # A permutation null is the standard resampling test for a two-sample ratio
    # and is stricter than resampling each arm separately.
    pooled = np.concatenate([ev_rv, nev_rv]); ne = len(ev_rv)
    null = np.empty(draws)
    for d in range(draws):
        perm = rng.permutation(pooled)
        null[d] = np.median(perm[:ne]) / np.median(perm[ne:])
    p_a = float((null >= ratio).mean())
    a_pass = ratio > 1.0 and p_a < 0.05
    print("── (a) MAGNITUDE ──")
    print(f"  median RV shock / non-shock = {ratio:.3f}   one-sided p = {p_a:.4f}   "
          f"{'PASS' if a_pass else 'FAIL'}")

    # (b) direction-null ------------------------------------------------------
    diff = float(ev_cr.mean() - nev_cr.mean())
    boot = stationary_block_bootstrap(ev_cr, L, draws, seed, np.mean, rng) - nev_cr.mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    b_holds = lo <= 0 <= hi
    print("── (b) DIRECTION-NULL ──")
    print(f"  mean 5d cumr shock − non-shock = {diff*100:+.3f}%   95% CI [{lo*100:+.3f}%, {hi*100:+.3f}%]   "
          f"{'HOLDS' if b_holds else 'FAILS -> directional'}")
    # what the shocks looked like, for the lead if it fails
    ev_sign = np.sign(ev_cr).mean()
    print(f"  (mean sign of post-shock 5d return: {ev_sign:+.3f})")

    # (c) tradeability on purged folds ---------------------------------------
    print("── (c) TRADEABILITY — 15 purged CPCV folds ──")
    dates = R.index.values
    entry = dates
    exit_ = np.append(dates[k:], [dates[-1]] * k)             # each obs "lives" k sessions
    fold_deltas = []
    for tr, te in combinatorial_purged_splits(entry, exit_, n_groups=S["cpcv"]["n_groups"],
                                              test_groups=S["cpcv"]["test_groups"],
                                              embargo_frac=k / len(dates)):
        d = sharpe(ab_net.values[te]) - sharpe(bh.values[te])
        fold_deltas.append(d)
    fold_deltas = np.asarray(fold_deltas)
    n_pos = int((fold_deltas > 0).sum())
    for i, d in enumerate(fold_deltas, 1):
        print(f"  fold {i:2d}  ΔSharpe {d:+.3f}")
    full_d = sharpe(ab_net.values) - sharpe(bh.values)
    _, dsr_prob = deflated_sharpe_ratio(sharpe(ab_net.values), n_trials=P["n_trials"],
                                        n_obs=len(ab_net))
    print(f"  positive folds {n_pos}/15   full-sample ΔSharpe {full_d:+.3f}   "
          f"abstain Sharpe {sharpe(ab_net.values):.3f}   DSR prob {dsr_prob:.3f}")
    if n_pos >= 12 and dsr_prob >= 0.95:
        c = "PASS"
    elif n_pos <= 7:
        c = "NULL"
    else:
        c = "INCONCLUSIVE"
    print(f"  -> {c}")

    # (d) raw return ----------------------------------------------------------
    d_day = float(ab_net.mean() - bh.mean())
    print("── (d) RAW RETURN ──")
    print(f"  abstain {ab_net.mean()*100:+.4f}%/day   buy&hold {bh.mean()*100:+.4f}%/day   "
          f"Δ {d_day*100:+.4f}%/day   floor {S['d_raw_return']['floor']*100:.2f}%/day")
    print(f"  abstain: total {float(np.expm1(ab_net.sum()))*100:+.1f}%  maxDD {max_dd(ab_net.values)*100:.1f}%  "
          f"time in market {float(exposed.mean().mean())*100:.1f}%\n")

    # companion: UVXY, descriptive only ---------------------------------------
    print("── COMPANION (descriptive only, no verdict weight) ──")
    u = load(doc["companion_instrument"]["symbol"], win)
    spy_sh = shocks(rets["SPY"], pct, warm)
    u = u.reindex(spy_sh.index).dropna()
    ua = u.values; cond = []
    for i in np.where(spy_sh.reindex(u.index).fillna(False).values)[0]:
        w = ua[i + 1: i + 1 + k]
        if len(w) == k:
            cond.append(w.sum())
    cond = np.asarray(cond)
    uncond = np.asarray([ua[i + 1: i + 1 + k].sum() for i in range(warm, len(ua) - k)])
    cb = stationary_block_bootstrap(cond, L, draws, seed, np.mean, rng)
    clo, chi = np.percentile(cb, [2.5, 97.5])
    print(f"  UVXY 5d cumr after SPY shock  {cond.mean()*100:+.2f}%  95% CI [{clo*100:+.2f}%, {chi*100:+.2f}%]  n={len(cond)}")
    print(f"  UVXY 5d cumr unconditional    {uncond.mean()*100:+.2f}%  (roll-decayed baseline)")
    print(f"  difference                    {(cond.mean()-uncond.mean())*100:+.2f}%   — a lead, not a result\n")

    # verdict --------------------------------------------------------------------
    if abort:
        verdict, sub = "INCONCLUSIVE", f"<30 shock events for {abort}"
    elif not b_holds:
        verdict, sub = "NULL", f"KILL_DIRECTIONAL (post-shock drift sign {ev_sign:+.2f}; record as a lead)"
    elif not a_pass or c == "NULL":
        verdict, sub = "NULL", ("magnitude not elevated" if not a_pass else "no Sharpe improvement")
    elif c == "INCONCLUSIVE":
        verdict, sub = "INCONCLUSIVE", f"{n_pos}/15 folds"
    elif d_day >= 0:
        verdict, sub = "CONFIRMED", ""
    else:
        verdict, sub = "NULL", "VALID_BUT_BELOW_FLOOR (Sharpe bought by giving up return)"
    print(f"══ VERDICT: {verdict}{'  — ' + sub if sub else ''} ══\n")

    result = {
        "id": doc["id"], "hash_lock": doc["hash_lock"], "run_at": datetime.now(timezone.utc).isoformat(),
        "n_shock_per_instrument": n_shock, "pooled_shock_windows": int(len(ev_rv)),
        "incumbent": {"total_return": float(np.expm1(bh.sum())), "sharpe": sharpe(bh.values),
                      "max_dd": max_dd(bh.values), "ret_per_day": float(bh.mean())},
        "a": {"ratio": ratio, "p": p_a, "pass": bool(a_pass)},
        "b": {"diff": diff, "ci": [float(lo), float(hi)], "holds": bool(b_holds), "mean_sign": float(ev_sign)},
        "c": {"fold_deltas": fold_deltas.tolist(), "positive_folds": n_pos, "full_delta": full_d,
              "dsr_prob": float(dsr_prob), "result": c},
        "d": {"delta_per_day": d_day, "abstain_per_day": float(ab_net.mean()),
              "time_in_market": float(exposed.mean().mean())},
        "companion_uvxy": {"cond": float(cond.mean()), "ci": [float(clo), float(chi)],
                           "uncond": float(uncond.mean()), "n": int(len(cond))},
        "verdict": verdict, "sub": sub,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2))

    # adjudicate ONCE -----------------------------------------------------------
    ledger = json.loads(LEDGER.read_text())
    for e in ledger:
        if e.get("id") == doc["id"]:
            e["status"] = "ADJUDICATED"
            e["verdict"] = verdict
            e["date_tested"] = result["run_at"][:10]
            e["result"] = (f"{verdict}{' — ' + sub if sub else ''}. a ratio {ratio:.3f} p={p_a:.4f}; "
                           f"b diff {diff*100:+.3f}% CI [{lo*100:+.3f},{hi*100:+.3f}]; "
                           f"c {n_pos}/15 folds, ΔSharpe {full_d:+.3f}, DSR {dsr_prob:.3f}; "
                           f"d Δ{d_day*100:+.4f}%/day. Incumbent Sharpe {sharpe(bh.values):.3f}.")
            e["p_value"] = p_a
            e["oos_sharpe"] = sharpe(ab_net.values)
    LEDGER.write_text(json.dumps(ledger, indent=2))
    print(f"ledger: {doc['id']} -> ADJUDICATED {verdict}")

    gate_zero("end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
