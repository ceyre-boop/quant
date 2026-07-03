#!/usr/bin/env python3
"""HYP-072/073/077/081 — COT-subset interim runner under the LOCKED family protocol.

Seals raw interim results as dated ledger ANNOTATIONS. Issues NO verdicts: the family
manifest corrects all 10 primary p-values together (Benjamini-Hochberg, alpha .05), and
six members await the options surface — adjudication happens when those legs exist.

Gate zero re-verifies every pre-registration hash lock before any data is read.
HYP-077 runs the operator-authorized COT-ONLY variant (rr25 term has no data) — the
deviation is stamped on its seal; the final family verdict requires the full composite.

Usage: python3 scripts/research/run_positioning_family.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sovereign.reporting.equity_curve import weighted_portfolio_sharpe  # noqa: E402  (import sanity)
from sovereign.research.positioning import event_study as es  # noqa: E402
from sovereign.research.positioning import v015_replay as vr  # noqa: E402
from sovereign.sentiment.store import connect  # noqa: E402

PREREG = ROOT / "data" / "research" / "preregister"
OUT = ROOT / "data" / "research" / "positioning_family"
SPOT_CACHE = OUT / "spot_cache"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
CAL_FILE = ROOT / "data" / "research" / "cb_meetings_historical.json"
FAMILY = "POSITIONING-BOARD-2026-07"
SEAL_BY = f"{FAMILY}/interim-cot-subset"
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "AUDNZD"]
FUNDED = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
N_PERM, SEED = 10000, 42
NULL_TEXT = ("null-interpretation: locked 'event-label shuffle preserving per-pair event counts' "
             "read as PER-PAIR DATE-SHUFFLE onto eligible dates (sign-flip rejected — destroyed by "
             "carry drift); event-level shuffle is the defense for overlapping forward windows")
PRIMARY_H_TEXT = "primary horizon interpretation: one pooled primary per HYP -> h=20 (072/073/077), h=5 (081); h=10 sealed secondary"

# FOMC/BOE 2023-2025 — re-declared from scripts/rq_rest_010_cb_blackout.py (import-unsafe script)
FOMC_2023_25 = ["2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13", "2024-01-31", "2024-03-20",
                "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
                "2025-01-29", "2025-03-19", "2025-05-07"]
BOE_2023_25 = ["2023-08-03", "2023-09-21", "2023-11-02", "2023-12-14", "2024-02-01", "2024-03-21",
               "2024-05-09", "2024-06-20", "2024-08-01", "2024-09-19", "2024-11-07", "2024-12-19",
               "2025-02-06", "2025-03-20", "2025-05-08"]


def _canonical_hash(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "hash_lock"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def gate_zero() -> dict:
    """Verify every prereg hash lock BEFORE any data read. SystemExit on mismatch."""
    checks = {}
    files = ["HYP-072_cot_extreme_fade.json", "HYP-073_flush_continuation.json",
             "HYP-077_crowded_carry_drawdown_gate.json", "HYP-081_extreme_into_event_fade.json",
             "HYP-072-081_positioning_family.json"]
    for name in files:
        doc = json.loads((PREREG / name).read_text())
        ok = _canonical_hash(doc) == doc.get("hash_lock")
        checks[name] = {"hash": doc.get("hash_lock", "")[:16], "ok": ok}
        if not ok:
            raise SystemExit(f"GATE ZERO: PREREGISTER HASH MISMATCH in {name} — the frozen design "
                             f"was altered after signing. HALT.")
    return checks


def fetch_spot(pair: str) -> pd.Series:
    """Daily closes (yfinance, parquet-pinned) from 2014-06 for warmup through today."""
    SPOT_CACHE.mkdir(parents=True, exist_ok=True)
    cache = SPOT_CACHE / f"{pair}.parquet"
    try:
        import yfinance as yf
        df = yf.download(f"{pair}=X", start="2014-06-01", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if len(df):
            df.to_parquet(cache)
    except Exception:
        df = None
    if (df is None or not len(df)) and cache.exists():
        df = pd.read_parquet(cache)
    if df is None or not len(df):
        raise SystemExit(f"no spot data for {pair} (network + cache both empty)")
    s = df["Close"].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


def load_cot() -> pd.DataFrame:
    con = connect(read_only=True)
    try:
        return con.execute(
            "SELECT pair, measurement_date, publish_date, net_pct_1y, flush_1w "
            "FROM sentiment_cot_weekly ORDER BY pair, publish_date").df()
    finally:
        con.close()


def corr_to_carry(events: list[es.Event], closes_by_pair: dict, h: int,
                  v015_equity: pd.Series) -> dict:
    """Pearson rho of the event-book daily return vs the v015 daily return (sealed info only)."""
    book = pd.Series(0.0, index=v015_equity.index)
    n_active = pd.Series(0, index=v015_equity.index)
    for e in events:
        closes = closes_by_pair.get(e.pair)
        if closes is None:
            continue
        t0 = es.effective_t0(e.publish_date, closes.index)
        if t0 is None:
            continue
        pos = closes.index.get_indexer([t0])[0]
        seg = closes.iloc[pos:pos + h + 1]
        rets = e.side * np.log(seg / seg.shift(1)).dropna()
        rets = rets.reindex(book.index).fillna(0.0)
        book = book + rets
        n_active = n_active + (rets != 0).astype(int)
    active = n_active > 0
    if active.sum() < 30:
        return {"rho_full": None, "n_overlap": int(active.sum()),
                "note": "under 30 overlapping active days"}
    book_ret = (book[active] / n_active[active])
    v015_ret = v015_equity.pct_change().reindex(book.index).fillna(0.0)[active]
    rho = float(np.corrcoef(book_ret, v015_ret)[0, 1])
    out = {"rho_full": round(rho, 4), "n_overlap": int(active.sum()),
           "benchmark": "v015 daily curve (DBV leg deferred to final family run per VRP-001 convention)"}
    for name, (a, b) in {"2020": ("2020-01-01", "2020-12-31"), "2022": ("2022-01-01", "2022-12-31")}.items():
        m = active & (book.index >= a) & (book.index <= b)
        if m.sum() >= 20:
            out[f"rho_{name}"] = round(float(np.corrcoef((book[m] / n_active[m]),
                                                         v015_ret.reindex(book.index)[m])[0, 1]), 4)
    return out


def run_crossing_hyp(hyp: str, cot: pd.DataFrame, closes: dict, rng: np.random.Generator,
                     metric_col: str, enter, rearm, direction_rule: str,
                     v015_equity: pd.Series) -> dict:
    events_by_pair, strengths = {}, {}
    for pair in PAIRS:
        sub = cot[cot["pair"] == pair].sort_values("publish_date")
        crossings = es.detect_crossings(list(sub["publish_date"]), list(sub[metric_col]), enter, rearm)
        if hyp == "HYP-072":     # fade: crowd long base -> base depreciates (metric side -> pair -side)
            to_side, strength = (lambda s: -s), (lambda v: max(v - 0.95, 0.05 - v))
        else:                    # HYP-073 continuation: same direction as the flush
            to_side, strength = (lambda s: s), (lambda v: abs(v) - 2.0)
        events_by_pair[pair] = es.make_events(pair, crossings, to_side, strength)
        strengths[pair] = [e.strength for e in events_by_pair[pair]]
    eligible = {p: es.eligible_dates(closes[p], sorted(cot[cot["pair"] == p]["publish_date"].unique()), 20)
                for p in PAIRS}
    res = {"hyp": hyp, "direction_rule": direction_rule, "per_pair_N": {}, "pair_spans": {}}
    all_events = []
    for p, evs in events_by_pair.items():
        res["per_pair_N"][p] = len(evs)
        sub = cot[cot["pair"] == p]
        res["pair_spans"][p] = [str(sub["publish_date"].min())[:10], str(sub["publish_date"].max())[:10]]
        all_events.extend(evs)
    prim = es.pooled_primary_p(events_by_pair, closes, eligible, 20, rng, N_PERM)
    res["primary"] = {"h": 20, "statistic_median": None if np.isnan(prim.obs) else round(prim.obs, 6),
                      "raw_p": None if np.isnan(prim.p) else round(prim.p, 5),
                      "n_perm": prim.n_perm, "N_deoverlapped": prim.n_events}
    rets20, kept20, dropped = es.signed_returns(all_events, closes, 20)
    rets20x, _, _ = es.signed_returns(all_events, closes, 20, ex2020=True)
    rets10, kept10, _ = es.signed_returns(all_events, closes, 10)
    res["dropped_missing_data"] = dropped
    res["medians"] = {"full_h20": round(float(np.median(rets20)), 6) if rets20 else None,
                      "ex2020_h20": round(float(np.median(rets20x)), 6) if rets20x else None,
                      "full_h10": round(float(np.median(rets10)), 6) if rets10 else None}
    res["hit_h20"] = es.binomial_hit(rets20)
    res["ic_h20"] = es.spearman_ic([e.strength for e in kept20], rets20)
    res["ic_h10"] = es.spearman_ic([e.strength for e in kept10], rets10)
    res["corr_carry"] = corr_to_carry(all_events, closes, 20, v015_equity)
    res["sample_status"] = ("OK" if prim.n_events >= 50 else "UNDERPOWERED_INTERIM")
    res["deviations"] = []
    return res


def run_hyp077(cot: pd.DataFrame, trades: pd.DataFrame, closes: dict,
               rng: np.random.Generator) -> tuple[dict, pd.Series]:
    got = vr.reconcile_guard(trades, closes)          # SystemExit on failure, BEFORE any write
    trading_index = closes["EURUSD"].index
    equity = vr.daily_portfolio_equity(trades, trading_index)
    dates, values, mapping = vr.crowding_composite(cot, trades, FUNDED)
    crossings = es.detect_crossings(dates, values,
                                    enter=lambda v: 1 if v >= 0.90 else 0,
                                    rearm=lambda v: v < 0.75)
    t0s = [es.effective_t0(d, trading_index) for d, _, _ in crossings]
    t0s = [t for t in t0s if t is not None]
    dds = [vr.fwd_max_drawdown(equity, t, 20) for t in t0s]
    dds = [d for d in dds if d is not None]
    elig_pos = np.arange(0, len(trading_index) - 21)
    all_dds = np.array([float((equity.iloc[p:p + 21] / equity.iloc[p:p + 21].cummax() - 1).min())
                        for p in elig_pos])
    obs = (float(np.median(dds)) - float(np.median(all_dds))) if dds else float("nan")
    n_ge = 0
    if dds:
        for _ in range(N_PERM):
            pick = rng.choice(elig_pos, size=len(dds), replace=False)
            if float(np.median(all_dds[pick])) - float(np.median(all_dds)) <= obs:
                n_ge += 1
    p = (n_ge + 1) / (N_PERM + 1) if dds else None
    res = {"hyp": "HYP-077", "reconcile_guard": {"target": 0.6886, "recomputed": got, "pass": True},
           "n_crossings": len(crossings), "N_deoverlapped": len(dds),
           "primary": {"h": 20, "N_deoverlapped": len(dds),
                       "statistic_median_dd_diff": round(obs, 6) if dds else None,
                       "raw_p": round(p, 5) if p else None, "n_perm": N_PERM,
                       "medians": {"event_dd": round(float(np.median(dds)), 6) if dds else None,
                                   "unconditional_dd": round(float(np.median(all_dds)), 6)}},
           "sample_status": "OK" if len(dds) >= 50 else "UNDERPOWERED_INTERIM",
           "deviations": ["DEVIATION (operator-authorized): rr25_z term omitted — options surface "
                          "has no data; COT-only interim variant. Hash-locked composite unchanged; "
                          "final family verdict requires the full composite rerun."],
           "composite_mapping_sample": mapping[:5], "diversifier_gate": "EXEMPT (prereg)"}
    return res, equity


def load_calendar() -> tuple[dict, dict]:
    """(verified_subset, full_history_or_None). Full-history refused without per-date sources."""
    from sovereign.forex.cb_calendar import CB_MEETINGS

    def _iso(d) -> str:
        return d if isinstance(d, str) else str(pd.Timestamp(d).date())

    subset: dict[str, set] = {"FED": set(FOMC_2023_25), "BOE": set(BOE_2023_25), "ECB": set(), "RBA": set()}
    for bank in ("FED", "BOE", "ECB", "RBA"):
        subset.setdefault(bank, set()).update(_iso(d) for d in CB_MEETINGS.get(bank, []))
    full = None
    if CAL_FILE.exists():
        doc = json.loads(CAL_FILE.read_text())
        banks, sources = doc.get("banks", {}), doc.get("sources", {})
        sourced = all(str(y) in sources.get(b, {}) or sources.get(b)
                      for b in banks for y in {d[:4] for d in banks[b]})
        if banks and sourced:
            full = {b: {_iso(d) for d in v} for b, v in banks.items()}
            for b in subset:
                full.setdefault(b, set()).update(subset[b])
        else:
            print("[081] calendar file present but not fully sourced — full-history branch REFUSED")
    return subset, full


def nfp_first_fridays(start_year=2015, end="2026-06-30") -> list[str]:
    out = []
    for y in range(start_year, 2027):
        for m in range(1, 13):
            d = date(y, m, 1)
            d = d.replace(day=1 + (4 - d.weekday()) % 7)  # first Friday
            if str(d) <= end:
                out.append(str(d))
    return out


def run_hyp081(cot: pd.DataFrame, closes: dict, rng: np.random.Generator,
               calendar: dict, branch: str, v015_equity: pd.Series) -> dict:
    event_days = sorted({d for bank in ("FED", "BOE", "ECB", "RBA") for d in calendar.get(bank, set())}
                        | set(nfp_first_fridays()))
    event_days = [d for d in event_days if "2015-01-01" <= d <= "2026-06-30"]
    events_by_pair, eligible = {}, {}
    for pair in PAIRS:
        sub = cot[cot["pair"] == pair].sort_values("publish_date").set_index("publish_date")["net_pct_1y"]
        evs, elig = [], []
        for dstr in event_days:
            d = date.fromisoformat(dstr)
            prior = sub[sub.index <= d]
            if prior.empty or pd.isna(prior.iloc[-1]):
                continue
            elig.append(d)
            v = float(prior.iloc[-1])
            if v >= 0.90 or v <= 0.10:
                metric_side = 1 if v >= 0.90 else -1          # crowd long/short base currency
                # AGAINST crowd-long-base = base depreciates => generic pair side -1; USDJPY flips
                pair_side = (-metric_side) * es.PAIR_FLIP.get(pair, 1)
                evs.append(es.Event(pair, d, pair_side, max(v - 0.90, 0.10 - v)))
        events_by_pair[pair] = evs
        eligible[pair] = elig
    prim = es.pooled_primary_p(events_by_pair, closes, eligible, 5, rng, N_PERM)
    all_events = [e for evs in events_by_pair.values() for e in evs]
    rets5, kept5, dropped = es.signed_returns(all_events, closes, 5)
    rets5x, _, _ = es.signed_returns(all_events, closes, 5, ex2020=True)
    return {"hyp": "HYP-081", "branch": branch,
            "calendar_counts": {b: len(calendar.get(b, [])) for b in ("FED", "BOE", "ECB", "RBA")},
            "n_event_days": len(event_days),
            "per_pair_N": {p: len(v) for p, v in events_by_pair.items()},
            "primary": {"h": 5, "statistic_median": None if np.isnan(prim.obs) else round(prim.obs, 6),
                        "raw_p": None if np.isnan(prim.p) else round(prim.p, 5),
                        "n_perm": prim.n_perm, "N_deoverlapped": prim.n_events},
            "medians": {"full_h5": round(float(np.median(rets5)), 6) if rets5 else None,
                        "ex2020_h5": round(float(np.median(rets5x)), 6) if rets5x else None},
            "hit_h5": es.binomial_hit(rets5), "ic_h5": es.spearman_ic([e.strength for e in kept5], rets5),
            "dropped_missing_data": dropped,
            "corr_carry": corr_to_carry(all_events, closes, 5, v015_equity),
            "sample_status": "OK" if prim.n_events >= 50 else "UNDERPOWERED_INTERIM",
            "deviations": [], "cross_sectional_caveat":
                "one event day can appear for multiple pairs (cross-sectional dependence) — "
                "known interim caveat, resolved at family stage"}


def _annotate_ledger(hyp_id: str, annotation: dict) -> str:
    ledger = json.loads(LEDGER.read_text())
    assert isinstance(ledger, list)
    backup = LEDGER.with_suffix(f".bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    shutil.copy2(LEDGER, backup)
    entry = next(e for e in ledger if e.get("id") == hyp_id)
    assert entry["status"] == "PREREGISTERED", f"{hyp_id} status is {entry['status']} — refusing"
    before = len(entry.get("annotations", []))
    entry.setdefault("annotations", []).append(annotation)
    tmp = tempfile.NamedTemporaryFile("w", dir=LEDGER.parent, delete=False, suffix=".tmp")
    tmp.write(json.dumps(ledger, indent=2) + "\n")
    tmp.close()
    Path(tmp.name).replace(LEDGER)
    check = json.loads(LEDGER.read_text())
    e2 = next(e for e in check if e.get("id") == hyp_id)
    assert len(e2["annotations"]) == before + 1 and e2["status"] == "PREREGISTERED"
    return str(backup.name)


def seal(hyp_id: str, res: dict, dry_run: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{hyp_id}.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    prim = res["primary"]
    note = (f"INTERIM SEAL — NO VERDICT (family BH across all 10 primaries awaits the options "
            f"legs). raw p={prim.get('raw_p')}, N={prim.get('N_deoverlapped')} de-overlapped, "
            f"h={prim.get('h')}, statistic={prim.get('statistic_median', prim.get('statistic_median_dd_diff'))}, "
            f"hit={json.dumps(res.get('hit_h20') or res.get('hit_h5') or {})}, "
            f"IC={res.get('ic_h20', res.get('ic_h5'))}, medians={json.dumps(res.get('medians', {}))}, "
            f"corr_carry={json.dumps(res.get('corr_carry', {}), default=str)[:220]}, "
            f"sample_status={res['sample_status']}, deviations={res['deviations']}; {NULL_TEXT}; "
            f"{PRIMARY_H_TEXT}; artifacts=data/research/positioning_family/{hyp_id}.json")
    if dry_run:
        print(f"[dry-run] would annotate {hyp_id}: {note[:160]}…")
        return
    backup = _annotate_ledger(hyp_id, {"date": datetime.now(timezone.utc).isoformat(),
                                       "by": SEAL_BY, "note": note})
    print(f"sealed {hyp_id} (ledger backup {backup})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    checks = gate_zero()
    print("gate zero: all prereg hashes verified", {k: v["ok"] for k, v in checks.items()})
    cot = load_cot()
    cot["publish_date"] = pd.to_datetime(cot["publish_date"]).dt.date
    closes = {p: fetch_spot(p) for p in PAIRS}
    trades = vr.load_trades()
    rng = np.random.default_rng(SEED)

    results = {}
    results["HYP-072"] = run_crossing_hyp(
        "HYP-072", cot, closes, rng, "net_pct_1y",
        enter=lambda v: 1 if v > 0.95 else (-1 if v < 0.05 else 0),
        rearm=lambda v: 0.10 <= v <= 0.90,
        direction_rule="fade: crowd long base -> base depreciates; USDJPY inverted",
        v015_equity=vr.daily_portfolio_equity(trades, closes["EURUSD"].index))
    results["HYP-073"] = run_crossing_hyp(
        "HYP-073", cot, closes, rng, "flush_1w",
        enter=lambda v: int(np.sign(v)) if abs(v) >= 2.0 else 0,
        rearm=lambda v: abs(v) < 1.0,
        direction_rule="continuation: forward move in the flush direction; USDJPY inverted",
        v015_equity=vr.daily_portfolio_equity(trades, closes["EURUSD"].index))
    res077, equity = run_hyp077(cot, trades, closes, rng)
    results["HYP-077"] = res077
    subset, full = load_calendar()
    results["HYP-081"] = run_hyp081(cot, closes, rng, full or subset,
                                    "full_history" if full else "verified_subset", equity)
    if full is None:
        results["HYP-081"]["deviations"].append(
            "full-history branch unavailable (calendar file missing/unsourced) — verified-subset only")

    manifest = {"run_ts": datetime.now(timezone.utc).isoformat(), "seed": SEED, "n_perm": N_PERM,
                "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip(),
                "gate_zero": checks, "null_interpretation": NULL_TEXT,
                "primary_h": PRIMARY_H_TEXT, "hyp_order": list(results),
                "cot_rows": int(len(cot)), "spot_spans": {p: [str(closes[p].index.min())[:10],
                                                              str(closes[p].index.max())[:10]] for p in PAIRS}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for hyp_id, res in results.items():
        seal(hyp_id, res, a.dry_run)
    print("\n== interim summary (NO verdicts) ==")
    for hyp_id, res in results.items():
        p = res["primary"]
        print(f"{hyp_id}: raw_p={p.get('raw_p')} N={p.get('N_deoverlapped')} "
              f"stat={p.get('statistic_median', p.get('statistic_median_dd_diff'))} "
              f"[{res['sample_status']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
