#!/usr/bin/env python3
"""HYP-074/075/076/078/079(/080) + HYP-077-FULL — options-leg interim runner, LOCKED protocol.

Sibling of run_positioning_family.py (which sealed the COT subset 2026-07-02 and stays
untouched for reproducibility). Same machinery: gate-zero hash verify on ALL 10 member
preregs + the family manifest BEFORE any data read; seals are dated ledger ANNOTATIONS;
statuses stay PREREGISTERED; NO verdicts here.

HYP-080 runs only if its prereg data_dependency holds (gdelt coverage >= 70% of the
2017+ window); otherwise a dated BLOCKED stamp is sealed instead.

--adjudicate: runs the family Benjamini-Hochberg across all 10 primaries ONLY when all
10 exist, exactly per the locked manifest; writes verdict annotations + the verdict
field. Refuses partial adjudication.

Usage: python3 scripts/research/run_positioning_family_options.py [--dry-run] [--adjudicate]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sovereign.research.positioning import event_study as es          # noqa: E402
from sovereign.research.positioning import v015_replay as vr          # noqa: E402
from sovereign.sentiment.store import connect                          # noqa: E402
from scripts.research import positioning_options_legs as ol           # noqa: E402

PREREG = ROOT / "data" / "research" / "preregister"
OUT = ROOT / "data" / "research" / "positioning_family"
SPOT_CACHE = OUT / "spot_cache"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
FAMILY = "POSITIONING-BOARD-2026-07"
SEAL_BY = f"{FAMILY}/interim-options-legs"
N_PERM, SEED = 10000, 42

ALL_PREREG = ["HYP-072_cot_extreme_fade.json", "HYP-073_flush_continuation.json",
              "HYP-074_rr_extreme_reversion.json", "HYP-075_spot_extreme_rr_nonconfirmation.json",
              "HYP-076_surprise_vs_crowding.json", "HYP-077_crowded_carry_drawdown_gate.json",
              "HYP-078_term_inversion_breakout.json", "HYP-079_butterfly_spike_timing.json",
              "HYP-080_gdelt_tone_x_positioning.json", "HYP-081_extreme_into_event_fade.json",
              "HYP-072-081_positioning_family.json"]

INTERP_TEXT = ("interpretations (declared 2026-07-03 pre-results, from data-availability arithmetic "
               "only — see positioning_options_legs.py docstring): 252-obs z = DAILY board series "
               "(ASOF-carried weekly surface; weekly reading rejected — <90 usable obs on 2020+ depth); "
               "EOD-signal t0 = first trading day STRICTLY AFTER signal date (surprise releases keep "
               "same-day t0, 08:30 ET); primary h: 074=20, 075=20, 076=5, 078=10, 079=10, 080=20; "
               "rr/bf/tone/cot are BASE-currency-signed, pair space via PAIR_FLIP once; 077-full rr "
               "term = Phi(rr25_z) aligned, per-pair mean with the cot term; 076 cells = crowded-"
               "against vs uncrowded control (extreme-aligned excluded), one event per publish_date "
               "(max |z|); rolling std ddof=1")
COVERAGE_TEXT = "coverage: options surface history starts 2020-01-03 (ThetaData Value-tier depth) — six years, not the decade; z warmup consumes 2020"


def _canonical_hash(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "hash_lock"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def gate_zero() -> dict:
    checks = {}
    for name in ALL_PREREG:
        doc = json.loads((PREREG / name).read_text())
        ok = _canonical_hash(doc) == doc.get("hash_lock")
        checks[name] = {"hash": doc.get("hash_lock", "")[:16], "ok": ok}
        if not ok:
            raise SystemExit(f"GATE ZERO: PREREGISTER HASH MISMATCH in {name}. HALT.")
    return checks


def fetch_ohlc(pair: str) -> pd.DataFrame:
    """Daily OHLC (yfinance, parquet-pinned) from 2014-06; Close feeds every leg, H/L feed 078."""
    SPOT_CACHE.mkdir(parents=True, exist_ok=True)
    cache = SPOT_CACHE / f"{pair}_ohlc.parquet"
    df = None
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
        raise SystemExit(f"no OHLC data for {pair} (network + cache both empty)")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[["Open", "High", "Low", "Close"]].astype(float)


def load_board() -> dict[str, pd.DataFrame]:
    con = connect(read_only=True)
    try:
        df = con.execute(
            "SELECT date, pair, rr25, bf25, atm_term_slope, cot_net_pct_1y, gdelt_tone "
            "FROM sentiment_board_state WHERE pair IN (?,?,?,?) ORDER BY pair, date",
            ol.OPT_PAIRS).df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return {p: g.set_index("date").sort_index() for p, g in df.groupby("pair")}


def load_cot() -> pd.DataFrame:
    con = connect(read_only=True)
    try:
        return con.execute(
            "SELECT pair, measurement_date, publish_date, net_pct_1y, flush_1w "
            "FROM sentiment_cot_weekly ORDER BY pair, publish_date").df()
    finally:
        con.close()


def load_releases() -> pd.DataFrame:
    con = connect(read_only=True)
    try:
        return con.execute(
            "SELECT publish_date, series, surprise_z, usd_sign "
            "FROM sentiment_surprise_release ORDER BY publish_date").df()
    finally:
        con.close()


def signed_leg_report(hyp: str, events_by_pair: dict, closes: dict, eligible: dict,
                      h_primary: int, h_secondary: int | None, rng, v015_equity) -> dict:
    res = {"hyp": hyp, "per_pair_N": {p: len(v) for p, v in events_by_pair.items()}}
    prim = es.pooled_primary_p(events_by_pair, closes, eligible, h_primary, rng, N_PERM)
    res["primary"] = {"h": h_primary,
                      "statistic_median": None if np.isnan(prim.obs) else round(prim.obs, 6),
                      "raw_p": None if np.isnan(prim.p) else round(prim.p, 5),
                      "n_perm": prim.n_perm, "N_deoverlapped": prim.n_events}
    all_events = [e for evs in events_by_pair.values() for e in evs]
    rets, kept, dropped = es.signed_returns(all_events, closes, h_primary)
    retsx, _, _ = es.signed_returns(all_events, closes, h_primary, ex2020=True)
    res["dropped_missing_data"] = dropped
    res["medians"] = {f"full_h{h_primary}": round(float(np.median(rets)), 6) if rets else None,
                      f"ex2020_h{h_primary}": round(float(np.median(retsx)), 6) if retsx else None}
    res[f"hit_h{h_primary}"] = es.binomial_hit(rets)
    res[f"ic_h{h_primary}"] = es.spearman_ic([e.strength for e in kept], rets)
    if h_secondary:
        s = es.pooled_primary_p(events_by_pair, closes, eligible, h_secondary, rng, N_PERM)
        res["secondary"] = {"h": h_secondary, "statistic_median": None if np.isnan(s.obs) else round(s.obs, 6),
                            "raw_p": None if np.isnan(s.p) else round(s.p, 5), "N": s.n_events}
    res["corr_carry"] = _corr_carry(all_events, closes, h_primary, v015_equity)
    res["sample_status"] = "OK" if prim.n_events >= 50 else "UNDERPOWERED_INTERIM"
    res["deviations"] = []
    return res


def _corr_carry(events, closes_by_pair, h, v015_equity) -> dict:
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
        book, n_active = book + rets, n_active + (rets != 0).astype(int)
    active = n_active > 0
    if active.sum() < 30:
        return {"rho_full": None, "n_overlap": int(active.sum()), "note": "under 30 overlapping active days"}
    rho = float(np.corrcoef(book[active] / n_active[active],
                            v015_equity.pct_change().reindex(book.index).fillna(0.0)[active])[0, 1])
    return {"rho_full": round(rho, 4), "n_overlap": int(active.sum()),
            "benchmark": "v015 daily curve (DBV leg deferred to final family run per VRP-001 convention)"}


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


def seal(hyp_id: str, res: dict, dry_run: bool, artifact_name: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    art = artifact_name or f"{hyp_id}.json"
    (OUT / art).write_text(json.dumps(res, indent=2, default=str) + "\n")
    prim = res.get("primary", {})
    stat = next((prim[k] for k in prim if k.startswith("statistic")), prim.get("obs"))
    p_raw = prim.get("raw_p", prim.get("p"))
    if isinstance(p_raw, float):
        p_raw = round(p_raw, 5)
    note = (f"INTERIM SEAL — NO VERDICT (family BH across all 10 primaries). "
            f"raw p={p_raw}, N={prim.get('N_deoverlapped', prim.get('N'))}, "
            f"h={prim.get('h')}, statistic={stat}, "
            f"sample_status={res.get('sample_status')}, deviations={res.get('deviations', [])}; "
            f"{COVERAGE_TEXT}; {INTERP_TEXT}; artifacts=data/research/positioning_family/{art}")
    if dry_run:
        print(f"[dry-run] would annotate {hyp_id}: {note[:150]}…")
        return
    backup = _annotate_ledger(hyp_id, {"date": datetime.now(timezone.utc).isoformat(),
                                       "by": SEAL_BY, "note": note})
    print(f"sealed {hyp_id} (ledger backup {backup})")


def gdelt_coverage(board: dict[str, pd.DataFrame]) -> float:
    tot = n = 0
    for p, df in board.items():
        w = df.loc[df.index >= "2017-01-01", "gdelt_tone"]
        tot += len(w)
        n += int(w.notna().sum())
    return (n / tot) if tot else 0.0


def family_bh(dry_run: bool) -> int:
    sources = {"HYP-072": "HYP-072.json", "HYP-073": "HYP-073.json", "HYP-074": "HYP-074.json",
               "HYP-075": "HYP-075.json", "HYP-076": "HYP-076.json", "HYP-077": "HYP-077_full.json",
               "HYP-078": "HYP-078.json", "HYP-079": "HYP-079.json", "HYP-080": "HYP-080.json",
               "HYP-081": "HYP-081.json"}
    ps, missing = {}, []
    for hyp, fname in sources.items():
        f = OUT / fname
        if not f.exists():
            missing.append(f"{hyp} ({fname} absent)")
            continue
        doc = json.loads(f.read_text())
        p = doc.get("primary", {}).get("raw_p", doc.get("primary", {}).get("p"))
        if p is None:
            missing.append(f"{hyp} (primary p null — {doc.get('sample_status')})")
        else:
            ps[hyp] = (float(p), doc)
    if missing:
        print("FAMILY BH REFUSED — not all 10 primaries exist:", "; ".join(missing))
        return 1
    m, alpha = len(ps), 0.05
    ranked = sorted(ps.items(), key=lambda kv: kv[1][0])
    kmax = 0
    for i, (hyp, (p, _)) in enumerate(ranked, 1):
        if p <= i / m * alpha:
            kmax = i
    table = []
    for i, (hyp, (p, doc)) in enumerate(ranked, 1):
        passed = i <= kmax
        n_ok = (doc.get("primary", {}).get("N_deoverlapped", doc.get("primary", {}).get("N", 0)) or 0) >= 50
        if not passed:
            verdict = "NOT_SIGNIFICANT"
        elif not n_ok:
            verdict = "UNDERPOWERED"
        else:
            verdict = "CANDIDATE_GATES_PENDING (diversifier/DBV + sign-stability review before CONFIRMED)"
        table.append({"hyp": hyp, "raw_p": p, "rank": i, "bh_threshold": round(i / m * alpha, 5),
                      "bh_pass": passed, "verdict": verdict})
    print(json.dumps(table, indent=2))
    if dry_run:
        print("[dry-run] no verdict annotations written")
        return 0
    ts = datetime.now(timezone.utc).isoformat()
    for row in table:
        note = (f"FAMILY VERDICT ({FAMILY}, BH alpha=0.05, m=10): raw_p={row['raw_p']}, rank={row['rank']}, "
                f"threshold={row['bh_threshold']}, bh_pass={row['bh_pass']} -> {row['verdict']}. "
                f"{COVERAGE_TEXT}. VISION kill-criterion link: an all-null family falsifies the "
                f"crowd-prediction thesis at current data resolution.")
        _annotate_ledger(row["hyp"], {"date": ts, "by": f"{FAMILY}/adjudication", "note": note})
        print(f"verdict annotated: {row['hyp']} -> {row['verdict']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--adjudicate", action="store_true", help="family BH (all 10 primaries required)")
    ap.add_argument("--only", default=None, metavar="HYP-ID",
                    help="run/seal ONLY this member (e.g. HYP-080 after its data dependency clears) — "
                         "prevents duplicate annotations on already-sealed legs; the single leg consumes "
                         "a fresh seed-42 stream (prereg specifies the seed, not stream position)")
    a = ap.parse_args()
    checks = gate_zero()
    print("gate zero: all 11 prereg hashes verified", all(v["ok"] for v in checks.values()))
    if a.adjudicate:
        return family_bh(a.dry_run)

    board = load_board()
    ohlc = {p: fetch_ohlc(p) for p in ol.OPT_PAIRS}
    closes = {p: ohlc[p]["Close"] for p in ol.OPT_PAIRS}
    trades = vr.load_trades()
    got = vr.reconcile_guard(trades, closes)
    print(f"reconcile guard: 0.6886 -> {got}")
    v015_eq = vr.daily_portfolio_equity(trades, closes["EURUSD"].index)
    cot = load_cot()
    cot["publish_date"] = pd.to_datetime(cot["publish_date"]).dt.date
    rng = np.random.default_rng(SEED)

    rr_z, bf_z, tone_z, slope, cot_pct = {}, {}, {}, {}, {}
    for p in ol.OPT_PAIRS:
        df = board[p]
        for src, dst in (("rr25", rr_z), ("bf25", bf_z), ("gdelt_tone", tone_z)):
            ol.assert_truncation_invariant(df[src].astype(float))
            dst[p] = ol.z_trailing(df[src].astype(float))
        slope[p] = df["atm_term_slope"].astype(float)
        cot_pct[p] = df["cot_net_pct_1y"].astype(float)
    print("truncation-invariance: PASS (rr25/bf25/gdelt_tone z, all pairs)")

    def elig(z_by_pair, h):
        out = {}
        for p in ol.OPT_PAIRS:
            ds = [ol.shift_pub(i.date()) for i in z_by_pair[p].dropna().index]
            out[p] = es.eligible_dates(closes[p], ds, h)
        return out

    results: dict[str, tuple[dict, str | None]] = {}
    want = (lambda h: a.only is None or a.only == h)

    if want("HYP-074"):
        ev074 = ol.events_074(rr_z)
        results["HYP-074"] = (signed_leg_report("HYP-074", ev074, closes, elig(rr_z, 20), 20, 10, rng, v015_eq), None)

    if want("HYP-075"):
        ev075 = ol.events_075(closes, rr_z)
        results["HYP-075"] = (signed_leg_report("HYP-075", ev075, closes, elig(rr_z, 20), 20, 10, rng, v015_eq), None)

    if want("HYP-076"):
        rels = load_releases()
        rels["publish_date"] = pd.to_datetime(rels["publish_date"]).dt.date
        ev076 = ol.events_076(rels, cot_pct)
        r076 = {"hyp": "HYP-076", "primary": None, "deviations": []}
        two = ol.perm_p_two_cell(ev076, closes, 5, rng, N_PERM)
        r076["primary"] = {"h": 5, "statistic_median_cell_diff": two.get("obs"),
                           "raw_p": round(two["p"], 5) if isinstance(two.get("p"), float) else two.get("p"),
                           "n_perm": N_PERM, "N_deoverlapped": two.get("n_crowded", 0),
                           "cells": {k: two.get(k) for k in ("n_crowded", "n_control", "median_crowded", "median_control", "note") if k in two}}
        two10 = ol.perm_p_two_cell(ev076, closes, 10, np.random.default_rng(SEED), N_PERM)
        r076["secondary"] = {"h": 10, "statistic": two10.get("obs"), "raw_p": two10.get("p")}
        r076["sample_status"] = "OK" if (two.get("n_crowded") or 0) >= 50 else "UNDERPOWERED_INTERIM"
        results["HYP-076"] = (r076, None)

    if want("HYP-078"):
        ev078 = ol.events_078(slope, cot_pct)
        stat078 = {p: pd.Series({ts: ol.range_ratio(ohlc[p]["High"], ohlc[p]["Low"], ts.date())
                                 for ts in slope[p].dropna().index}).dropna() for p in ol.OPT_PAIRS}
        pr078 = ol.perm_p_stat(stat078, {p: [e.signal_date for e in ev078[p]] for p in ol.OPT_PAIRS}, rng, N_PERM)
        results["HYP-078"] = ({"hyp": "HYP-078", "per_pair_N": {p: len(ev078[p]) for p in ol.OPT_PAIRS},
                               "primary": {"h": 10, "statistic_median_range_ratio": None if np.isnan(pr078.obs) else round(pr078.obs, 6),
                                           "raw_p": None if np.isnan(pr078.p) else round(pr078.p, 5),
                                           "n_perm": pr078.n_perm, "N_deoverlapped": pr078.n_events},
                               "sample_status": "OK" if pr078.n_events >= 50 else "UNDERPOWERED_INTERIM",
                               "deviations": []}, None)

    if want("HYP-079"):
        ev079 = ol.events_079(bf_z, cot_pct)
        stat079 = {p: pd.Series({ts: ol.abs_move(closes[p], ts.date())
                                 for ts in bf_z[p].dropna().index}).dropna() for p in ol.OPT_PAIRS}
        pr079 = ol.perm_p_stat(stat079, {p: [e.signal_date for e in ev079[p]] for p in ol.OPT_PAIRS}, rng, N_PERM)
        sec079 = {p: [es.Event(p, ol.shift_pub(e.signal_date), -e.crowd_pair_side, 1.0) for e in ev079[p]]
                  for p in ol.OPT_PAIRS}
        sec_rets, _, _ = es.signed_returns([e for v in sec079.values() for e in v], closes, 10)
        results["HYP-079"] = ({"hyp": "HYP-079", "per_pair_N": {p: len(ev079[p]) for p in ol.OPT_PAIRS},
                               "primary": {"h": 10, "statistic_median_abs_move": None if np.isnan(pr079.obs) else round(pr079.obs, 6),
                                           "raw_p": None if np.isnan(pr079.p) else round(pr079.p, 5),
                                           "n_perm": pr079.n_perm, "N_deoverlapped": pr079.n_events},
                               "secondary_signed_vs_crowd": {"h": 10, "median": round(float(np.median(sec_rets)), 6) if sec_rets else None,
                                                              "hit": es.binomial_hit(sec_rets)},
                               "sample_status": "OK" if pr079.n_events >= 50 else "UNDERPOWERED_INTERIM",
                               "deviations": []}, None)

    if want("HYP-077"):
        dts, vals, mapping = ol.crowding_composite_full(cot, rr_z, trades, ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
                                                        vr.open_positions_on)
        cross = es.detect_crossings(dts, vals, enter=lambda v: 1 if v >= 0.90 else 0, rearm=lambda v: v < 0.75)
        t0s = [es.effective_t0(d, closes["EURUSD"].index) for d, _, _ in cross]
        t0s = [t for t in t0s if t is not None]
        dds = [vr.fwd_max_drawdown(v015_eq, t, 20) for t in t0s]
        dds = [d for d in dds if d is not None]
        elig_pos = np.arange(0, len(closes["EURUSD"].index) - 21)
        all_dds = np.array([float((v015_eq.iloc[q:q + 21] / v015_eq.iloc[q:q + 21].cummax() - 1).min()) for q in elig_pos])
        obs77 = (float(np.median(dds)) - float(np.median(all_dds))) if dds else float("nan")
        n_ge = 0
        if dds:
            for _ in range(N_PERM):
                pick = rng.choice(elig_pos, size=len(dds), replace=False)
                if float(np.median(all_dds[pick])) - float(np.median(all_dds)) <= obs77:
                    n_ge += 1
        r077 = {"hyp": "HYP-077", "variant": "FULL COMPOSITE (cot + Phi(rr25_z) aligned) — supersedes the "
                                              "COT-only interim for the family stage; both annotations remain",
                "reconcile_guard": {"target": 0.6886, "recomputed": got, "pass": True},
                "n_crossings": len(cross), "N_deoverlapped": len(dds),
                "primary": {"h": 20, "N_deoverlapped": len(dds),
                            "statistic_median_dd_diff": round(obs77, 6) if dds else None,
                            "raw_p": round((n_ge + 1) / (N_PERM + 1), 5) if dds else None, "n_perm": N_PERM},
                "sample_status": "OK" if len(dds) >= 50 else "UNDERPOWERED_INTERIM",
                "composite_mapping_sample": mapping[:5], "diversifier_gate": "EXEMPT (prereg)",
                "deviations": []}
        results["HYP-077"] = (r077, "HYP-077_full.json")

    cov = gdelt_coverage(board)
    if not want("HYP-080"):
        pass
    elif cov >= 0.70:
        ev080 = ol.events_080(tone_z, cot_pct)
        results["HYP-080"] = (signed_leg_report("HYP-080", ev080, closes, elig(tone_z, 20), 20, 10, rng, v015_eq), None)
        results["HYP-080"][0]["gdelt_coverage"] = round(cov, 4)
    else:
        blocked = {"hyp": "HYP-080", "primary": None, "sample_status": "BLOCKED",
                   "gdelt_coverage": round(cov, 4),
                   "deviations": [f"BLOCKED — prereg data_dependency unmet: gdelt coverage {cov:.1%} < 70% of the 2017+ window"]}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "HYP-080_blocked.json").write_text(json.dumps(blocked, indent=2) + "\n")
        note = (f"BLOCKED STAMP — data_dependency unmet: gdelt_tone coverage {cov:.1%} < 70% "
                f"(prereg HYP-080 line data_dependency). Family BH remains open. {COVERAGE_TEXT}")
        if a.dry_run:
            print(f"[dry-run] would stamp HYP-080 BLOCKED ({cov:.1%})")
        else:
            _annotate_ledger("HYP-080", {"date": datetime.now(timezone.utc).isoformat(),
                                         "by": SEAL_BY, "note": note})
            print(f"stamped HYP-080 BLOCKED (gdelt coverage {cov:.1%})")

    manifest = {"run_ts": datetime.now(timezone.utc).isoformat(), "seed": SEED, "n_perm": N_PERM,
                "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip(),
                "gate_zero": checks, "interpretations": INTERP_TEXT, "coverage": COVERAGE_TEXT,
                "gdelt_coverage": round(cov, 4), "hyp_order": list(results)}
    OUT.mkdir(parents=True, exist_ok=True)
    mname = "run_manifest_options.json" if a.only is None else f"run_manifest_options_{a.only}.json"
    (OUT / mname).write_text(json.dumps(manifest, indent=2) + "\n")
    for hyp_id, (res, art) in results.items():
        seal(hyp_id, res, a.dry_run, art)
    print("\n== options-leg interim summary (NO verdicts) ==")
    for hyp_id, (res, _) in results.items():
        prim = res.get("primary") or {}
        print(f"{hyp_id}: raw_p={prim.get('raw_p')} N={prim.get('N_deoverlapped')} [{res.get('sample_status')}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
