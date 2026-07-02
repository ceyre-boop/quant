#!/usr/bin/env python3
"""HYP-072..081 — positioning board-state falsifiability program (pre-registration).

Builds, hash-locks, and verifies the ten pre-registration files plus the family
manifest, then appends ten PREREGISTERED entries to the hypothesis ledger.

Discipline:
- Fields are frozen BEFORE the Track-B positioning data lands (the substrate —
  TFF COT features, options surface — is being built in the same sprint; no
  feature values have been read).
- Hash method mirrors scripts/research/exit_regime_conditioning.py::_canonical_hash:
  sha256 of json.dumps(doc minus hash_lock, sort_keys=True, separators=(',',':')).
- The BH family is locked here: the ten primary p-values correct together
  (alpha 0.05). Running any member outside the family invalidates the program.
- PREREGISTERED is a NEW ledger status (first use 2026-07-02): the entry exists
  before any result; verdict-bearing fields are null until resolution.

Usage:
  python3 scripts/research/preregister_positioning.py --write   # sign + write + ledger
  python3 scripts/research/preregister_positioning.py --verify  # re-verify all hashes
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREREG_DIR = ROOT / "data" / "research" / "preregister"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"

FROZEN_AT = "2026-07-02T05:30:00Z"
FAMILY_ID = "POSITIONING-BOARD-2026-07"
UNIVERSE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
UNIVERSE_NOTE = ("4 live pairs; AUDNZD included wherever both futures legs (6A 232741, 6N 112741) "
                 "carry data — constitutional carry-complex scope, not the live trading universe")

# Shared validation protocol — locked once, referenced by every member.
PROTOCOL = {
    "permutation": {"n": 10000, "seed": 42, "p_formula": "(n_ge + 1) / (N + 1)",
                    "null": "event-label shuffle preserving per-pair event counts"},
    "multiple_testing": {"family": FAMILY_ID, "method": "benjamini_hochberg",
                         "alpha": 0.05,
                         "rule": "the 10 primary p-values (one per HYP, pooled-universe cell) correct together"},
    "ic_threshold_oos": 0.15,
    "de_overlap_standard": ("ONE observation per distinct event/crossing (NOT per day in state); "
                            "binomial null at the REAL de-overlapped N; median primary (mean is "
                            "outlier-dragged); report full sample AND ex-2020"),
    "min_sample": {"per_pair_cell": 30, "pooled_primary": 50,
                   "underpowered_rule": "below pooled minimum -> verdict UNDERPOWERED, never a directional claim"},
    "diversifier_gate": {"applies": "all members except HYP-077",
                         "benchmark_primary": "DBV carry proxy (VRP-001 convention)",
                         "benchmark_secondary": "v015 forex backtest pnl_pct (logs/forex_backtest_trades.json)",
                         "true_diversifier_iff": "abs(full_rho) < 0.25 AND max_crisis_rho < 0.35",
                         "crisis_windows": [["2008-06-01", "2009-03-31"], ["2020-02-20", "2020-04-30"],
                                            ["2022-01-01", "2022-12-31"]]},
    "no_model_training": "event studies only; no fitted parameters beyond pre-registered thresholds",
    "data_substrate": ("positioning board tables (sentiment_cot_weekly legacy+TFF features, "
                       "sentiment_options_surface, econ_surprise_z, gdelt_tone) — release/publish-date "
                       "keyed, look-ahead audited (scripts/audit_look_ahead.py)"),
}

def _hyp(num, slug, name, thesis, mechanism, event, direction, horizon_td, extra=None):
    doc = {
        "id": f"HYP-{num:03d}",
        "slug": slug,
        "name": name,
        "status": "PREREGISTERED",
        "frozen_at": FROZEN_AT,
        "family": FAMILY_ID,
        "phase": "prereg — data substrate in flight (Track B); no feature values read",
        "thesis": thesis,
        "mechanism": mechanism,
        "event_definition": event,
        "direction_rule": direction,
        "universe": UNIVERSE,
        "universe_note": UNIVERSE_NOTE,
        "horizon_trading_days": horizon_td,
        "validation_protocol": PROTOCOL,
        "success_criteria": ("candidate iff: pooled primary permutation p < 0.05 AND survives family BH "
                             "AND median effect sign stable full/ex-2020 AND pooled N >= 50 de-overlapped "
                             "AND (diversifier gate passes, unless exempt)"),
        "failure_criteria": "anything else -> NOT_SIGNIFICANT (the prior); direction-unstable -> REJECTED",
        "prior_expectation": "NOT_SIGNIFICANT",
        "verdict": None,
        "hash_method": ("sha256 of json.dumps(doc, sort_keys=True, separators=(',',':')) "
                        "where doc = this object MINUS the hash_lock field"),
    }
    if extra:
        doc.update(extra)
    return doc

HYPS = [
    _hyp(72, "cot_extreme_fade",
         "COT net-spec 1y-percentile > 0.95 fades over 2-4 weeks",
         "Extreme speculative crowding in a currency future marks positioning exhaustion; "
         "the crowded side lacks marginal buyers and mean-reverts over 2-4 weeks.",
         "Late-cycle speculators chase; when net-spec percentile is extreme the marginal flow "
         "that drove the move is spent (Tenet: race-to-the-bottom is the caution signal).",
         "cot_net_pct_1y crossing above 0.95 (or below 0.05), base-currency future, "
         "de-overlap: one event per crossing, re-arm only after re-entering [0.10, 0.90]",
         "crossing >0.95 (crowd long base) -> fade = base depreciates vs USD; <0.05 mirrored. "
         "USDJPY inverted (6J is JPY/USD).",
         [10, 20]),
    _hyp(73, "flush_continuation",
         "Largest weekly net-spec flush continues in the flush direction",
         "A top-decile week-over-week net-spec unwind (flush) marks forced de-risking that "
         "continues: stopped-out positioning takes more than a week to clear.",
         "Forced liquidation is autocorrelated — margin/stop cascades unwind over weeks, "
         "not in one print.",
         "|flush_1w| (WoW delta net_spec / trailing-1y delta std) >= 2.0; de-overlap: one event "
         "per flush episode, re-arm after |flush_1w| < 1.0",
         "forward spot move in the SAME direction as the flush (crowd cut longs -> pair falls further)",
         [10, 20]),
    _hyp(74, "rr_extreme_reversion",
         "25-delta risk-reversal z beyond +/-2 precedes spot mean-reversion",
         "Extreme skew pricing marks one-sided hedging/fear that overshoots; spot mean-reverts "
         "as the skew normalizes.",
         "Options skew embeds crowd fear; extremes are typically hedging climaxes, not "
         "information about drift.",
         "rr25_z (vs trailing 252 obs) crossing beyond +/-2.0; de-overlap: one event per "
         "crossing, re-arm inside +/-1.0",
         "spot moves OPPOSITE the skew extreme (rr25_z > +2 = calls bid = crowd long topside -> fade down)",
         [10, 20]),
    _hyp(75, "spot_extreme_rr_nonconfirmation",
         "Spot new extreme without risk-reversal confirmation reverses",
         "A fresh 60-day spot extreme that options skew refuses to confirm (rr25_z not making a "
         "same-sign 60-day extreme) is a move without conviction behind it and reverses.",
         "Divergence between price and the hedging market = the marginal participant is not "
         "paying up to chase; classic non-confirmation.",
         "spot closes at a new 60-trading-day extreme AND rr25_z 60-day extreme of the same sign "
         "is absent that day; de-overlap: one event per 20-td window per pair per side",
         "reversal: forward move opposite the spot extreme's direction",
         [10, 20]),
    _hyp(76, "surprise_vs_crowding",
         "Economic surprise against extreme positioning forces a move, scaled by crowding",
         "A macro surprise that hits a crowded trade forces repositioning; the move in the "
         "surprise's direction is larger when positioning is extreme against it.",
         "Surprise + trapped crowd = forced flow (the urgent-buyer-meets-urgent-seller day "
         "from VISION).",
         "de-overlapped distinct release events with |econ_surprise_z| >= 1.5 (existing "
         "SENTIMENT-ECON-SURPRISE standard) WHERE cot_net_pct_1y is extreme (>=0.90 or <=0.10) "
         "AGAINST the USD-signed surprise direction",
         "forward move in the SURPRISE direction; primary test vs the same-surprise-uncrowded "
         "control cell (median difference); secondary: monotonicity in crowding decile",
         [5, 10],
         {"note": "conditional variant of the resolved SENTIMENT-ECON-SURPRISE null: that test "
                  "found no UNCONDITIONAL daily-horizon signal; this pre-registers the "
                  "crowding-conditional cell only."}),
    _hyp(77, "crowded_carry_drawdown_gate",
         "Crowded carry (COT + risk-reversals on funded pairs) precedes carry drawdowns — defensive gate",
         "When the carry complex itself is the crowded trade, its unwind risk is elevated; a "
         "crowding composite should lead carry-portfolio drawdowns and can gate size DOWN.",
         "The carry crowd IS the large-spec book (SENTIMENT-COT-POSITIONING found +0.18 "
         "correlation); at extremes the exit door is small (Tenet 6).",
         "crowding composite = mean over funded pairs of (cot_net_pct_1y aligned with the carry "
         "position's direction, and rr25_z aligned) >= 0.90; state variable, de-overlap: one "
         "event per crossing, re-arm below 0.75",
         "subsequent 20-td v015 carry-portfolio max drawdown is DEEPER than the unconditional "
         "20-td max-drawdown distribution (median difference, permutation on state labels)",
         [20],
         {"diversifier_gate_exempt": True,
          "exemption_reason": "defensive gate candidate — deliberately carry-correlated; its job "
                              "is to fire WHEN carry is at risk, not to diversify it",
          "reconcile_guard": {"target": 0.6886, "tolerance": 0.01,
                              "rule": "the v015 decade replay used for drawdown scoring must "
                                      "reproduce the canonical weighted portfolio Sharpe first"}}),
    _hyp(78, "term_inversion_breakout",
         "ATM IV term-structure inversion at a positioning extreme precedes a breakout regime",
         "Inverted FX vol term structure (front > back) at a positioning extreme marks stressed "
         "hedging into a crowded book — conditions for range expansion rather than drift.",
         "Front-loaded vol demand = near-dated event/stress pricing; a crowded book amplifies "
         "realized movement when it breaks.",
         "atm_term_slope (atm_iv_1m - atm_iv_3m) > 0 AND (cot_net_pct_1y >= 0.90 or <= 0.10); "
         "de-overlap: one event per joint-condition onset, re-arm when slope < 0",
         "regime claim, not directional: forward 10-td realized range (high-low sum) exceeds the "
         "trailing-60-td median range (ratio > 1; permutation on event labels)",
         [10]),
    _hyp(79, "butterfly_spike_timing",
         "25-delta butterfly spike at a positioning extreme times the forced move",
         "A tail-demand spike (butterfly bid) while positioning is extreme marks smart hedging "
         "immediately before forced unwinds — a timing signal for the move against the crowd.",
         "Whoever pays up for wings at a crowding extreme is pricing the unwind; wings lead.",
         "bf25_z (vs trailing 252 obs) >= 2.0 AND (cot_net_pct_1y >= 0.90 or <= 0.10); "
         "de-overlap: one event per spike, re-arm below bf25_z < 1.0",
         "|forward 10-td move| exceeds unconditional median (timing claim) AND signed direction "
         "is AGAINST the crowded side (secondary)",
         [10]),
    _hyp(80, "gdelt_tone_x_positioning",
         "GDELT tone extremes read opposite ways by positioning alignment: aligned = exhaustion, opposed = fuel",
         "News tone that agrees with an already-crowded position is late (exhaustion — fade it); "
         "tone that fights the crowd is early information the crowd must eventually price (fuel).",
         "Narrative confirmation lags positioning; narrative opposition forces repricing.",
         "gdelt tone_z |z| >= 1.5 events, split: ALIGNED (tone sign == crowded side sign, "
         "cot_net_pct_1y >= 0.80 or <= 0.20) vs OPPOSED; de-overlap: one event per tone episode "
         "(re-arm |z| < 0.75); the two cells are ONE hypothesis with paired predictions",
         "ALIGNED -> fade tone direction; OPPOSED -> follow tone direction; primary = pooled "
         "two-cell median effect with signs as predicted",
         [10, 20],
         {"data_dependency": "requires gdelt_tone backfill (currently NULL pending off-peak run) "
                             "— runs only when coverage >= 70% of window"}),
    _hyp(81, "extreme_into_event_fade",
         "Positioning extreme into a CB decision or NFP resolves against the crowd",
         "Scheduled binary events force crowded books to de-risk; the post-event move "
         "disproportionately runs against the pre-event crowd.",
         "Event risk + crowded positioning = asymmetric squeeze potential (the known "
         "pre-CB chop/positioning effect, HYP-061 family, now positioning-conditioned).",
         "scheduled event day (FOMC/BOE/ECB/RBA from cb_calendar.CB_MEETINGS + NFP first-Friday "
         "schedule) with cot_net_pct_1y >= 0.90 or <= 0.10 on the prior Friday's report; "
         "de-overlap: one observation per event day per pair",
         "post-event forward move AGAINST the crowded side (crowd long base -> base falls after "
         "the event)",
         [5],
         {"note": "cb_calendar.CB_MEETINGS must be back-extended (known gap, HYP-061 wiring "
                  "note) before the full-history run; prereg does not depend on it."}),
]

def _canonical_hash(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "hash_lock"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _family_manifest(members):
    return {
        "id": FAMILY_ID,
        "kind": "bh_family_manifest",
        "status": "PREREGISTERED",
        "frozen_at": FROZEN_AT,
        "members": [m["id"] for m in members],
        "rule": ("The 10 primary p-values (one pooled-universe cell per member) are corrected "
                 "TOGETHER via Benjamini-Hochberg at alpha=0.05. Members may not be run/reported "
                 "outside the family. Secondary/per-pair cells are exploratory only."),
        "kill_criterion_link": ("VISION.md Falsifiability: ~10 pre-registered nulls on the "
                                "positioning board-state falsify the crowd-prediction thesis at "
                                "current data resolution."),
        "protocol": PROTOCOL,
        "hash_method": ("sha256 of json.dumps(doc, sort_keys=True, separators=(',',':')) "
                        "where doc = this object MINUS the hash_lock field"),
    }

def write_all():
    PREREG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for doc in HYPS:
        doc["hash_lock"] = _canonical_hash(doc)
        path = PREREG_DIR / f"{doc['id']}_{doc['slug']}.json"
        if path.exists():
            print(f"REFUSING to overwrite existing prereg: {path}", file=sys.stderr)
            sys.exit(1)
        path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
        written.append((path, doc))
    fam = _family_manifest(HYPS)
    fam["hash_lock"] = _canonical_hash(fam)
    fam_path = PREREG_DIR / "HYP-072-081_positioning_family.json"
    if fam_path.exists():
        print(f"REFUSING to overwrite existing manifest: {fam_path}", file=sys.stderr)
        sys.exit(1)
    fam_path.write_text(json.dumps(fam, indent=2, sort_keys=False) + "\n")
    written.append((fam_path, fam))
    return written, fam

def append_ledger(docs, fam):
    ledger = json.loads(LEDGER.read_text())
    assert isinstance(ledger, list), "ledger must be a JSON array"
    existing = {e.get("id") for e in ledger}
    backup = LEDGER.with_suffix(f".bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    shutil.copy2(LEDGER, backup)
    added = 0
    for doc in docs:
        if doc["id"] in existing:
            print(f"REFUSING duplicate ledger id {doc['id']}", file=sys.stderr)
            sys.exit(1)
        ledger.append({
            "id": doc["id"],
            "name": doc["name"],
            "status": "PREREGISTERED",
            "date_registered": FROZEN_AT[:10],
            "family": FAMILY_ID,
            "hash_lock": doc["hash_lock"],
            "prereg_file": f"data/research/preregister/{doc['id']}_{doc['slug']}.json",
            "mechanism": doc["mechanism"],
            "methodology_note": ("Pre-registered before the positioning board-state data existed "
                                 "(Track B in flight, 2026-07-02). Family BH: " + FAMILY_ID +
                                 " (10 members, alpha 0.05, manifest hash " + fam["hash_lock"][:12] +
                                 "...). Verdict empty until resolved under family protocol. "
                                 "PREREGISTERED status is new to the ledger as of this program."),
            "prior_expectation": "NOT_SIGNIFICANT",
            "result": None,
            "p_value": None,
            "bh_survives": None,
            "oos_sharpe": None,
            "is_sharpe": None,
            "standalone": False,
            "auto_generated": False,
            "source": "manual",
        })
        added += 1
    tmp = tempfile.NamedTemporaryFile("w", dir=LEDGER.parent, delete=False, suffix=".tmp")
    tmp.write(json.dumps(ledger, indent=2) + "\n")
    tmp.close()
    Path(tmp.name).replace(LEDGER)
    print(f"ledger: +{added} PREREGISTERED entries (backup {backup.name})")

def verify():
    ok = True
    for path in sorted(PREREG_DIR.glob("HYP-07*_*.json")) + sorted(PREREG_DIR.glob("HYP-08*_*.json")) \
            + [PREREG_DIR / "HYP-072-081_positioning_family.json"]:
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        if doc.get("id", "").startswith(("HYP-071",)):
            continue
        h = _canonical_hash(doc)
        good = doc.get("hash_lock") == h
        ok &= good
        print(f"{'OK  ' if good else 'FAIL'} {path.name} {doc.get('hash_lock', '')[:16]}")
    ledger = json.loads(LEDGER.read_text())
    pre = [e for e in ledger if e.get("status") == "PREREGISTERED"]
    print(f"ledger PREREGISTERED entries: {len(pre)}")
    files = {json.loads((PREREG_DIR / Path(e['prereg_file']).name).read_text())['hash_lock'] == e['hash_lock']
             for e in pre}
    print("ledger hash_locks match prereg files:", files == {True})
    if not ok or files != {True}:
        raise SystemExit("PREREGISTRATION VERIFY FAILED — do not proceed")
    print("VERIFY PASS")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.write:
        written, fam = write_all()
        for p, d in written:
            print(f"signed {p.name}  {d['hash_lock'][:16]}")
        append_ledger(HYPS, fam)
    if a.verify or a.write:
        verify()
    if not (a.write or a.verify):
        ap.print_help()
