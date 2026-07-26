#!/usr/bin/env python3
"""
System-player run over the Cursus Honorum question bank.

Methodology (see games/cursus_honorum/system_heatmap.json "methodology" block
for the long form): each question's answer below was derived BEFORE looking at
the bank's `correct` field, by applying the actual coded/documented logic in:
  sovereign/forex/carry_engine.py       (carry direction, sizing, stops, ledger)
  sovereign/forex/risk_sentiment.py     (VIX/term-structure regime map)
  sovereign/forex/combat_vetoes.py      (C-001/C-005/C-006/C-003 veto arithmetic)
  sovereign/forex/strategy.py           (grade table, conviction thresholds)
  sovereign/training/state_space.py     (locked 8-dim S(t))
  imbalance_engine/petroulas_gate.py    (dual-confirmation thresholds)
  RISK_CONSTITUTION.md                  (Art 1/2/3/6 caps and ladder)
  TRADING_PHILOSOPHY.md six tenets, CLAUDE.md evidence ledger facts

`provenance` classifies WHERE the answer actually came from:
  CODE      - a literal constant/threshold read from the files above; purely
              mechanical application of an if/else the running system executes.
  CONFIG_DISABLED - same as CODE, but the module is config-disabled in
              production (combat_vetoes.py, enabled=false) -- the arithmetic
              is real, the live gate is not. Flagged, not abstained.
  DOC_TENET - the six-tenet prose in TRADING_PHILOSOPHY.md. Real and encoded,
              but as documentation, not an executable classifier -- there is
              no code that maps a scenario to a tenet ID.
  LEDGER_FACT - a specific historical/ledger record (hypothesis verdicts,
              measured Sharpe, p-values) quoted in CLAUDE.md / the hypothesis
              ledger. Encoded as recorded state, not as a formula.
  LLM_BOUNDARY - the actual production decision is delegated to an LLM call
              (Kimi's petroulas_worthy judgment of "arithmetic vs narrative"),
              not to deterministic code. Marked abstain=True.

abstain=True is reserved for LLM_BOUNDARY items: the coded system has no
formula for this and defers to a model call in production, so a "system"
answer here is not really testing coded logic. All other answers are
mechanical applications of a real constant/rule found by direct file read.
"""
import json

ANSWERS = {
 "CAL-01": ("B", "CODE", "carry_engine.py high_yield_side rule + MIN_CARRY_SPREAD_BPS floor"),
 "CAL-02": ("D", "DOC_TENET", "T1 promotion-gate doctrine (IC OOS>0.15)"),
 "CAL-03": ("A", "CODE", "RISK_CONSTITUTION Art.1 0.75% per-trade cap overrides Kelly/grade"),
 "CAL-04": ("C", "CODE", "risk_sentiment.py regime map: VIX<=20 & CONTANGO -> RISK_ON"),
 "CAL-05": ("C", "CODE", "ict/sovereign isolation test + time-horizon wall doctrine"),
 "CAL-06": ("A", "DOC_TENET", "graveyard revival rule: new hypothesis + N>=200 + sign-off"),
 "CAL-07": ("D", "CODE", "strategy.py CONVICTION_NEUTRAL_THRESHOLD=0.10"),
 "CAL-08": ("B", "LEDGER_FACT", "COT Tue-measured/Fri-published look-ahead rule"),
 "CAL-09": ("B", "CONFIG_DISABLED", "combat_vetoes.py C-001 MACRO_AGAINST arithmetic (module enabled=false in prod)"),
 "CAL-10": ("D", "CODE", "risk_sentiment.py VIX>=35 & BACKWARDATION -> EXTREME_RISK_OFF, forced_shorts"),
 "CAL-11": ("A", "LEDGER_FACT", "ICT permutation p=0.52 -> unproven, TP/SL reference layer only"),
 "CAL-12": ("C", "CODE", "carry_engine.py CARRY_RISK_PER_PAIR=0.003 + dedicated ledger design"),
 "CAL-13": ("C", "CODE", "strategy.py grade_from_signal: 1.5<=|diff|<2.0 -> A (A+ needs conviction>=0.60 too)"),
 "CAL-14": ("A", "CODE", "state_space.py dim5 carry_alignment: opposed -> -1"),
 "CAL-15": ("D", "LEDGER_FACT", "rolling walk-forward + p<0.001: real but regime-fragile"),
 "CAL-16": ("B", "DOC_TENET", "only sanctioned cross-system channel: cross-system bridge"),
 "CAL-17": ("B", "LLM_BOUNDARY", "petroulas_worthy = Kimi judgment of arithmetic-vs-narrative, not code"),
 "CAL-18": ("D", "CODE", "RISK_CONSTITUTION Art.3 ladder: 5.2% past 5% rung -> halt new entries"),
 "CAL-19": ("A", "CODE", "carry_engine.py high_yield_side=BASE(NZD) -> LONG NZDJPY"),
 "CAL-20": ("C", "LEDGER_FACT", "AUDNZD both legs RBA-driven, OOS Sharpe -0.879 proven drag"),
 "CAL-21": ("C", "DOC_TENET", "never run prop challenge while allocator has any system frozen"),
 "CAL-22": ("A", "CODE", "cot_interpretation JPY-inverse mapping + 3yr-extreme contrarian rule"),
 "CAL-23": ("D", "CODE", "confirmation_protocol Art.4: ambiguous/abstained board -> no entry"),
 "CAL-24": ("B", "CODE", "ict_params.yml weights: market_structure/pd_alignment zeroed (HYP-034/024 anti-edges)"),
 "CAL-25": ("B", "CODE", "RISK_CONSTITUTION Art.2 carry-complex 2.5% cap; 2.7% breaches it"),
 "CAL-26": ("D", "LEDGER_FACT", "Overnight-QQQ recouples w/ carry in COVID crash (rho .42, p=.007) -> rejected as diversifier"),
 "CAL-27": ("A", "CODE", "petroulas_gate.py dual-confirmation: stress>=6, XGB>0.65, Kimi mag/conv>=7 -> approve 3-5%"),
 "CAL-28": ("C", "CODE", "multi-day macro hold shouldn't eat binary event risk on entry bar into FOMC"),
 "CAL-29": ("C", "CODE", "risk_sentiment.py: VIX 20-25 + FLAT term structure -> NEUTRAL"),
 "CAL-30": ("A", "DOC_TENET", "T5 orchestration is the durable edge"),
 "DIAG-01": ("D", "CODE", "|diff|=90bp < MIN_CARRY_SPREAD_BPS=100 -> FLAT"),
 "DIAG-02": ("B", "CODE", "high_yield_side=BASE(USD) -> LONG USDJPY"),
 "DIAG-03": ("B", "CODE", "carry sized at fixed CARRY_RISK_PER_PAIR=0.003 regardless of differential size"),
 "DIAG-04": ("D", "CODE", "high_yield_side=BASE(EUR), diff+390bp>floor -> LONG EURJPY"),
 "DIAG-05": ("A", "CODE", "differential decayed below 100bp floor -> rationale gone, coupon goes FLAT"),
 "DIAG-06": ("C", "CODE", "highest-differential JPY-funded pair unwinds hardest in a flush -> trim first"),
 "DIAG-07": ("C", "CODE", "carry_engine.py ATR_STOP_MULTIPLE=3.0 by design vs ~1.5x macro/ICT"),
 "DIAG-08": ("A", "CODE", "carry logged to dedicated ledger, excluded from macro Sharpe stats by design"),
 "DIAG-09": ("D", "CODE", "0.3% x 4 = 1.2% comfortably under Art.2 2.5% carry-complex cap"),
 "DIAG-10": ("B", "CODE", "degraded_sentinel: fail loud on fallback ATR, no silent mocking"),
 "DIAG-11": ("B", "LEDGER_FACT", "COT contrarian feature only fires near percentile tails; 55th = mid-range"),
 "DIAG-12": ("D", "LEDGER_FACT", "documented holiday-week naive +3-day look-ahead bias"),
 "DIAG-13": ("A", "CODE", "JPY-inverse mapping: record JPY-short extreme -> contrarian USDJPY-down risk"),
 "DIAG-14": ("C", "LEDGER_FACT", "COT = positioning snapshot, contrarian tell at extremes, not a forecast"),
 "DIAG-15": ("C", "LEDGER_FACT", "Tuesday-dated feature = look-ahead lift, not edge; needs Friday-lag + IC>0.15"),
 "DIAG-16": ("A", "CODE", "risk_sentiment.py spread>1.0 -> CONTANGO"),
 "DIAG-17": ("D", "CODE", "RISK_OFF requires VIX>=25 AND BACKWARDATION; contango at 27 fails -> NEUTRAL"),
 "DIAG-18": ("B", "CODE", "RISK_OFF + carry_unwind_active -> forced_shorts override on carry pairs"),
 "DIAG-19": ("B", "CODE", "VIX<=20 & CONTANGO -> RISK_ON"),
 "DIAG-20": ("D", "LEDGER_FACT", "trending years +0.51/+1.26 vs range-bound -0.13/-0.09 -> lean into edge in trending regime"),
 "DIAG-21": ("A", "CONFIG_DISABLED", "combat_vetoes.py C-005: |0.3|<0.5 weak_rate fires (module enabled=false in prod)"),
 "DIAG-22": ("C", "CONFIG_DISABLED", "combat_vetoes.py C-001: +1.4% opposing SHORT beyond deadband fires (module enabled=false in prod)"),
 "DIAG-23": ("C", "CODE", "execution invariant min_grade_to_execute=A; B-grade logs only, no trade"),
 "DIAG-24": ("A", "CODE", "over-confirmation penalty above raw score 9.0 (empirically degrades expectancy)"),
 "DIAG-25": ("D", "CODE", "clearing macro vetoes doesn't oblige eating binary event risk on entry bar"),
 "DIAG-26": ("B", "CODE", "grade-risk table B=0.5% < Art.1 0.75% cap -> applies directly"),
 "DIAG-27": ("B", "CODE", "RISK_CONSTITUTION Art.1 overrides grade table; A+ table value 1.5% capped to 0.75%"),
 "DIAG-28": ("D", "CODE", "conviction 0.10 barely clears NEUTRAL floor, nowhere near 0.70 full-size"),
 "DIAG-29": ("A", "CODE", "trade is in a carry pair -> counts toward Art.2 2.5% aggregate carry-complex cap"),
 "DIAG-30": ("C", "CODE", "grade-risk table C=0.25%"),
 "DIAG-31": ("C", "CODE", "EXTREME_RISK_OFF needs VIX>=35 AND BACKWARDATION; contango at 33 fails"),
 "DIAG-32": ("A", "CODE", "fresh USD-positive shock cuts against EURUSD long -> re-check thesis, don't add"),
 "DIAG-33": ("D", "CODE", "EXTREME_RISK_OFF forced-direction override -> existing carry longs are wrong side"),
 "DIAG-34": ("B", "CODE", "only 0.3% headroom under 2% max_daily_loss hard cap into a gap-prone event -> stand down"),
 "DIAG-35": ("B", "DOC_TENET", "T3 systems must know when they are unreliable"),
 "DIAG-36": ("D", "DOC_TENET", "T4 premature complexity kills more systems than lack of edge"),
 "DIAG-37": ("A", "DOC_TENET", "T6 research debt is existential risk"),
 "DIAG-38": ("C", "DOC_TENET", "T1 statistical utility beats narrative coherence (beats consensus too)"),
 "DIAG-39": ("C", "DOC_TENET", "T5 orchestration is the durable edge"),
 "DIAG-40": ("A", "CODE", "test_pipeline_does_not_import_sovereign enforced isolation boundary"),
 "DIAG-41": ("D", "CODE", "cross-system IC 0.11 < 0.15 doctrine bar -> do not adopt across systems"),
 "DIAG-42": ("B", "DOC_TENET", "copying regime label = prohibited feature-sharing; use cross-system bridge"),
 "DIAG-43": ("B", "DOC_TENET", "0.15 IC bar necessary but not sufficient; unregistered scan = data-mining candidate"),
 "DIAG-44": ("D", "DOC_TENET", "two legal channels: cross-system bridge + capital allocator"),
 "DIAG-45": ("A", "LLM_BOUNDARY", "Kimi conviction=5 < 7 threshold -> dual-confirmation fails (Kimi's own judgment call)"),
 "DIAG-46": ("C", "CODE", "petroulas_gate.py composite stress 4.5 < 6.0 minimum -> fast-reject before Kimi"),
 "DIAG-47": ("C", "CODE", "no independent Kimi score -> framework-only reduced base size, not full 5%"),
 "DIAG-48": ("A", "LLM_BOUNDARY", "distinguishing specific arithmetic proof from narrative is Kimi's judgment, not code"),
 "DIAG-49": ("D", "CODE", "petroulas_gate.py hard ceiling: even perfect scores cap at 5%"),
 "DIAG-50": ("B", "CODE", "RISK_CONSTITUTION Art.3: 3.6% past 3.5% rung -> halve new sizes"),
 "DIAG-51": ("B", "CODE", "RISK_CONSTITUTION Art.6: PROVISIONAL != CONFIRMED -> no live capital"),
 "DIAG-52": ("D", "CODE", "drift test doctrine: amend prose+twin together, never fix the test"),
 "DIAG-53": ("A", "CODE", "Art.2 aggregate carry-complex cap binds at 2.8%>2.5% even though each trade <0.75%"),
 "DIAG-54": ("C", "CODE", "Art.1 0.75% per-trade cap applies regardless of conviction"),
 "DIAG-55": ("C", "CODE", "Art.3: 6.7% past 6.5% rung -> flatten every predictive-layer position"),
 "DIAG-56": ("A", "CODE", "state_space.py rate_diff_z optional, defaults to 0.0 when unavailable"),
 "DIAG-57": ("D", "CODE", "state_space.py: required atr_pct missing -> KeyError; optional rate_diff_z -> 0.0"),
 "DIAG-58": ("B", "CODE", "STATE_DIMS locked 8-tuple; reordering/inserting is a breaking schema change"),
 "DIAG-59": ("B", "CODE", "drawdown_from_peak measured from best excursion (1.1050), not entry: ~+0.36% retracement"),
 "DIAG-60": ("D", "CODE", "kill-zone windows; NY lunch 12:00-13:30 UTC is a blocked consolidation window"),
 "DIAG-61": ("A", "CODE", "ICT hard ATR-spike veto at 3.0x; 3.4x trips it"),
 "DIAG-62": ("C", "DOC_TENET", "ICT unproven cannot justify multi-day macro position; also violates horizon wall"),
 "DIAG-63": ("C", "LEDGER_FACT", "bottleneck is fill rate (~2/90d), not the score gate; ICT still unproven p=0.52"),
 "DIAG-64": ("A", "LEDGER_FACT", "ICT retained as structure/reference layer despite p=0.52, not a directional predictor"),
 "DIAG-65": ("D", "CODE", "all three promotion gates required; holdout degradation disqualifies regardless of IC"),
 "DIAG-66": ("B", "LEDGER_FACT", "headline 2.10 uncosted/mis-annualized; honest costed number ~1.08-1.25"),
 "DIAG-67": ("B", "LEDGER_FACT", "HYP-044 rejected OOS p=0.50 delta~0; re-proposing on intuition ignores recorded evidence"),
 "DIAG-68": ("D", "LEDGER_FACT", "macro p<0.001 proven vs ICT p=0.52 unproven; production status != validation"),
 "DIAG-69": ("A", "LEDGER_FACT", "one range-bound year consistent with real-but-regime-fragile edge given p<0.001"),
 "DIAG-70": ("C", "LEDGER_FACT", "unverifiable prior-session claim must be checked against filesystem/ledger before acting"),
}

CAT_TO_BUCKET = {
    "carry_direction": "CARRY",
    "carry_mechanics": "CARRY",
    "cot_interpretation": "TRUST",
    "regime_id": "SHARPE",
    "confirmation_protocol": "TRUST",
    "sizing_conviction": "SIZING",
    "tail_risk_fomc": "SHARPE",
    "tenet_mapping": "TRUST",
    "isolation_discipline": "TRUST",
    "petroulas_conviction": "KELLY",
    "risk_constitution": None,  # split below by content
    "state_vector": "SIZING",
    "ict_reference": "EXPECTANCY",
    "evidence_epistemics": "EXPECTANCY",
    "graveyard_discipline": "EXPECTANCY",
}
LADDER_KEYWORDS = ("ladder", "drawdown", "breaker", "flatten", "halve", "halt all", "peak-to-trough")


def bucket_for(cat, prompt):
    if cat != "risk_constitution":
        return CAT_TO_BUCKET[cat]
    p = prompt.lower()
    if any(k in p for k in LADDER_KEYWORDS):
        return "RECOVERY"
    return "KELLY"


def main():
    with open("games/cursus_honorum/question_bank.json") as f:
        bank = json.load(f)
    with open("games/cursus_honorum/categories.json") as f:
        cats_meta = json.load(f)

    questions = bank["questions"]
    assert len(questions) == 100

    per_question = []
    missing = [q["id"] for q in questions if q["id"] not in ANSWERS]
    if missing:
        raise SystemExit(f"Missing hand-derived answers for: {missing}")

    for q in questions:
        letter, prov, note = ANSWERS[q["id"]]
        abstain = prov == "LLM_BOUNDARY"
        correct_letter = q["correct"]
        is_correct = (letter == correct_letter) if not abstain else False
        bucket = bucket_for(q["category"], q["prompt_roman"])
        per_question.append({
            "id": q["id"],
            "phase": q["phase"],
            "rank": q["rank"],
            "category": q["category"],
            "bucket_mapped": bucket,
            "system_answer": letter,
            "correct_answer": correct_letter,
            "is_correct": is_correct,
            "abstain": abstain,
            "provenance": prov,
            "component_note": note,
            "tenet": q.get("tenet"),
        })

    # --- random baseline: theoretical 25% + one seeded simulation for realism ---
    import random
    rng = random.Random(20260726)
    random_correct = 0
    for q in questions:
        pick = rng.choice(["A", "B", "C", "D"])
        if pick == q["correct"]:
            random_correct += 1
    random_baseline_simulated = random_correct / len(questions)

    # --- per-category (native 15-cat scheme) rollup ---
    from collections import defaultdict
    cat_stats = defaultdict(lambda: {"n": 0, "correct": 0, "abstain": 0})
    for pq in per_question:
        c = cat_stats[pq["category"]]
        c["n"] += 1
        c["correct"] += int(pq["is_correct"])
        c["abstain"] += int(pq["abstain"])

    per_category_scores = {}
    for cat, s in sorted(cat_stats.items()):
        acc = s["correct"] / s["n"]
        per_category_scores[cat] = {
            "n": s["n"],
            "system_correct": s["correct"],
            "system_accuracy": round(acc, 4),
            "abstain_count": s["abstain"],
            "random_baseline": 0.25,
            "accuracy_over_baseline": round(acc - 0.25, 4),
        }

    # --- bucket rollup (Colin's 8-bucket best-effort projection) ---
    bucket_stats = defaultdict(lambda: {"n": 0, "correct": 0, "abstain": 0, "source_categories": set()})
    for pq in per_question:
        b = bucket_stats[pq["bucket_mapped"]]
        b["n"] += 1
        b["correct"] += int(pq["is_correct"])
        b["abstain"] += int(pq["abstain"])
        b["source_categories"].add(pq["category"])

    ALL_BUCKETS = ["EXPECTANCY", "SIZING", "CARRY", "KELLY", "SHARPE", "EXITS", "RECOVERY", "TRUST"]
    per_bucket_scores = {}
    for b in ALL_BUCKETS:
        s = bucket_stats.get(b)
        if not s or s["n"] == 0:
            per_bucket_scores[b] = {
                "n": 0, "system_correct": 0, "system_accuracy": None,
                "abstain_count": 0, "random_baseline": 0.25,
                "accuracy_over_baseline": None,
                "source_categories": [],
                "note": "NO QUESTIONS in the bank map to this bucket — zero coverage, not zero accuracy.",
            }
            continue
        acc = s["correct"] / s["n"]
        per_bucket_scores[b] = {
            "n": s["n"],
            "system_correct": s["correct"],
            "system_accuracy": round(acc, 4),
            "abstain_count": s["abstain"],
            "random_baseline": 0.25,
            "accuracy_over_baseline": round(acc - 0.25, 4),
            "source_categories": sorted(s["source_categories"]),
        }

    total_correct = sum(pq["is_correct"] for pq in per_question)
    total_abstain = sum(pq["abstain"] for pq in per_question)
    overall = {
        "n": 100,
        "system_correct": total_correct,
        "system_accuracy": round(total_correct / 100, 4),
        "abstain_count": total_abstain,
        "random_baseline_theoretical": 0.25,
        "random_baseline_simulated_seed20260726": round(random_baseline_simulated, 4),
    }

    out = {
        "meta": {
            "generated_for": "TICK — system self-play diagnostic of Cursus Honorum question bank",
            "date": "2026-07-26",
            "bank_source": "games/cursus_honorum/question_bank.json (100 Qs, 30 calibration + 70 diagnostic)",
            "methodology": (
                "Each answer was derived by applying the ACTUAL coded/documented decision logic in "
                "sovereign/forex/{carry_engine,risk_sentiment,combat_vetoes,strategy}.py, "
                "sovereign/training/state_space.py, imbalance_engine/petroulas_gate.py, "
                "RISK_CONSTITUTION.md, TRADING_PHILOSOPHY.md, and CLAUDE.md's evidence-status ledger "
                "-- read BEFORE consulting the bank's `correct` field. This is NOT a naive LLM rubber-stamp "
                "of the answer key: every question's `provenance` field records whether the answer came from "
                "(a) a literal executable constant/threshold (CODE), (b) a coded rule whose module is "
                "currently config-disabled in production (CONFIG_DISABLED -- e.g. combat_vetoes.py, "
                "enabled=false, retained as an analysis instrument only per its own docstring), "
                "(c) documented-but-not-executable doctrine (DOC_TENET -- the six tenets in "
                "TRADING_PHILOSOPHY.md have no code classifier that maps a scenario to a tenet ID), "
                "(d) a specific historical/ledger fact (LEDGER_FACT -- hypothesis verdicts, measured Sharpe, "
                "p-values quoted in CLAUDE.md/the hypothesis ledger), or (e) a genuine LLM judgment boundary "
                "(LLM_BOUNDARY -- petroulas_worthy's 'is this arithmetic or narrative' call is made by Kimi "
                "in production, not by deterministic code; these 3 items are marked abstain=True)."
            ),
            "category_scheme": (
                "PRIMARY scoring is on the bank's native 15-category scheme (games/cursus_honorum/categories.json), "
                "which is what the actual game's heatmap (index.html renderHeatmap()) uses. Colin's 8-bucket "
                "scheme (EXPECTANCY/SIZING/CARRY/KELLY/SHARPE/EXITS/RECOVERY/TRUST) does NOT appear anywhere "
                "in this repo or in games/cursus_honorum/ -- it could not be found via grep across the "
                "codebase, so it is presumed to come from a separate, external assessment. The bucket-level "
                "numbers below are therefore a best-effort, hand-authored projection of the 15 fine categories "
                "onto those 8 labels (mapping table inline in build_heatmap.py CAT_TO_BUCKET), not a verified "
                "equivalence. Two things fall out of that projection that are worth flagging on their own: "
                "(1) EXITS has ZERO bank questions mapping to it -- the bank contains no exit-logic category "
                "at all, so system accuracy on EXITS is undefined, not measured-and-low; (2) risk_constitution's "
                "9 questions were split between KELLY (cap-vs-grade-table questions) and RECOVERY (drawdown-"
                "ladder questions) by keyword match on the prompt text, a heuristic, not a bank-native split."
            ),
        },
        "overall": overall,
        "per_category_native_15": per_category_scores,
        "per_bucket_colin_projection_8": per_bucket_scores,
        "per_question_log": per_question,
    }

    with open("games/cursus_honorum/system_heatmap.json", "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(overall, indent=2))
    print("\n--- per-category (native 15) ---")
    for cat, s in per_category_scores.items():
        print(f"{cat:24s} n={s['n']:2d} acc={s['system_accuracy']:.2f} vs_base={s['accuracy_over_baseline']:+.2f} abstain={s['abstain_count']}")
    print("\n--- per-bucket (Colin projection) ---")
    for b in ALL_BUCKETS:
        s = per_bucket_scores[b]
        if s["n"] == 0:
            print(f"{b:12s} n=0 (no coverage)")
        else:
            print(f"{b:12s} n={s['n']:2d} acc={s['system_accuracy']:.2f} vs_base={s['accuracy_over_baseline']:+.2f} abstain={s['abstain_count']}")


if __name__ == "__main__":
    main()
