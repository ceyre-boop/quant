#!/usr/bin/env python3
"""
COLD system-player run over the Cursus Honorum bank.

Differs from the warm run (games/cursus_honorum/build_heatmap.py /
system_heatmap.json) in exactly one way: the cold player is NOT allowed to use
a numeric threshold value that lives ONLY in Python source (no config/*.yml
backing) when a question's prompt or options hand that exact number over as
scaffolding. It must ABSTAIN on those, even where the warm run answered
correctly, because reciting a hardcoded magic number the question just fed you
is not the same as the system "knowing" a principle.

Threshold provenance was checked file-by-file before scoring (see
HARDCODE_ONLY_THRESHOLDS below and the cold_summary.md report) by grepping
config/*.yml for every constant defined in:
  sovereign/forex/carry_engine.py, sovereign/forex/strategy.py,
  sovereign/forex/risk_sentiment.py, imbalance_engine/petroulas_gate.py
against sovereign/forex/config/combat_vetoes.yaml, config/parameters.yml,
config/ict_params.yml, config/risk_constitution.yaml, config/training.yml.

Result of that check:
  - config/risk_constitution.yaml is a genuine machine twin of Article 1/2/3
    (0.75%, 2.5%, 3.5/5/6.5) -> risk_constitution category stays fully
    config-derivable, no abstention needed there.
  - config/ict_params.yml externalizes ICT's weights, atr_spike_veto_multiplier,
    min_grade_to_execute, kill zones, over-confirmation penalty -> ict_reference
    stays fully config-derivable.
  - sovereign/forex/config/combat_vetoes.yaml externalizes the 4 combat-veto
    constants (deadband, weak_rate, atr_floor, momentum_opp) -> those
    confirmation_protocol questions stay config-derivable (still flagged
    CONFIG_DISABLED since the module itself is enabled:false in prod).
  - sovereign/forex/carry_engine.py, sovereign/forex/strategy.py,
    sovereign/forex/risk_sentiment.py, and imbalance_engine/petroulas_gate.py
    load NO config file at all (verified: no `import yaml`, no config read) --
    every constant in them is a bare Python module/class attribute. THIS is
    the real finding: exactly the four modules CLAUDE.md's "never hardcode
    thresholds" rule warns about.

abstain=True below is set ONLY where the question's own text hands over one
of those hardcoded-only numbers as the deciding fact (i.e. removing the
number from the prompt would make the question unanswerable from config
alone). Where the same category's question is answerable from a doctrine
statement, a config-backed number, or a decision the prompt already hands you
pre-classified (e.g. "the engine has flagged EXTREME_RISK_OFF"), it is still
answered, not abstained -- abstaining there would just be sandbagging a
question that never actually required the hardcoded number.
"""
import json

# id -> (letter, provenance, note, abstain, threshold_source)
# threshold_source: CONFIG | HARDCODE_ONLY | LEDGER_FACT | DOC_TENET | STRUCTURAL | LLM_BOUNDARY | NONE
ANSWERS = {
 "CAL-01": ("B", "CODE", "carry_engine.py high_yield_side rule; 285bp clears any sane floor without needing the exact 100bp constant", False, "NONE"),
 "CAL-02": ("D", "DOC_TENET", "T1 promotion-gate doctrine (an OOS-testing requirement exists); doesn't hinge on the exact 0.15 IC number", False, "NONE"),
 "CAL-03": ("A", "CODE", "RISK_CONSTITUTION Art.1 0.75% -- config/risk_constitution.yaml hard_cap_frac=0.0075", False, "CONFIG"),
 "CAL-04": ("None", "ABSTAIN", "regime classification needs VIX_RISK_ON_THRESHOLD=20.0, hardcoded only in risk_sentiment.py, no config; option text ('VIX<=20') hands it over", True, "HARDCODE_ONLY"),
 "CAL-05": ("C", "CODE", "test_pipeline_does_not_import_sovereign + horizon-wall doctrine, no numeric threshold", False, "NONE"),
 "CAL-06": ("A", "CODE", "N>=200 fresh trades -- config/parameters.yml min_training_samples: 200", False, "CONFIG"),
 "CAL-07": ("None", "ABSTAIN", "CONVICTION_NEUTRAL_THRESHOLD=0.10 hardcoded only in strategy.py, no config; 0.07 vs 0.10 boundary spoon-fed", True, "HARDCODE_ONLY"),
 "CAL-08": ("B", "LEDGER_FACT", "COT Tue-measured/Fri-published procedural fact, not a tunable threshold", False, "NONE"),
 "CAL-09": ("B", "CODE", "combat_vetoes.yaml deadband=0.2 -- config-backed (module enabled=false in prod)", False, "CONFIG"),
 "CAL-10": ("None", "ABSTAIN", "EXTREME_RISK_OFF needs VIX_EXTREME_THRESHOLD=35.0, hardcoded only in risk_sentiment.py, no config", True, "HARDCODE_ONLY"),
 "CAL-11": ("A", "LEDGER_FACT", "ICT permutation p=0.52 is a recorded research result, not a tunable code constant", False, "NONE"),
 "CAL-12": ("C", "CODE", "carry=infrastructure design principle; doesn't hinge on the exact 0.3%/3xATR numbers", False, "NONE"),
 "CAL-13": ("None", "ABSTAIN", "grade_from_signal boundaries (2.0/1.5/0.60) hardcoded only in strategy.py, no config; 1.7/0.55 sits on those exact boundaries", True, "HARDCODE_ONLY"),
 "CAL-14": ("A", "CODE", "state_space.py carry_alignment definition -- structural/definitional, not a tunable threshold", False, "STRUCTURAL"),
 "CAL-15": ("D", "LEDGER_FACT", "rolling walk-forward + p<0.001 recorded results, not a code constant", False, "NONE"),
 "CAL-16": ("B", "DOC_TENET", "cross-system bridge is the sanctioned channel -- doctrine, no number", False, "NONE"),
 "CAL-17": ("None", "ABSTAIN", "petroulas_worthy is Kimi's (LLM) judgment call, not code, AND references MIN_XGB_CONFIDENCE=0.65 (hardcoded only)", True, "LLM_BOUNDARY"),
 "CAL-18": ("D", "CODE", "RISK_CONSTITUTION Art.3 ladder -- config/risk_constitution.yaml article_3_breakers", False, "CONFIG"),
 "CAL-19": ("A", "CODE", "high_yield_side=BASE(NZD) direction rule; 540bp obviously carry-positive without needing exact floor", False, "NONE"),
 "CAL-20": ("C", "LEDGER_FACT", "AUDNZD RBA-correlation + OOS Sharpe -0.879 recorded facts, not a code constant", False, "NONE"),
 "CAL-21": ("C", "DOC_TENET", "never run prop challenge while any system frozen -- doctrine, no number", False, "NONE"),
 "CAL-22": ("A", "CODE", "JPY-inverse mapping + percentile-extreme contrarian rule; qualitative, no exact cutoff needed", False, "NONE"),
 "CAL-23": ("D", "CODE", "Article 4 abstain-on-ambiguity doctrine, no numeric threshold", False, "NONE"),
 "CAL-24": ("B", "CODE", "ict_params.yml scoring.weights (market_structure/pd_alignment=0.0) -- config-backed", False, "CONFIG"),
 "CAL-25": ("B", "CODE", "RISK_CONSTITUTION Art.2 carry-complex 2.5% -- config/risk_constitution.yaml carry_heat_cap_frac=0.025", False, "CONFIG"),
 "CAL-26": ("D", "LEDGER_FACT", "Overnight-QQQ COVID recoupling (rho .42, p=.007) recorded result, not a code constant", False, "NONE"),
 "CAL-27": ("None", "ABSTAIN", "petroulas_gate.py MIN_COMPOSITE_STRESS=6.0 / MIN_XGB_CONFIDENCE=0.65 / MIN_KIMI_MAGNITUDE=7 / MIN_KIMI_CONVICTION=7 all hardcoded only, no config; every gate value spoon-fed", True, "HARDCODE_ONLY"),
 "CAL-28": ("C", "CODE", "multi-day macro hold vs binary event-bar risk -- principle, no numeric threshold", False, "NONE"),
 "CAL-29": ("None", "ABSTAIN", "NEUTRAL needs VIX 20-25 band + FLAT +/-1 term-structure band, both hardcoded only in risk_sentiment.py", True, "HARDCODE_ONLY"),
 "CAL-30": ("A", "DOC_TENET", "T5 orchestration is the durable edge -- doctrine, no number", False, "NONE"),
 "DIAG-01": ("None", "ABSTAIN", "|diff|=90bp vs MIN_CARRY_SPREAD_BPS=100 boundary case; 100bp hardcoded only in carry_engine.py, no config, and question names the constant directly", True, "HARDCODE_ONLY"),
 "DIAG-02": ("B", "CODE", "high_yield_side=BASE(USD) direction rule -- structural rate comparison, not a tunable threshold", False, "STRUCTURAL"),
 "DIAG-03": ("B", "CODE", "carry=fixed-coupon-not-conviction design principle; doesn't hinge on the exact 0.3% number", False, "NONE"),
 "DIAG-04": ("D", "CODE", "high_yield_side=BASE(EUR) direction rule; 390bp obviously carry-positive without needing exact floor", False, "NONE"),
 "DIAG-05": ("None", "ABSTAIN", "285bp->85bp decay crossing the 100bp floor is a genuine boundary case; floor hardcoded only in carry_engine.py, no config", True, "HARDCODE_ONLY"),
 "DIAG-06": ("C", "CODE", "highest-differential JPY-funded pair unwinds hardest -- relative/qualitative reasoning, not floor-dependent", False, "NONE"),
 "DIAG-07": ("C", "CODE", "wide-stop-survives-flushes design principle; doesn't hinge on the exact 3x multiple", False, "NONE"),
 "DIAG-08": ("A", "CODE", "dedicated-ledger-protects-attribution design principle, no numeric threshold", False, "NONE"),
 "DIAG-09": ("None", "ABSTAIN", "0.3%x4=1.2% arithmetic needs the exact CARRY_RISK_PER_PAIR=0.003, hardcoded only in carry_engine.py, no config", True, "HARDCODE_ONLY"),
 "DIAG-10": ("B", "CODE", "fail-loud-on-degraded-data principle, no numeric threshold needed", False, "NONE"),
 "DIAG-11": ("B", "LEDGER_FACT", "COT contrarian feature fires near percentile tails -- qualitative, no exact cutoff given or needed", False, "NONE"),
 "DIAG-12": ("D", "LEDGER_FACT", "holiday-week +3-day look-ahead bias -- procedural fact, not a tunable threshold", False, "NONE"),
 "DIAG-13": ("A", "CODE", "JPY-inverse mapping at a percentile extreme -- qualitative, no exact cutoff needed", False, "NONE"),
 "DIAG-14": ("C", "LEDGER_FACT", "definitional fact about what COT measures, no threshold", False, "NONE"),
 "DIAG-15": ("C", "LEDGER_FACT", "look-ahead-vs-edge doctrine; deciding factor is the lag rule, not the exact IC number", False, "NONE"),
 "DIAG-16": ("None", "ABSTAIN", "CONTANGO/BACKWARDATION classification needs the +/-1.0 term-structure band, hardcoded only (inline magic number) in risk_sentiment.py, no config", True, "HARDCODE_ONLY"),
 "DIAG-17": ("None", "ABSTAIN", "RISK_OFF requires VIX>=25.0 AND BACKWARDATION -- 25.0 hardcoded only in risk_sentiment.py, no config", True, "HARDCODE_ONLY"),
 "DIAG-18": ("B", "CODE", "classification already handed to the player (\"Classifier returns RISK_OFF\"); decision is the forced-override consequence, not deriving the VIX/backwardation boundary", False, "NONE"),
 "DIAG-19": ("None", "ABSTAIN", "RISK_ON needs VIX_RISK_ON_THRESHOLD=20.0, hardcoded only in risk_sentiment.py, no config; option states 'VIX well under 20'", True, "HARDCODE_ONLY"),
 "DIAG-20": ("D", "LEDGER_FACT", "trending-year +0.51/+1.26 vs range-bound -0.13/-0.09 recorded results, not a code constant", False, "NONE"),
 "DIAG-21": ("A", "CODE", "combat_vetoes.yaml weak_rate=0.5 -- config-backed (module enabled=false in prod)", False, "CONFIG"),
 "DIAG-22": ("C", "CODE", "combat_vetoes.yaml deadband=0.2 -- config-backed (module enabled=false in prod)", False, "CONFIG"),
 "DIAG-23": ("C", "CODE", "ict_params.yml execution.min_grade_to_execute: 'A' -- config-backed", False, "CONFIG"),
 "DIAG-24": ("A", "CODE", "ict_params.yml overconfirmation_penalty_threshold=9.0 / slope=0.5 -- config-backed", False, "CONFIG"),
 "DIAG-25": ("D", "CODE", "event-bar entry-risk principle, no numeric threshold", False, "NONE"),
 "DIAG-26": ("B", "CODE", "grade already given (B); config/parameters.yml grade_risk.B=0.005 vs Art.1 cap 0.0075, both config", False, "CONFIG"),
 "DIAG-27": ("B", "CODE", "grade already given (A+); config/parameters.yml grade_risk.A_plus=0.015 vs Art.1 cap 0.0075, both config", False, "CONFIG"),
 "DIAG-28": ("None", "ABSTAIN", "conviction=0.10 sits exactly on CONVICTION_NEUTRAL_THRESHOLD, hardcoded only in strategy.py, no config", True, "HARDCODE_ONLY"),
 "DIAG-29": ("A", "CODE", "aggregate carry-complex heat vs Art.2 2.5% cap -- both config/risk_constitution.yaml", False, "CONFIG"),
 "DIAG-30": ("C", "CODE", "grade already given (C); config/parameters.yml grade_risk.C=0.0025, config", False, "CONFIG"),
 "DIAG-31": ("None", "ABSTAIN", "EXTREME_RISK_OFF needs VIX_EXTREME_THRESHOLD=35.0, hardcoded only, no config; 33 sits just under it, option names '35 extreme line'", True, "HARDCODE_ONLY"),
 "DIAG-32": ("A", "CODE", "USD-shock-cuts-against-long principle, no regime threshold needed", False, "NONE"),
 "DIAG-33": ("D", "CODE", "classification already handed to the player (\"engine has flagged EXTREME_RISK_OFF\"); decision is the forced-override consequence, not deriving the VIX/backwardation boundary", False, "NONE"),
 "DIAG-34": ("B", "CODE", "config/parameters.yml hard_constraints.max_daily_loss_pct=0.02 -- config-backed", False, "CONFIG"),
 "DIAG-35": ("B", "DOC_TENET", "T3 systems must know when unreliable -- doctrine, no number", False, "NONE"),
 "DIAG-36": ("D", "DOC_TENET", "T4 premature complexity -- doctrine, no number", False, "NONE"),
 "DIAG-37": ("A", "DOC_TENET", "T6 research debt -- doctrine, no number", False, "NONE"),
 "DIAG-38": ("C", "DOC_TENET", "T1 statistical utility beats consensus; the tenet mapping doesn't hinge on the exact 0.15 IC number, only on 'low IC + unanimous agreement'", False, "NONE"),
 "DIAG-39": ("C", "DOC_TENET", "T5 orchestration -- doctrine, no number", False, "NONE"),
 "DIAG-40": ("A", "CODE", "test_pipeline_does_not_import_sovereign -- structural/enforced, not a tunable threshold", False, "STRUCTURAL"),
 "DIAG-41": ("None", "ABSTAIN", "cross-system IC=0.11 vs the 0.15 bar is a genuine boundary case; 0.15 lives only as a documented number in sovereign/forensics/latent_feature_search.py, no config", True, "HARDCODE_ONLY"),
 "DIAG-42": ("B", "DOC_TENET", "cross-system bridge is the sanctioned channel -- doctrine, no number", False, "NONE"),
 "DIAG-43": ("B", "DOC_TENET", "0.15 IC clearing already stated in the prompt (not derived); deciding factor is the pre-registration/data-mining doctrine, not the exact bar", False, "NONE"),
 "DIAG-44": ("D", "DOC_TENET", "naming the two sanctioned channels -- doctrine, no number", False, "NONE"),
 "DIAG-45": ("None", "ABSTAIN", "Kimi conviction=5 vs MIN_KIMI_CONVICTION=7 (hardcoded only, no config) AND petroulas_worthy is an LLM judgment", True, "LLM_BOUNDARY"),
 "DIAG-46": ("None", "ABSTAIN", "composite stress=4.5 vs MIN_COMPOSITE_STRESS=6.0, hardcoded only in petroulas_gate.py, no config", True, "HARDCODE_ONLY"),
 "DIAG-47": ("None", "ABSTAIN", "fallback-size formula uses PETROULAS_BASE_PCT=3.0/NORMAL_SIZE_PCT=1.5, hardcoded only, no config", True, "HARDCODE_ONLY"),
 "DIAG-48": ("None", "ABSTAIN", "distinguishing arithmetic from narrative is Kimi's (LLM) judgment, not code, AND cites MIN_XGB_CONFIDENCE=0.65 (hardcoded only)", True, "LLM_BOUNDARY"),
 "DIAG-49": ("None", "ABSTAIN", "5% ceiling is PETROULAS_MAX_PCT=5.0, hardcoded only in petroulas_gate.py, no config", True, "HARDCODE_ONLY"),
 "DIAG-50": ("B", "CODE", "RISK_CONSTITUTION Art.3 -- config/risk_constitution.yaml article_3_breakers", False, "CONFIG"),
 "DIAG-51": ("B", "CODE", "RISK_CONSTITUTION Art.6 PROVISIONAL-vs-CONFIRMED doctrine, no numeric threshold", False, "NONE"),
 "DIAG-52": ("D", "CODE", "drift-test/twin-amendment doctrine, no numeric threshold", False, "NONE"),
 "DIAG-53": ("A", "CODE", "aggregate carry-complex cap vs per-trade cap -- both config/risk_constitution.yaml", False, "CONFIG"),
 "DIAG-54": ("C", "CODE", "Art.1 0.75% per-trade cap -- config/risk_constitution.yaml hard_cap_frac=0.0075", False, "CONFIG"),
 "DIAG-55": ("C", "CODE", "RISK_CONSTITUTION Art.3 -- config/risk_constitution.yaml article_3_breakers", False, "CONFIG"),
 "DIAG-56": ("A", "CODE", "state_space.py optional-field default-to-0.0 -- structural/definitional, not a tunable threshold", False, "STRUCTURAL"),
 "DIAG-57": ("D", "CODE", "state_space.py required-vs-optional field handling -- structural, not a tunable threshold", False, "STRUCTURAL"),
 "DIAG-58": ("B", "CODE", "STATE_DIMS locked 8-tuple -- structural schema contract, not a tunable threshold", False, "STRUCTURAL"),
 "DIAG-59": ("B", "CODE", "drawdown_from_peak arithmetic definition -- structural formula, not a tunable threshold", False, "STRUCTURAL"),
 "DIAG-60": ("D", "CODE", "ict_params.yml kill_zones -- config-backed", False, "CONFIG"),
 "DIAG-61": ("A", "CODE", "ict_params.yml atr_spike_veto_multiplier=3.0 -- config-backed", False, "CONFIG"),
 "DIAG-62": ("C", "DOC_TENET", "ICT-unproven + horizon-isolation doctrine, no numeric threshold", False, "NONE"),
 "DIAG-63": ("C", "LEDGER_FACT", "fill-rate-is-the-bottleneck recorded finding + ICT unproven p=0.52, not a code constant", False, "NONE"),
 "DIAG-64": ("A", "LEDGER_FACT", "ICT retained as reference layer despite p=0.52 -- recorded status, not a code constant", False, "NONE"),
 "DIAG-65": ("D", "CODE", "all-three-gates-required doctrine; deciding factor is holdout degradation, not the exact 0.15 IC number (already stated as clearing it)", False, "NONE"),
 "DIAG-66": ("B", "LEDGER_FACT", "uncosted/mis-annualized 2.10 vs costed 1.08-1.25 -- recorded measurement, not a code constant", False, "NONE"),
 "DIAG-67": ("B", "LEDGER_FACT", "HYP-044 rejected OOS p=0.50 -- recorded ledger verdict, not a code constant", False, "NONE"),
 "DIAG-68": ("D", "LEDGER_FACT", "macro p<0.001 vs ICT p=0.52 -- recorded statistics, not code constants", False, "NONE"),
 "DIAG-69": ("A", "LEDGER_FACT", "one range-bound year vs p<0.001 permutation -- recorded statistics, not a code constant", False, "NONE"),
 "DIAG-70": ("C", "DOC_TENET", "verify-before-acting doctrine, no numeric threshold", False, "NONE"),
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
    "risk_constitution": None,
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


HARDCODE_ONLY_THRESHOLDS = [
    {"file": "sovereign/forex/carry_engine.py", "constant": "MIN_CARRY_SPREAD_BPS", "value": "100 (bps)",
     "used_by": ["DIAG-01", "DIAG-05"]},
    {"file": "sovereign/forex/carry_engine.py", "constant": "CARRY_RISK_PER_PAIR", "value": "0.003 (0.3%)",
     "used_by": ["DIAG-09"]},
    {"file": "sovereign/forex/carry_engine.py", "constant": "ATR_STOP_MULTIPLE", "value": "3.0",
     "used_by": []},
    {"file": "sovereign/forex/strategy.py", "constant": "CONVICTION_NEUTRAL_THRESHOLD", "value": "0.10",
     "used_by": ["CAL-07", "DIAG-28"]},
    {"file": "sovereign/forex/strategy.py", "constant": "CONVICTION_FULL_SIZE", "value": "0.70",
     "used_by": ["DIAG-28"]},
    {"file": "sovereign/forex/strategy.py", "constant": "grade_from_signal() inline boundaries", "value": "|diff|>=2.0 & conv>=0.60 (A+), >=1.5 (A), >=0.5 (B)",
     "used_by": ["CAL-13"]},
    {"file": "sovereign/forex/risk_sentiment.py", "constant": "VIX_RISK_OFF_THRESHOLD", "value": "25.0",
     "used_by": ["CAL-29", "DIAG-17"]},
    {"file": "sovereign/forex/risk_sentiment.py", "constant": "VIX_RISK_ON_THRESHOLD", "value": "20.0",
     "used_by": ["CAL-04", "CAL-29", "DIAG-19"]},
    {"file": "sovereign/forex/risk_sentiment.py", "constant": "VIX_EXTREME_THRESHOLD", "value": "35.0",
     "used_by": ["CAL-10", "DIAG-31"]},
    {"file": "sovereign/forex/risk_sentiment.py", "constant": "term-structure CONTANGO/BACKWARDATION band (inline, unnamed)", "value": "+/-1.0",
     "used_by": ["DIAG-16", "CAL-29"]},
    {"file": "imbalance_engine/petroulas_gate.py", "constant": "MIN_COMPOSITE_STRESS", "value": "6.0",
     "used_by": ["CAL-27", "DIAG-46"]},
    {"file": "imbalance_engine/petroulas_gate.py", "constant": "MIN_XGB_CONFIDENCE", "value": "0.65",
     "used_by": ["CAL-17", "CAL-27", "DIAG-45", "DIAG-48"]},
    {"file": "imbalance_engine/petroulas_gate.py", "constant": "MIN_KIMI_MAGNITUDE", "value": "7",
     "used_by": ["CAL-27", "DIAG-45"]},
    {"file": "imbalance_engine/petroulas_gate.py", "constant": "MIN_KIMI_CONVICTION", "value": "7",
     "used_by": ["CAL-27", "DIAG-45"]},
    {"file": "imbalance_engine/petroulas_gate.py", "constant": "NORMAL_SIZE_PCT", "value": "1.5",
     "used_by": ["DIAG-47"]},
    {"file": "imbalance_engine/petroulas_gate.py", "constant": "PETROULAS_BASE_PCT", "value": "3.0",
     "used_by": ["DIAG-47"]},
    {"file": "imbalance_engine/petroulas_gate.py", "constant": "PETROULAS_MAX_PCT", "value": "5.0",
     "used_by": ["DIAG-49"]},
    {"file": "sovereign/forensics/latent_feature_search.py (doctrine comment, not config)", "constant": "cross-system IC promotion bar", "value": "0.15",
     "used_by": ["DIAG-41"]},
]

CONFIG_BACKED_FOR_CONTRAST = [
    "config/risk_constitution.yaml (Art.1 0.75%, Art.2 2.5%, Art.3 3.5/5/6.5 ladder) -- proper pattern",
    "config/parameters.yml (grade_risk table A_plus/A/B/C, hard_constraints.max_daily_loss_pct, min_training_samples: 200) -- proper pattern",
    "config/ict_params.yml (scoring weights, atr_spike_veto_multiplier, min_grade_to_execute, kill_zones, overconfirmation_penalty_*) -- proper pattern",
    "sovereign/forex/config/combat_vetoes.yaml (deadband, weak_rate, atr_floor, momentum_opp) -- proper pattern, though the module itself is enabled:false in prod",
]


def main():
    with open("games/cursus_honorum/question_bank.json") as f:
        bank = json.load(f)

    questions = bank["questions"]
    assert len(questions) == 100
    missing = [q["id"] for q in questions if q["id"] not in ANSWERS]
    if missing:
        raise SystemExit(f"Missing hand-derived answers for: {missing}")

    per_question = []
    for q in questions:
        letter, prov, note, abstain, tsrc = ANSWERS[q["id"]]
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
            "threshold_source": tsrc,
            "component_note": note,
            "tenet": q.get("tenet"),
        })

    import random
    rng = random.Random(20260726)
    random_correct = sum(1 for q in questions if rng.choice(["A", "B", "C", "D"]) == q["correct"])
    random_baseline_simulated = random_correct / len(questions)

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
            "n": s["n"], "system_correct": s["correct"], "system_accuracy": round(acc, 4),
            "abstain_count": s["abstain"], "random_baseline": 0.25,
            "accuracy_over_baseline": round(acc - 0.25, 4),
        }

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
                "n": 0, "system_correct": 0, "system_accuracy": None, "abstain_count": 0,
                "random_baseline": 0.25, "accuracy_over_baseline": None, "source_categories": [],
                "note": "NO QUESTIONS in the bank map to this bucket -- zero coverage, not zero accuracy.",
            }
            continue
        acc = s["correct"] / s["n"]
        per_bucket_scores[b] = {
            "n": s["n"], "system_correct": s["correct"], "system_accuracy": round(acc, 4),
            "abstain_count": s["abstain"], "random_baseline": 0.25,
            "accuracy_over_baseline": round(acc - 0.25, 4),
            "source_categories": sorted(s["source_categories"]),
        }

    total_correct = sum(pq["is_correct"] for pq in per_question)
    total_abstain = sum(pq["abstain"] for pq in per_question)
    overall = {
        "n": 100, "system_correct": total_correct, "system_accuracy": round(total_correct / 100, 4),
        "abstain_count": total_abstain,
        "answered_accuracy_excl_abstentions": round(total_correct / (100 - total_abstain), 4),
        "random_baseline_theoretical": 0.25,
        "random_baseline_simulated_seed20260726": round(random_baseline_simulated, 4),
    }

    out = {
        "meta": {
            "generated_for": "Cold system self-play diagnostic (hardcoded-threshold-blind) over the Cursus Honorum bank",
            "date": "2026-07-26",
            "relationship_to_warm_run": (
                "This is a SEPARATE, cold-blind pass. games/cursus_honorum/system_heatmap.json (the warm "
                "run) let the player apply any numeric threshold the question text supplied, even when that "
                "number lives only in Python source with no config backing (e.g. Petroulas's Kimi thresholds, "
                "combat-veto deadbands, CONVICTION_NEUTRAL_THRESHOLD) -- so it wasn't fully testing whether "
                "the system's OWN knowledge of those numbers is principled/config-driven vs merely hardcoded "
                "and handed back to it. This cold run abstains wherever a question's correct answer genuinely "
                "depends on such a hardcoded-only number. data/cursus_honorum/alphazero_run_01.json is a THIRD, "
                "independent parallel run (Colin's) with the same caveat as the original warm run -- preserved "
                "untouched; not modified or read into this script."
            ),
            "methodology": (
                "Every constant referenced by a question was traced to its defining file BEFORE abstaining or "
                "answering. config/risk_constitution.yaml, config/parameters.yml, config/ict_params.yml, and "
                "sovereign/forex/config/combat_vetoes.yaml DO externalize their respective thresholds (risk "
                "caps/ladder, grade-size table + daily-loss cap + training sample floor, ICT weights/gates, "
                "combat-veto deadbands) -- questions resting on those stay answered. sovereign/forex/"
                "carry_engine.py, sovereign/forex/strategy.py, sovereign/forex/risk_sentiment.py, and "
                "imbalance_engine/petroulas_gate.py load NO config file at all (grepped for `import yaml` / "
                "config reads -- none found); every constant in them is a bare module/class attribute. Where a "
                "question's correct answer hinges on one of THOSE numbers, and the question hands the number "
                "over as scaffolding, the cold player abstains. Where the same category's question is doctrine, "
                "a recorded ledger fact, a structural/definitional fact (state-vector schema, direction rule "
                "from raw rate comparison), or a config-backed number, it stays answered -- abstaining there "
                "would understate the system's real, non-hardcoded knowledge just as badly as the warm run "
                "overstated it."
            ),
        },
        "overall": overall,
        "per_category_native_15": per_category_scores,
        "per_bucket_colin_projection_8": per_bucket_scores,
        "hardcoded_threshold_findings": HARDCODE_ONLY_THRESHOLDS,
        "config_backed_for_contrast": CONFIG_BACKED_FOR_CONTRAST,
        "per_question_log": per_question,
    }

    with open("games/cursus_honorum/system_heatmap_cold.json", "w") as f:
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
