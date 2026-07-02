"""Risk Constitution drift tripwire.

RISK_CONSTITUTION.md is the human-readable authority; config/risk_constitution.yaml
is its machine twin. This test extracts the bold percent tokens from the prose and
proves they match the YAML constants exactly.

If a test here fails, the constitution and its twin have drifted. Do NOT fix the
test — amend both files together in the same commit, with a dated entry in the
MD Amendments section (Article 5).

Isolation: imports yaml/re/pathlib only. No live trading module may be imported
here, and no live module may read config/risk_constitution.yaml.
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "RISK_CONSTITUTION.md"
YML_PATH = ROOT / "config" / "risk_constitution.yaml"

# Binding numerics are bold percent tokens: **0.75%**. Sign convention keeps the
# sign outside the bold, but the regex tolerates U+2212 / en-dash / hyphen inside
# and captures the magnitude only.
BOLD_PCT = re.compile(r"\*\*[−–-]?(\d+(?:\.\d+)?)\s*%\*\*")
BOLD_PCT_DRAFT = re.compile(r"\*\*[−–-]?\d+(?:\.\d+)?\s*%\*\*\s*\(DRAFT\)")
PAIR = re.compile(r"`([A-Z]{6})`")
ARTICLE_HEAD = re.compile(r"^## Article (\d+)\b", re.M)

# Exact bold-percent count per article, and in total. A de-bolded binding value,
# an accidentally bolded rationale number, or a new number added without wiring
# all change these counts and MUST fail loudly.
EXPECTED_COUNTS = {1: 1, 2: 1, 3: 3, 4: 0, 5: 0, 6: 0}
EXPECTED_TOTAL = 5

AMEND = "amend RISK_CONSTITUTION.md and config/risk_constitution.yaml together (Article 5)"


def _doc() -> str:
    return MD_PATH.read_text(encoding="utf-8")


def _cfg() -> dict:
    return yaml.safe_load(YML_PATH.read_text(encoding="utf-8"))


def _articles(text: str) -> dict:
    heads = list(ARTICLE_HEAD.finditer(text))
    nums = [int(m.group(1)) for m in heads]
    assert nums == [1, 2, 3, 4, 5, 6], f"expected Articles 1..6 in order, found {nums}"
    bounds = [m.start() for m in heads] + [len(text)]
    return {nums[i]: text[bounds[i]:bounds[i + 1]] for i in range(6)}


def _pcts(section: str) -> list:
    """Percent magnitudes of the bold tokens in a section, e.g. [5.0, 7.0, 8.5]."""
    return [float(v) for v in BOLD_PCT.findall(section)]


def _match(yaml_frac: float, md_pct: float, name: str) -> None:
    assert abs(yaml_frac - md_pct / 100.0) < 1e-12, (
        f"DRIFT on {name}: MD says {md_pct}% but YAML says {yaml_frac} — {AMEND}")


def test_files_exist():
    assert MD_PATH.exists(), f"missing {MD_PATH.name} — the prose half of the constitution pair"
    assert YML_PATH.exists(), f"missing config/{YML_PATH.name} — the machine half of the constitution pair"


def test_yaml_schema_complete():
    cfg = _cfg()
    assert set(cfg) == {"meta", "article_1_per_trade", "article_2_carry_complex", "article_3_breakers"}, (
        f"unexpected top-level YAML keys: {sorted(cfg)}")
    assert cfg["meta"]["source"] == "RISK_CONSTITUTION.md"
    fracs = {
        "article_1_per_trade.hard_cap_frac": cfg["article_1_per_trade"]["hard_cap_frac"],
        "article_2_carry_complex.carry_heat_cap_frac": cfg["article_2_carry_complex"]["carry_heat_cap_frac"],
        "article_3_breakers.halve_sizing_at_dd_frac": cfg["article_3_breakers"]["halve_sizing_at_dd_frac"],
        "article_3_breakers.halt_new_entries_at_dd_frac": cfg["article_3_breakers"]["halt_new_entries_at_dd_frac"],
        "article_3_breakers.flatten_predictive_at_dd_frac": cfg["article_3_breakers"]["flatten_predictive_at_dd_frac"],
    }
    for name, v in fracs.items():
        assert isinstance(v, float), f"{name} must be a float fraction, got {type(v).__name__}"
        assert 0 < v <= 0.10, f"{name}={v} outside sane fraction range (0, 0.10] — wrong unit?"
    pairs = cfg["article_2_carry_complex"]["carry_pairs"]
    assert isinstance(pairs, list) and all(
        isinstance(p, str) and len(p) == 6 and p.isupper() for p in pairs), (
        f"carry_pairs must be 6-char uppercase symbols, got {pairs}")


def test_md_has_six_articles():
    _articles(_doc())


def test_article1_per_trade_cap():
    pcts = _pcts(_articles(_doc())[1])
    assert pcts == [0.75], f"Article 1 must bind exactly one value (0.75%), found {pcts} — {AMEND}"
    _match(_cfg()["article_1_per_trade"]["hard_cap_frac"], pcts[0], "per-trade hard cap")


def test_article2_heat_cap():
    pcts = _pcts(_articles(_doc())[2])
    assert pcts == [2.5], f"Article 2 must bind exactly one value (2.5%), found {pcts} — {AMEND}"
    _match(_cfg()["article_2_carry_complex"]["carry_heat_cap_frac"], pcts[0], "carry heat cap")


def test_article2_pairs_match_yaml():
    """Locks MD<->YAML pair lists only. The constitution deliberately names FIVE
    pairs (AUDNZD included) although the live universe trades four (HYP-045) —
    constitutional scope > live scope. Never compare against the live universe."""
    md_pairs = PAIR.findall(_articles(_doc())[2])
    yml_pairs = _cfg()["article_2_carry_complex"]["carry_pairs"]
    assert len(md_pairs) == 5, f"Article 2 must name exactly 5 backticked pairs, found {md_pairs}"
    assert len(set(md_pairs)) == 5, f"duplicate pair in Article 2: {md_pairs}"
    assert sorted(md_pairs) == sorted(yml_pairs), (
        f"DRIFT: MD pairs {sorted(md_pairs)} != YAML pairs {sorted(yml_pairs)} — {AMEND}")


def test_article3_breakers_match_and_ordered():
    pcts = _pcts(_articles(_doc())[3])
    assert sorted(pcts) == [5.0, 7.0, 8.5], (
        f"Article 3 must bind exactly [5%, 7%, 8.5%], found {pcts} — {AMEND}")
    br = _cfg()["article_3_breakers"]
    _match(br["halve_sizing_at_dd_frac"], 5.0, "halve-sizing breaker")
    _match(br["halt_new_entries_at_dd_frac"], 7.0, "halt-new-entries breaker")
    _match(br["flatten_predictive_at_dd_frac"], 8.5, "flatten-predictive breaker")
    assert (br["halve_sizing_at_dd_frac"] < br["halt_new_entries_at_dd_frac"]
            < br["flatten_predictive_at_dd_frac"] < 0.10), (
        "breaker ladder must escalate and stay below the 10% prop kill line")


def test_no_stray_bold_percentages():
    doc = _doc()
    sections = _articles(doc)
    for n, expected in EXPECTED_COUNTS.items():
        found = len(_pcts(sections[n]))
        assert found == expected, (
            f"Article {n}: expected {expected} bold percent token(s), found {found}. "
            f"A new bold percentage must be wired into config/risk_constitution.yaml AND this "
            f"test; if it is rationale (like the 10% prop kill line), unbold it. {AMEND}")
    total = len(_pcts(doc))
    assert total == EXPECTED_TOTAL, (
        f"whole document must contain exactly {EXPECTED_TOTAL} bold percent tokens, found {total} "
        f"— a number is loose outside the articles. {AMEND}")


def test_draft_markers_until_ratified():
    doc, cfg = _doc(), _cfg()
    if not cfg["meta"]["ratified"]:
        assert cfg["meta"]["status"] == "DRAFT", "unratified constitution must carry meta.status DRAFT"
        assert "STATUS: DRAFT" in doc, "unratified constitution must carry the STATUS: DRAFT banner"
        marked = len(BOLD_PCT_DRAFT.findall(doc))
        assert marked == EXPECTED_TOTAL, (
            f"every binding numeric must carry an individual (DRAFT) marker until ratification: "
            f"{marked}/{EXPECTED_TOTAL} marked")
    else:
        assert cfg["meta"]["status"] != "DRAFT", "ratified constitution must not carry status DRAFT"
        assert "STATUS: DRAFT" not in doc, "ratified constitution must drop the DRAFT banner"
        assert not BOLD_PCT_DRAFT.findall(doc), "ratified constitution must drop all (DRAFT) markers"


def test_yaml_not_wired_into_live_config_loader():
    """The twin exists for this test only. The day live code reads it is the day
    it must be ratified and wired deliberately — not inherited by accident."""
    for py in (ROOT / "config").glob("*.py"):
        src = py.read_text(encoding="utf-8")
        assert "risk_constitution" not in src, (
            f"{py.name} references risk_constitution — the twin must not be wired into live "
            f"config loading while DRAFT")
