"""Phase 1 — Trump statement event catalog + classification (HYP-085, spec §7-P1).

Sources (spec §3, primary venues only), each cached raw under data/raw/<source>/:
  1. Federal Register API (executive orders, documented public JSON API)
  2. whitehouse.gov listings (presidential-actions, briefings-statements, fact-sheets)
  3. Truth Social probe chain (archive mirrors; polite, no bot-wall fighting)
  4. data/manual_events.jsonl — spec-authorized curated fallback (rows classified
     by the SAME rules; scraped rows beat manual duplicates)

Classification is DETERMINISTIC (classification_rules.py, committed before this
script first ran on scraped text): qualifies = entity AND policy AND action.
Matched literals land in `notes` for audit. One output row per event x instrument;
5-trading-day per-instrument keep-first de-dup (drops logged).

HONESTY GATE (spec §7): fewer than 30 qualifying events -> proceed, state the
shortfall loudly, NEVER loosen the definition.

Run:  python3 research/political_alpha/build_event_catalog.py [--offline] [--max-pages N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402
from classification_rules import (  # noqa: E402
    ACTION_PATTERNS, ENTITY_MAP, POLICY_PATTERNS, STUDY_START,
)

RAW = _lib.RAW_DIR
UA = {"User-Agent": "alta-research-event-study/1.0 (public-data academic event study)"}
TEXT_CAP = 2000          # chars of statement_text committed to the catalog (full text in raw/)

_ENTITY_RX = [(label, re.compile(rx, re.I), pairs) for label, rx, pairs in ENTITY_MAP]
_POLICY_RX = [(label, re.compile(rx, re.I)) for label, rx in POLICY_PATTERNS]
_ACTION_RX = re.compile(ACTION_PATTERNS, re.I)


# ── tiny cached HTTP layer ───────────────────────────────────────────────────────────

def _cache_file(source: str, key: str, ext: str) -> Path:
    h = hashlib.sha256(key.encode()).hexdigest()[:20]
    return RAW / source / f"{h}{ext}"


def http_get(url: str, source: str, offline: bool, timeout: int = 30,
             pause: float = 0.4) -> str | None:
    """GET with raw-response caching. offline=True never touches the network.
    Returns None on any failure (callers proceed; nothing is fabricated)."""
    cache = _cache_file(source, url, ".body")
    if cache.exists():
        return cache.read_text(errors="replace")
    if offline:
        return None
    try:
        import requests
        resp = requests.get(url, headers=UA, timeout=timeout)
        time.sleep(pause)
        if resp.status_code != 200:
            print(f"  [{source}] HTTP {resp.status_code}: {url[:100]}")
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text)
        return resp.text
    except Exception as exc:
        print(f"  [{source}] FAIL {type(exc).__name__}: {url[:100]}")
        return None


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", html)
    m = re.search(r"(?is)<main[^>]*>(.*?)</main>", html)
    if m:
        html = m.group(1)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def _noon_et_utc(date_str: str) -> str:
    """A clock-less date (FR signing_date) pinned to 12:00 ET, expressed UTC."""
    local = datetime.fromisoformat(date_str + "T12:00:00").replace(tzinfo=_lib.ET)
    return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── source 1: Federal Register executive orders ──────────────────────────────────────

def fetch_federal_register(offline: bool) -> list[dict]:
    # DISCLOSED OPERATIONALIZATION (decided before any statistics were computed):
    # proclamations are included alongside executive orders — Section 232 tariff
    # actions (steel/aluminum/autos, the spec's own SLX mapping) are issued as
    # signed, FR-published, timestamped PROCLAMATIONS, the same venue class as EOs.
    # Presidential memoranda remain excluded. Stated in the Phase-1 note + report.
    base = ("https://www.federalregister.gov/api/v1/documents.json"
            "?conditions%5Bpresidential_document_type%5D%5B%5D=executive_order"
            "&conditions%5Bpresidential_document_type%5D%5B%5D=proclamation"
            "&conditions%5Bpresident%5D=donald-trump"
            "&conditions%5Bcorrection%5D=0"
            f"&conditions%5Bsigning_date%5D%5Bgte%5D={STUDY_START}"
            "&conditions%5Btype%5D%5B%5D=PRESDOCU"
            "&fields%5B%5D=document_number&fields%5B%5D=title&fields%5B%5D=abstract"
            "&fields%5B%5D=signing_date&fields%5B%5D=publication_date"
            "&fields%5B%5D=html_url&fields%5B%5D=raw_text_url&fields%5B%5D=executive_order_number"
            "&per_page=100&order=oldest")
    out, url, page = [], base, 1
    while url and page <= 20:
        body = http_get(url, "federal_register", offline)
        if body is None:
            break
        doc = json.loads(body)
        for d in doc.get("results", []):
            text = ""
            if d.get("raw_text_url"):
                text = _strip_html(http_get(d["raw_text_url"], "federal_register", offline) or "")
            out.append({
                "timestamp_utc": _noon_et_utc(d["signing_date"]),
                "source": "federal_register",
                "source_url": d.get("html_url", ""),
                "title": d.get("title", ""),
                "text": (d.get("abstract") or "") + " " + text,
                "venue_class": "primary",
                "ts_note": "signing time unknown; pinned 12:00 ET",
            })
        url = doc.get("next_page_url")
        page += 1
    print(f"  federal_register: {len(out)} executive orders")
    return out


# ── source 2: whitehouse.gov listings ────────────────────────────────────────────────

WH_LISTINGS = ["presidential-actions", "briefings-statements", "fact-sheets", "articles"]
_WH_LINK_RX = re.compile(
    r'href="(https://www\.whitehouse\.gov/(?:presidential-actions|briefings-statements|'
    r'fact-sheets|articles|remarks)/20(?:25|26)/[^"#?]+/?)"')
_WH_TIME_RX = re.compile(r'property="article:published_time"\s+content="([^"]+)"')
_WH_TIME_RX2 = re.compile(r'"datePublished"\s*:\s*"([^"]+)"')
_WH_TITLE_RX = re.compile(r'property="og:title"\s+content="([^"]+)"')


def fetch_whitehouse(offline: bool, max_pages: int) -> list[dict]:
    links: list[str] = []
    seen = set()
    for listing in WH_LISTINGS:
        empty_streak = 0
        for page in range(1, max_pages + 1):
            url = (f"https://www.whitehouse.gov/{listing}/"
                   if page == 1 else f"https://www.whitehouse.gov/{listing}/page/{page}/")
            body = http_get(url, "whitehouse", offline)
            if body is None:
                break
            new = [u for u in _WH_LINK_RX.findall(body) if u not in seen]
            for u in new:
                seen.add(u)
                links.append(u)
            empty_streak = empty_streak + 1 if not new else 0
            if empty_streak >= 2:
                break
    print(f"  whitehouse: {len(links)} article links found")

    out = []
    for u in links:
        body = http_get(u, "whitehouse", offline)
        if body is None:
            continue
        tm = _WH_TIME_RX.search(body) or _WH_TIME_RX2.search(body)
        if not tm:
            continue
        try:
            ts = datetime.fromisoformat(tm.group(1).replace("Z", "+00:00"))
            ts_utc = ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
        if ts_utc[:10] < STUDY_START:
            continue
        title_m = _WH_TITLE_RX.search(body)
        out.append({
            "timestamp_utc": ts_utc,
            "source": "whitehouse",
            "source_url": u,
            "title": title_m.group(1) if title_m else u.rstrip("/").rsplit("/", 1)[-1],
            "text": _strip_html(body),
            "venue_class": "primary",
            "ts_note": "",
        })
    print(f"  whitehouse: {len(out)} dated articles in study window")
    return out


# ── source 3: Truth Social probe chain ───────────────────────────────────────────────

_TS_TIME_RX = re.compile(r"([A-Z][a-z]+ \d{1,2}, \d{4}, \d{1,2}:\d{2} (?:AM|PM))")
_TS_NEXT_RX = re.compile(r'href="(https://trumpstruth\.org/?\?[^"]*cursor=[^"]+)"[^>]*>\s*Next Page')


def _ts_et_to_utc(display: str) -> str | None:
    """Mirror displays EASTERN time (verified 2026-07-08 two ways: the 2025-04-09
    '90 Day PAUSE' status shows 1:18 PM — its extensively documented 13:18 ET post
    time — and the pagination cursor's status_created_at is the UTC equivalent of
    the displayed time, e.g. '11:06 AM' display ↔ '15:06:29' cursor)."""
    try:
        naive = datetime.strptime(display, "%B %d, %Y, %I:%M %p")
        return naive.replace(tzinfo=_lib.ET).astimezone(timezone.utc)\
            .strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def fetch_truth_social(offline: bool) -> list[dict]:
    """trumpstruth.org mirror of @realDonaldTrump, walked via its cursor-paginated
    listing (page=N is ignored by the site; the Next-Page cursor link is the real
    mechanism). Listing blocks embed full post text + display time + permalink, so
    no per-status fetches are needed. ReTruths of other accounts are excluded
    (not Trump's own words). All-fail is an accepted, reported outcome."""
    out: list[dict] = []

    url = f"https://trumpstruth.org/?sort=desc&per_page=50&start_date={STUDY_START}"
    for page_n in range(1, 401):                     # ~215 pages expected; hard cap
        body = http_get(url, "truth_social", offline, timeout=25, pause=0.35)
        if body is None:
            break
        blocks = body.split('data-status-url=')
        oldest = None
        for b in blocks[1:]:
            m_link = re.match(r'"(https://trumpstruth\.org/statuses/\d+)\s*"', b)
            m_time = _TS_TIME_RX.search(b)
            if not (m_link and m_time):
                continue
            ts_utc = _ts_et_to_utc(m_time.group(1))
            if ts_utc is None:
                continue
            oldest = ts_utc
            if ts_utc[:10] < STUDY_START:
                continue
            b = re.sub(r"<[^>]*$", "", b)            # drop unterminated tag at block tail
            # excise shared-link preview cards (og-title/description are the ARTICLE'S
            # words, not Trump's — same disclosed tightening as the URL strip)
            b = re.sub(r'(?s)<a[^>]*class="status-card[^"]*"[^>]*>.*?</a>', " ", b)
            b = re.sub(r'(?s)<div[^>]*class="status-card[^"]*"[^>]*>.*?</div>', " ", b)
            txt = re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", b))
            prefix = re.compile(
                r'^"[^"]*"\s*>\s*Donald J\. Trump\s+@realDonaldTrump\s*·\s*'
                r"[A-Z][a-z]+ \d{1,2}, \d{4}, \d{1,2}:\d{2} (?:AM|PM)\s*")
            if not prefix.match(txt):
                continue                             # not authored by @realDonaldTrump (retruth)
            txt = prefix.sub("", txt)
            txt = re.sub(r"(&nbsp;|\s)*Original Post\s*", " ", txt).strip()
            orig = re.search(r'href="(https://truthsocial\.com/@realDonaldTrump/\d+[^"]*)"', b)
            out.append({
                "timestamp_utc": ts_utc,
                "source": "truth_social",
                "source_url": orig.group(1) if orig else m_link.group(1),
                "title": txt[:120],
                "text": txt,
                "venue_class": "primary",
                "ts_note": "via trumpstruth.org mirror (ET display, verified; converted to UTC)",
            })
        m_next = _TS_NEXT_RX.search(body)
        if not m_next or (oldest and oldest[:10] < STUDY_START):
            break
        url = m_next.group(1).replace("&amp;", "&")
    if out:
        print(f"  truth_social: {len(out)} own statuses via trumpstruth.org ({page_n} pages)")
        return out

    # probe B: truthsocial.com Mastodon-fork API (one polite attempt; Cloudflare likely)
    body = http_get("https://truthsocial.com/api/v1/accounts/lookup?acct=realDonaldTrump",
                    "truth_social", offline, timeout=15)
    if body:
        try:
            acct_id = json.loads(body)["id"]
            feed = http_get(f"https://truthsocial.com/api/v1/accounts/{acct_id}/statuses?limit=40",
                            "truth_social", offline, timeout=15)
            for st in json.loads(feed or "[]"):
                ts_utc = datetime.fromisoformat(
                    st["created_at"].replace("Z", "+00:00")).astimezone(timezone.utc)\
                    .strftime("%Y-%m-%dT%H:%M:%SZ")
                if ts_utc[:10] < STUDY_START:
                    continue
                out.append({
                    "timestamp_utc": ts_utc, "source": "truth_social",
                    "source_url": st.get("url", ""), "title": _strip_html(st.get("content", ""))[:120],
                    "text": _strip_html(st.get("content", "")), "venue_class": "primary",
                    "ts_note": "via truthsocial.com API",
                })
        except Exception as exc:
            print(f"  truth_social API parse fail: {type(exc).__name__}")
    if out:
        print(f"  truth_social: {len(out)} statuses via API")
    else:
        print("  truth_social: ALL PROBES FAILED — coverage gap will be stated in the report")
    return out


# ── source 4: manual fallback ────────────────────────────────────────────────────────

def load_manual() -> list[dict]:
    rows = _lib.read_jsonl(_lib.DATA_DIR / "manual_events.jsonl")
    out = []
    for r in rows:
        out.append({
            "timestamp_utc": r["timestamp_utc"],
            "source": "manual",
            "source_url": r["source_url"],
            "title": r.get("title", ""),
            "text": r.get("text", r.get("title", "")),
            "venue_class": "primary_manual",
            "ts_note": r.get("notes", ""),
        })
    if out:
        print(f"  manual: {len(out)} curated rows")
    return out


# ── classification ───────────────────────────────────────────────────────────────────

_URL_RX = re.compile(r"(?:https?:)?\s*//\s*\S+(?:\s+[a-z0-9./_-]+)*")


def classify(cand: dict) -> dict:
    title, body = cand.get("title", ""), cand.get("text", "")
    if cand.get("source") == "truth_social":
        # DISCLOSED TIGHTENING (post-first-run, 2026-07-08): shared-link slugs are
        # stripped before matching, so a post qualifies only on Trump's OWN words —
        # sharing an article titled "...steel-tariffs-must-be-permanent" is
        # endorsement of commentary, not "announcing/previewing a policy action"
        # (locked §2). Fields are stripped SEPARATELY (title is derived from text;
        # concatenating first let a duplicated URL shed its scheme and leak slug
        # words past the filter). The mirror space-breaks long URLs; the
        # continuation class eats only lowercase slug tokens, so composed text
        # (capitalized / ALL CAPS) survives.
        title, body = _URL_RX.sub(" ", title), _URL_RX.sub(" ", body)
    text = f"{title} {body}"
    entities, pairs = [], []
    for label, rx, pp in _ENTITY_RX:
        if rx.search(text):
            entities.append(label)
            pairs.extend(pp)
    policy = next((label for label, rx in _POLICY_RX if rx.search(text)), None)
    action = bool(_ACTION_RX.search(text)) or cand["source"] == "federal_register"
    qualifies = bool(entities) and policy is not None and action
    # unique (sector, instrument) preserving order
    seen, uniq = set(), []
    for sec, inst in pairs:
        if inst not in seen:
            seen.add(inst)
            uniq.append((sec, inst))
    return {"qualifies": qualifies, "policy_action": policy or "",
            "pairs": uniq, "entities": entities,
            "action_matched": action}


# ── 5-trading-day per-instrument de-dup (keep first; log drops) ──────────────────────

def dedup_min_separation(rows: list[dict], index_by_instrument: dict, min_sep: int = 5):
    kept, dropped = [], []
    by_inst: dict[str, list[dict]] = {}
    for r in rows:
        by_inst.setdefault(r["instrument_tagged"], []).append(r)
    for inst, group in by_inst.items():
        idx = index_by_instrument.get(inst)
        group = sorted(group, key=lambda r: (r["t0"], r["event_id"]))
        last_pos, last_id = None, None
        for r in group:
            if idx is None or r["t0"] is None:
                kept.append(r)          # no calendar -> Phase 2 will gap it, not here
                continue
            pos = int(idx.searchsorted(r["t0"]))
            if last_pos is not None and (pos - last_pos) < min_sep:
                dropped.append({"event_id": r["event_id"], "instrument_tagged": inst,
                                "kept_event_id": last_id,
                                "gap_trading_days": int(pos - last_pos),
                                "reason": "min_separation_5td"})
                continue
            kept.append(r)
            last_pos, last_id = pos, r["event_id"]
    return kept, dropped


# ── main ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="replay raw cache only")
    ap.add_argument("--max-pages", type=int, default=80)
    args = ap.parse_args()

    print("Phase 1 — building Trump statement event catalog")
    cands = (fetch_federal_register(args.offline)
             + fetch_whitehouse(args.offline, args.max_pages)
             + fetch_truth_social(args.offline)
             + load_manual())
    print(f"  candidates total: {len(cands)}")

    # classify everything; keep the full candidate record in raw/ for audit
    classified = []
    for c in cands:
        cls = classify(c)
        classified.append({**c, **cls})
    _lib.write_jsonl(RAW / "candidates_all.jsonl",
                     [{k: v for k, v in c.items() if k != "pairs"} |
                      {"instruments": [i for _, i in c["pairs"]]} for c in classified])

    qual = [c for c in classified if c["qualifies"]]
    print(f"  qualifying statements: {len(qual)} / {len(cands)}")

    # scraped beats manual on (UTC date, policy_action, instrument-set overlap)
    scraped_keys = set()
    for c in qual:
        if c["source"] != "manual":
            for _, inst in c["pairs"]:
                scraped_keys.add((c["timestamp_utc"][:10], c["policy_action"], inst))
    manual_dropped = []
    filtered = []
    for c in qual:
        if c["source"] == "manual":
            overlap = [inst for _, inst in c["pairs"]
                       if (c["timestamp_utc"][:10], c["policy_action"], inst) in scraped_keys]
            if overlap:
                manual_dropped.append({"timestamp_utc": c["timestamp_utc"],
                                       "instruments": overlap,
                                       "reason": "manual_duplicate_of_scraped"})
                continue
        filtered.append(c)
    qual = filtered

    # event ids per statement, then expand to one row per (event x instrument)
    qual.sort(key=lambda c: (c["timestamp_utc"], c["source"], c["source_url"]))
    rows = []
    for i, c in enumerate(qual, start=1):
        eid = f"PA-{i:04d}"
        for sector, inst in c["pairs"]:
            note = (f"entities={'+'.join(c['entities'])}; "
                    f"{c['ts_note']}" if c["ts_note"] else f"entities={'+'.join(c['entities'])}")
            rows.append({
                "event_id": eid,
                "timestamp_utc": c["timestamp_utc"],
                "source": c["source"],
                "source_url": c["source_url"],
                "statement_text": (c["text"][:TEXT_CAP] + " …(truncated; full text in data/raw/)"
                                   if len(c["text"]) > TEXT_CAP else c["text"]),
                "policy_action": c["policy_action"],
                "sector_tagged": sector,
                "instrument_tagged": inst,
                "venue_class": c["venue_class"],
                "qualifies": True,
                "notes": note,
            })

    # T0 mapping + 5-trading-day per-instrument de-dup
    print("  fetching daily OHLCV for the universe (cached)…")
    px = {t: _lib.fetch_daily(t) for t in _lib.UNIVERSE}
    for t, df in px.items():
        print(f"    {t}: {len(df)} rows" + ("  ⚠️ EMPTY" if df.empty else ""))
    idx_by_inst = {t: (df.index if not df.empty else None) for t, df in px.items()}
    for r in rows:
        idx = idx_by_inst.get(r["instrument_tagged"])
        r["t0"] = (_lib.map_t0(r["timestamp_utc"], idx, _lib.ASSET_CLASS[r["instrument_tagged"]])
                   if idx is not None else None)

    kept, dropped = dedup_min_separation(rows, idx_by_inst, min_sep=5)
    for r in kept:
        r["t0"] = str(r["t0"].date()) if r["t0"] is not None else None
    kept.sort(key=lambda r: (r["event_id"], r["instrument_tagged"]))

    _lib.write_jsonl(_lib.DATA_DIR / "trump_events.jsonl", kept)
    _lib.write_jsonl(_lib.DATA_DIR / "dropped_events.jsonl",
                     dropped + [{"event_id": None, **d} for d in manual_dropped])

    n_events = len({r["event_id"] for r in kept})
    n_rows = len(kept)
    per_source = pd.Series([r["source"] for r in kept]).value_counts().to_dict() if kept else {}
    per_inst = pd.Series([r["instrument_tagged"] for r in kept]).value_counts().to_dict() if kept else {}
    print("\n  ── CATALOG SUMMARY ──")
    print(f"  qualifying events: {n_events}   rows (event x instrument): {n_rows}")
    print(f"  dropped (5-td separation): {len(dropped)}   manual duplicates: {len(manual_dropped)}")
    print(f"  by source: {per_source}")
    print(f"  by instrument: {per_inst}")
    if n_events < 30:
        print(f"\n  ⚠️ HONESTY GATE: only {n_events} qualifying events (<30). The definition is")
        print("     NOT loosened; the shortfall will be stated in the Phase-4 report (spec §7).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
