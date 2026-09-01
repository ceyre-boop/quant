"""What facts exist, and what each one's two timestamps are.

Every fact declares:

  valid_col      WHEN IT HAPPENED. The event or period the row describes —
                 a fiscal quarter end, a transaction date, a settlement date.
  published_col  WHEN IT BECAME KNOWABLE. The filing or publication instant.
                 NEVER the event date. A Form 4 transaction on the 3rd filed on
                 the 5th was not knowable until the 5th.

A fact with `published_col=None` is NOT point-in-time readable. Reading it
as-of raises NotPointInTime rather than returning rows. That is deliberate and
is the load-bearing decision in this module: the alternative — returning rows
whose knowability is unknown — is the exact silent leak the layer exists to
prevent. Such facts appear here explicitly, with the reason and the fix, so the
gap is a visible TODO rather than an invisible assumption.

Declaring facts here (rather than accepting arbitrary table names) is what stops
an ad-hoc query slipping past the temporal contract, the same way
backtester/holdout_guard.py refuses an unregistered dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sovereign.pit.errors import UnknownFact


@dataclass(frozen=True)
class FactSpec:
    name: str
    table: str
    entity_col: str
    valid_col: str
    #: None means this fact cannot be read as-of. `blocked_reason` says why.
    published_col: str | None
    #: Columns identifying one logical observation, EXCLUDING the temporal ones.
    #: Two rows sharing these but differing in published_ts are two VINTAGES of
    #: the same fact — a restatement — and both must survive.
    identity: tuple[str, ...]
    blocked_reason: str | None = None
    notes: str = ""

    @property
    def is_point_in_time(self) -> bool:
        return self.published_col is not None


#: The temporal contract for the fundamentals layer.
#:
#: Audited 2026-09-01 against sovereign/fundamentals/store.py. Four facts are
#: blocked because their source genuinely gives us no publication instant; each
#: names the concrete fix rather than being quietly excluded.
FACTS: dict[str, FactSpec] = {
    "earnings": FactSpec(
        name="earnings",
        table="fund_earnings_event",
        entity_col="ticker",
        valid_col="fiscal_end",
        published_col="published_ts",
        identity=("ticker", "fiscal_end"),
        notes=(
            "Restatements and the pre-print -> post-print transition are separate "
            "vintages of the same (ticker, fiscal_end). Both must survive: the "
            "pre-announcement consensus view of a quarter is a real historical fact."
        ),
    ),
    "insider": FactSpec(
        name="insider",
        table="fund_insider_txn",
        entity_col="ticker",
        valid_col="txn_date",
        # published_ts, NOT filing_date. filing_date is a DATE, so using it as
        # knowable_at claimed a Form 4 was public at midnight when EDGAR
        # accepted it at ~18:30 ET — ~22 hours of overstated knowledge, and a
        # measured leak on same-day reads. published_ts is the acceptance instant.
        published_col="published_ts",
        identity=("accession", "line_no"),
        notes="The one fact that was already as-of-ready before this layer existed.",
    ),
    "institutions": FactSpec(
        name="institutions",
        table="fund_institution_holding",
        entity_col="ticker",
        valid_col="period_end",
        # Same reason as insider: an instant, not a filing DAY.
        published_col="published_ts",
        identity=("period_end", "filer_cik", "cusip"),
        notes=(
            "13F is filed ~45 days after quarter end, so reading on period_end "
            "leaks six weeks. A 13F-A amendment is a NEW VINTAGE of the same "
            "(period_end, filer_cik, cusip), not a replacement — the original "
            "filing was knowable earlier and that must stay true."
        ),
    ),
    "short_interest": FactSpec(
        name="short_interest",
        table="fund_short_interest",
        entity_col="ticker",
        valid_col="settlement_date",
        published_col="published_ts",
        identity=("settlement_date", "ticker", "source"),
        notes=(
            "Published ~8 days after settlement. The column exists but the Nasdaq "
            "transport currently writes NULL, so rows are stored and correctly "
            "refused by as-of reads until the publication date is backfilled."
        ),
    ),
    "estimates": FactSpec(
        name="estimates",
        table="fund_estimate_snapshot",
        entity_col="ticker",
        valid_col="period_end",
        # snapshot_date, NOT a published_ts column — the table has none. The date
        # we took the snapshot is the instant that view of consensus became
        # knowable TO US. It is conservative (the street held that view slightly
        # earlier), and conservative is the only safe direction: we under-claim
        # knowledge rather than over-claim it.
        published_col="snapshot_date",
        identity=("ticker", "period"),
        notes=(
            "Already vintage-keyed on snapshot_date before this layer, and the "
            "one table that got point-in-time right by accident. This is the "
            "revision path no free source will sell us. Identity excludes "
            "snapshot_date on purpose: successive snapshots of the same "
            "(ticker, period) ARE the vintages."
        ),
    ),

    # ── Spec tables (Phase 0.5): these use event_date / knowable_at directly ──
    "filings": FactSpec(
        name="filings",
        table="filings",
        entity_col="ticker",
        valid_col="event_date",
        published_col="knowable_at",
        identity=("accession_no",),
        notes=(
            "knowable_at is the EDGAR ACCEPTANCE INSTANT. An 8-K accepted at "
            "16:30 ET was not knowable that morning; using the filing DAY "
            "instead claims a full extra trading day of knowledge and inverts "
            "the entry from tomorrow's open to today's close."
        ),
    ),
    "prices": FactSpec(
        name="prices",
        table="prices",
        entity_col="ticker",
        valid_col="event_date",
        published_col="knowable_at",
        identity=("ticker", "event_date"),
        notes=(
            "knowable_at is the SESSION CLOSE, not ingest time. This is what "
            "makes price_reaction computable inside an as-of view rather than "
            "stored and re-read — see that fact's blocked_reason."
        ),
    ),

    # ── Blocked: no publication instant exists ──────────────────────────────
    "short_volume": FactSpec(
        name="short_volume",
        table="fund_short_volume_daily",
        entity_col="ticker",
        valid_col="date",
        published_col=None,
        identity=("date", "ticker"),
        blocked_reason=(
            "FINRA posts the daily file AFTER the close of the day it describes, "
            "so `date` is the event day, not the publication instant, and the "
            "table has neither a published_ts nor a fetched_at. Filtering on "
            "date <= as_of would return rows published that evening. "
            "FIX: add published_ts to the schema and set it from the FINRA file's "
            "posting time in transports/finra.py."
        ),
    ),
    "borrow": FactSpec(
        name="borrow",
        table="fund_borrow",
        entity_col="ticker",
        valid_col="date",
        published_col=None,
        identity=("date", "ticker"),
        blocked_reason=(
            "IB locate snapshots are an intraday observation stamped with a bare "
            "date and no fetched_at. FIX: record the snapshot instant in "
            "scripts/ib_shortable_snapshot.py and add published_ts."
        ),
    ),
    "institutions_agg": FactSpec(
        name="institutions_agg",
        table="fund_institution_agg",
        entity_col="ticker",
        valid_col="period_end",
        published_col=None,
        identity=("period_end", "ticker"),
        blocked_reason=(
            "Aggregated from holdings with the filing_date discarded, so it leaks "
            "~45 days BY CONSTRUCTION and cannot be fixed by a filter. "
            "FIX: carry max(filing_date) of the contributing holdings as the "
            "aggregate's publication instant — the provenance of an aggregate is "
            "its latest contributor, per research/petrules/free_features.py."
        ),
    ),
    "price_reaction": FactSpec(
        name="price_reaction",
        table="fund_price_reaction",
        entity_col="ticker",
        valid_col="report_date",
        published_col=None,
        identity=("ticker", "report_date"),
        blocked_reason=(
            "Derived, not sourced: it has no publication instant of its own. A "
            "derived value is knowable only when its LATEST INPUT was knowable, "
            "so it must not be read as-of directly. "
            "FIX: compute it inside an as-of view from inputs that are themselves "
            "as-of filtered, rather than storing and re-reading it."
        ),
    ),
}


def get(name: str) -> FactSpec:
    try:
        return FACTS[name]
    except KeyError:
        raise UnknownFact(
            f"{name!r} is not a registered fact. Known: {sorted(FACTS)}. "
            "Facts must be declared in sovereign/pit/spec.py so that every read "
            "has a stated temporal contract."
        ) from None


def point_in_time_facts() -> list[str]:
    return sorted(n for n, f in FACTS.items() if f.is_point_in_time)


def blocked_facts() -> list[str]:
    return sorted(n for n, f in FACTS.items() if not f.is_point_in_time)
