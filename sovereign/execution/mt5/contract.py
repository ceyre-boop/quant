"""order_intent contract — parse + validate (spec §5).

The bridge's ONLY input. A producer writes one JSON file per intent to
data/execution/mt5_intents/<intent_id>.json; the bridge consumes it. Decoupling by
data contract (not import) is what keeps the bridge isolation-safe: the contract
references no execution-path symbols.

The bridge validates but does NOT compute edge — the intent arrives pre-sized.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

VALID_SIDES = {"BUY", "SELL"}
VALID_ORDER_TYPES = {"MARKET"}  # PENDING deferred (spec §5)
MAX_COMMENT_LEN = 31  # MT5 limit


class IntentError(ValueError):
    """Raised when an order_intent is malformed. Message names the failing field."""


@dataclass(frozen=True)
class OrderIntent:
    intent_id: str
    created_at: str
    symbol: str
    side: str            # BUY | SELL
    order_type: str      # MARKET
    volume_lots: float
    sl_price: float
    tp_price: Optional[float]
    comment: str
    magic: int
    source_strategy: str
    signal_hash: str
    max_slippage_points: int

    # --- construction -----------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OrderIntent":
        """Parse a raw dict into an OrderIntent, applying structural validation.

        Raises IntentError naming the first failing field. Value/range checks that
        need config (lot bounds) live in validate(); everything self-contained is
        checked here so a malformed file is rejected before any terminal contact.
        """
        required = [
            "intent_id", "created_at", "symbol", "side", "order_type",
            "volume_lots", "sl_price", "comment", "magic",
            "source_strategy", "signal_hash", "max_slippage_points",
        ]
        for field in required:
            if field not in d:
                raise IntentError(f"missing required field: '{field}'")
            if d[field] is None:
                raise IntentError(f"required field is null: '{field}'")

        side = str(d["side"]).upper()
        if side not in VALID_SIDES:
            raise IntentError(f"side must be one of {sorted(VALID_SIDES)}, got '{d['side']}'")

        order_type = str(d["order_type"]).upper()
        if order_type not in VALID_ORDER_TYPES:
            raise IntentError(
                f"order_type must be one of {sorted(VALID_ORDER_TYPES)} in v1, got '{d['order_type']}'"
            )

        try:
            volume_lots = float(d["volume_lots"])
        except (TypeError, ValueError):
            raise IntentError(f"volume_lots must be numeric, got '{d['volume_lots']}'")
        if volume_lots <= 0:
            raise IntentError(f"volume_lots must be > 0, got {volume_lots}")

        try:
            sl_price = float(d["sl_price"])
        except (TypeError, ValueError):
            raise IntentError(f"sl_price must be numeric, got '{d['sl_price']}'")
        if sl_price <= 0:
            raise IntentError(f"sl_price must be > 0, got {sl_price}")

        tp_raw = d.get("tp_price")
        tp_price: Optional[float]
        if tp_raw is None:
            tp_price = None
        else:
            try:
                tp_price = float(tp_raw)
            except (TypeError, ValueError):
                raise IntentError(f"tp_price must be numeric or null, got '{tp_raw}'")
            if tp_price <= 0:
                raise IntentError(f"tp_price must be > 0 when present, got {tp_price}")

        comment = str(d["comment"])
        if len(comment) > MAX_COMMENT_LEN:
            raise IntentError(
                f"comment must be <= {MAX_COMMENT_LEN} chars (MT5 limit), got {len(comment)}"
            )

        if isinstance(d["magic"], bool) or not isinstance(d["magic"], int):
            raise IntentError(f"magic must be an int, got '{d['magic']}'")

        try:
            max_slip = int(d["max_slippage_points"])
        except (TypeError, ValueError):
            raise IntentError(
                f"max_slippage_points must be an int, got '{d['max_slippage_points']}'"
            )
        if max_slip < 0:
            raise IntentError(f"max_slippage_points must be >= 0, got {max_slip}")

        intent_id = str(d["intent_id"]).strip()
        if not intent_id:
            raise IntentError("intent_id must be a non-empty string")

        symbol = str(d["symbol"]).strip()
        if not symbol:
            raise IntentError("symbol must be a non-empty string")

        return cls(
            intent_id=intent_id,
            created_at=str(d["created_at"]),
            symbol=symbol,
            side=side,
            order_type=order_type,
            volume_lots=volume_lots,
            sl_price=sl_price,
            tp_price=tp_price,
            comment=comment,
            magic=int(d["magic"]),
            source_strategy=str(d["source_strategy"]),
            signal_hash=str(d["signal_hash"]),
            max_slippage_points=max_slip,
        )

    @classmethod
    def load(cls, path: Path | str) -> "OrderIntent":
        p = Path(path)
        if not p.exists():
            raise IntentError(f"intent file not found: {p}")
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise IntentError(f"intent file is not valid JSON ({p}): {e}")
        if not isinstance(raw, dict):
            raise IntentError(f"intent file must be a JSON object, got {type(raw).__name__}")
        return cls.from_dict(raw)

    # --- config-dependent validation --------------------------------------

    def validate(self, *, min_lot: float, max_lot: float, allowed_symbols: Optional[set[str]] = None) -> None:
        """Range/consistency checks that need config. Raises IntentError on failure.

        Structural checks already ran in from_dict(); this layer enforces broker/config
        bounds and the SL-side rule (spec §5).
        """
        if self.volume_lots < min_lot or self.volume_lots > max_lot:
            raise IntentError(
                f"volume_lots {self.volume_lots} outside configured bounds [{min_lot}, {max_lot}]"
            )

        if allowed_symbols is not None and self.symbol not in allowed_symbols:
            raise IntentError(
                f"symbol '{self.symbol}' not in configured symbol map {sorted(allowed_symbols)}"
            )

        # SL must be on the protective side of the trade (spec §5):
        # SELL → SL above the market entry; BUY → SL below. We enforce it against the
        # TP as the directional reference when TP is present, else this is a soft check
        # that the SL/TP ordering is internally consistent for the side.
        if self.tp_price is not None:
            if self.side == "SELL" and not (self.sl_price > self.tp_price):
                raise IntentError(
                    f"SELL intent requires sl_price ({self.sl_price}) above tp_price "
                    f"({self.tp_price}) — SL must sit above a short's target"
                )
            if self.side == "BUY" and not (self.sl_price < self.tp_price):
                raise IntentError(
                    f"BUY intent requires sl_price ({self.sl_price}) below tp_price "
                    f"({self.tp_price}) — SL must sit below a long's target"
                )

    def order_card(self, *, login: Any, server: str, trade_mode_name: str) -> str:
        """Human-readable order card for the approval step (spec §6.1)."""
        tp = "none" if self.tp_price is None else f"{self.tp_price}"
        return (
            "\n".join([
                "┌─ MT5 ORDER CARD ─────────────────────────────────",
                f"│ intent_id : {self.intent_id}",
                f"│ symbol    : {self.symbol}",
                f"│ side      : {self.side}  ({self.order_type})",
                f"│ volume    : {self.volume_lots} lots",
                f"│ SL / TP   : {self.sl_price} / {tp}",
                f"│ max slip  : {self.max_slippage_points} points",
                f"│ comment   : {self.comment}  (magic {self.magic})",
                f"│ strategy  : {self.source_strategy}",
                f"│ ACCOUNT   : login={login} server={server}  [{trade_mode_name}]",
                "└──────────────────────────────────────────────────",
            ])
        )
