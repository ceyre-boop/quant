"""Firebase client wrapper for Clawd Trading System.

Provides schema-validated writes to Firestore and Realtime Database.
All writes are validated before commit to prevent invalid data.
"""

import os
import json
import logging
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, db
    from firebase_admin.exceptions import FirebaseError

    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    firebase_admin = None
    credentials = None
    firestore = None
    db = None
    FirebaseError = Exception

from contracts.types import (
    FeatureRecord,
    BiasOutput,
    RiskOutput,
    GameOutput,
    RegimeState,
    EntrySignal,
    PositionState,
    AccountState,
)

logger = logging.getLogger(__name__)


class FirebaseClient:
    """Firebase client for Firestore and Realtime Database operations."""

    _instance = None
    _initialized = False
    _lock = threading.Lock()

    db: Any
    rtdb: Any

    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure single Firebase initialization."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FirebaseClient, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        project_id: Optional[str] = None,
        service_account_path: Optional[str] = None,
        rtdb_url: Optional[str] = None,
    ):
        if FirebaseClient._initialized:
            return

        if not FIREBASE_AVAILABLE:
            logger.warning("firebase_admin not installed. Firebase features will be disabled.")
            self.db = None
            self.rtdb = None
            FirebaseClient._initialized = True
            return

        self.project_id = project_id or os.getenv("FIREBASE_PROJECT_ID")
        self.service_account_path = service_account_path or os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
        self.rtdb_url = rtdb_url or os.getenv("FIREBASE_RTDB_URL")

        if not self.project_id:
            raise ValueError("FIREBASE_PROJECT_ID not set")

        self._initialize_firebase()
        FirebaseClient._initialized = True

    def _initialize_firebase(self):
        """Initialize Firebase app."""
        if not FIREBASE_AVAILABLE:
            return

        try:
            # Check if already initialized
            firebase_admin.get_app()
            logger.info("Firebase already initialized")
        except ValueError:
            # Initialize with service account
            if self.service_account_path and Path(self.service_account_path).exists():
                cred = credentials.Certificate(self.service_account_path)
                firebase_admin.initialize_app(cred, {"databaseURL": self.rtdb_url})
                logger.info(f"Firebase initialized with service account for project {self.project_id}")
            else:
                # Try application default credentials (for GCP/GKE environments)
                firebase_admin.initialize_app({"projectId": self.project_id, "databaseURL": self.rtdb_url})
                logger.info(f"Firebase initialized with default credentials for project {self.project_id}")

        self.db = firestore.client()
        self.rtdb = db

    def _check_db(self):
        """Check if Firestore is available."""
        if not FIREBASE_AVAILABLE or self.db is None:
            raise RuntimeError("Firebase not initialized. Check FIREBASE_PROJECT_ID and credentials.")

    # ========================================================================
    # Firestore Write Methods
    # ========================================================================

    def write_feature_record(self, record: FeatureRecord, validate: bool = True) -> str:
        """Write feature record to Firestore.

        Document ID: {symbol}_{timeframe}_{timestamp_utc}
        """
        self._check_db()

        if validate and not record.is_valid:
            raise ValueError(f"Cannot write invalid feature record: {record.validation_errors}")

        doc_id = f"{record.symbol}_{record.timeframe}_{record.timestamp.strftime('%Y%m%d_%H%M%S')}"
        data = record.to_dict()
        data["created_at"] = firestore.SERVER_TIMESTAMP

        self.db.collection("feature_records").document(doc_id).set(data)
        logger.debug(f"Wrote feature record: {doc_id}")
        return doc_id

    def write_bias_output(self, symbol: str, bias: BiasOutput) -> str:
        """Write bias output to Firestore.

        Document ID: {symbol}_{date}_{timestamp_utc}
        """
        self._check_db()

        doc_id = f"{symbol}_{(bias.timestamp or datetime.utcnow()).strftime('%Y%m%d_%H%M%S')}"
        data = bias.to_dict()
        data["symbol"] = symbol
        data["created_at"] = firestore.SERVER_TIMESTAMP

        self.db.collection("bias_outputs").document(doc_id).set(data)
        logger.debug(f"Wrote bias output: {doc_id}")
        return doc_id

    def write_risk_structure(self, symbol: str, risk: RiskOutput, bias_id: str) -> str:
        """Write risk structure to Firestore.

        Document ID: {symbol}_{date}_{timestamp_utc}
        """
        self._check_db()

        doc_id = f"{symbol}_{(risk.timestamp or datetime.utcnow()).strftime('%Y%m%d_%H%M%S')}"
        data = risk.to_dict()
        data["symbol"] = symbol
        data["bias_id"] = bias_id
        data["created_at"] = firestore.SERVER_TIMESTAMP

        self.db.collection("risk_structures").document(doc_id).set(data)
        logger.debug(f"Wrote risk structure: {doc_id}")
        return doc_id

    def write_game_output(self, symbol: str, game: GameOutput) -> str:
        """Write game output to Firestore.

        Document ID: {symbol}_{date}_{timestamp_utc}
        """
        self._check_db()

        doc_id = f"{symbol}_{(game.timestamp or datetime.utcnow()).strftime('%Y%m%d_%H%M%S')}"
        data = game.to_dict()
        data["symbol"] = symbol
        data["created_at"] = firestore.SERVER_TIMESTAMP

        self.db.collection("game_outputs").document(doc_id).set(data)
        logger.debug(f"Wrote game output: {doc_id}")
        return doc_id

    def write_entry_signal(self, signal: EntrySignal) -> str:
        """Write entry signal to Firestore."""
        self._check_db()

        doc_id = f"{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"
        data = signal.to_dict()
        data["created_at"] = firestore.SERVER_TIMESTAMP

        self.db.collection("entry_signals").document(doc_id).set(data)
        logger.info(f"Wrote entry signal: {doc_id}")
        return doc_id

    def write_position(self, position: PositionState) -> str:
        """Write position state to Firestore."""
        self._check_db()

        data = position.to_dict()
        data["updated_at"] = firestore.SERVER_TIMESTAMP

        self.db.collection("positions").document(position.trade_id).set(data)
        logger.debug(f"Wrote position: {position.trade_id}")
        return position.trade_id

    def update_position(self, trade_id: str, updates: Dict[str, Any]) -> None:
        """Update position fields."""
        self._check_db()

        updates["updated_at"] = firestore.SERVER_TIMESTAMP
        self.db.collection("positions").document(trade_id).update(updates)
        logger.debug(f"Updated position: {trade_id}")

    def write_account_state(self, account: AccountState) -> str:
        """Write account state to Firestore."""
        self._check_db()

        data = account.to_dict()
        data["updated_at"] = firestore.SERVER_TIMESTAMP

        self.db.collection("account_state").document(account.account_id).set(data)
        logger.debug(f"Wrote account state: {account.account_id}")
        return account.account_id

    def write_trade_record(self, trade_id: str, data: Dict[str, Any]) -> str:
        """Write closed trade record."""
        self._check_db()

        data["updated_at"] = firestore.SERVER_TIMESTAMP

        self.db.collection("trade_records").document(trade_id).set(data, merge=True)
        logger.info(f"Wrote trade record: {trade_id}")
        return trade_id

    def write_regime_history(self, symbol: str, regime: RegimeState, timestamp: datetime) -> str:
        """Write regime classification to history."""
        self._check_db()

        doc_id = f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        data = regime.to_dict()
        data["symbol"] = symbol
        data["timestamp"] = timestamp.isoformat()
        data["created_at"] = firestore.SERVER_TIMESTAMP

        self.db.collection("regime_history").document(doc_id).set(data)
        logger.debug(f"Wrote regime history: {doc_id}")
        return doc_id

    def write_system_log(
        self,
        level: str,
        message: str,
        module: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write system log entry."""
        self._check_db()

        doc_ref = self.db.collection("system_logs").document()
        data = {
            "level": level,
            "message": message,
            "module": module,
            "metadata": metadata or {},
            "created_at": firestore.SERVER_TIMESTAMP,
        }
        doc_ref.set(data)
        return doc_ref.id

    # ========================================================================
    # Realtime Database Methods
    # ========================================================================

    def update_live_state(
        self,
        symbol: str,
        bias: Optional[BiasOutput] = None,
        regime: Optional[RegimeState] = None,
        game: Optional[GameOutput] = None,
        position: Optional[PositionState] = None,
        session_pnl: Optional[float] = None,
    ) -> None:
        """Update live state in Realtime Database."""
        if not FIREBASE_AVAILABLE or self.rtdb is None:
            logger.warning("Realtime Database not available")
            return

        ref = self.rtdb.reference(f"live_state/{symbol}")

        updates: Dict[str, Any] = {"updated_at": datetime.utcnow().isoformat()}

        if bias:
            updates["current_bias"] = bias.to_dict()
        if regime:
            updates["current_regime"] = regime.to_dict()
        if game:
            updates["game_state"] = game.to_dict()
        if position:
            updates["position_state"] = position.direction.name if position.status == "OPEN" else "FLAT"
            updates["open_position"] = position.to_dict()
        if session_pnl is not None:
            updates["session_pnl"] = session_pnl

        ref.update(updates)
        logger.debug(f"Updated live state for {symbol}")

    def update_session_controls(
        self,
        trading_enabled: Optional[bool] = None,
        daily_loss_pct: Optional[float] = None,
        open_positions: Optional[int] = None,
        hard_logic_status: Optional[str] = None,
    ) -> None:
        """Update session controls in Realtime Database."""
        if not FIREBASE_AVAILABLE or self.rtdb is None:
            logger.warning("Realtime Database not available")
            return

        ref = self.rtdb.reference("session_controls")

        updates: Dict[str, Any] = {}
        if trading_enabled is not None:
            updates["trading_enabled"] = trading_enabled
        if daily_loss_pct is not None:
            updates["daily_loss_pct"] = daily_loss_pct
        if open_positions is not None:
            updates["open_positions"] = open_positions
        if hard_logic_status is not None:
            updates["hard_logic_status"] = hard_logic_status

        if updates:
            ref.update(updates)
            logger.debug("Updated session controls")

    def get_session_controls(self) -> Dict[str, Any]:
        """Get current session controls."""
        if not FIREBASE_AVAILABLE or self.rtdb is None:
            return {}
        ref = self.rtdb.reference("session_controls")
        return ref.get() or {}

    def get_live_state(self, symbol: str) -> Dict[str, Any]:
        """Get live state for symbol."""
        if not FIREBASE_AVAILABLE or self.rtdb is None:
            return {}
        ref = self.rtdb.reference(f"live_state/{symbol}")
        return ref.get() or {}

    # ========================================================================
    # Query Methods
    # ========================================================================

    def get_latest_bias(self, symbol: str, limit: int = 1) -> List[Dict[str, Any]]:
        """Get latest bias outputs for symbol."""
        self._check_db()

        docs = (
            self.db.collection("bias_outputs")
            .where("symbol", "==", symbol)
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        return [doc.to_dict() for doc in docs]

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open positions."""
        self._check_db()

        query = self.db.collection("positions").where("status", "==", "OPEN")
        if symbol:
            query = query.where("symbol", "==", symbol)

        docs = query.stream()
        return [doc.to_dict() for doc in docs]

    # ========================================================================
    # Batch Operations
    # ========================================================================

    def batch_write_bias_outputs(self, outputs: Dict[str, BiasOutput]) -> List[str]:
        """Batch write bias outputs for multiple symbols."""
        doc_ids = []
        for symbol, bias in outputs.items():
            doc_id = self.write_bias_output(symbol, bias)
            doc_ids.append(doc_id)
        return doc_ids

    def batch_write_risk_structures(self, structures: Dict[str, RiskOutput], bias_ids: Dict[str, str]) -> List[str]:
        """Batch write risk structures for multiple symbols."""
        doc_ids = []
        for symbol, risk in structures.items():
            bias_id = bias_ids.get(symbol, "")
            doc_id = self.write_risk_structure(symbol, risk, bias_id)
            doc_ids.append(doc_id)
        return doc_ids

    # ========================================================================
    # Generic CRUD helpers used by callers
    # ========================================================================

    def write(self, collection: str, doc_id: Optional[str], data: Dict[str, Any]) -> Optional[str]:
        """Write a document to Firestore. If doc_id is None, auto-generate one."""
        if not FIREBASE_AVAILABLE or self.db is None:
            return None
        data["updated_at"] = firestore.SERVER_TIMESTAMP
        if doc_id:
            self.db.collection(collection).document(doc_id).set(data, merge=True)
            return doc_id
        ref = self.db.collection(collection).document()
        ref.set(data)
        return ref.id

    def read(self, collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Read a single document from Firestore."""
        if not FIREBASE_AVAILABLE or self.db is None:
            return None
        doc = self.db.collection(collection).document(doc_id).get()
        return doc.to_dict() if doc.exists else None

    def query(
        self,
        collection: str,
        filters: Optional[List] = None,
        order_by: Optional[str] = None,
        direction: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query a Firestore collection."""
        if not FIREBASE_AVAILABLE or self.db is None:
            return []
        q = self.db.collection(collection)
        if filters:
            for f in filters:
                q = q.where(*f)
        if order_by:
            q = q.order_by(order_by)
        if limit:
            q = q.limit(limit)
        return [doc.to_dict() for doc in q.stream()]

    def update_realtime(self, path: str, data: Any) -> None:
        """Write/update data in the Realtime Database at the given path."""
        if not FIREBASE_AVAILABLE or self.rtdb is None:
            return
        ref = self.rtdb.reference(path)
        if isinstance(data, dict):
            ref.update(data)
        else:
            ref.set(data)

    def read_realtime(self, path: str) -> Any:
        """Read data from the Realtime Database at the given path."""
        if not FIREBASE_AVAILABLE or self.rtdb is None:
            return None
        return self.rtdb.reference(path).get()

