"""Firebase Client - Wrapper for Firebase Realtime Database and Firestore.

Provides unified interface for both RTDB and Firestore operations.
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class FirebaseClient:
    """Firebase client for Realtime Database and Firestore.
    
    In demo/test mode, operations are logged but not executed.
    In production, uses Firebase Admin SDK.
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        service_account_path: Optional[str] = None,
        rtdb_url: Optional[str] = None,
        demo_mode: bool = True
    ):
        self.project_id = project_id or os.getenv('FIREBASE_PROJECT_ID', 'taboost-platform')
        self.service_account_path = service_account_path or os.getenv(
            'FIREBASE_SERVICE_ACCOUNT_PATH'
        )
        self.rtdb_url = rtdb_url or os.getenv('FIREBASE_RTDB_URL')
        self.demo_mode = demo_mode or (os.getenv('TRADING_MODE', 'paper') == 'paper')
        
        self._rtdb = None
        self._firestore = None
        
        if not self.demo_mode:
            self._initialize_admin_sdk()
    
    def _initialize_admin_sdk(self):
        """Initialize Firebase Admin SDK."""
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore, db
            
            if not firebase_admin._apps:
                if self.service_account_path:
                    cred = credentials.Certificate(self.service_account_path)
                else:
                    # Use application default credentials
                    cred = credentials.ApplicationDefault()
                
                firebase_admin.initialize_app(
                    cred,
                    {
                        'databaseURL': self.rtdb_url,
                        'projectId': self.project_id
                    }
                )
            
            self._firestore = firestore.client()
            self._rtdb = db.reference()
            
            logger.info("Firebase Admin SDK initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
            self.demo_mode = True
    
    def rtdb_update(self, path: str, data: Dict[str, Any]):
        """Update Realtime Database at path.
        
        Args:
            path: RTDB path (e.g., '/live_state/NAS100')
            data: Data to write
        """
        if self.demo_mode:
            logger.debug(f"[DEMO RTDB] UPDATE {path}: {data}")
            return
        
        try:
            if self._rtdb:
                ref = self._rtdb.child(path.lstrip('/'))
                ref.update(data)
                logger.debug(f"RTDB updated: {path}")
        except Exception as e:
            logger.error(f"RTDB update failed for {path}: {e}")
    
    def rtdb_set(self, path: str, data: Dict[str, Any]):
        """Set Realtime Database at path (overwrite).
        
        Args:
            path: RTDB path
            data: Data to write
        """
        if self.demo_mode:
            logger.debug(f"[DEMO RTDB] SET {path}: {data}")
            return
        
        try:
            if self._rtdb:
                ref = self._rtdb.child(path.lstrip('/'))
                ref.set(data)
                logger.debug(f"RTDB set: {path}")
        except Exception as e:
            logger.error(f"RTDB set failed for {path}: {e}")
    
    def rtdb_get(self, path: str) -> Optional[Dict[str, Any]]:
        """Get data from Realtime Database.
        
        Args:
            path: RTDB path
        
        Returns:
            Data at path or None
        """
        if self.demo_mode:
            logger.debug(f"[DEMO RTDB] GET {path}")
            return {}
        
        try:
            if self._rtdb:
                ref = self._rtdb.child(path.lstrip('/'))
                return ref.get()
        except Exception as e:
            logger.error(f"RTDB get failed for {path}: {e}")
        
        return None
    
    def firestore_set(self, collection_doc: str, data: Dict[str, Any]):
        """Set document in Firestore.
        
        Args:
            collection_doc: Path like 'entry_signals/symbol_20240115_143000'
            data: Document data
        """
        if self.demo_mode:
            logger.debug(f"[DEMO FIRESTORE] SET {collection_doc}: {data}")
            return
        
        try:
            if self._firestore:
                parts = collection_doc.split('/')
                if len(parts) >= 2:
                    collection = parts[0]
                    doc_id = '/'.join(parts[1:])
                    self._firestore.collection(collection).document(doc_id).set(data)
                    logger.debug(f"Firestore set: {collection_doc}")
        except Exception as e:
            logger.error(f"Firestore set failed for {collection_doc}: {e}")
    
    def firestore_get(self, collection_doc: str) -> Optional[Dict[str, Any]]:
        """Get document from Firestore.
        
        Args:
            collection_doc: Path like 'entry_signals/symbol_20240115_143000'
        
        Returns:
            Document data or None
        """
        if self.demo_mode:
            logger.debug(f"[DEMO FIRESTORE] GET {collection_doc}")
            return {}
        
        try:
            if self._firestore:
                parts = collection_doc.split('/')
                if len(parts) >= 2:
                    collection = parts[0]
                    doc_id = '/'.join(parts[1:])
                    doc = self._firestore.collection(collection).document(doc_id).get()
                    return doc.to_dict() if doc.exists else None
        except Exception as e:
            logger.error(f"Firestore get failed for {collection_doc}: {e}")
        
        return None
    
    def firestore_query(
        self,
        collection: str,
        field: str,
        operator: str,
        value: Any,
        limit: int = 100
    ) -> list:
        """Query Firestore collection.
        
        Args:
            collection: Collection name
            field: Field to filter on
            operator: Comparison operator ('==', '>', '<', etc.)
            value: Value to compare
            limit: Max results
        
        Returns:
            List of documents
        """
        if self.demo_mode:
            logger.debug(f"[DEMO FIRESTORE] QUERY {collection} WHERE {field} {operator} {value}")
            return []
        
        try:
            if self._firestore:
                query = self._firestore.collection(collection).where(field, operator, value)
                query = query.limit(limit)
                docs = query.stream()
                return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Firestore query failed for {collection}: {e}")
        
        return []
    
    def is_connected(self) -> bool:
        """Check if Firebase is connected."""
        if self.demo_mode:
            return True
        return self._rtdb is not None or self._firestore is not None
