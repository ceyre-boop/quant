"""Firebase Client - Wrapper for Firebase Realtime Database and Firestore.

Provides unified interface for both RTDB and Firestore operations.
Uses Firebase Admin SDK for production operations.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class FirebaseClient:
    """Firebase client for Realtime Database and Firestore.
    
    Uses Firebase Admin SDK for production operations.
    Falls back to demo mode only if initialization fails.
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        service_account_path: Optional[str] = None,
        rtdb_url: Optional[str] = None,
        demo_mode: bool = False
    ):
        self.project_id = project_id or os.getenv('FIREBASE_PROJECT_ID', 'clawd-trading-7b8de')
        self.service_account_path = service_account_path or os.getenv(
            'FIREBASE_SERVICE_ACCOUNT_PATH'
        )
        self.rtdb_url = rtdb_url or os.getenv(
            'FIREBASE_RTDB_URL', 
            'https://clawd-trading-7b8de-default-rtdb.firebaseio.com'
        )
        
        self._rtdb = None
        self._firestore = None
        self._demo_mode = False
        
        # Try to initialize Admin SDK
        try:
            self._initialize_admin_sdk()
        except Exception as e:
            logger.error(f"Firebase Admin SDK initialization failed: {e}")
            if demo_mode:
                logger.warning("Falling back to demo mode")
                self._demo_mode = True
            else:
                raise
    
    def _initialize_admin_sdk(self):
        """Initialize Firebase Admin SDK."""
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore, db
            
            if not firebase_admin._apps:
                cred = None
                
                # Try service account JSON from environment variable first
                service_account_json = os.getenv('FIREBASE_SERVICE_ACCOUNT')
                if service_account_json:
                    try:
                        cred_dict = json.loads(service_account_json)
                        cred = credentials.Certificate(cred_dict)
                        logger.info("Using Firebase credentials from FIREBASE_SERVICE_ACCOUNT env var")
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid FIREBASE_SERVICE_ACCOUNT JSON: {e}")
                
                # Try service account file path
                if not cred and self.service_account_path and os.path.exists(self.service_account_path):
                    cred = credentials.Certificate(self.service_account_path)
                    logger.info(f"Using Firebase credentials from file: {self.service_account_path}")
                
                # Try application default credentials (Cloud Run, App Engine, etc.)
                if not cred:
                    try:
                        cred = credentials.ApplicationDefault()
                        logger.info("Using Firebase Application Default Credentials")
                    except Exception:
                        logger.warning("No ADC available")
                
                if not cred:
                    raise ValueError(
                        "No Firebase credentials found. Set FIREBASE_SERVICE_ACCOUNT "
                        "environment variable with service account JSON."
                    )
                
                # Initialize the app
                firebase_admin.initialize_app(
                    cred,
                    {
                        'databaseURL': self.rtdb_url,
                        'projectId': self.project_id
                    }
                )
            
            self._firestore = firestore.client()
            self._rtdb = db.reference()
            
            logger.info("Firebase Admin SDK initialized successfully")
            logger.info(f"Project ID: {self.project_id}")
            logger.info(f"RTDB URL: {self.rtdb_url}")
            
        except ImportError:
            logger.error("firebase_admin not installed. Run: pip install firebase-admin")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
            raise
    
    def rtdb_update(self, path: str, data: Dict[str, Any]):
        """Update Realtime Database at path.
        
        Args:
            path: RTDB path (e.g., '/live_state/NAS100')
            data: Data to write
        """        
        try:
            if self._rtdb:
                ref = self._rtdb.child(path.lstrip('/'))
                ref.update(data)
                logger.debug(f"RTDB updated: {path}")
            elif self._demo_mode:
                logger.debug(f"[DEMO RTDB] UPDATE {path}")
        except Exception as e:
            logger.error(f"RTDB update failed for {path}: {e}")
    
    def rtdb_set(self, path: str, data: Dict[str, Any]):
        """Set Realtime Database at path (overwrite).
        
        Args:
            path: RTDB path
            data: Data to write
        """
        try:
            if self._rtdb:
                ref = self._rtdb.child(path.lstrip('/'))
                ref.set(data)
                logger.debug(f"RTDB set: {path}")
            elif self._demo_mode:
                logger.debug(f"[DEMO RTDB] SET {path}")
        except Exception as e:
            logger.error(f"RTDB set failed for {path}: {e}")
    
    def rtdb_get(self, path: str) -> Optional[Dict[str, Any]]:
        """Get data from Realtime Database.
        
        Args:
            path: RTDB path
        
        Returns:
            Data at path or None
        """
        try:
            if self._rtdb:
                ref = self._rtdb.child(path.lstrip('/'))
                return ref.get()
        except Exception as e:
            logger.error(f"RTDB get failed for {path}: {e}")
        
        return None
    
    def rtdb_push(self, path: str, data: Dict[str, Any]) -> Optional[str]:
        """Push data to a list in Realtime Database.
        
        Args:
            path: RTDB path
            data: Data to push
        
        Returns:
            Key of pushed data or None
        """
        try:
            if self._rtdb:
                ref = self._rtdb.child(path.lstrip('/'))
                new_ref = ref.push(data)
                logger.debug(f"RTDB pushed: {path}")
                return new_ref.key
            elif self._demo_mode:
                logger.debug(f"[DEMO RTDB] PUSH {path}")
        except Exception as e:
            logger.error(f"RTDB push failed for {path}: {e}")
        
        return None
    
    def rtdb_delete(self, path: str):
        """Delete data from Realtime Database.
        
        Args:
            path: RTDB path to delete
        """
        try:
            if self._rtdb:
                ref = self._rtdb.child(path.lstrip('/'))
                ref.delete()
                logger.debug(f"RTDB deleted: {path}")
            elif self._demo_mode:
                logger.debug(f"[DEMO RTDB] DELETE {path}")
        except Exception as e:
            logger.error(f"RTDB delete failed for {path}: {e}")
    
    def firestore_set(self, collection_doc: str, data: Dict[str, Any]):
        """Set document in Firestore.
        
        Args:
            collection_doc: Path like 'entry_signals/symbol_20240115_143000'
            data: Document data
        """
        try:
            if self._firestore:
                parts = collection_doc.split('/')
                if len(parts) >= 2:
                    collection = parts[0]
                    doc_id = '/'.join(parts[1:])
                    self._firestore.collection(collection).document(doc_id).set(data)
                    logger.debug(f"Firestore set: {collection_doc}")
            elif self._demo_mode:
                logger.debug(f"[DEMO FIRESTORE] SET {collection_doc}")
        except Exception as e:
            logger.error(f"Firestore set failed for {collection_doc}: {e}")
    
    def firestore_get(self, collection_doc: str) -> Optional[Dict[str, Any]]:
        """Get document from Firestore.
        
        Args:
            collection_doc: Path like 'entry_signals/symbol_20240115_143000'
        
        Returns:
            Document data or None
        """
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
        try:
            if self._firestore:
                query = self._firestore.collection(collection).where(field, operator, value)
                query = query.limit(limit)
                docs = query.stream()
                return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Firestore query failed for {collection}: {e}")
        
        return []
    
    def firestore_update(self, collection_doc: str, data: Dict[str, Any]):
        """Update fields in a Firestore document.
        
        Args:
            collection_doc: Path like 'entry_signals/symbol_20240115_143000'
            data: Fields to update
        """
        try:
            if self._firestore:
                parts = collection_doc.split('/')
                if len(parts) >= 2:
                    collection = parts[0]
                    doc_id = '/'.join(parts[1:])
                    self._firestore.collection(collection).document(doc_id).update(data)
                    logger.debug(f"Firestore updated: {collection_doc}")
            elif self._demo_mode:
                logger.debug(f"[DEMO FIRESTORE] UPDATE {collection_doc}")
        except Exception as e:
            logger.error(f"Firestore update failed for {collection_doc}: {e}")
    
    def is_connected(self) -> bool:
        """Check if Firebase is connected."""
        return self._rtdb is not None or self._firestore is not None
    
    def get_server_timestamp(self) -> Dict[str, Any]:
        """Get server timestamp for Firebase operations."""
        from firebase_admin import db
        return db.ServerValue.TIMESTAMP
