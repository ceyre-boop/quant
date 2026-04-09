"""
Firebase REST Client - Works with Web API Key
No Admin SDK required - uses REST API
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FirebaseRESTClient:
    """
    Firebase client using REST API (works with Web API key).
    No Admin SDK service account required.
    """

    def __init__(self):
        self.api_key = os.getenv("FIREBASE_API_KEY")
        self.database_url = os.getenv("FIREBASE_RTDB_URL")
        self.project_id = os.getenv("FIREBASE_PROJECT_ID")

        if not all([self.api_key, self.database_url]):
            logger.error(
                "Firebase configuration missing. Check FIREBASE_API_KEY and FIREBASE_RTDB_URL"
            )
            self._enabled = False
        else:
            self._enabled = True
            logger.info(f"Firebase REST client initialized for {self.project_id}")

    def _make_request(
        self, method: str, path: str, data: Dict = None
    ) -> Optional[Dict]:
        """Make HTTP request to Firebase RTDB."""
        if not self._enabled:
            return None

        # Construct URL with auth
        url = f"{self.database_url}/{path}.json?auth={self.api_key}"

        try:
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "PUT":
                response = requests.put(url, json=data, timeout=10)
            elif method == "PATCH":
                response = requests.patch(url, json=data, timeout=10)
            else:
                return None

            if response.status_code in [200, 201]:
                return response.json() if response.text else {}
            else:
                logger.error(
                    f"Firebase {method} failed: {response.status_code} - {response.text[:200]}"
                )
                return None

        except Exception as e:
            logger.error(f"Firebase request error: {e}")
            return None

    def read(self, path: str) -> Optional[Dict]:
        """Read data from Firebase."""
        return self._make_request("GET", path)

    def write(self, path: str, data: Dict) -> bool:
        """Write data to Firebase."""
        result = self._make_request("PUT", path, data)
        return result is not None

    def update(self, path: str, data: Dict) -> bool:
        """Update data at path."""
        result = self._make_request("PATCH", path, data)
        return result is not None


# Singleton instance
_firebase_client = None


def get_firebase_client() -> FirebaseRESTClient:
    """Get or create Firebase client instance."""
    global _firebase_client
    if _firebase_client is None:
        _firebase_client = FirebaseRESTClient()
    return _firebase_client
