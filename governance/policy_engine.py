"""
Institutional Governance & Audit Layer (Pillar 10)
Handles run manifests, parameter hash-locking, and policy enforcement.
"""

import yaml
import hashlib
import logging
import os
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GovernanceLayer:
    def __init__(self, config_path: str = "config/parameters.yml"):
        self.config_path = config_path
        self.parameters = self._load_parameters()
        self.config_hash = self._compute_hash()
        
    def _load_parameters(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Pillar 6 Violation: Config missing at {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _compute_hash(self) -> str:
        """Hash-locked config (Pillar 6)."""
        with open(self.config_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
            
    def get_manifest(self) -> Dict[str, Any]:
        """Generate audit manifest for the current run (Pillar 10)."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "config_hash": self.config_hash,
            "version": self.parameters.get("version"),
            "environment": self.parameters.get("environment")
        }

    def log_run_start(self):
        manifest = self.get_manifest()
        logger.info(f"AUDIT | Run started with hash: {manifest['config_hash'][:12]}... [v{manifest['version']}]")

# Global Governance Instance
GOVERNANCE = GovernanceLayer()
