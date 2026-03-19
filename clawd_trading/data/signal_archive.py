"""Signal Archive - Persistent storage for trading signals.

Archives every signal with full context (price, features, model outputs)
for later analysis and model improvement.

Storage structure:
    signals_history/YYYY-MM-DD/signal_HHMMSS.json
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import os

from integration.firebase_client import FirebaseClient
from contracts.types import EntrySignal, ThreeLayerContext

logger = logging.getLogger(__name__)


@dataclass
class SignalArchiveRecord:
    """Complete signal archive record with full context."""
    # Signal metadata
    archive_id: str
    timestamp: datetime
    date_folder: str
    
    # Original signal
    signal: Dict[str, Any]
    
    # Full layer context
    layer_context: Dict[str, Any]
    
    # Market data at signal time
    market_data: Dict[str, Any] = field(default_factory=dict)
    
    # Feature snapshot
    features: Dict[str, Any] = field(default_factory=dict)
    
    # Raw model outputs
    model_outputs: Dict[str, Any] = field(default_factory=dict)
    
    # Outcome (filled later)
    outcome: Optional[Dict[str, Any]] = None
    
    # Metadata
    archived_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'archive_id': self.archive_id,
            'timestamp': self.timestamp.isoformat(),
            'date_folder': self.date_folder,
            'signal': self.signal,
            'layer_context': self.layer_context,
            'market_data': self.market_data,
            'features': self.features,
            'model_outputs': self.model_outputs,
            'outcome': self.outcome,
            'archived_at': self.archived_at.isoformat(),
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalArchiveRecord':
        """Create from dictionary."""
        return cls(
            archive_id=data['archive_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            date_folder=data['date_folder'],
            signal=data['signal'],
            layer_context=data['layer_context'],
            market_data=data.get('market_data', {}),
            features=data.get('features', {}),
            model_outputs=data.get('model_outputs', {}),
            outcome=data.get('outcome'),
            archived_at=datetime.fromisoformat(data['archived_at']) if data.get('archived_at') else datetime.utcnow(),
            version=data.get('version', '1.0')
        )


class SignalArchive:
    """Signal archive system for storing and retrieving trading signals.
    
    Archives signals to both local filesystem and Firebase for redundancy.
    
    Storage structure:
        signals_history/
            2024-01-15/
                signal_143022.json
                signal_143045.json
            2024-01-16/
                signal_090015.json
    
    Usage:
        archive = SignalArchive(base_path='./signals_history')
        
        # Archive a signal
        archive.archive_signal(
            signal=entry_signal,
            layer_context=three_layer_context,
            market_data=current_market_data,
            features=feature_snapshot
        )
        
        # Retrieve signals for analysis
        signals = archive.get_signals_for_date('2024-01-15')
    """
    
    def __init__(
        self,
        base_path: str = './signals_history',
        firebase_client: Optional[FirebaseClient] = None,
        use_firebase: bool = True
    ):
        """Initialize signal archive.
        
        Args:
            base_path: Local filesystem path for archive storage
            firebase_client: Firebase client for cloud storage
            use_firebase: Whether to also archive to Firebase
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.client = firebase_client or FirebaseClient(demo_mode=not use_firebase)
        self.use_firebase = use_firebase
        
        logger.info(f"SignalArchive initialized: base_path={base_path}")
    
    def _generate_archive_id(self, signal: EntrySignal) -> str:
        """Generate unique archive ID for a signal."""
        timestamp_str = signal.timestamp.strftime('%Y%m%d_%H%M%S')
        return f"{signal.symbol}_{signal.direction.name}_{timestamp_str}"
    
    def _get_date_folder(self, timestamp: datetime) -> str:
        """Get date folder name from timestamp."""
        return timestamp.strftime('%Y-%m-%d')
    
    def _get_filename(self, timestamp: datetime) -> str:
        """Get filename from timestamp."""
        return f"signal_{timestamp.strftime('%H%M%S')}.json"
    
    def _get_local_path(self, date_folder: str, filename: str) -> Path:
        """Get full local path for archive file."""
        folder_path = self.base_path / date_folder
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path / filename
    
    def archive_signal(
        self,
        signal: EntrySignal,
        layer_context: Optional[ThreeLayerContext] = None,
        market_data: Optional[Dict[str, Any]] = None,
        features: Optional[Dict[str, Any]] = None,
        model_outputs: Optional[Dict[str, Any]] = None
    ) -> SignalArchiveRecord:
        """Archive a trading signal with full context.
        
        Args:
            signal: The entry signal to archive
            layer_context: Three-layer context (bias, risk, game, regime)
            market_data: Market data at signal time
            features: Feature values used for prediction
            model_outputs: Raw model output values
            
        Returns:
            SignalArchiveRecord that was stored
        """
        # Generate archive metadata
        archive_id = self._generate_archive_id(signal)
        date_folder = self._get_date_folder(signal.timestamp)
        filename = self._get_filename(signal.timestamp)
        
        # Create record
        record = SignalArchiveRecord(
            archive_id=archive_id,
            timestamp=signal.timestamp,
            date_folder=date_folder,
            signal=signal.to_dict(),
            layer_context=layer_context.to_dict() if layer_context else {},
            market_data=market_data or {},
            features=features or {},
            model_outputs=model_outputs or {}
        )
        
        # Save to local filesystem
        self._save_local(record, date_folder, filename)
        
        # Save to Firebase if enabled
        if self.use_firebase:
            self._save_firebase(record, date_folder, filename)
        
        logger.info(f"Archived signal: {archive_id}")
        return record
    
    def _save_local(self, record: SignalArchiveRecord, date_folder: str, filename: str):
        """Save record to local filesystem."""
        try:
            file_path = self._get_local_path(date_folder, filename)
            with open(file_path, 'w') as f:
                json.dump(record.to_dict(), f, indent=2, default=str)
            logger.debug(f"Saved signal to local: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save signal locally: {e}")
    
    def _save_firebase(self, record: SignalArchiveRecord, date_folder: str, filename: str):
        """Save record to Firebase."""
        try:
            # Remove .json extension for Firebase key
            key = filename.replace('.json', '')
            path = f'/signals_history/{date_folder}/{key}'
            
            self.client.rtdb_set(path, record.to_dict())
            logger.debug(f"Saved signal to Firebase: {path}")
        except Exception as e:
            logger.error(f"Failed to save signal to Firebase: {e}")
    
    def get_signal(
        self,
        date_folder: str,
        signal_id: str
    ) -> Optional[SignalArchiveRecord]:
        """Retrieve a specific signal from archive.
        
        Args:
            date_folder: Date folder (YYYY-MM-DD)
            signal_id: Signal ID (signal_HHMMSS)
            
        Returns:
            SignalArchiveRecord or None if not found
        """
        # Try local first
        try:
            file_path = self._get_local_path(date_folder, f"{signal_id}.json")
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    return SignalArchiveRecord.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load signal from local: {e}")
        
        # Try Firebase
        if self.use_firebase:
            try:
                path = f'/signals_history/{date_folder}/{signal_id}'
                data = self.client.rtdb_get(path)
                if data:
                    return SignalArchiveRecord.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to load signal from Firebase: {e}")
        
        return None
    
    def get_signals_for_date(
        self,
        date: str,
        symbol: Optional[str] = None
    ) -> List[SignalArchiveRecord]:
        """Get all signals for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD)
            symbol: Optional symbol filter
            
        Returns:
            List of SignalArchiveRecord
        """
        records = []
        
        # Try local
        try:
            folder_path = self.base_path / date
            if folder_path.exists():
                for file_path in folder_path.glob('signal_*.json'):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            record = SignalArchiveRecord.from_dict(data)
                            
                            # Apply symbol filter
                            if symbol is None or record.signal.get('symbol') == symbol:
                                records.append(record)
                    except Exception as e:
                        logger.error(f"Failed to load signal file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to load signals from local: {e}")
        
        # Sort by timestamp
        records.sort(key=lambda r: r.timestamp)
        
        return records
    
    def get_signals_for_range(
        self,
        start_date: str,
        end_date: str,
        symbol: Optional[str] = None
    ) -> List[SignalArchiveRecord]:
        """Get signals for a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbol: Optional symbol filter
            
        Returns:
            List of SignalArchiveRecord
        """
        from datetime import timedelta
        
        records = []
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            records.extend(self.get_signals_for_date(date_str, symbol))
            current += timedelta(days=1)
        
        return records
    
    def update_outcome(
        self,
        archive_id: str,
        outcome: Dict[str, Any]
    ) -> bool:
        """Update signal with outcome data after trade completes.
        
        Args:
            archive_id: Archive ID of the signal
            outcome: Outcome data dict with keys like:
                - exit_price
                - exit_time
                - pnl
                - pnl_pct
                - exit_reason
                
        Returns:
            True if update successful
        """
        # Parse archive_id to find file
        parts = archive_id.split('_')
        if len(parts) < 3:
            logger.error(f"Invalid archive_id format: {archive_id}")
            return False
        
        symbol = parts[0]
        direction = parts[1]
        timestamp_str = '_'.join(parts[2:])
        
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            date_folder = self._get_date_folder(timestamp)
            filename = self._get_filename(timestamp)
            signal_id = filename.replace('.json', '')
            
            # Load existing record
            record = self.get_signal(date_folder, signal_id)
            if not record:
                logger.error(f"Signal not found: {archive_id}")
                return False
            
            # Update outcome
            record.outcome = outcome
            record.archived_at = datetime.utcnow()
            
            # Save back
            self._save_local(record, date_folder, filename)
            if self.use_firebase:
                self._save_firebase(record, date_folder, filename)
            
            logger.info(f"Updated outcome for signal: {archive_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update outcome: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get archive statistics.
        
        Returns:
            Dictionary with archive stats
        """
        total_signals = 0
        total_with_outcomes = 0
        date_folders = 0
        
        try:
            for date_folder in self.base_path.iterdir():
                if date_folder.is_dir():
                    date_folders += 1
                    signals = list(date_folder.glob('signal_*.json'))
                    total_signals += len(signals)
                    
                    # Count with outcomes
                    for signal_file in signals:
                        try:
                            with open(signal_file, 'r') as f:
                                data = json.load(f)
                                if data.get('outcome'):
                                    total_with_outcomes += 1
                        except:
                            pass
        except Exception as e:
            logger.error(f"Failed to calculate stats: {e}")
        
        return {
            'total_signals': total_signals,
            'total_with_outcomes': total_with_outcomes,
            'date_folders': date_folders,
            'base_path': str(self.base_path)
        }
    
    def list_dates(self) -> List[str]:
        """List all dates with archived signals.
        
        Returns:
            List of date strings (YYYY-MM-DD)
        """
        dates = []
        try:
            for item in self.base_path.iterdir():
                if item.is_dir():
                    dates.append(item.name)
        except Exception as e:
            logger.error(f"Failed to list dates: {e}")
        
        return sorted(dates)
