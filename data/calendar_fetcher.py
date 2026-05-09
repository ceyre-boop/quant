import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import urljoin

import requests
from data.forex_factory_scraper import ForexFactoryScraper

logger = logging.getLogger(__name__)


class CalendarFetcher:
    """Fetches economic calendar events, primarily via Forex Factory scraping."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = 'https://api.tradingeconomics.com'
    ):
        self.api_key = api_key or os.getenv('TRADING_ECON_API_KEY')
        self.base_url = base_url
        self.ff_scraper = ForexFactoryScraper()
    
    def fetch_events(
        self,
        country: str = 'united states',
        days_ahead: int = 1
    ) -> List[Dict[str, Any]]:
        """Fetch economic calendar events.
        
        Args:
            country: Country filter
            days_ahead: Number of days to look ahead (Forex Factory primarily today)
        
        Returns:
            List of calendar events
        """
        # Try Forex Factory first (it's free and preferred by retail/ICT)
        try:
            ff_events = self.ff_scraper.fetch_today_events()
            if ff_events:
                # Convert FF format to internal format
                internal_events = []
                for e in ff_events:
                    internal_events.append({
                        'Date': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                        'Country': e['currency'],
                        'Event': e['event'],
                        'Importance': 3 if e['impact'] == 'High' else (2 if e['impact'] == 'Medium' else 1),
                        'Impact': e['impact']
                    })
                return internal_events
        except Exception as e:
            logger.warning(f"ForexFactoryScraper failed: {e}")

        # Fallback to Trading Economics if API key exists
        if self.api_key:
            try:
                endpoint = f'/calendar/country/{country}'
                url = urljoin(self.base_url, endpoint)
                
                params = {
                    'c': self.api_key,
                    'format': 'json'
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                events = response.json()
                
                # Filter to upcoming events
                today = datetime.now().date()
                end_date = today + timedelta(days=days_ahead)
                
                filtered_events = []
                for event in events:
                    try:
                        event_date = datetime.strptime(event.get('Date', ''), '%Y-%m-%dT%H:%M:%S').date()
                        if today <= event_date <= end_date:
                            filtered_events.append(event)
                    except (ValueError, TypeError):
                        continue
                
                return filtered_events
                
            except Exception as e:
                logger.error(f"Failed to fetch Trading Economics events: {e}")

        return self._get_mock_events()
    
    def get_high_impact_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter for high impact events."""
        high_impact_keywords = [
            'fed', 'fomc', 'interest rate', 'nfp', 'non-farm payrolls',
            'cpi', 'inflation', 'gdp', 'employment', 'unemployment'
        ]
        
        high_impact = []
        for event in events:
            event_name = event.get('Event', '').lower()
            importance = event.get('Importance', 1)
            impact = event.get('Impact', '')
            
            if impact == 'High' or importance >= 3 or any(kw in event_name for kw in high_impact_keywords):
                high_impact.append(event)
        
        return high_impact
    
    def _get_mock_events(self) -> List[Dict[str, Any]]:
        """Return mock events for testing."""
        return [
            {
                'Date': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                'Country': 'USD',
                'Event': 'FOMC Statement',
                'Importance': 3,
                'Impact': 'High'
            }
        ]
    
    def calculate_event_risk(self, events: List[Dict[str, Any]]) -> str:
        """Calculate event risk level for the day.
        
        Returns:
            'CLEAR', 'ELEVATED', 'HIGH', or 'EXTREME'
        """
        high_impact = self.get_high_impact_events(events)
        
        if len(high_impact) == 0:
            return 'CLEAR'
        
        # Check for extreme events (FOMC, CPI, NFP)
        critical_keywords = ['fomc', 'cpi', 'nfp', 'interest rate', 'fed']
        critical_events = [e for e in high_impact if any(kw in e.get('Event', '').lower() for kw in critical_keywords)]
        
        if critical_events:
            return 'EXTREME'
        elif len(high_impact) >= 3:
            return 'HIGH'
        else:
            return 'ELEVATED'

