"""TradeLocker Client - WebSocket client for TradeLocker execution."""

import os
import json
import logging
from typing import Optional, Dict, Any, Callable
from urllib.parse import urljoin

import requests
from websocket import WebSocketApp

logger = logging.getLogger(__name__)


class TradeLockerClient:
    """TradeLocker API client for order execution."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        base_url: Optional[str] = None,
        env: str = 'demo'
    ):
        self.api_key = api_key or os.getenv('TRADELOCKER_API_KEY')
        self.account_id = account_id or os.getenv('TRADELOCKER_ACCOUNT_ID')
        self.base_url = base_url or os.getenv(
            'TRADELOCKER_BASE_URL',
            'https://api.tradelocker.com'
        )
        self.env = env or os.getenv('TRADELOCKER_ENV', 'demo')
        
        self.ws: Optional[WebSocketApp] = None
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            url = urljoin(self.base_url, f'/api/v1/accounts/{self.account_id}')
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
    
    def get_positions(self) -> list:
        """Get open positions."""
        try:
            url = urljoin(self.base_url, f'/api/v1/accounts/{self.account_id}/positions')
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json().get('positions', [])
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def place_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        quantity: float,
        order_type: str = 'market',
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """Place an order.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: 'market', 'limit', 'stop'
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Order response
        """
        if self.env == 'demo':
            logger.info(f"DEMO: Would place {side} order for {quantity} {symbol}")
            return {'demo': True, 'symbol': symbol, 'side': side, 'quantity': quantity}
        
        try:
            url = urljoin(self.base_url, f'/api/v1/accounts/{self.account_id}/orders')
            
            payload = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'type': order_type
            }
            
            if stop_loss:
                payload['stop_loss'] = stop_loss
            if take_profit:
                payload['take_profit'] = take_profit
            
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            order = response.json()
            logger.info(f"Order placed: {order.get('id')}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {'error': str(e)}
    
    def close_position(self, position_id: str) -> Dict[str, Any]:
        """Close a position."""
        if self.env == 'demo':
            logger.info(f"DEMO: Would close position {position_id}")
            return {'demo': True, 'position_id': position_id}
        
        try:
            url = urljoin(
                self.base_url,
                f'/api/v1/accounts/{self.account_id}/positions/{position_id}/close'
            )
            response = self.session.post(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return {'error': str(e)}
    
    def connect_websocket(
        self,
        on_price: Optional[Callable] = None,
        on_order_update: Optional[Callable] = None
    ):
        """Connect to TradeLocker WebSocket for real-time updates."""
        ws_url = self.base_url.replace('https://', 'wss://').replace('http://', 'ws://')
        ws_url = urljoin(ws_url, '/ws')
        
        def on_message(ws, message):
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'price' and on_price:
                on_price(data)
            elif msg_type == 'order' and on_order_update:
                on_order_update(data)
        
        def on_error(ws, error):
            logger.error(f"TradeLocker WebSocket error: {error}")
        
        def on_close(ws, code, msg):
            logger.info(f"TradeLocker WebSocket closed: {code} - {msg}")
        
        def on_open(ws):
            logger.info("TradeLocker WebSocket connected")
            # Authenticate
            ws.send(json.dumps({
                'action': 'auth',
                'token': self.api_key
            }))
        
        self.ws = WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        import threading
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def disconnect(self):
        """Disconnect WebSocket."""
        if self.ws:
            self.ws.close()
