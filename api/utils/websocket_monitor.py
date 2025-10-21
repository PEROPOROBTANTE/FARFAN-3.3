# coding=utf-8
"""
WebSocket Connection Monitoring
================================

SIN_CARRETA: Monitor WebSocket connections for stability and security.
Tracks connection/disconnection events, rates, and authentication.

Author: FARFAN 3.3 Team
Version: 1.0.0
Python: 3.10+
"""

import logging
from typing import Optional, Dict, Set
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)


class WebSocketMonitor:
    """
    SIN_CARRETA: WebSocket connection monitoring and security
    
    Rationale: Track WebSocket stability per dashboard requirements.
    Monitor disconnect rates and authentication to ensure secure
    real-time data delivery.
    """
    
    def __init__(self):
        """Initialize WebSocket monitor"""
        self._lock = Lock()
        self._active_connections: Set[str] = set()
        self._connection_history: Dict[str, Dict] = {}
        
        logger.info("WebSocketMonitor initialized")
    
    def register_connection(
        self,
        connection_id: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> None:
        """
        SIN_CARRETA: Register new WebSocket connection
        
        Rationale: Track active connections for monitoring and security.
        
        Args:
            connection_id: Unique connection identifier
            user_id: Optional authenticated user ID
            ip_address: Optional client IP address
        """
        with self._lock:
            self._active_connections.add(connection_id)
            self._connection_history[connection_id] = {
                "connected_at": datetime.now(),
                "user_id": user_id,
                "ip_address": ip_address,
                "disconnected_at": None,
                "disconnect_reason": None
            }
        
        logger.info(
            f"WebSocket connection registered: {connection_id}",
            extra={
                "event_type": "websocket",
                "action": "connect",
                "connection_id": connection_id,
                "user_id": user_id,
                "ip_address": ip_address,
                "active_connections": len(self._active_connections)
            }
        )
        
        # Record in metrics collector if available
        try:
            from api.utils.monitoring import get_metrics_collector
            get_metrics_collector().record_ws_connect(connection_id)
        except ImportError:
            pass
    
    def unregister_connection(
        self,
        connection_id: str,
        reason: Optional[str] = None
    ) -> None:
        """
        SIN_CARRETA: Unregister WebSocket connection
        
        Rationale: Track disconnections to monitor stability and
        detect abnormal patterns.
        
        Args:
            connection_id: Unique connection identifier
            reason: Optional disconnect reason
        """
        with self._lock:
            if connection_id in self._active_connections:
                self._active_connections.remove(connection_id)
            
            if connection_id in self._connection_history:
                self._connection_history[connection_id]["disconnected_at"] = datetime.now()
                self._connection_history[connection_id]["disconnect_reason"] = reason
        
        logger.info(
            f"WebSocket connection unregistered: {connection_id}",
            extra={
                "event_type": "websocket",
                "action": "disconnect",
                "connection_id": connection_id,
                "reason": reason,
                "active_connections": len(self._active_connections)
            }
        )
        
        # Record in metrics collector if available
        try:
            from api.utils.monitoring import get_metrics_collector
            get_metrics_collector().record_ws_disconnect(connection_id, reason)
        except ImportError:
            pass
    
    def is_connected(self, connection_id: str) -> bool:
        """
        Check if connection is active
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            True if connection is active
        """
        with self._lock:
            return connection_id in self._active_connections
    
    def get_active_count(self) -> int:
        """
        Get count of active connections
        
        Returns:
            Number of active connections
        """
        with self._lock:
            return len(self._active_connections)
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict]:
        """
        Get connection information
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            Connection info dict or None
        """
        with self._lock:
            return self._connection_history.get(connection_id)
    
    def get_statistics(self) -> Dict:
        """
        SIN_CARRETA: Get WebSocket statistics
        
        Rationale: Provide consolidated view of connection health
        for monitoring and alerting.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            total_connections = len(self._connection_history)
            active_connections = len(self._active_connections)
            
            # Calculate average connection duration
            completed = [
                c for c in self._connection_history.values()
                if c["disconnected_at"] is not None
            ]
            
            avg_duration = 0.0
            if completed:
                durations = [
                    (c["disconnected_at"] - c["connected_at"]).total_seconds()
                    for c in completed
                ]
                avg_duration = sum(durations) / len(durations)
            
            # Count abnormal disconnects (< 5 seconds)
            abnormal_disconnects = sum(
                1 for c in completed
                if (c["disconnected_at"] - c["connected_at"]).total_seconds() < 5.0
            )
            
            return {
                "active_connections": active_connections,
                "total_connections": total_connections,
                "completed_connections": len(completed),
                "average_duration_seconds": round(avg_duration, 2),
                "abnormal_disconnects": abnormal_disconnects,
                "abnormal_disconnect_rate": (
                    (abnormal_disconnects / len(completed) * 100)
                    if completed else 0.0
                )
            }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_ws_monitor: Optional[WebSocketMonitor] = None


def get_websocket_monitor() -> WebSocketMonitor:
    """
    SIN_CARRETA: Get singleton WebSocket monitor instance
    
    Rationale: Ensure single source of truth for WebSocket monitoring.
    
    Returns:
        WebSocketMonitor instance
    """
    global _ws_monitor
    if _ws_monitor is None:
        _ws_monitor = WebSocketMonitor()
    return _ws_monitor
