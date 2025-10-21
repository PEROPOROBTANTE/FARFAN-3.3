# coding=utf-8
"""
Performance Monitoring and Metrics Collection
==============================================

SIN_CARRETA: Centralized performance monitoring for AtroZ Dashboard API.
Tracks response times, memory usage, cache stats, error rates, data freshness,
and frame rates. Emits alerts when thresholds exceeded.

Author: FARFAN 3.3 Team
Version: 1.0.0
Python: 3.10+
"""

import logging
import psutil
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from threading import Lock

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ALERT THRESHOLDS (per requirements)
# ============================================================================

ALERT_THRESHOLDS = {
    "latency_ms": 500,  # API response time > 500ms
    "error_rate_percent": 1.0,  # Error rate > 1%
    "memory_percent": 80.0,  # Memory usage > 80%
    "fps": 50,  # Frame rate < 50 fps
    "data_staleness_minutes": 15,  # Data staleness > 15 minutes
    "ws_disconnects_per_min": 5,  # WebSocket disconnects > 5/min
}


# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# Create registry for metrics
registry = CollectorRegistry()

# HTTP request duration histogram
http_request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint", "status"],
    registry=registry,
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
)

# HTTP requests total counter
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=registry,
)

# Active requests gauge
active_requests = Gauge(
    "http_active_requests", "Number of active HTTP requests", registry=registry
)

# Memory usage gauge
memory_usage = Gauge(
    "system_memory_usage_percent", "System memory usage percentage", registry=registry
)

# Cache hit rate gauge
cache_hit_rate = Gauge(
    "cache_hit_rate", "Cache hit rate percentage", ["cache_name"], registry=registry
)

# Data freshness gauge
data_freshness = Gauge(
    "data_freshness_seconds",
    "Age of data in seconds",
    ["data_source"],
    registry=registry,
)

# Error rate gauge
error_rate = Gauge("error_rate_percent", "Error rate as percentage", registry=registry)

# Frame rate gauge (for UI monitoring)
frame_rate = Gauge("ui_frame_rate", "UI frame rate in FPS", registry=registry)

# WebSocket disconnects counter
ws_disconnects = Counter(
    "websocket_disconnects_total", "Total WebSocket disconnections", registry=registry
)


# ============================================================================
# PERFORMANCE METRICS DATA CLASSES
# ============================================================================


@dataclass
class RequestMetrics:
    """
    SIN_CARRETA: Metrics for a single request

    Rationale: Encapsulate all timing and status data for a request
    to enable comprehensive analysis and alerting.
    """

    timestamp: datetime
    method: str
    endpoint: str
    status_code: int
    duration_ms: float
    memory_mb: float

    @property
    def is_error(self) -> bool:
        """Check if request resulted in error"""
        return self.status_code >= 400

    @property
    def is_slow(self) -> bool:
        """Check if request exceeded latency threshold"""
        return self.duration_ms > ALERT_THRESHOLDS["latency_ms"]


@dataclass
class CacheMetrics:
    """
    SIN_CARRETA: Cache performance metrics

    Rationale: Track cache efficiency to optimize data access patterns
    and identify caching opportunities.
    """

    cache_name: str
    hits: int = 0
    misses: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage"""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100.0

    @property
    def is_inefficient(self) -> bool:
        """Check if cache hit rate is too low"""
        return self.hit_rate < 50.0  # Alert if <50% hit rate


@dataclass
class WebSocketMetrics:
    """
    SIN_CARRETA: WebSocket connection metrics

    Rationale: Monitor WebSocket stability per dashboard requirements
    to ensure real-time data delivery reliability.
    """

    connection_id: str
    connected_at: datetime
    disconnected_at: Optional[datetime] = None
    disconnect_reason: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        """Calculate connection duration"""
        end = self.disconnected_at or datetime.now()
        return (end - self.connected_at).total_seconds()

    @property
    def is_abnormal_disconnect(self) -> bool:
        """Check if disconnect was abnormal (too quick)"""
        return self.duration_seconds < 5.0  # Alert if disconnected in <5s


# ============================================================================
# METRICS COLLECTOR
# ============================================================================


class MetricsCollector:
    """
    SIN_CARRETA: Centralized metrics collection and alerting

    Rationale: Single source of truth for all performance metrics.
    Automatically checks thresholds and emits alerts when exceeded.
    Thread-safe for concurrent request handling.
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics collector

        Args:
            window_size: Number of recent requests to keep for analysis
        """
        self._lock = Lock()
        self._request_history: deque = deque(maxlen=window_size)
        self._cache_metrics: Dict[str, CacheMetrics] = {}
        self._ws_metrics: Dict[str, WebSocketMetrics] = {}
        self._data_freshness: Dict[str, datetime] = {}
        self._ws_disconnect_history: deque = deque(maxlen=100)

        logger.info(
            "MetricsCollector initialized",
            extra={"window_size": window_size, "thresholds": ALERT_THRESHOLDS},
        )

    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        memory_mb: float,
    ) -> None:
        """
        SIN_CARRETA: Record a completed request

        Rationale: Capture all request metrics for analysis and alerting.
        Updates Prometheus metrics and checks alert thresholds.

        Args:
            method: HTTP method
            endpoint: Request endpoint
            status_code: Response status code
            duration_ms: Request duration in milliseconds
            memory_mb: Memory usage in MB
        """
        with self._lock:
            # Create metrics object
            metrics = RequestMetrics(
                timestamp=datetime.now(),
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration_ms=duration_ms,
                memory_mb=memory_mb,
            )

            # Add to history
            self._request_history.append(metrics)

            # Update Prometheus metrics
            http_requests_total.labels(
                method=method, endpoint=endpoint, status=status_code
            ).inc()

            http_request_duration.labels(
                method=method, endpoint=endpoint, status=status_code
            ).observe(
                duration_ms / 1000.0
            )  # Convert to seconds

            # Update system memory
            memory_usage.set(psutil.virtual_memory().percent)

            # Check and emit alerts
            self._check_latency_alert(metrics)
            self._check_error_rate_alert()
            self._check_memory_alert()

        logger.debug(
            f"Request recorded: {method} {endpoint} -> {status_code} ({duration_ms:.2f}ms)",
            extra={"metrics": metrics},
        )

    def record_cache_hit(self, cache_name: str) -> None:
        """
        SIN_CARRETA: Record a cache hit

        Rationale: Track cache efficiency to optimize data access.

        Args:
            cache_name: Name of the cache
        """
        with self._lock:
            if cache_name not in self._cache_metrics:
                self._cache_metrics[cache_name] = CacheMetrics(cache_name=cache_name)

            self._cache_metrics[cache_name].hits += 1
            self._cache_metrics[cache_name].last_updated = datetime.now()

            # Update Prometheus metric
            hit_rate = self._cache_metrics[cache_name].hit_rate
            cache_hit_rate.labels(cache_name=cache_name).set(hit_rate)

            logger.debug(
                f"Cache hit recorded: {cache_name} (hit rate: {hit_rate:.2f}%)"
            )

    def record_cache_miss(self, cache_name: str) -> None:
        """
        SIN_CARRETA: Record a cache miss

        Rationale: Track cache efficiency to identify optimization opportunities.

        Args:
            cache_name: Name of the cache
        """
        with self._lock:
            if cache_name not in self._cache_metrics:
                self._cache_metrics[cache_name] = CacheMetrics(cache_name=cache_name)

            self._cache_metrics[cache_name].misses += 1
            self._cache_metrics[cache_name].last_updated = datetime.now()

            # Update Prometheus metric
            hit_rate = self._cache_metrics[cache_name].hit_rate
            cache_hit_rate.labels(cache_name=cache_name).set(hit_rate)

            # Alert if inefficient
            if self._cache_metrics[cache_name].is_inefficient:
                self._emit_alert(
                    alert_type="low_cache_hit_rate",
                    message=f"Cache '{cache_name}' has low hit rate: {hit_rate:.2f}%",
                    context={"cache_name": cache_name, "hit_rate": hit_rate},
                )

            logger.debug(
                f"Cache miss recorded: {cache_name} (hit rate: {hit_rate:.2f}%)"
            )

    def update_data_freshness(self, data_source: str, timestamp: datetime) -> None:
        """
        SIN_CARRETA: Update data freshness timestamp

        Rationale: Track data staleness to ensure timely updates per requirements.

        Args:
            data_source: Name of data source
            timestamp: Timestamp of latest data
        """
        with self._lock:
            self._data_freshness[data_source] = timestamp

            # Calculate age in seconds
            age_seconds = (datetime.now() - timestamp).total_seconds()
            data_freshness.labels(data_source=data_source).set(age_seconds)

            # Check staleness threshold (15 minutes)
            age_minutes = age_seconds / 60.0
            if age_minutes > ALERT_THRESHOLDS["data_staleness_minutes"]:
                self._emit_alert(
                    alert_type="data_staleness",
                    message=f"Data source '{data_source}' is stale: {age_minutes:.1f} minutes old",
                    context={"data_source": data_source, "age_minutes": age_minutes},
                )

            logger.debug(
                f"Data freshness updated: {data_source} ({age_seconds:.1f}s old)"
            )

    def record_ws_connect(self, connection_id: str) -> None:
        """
        SIN_CARRETA: Record WebSocket connection

        Rationale: Track WebSocket stability per dashboard requirements.

        Args:
            connection_id: Unique connection identifier
        """
        with self._lock:
            self._ws_metrics[connection_id] = WebSocketMetrics(
                connection_id=connection_id, connected_at=datetime.now()
            )

        logger.info(f"WebSocket connected: {connection_id}")

    def record_ws_disconnect(
        self, connection_id: str, reason: Optional[str] = None
    ) -> None:
        """
        SIN_CARRETA: Record WebSocket disconnection

        Rationale: Monitor disconnect rate to detect stability issues.

        Args:
            connection_id: Unique connection identifier
            reason: Optional disconnect reason
        """
        with self._lock:
            if connection_id in self._ws_metrics:
                self._ws_metrics[connection_id].disconnected_at = datetime.now()
                self._ws_metrics[connection_id].disconnect_reason = reason

                # Track disconnect history
                self._ws_disconnect_history.append(datetime.now())

                # Update Prometheus counter
                ws_disconnects.inc()

                # Check disconnect rate
                self._check_ws_disconnect_rate()

                # Check if abnormal disconnect
                if self._ws_metrics[connection_id].is_abnormal_disconnect:
                    self._emit_alert(
                        alert_type="abnormal_ws_disconnect",
                        message=f"WebSocket {connection_id} disconnected abnormally after {self._ws_metrics[connection_id].duration_seconds:.1f}s",
                        context={
                            "connection_id": connection_id,
                            "duration_seconds": self._ws_metrics[
                                connection_id
                            ].duration_seconds,
                            "reason": reason,
                        },
                    )

        logger.info(f"WebSocket disconnected: {connection_id} (reason: {reason})")

    def update_frame_rate(self, fps: float) -> None:
        """
        SIN_CARRETA: Update UI frame rate

        Rationale: Monitor UI performance to ensure smooth user experience.

        Args:
            fps: Current frame rate
        """
        with self._lock:
            frame_rate.set(fps)

            # Check FPS threshold
            if fps < ALERT_THRESHOLDS["fps"]:
                self._emit_alert(
                    alert_type="low_frame_rate",
                    message=f"Frame rate dropped to {fps:.1f} FPS (threshold: {ALERT_THRESHOLDS['fps']})",
                    context={"fps": fps, "threshold": ALERT_THRESHOLDS["fps"]},
                )

        logger.debug(f"Frame rate updated: {fps:.1f} FPS")

    def increment_active_requests(self) -> None:
        """Increment active request counter"""
        active_requests.inc()

    def decrement_active_requests(self) -> None:
        """Decrement active request counter"""
        active_requests.dec()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        SIN_CARRETA: Get summary of all metrics

        Rationale: Provide consolidated view for health checks and monitoring.

        Returns:
            Dictionary with all current metrics
        """
        with self._lock:
            recent_requests = list(self._request_history)[-100:]  # Last 100 requests

            total_requests = len(recent_requests)
            error_requests = sum(1 for r in recent_requests if r.is_error)
            slow_requests = sum(1 for r in recent_requests if r.is_slow)

            avg_latency = (
                sum(r.duration_ms for r in recent_requests) / total_requests
                if total_requests > 0
                else 0.0
            )

            return {
                "requests": {
                    "total": total_requests,
                    "errors": error_requests,
                    "error_rate_percent": (
                        (error_requests / total_requests * 100)
                        if total_requests > 0
                        else 0.0
                    ),
                    "slow": slow_requests,
                    "avg_latency_ms": round(avg_latency, 2),
                },
                "memory": {
                    "percent": psutil.virtual_memory().percent,
                    "available_mb": psutil.virtual_memory().available / (1024 * 1024),
                },
                "cache": {
                    name: {
                        "hits": cache.hits,
                        "misses": cache.misses,
                        "hit_rate_percent": round(cache.hit_rate, 2),
                    }
                    for name, cache in self._cache_metrics.items()
                },
                "websockets": {
                    "active": sum(
                        1
                        for ws in self._ws_metrics.values()
                        if ws.disconnected_at is None
                    ),
                    "total": len(self._ws_metrics),
                    "recent_disconnects": len(
                        [
                            d
                            for d in self._ws_disconnect_history
                            if (datetime.now() - d).total_seconds() < 60
                        ]
                    ),
                },
                "data_freshness": {
                    source: (datetime.now() - ts).total_seconds()
                    for source, ts in self._data_freshness.items()
                },
                "thresholds": ALERT_THRESHOLDS,
            }

    def get_prometheus_metrics(self) -> bytes:
        """
        SIN_CARRETA: Get Prometheus-formatted metrics

        Rationale: Enable Prometheus scraping for external monitoring.

        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(registry)

    def _check_latency_alert(self, metrics: RequestMetrics) -> None:
        """Check if latency threshold exceeded and emit alert"""
        if metrics.is_slow:
            self._emit_alert(
                alert_type="high_latency",
                message=f"Request latency exceeded threshold: {metrics.duration_ms:.2f}ms (threshold: {ALERT_THRESHOLDS['latency_ms']}ms)",
                context={
                    "method": metrics.method,
                    "endpoint": metrics.endpoint,
                    "duration_ms": metrics.duration_ms,
                    "threshold": ALERT_THRESHOLDS["latency_ms"],
                },
            )

    def _check_error_rate_alert(self) -> None:
        """Check if error rate threshold exceeded and emit alert"""
        recent_requests = list(self._request_history)[-100:]  # Last 100 requests
        if len(recent_requests) < 10:  # Need minimum sample
            return

        error_count = sum(1 for r in recent_requests if r.is_error)
        error_rate_percent = (error_count / len(recent_requests)) * 100.0

        # Update Prometheus gauge
        error_rate.set(error_rate_percent)

        if error_rate_percent > ALERT_THRESHOLDS["error_rate_percent"]:
            self._emit_alert(
                alert_type="high_error_rate",
                message=f"Error rate exceeded threshold: {error_rate_percent:.2f}% (threshold: {ALERT_THRESHOLDS['error_rate_percent']}%)",
                context={
                    "error_rate_percent": error_rate_percent,
                    "threshold": ALERT_THRESHOLDS["error_rate_percent"],
                    "sample_size": len(recent_requests),
                },
            )

    def _check_memory_alert(self) -> None:
        """Check if memory usage threshold exceeded and emit alert"""
        mem_percent = psutil.virtual_memory().percent

        if mem_percent > ALERT_THRESHOLDS["memory_percent"]:
            self._emit_alert(
                alert_type="high_memory",
                message=f"Memory usage exceeded threshold: {mem_percent:.1f}% (threshold: {ALERT_THRESHOLDS['memory_percent']}%)",
                context={
                    "memory_percent": mem_percent,
                    "threshold": ALERT_THRESHOLDS["memory_percent"],
                    "available_mb": psutil.virtual_memory().available / (1024 * 1024),
                },
            )

    def _check_ws_disconnect_rate(self) -> None:
        """Check if WebSocket disconnect rate exceeded threshold"""
        # Count disconnects in last minute
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        recent_disconnects = [
            d for d in self._ws_disconnect_history if d >= one_minute_ago
        ]

        disconnect_rate = len(recent_disconnects)

        if disconnect_rate > ALERT_THRESHOLDS["ws_disconnects_per_min"]:
            self._emit_alert(
                alert_type="high_ws_disconnect_rate",
                message=f"WebSocket disconnect rate exceeded threshold: {disconnect_rate}/min (threshold: {ALERT_THRESHOLDS['ws_disconnects_per_min']}/min)",
                context={
                    "disconnect_rate": disconnect_rate,
                    "threshold": ALERT_THRESHOLDS["ws_disconnects_per_min"],
                },
            )

    def _emit_alert(
        self, alert_type: str, message: str, context: Dict[str, Any]
    ) -> None:
        """
        SIN_CARRETA: Emit structured alert

        Rationale: Centralize alert emission for consistent logging and
        future integration with external alerting systems.

        Args:
            alert_type: Type of alert
            message: Human-readable alert message
            context: Additional context data
        """
        logger.warning(
            message,
            extra={
                "event_type": "alert",
                "alert_type": alert_type,
                "timestamp": datetime.now().isoformat(),
                "context": context,
            },
        )


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    SIN_CARRETA: Get singleton metrics collector instance

    Rationale: Ensure single source of truth for metrics across application.

    Returns:
        MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
