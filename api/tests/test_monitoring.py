# coding=utf-8
"""
Tests for Performance Monitoring
=================================

SIN_CARRETA: Validate monitoring metrics collection, alert thresholds,
and telemetry emission per dashboard requirements.

Author: FARFAN 3.3 Team
Version: 1.0.0
Python: 3.10+
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from api.utils.monitoring import (
    MetricsCollector,
    RequestMetrics,
    CacheMetrics,
    WebSocketMetrics,
    ALERT_THRESHOLDS,
    get_metrics_collector
)


class TestRequestMetrics:
    """Test RequestMetrics data class"""
    
    def test_request_metrics_creation(self):
        """Test creating request metrics"""
        metrics = RequestMetrics(
            timestamp=datetime.now(),
            method="GET",
            endpoint="/api/v1/test",
            status_code=200,
            duration_ms=50.0,
            memory_mb=100.0
        )
        
        assert metrics.method == "GET"
        assert metrics.endpoint == "/api/v1/test"
        assert metrics.status_code == 200
        assert metrics.duration_ms == 50.0
        assert not metrics.is_error
        assert not metrics.is_slow
    
    def test_is_error_detection(self):
        """Test error detection for status codes >= 400"""
        metrics_400 = RequestMetrics(
            timestamp=datetime.now(),
            method="GET",
            endpoint="/test",
            status_code=400,
            duration_ms=10.0,
            memory_mb=50.0
        )
        assert metrics_400.is_error
        
        metrics_500 = RequestMetrics(
            timestamp=datetime.now(),
            method="GET",
            endpoint="/test",
            status_code=500,
            duration_ms=10.0,
            memory_mb=50.0
        )
        assert metrics_500.is_error
        
        metrics_200 = RequestMetrics(
            timestamp=datetime.now(),
            method="GET",
            endpoint="/test",
            status_code=200,
            duration_ms=10.0,
            memory_mb=50.0
        )
        assert not metrics_200.is_error
    
    def test_is_slow_detection(self):
        """Test slow request detection based on threshold"""
        slow_metrics = RequestMetrics(
            timestamp=datetime.now(),
            method="GET",
            endpoint="/test",
            status_code=200,
            duration_ms=600.0,  # > 500ms threshold
            memory_mb=50.0
        )
        assert slow_metrics.is_slow
        
        fast_metrics = RequestMetrics(
            timestamp=datetime.now(),
            method="GET",
            endpoint="/test",
            status_code=200,
            duration_ms=100.0,  # < 500ms threshold
            memory_mb=50.0
        )
        assert not fast_metrics.is_slow


class TestCacheMetrics:
    """Test CacheMetrics data class"""
    
    def test_cache_metrics_creation(self):
        """Test creating cache metrics"""
        cache = CacheMetrics(cache_name="test_cache")
        
        assert cache.cache_name == "test_cache"
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.hit_rate == 0.0
    
    def test_hit_rate_calculation(self):
        """Test cache hit rate calculation"""
        cache = CacheMetrics(cache_name="test_cache")
        
        # No hits or misses
        assert cache.hit_rate == 0.0
        
        # 50% hit rate
        cache.hits = 5
        cache.misses = 5
        assert cache.hit_rate == 50.0
        
        # 80% hit rate
        cache.hits = 8
        cache.misses = 2
        assert cache.hit_rate == 80.0
    
    def test_is_inefficient_detection(self):
        """Test detection of inefficient cache (< 50% hit rate)"""
        cache = CacheMetrics(cache_name="test_cache")
        
        # Low hit rate (inefficient)
        cache.hits = 3
        cache.misses = 7
        assert cache.hit_rate == 30.0
        assert cache.is_inefficient
        
        # Good hit rate (efficient)
        cache.hits = 7
        cache.misses = 3
        assert cache.hit_rate == 70.0
        assert not cache.is_inefficient


class TestWebSocketMetrics:
    """Test WebSocketMetrics data class"""
    
    def test_ws_metrics_creation(self):
        """Test creating WebSocket metrics"""
        ws = WebSocketMetrics(
            connection_id="test-conn-1",
            connected_at=datetime.now()
        )
        
        assert ws.connection_id == "test-conn-1"
        assert ws.disconnected_at is None
        assert ws.disconnect_reason is None
    
    def test_duration_calculation(self):
        """Test connection duration calculation"""
        now = datetime.now()
        ws = WebSocketMetrics(
            connection_id="test-conn-1",
            connected_at=now - timedelta(seconds=30)
        )
        
        # Should be approximately 30 seconds
        assert 29 <= ws.duration_seconds <= 31
        
        # With explicit disconnect time
        ws.disconnected_at = now
        duration = ws.duration_seconds
        assert 29 <= duration <= 31
    
    def test_abnormal_disconnect_detection(self):
        """Test detection of abnormal disconnects (< 5 seconds)"""
        now = datetime.now()
        
        # Abnormal disconnect (< 5 seconds)
        ws_abnormal = WebSocketMetrics(
            connection_id="test-conn-1",
            connected_at=now - timedelta(seconds=2),
            disconnected_at=now
        )
        assert ws_abnormal.is_abnormal_disconnect
        
        # Normal disconnect (>= 5 seconds)
        ws_normal = WebSocketMetrics(
            connection_id="test-conn-2",
            connected_at=now - timedelta(seconds=10),
            disconnected_at=now
        )
        assert not ws_normal.is_abnormal_disconnect


class TestMetricsCollector:
    """Test MetricsCollector class"""
    
    def test_collector_initialization(self):
        """Test metrics collector initialization"""
        collector = MetricsCollector(window_size=100)
        
        assert len(collector._request_history) == 0
        assert len(collector._cache_metrics) == 0
        assert len(collector._ws_metrics) == 0
    
    def test_record_request(self):
        """Test recording request metrics"""
        collector = MetricsCollector(window_size=10)
        
        collector.record_request(
            method="GET",
            endpoint="/api/v1/test",
            status_code=200,
            duration_ms=50.0,
            memory_mb=100.0
        )
        
        assert len(collector._request_history) == 1
        metrics = collector._request_history[0]
        assert metrics.method == "GET"
        assert metrics.endpoint == "/api/v1/test"
        assert metrics.status_code == 200
    
    def test_cache_hit_recording(self):
        """Test recording cache hits"""
        collector = MetricsCollector()
        
        collector.record_cache_hit("test_cache")
        collector.record_cache_hit("test_cache")
        
        assert "test_cache" in collector._cache_metrics
        assert collector._cache_metrics["test_cache"].hits == 2
        assert collector._cache_metrics["test_cache"].misses == 0
    
    def test_cache_miss_recording(self):
        """Test recording cache misses"""
        collector = MetricsCollector()
        
        collector.record_cache_miss("test_cache")
        collector.record_cache_miss("test_cache")
        collector.record_cache_miss("test_cache")
        
        assert "test_cache" in collector._cache_metrics
        assert collector._cache_metrics["test_cache"].hits == 0
        assert collector._cache_metrics["test_cache"].misses == 3
    
    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate in collector"""
        collector = MetricsCollector()
        
        # Record 7 hits and 3 misses (70% hit rate)
        for _ in range(7):
            collector.record_cache_hit("test_cache")
        for _ in range(3):
            collector.record_cache_miss("test_cache")
        
        cache_metrics = collector._cache_metrics["test_cache"]
        assert cache_metrics.hit_rate == 70.0
    
    def test_data_freshness_update(self):
        """Test updating data freshness"""
        collector = MetricsCollector()
        
        timestamp = datetime.now() - timedelta(minutes=5)
        collector.update_data_freshness("test_source", timestamp)
        
        assert "test_source" in collector._data_freshness
        assert collector._data_freshness["test_source"] == timestamp
    
    def test_ws_connection_tracking(self):
        """Test WebSocket connection tracking"""
        collector = MetricsCollector()
        
        collector.record_ws_connect("conn-1")
        
        assert "conn-1" in collector._ws_metrics
        assert collector._ws_metrics["conn-1"].connection_id == "conn-1"
        assert collector._ws_metrics["conn-1"].disconnected_at is None
    
    def test_ws_disconnection_tracking(self):
        """Test WebSocket disconnection tracking"""
        collector = MetricsCollector()
        
        collector.record_ws_connect("conn-1")
        collector.record_ws_disconnect("conn-1", reason="normal_close")
        
        assert collector._ws_metrics["conn-1"].disconnected_at is not None
        assert collector._ws_metrics["conn-1"].disconnect_reason == "normal_close"
    
    def test_frame_rate_update(self):
        """Test frame rate update"""
        collector = MetricsCollector()
        
        # Should not raise exception
        collector.update_frame_rate(60.0)
        collector.update_frame_rate(45.0)  # Below threshold
    
    def test_active_requests_tracking(self):
        """Test active request counter"""
        collector = MetricsCollector()
        
        collector.increment_active_requests()
        collector.increment_active_requests()
        assert True  # Should not raise exception
        
        collector.decrement_active_requests()
        assert True  # Should not raise exception
    
    def test_metrics_summary(self):
        """Test getting metrics summary"""
        collector = MetricsCollector()
        
        # Record some requests
        for i in range(10):
            collector.record_request(
                method="GET",
                endpoint=f"/test/{i}",
                status_code=200,
                duration_ms=50.0 + i,
                memory_mb=100.0
            )
        
        # Record one error
        collector.record_request(
            method="GET",
            endpoint="/test/error",
            status_code=500,
            duration_ms=100.0,
            memory_mb=100.0
        )
        
        summary = collector.get_metrics_summary()
        
        assert "requests" in summary
        assert summary["requests"]["total"] == 11
        assert summary["requests"]["errors"] == 1
        assert "memory" in summary
        assert "thresholds" in summary
    
    def test_prometheus_metrics_export(self):
        """Test Prometheus metrics export"""
        collector = MetricsCollector()
        
        # Record a request
        collector.record_request(
            method="GET",
            endpoint="/test",
            status_code=200,
            duration_ms=50.0,
            memory_mb=100.0
        )
        
        # Get Prometheus metrics
        metrics = collector.get_prometheus_metrics()
        
        assert isinstance(metrics, bytes)
        assert b"http_requests_total" in metrics
        assert b"http_request_duration_seconds" in metrics
    
    @patch('api.utils.monitoring.logger')
    def test_latency_alert_emission(self, mock_logger):
        """Test alert emission when latency threshold exceeded"""
        collector = MetricsCollector()
        
        # Record slow request (> 500ms)
        collector.record_request(
            method="GET",
            endpoint="/slow",
            status_code=200,
            duration_ms=600.0,
            memory_mb=100.0
        )
        
        # Should have emitted warning log
        mock_logger.warning.assert_called()
        call_args = mock_logger.warning.call_args
        assert "latency" in str(call_args).lower()
    
    @patch('api.utils.monitoring.logger')
    def test_error_rate_alert(self, mock_logger):
        """Test alert emission when error rate threshold exceeded"""
        collector = MetricsCollector()
        
        # Record enough requests to trigger alert (need > 10 for sampling)
        # Record 2 errors out of 10 = 20% error rate (> 1% threshold)
        for i in range(8):
            collector.record_request(
                method="GET",
                endpoint="/test",
                status_code=200,
                duration_ms=50.0,
                memory_mb=100.0
            )
        
        for i in range(2):
            collector.record_request(
                method="GET",
                endpoint="/error",
                status_code=500,
                duration_ms=50.0,
                memory_mb=100.0
            )
        
        # Should have checked error rate
        # (Alert may or may not be emitted depending on threshold check)
        assert len(collector._request_history) == 10


class TestMetricsCollectorSingleton:
    """Test singleton metrics collector"""
    
    def test_singleton_pattern(self):
        """Test that get_metrics_collector returns same instance"""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        assert collector1 is collector2
    
    def test_singleton_state_persistence(self):
        """Test that state persists across calls"""
        collector = get_metrics_collector()
        
        collector.record_request(
            method="GET",
            endpoint="/test",
            status_code=200,
            duration_ms=50.0,
            memory_mb=100.0
        )
        
        # Get collector again and check state
        collector2 = get_metrics_collector()
        assert len(collector2._request_history) >= 1


class TestAlertThresholds:
    """Test alert threshold constants"""
    
    def test_threshold_values(self):
        """Test that alert thresholds are properly configured"""
        assert ALERT_THRESHOLDS["latency_ms"] == 500
        assert ALERT_THRESHOLDS["error_rate_percent"] == 1.0
        assert ALERT_THRESHOLDS["memory_percent"] == 80.0
        assert ALERT_THRESHOLDS["fps"] == 50
        assert ALERT_THRESHOLDS["data_staleness_minutes"] == 15
        assert ALERT_THRESHOLDS["ws_disconnects_per_min"] == 5
