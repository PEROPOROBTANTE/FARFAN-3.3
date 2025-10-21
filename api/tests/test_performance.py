# coding=utf-8
"""
Performance Tests for API Endpoints
====================================

SIN_CARRETA: Validate API performance requirements:
- API response < 200ms for simple endpoints
- Streaming latency < 50ms (if applicable)
- Animation >= 60fps (if applicable)

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import pytest
import time
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestAPIPerformance:
    """Test API endpoint performance requirements"""
    
    def test_root_endpoint_performance(self):
        """SIN_CARRETA: Root endpoint responds in < 200ms"""
        start = time.time()
        response = client.get("/")
        duration_ms = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert duration_ms < 200, f"Response took {duration_ms:.2f}ms, expected < 200ms"
    
    def test_health_endpoint_performance(self):
        """SIN_CARRETA: Health check responds in < 200ms"""
        start = time.time()
        response = client.get("/health")
        duration_ms = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert duration_ms < 200, f"Response took {duration_ms:.2f}ms, expected < 200ms"
    
    def test_list_regions_performance(self):
        """SIN_CARRETA: List regions responds in < 200ms"""
        start = time.time()
        response = client.get("/api/v1/pdet/regions")
        duration_ms = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert duration_ms < 200, f"Response took {duration_ms:.2f}ms, expected < 200ms"
    
    def test_get_region_performance(self):
        """SIN_CARRETA: Get region detail responds in < 200ms"""
        start = time.time()
        response = client.get("/api/v1/pdet/regions/REGION_001")
        duration_ms = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert duration_ms < 200, f"Response took {duration_ms:.2f}ms, expected < 200ms"
    
    def test_list_municipalities_performance(self):
        """SIN_CARRETA: List municipalities responds in < 200ms"""
        start = time.time()
        response = client.get("/api/v1/pdet/regions/REGION_001/municipalities")
        duration_ms = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert duration_ms < 200, f"Response took {duration_ms:.2f}ms, expected < 200ms"
    
    def test_get_municipality_performance(self):
        """SIN_CARRETA: Get municipality detail responds in < 200ms"""
        start = time.time()
        response = client.get("/api/v1/municipalities/MUN_00101")
        duration_ms = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert duration_ms < 200, f"Response took {duration_ms:.2f}ms, expected < 200ms"
    
    def test_municipality_analysis_performance(self):
        """SIN_CARRETA: Municipality analysis responds in < 200ms"""
        start = time.time()
        response = client.get("/api/v1/municipalities/MUN_00101/analysis")
        duration_ms = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert duration_ms < 200, f"Response took {duration_ms:.2f}ms, expected < 200ms"
    
    def test_cluster_analysis_performance(self):
        """SIN_CARRETA: Cluster analysis responds in < 200ms"""
        start = time.time()
        response = client.get("/api/v1/analysis/clusters/REGION_001")
        duration_ms = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert duration_ms < 200, f"Response took {duration_ms:.2f}ms, expected < 200ms"
    
    def test_question_analysis_performance(self):
        """SIN_CARRETA: Question analysis (300 items) responds in < 200ms"""
        start = time.time()
        response = client.get("/api/v1/analysis/questions/MUN_00101")
        duration_ms = (time.time() - start) * 1000
        
        assert response.status_code == 200
        # This endpoint returns 300 questions, so allow slightly more time but still under 200ms target
        assert duration_ms < 200, f"Response took {duration_ms:.2f}ms, expected < 200ms"
    
    def test_response_time_header_accuracy(self):
        """SIN_CARRETA: X-Response-Time-Ms header is accurate"""
        start = time.time()
        response = client.get("/api/v1/pdet/regions")
        actual_duration_ms = (time.time() - start) * 1000
        
        assert "X-Response-Time-Ms" in response.headers
        reported_duration_ms = float(response.headers["X-Response-Time-Ms"])
        
        # Reported time should be close to actual (within 10ms tolerance)
        assert abs(reported_duration_ms - actual_duration_ms) < 10, \
            f"Reported {reported_duration_ms:.2f}ms vs actual {actual_duration_ms:.2f}ms"
    
    def test_concurrent_requests_performance(self):
        """SIN_CARRETA: Multiple concurrent requests maintain performance"""
        import concurrent.futures
        
        def make_request():
            start = time.time()
            response = client.get("/api/v1/pdet/regions")
            duration_ms = (time.time() - start) * 1000
            return response.status_code == 200, duration_ms
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert all(success for success, _ in results)
        
        # Average response time should still be under 200ms
        avg_duration = sum(duration for _, duration in results) / len(results)
        assert avg_duration < 200, f"Average response time {avg_duration:.2f}ms, expected < 200ms"


class TestPerformanceConsistency:
    """Test that performance is consistent across multiple calls"""
    
    def test_repeated_calls_stable_performance(self):
        """SIN_CARRETA: Performance is stable across 100 repeated calls"""
        durations = []
        
        for _ in range(100):
            start = time.time()
            response = client.get("/api/v1/pdet/regions")
            duration_ms = (time.time() - start) * 1000
            
            assert response.status_code == 200
            durations.append(duration_ms)
        
        # Calculate statistics
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)
        
        # All should be under 200ms
        assert max_duration < 200, f"Max duration {max_duration:.2f}ms exceeds 200ms threshold"
        
        # Average should be well under threshold
        assert avg_duration < 100, f"Average duration {avg_duration:.2f}ms should be well under 200ms"
        
        # Variance should be reasonable (max should not be more than 3x min)
        assert max_duration < min_duration * 3, \
            f"Performance variance too high: min={min_duration:.2f}ms, max={max_duration:.2f}ms"
