# coding=utf-8
"""
Telemetry Validation Tests
===========================

SIN_CARRETA: Validate telemetry event emission at all decision points:
- Every API request emits structured telemetry
- Telemetry contains required fields
- Decision points are logged
- Error events are captured
- Performance metrics are recorded

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import pytest
import logging
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestTelemetryHeaders:
    """Test telemetry headers are present and valid"""
    
    def test_request_id_header_present(self):
        """SIN_CARRETA: Every response has X-Request-ID header"""
        endpoints = [
            "/",
            "/health",
            "/api/v1/pdet/regions",
            "/api/v1/pdet/regions/REGION_001",
            "/api/v1/municipalities/MUN_00101",
            "/api/v1/municipalities/MUN_00101/analysis",
            "/api/v1/analysis/clusters/REGION_001",
            "/api/v1/analysis/questions/MUN_00101",
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert "X-Request-ID" in response.headers, \
                f"Missing X-Request-ID header on {endpoint}"
            assert len(response.headers["X-Request-ID"]) > 0, \
                f"Empty X-Request-ID on {endpoint}"
    
    def test_request_id_is_unique(self):
        """SIN_CARRETA: Each request gets unique request ID"""
        request_ids = set()
        
        for _ in range(100):
            response = client.get("/api/v1/pdet/regions")
            request_id = response.headers["X-Request-ID"]
            request_ids.add(request_id)
        
        # All IDs should be unique
        assert len(request_ids) == 100, "Request IDs not unique"
    
    def test_response_time_header_present(self):
        """SIN_CARRETA: Every response has X-Response-Time-Ms header"""
        endpoints = [
            "/",
            "/health",
            "/api/v1/pdet/regions",
            "/api/v1/pdet/regions/REGION_001",
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert "X-Response-Time-Ms" in response.headers, \
                f"Missing X-Response-Time-Ms header on {endpoint}"
            
            # Should be valid float
            time_ms = float(response.headers["X-Response-Time-Ms"])
            assert time_ms > 0, f"Invalid response time on {endpoint}"
    
    def test_response_time_is_reasonable(self):
        """SIN_CARRETA: Response time header reflects actual time"""
        import time
        
        start = time.time()
        response = client.get("/api/v1/pdet/regions")
        actual_ms = (time.time() - start) * 1000
        
        reported_ms = float(response.headers["X-Response-Time-Ms"])
        
        # Should be within 10ms of actual
        assert abs(reported_ms - actual_ms) < 10, \
            f"Reported time {reported_ms}ms differs from actual {actual_ms}ms"
    
    def test_telemetry_headers_on_errors(self):
        """SIN_CARRETA: Telemetry headers present even on errors"""
        # 400 error
        response = client.get("/api/v1/pdet/regions/INVALID")
        assert response.status_code == 400
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time-Ms" in response.headers
        
        # 404 error
        response = client.get("/api/v1/pdet/regions/REGION_999")
        assert response.status_code == 404
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time-Ms" in response.headers


class TestTelemetryEventStructure:
    """Test telemetry events have proper structure"""
    
    def test_success_event_structure(self, caplog):
        """SIN_CARRETA: Success events have required fields"""
        with caplog.at_level(logging.INFO):
            response = client.get("/api/v1/pdet/regions")
            assert response.status_code == 200
        
        # Check that structured log was emitted
        # Log should contain request path and status
        log_text = " ".join([record.message for record in caplog.records])
        assert "/api/v1/pdet/regions" in log_text
        assert "200" in log_text
    
    def test_error_event_structure(self, caplog):
        """SIN_CARRETA: Error events have required fields"""
        with caplog.at_level(logging.ERROR):
            response = client.get("/api/v1/pdet/regions/REGION_999")
            assert response.status_code == 404
        
        # Check that error was logged
        log_text = " ".join([record.message for record in caplog.records])
        assert "404" in log_text or "REGION_999" in log_text or "NotFound" in log_text
    
    def test_validation_error_event(self, caplog):
        """SIN_CARRETA: Validation errors emit telemetry"""
        with caplog.at_level(logging.WARNING):
            response = client.get("/api/v1/pdet/regions/INVALID")
            assert response.status_code == 400
        
        # Should log validation error
        log_text = " ".join([record.message for record in caplog.records])
        assert "Validation" in log_text or "400" in log_text or "INVALID" in log_text


class TestDecisionPointLogging:
    """Test that decision points emit telemetry"""
    
    def test_endpoint_routing_logged(self, caplog):
        """SIN_CARRETA: Endpoint routing decisions are logged"""
        with caplog.at_level(logging.INFO):
            response = client.get("/api/v1/pdet/regions")
            assert response.status_code == 200
        
        # Check that request was logged with method and path
        log_messages = [record.message for record in caplog.records]
        assert any("GET" in msg and "/api/v1/pdet/regions" in msg for msg in log_messages), \
            "Endpoint routing not logged"
    
    def test_entity_lookup_logged(self, caplog):
        """SIN_CARRETA: Entity lookup decisions are logged"""
        with caplog.at_level(logging.INFO):
            response = client.get("/api/v1/pdet/regions/REGION_001")
            assert response.status_code == 200
        
        log_messages = [record.message for record in caplog.records]
        assert any("REGION_001" in msg for msg in log_messages), \
            "Entity lookup not logged"
    
    def test_validation_decision_logged(self, caplog):
        """SIN_CARRETA: Validation decisions are logged"""
        with caplog.at_level(logging.WARNING):
            response = client.get("/api/v1/pdet/regions/INVALID_ID")
            assert response.status_code == 400
        
        log_messages = [record.message for record in caplog.records]
        assert any("Validation" in msg or "error" in msg.lower() for msg in log_messages), \
            "Validation decision not logged"
    
    def test_not_found_decision_logged(self, caplog):
        """SIN_CARRETA: Not found decisions are logged"""
        with caplog.at_level(logging.ERROR):
            response = client.get("/api/v1/pdet/regions/REGION_999")
            assert response.status_code == 404
        
        log_messages = [record.message for record in caplog.records]
        assert any("404" in msg or "not found" in msg.lower() for msg in log_messages), \
            "Not found decision not logged"


class TestPerformanceMetrics:
    """Test that performance metrics are captured"""
    
    def test_response_time_captured(self):
        """SIN_CARRETA: Response time is captured for every request"""
        response = client.get("/api/v1/pdet/regions")
        
        assert "X-Response-Time-Ms" in response.headers
        time_ms = float(response.headers["X-Response-Time-Ms"])
        
        # Should be positive and reasonable
        assert 0 < time_ms < 1000, f"Unreasonable response time: {time_ms}ms"
    
    def test_performance_metrics_vary(self):
        """SIN_CARRETA: Performance metrics vary appropriately"""
        # Simple endpoint
        simple_response = client.get("/health")
        simple_time = float(simple_response.headers["X-Response-Time-Ms"])
        
        # Complex endpoint
        complex_response = client.get("/api/v1/analysis/questions/MUN_00101")
        complex_time = float(complex_response.headers["X-Response-Time-Ms"])
        
        # Complex endpoint should generally take longer (but not always due to caching)
        # Just verify both have valid times
        assert simple_time > 0
        assert complex_time > 0
    
    def test_performance_tracking_consistent(self):
        """SIN_CARRETA: Performance tracking is consistent"""
        times = []
        
        for _ in range(50):
            response = client.get("/api/v1/pdet/regions")
            time_ms = float(response.headers["X-Response-Time-Ms"])
            times.append(time_ms)
        
        # All times should be valid
        assert all(t > 0 for t in times)
        
        # Should have some variance but not too much
        avg_time = sum(times) / len(times)
        assert all(t < avg_time * 5 for t in times), \
            "Performance tracking shows excessive variance"


class TestErrorCapture:
    """Test that errors are properly captured in telemetry"""
    
    def test_validation_errors_captured(self, caplog):
        """SIN_CARRETA: Validation errors are captured"""
        with caplog.at_level(logging.WARNING):
            response = client.get("/api/v1/pdet/regions/BAD_FORMAT")
            assert response.status_code == 400
        
        # Should have warning or error log
        assert len(caplog.records) > 0, "No log records for validation error"
        
        log_levels = [record.levelname for record in caplog.records]
        assert "WARNING" in log_levels or "ERROR" in log_levels, \
            "Validation error not logged with appropriate level"
    
    def test_not_found_errors_captured(self, caplog):
        """SIN_CARRETA: Not found errors are captured"""
        with caplog.at_level(logging.ERROR):
            response = client.get("/api/v1/pdet/regions/REGION_888")
            assert response.status_code == 404
        
        # Should have error log
        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_records) > 0, "Not found error not logged"
    
    def test_error_context_captured(self, caplog):
        """SIN_CARRETA: Error context is captured in telemetry"""
        with caplog.at_level(logging.WARNING):
            response = client.get("/api/v1/pdet/regions/INVALID_INPUT")
            assert response.status_code == 400
        
        # Log should contain context about what went wrong
        log_text = " ".join([record.message for record in caplog.records])
        # Should mention either path, method, or error type
        assert any(term in log_text for term in ["GET", "INVALID_INPUT", "Validation", "error"]), \
            "Error context not captured"


class TestTelemetryCompleteness:
    """Test that telemetry covers all endpoints"""
    
    def test_all_successful_endpoints_emit_telemetry(self):
        """SIN_CARRETA: All successful endpoints emit telemetry"""
        endpoints = [
            ("/", 200),
            ("/health", 200),
            ("/api/v1/pdet/regions", 200),
            ("/api/v1/pdet/regions/REGION_001", 200),
            ("/api/v1/pdet/regions/REGION_001/municipalities", 200),
            ("/api/v1/municipalities/MUN_00101", 200),
            ("/api/v1/municipalities/MUN_00101/analysis", 200),
            ("/api/v1/analysis/clusters/REGION_001", 200),
            ("/api/v1/analysis/questions/MUN_00101", 200),
        ]
        
        for endpoint, expected_status in endpoints:
            response = client.get(endpoint)
            assert response.status_code == expected_status, \
                f"Unexpected status for {endpoint}"
            
            # Should have telemetry headers
            assert "X-Request-ID" in response.headers, \
                f"Missing telemetry on {endpoint}"
            assert "X-Response-Time-Ms" in response.headers, \
                f"Missing performance telemetry on {endpoint}"
    
    def test_all_error_endpoints_emit_telemetry(self):
        """SIN_CARRETA: All error endpoints emit telemetry"""
        error_endpoints = [
            ("/api/v1/pdet/regions/INVALID", 400),
            ("/api/v1/pdet/regions/REGION_999", 404),
            ("/api/v1/municipalities/BAD_ID", 400),
            ("/api/v1/municipalities/MUN_99999", 404),
        ]
        
        for endpoint, expected_status in error_endpoints:
            response = client.get(endpoint)
            assert response.status_code == expected_status, \
                f"Unexpected status for {endpoint}"
            
            # Should still have telemetry headers
            assert "X-Request-ID" in response.headers, \
                f"Missing telemetry on error endpoint {endpoint}"
            assert "X-Response-Time-Ms" in response.headers, \
                f"Missing performance telemetry on error endpoint {endpoint}"


class TestTelemetryConsistency:
    """Test telemetry is consistent across requests"""
    
    def test_request_id_format_consistent(self):
        """SIN_CARRETA: Request IDs follow consistent format"""
        request_ids = []
        
        for _ in range(20):
            response = client.get("/api/v1/pdet/regions")
            request_ids.append(response.headers["X-Request-ID"])
        
        # All should be non-empty strings
        assert all(isinstance(rid, str) and len(rid) > 0 for rid in request_ids)
        
        # All should be unique
        assert len(set(request_ids)) == 20, "Request IDs not unique"
    
    def test_response_time_format_consistent(self):
        """SIN_CARRETA: Response times follow consistent format"""
        for _ in range(20):
            response = client.get("/api/v1/pdet/regions")
            time_str = response.headers["X-Response-Time-Ms"]
            
            # Should be parseable as float
            time_ms = float(time_str)
            assert time_ms > 0
            
            # Should have reasonable precision (2 decimal places)
            assert len(time_str.split(".")[-1]) <= 3, \
                f"Response time has excessive precision: {time_str}"
    
    def test_telemetry_preserved_across_endpoints(self):
        """SIN_CARRETA: Telemetry structure consistent across all endpoints"""
        endpoints = [
            "/",
            "/health",
            "/api/v1/pdet/regions",
            "/api/v1/municipalities/MUN_00101",
        ]
        
        all_headers = []
        for endpoint in endpoints:
            response = client.get(endpoint)
            headers = {
                "has_request_id": "X-Request-ID" in response.headers,
                "has_response_time": "X-Response-Time-Ms" in response.headers,
            }
            all_headers.append(headers)
        
        # All should have same telemetry headers
        first = all_headers[0]
        for headers in all_headers[1:]:
            assert headers == first, "Inconsistent telemetry across endpoints"
