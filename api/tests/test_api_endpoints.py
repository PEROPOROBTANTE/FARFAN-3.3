# coding=utf-8
"""
Tests for API Endpoints
========================

SIN_CARRETA: Test all API endpoints for contract validation, determinism, and telemetry.

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestRootEndpoints:
    """Test root and health endpoints"""
    
    def test_root_endpoint(self):
        """SIN_CARRETA: Root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "AtroZ Dashboard API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
    
    def test_health_endpoint(self):
        """SIN_CARRETA: Health endpoint returns status"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "atroz-api"


class TestPDETRegionsEndpoints:
    """Test PDET regions endpoints"""
    
    def test_list_regions_success(self):
        """SIN_CARRETA: List regions returns valid data"""
        response = client.get("/api/v1/pdet/regions")
        assert response.status_code == 200
        
        data = response.json()
        assert "regions" in data
        assert "total" in data
        assert "timestamp" in data
        
        assert data["total"] == 10
        assert len(data["regions"]) == 10
        
        # Verify first region structure
        region = data["regions"][0]
        assert region["id"] == "REGION_001"
        assert "name" in region
        assert "coordinates" in region
        assert "overall_score" in region
        
        # Verify coordinates
        coords = region["coordinates"]
        assert "latitude" in coords
        assert "longitude" in coords
    
    def test_list_regions_determinism(self):
        """SIN_CARRETA: Multiple calls return identical data"""
        response1 = client.get("/api/v1/pdet/regions")
        response2 = client.get("/api/v1/pdet/regions")
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Timestamps will differ, so compare regions only
        assert data1["regions"] == data2["regions"]
        assert data1["total"] == data2["total"]
    
    def test_get_region_success(self):
        """SIN_CARRETA: Get region returns valid data"""
        response = client.get("/api/v1/pdet/regions/REGION_001")
        assert response.status_code == 200
        
        data = response.json()
        assert "region" in data
        assert "timestamp" in data
        
        region = data["region"]
        assert region["id"] == "REGION_001"
        assert "dimension_scores" in region
        assert "policy_area_scores" in region
        assert "metadata" in region
        
        # Verify all 6 dimensions present
        assert len(region["dimension_scores"]) == 6
        assert all(f"D{i}" in region["dimension_scores"] for i in range(1, 7))
        
        # Verify all 10 policy areas present
        assert len(region["policy_area_scores"]) == 10
        assert all(f"P{i}" in region["policy_area_scores"] for i in range(1, 11))
    
    def test_get_region_determinism(self):
        """SIN_CARRETA: Same region ID returns identical data"""
        response1 = client.get("/api/v1/pdet/regions/REGION_005")
        response2 = client.get("/api/v1/pdet/regions/REGION_005")
        
        region1 = response1.json()["region"]
        region2 = response2.json()["region"]
        
        # Compare excluding timestamp fields
        assert region1["id"] == region2["id"]
        assert region1["name"] == region2["name"]
        assert region1["overall_score"] == region2["overall_score"]
        assert region1["dimension_scores"] == region2["dimension_scores"]
        assert region1["policy_area_scores"] == region2["policy_area_scores"]
    
    def test_get_region_not_found(self):
        """SIN_CARRETA: Invalid region returns 404"""
        response = client.get("/api/v1/pdet/regions/REGION_999")
        assert response.status_code == 404
        
        data = response.json()
        assert data["detail"]["error"] == "NotFound"
    
    def test_get_region_invalid_format(self):
        """SIN_CARRETA: Invalid format returns 400"""
        response = client.get("/api/v1/pdet/regions/INVALID")
        assert response.status_code == 400  # Custom validation error handler
    
    def test_list_municipalities_success(self):
        """SIN_CARRETA: List municipalities returns valid data"""
        response = client.get("/api/v1/pdet/regions/REGION_001/municipalities")
        assert response.status_code == 200
        
        data = response.json()
        assert "municipalities" in data
        assert "region_id" in data
        assert "total" in data
        
        assert data["region_id"] == "REGION_001"
        assert data["total"] == 10
        assert len(data["municipalities"]) == 10
        
        # Verify municipality structure
        mun = data["municipalities"][0]
        assert "id" in mun
        assert mun["id"].startswith("MUN_")
        assert mun["region_id"] == "REGION_001"
    
    def test_list_municipalities_determinism(self):
        """SIN_CARRETA: Multiple calls return identical municipalities"""
        response1 = client.get("/api/v1/pdet/regions/REGION_003/municipalities")
        response2 = client.get("/api/v1/pdet/regions/REGION_003/municipalities")
        
        muns1 = response1.json()["municipalities"]
        muns2 = response2.json()["municipalities"]
        
        assert muns1 == muns2


class TestMunicipalitiesEndpoints:
    """Test municipalities endpoints"""
    
    def test_get_municipality_success(self):
        """SIN_CARRETA: Get municipality returns valid data"""
        response = client.get("/api/v1/municipalities/MUN_00101")
        assert response.status_code == 200
        
        data = response.json()
        assert "municipality" in data
        
        mun = data["municipality"]
        assert mun["id"] == "MUN_00101"
        assert "dimension_scores" in mun
        assert "policy_area_scores" in mun
        assert "metadata" in mun
        
        # Verify dimensions and policies
        assert len(mun["dimension_scores"]) == 6
        assert len(mun["policy_area_scores"]) == 10
    
    def test_get_municipality_determinism(self):
        """SIN_CARRETA: Same municipality returns identical data"""
        response1 = client.get("/api/v1/municipalities/MUN_00205")
        response2 = client.get("/api/v1/municipalities/MUN_00205")
        
        mun1 = response1.json()["municipality"]
        mun2 = response2.json()["municipality"]
        
        assert mun1["id"] == mun2["id"]
        assert mun1["overall_score"] == mun2["overall_score"]
        assert mun1["dimension_scores"] == mun2["dimension_scores"]
    
    def test_get_municipality_not_found(self):
        """SIN_CARRETA: Invalid municipality returns 404"""
        response = client.get("/api/v1/municipalities/MUN_99999")
        assert response.status_code == 404
    
    def test_get_municipality_analysis_success(self):
        """SIN_CARRETA: Get analysis returns 6 dimensions"""
        response = client.get("/api/v1/municipalities/MUN_00101/analysis")
        assert response.status_code == 200
        
        data = response.json()
        assert data["municipality_id"] == "MUN_00101"
        assert "dimensions" in data
        assert "summary" in data
        
        # Must have exactly 6 dimensions
        assert len(data["dimensions"]) == 6
        
        # Each dimension must have 5 questions
        for dim in data["dimensions"]:
            assert "dimension_id" in dim
            assert "questions" in dim
            assert len(dim["questions"]) == 5
            assert "strengths" in dim
            assert "weaknesses" in dim
    
    def test_get_municipality_analysis_determinism(self):
        """SIN_CARRETA: Analysis is deterministic"""
        response1 = client.get("/api/v1/municipalities/MUN_00302/analysis")
        response2 = client.get("/api/v1/municipalities/MUN_00302/analysis")
        
        dims1 = response1.json()["dimensions"]
        dims2 = response2.json()["dimensions"]
        
        # Compare dimension scores and questions
        for d1, d2 in zip(dims1, dims2):
            assert d1["dimension_id"] == d2["dimension_id"]
            assert d1["score"] == d2["score"]
            assert len(d1["questions"]) == len(d2["questions"])


class TestAnalysisEndpoints:
    """Test analysis endpoints"""
    
    def test_get_clusters_success(self):
        """SIN_CARRETA: Cluster analysis returns valid data"""
        response = client.get("/api/v1/analysis/clusters/REGION_001")
        assert response.status_code == 200
        
        data = response.json()
        assert data["region_id"] == "REGION_001"
        assert "clusters" in data
        assert "summary" in data
        
        # Verify cluster structure
        assert len(data["clusters"]) >= 1
        
        cluster = data["clusters"][0]
        assert "cluster_id" in cluster
        assert "centroid_scores" in cluster
        assert "members" in cluster
        assert "characteristics" in cluster
        
        # Centroid must have all 6 dimensions
        assert len(cluster["centroid_scores"]) == 6
    
    def test_get_clusters_determinism(self):
        """SIN_CARRETA: Clusters are deterministic"""
        response1 = client.get("/api/v1/analysis/clusters/REGION_002")
        response2 = client.get("/api/v1/analysis/clusters/REGION_002")
        
        clusters1 = response1.json()["clusters"]
        clusters2 = response2.json()["clusters"]
        
        assert len(clusters1) == len(clusters2)
        
        for c1, c2 in zip(clusters1, clusters2):
            assert c1["cluster_id"] == c2["cluster_id"]
            assert c1["centroid_scores"] == c2["centroid_scores"]
    
    def test_get_questions_success(self):
        """SIN_CARRETA: Question analysis returns exactly 300 questions"""
        response = client.get("/api/v1/analysis/questions/MUN_00101")
        assert response.status_code == 200
        
        data = response.json()
        assert data["municipality_id"] == "MUN_00101"
        assert data["total_questions"] == 300
        assert len(data["questions"]) == 300
        
        # Verify groupings
        assert "by_dimension" in data
        assert "by_policy_area" in data
        
        # Each dimension should have 50 questions (10 policies × 5 questions)
        for dim_id, questions in data["by_dimension"].items():
            assert len(questions) == 50, f"Dimension {dim_id} should have 50 questions"
        
        # Each policy area should have 30 questions (6 dimensions × 5 questions)
        for policy_id, questions in data["by_policy_area"].items():
            assert len(questions) == 30, f"Policy {policy_id} should have 30 questions"
    
    def test_get_questions_determinism(self):
        """SIN_CARRETA: Questions are deterministic"""
        response1 = client.get("/api/v1/analysis/questions/MUN_00405")
        response2 = client.get("/api/v1/analysis/questions/MUN_00405")
        
        questions1 = response1.json()["questions"]
        questions2 = response2.json()["questions"]
        
        assert len(questions1) == len(questions2) == 300
        
        # Compare first 10 questions
        for q1, q2 in zip(questions1[:10], questions2[:10]):
            assert q1["question_id"] == q2["question_id"]
            assert q1["quantitative_score"] == q2["quantitative_score"]
            assert q1["qualitative_level"] == q2["qualitative_level"]


class TestContractEnforcement:
    """Test strict contract validation (400/403 errors)"""
    
    def test_invalid_region_id_format(self):
        """SIN_CARRETA: Invalid region ID format returns validation error"""
        response = client.get("/api/v1/pdet/regions/INVALID_FORMAT")
        assert response.status_code == 400  # Custom validation error handler
    
    def test_invalid_municipality_id_format(self):
        """SIN_CARRETA: Invalid municipality ID format returns validation error"""
        response = client.get("/api/v1/municipalities/INVALID")
        assert response.status_code == 400  # Custom validation error handler
    
    def test_region_id_out_of_range(self):
        """SIN_CARRETA: Region ID out of range returns 404"""
        response = client.get("/api/v1/pdet/regions/REGION_999")
        assert response.status_code == 404
        
        data = response.json()
        assert "error" in data["detail"]


class TestTelemetryHeaders:
    """Test telemetry headers are present"""
    
    def test_request_id_header(self):
        """SIN_CARRETA: Response includes X-Request-ID header"""
        response = client.get("/api/v1/pdet/regions")
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"]
    
    def test_response_time_header(self):
        """SIN_CARRETA: Response includes X-Response-Time-Ms header"""
        response = client.get("/api/v1/pdet/regions")
        assert "X-Response-Time-Ms" in response.headers
        
        # Should be a valid float
        time_ms = float(response.headers["X-Response-Time-Ms"])
        assert time_ms > 0


class TestScoreValidation:
    """Test score validation rules"""
    
    def test_region_scores_in_range(self):
        """SIN_CARRETA: All region scores are in [0, 100]"""
        response = client.get("/api/v1/pdet/regions/REGION_001")
        region = response.json()["region"]
        
        assert 0.0 <= region["overall_score"] <= 100.0
        
        for score in region["dimension_scores"].values():
            assert 0.0 <= score <= 100.0
        
        for score in region["policy_area_scores"].values():
            assert 0.0 <= score <= 100.0
    
    def test_question_scores_in_range(self):
        """SIN_CARRETA: Question scores are in [0, 3]"""
        response = client.get("/api/v1/municipalities/MUN_00101/analysis")
        dimensions = response.json()["dimensions"]
        
        for dim in dimensions:
            for question in dim["questions"]:
                score = question["quantitative_score"]
                assert 0.0 <= score <= 3.0, f"Score {score} out of range"
                
                # Should have 2 decimal places
                assert score == round(score, 2)
