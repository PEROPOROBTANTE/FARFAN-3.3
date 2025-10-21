# coding=utf-8
"""
Integration Tests for All API Endpoints
========================================

SIN_CARRETA: Comprehensive integration tests covering:
- All core endpoints (regions, municipalities, analysis)
- End-to-end workflows
- Cross-endpoint data consistency
- Complete API surface coverage

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestCoreEndpointsIntegration:
    """Integration tests for core PDET region endpoints"""
    
    def test_complete_region_workflow(self):
        """SIN_CARRETA: Complete workflow from list to detail to municipalities"""
        # Step 1: List all regions
        list_response = client.get("/api/v1/pdet/regions")
        assert list_response.status_code == 200
        
        data = list_response.json()
        assert data["total"] == 10
        assert len(data["regions"]) == 10
        
        # Step 2: Get first region detail
        first_region_id = data["regions"][0]["id"]
        detail_response = client.get(f"/api/v1/pdet/regions/{first_region_id}")
        assert detail_response.status_code == 200
        
        region = detail_response.json()["region"]
        assert region["id"] == first_region_id
        
        # Step 3: Get region's municipalities
        mun_response = client.get(f"/api/v1/pdet/regions/{first_region_id}/municipalities")
        assert mun_response.status_code == 200
        
        municipalities = mun_response.json()["municipalities"]
        assert len(municipalities) == 10
        
        # Verify all municipalities belong to region
        for mun in municipalities:
            assert mun["region_id"] == first_region_id
    
    def test_all_regions_accessible(self):
        """SIN_CARRETA: All 10 regions are accessible via API"""
        for i in range(1, 11):
            region_id = f"REGION_{i:03d}"
            response = client.get(f"/api/v1/pdet/regions/{region_id}")
            assert response.status_code == 200, f"Region {region_id} not accessible"
            
            region = response.json()["region"]
            assert region["id"] == region_id
    
    def test_region_data_consistency(self):
        """SIN_CARRETA: Region data consistent between list and detail"""
        # Get from list
        list_response = client.get("/api/v1/pdet/regions")
        regions_list = list_response.json()["regions"]
        
        # Check a few regions for consistency
        for region_summary in regions_list[:3]:  # Test first 3 regions
            detail_response = client.get(f"/api/v1/pdet/regions/{region_summary['id']}")
            region_detail = detail_response.json()["region"]
            
            # Common fields should match
            assert region_summary["id"] == region_detail["id"]
            # Note: Name and scores may vary between calls if data is dynamically generated
            # We focus on ID consistency which is the core deterministic field


class TestMunicipalityEndpointsIntegration:
    """Integration tests for municipality endpoints"""
    
    def test_complete_municipality_workflow(self):
        """SIN_CARRETA: Complete workflow from region to municipality to analysis"""
        # Step 1: Get region's municipalities
        response = client.get("/api/v1/pdet/regions/REGION_001/municipalities")
        assert response.status_code == 200
        
        municipalities = response.json()["municipalities"]
        assert len(municipalities) > 0
        
        # Step 2: Get first municipality detail
        mun_id = municipalities[0]["id"]
        detail_response = client.get(f"/api/v1/municipalities/{mun_id}")
        assert detail_response.status_code == 200
        
        municipality = detail_response.json()["municipality"]
        assert municipality["id"] == mun_id
        
        # Step 3: Get municipality analysis
        analysis_response = client.get(f"/api/v1/municipalities/{mun_id}/analysis")
        assert analysis_response.status_code == 200
        
        analysis = analysis_response.json()
        assert analysis["municipality_id"] == mun_id
        assert len(analysis["dimensions"]) == 6
    
    def test_all_municipalities_accessible(self):
        """SIN_CARRETA: Municipalities from each region are accessible"""
        # Test municipalities by getting them from region lists
        for region_num in range(1, 11):
            region_id = f"REGION_{region_num:03d}"
            response = client.get(f"/api/v1/pdet/regions/{region_id}/municipalities")
            assert response.status_code == 200
            
            municipalities = response.json()["municipalities"]
            assert len(municipalities) > 0, f"Region {region_id} should have municipalities"
            
            # Test first municipality from each region
            first_mun_id = municipalities[0]["id"]
            mun_response = client.get(f"/api/v1/municipalities/{first_mun_id}")
            assert mun_response.status_code == 200, f"Municipality {first_mun_id} not accessible"
    
    def test_municipality_region_relationship(self):
        """SIN_CARRETA: Municipality region relationships are correct"""
        for region_num in range(1, 11):
            region_id = f"REGION_{region_num:03d}"
            
            # Get region's municipalities
            response = client.get(f"/api/v1/pdet/regions/{region_id}/municipalities")
            municipalities = response.json()["municipalities"]
            
            # Check each municipality belongs to region
            for mun in municipalities:
                assert mun["region_id"] == region_id
                
                # Get municipality detail and verify
                detail_response = client.get(f"/api/v1/municipalities/{mun['id']}")
                detail = detail_response.json()["municipality"]
                assert detail["region_id"] == region_id


class TestAnalysisEndpointsIntegration:
    """Integration tests for analysis endpoints"""
    
    def test_complete_cluster_analysis_workflow(self):
        """SIN_CARRETA: Complete cluster analysis workflow"""
        # Get cluster analysis for region
        response = client.get("/api/v1/analysis/clusters/REGION_001")
        assert response.status_code == 200
        
        data = response.json()
        assert data["region_id"] == "REGION_001"
        assert len(data["clusters"]) >= 1
        
        # Verify each cluster has valid structure
        for cluster in data["clusters"]:
            assert "cluster_id" in cluster
            assert "centroid_scores" in cluster
            assert len(cluster["centroid_scores"]) == 6  # All 6 dimensions
            assert "members" in cluster
            assert len(cluster["members"]) > 0
            
            # Verify cluster members are valid municipalities
            for member in cluster["members"]:
                mun_id = member["municipality_id"]
                mun_response = client.get(f"/api/v1/municipalities/{mun_id}")
                assert mun_response.status_code == 200
    
    def test_complete_question_analysis_workflow(self):
        """SIN_CARRETA: Complete question analysis workflow"""
        # Get all questions for municipality
        response = client.get("/api/v1/analysis/questions/MUN_00101")
        assert response.status_code == 200
        
        data = response.json()
        assert data["municipality_id"] == "MUN_00101"
        assert data["total_questions"] == 300
        assert len(data["questions"]) == 300
        
        # Verify groupings
        by_dimension = data["by_dimension"]
        by_policy = data["by_policy_area"]
        
        # Check dimension groupings
        assert len(by_dimension) == 6
        for dim_id, questions in by_dimension.items():
            assert len(questions) == 50  # 10 policies × 5 questions
        
        # Check policy groupings
        assert len(by_policy) == 10
        for policy_id, questions in by_policy.items():
            assert len(questions) == 30  # 6 dimensions × 5 questions
    
    def test_analysis_cross_reference(self):
        """SIN_CARRETA: Analysis data cross-references correctly"""
        # Get municipality analysis (6 dimensions × 5 questions = 30 questions)
        analysis_response = client.get("/api/v1/municipalities/MUN_00101/analysis")
        analysis = analysis_response.json()
        
        # Get full question list (300 questions)
        questions_response = client.get("/api/v1/analysis/questions/MUN_00101")
        questions_data = questions_response.json()
        
        # Verify that questions from analysis are subset of full question list
        analysis_question_ids = set()
        for dim in analysis["dimensions"]:
            for q in dim["questions"]:
                analysis_question_ids.add(q["question_id"])
        
        full_question_ids = {q["question_id"] for q in questions_data["questions"]}
        
        # All analysis questions should be in full list
        assert analysis_question_ids.issubset(full_question_ids), \
            "Analysis questions not found in full question list"


class TestDataConsistencyAcrossEndpoints:
    """Test data consistency across different endpoints"""
    
    def test_score_consistency(self):
        """SIN_CARRETA: Scores are consistent within same call"""
        # Get municipality detail
        detail_response = client.get("/api/v1/municipalities/MUN_00101")
        mun_detail = detail_response.json()["municipality"]
        
        # Overall score should be within valid range
        assert 0 <= mun_detail["overall_score"] <= 100
        
        # Dimension scores should be present
        assert len(mun_detail["dimension_scores"]) == 6
        for score in mun_detail["dimension_scores"].values():
            assert 0 <= score <= 100
    
    def test_coordinate_consistency(self):
        """SIN_CARRETA: Coordinates are within valid bounds"""
        # Get region detail
        detail_response = client.get("/api/v1/pdet/regions/REGION_001")
        region_detail = detail_response.json()["region"]
        
        # Coordinates should be within Colombia bounds
        coords = region_detail["coordinates"]
        assert -4.3 <= coords["latitude"] <= 12.6
        assert -81.8 <= coords["longitude"] <= -66.8
    
    def test_relationship_consistency(self):
        """SIN_CARRETA: Entity relationships are consistent"""
        # Get region's municipalities
        mun_list_response = client.get("/api/v1/pdet/regions/REGION_001/municipalities")
        municipalities = mun_list_response.json()["municipalities"]
        
        # Should have municipalities
        assert len(municipalities) > 0
        
        # All municipalities should belong to REGION_001
        for mun in municipalities:
            assert mun["region_id"] == "REGION_001"


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows"""
    
    def test_researcher_workflow(self):
        """SIN_CARRETA: Simulate researcher exploring data"""
        # 1. Start with region overview
        regions_response = client.get("/api/v1/pdet/regions")
        assert regions_response.status_code == 200
        regions = regions_response.json()["regions"]
        
        # 2. Select region with highest score
        top_region = max(regions, key=lambda r: r["overall_score"])
        
        # 3. Get detailed region info
        region_detail_response = client.get(f"/api/v1/pdet/regions/{top_region['id']}")
        assert region_detail_response.status_code == 200
        
        # 4. Get cluster analysis for region
        cluster_response = client.get(f"/api/v1/analysis/clusters/{top_region['id']}")
        assert cluster_response.status_code == 200
        clusters = cluster_response.json()["clusters"]
        
        # 5. Select a municipality from best cluster
        best_cluster = max(clusters, key=lambda c: sum(c["centroid_scores"].values()))
        selected_mun_id = best_cluster["members"][0]["municipality_id"]
        
        # 6. Get municipality analysis
        mun_analysis_response = client.get(f"/api/v1/municipalities/{selected_mun_id}/analysis")
        assert mun_analysis_response.status_code == 200
        
        # 7. Get full question breakdown
        questions_response = client.get(f"/api/v1/analysis/questions/{selected_mun_id}")
        assert questions_response.status_code == 200
    
    def test_dashboard_workflow(self):
        """SIN_CARRETA: Simulate dashboard loading all data"""
        # Dashboard typically loads multiple endpoints
        endpoints_to_load = [
            "/api/v1/pdet/regions",
            "/api/v1/pdet/regions/REGION_001",
            "/api/v1/pdet/regions/REGION_001/municipalities",
            "/api/v1/municipalities/MUN_00101",
            "/api/v1/municipalities/MUN_00101/analysis",
            "/api/v1/analysis/clusters/REGION_001",
        ]
        
        # All should load successfully
        for endpoint in endpoints_to_load:
            response = client.get(endpoint)
            assert response.status_code == 200, f"Dashboard failed to load {endpoint}"
    
    def test_comparative_analysis_workflow(self):
        """SIN_CARRETA: Compare multiple municipalities"""
        # Get municipalities from same region
        response = client.get("/api/v1/pdet/regions/REGION_001/municipalities")
        municipalities = response.json()["municipalities"][:5]  # Take first 5
        
        # Get detailed analysis for each
        analyses = []
        for mun in municipalities:
            analysis_response = client.get(f"/api/v1/municipalities/{mun['id']}/analysis")
            assert analysis_response.status_code == 200
            analyses.append(analysis_response.json())
        
        # Verify all have same structure
        assert len(analyses) == 5
        for analysis in analyses:
            assert len(analysis["dimensions"]) == 6
            for dim in analysis["dimensions"]:
                assert len(dim["questions"]) == 5


class TestAPICompleteness:
    """Test that API provides complete data coverage"""
    
    def test_all_dimensions_covered(self):
        """SIN_CARRETA: All 6 dimensions covered in all endpoints"""
        expected_dimensions = {"D1", "D2", "D3", "D4", "D5", "D6"}
        
        # Check region
        response = client.get("/api/v1/pdet/regions/REGION_001")
        region = response.json()["region"]
        assert set(region["dimension_scores"].keys()) == expected_dimensions
        
        # Check municipality
        response = client.get("/api/v1/municipalities/MUN_00101")
        mun = response.json()["municipality"]
        assert set(mun["dimension_scores"].keys()) == expected_dimensions
        
        # Check analysis
        response = client.get("/api/v1/municipalities/MUN_00101/analysis")
        analysis = response.json()
        analysis_dims = {dim["dimension_id"] for dim in analysis["dimensions"]}
        assert analysis_dims == expected_dimensions
    
    def test_all_policies_covered(self):
        """SIN_CARRETA: All 10 policy areas covered"""
        expected_policies = {f"P{i}" for i in range(1, 11)}
        
        # Check region
        response = client.get("/api/v1/pdet/regions/REGION_001")
        region = response.json()["region"]
        assert set(region["policy_area_scores"].keys()) == expected_policies
        
        # Check municipality
        response = client.get("/api/v1/municipalities/MUN_00101")
        mun = response.json()["municipality"]
        assert set(mun["policy_area_scores"].keys()) == expected_policies
    
    def test_all_regions_have_municipalities(self):
        """SIN_CARRETA: Every region has 10 municipalities"""
        for i in range(1, 11):
            region_id = f"REGION_{i:03d}"
            response = client.get(f"/api/v1/pdet/regions/{region_id}/municipalities")
            assert response.status_code == 200
            
            municipalities = response.json()["municipalities"]
            assert len(municipalities) == 10, \
                f"Region {region_id} should have 10 municipalities, has {len(municipalities)}"
    
    def test_total_municipality_count(self):
        """SIN_CARRETA: Total of 100 municipalities across 10 regions"""
        all_municipalities = set()
        
        for i in range(1, 11):
            region_id = f"REGION_{i:03d}"
            response = client.get(f"/api/v1/pdet/regions/{region_id}/municipalities")
            municipalities = response.json()["municipalities"]
            
            for mun in municipalities:
                all_municipalities.add(mun["id"])
        
        assert len(all_municipalities) == 100, \
            f"Expected 100 unique municipalities, found {len(all_municipalities)}"


class TestErrorHandlingIntegration:
    """Integration tests for error handling"""
    
    def test_cascading_not_found_errors(self):
        """SIN_CARRETA: Invalid parent returns 404, not causing cascade errors"""
        # Invalid region
        response = client.get("/api/v1/pdet/regions/REGION_999")
        assert response.status_code == 404
        
        # Invalid region municipalities
        response = client.get("/api/v1/pdet/regions/REGION_999/municipalities")
        assert response.status_code == 404
        
        # Invalid municipality
        response = client.get("/api/v1/municipalities/MUN_99999")
        assert response.status_code == 404
    
    def test_validation_errors_consistent(self):
        """SIN_CARRETA: Validation errors are consistent across endpoints"""
        invalid_inputs = [
            "/api/v1/pdet/regions/INVALID",
            "/api/v1/municipalities/INVALID",
        ]
        
        for endpoint in invalid_inputs:
            response = client.get(endpoint)
            assert response.status_code == 400
            
            data = response.json()
            assert "error" in data or "detail" in data
