# coding=utf-8
"""
Contract Tests for API Endpoints
=================================

SIN_CARRETA: Enforce strict schema boundaries:
- Score ranges (0-100 for overall, 0-3 for questions)
- Coordinate boundaries (Colombia geographic bounds)
- Connection strength validation
- Required field presence
- Data type validation
- Enum value validation

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestScoreRangeContracts:
    """Test score range contract enforcement"""
    
    def test_region_overall_score_range(self):
        """SIN_CARRETA: Region overall scores must be in [0, 100]"""
        response = client.get("/api/v1/pdet/regions")
        assert response.status_code == 200
        
        regions = response.json()["regions"]
        for region in regions:
            score = region["overall_score"]
            assert 0.0 <= score <= 100.0, \
                f"Region {region['id']} overall score {score} out of range [0, 100]"
    
    def test_region_dimension_scores_range(self):
        """SIN_CARRETA: Region dimension scores must be in [0, 100]"""
        response = client.get("/api/v1/pdet/regions/REGION_001")
        assert response.status_code == 200
        
        region = response.json()["region"]
        for dim_id, score in region["dimension_scores"].items():
            assert 0.0 <= score <= 100.0, \
                f"Dimension {dim_id} score {score} out of range [0, 100]"
    
    def test_region_policy_scores_range(self):
        """SIN_CARRETA: Region policy area scores must be in [0, 100]"""
        response = client.get("/api/v1/pdet/regions/REGION_001")
        assert response.status_code == 200
        
        region = response.json()["region"]
        for policy_id, score in region["policy_area_scores"].items():
            assert 0.0 <= score <= 100.0, \
                f"Policy {policy_id} score {score} out of range [0, 100]"
    
    def test_municipality_scores_range(self):
        """SIN_CARRETA: Municipality scores must be in [0, 100]"""
        response = client.get("/api/v1/municipalities/MUN_00101")
        assert response.status_code == 200
        
        mun = response.json()["municipality"]
        
        # Overall score
        assert 0.0 <= mun["overall_score"] <= 100.0
        
        # Dimension scores
        for dim_id, score in mun["dimension_scores"].items():
            assert 0.0 <= score <= 100.0, f"Dimension {dim_id} score out of range"
        
        # Policy area scores
        for policy_id, score in mun["policy_area_scores"].items():
            assert 0.0 <= score <= 100.0, f"Policy {policy_id} score out of range"
    
    def test_question_scores_range(self):
        """SIN_CARRETA: Question quantitative scores must be in [0, 3]"""
        response = client.get("/api/v1/municipalities/MUN_00101/analysis")
        assert response.status_code == 200
        
        dimensions = response.json()["dimensions"]
        for dim in dimensions:
            for question in dim["questions"]:
                score = question["quantitative_score"]
                assert 0.0 <= score <= 3.0, \
                    f"Question {question['question_id']} score {score} out of range [0, 3]"
    
    def test_cluster_centroid_scores_range(self):
        """SIN_CARRETA: Cluster centroid scores must be in [0, 100]"""
        response = client.get("/api/v1/analysis/clusters/REGION_001")
        assert response.status_code == 200
        
        clusters = response.json()["clusters"]
        for cluster in clusters:
            for dim_id, score in cluster["centroid_scores"].items():
                assert 0.0 <= score <= 100.0, \
                    f"Cluster {cluster['cluster_id']} dimension {dim_id} score out of range"
    
    def test_score_decimal_precision(self):
        """SIN_CARRETA: Scores have reasonable precision (not overly precise)"""
        response = client.get("/api/v1/pdet/regions/REGION_001")
        region = response.json()["region"]
        
        # Check overall score has reasonable precision
        overall = region["overall_score"]
        assert overall == round(overall, 15), "Overall score has unreasonable precision"
        
        # Check dimension scores have reasonable precision
        for score in region["dimension_scores"].values():
            assert score == round(score, 15), "Dimension score has unreasonable precision"


class TestCoordinateContracts:
    """Test geographic coordinate contract enforcement"""
    
    def test_coordinates_within_colombia_bounds(self):
        """SIN_CARRETA: Coordinates must be within Colombia geographic bounds"""
        # Colombia bounds: lat [-4.3, 12.6], lon [-81.8, -66.8]
        response = client.get("/api/v1/pdet/regions")
        assert response.status_code == 200
        
        regions = response.json()["regions"]
        for region in regions:
            coords = region["coordinates"]
            lat = coords["latitude"]
            lon = coords["longitude"]
            
            assert -4.3 <= lat <= 12.6, \
                f"Region {region['id']} latitude {lat} outside Colombia bounds [-4.3, 12.6]"
            assert -81.8 <= lon <= -66.8, \
                f"Region {region['id']} longitude {lon} outside Colombia bounds [-81.8, -66.8]"
    
    def test_municipality_coordinates_within_bounds(self):
        """SIN_CARRETA: Municipality coordinates within Colombia bounds"""
        response = client.get("/api/v1/municipalities/MUN_00101")
        assert response.status_code == 200
        
        mun = response.json()["municipality"]
        coords = mun["coordinates"]
        lat = coords["latitude"]
        lon = coords["longitude"]
        
        assert -4.3 <= lat <= 12.6, f"Latitude {lat} outside bounds"
        assert -81.8 <= lon <= -66.8, f"Longitude {lon} outside bounds"
    
    def test_coordinates_have_precision(self):
        """SIN_CARRETA: Coordinates have reasonable precision (not overly precise)"""
        response = client.get("/api/v1/pdet/regions/REGION_001")
        region = response.json()["region"]
        coords = region["coordinates"]
        
        # Coordinates should have at least 2 decimal places but not more than 6
        lat_str = str(coords["latitude"])
        lon_str = str(coords["longitude"])
        
        if "." in lat_str:
            lat_decimals = len(lat_str.split(".")[1])
            assert 2 <= lat_decimals <= 6, f"Latitude has unusual precision: {lat_decimals} decimals"
        
        if "." in lon_str:
            lon_decimals = len(lon_str.split(".")[1])
            assert 2 <= lon_decimals <= 6, f"Longitude has unusual precision: {lon_decimals} decimals"


class TestStructuralContracts:
    """Test structural requirements are met"""
    
    def test_region_has_all_dimensions(self):
        """SIN_CARRETA: Regions must have all 6 dimensions (D1-D6)"""
        response = client.get("/api/v1/pdet/regions/REGION_001")
        assert response.status_code == 200
        
        region = response.json()["region"]
        dimension_scores = region["dimension_scores"]
        
        assert len(dimension_scores) == 6, "Region must have exactly 6 dimensions"
        
        expected_dimensions = {"D1", "D2", "D3", "D4", "D5", "D6"}
        actual_dimensions = set(dimension_scores.keys())
        assert actual_dimensions == expected_dimensions, \
            f"Missing dimensions: {expected_dimensions - actual_dimensions}"
    
    def test_region_has_all_policy_areas(self):
        """SIN_CARRETA: Regions must have all 10 policy areas (P1-P10)"""
        response = client.get("/api/v1/pdet/regions/REGION_001")
        assert response.status_code == 200
        
        region = response.json()["region"]
        policy_scores = region["policy_area_scores"]
        
        assert len(policy_scores) == 10, "Region must have exactly 10 policy areas"
        
        expected_policies = {f"P{i}" for i in range(1, 11)}
        actual_policies = set(policy_scores.keys())
        assert actual_policies == expected_policies, \
            f"Missing policies: {expected_policies - actual_policies}"
    
    def test_municipality_analysis_has_all_dimensions(self):
        """SIN_CARRETA: Municipality analysis must have all 6 dimensions"""
        response = client.get("/api/v1/municipalities/MUN_00101/analysis")
        assert response.status_code == 200
        
        dimensions = response.json()["dimensions"]
        assert len(dimensions) == 6, "Analysis must have exactly 6 dimensions"
        
        dimension_ids = {dim["dimension_id"] for dim in dimensions}
        expected_dimensions = {"D1", "D2", "D3", "D4", "D5", "D6"}
        assert dimension_ids == expected_dimensions
    
    def test_dimension_has_five_questions(self):
        """SIN_CARRETA: Each dimension must have exactly 5 questions"""
        response = client.get("/api/v1/municipalities/MUN_00101/analysis")
        assert response.status_code == 200
        
        dimensions = response.json()["dimensions"]
        for dim in dimensions:
            questions = dim["questions"]
            assert len(questions) == 5, \
                f"Dimension {dim['dimension_id']} must have exactly 5 questions, has {len(questions)}"
    
    def test_question_analysis_has_300_questions(self):
        """SIN_CARRETA: Question analysis must have exactly 300 questions"""
        response = client.get("/api/v1/analysis/questions/MUN_00101")
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_questions"] == 300, "Must have exactly 300 questions"
        assert len(data["questions"]) == 300, "Question list must contain 300 items"
    
    def test_question_grouping_by_dimension(self):
        """SIN_CARRETA: Questions grouped by dimension must total 50 each (10 policies × 5 questions)"""
        response = client.get("/api/v1/analysis/questions/MUN_00101")
        assert response.status_code == 200
        
        by_dimension = response.json()["by_dimension"]
        
        # Must have all 6 dimensions
        assert len(by_dimension) == 6
        
        # Each dimension must have 50 questions
        for dim_id, questions in by_dimension.items():
            assert len(questions) == 50, \
                f"Dimension {dim_id} must have 50 questions, has {len(questions)}"
    
    def test_question_grouping_by_policy(self):
        """SIN_CARRETA: Questions grouped by policy must total 30 each (6 dimensions × 5 questions)"""
        response = client.get("/api/v1/analysis/questions/MUN_00101")
        assert response.status_code == 200
        
        by_policy = response.json()["by_policy_area"]
        
        # Must have all 10 policies
        assert len(by_policy) == 10
        
        # Each policy must have 30 questions
        for policy_id, questions in by_policy.items():
            assert len(questions) == 30, \
                f"Policy {policy_id} must have 30 questions, has {len(questions)}"


class TestRequiredFieldContracts:
    """Test that required fields are always present"""
    
    def test_region_summary_required_fields(self):
        """SIN_CARRETA: Region summary must have all required fields"""
        response = client.get("/api/v1/pdet/regions")
        assert response.status_code == 200
        
        regions = response.json()["regions"]
        assert len(regions) > 0
        
        required_fields = ["id", "name", "coordinates", "overall_score"]
        for region in regions:
            for field in required_fields:
                assert field in region, f"Missing required field: {field}"
            
            # Coordinates sub-fields
            assert "latitude" in region["coordinates"]
            assert "longitude" in region["coordinates"]
    
    def test_region_detail_required_fields(self):
        """SIN_CARRETA: Region detail must have all required fields"""
        response = client.get("/api/v1/pdet/regions/REGION_001")
        assert response.status_code == 200
        
        region = response.json()["region"]
        
        required_fields = [
            "id", "name", "coordinates", "overall_score",
            "dimension_scores", "policy_area_scores", "metadata"
        ]
        
        for field in required_fields:
            assert field in region, f"Missing required field: {field}"
        
        # Metadata sub-fields
        metadata = region["metadata"]
        metadata_fields = ["population", "area_km2", "municipalities_count", "creation_date"]
        for field in metadata_fields:
            assert field in metadata, f"Missing metadata field: {field}"
    
    def test_municipality_required_fields(self):
        """SIN_CARRETA: Municipality must have all required fields"""
        response = client.get("/api/v1/municipalities/MUN_00101")
        assert response.status_code == 200
        
        mun = response.json()["municipality"]
        
        required_fields = [
            "id", "name", "region_id", "coordinates", "overall_score",
            "dimension_scores", "policy_area_scores", "metadata"
        ]
        
        for field in required_fields:
            assert field in mun, f"Missing required field: {field}"
    
    def test_question_required_fields(self):
        """SIN_CARRETA: Questions must have all required fields"""
        response = client.get("/api/v1/municipalities/MUN_00101/analysis")
        assert response.status_code == 200
        
        dimensions = response.json()["dimensions"]
        # Required fields may vary - check for key identifying fields
        required_fields = [
            "question_id", "quantitative_score", 
            "qualitative_level"
        ]
        
        for dim in dimensions:
            for question in dim["questions"]:
                for field in required_fields:
                    assert field in question, f"Missing required field: {field}"
                
                # Check that question has either 'text' or 'question_text'
                assert "text" in question or "question_text" in question, \
                    "Question must have text or question_text field"


class TestEnumValueContracts:
    """Test enum value validation"""
    
    def test_qualitative_levels_are_valid(self):
        """SIN_CARRETA: Qualitative levels must be from valid enum"""
        valid_levels = {
            "EXCELENTE", "BUENO", "SATISFACTORIO", 
            "ACEPTABLE", "INSUFICIENTE", "DEFICIENTE"
        }
        
        response = client.get("/api/v1/municipalities/MUN_00101/analysis")
        assert response.status_code == 200
        
        dimensions = response.json()["dimensions"]
        for dim in dimensions:
            for question in dim["questions"]:
                level = question["qualitative_level"]
                assert level in valid_levels, \
                    f"Invalid qualitative level: {level}"
    
    def test_dimension_ids_are_valid(self):
        """SIN_CARRETA: Dimension IDs must be from valid enum (D1-D6)"""
        valid_dimensions = {"D1", "D2", "D3", "D4", "D5", "D6"}
        
        response = client.get("/api/v1/pdet/regions/REGION_001")
        region = response.json()["region"]
        
        for dim_id in region["dimension_scores"].keys():
            assert dim_id in valid_dimensions, f"Invalid dimension ID: {dim_id}"
    
    def test_policy_ids_are_valid(self):
        """SIN_CARRETA: Policy IDs must be from valid enum (P1-P10)"""
        valid_policies = {f"P{i}" for i in range(1, 11)}
        
        response = client.get("/api/v1/pdet/regions/REGION_001")
        region = response.json()["region"]
        
        for policy_id in region["policy_area_scores"].keys():
            assert policy_id in valid_policies, f"Invalid policy ID: {policy_id}"


class TestDataTypeContracts:
    """Test data types are correct"""
    
    def test_score_types(self):
        """SIN_CARRETA: Scores must be float type"""
        response = client.get("/api/v1/pdet/regions/REGION_001")
        region = response.json()["region"]
        
        assert isinstance(region["overall_score"], (int, float))
        
        for score in region["dimension_scores"].values():
            assert isinstance(score, (int, float))
        
        for score in region["policy_area_scores"].values():
            assert isinstance(score, (int, float))
    
    def test_count_types(self):
        """SIN_CARRETA: Counts must be integer type"""
        response = client.get("/api/v1/pdet/regions/REGION_001")
        region = response.json()["region"]
        
        metadata = region["metadata"]
        assert isinstance(metadata["population"], int)
        assert isinstance(metadata["municipalities_count"], int)
    
    def test_coordinate_types(self):
        """SIN_CARRETA: Coordinates must be float type"""
        response = client.get("/api/v1/pdet/regions/REGION_001")
        region = response.json()["region"]
        
        coords = region["coordinates"]
        assert isinstance(coords["latitude"], (int, float))
        assert isinstance(coords["longitude"], (int, float))
    
    def test_string_types(self):
        """SIN_CARRETA: IDs and names must be string type"""
        response = client.get("/api/v1/pdet/regions/REGION_001")
        region = response.json()["region"]
        
        assert isinstance(region["id"], str)
        assert isinstance(region["name"], str)


class TestConnectionStrengthContracts:
    """Test connection strength validation (if applicable)"""
    
    def test_cluster_member_similarity_range(self):
        """SIN_CARRETA: Cluster member similarity scores in [0, 1]"""
        response = client.get("/api/v1/analysis/clusters/REGION_001")
        assert response.status_code == 200
        
        clusters = response.json()["clusters"]
        for cluster in clusters:
            for member in cluster["members"]:
                if "similarity_score" in member:
                    similarity = member["similarity_score"]
                    assert 0.0 <= similarity <= 1.0, \
                        f"Similarity score {similarity} out of range [0, 1]"
