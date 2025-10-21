# coding=utf-8
"""
Tests for New API Endpoints
============================

SIN_CARRETA: Test all new visualization, temporal, evidence, and export endpoints
for contract validation, determinism, and telemetry.

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestVisualizationEndpoints:
    """Test visualization endpoints"""
    
    def test_constellation_success(self):
        """SIN_CARRETA: Constellation returns valid data"""
        response = client.get("/api/v1/visualization/constellation")
        assert response.status_code == 200
        
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert "timestamp" in data
        
        # Verify nodes structure
        assert len(data["nodes"]) >= 1
        node = data["nodes"][0]
        assert "region_id" in node
        assert "x" in node and 0 <= node["x"] <= 100
        assert "y" in node and 0 <= node["y"] <= 100
        assert "score" in node and 0 <= node["score"] <= 100
        assert "size" in node and 1 <= node["size"] <= 10
    
    def test_constellation_determinism(self):
        """SIN_CARRETA: Constellation is deterministic"""
        response1 = client.get("/api/v1/visualization/constellation")
        response2 = client.get("/api/v1/visualization/constellation")
        
        nodes1 = response1.json()["nodes"]
        nodes2 = response2.json()["nodes"]
        
        assert nodes1 == nodes2
    
    def test_phylogram_success(self):
        """SIN_CARRETA: Phylogram returns valid data"""
        response = client.get("/api/v1/visualization/phylogram/REGION_001")
        assert response.status_code == 200
        
        data = response.json()
        assert data["region_id"] == "REGION_001"
        assert "nodes" in data
        assert len(data["nodes"]) >= 1
        
        # Verify tree structure
        root = [n for n in data["nodes"] if n["parent_id"] is None][0]
        assert root["depth"] == 0
        assert root["id"] == "REGION_001"
    
    def test_phylogram_not_found(self):
        """SIN_CARRETA: Invalid phylogram region returns 404"""
        response = client.get("/api/v1/visualization/phylogram/REGION_999")
        assert response.status_code == 404
    
    def test_mesh_success(self):
        """SIN_CARRETA: Mesh returns valid 3D data"""
        response = client.get("/api/v1/visualization/mesh/REGION_002")
        assert response.status_code == 200
        
        data = response.json()
        assert data["region_id"] == "REGION_002"
        assert "nodes" in data
        
        # Verify 3D coordinates
        node = data["nodes"][0]
        assert "x" in node and 0 <= node["x"] <= 100
        assert "y" in node and 0 <= node["y"] <= 100
        assert "z" in node and 0 <= node["z"] <= 100
        assert "dimension_scores" in node
        assert len(node["dimension_scores"]) == 6
    
    def test_helix_success(self):
        """SIN_CARRETA: Helix returns 6 dimension points"""
        response = client.get("/api/v1/visualization/helix/MUN_00101")
        assert response.status_code == 200
        
        data = response.json()
        assert data["municipality_id"] == "MUN_00101"
        assert "points" in data
        assert len(data["points"]) == 6
        
        # Verify helix structure
        point = data["points"][0]
        assert "dimension" in point
        assert "angle" in point and 0 <= point["angle"] <= 360
        assert "height" in point and 0 <= point["height"] <= 100
    
    def test_radar_success(self):
        """SIN_CARRETA: Radar returns 10 policy axes"""
        response = client.get("/api/v1/visualization/radar/MUN_00201")
        assert response.status_code == 200
        
        data = response.json()
        assert data["municipality_id"] == "MUN_00201"
        assert "axes" in data
        assert len(data["axes"]) == 10
        
        # Verify radar structure
        axis = data["axes"][0]
        assert "policy_area" in axis
        assert "score" in axis and 0 <= axis["score"] <= 100


class TestTemporalEndpoints:
    """Test temporal endpoints"""
    
    def test_timeline_regions_success(self):
        """SIN_CARRETA: Region timeline returns events"""
        response = client.get("/api/v1/timeline/regions/REGION_003")
        assert response.status_code == 200
        
        data = response.json()
        assert data["region_id"] == "REGION_003"
        assert "events" in data
        assert len(data["events"]) >= 1
        assert "start_date" in data
        assert "end_date" in data
        
        # Verify event structure
        event = data["events"][0]
        assert "timestamp" in event
        assert "event_type" in event
        assert "description" in event
    
    def test_timeline_municipalities_success(self):
        """SIN_CARRETA: Municipality timeline returns events"""
        response = client.get("/api/v1/timeline/municipalities/MUN_00301")
        assert response.status_code == 200
        
        data = response.json()
        assert data["municipality_id"] == "MUN_00301"
        assert "events" in data
        assert len(data["events"]) >= 1
    
    def test_comparison_regions_success(self):
        """SIN_CARRETA: Region comparison returns all regions"""
        response = client.get("/api/v1/comparison/regions")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data
        assert len(data["items"]) >= 2
        
        # Verify comparison item structure
        item = data["items"][0]
        assert "entity_id" in item
        assert "entity_name" in item
        assert "dimension_scores" in item
        assert "overall_score" in item
        assert len(item["dimension_scores"]) == 6
    
    def test_comparison_matrix_success(self):
        """SIN_CARRETA: Comparison matrix returns similarity scores"""
        request_data = {
            "entity_ids": ["REGION_001", "REGION_002", "REGION_003"]
        }
        response = client.post("/api/v1/comparison/matrix", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "entity_ids" in data
        assert "matrix" in data
        assert len(data["entity_ids"]) == 3
        assert len(data["matrix"]) == 9  # 3x3 matrix
        
        # Verify matrix cell structure
        cell = data["matrix"][0]
        assert "row_entity" in cell
        assert "col_entity" in cell
        assert "similarity" in cell
        assert 0 <= cell["similarity"] <= 1
    
    def test_comparison_matrix_validation(self):
        """SIN_CARRETA: Matrix requires at least 2 entities"""
        request_data = {
            "entity_ids": ["REGION_001"]
        }
        response = client.post("/api/v1/comparison/matrix", json=request_data)
        assert response.status_code == 400
    
    def test_historical_data_success(self):
        """SIN_CARRETA: Historical data returns yearly scores"""
        response = client.get("/api/v1/historical/region/REGION_001/years/2018/2023")
        assert response.status_code == 200
        
        data = response.json()
        assert data["entity_type"] == "region"
        assert data["entity_id"] == "REGION_001"
        assert data["start_year"] == 2018
        assert data["end_year"] == 2023
        assert "data_points" in data
        assert len(data["data_points"]) == 6  # 2018-2023 inclusive
        
        # Verify data point structure
        point = data["data_points"][0]
        assert "year" in point
        assert "dimension_scores" in point
        assert "overall_score" in point
        assert len(point["dimension_scores"]) == 6
    
    def test_historical_data_municipality(self):
        """SIN_CARRETA: Historical data works for municipalities"""
        response = client.get("/api/v1/historical/municipality/MUN_00101/years/2020/2023")
        assert response.status_code == 200
        
        data = response.json()
        assert data["entity_type"] == "municipality"
        assert len(data["data_points"]) == 4
    
    def test_historical_data_invalid_range(self):
        """SIN_CARRETA: Invalid year range returns 400"""
        response = client.get("/api/v1/historical/region/REGION_001/years/2023/2018")
        assert response.status_code == 400


class TestEvidenceEndpoints:
    """Test evidence and documents endpoints"""
    
    def test_evidence_stream_success(self):
        """SIN_CARRETA: Evidence stream returns paginated items"""
        response = client.get("/api/v1/evidence/stream?page=1&per_page=20")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "per_page" in data
        assert data["page"] == 1
        assert data["per_page"] == 20
        assert len(data["items"]) <= 20
        
        # Verify evidence item structure
        if data["items"]:
            item = data["items"][0]
            assert "evidence_id" in item
            assert item["evidence_id"].startswith("EV_")
            assert "text" in item
            assert "confidence" in item
            assert 0 <= item["confidence"] <= 1
            assert "timestamp" in item
    
    def test_evidence_stream_pagination(self):
        """SIN_CARRETA: Evidence stream pagination is deterministic"""
        response1 = client.get("/api/v1/evidence/stream?page=1&per_page=10")
        response2 = client.get("/api/v1/evidence/stream?page=1&per_page=10")
        
        items1 = response1.json()["items"]
        items2 = response2.json()["items"]
        
        assert items1 == items2
    
    def test_document_references_success(self):
        """SIN_CARRETA: Document references returns list"""
        response = client.get("/api/v1/documents/references/REGION_001")
        assert response.status_code == 200
        
        data = response.json()
        assert data["region_id"] == "REGION_001"
        assert "references" in data
        assert "total" in data
        assert len(data["references"]) >= 1
        
        # Verify reference structure
        ref = data["references"][0]
        assert "document_id" in ref
        assert ref["document_id"].startswith("DOC_")
        assert "title" in ref
        assert "author" in ref
        assert "date" in ref
    
    def test_document_sources_success(self):
        """SIN_CARRETA: Document sources returns for question"""
        response = client.get("/api/v1/documents/sources/P1-D1-Q1")
        assert response.status_code == 200
        
        data = response.json()
        assert data["question_id"] == "P1-D1-Q1"
        assert "sources" in data
        assert "total" in data
        
        # Verify source structure
        if data["sources"]:
            source = data["sources"][0]
            assert "source_id" in source
            assert source["source_id"].startswith("SRC_")
            assert "excerpt" in source
            assert "relevance" in source
            assert 0 <= source["relevance"] <= 1
    
    def test_citations_success(self):
        """SIN_CARRETA: Citations returns formatted citations"""
        response = client.get("/api/v1/citations/IND_001")
        assert response.status_code == 200
        
        data = response.json()
        assert data["indicator_id"] == "IND_001"
        assert "citations" in data
        assert "total" in data
        
        # Verify citation structure
        if data["citations"]:
            cit = data["citations"][0]
            assert "citation_id" in cit
            assert cit["citation_id"].startswith("CIT_")
            assert "citation_format" in cit
            assert "year" in cit
            assert 2000 <= cit["year"] <= 2030


class TestExportEndpoints:
    """Test export and reporting endpoints"""
    
    def test_export_dashboard_success(self):
        """SIN_CARRETA: Dashboard export returns export info"""
        request_data = {
            "format": "pdf",
            "include_visualizations": True,
            "include_raw_data": False
        }
        response = client.post("/api/v1/export/dashboard", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "export_id" in data
        assert data["export_id"].startswith("EXP_")
        assert "format" in data
        assert data["format"] == "pdf"
        assert "download_url" in data
        assert "expires_at" in data
        assert "size_bytes" in data
        assert data["size_bytes"] > 0
    
    def test_export_region_success(self):
        """SIN_CARRETA: Region export returns export info"""
        request_data = {
            "format": "xlsx",
            "include_municipalities": True,
            "include_analysis": True
        }
        response = client.post("/api/v1/export/region/REGION_001", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["export_id"].startswith("EXP_")
        assert data["format"] == "xlsx"
    
    def test_export_comparison_success(self):
        """SIN_CARRETA: Comparison export returns export info"""
        request_data = {
            "entity_ids": ["REGION_001", "REGION_002"],
            "format": "csv"
        }
        response = client.post("/api/v1/export/comparison", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["export_id"].startswith("EXP_")
        assert data["format"] == "csv"
    
    def test_report_generate_success(self):
        """SIN_CARRETA: Standard report generation works"""
        response = client.get("/api/v1/reports/generate/executive_summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "report_id" in data
        assert data["report_id"].startswith("RPT_")
        assert "report_type" in data
        assert data["report_type"] == "executive_summary"
        assert "download_url" in data
        assert "expires_at" in data
        assert "size_bytes" in data
    
    def test_report_custom_success(self):
        """SIN_CARRETA: Custom report generation works"""
        request_data = {
            "title": "Custom Analysis Report",
            "entity_ids": ["REGION_001", "REGION_002"],
            "sections": ["overview", "analysis", "recommendations"],
            "format": "pdf"
        }
        response = client.post("/api/v1/reports/custom", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["report_id"].startswith("RPT_")
        assert data["report_type"] == "custom"


class TestContractValidation:
    """Test strict contract validation for new endpoints"""
    
    def test_constellation_coordinates_in_range(self):
        """SIN_CARRETA: Constellation coordinates are in [0, 100]"""
        response = client.get("/api/v1/visualization/constellation")
        nodes = response.json()["nodes"]
        
        for node in nodes:
            assert 0 <= node["x"] <= 100
            assert 0 <= node["y"] <= 100
    
    def test_comparison_matrix_similarity_in_range(self):
        """SIN_CARRETA: Matrix similarity scores are in [0, 1]"""
        request_data = {
            "entity_ids": ["REGION_001", "REGION_002"]
        }
        response = client.post("/api/v1/comparison/matrix", json=request_data)
        matrix = response.json()["matrix"]
        
        for cell in matrix:
            assert 0 <= cell["similarity"] <= 1
    
    def test_evidence_confidence_in_range(self):
        """SIN_CARRETA: Evidence confidence scores are in [0, 1]"""
        response = client.get("/api/v1/evidence/stream?page=1&per_page=10")
        items = response.json()["items"]
        
        for item in items:
            assert 0 <= item["confidence"] <= 1
    
    def test_historical_years_in_range(self):
        """SIN_CARRETA: Historical years are in [2016, 2030]"""
        response = client.get("/api/v1/historical/region/REGION_001/years/2016/2030")
        points = response.json()["data_points"]
        
        for point in points:
            assert 2016 <= point["year"] <= 2030


class TestDeterminism:
    """Test determinism of new endpoints"""
    
    def test_phylogram_determinism(self):
        """SIN_CARRETA: Phylogram is deterministic"""
        response1 = client.get("/api/v1/visualization/phylogram/REGION_002")
        response2 = client.get("/api/v1/visualization/phylogram/REGION_002")
        
        nodes1 = response1.json()["nodes"]
        nodes2 = response2.json()["nodes"]
        
        assert nodes1 == nodes2
    
    def test_timeline_determinism(self):
        """SIN_CARRETA: Timeline is deterministic"""
        response1 = client.get("/api/v1/timeline/regions/REGION_003")
        response2 = client.get("/api/v1/timeline/regions/REGION_003")
        
        events1 = response1.json()["events"]
        events2 = response2.json()["events"]
        
        assert events1 == events2
    
    def test_export_determinism(self):
        """SIN_CARRETA: Export IDs are deterministic"""
        request_data = {
            "format": "pdf",
            "include_visualizations": True,
            "include_raw_data": False
        }
        response1 = client.post("/api/v1/export/dashboard", json=request_data)
        response2 = client.post("/api/v1/export/dashboard", json=request_data)
        
        # Export IDs should be deterministic
        assert response1.json()["export_id"] == response2.json()["export_id"]


class TestTelemetryNewEndpoints:
    """Test telemetry for new endpoints"""
    
    def test_visualization_telemetry_headers(self):
        """SIN_CARRETA: Visualization endpoints include telemetry"""
        response = client.get("/api/v1/visualization/constellation")
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time-Ms" in response.headers
    
    def test_temporal_telemetry_headers(self):
        """SIN_CARRETA: Temporal endpoints include telemetry"""
        response = client.get("/api/v1/timeline/regions/REGION_001")
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time-Ms" in response.headers
    
    def test_evidence_telemetry_headers(self):
        """SIN_CARRETA: Evidence endpoints include telemetry"""
        response = client.get("/api/v1/evidence/stream")
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time-Ms" in response.headers
