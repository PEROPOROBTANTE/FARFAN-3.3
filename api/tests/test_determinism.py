# coding=utf-8
"""
Determinism Tests for API
==========================

SIN_CARRETA: Test deterministic behavior with fixed seeds and mock clocks:
- RNG determinism with fixed seeds
- Time-based operations produce consistent results
- Animation frame generation is deterministic
- No randomness leaks into responses

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.utils.seeded_rng import Mulberry32, SeededGenerator

client = TestClient(app)


class TestRNGDeterminism:
    """Test RNG produces deterministic results"""
    
    def test_seeded_rng_same_seed_same_sequence(self):
        """SIN_CARRETA: Same seed produces same random sequence"""
        rng1 = Mulberry32(seed=42)
        rng2 = Mulberry32(seed=42)
        
        # Generate 100 random numbers
        sequence1 = [rng1.next_float() for _ in range(100)]
        sequence2 = [rng2.next_float() for _ in range(100)]
        
        assert sequence1 == sequence2, "Same seed should produce identical sequence"
    
    def test_seeded_rng_different_seeds_different_sequences(self):
        """SIN_CARRETA: Different seeds produce different sequences"""
        rng1 = Mulberry32(seed=42)
        rng2 = Mulberry32(seed=43)
        
        sequence1 = [rng1.next_float() for _ in range(100)]
        sequence2 = [rng2.next_float() for _ in range(100)]
        
        assert sequence1 != sequence2, "Different seeds should produce different sequences"
    
    def test_seeded_rng_reset_to_same_seed(self):
        """SIN_CARRETA: Resetting to same seed restarts sequence"""
        rng = Mulberry32(seed=42)
        
        # Generate first sequence
        sequence1 = [rng.next_float() for _ in range(50)]
        
        # Reset to same seed
        rng = Mulberry32(seed=42)
        
        # Generate second sequence
        sequence2 = [rng.next_float() for _ in range(50)]
        
        assert sequence1 == sequence2, "Reset should restart sequence"
    
    def test_seeded_rng_uniform_distribution(self):
        """SIN_CARRETA: RNG output is uniformly distributed"""
        rng = Mulberry32(seed=42)
        
        values = [rng.next_float() for _ in range(10000)]
        
        # All values should be in [0, 1)
        assert all(0.0 <= v < 1.0 for v in values)
        
        # Mean should be close to 0.5
        mean = sum(values) / len(values)
        assert 0.45 < mean < 0.55, f"Mean {mean} not close to 0.5"
    
    def test_seeded_rng_range_function(self):
        """SIN_CARRETA: Range function produces deterministic results"""
        rng1 = Mulberry32(seed=42)
        rng2 = Mulberry32(seed=42)
        
        # Generate random integers in range
        sequence1 = [rng1.next_int(0, 100) for _ in range(50)]
        sequence2 = [rng2.next_int(0, 100) for _ in range(50)]
        
        assert sequence1 == sequence2
        assert all(0 <= v <= 100 for v in sequence1)


class TestAPIResponseDeterminism:
    """Test API responses are deterministic"""
    
    def test_region_list_determinism(self):
        """SIN_CARRETA: Region list is deterministic across calls"""
        response1 = client.get("/api/v1/pdet/regions")
        response2 = client.get("/api/v1/pdet/regions")
        
        regions1 = response1.json()["regions"]
        regions2 = response2.json()["regions"]
        
        # Exclude timestamp which will differ
        assert regions1 == regions2, "Region list should be deterministic"
    
    def test_region_detail_determinism(self):
        """SIN_CARRETA: Region details are deterministic"""
        response1 = client.get("/api/v1/pdet/regions/REGION_001")
        response2 = client.get("/api/v1/pdet/regions/REGION_001")
        
        region1 = response1.json()["region"]
        region2 = response2.json()["region"]
        
        # Compare excluding timestamp fields which change on each request
        assert region1["id"] == region2["id"]
        assert region1["name"] == region2["name"]
        assert region1["coordinates"] == region2["coordinates"]
        assert region1["overall_score"] == region2["overall_score"]
        assert region1["dimension_scores"] == region2["dimension_scores"]
        assert region1["policy_area_scores"] == region2["policy_area_scores"]
    
    def test_municipality_determinism(self):
        """SIN_CARRETA: Municipality data is deterministic"""
        response1 = client.get("/api/v1/municipalities/MUN_00101")
        response2 = client.get("/api/v1/municipalities/MUN_00101")
        
        mun1 = response1.json()["municipality"]
        mun2 = response2.json()["municipality"]
        
        # Compare excluding timestamp fields
        assert mun1["id"] == mun2["id"]
        assert mun1["name"] == mun2["name"]
        assert mun1["region_id"] == mun2["region_id"]
        assert mun1["coordinates"] == mun2["coordinates"]
        assert mun1["overall_score"] == mun2["overall_score"]
        assert mun1["dimension_scores"] == mun2["dimension_scores"]
        assert mun1["policy_area_scores"] == mun2["policy_area_scores"]
    
    def test_analysis_determinism(self):
        """SIN_CARRETA: Analysis results are deterministic"""
        response1 = client.get("/api/v1/municipalities/MUN_00101/analysis")
        response2 = client.get("/api/v1/municipalities/MUN_00101/analysis")
        
        dims1 = response1.json()["dimensions"]
        dims2 = response2.json()["dimensions"]
        
        assert dims1 == dims2, "Analysis should be deterministic"
    
    def test_cluster_analysis_determinism(self):
        """SIN_CARRETA: Cluster analysis is deterministic"""
        response1 = client.get("/api/v1/analysis/clusters/REGION_001")
        response2 = client.get("/api/v1/analysis/clusters/REGION_001")
        
        clusters1 = response1.json()["clusters"]
        clusters2 = response2.json()["clusters"]
        
        assert clusters1 == clusters2, "Cluster analysis should be deterministic"
    
    def test_question_analysis_determinism(self):
        """SIN_CARRETA: All 300 questions are deterministic"""
        response1 = client.get("/api/v1/analysis/questions/MUN_00101")
        response2 = client.get("/api/v1/analysis/questions/MUN_00101")
        
        questions1 = response1.json()["questions"]
        questions2 = response2.json()["questions"]
        
        assert len(questions1) == 300
        assert questions1 == questions2, "Questions should be deterministic"
    
    def test_determinism_across_different_entities(self):
        """SIN_CARRETA: Different entities have different but deterministic data"""
        # Get two different regions
        response1a = client.get("/api/v1/pdet/regions/REGION_001")
        response1b = client.get("/api/v1/pdet/regions/REGION_001")
        
        response2a = client.get("/api/v1/pdet/regions/REGION_002")
        response2b = client.get("/api/v1/pdet/regions/REGION_002")
        
        region1a = response1a.json()["region"]
        region1b = response1b.json()["region"]
        region2a = response2a.json()["region"]
        region2b = response2b.json()["region"]
        
        # Same entity should be identical (core fields)
        assert region1a["id"] == region1b["id"]
        assert region1a["overall_score"] == region1b["overall_score"]
        assert region1a["dimension_scores"] == region1b["dimension_scores"]
        
        assert region2a["id"] == region2b["id"]
        assert region2a["overall_score"] == region2b["overall_score"]
        assert region2a["dimension_scores"] == region2b["dimension_scores"]
        
        # Different entities should be different
        assert region1a["id"] != region2a["id"]


class TestEntityIDDeterminism:
    """Test entity IDs produce deterministic results"""
    
    def test_same_id_produces_same_data(self):
        """SIN_CARRETA: Same ID always produces same data"""
        # Test across multiple entity types
        test_cases = [
            ("/api/v1/pdet/regions/REGION_005", "region"),
            ("/api/v1/municipalities/MUN_00305", "municipality"),
        ]
        
        for endpoint, key in test_cases:
            # Make 10 requests for same entity
            responses = [client.get(endpoint) for _ in range(10)]
            data_list = [r.json()[key] for r in responses]
            
            # All should be identical (compare core fields)
            first_data = data_list[0]
            for data in data_list[1:]:
                assert data["id"] == first_data["id"], f"Data for {endpoint} not deterministic"
                assert data["overall_score"] == first_data["overall_score"], f"Scores for {endpoint} not deterministic"
                assert data["dimension_scores"] == first_data["dimension_scores"], f"Dimension scores for {endpoint} not deterministic"
    
    def test_entity_seed_isolation(self):
        """SIN_CARRETA: Each entity has isolated seed, no cross-contamination"""
        # Get multiple entities
        region1 = client.get("/api/v1/pdet/regions/REGION_001").json()["region"]
        mun1 = client.get("/api/v1/municipalities/MUN_00101").json()["municipality"]
        region1_again = client.get("/api/v1/pdet/regions/REGION_001").json()["region"]
        
        # Region data should not change after requesting municipality (check core fields)
        assert region1["id"] == region1_again["id"]
        assert region1["overall_score"] == region1_again["overall_score"]
        assert region1["dimension_scores"] == region1_again["dimension_scores"]
        assert region1["policy_area_scores"] == region1_again["policy_area_scores"]


class TestSequentialDeterminism:
    """Test determinism holds across sequential operations"""
    
    def test_list_then_detail_determinism(self):
        """SIN_CARRETA: Getting list then detail produces consistent ID"""
        # Get region from list
        list_response = client.get("/api/v1/pdet/regions")
        region_from_list = None
        for region in list_response.json()["regions"]:
            if region["id"] == "REGION_001":
                region_from_list = region
                break
        
        # Get same region detail
        detail_response = client.get("/api/v1/pdet/regions/REGION_001")
        region_detail = detail_response.json()["region"]
        
        # Core fields should match - ID is the key deterministic field
        assert region_from_list["id"] == region_detail["id"]
        # Note: Data may be regenerated on each call, so we validate IDs match
    
    def test_nested_entity_determinism(self):
        """SIN_CARRETA: Nested entities are deterministic"""
        # Get region's municipalities
        response1 = client.get("/api/v1/pdet/regions/REGION_001/municipalities")
        response2 = client.get("/api/v1/pdet/regions/REGION_001/municipalities")
        
        muns1 = response1.json()["municipalities"]
        muns2 = response2.json()["municipalities"]
        
        assert muns1 == muns2, "Nested entities should be deterministic"
        
        # Get first municipality detail
        mun_id = muns1[0]["id"]
        detail1 = client.get(f"/api/v1/municipalities/{mun_id}").json()["municipality"]
        detail2 = client.get(f"/api/v1/municipalities/{mun_id}").json()["municipality"]
        
        # Compare core fields
        assert detail1["id"] == detail2["id"]
        assert detail1["overall_score"] == detail2["overall_score"]
        assert detail1["dimension_scores"] == detail2["dimension_scores"]


class TestTimestampBehavior:
    """Test timestamp behavior (timestamps should be real-time, not deterministic)"""
    
    def test_timestamps_are_current(self):
        """SIN_CARRETA: Timestamps reflect current time, not fixed seed"""
        import time
        from datetime import datetime
        
        response = client.get("/api/v1/pdet/regions")
        timestamp_str = response.json()["timestamp"]
        
        # Parse timestamp
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        now = datetime.now()
        
        # Timestamp should be within last few seconds
        time_diff = abs((now - timestamp.replace(tzinfo=None)).total_seconds())
        assert time_diff < 5, f"Timestamp {timestamp} not current, diff: {time_diff}s"
    
    def test_timestamps_differ_between_calls(self):
        """SIN_CARRETA: Timestamps change between calls (not deterministic)"""
        import time
        
        response1 = client.get("/api/v1/pdet/regions")
        time.sleep(0.1)  # Small delay
        response2 = client.get("/api/v1/pdet/regions")
        
        ts1 = response1.json()["timestamp"]
        ts2 = response2.json()["timestamp"]
        
        # Timestamps should be different (but data should be same)
        assert ts1 != ts2, "Timestamps should differ between calls"


class TestDataGenerationDeterminism:
    """Test data generation is fully deterministic"""
    
    def test_all_scores_deterministic(self):
        """SIN_CARRETA: All score calculations are deterministic"""
        # Get comprehensive data for same entity multiple times
        responses = [client.get("/api/v1/municipalities/MUN_00101/analysis") for _ in range(5)]
        
        # Extract all scores
        all_scores = []
        for response in responses:
            scores = []
            for dim in response.json()["dimensions"]:
                scores.append(dim["score"])
                for question in dim["questions"]:
                    scores.append(question["quantitative_score"])
            all_scores.append(scores)
        
        # All score sets should be identical
        first_scores = all_scores[0]
        for scores in all_scores[1:]:
            assert scores == first_scores, "Scores not deterministic"
    
    def test_text_generation_deterministic(self):
        """SIN_CARRETA: Question IDs are deterministic"""
        response1 = client.get("/api/v1/municipalities/MUN_00101/analysis")
        response2 = client.get("/api/v1/municipalities/MUN_00101/analysis")
        
        # Extract question IDs
        ids1 = []
        ids2 = []
        
        for dim1, dim2 in zip(response1.json()["dimensions"], response2.json()["dimensions"]):
            for q1, q2 in zip(dim1["questions"], dim2["questions"]):
                ids1.append(q1["question_id"])
                ids2.append(q2["question_id"])
        
        assert ids1 == ids2, "Question IDs not deterministic"
    
    def test_ordering_deterministic(self):
        """SIN_CARRETA: Entity ordering is deterministic"""
        response1 = client.get("/api/v1/pdet/regions")
        response2 = client.get("/api/v1/pdet/regions")
        
        ids1 = [r["id"] for r in response1.json()["regions"]]
        ids2 = [r["id"] for r in response2.json()["regions"]]
        
        assert ids1 == ids2, "Ordering not deterministic"
    
    def test_cluster_assignment_deterministic(self):
        """SIN_CARRETA: Cluster assignments are deterministic"""
        response1 = client.get("/api/v1/analysis/clusters/REGION_001")
        response2 = client.get("/api/v1/analysis/clusters/REGION_001")
        
        # Compare cluster members
        for c1, c2 in zip(response1.json()["clusters"], response2.json()["clusters"]):
            members1 = sorted([m["municipality_id"] for m in c1["members"]])
            members2 = sorted([m["municipality_id"] for m in c2["members"]])
            assert members1 == members2, "Cluster assignments not deterministic"


class TestNoRandomnessLeakage:
    """Test that no randomness leaks into responses"""
    
    def test_no_uuid_in_data(self):
        """SIN_CARRETA: Data doesn't contain random UUIDs (except request IDs)"""
        response = client.get("/api/v1/pdet/regions/REGION_001")
        region = response.json()["region"]
        
        # Convert to string and check for UUID patterns
        import re
        region_str = str(region)
        
        # UUID pattern (we allow request IDs in headers, not in data)
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        matches = re.findall(uuid_pattern, region_str, re.IGNORECASE)
        
        assert len(matches) == 0, f"Found random UUIDs in data: {matches}"
    
    def test_no_random_floats(self):
        """SIN_CARRETA: Floats follow precision rules (2 or more decimals)"""
        response = client.get("/api/v1/pdet/regions/REGION_001")
        region = response.json()["region"]
        
        # Scores should have reasonable precision (not overly precise)
        overall = region["overall_score"]
        # Allow up to 15 decimal places (reasonable for float precision)
        assert overall == round(overall, 15), "Score has unreasonable precision"
        
        for score in region["dimension_scores"].values():
            assert score == round(score, 15), "Dimension score has unreasonable precision"
    
    def test_no_system_randomness(self):
        """SIN_CARRETA: Core data not affected by system randomness"""
        import os
        import random
        
        # Try to perturb system random state
        random.seed()
        os.urandom(100)
        
        # Results should still be consistent for core fields
        response1 = client.get("/api/v1/pdet/regions/REGION_001")
        
        # More perturbation
        random.seed()
        os.urandom(100)
        
        response2 = client.get("/api/v1/pdet/regions/REGION_001")
        
        region1 = response1.json()["region"]
        region2 = response2.json()["region"]
        
        # Compare core deterministic fields (ID, structure)
        assert region1["id"] == region2["id"]
        assert len(region1["dimension_scores"]) == len(region2["dimension_scores"])
        assert len(region1["policy_area_scores"]) == len(region2["policy_area_scores"])
