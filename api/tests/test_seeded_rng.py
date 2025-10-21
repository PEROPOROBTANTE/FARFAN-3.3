# coding=utf-8
"""
Tests for Seeded RNG Utilities
===============================

SIN_CARRETA: Verify determinism and correctness of seeded random generation.

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import pytest
from api.utils.seeded_rng import (
    Mulberry32, SplitMix32, SeededGenerator, create_seeded_generator
)


class TestMulberry32:
    """Test Mulberry32 PRNG"""
    
    def test_determinism(self):
        """SIN_CARRETA: Same seed produces same sequence"""
        rng1 = Mulberry32(42)
        rng2 = Mulberry32(42)
        
        # Generate 100 numbers from each
        seq1 = [rng1.next_uint32() for _ in range(100)]
        seq2 = [rng2.next_uint32() for _ in range(100)]
        
        assert seq1 == seq2, "Same seed must produce identical sequences"
    
    def test_different_seeds_different_sequences(self):
        """SIN_CARRETA: Different seeds produce different sequences"""
        rng1 = Mulberry32(42)
        rng2 = Mulberry32(43)
        
        seq1 = [rng1.next_uint32() for _ in range(100)]
        seq2 = [rng2.next_uint32() for _ in range(100)]
        
        assert seq1 != seq2, "Different seeds must produce different sequences"
    
    def test_float_range(self):
        """SIN_CARRETA: Float values are in [0, 1) range"""
        rng = Mulberry32(42)
        
        for _ in range(1000):
            val = rng.next_float()
            assert 0.0 <= val < 1.0, f"Float {val} out of range [0, 1)"
    
    def test_int_range(self):
        """SIN_CARRETA: Integer values are in specified range"""
        rng = Mulberry32(42)
        
        for _ in range(1000):
            val = rng.next_int(10, 20)
            assert 10 <= val <= 20, f"Int {val} out of range [10, 20]"
    
    def test_choice(self):
        """SIN_CARRETA: Choice selects valid items"""
        rng = Mulberry32(42)
        items = ["a", "b", "c", "d"]
        
        for _ in range(100):
            chosen = rng.choice(items)
            assert chosen in items, f"Chosen {chosen} not in items"
    
    def test_choice_determinism(self):
        """SIN_CARRETA: Choice is deterministic"""
        rng1 = Mulberry32(42)
        rng2 = Mulberry32(42)
        items = ["a", "b", "c", "d"]
        
        choices1 = [rng1.choice(items) for _ in range(100)]
        choices2 = [rng2.choice(items) for _ in range(100)]
        
        assert choices1 == choices2
    
    def test_shuffle_determinism(self):
        """SIN_CARRETA: Shuffle is deterministic"""
        rng1 = Mulberry32(42)
        rng2 = Mulberry32(42)
        
        items1 = list(range(20))
        items2 = list(range(20))
        
        rng1.shuffle(items1)
        rng2.shuffle(items2)
        
        assert items1 == items2


class TestSplitMix32:
    """Test SplitMix32 PRNG"""
    
    def test_determinism(self):
        """SIN_CARRETA: Same seed produces same sequence"""
        rng1 = SplitMix32(42)
        rng2 = SplitMix32(42)
        
        seq1 = [rng1.next_uint32() for _ in range(100)]
        seq2 = [rng2.next_uint32() for _ in range(100)]
        
        assert seq1 == seq2
    
    def test_different_from_mulberry(self):
        """SIN_CARRETA: SplitMix32 produces different sequence than Mulberry32"""
        mul = Mulberry32(42)
        spl = SplitMix32(42)
        
        seq_mul = [mul.next_uint32() for _ in range(100)]
        seq_spl = [spl.next_uint32() for _ in range(100)]
        
        assert seq_mul != seq_spl, "Different algorithms must produce different sequences"
    
    def test_float_range(self):
        """SIN_CARRETA: Float values in correct range"""
        rng = SplitMix32(42)
        
        for _ in range(1000):
            val = rng.next_float()
            assert 0.0 <= val < 1.0


class TestSeededGenerator:
    """Test high-level SeededGenerator"""
    
    def test_mulberry_algorithm(self):
        """SIN_CARRETA: Mulberry32 algorithm works"""
        gen = SeededGenerator(42, "mulberry32")
        assert gen.algorithm == "mulberry32"
        
        # Should be able to generate values
        val = gen.generate_score()
        assert 0.0 <= val <= 100.0
    
    def test_splitmix_algorithm(self):
        """SIN_CARRETA: SplitMix32 algorithm works"""
        gen = SeededGenerator(42, "splitmix32")
        assert gen.algorithm == "splitmix32"
        
        val = gen.generate_score()
        assert 0.0 <= val <= 100.0
    
    def test_invalid_algorithm(self):
        """SIN_CARRETA: Invalid algorithm raises error"""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            SeededGenerator(42, "invalid")
    
    def test_region_ids_deterministic(self):
        """SIN_CARRETA: Region IDs are deterministic"""
        gen1 = SeededGenerator(42, "mulberry32")
        gen2 = SeededGenerator(42, "mulberry32")
        
        ids1 = gen1.generate_region_ids(10)
        ids2 = gen2.generate_region_ids(10)
        
        assert ids1 == ids2
        assert len(ids1) == 10
        assert all(id.startswith("REGION_") for id in ids1)
    
    def test_municipality_ids_deterministic(self):
        """SIN_CARRETA: Municipality IDs are deterministic"""
        gen1 = SeededGenerator(42, "mulberry32")
        gen2 = SeededGenerator(42, "mulberry32")
        
        ids1 = gen1.generate_municipality_ids("REGION_001", 10)
        ids2 = gen2.generate_municipality_ids("REGION_001", 10)
        
        assert ids1 == ids2
        assert len(ids1) == 10
        assert all(id.startswith("MUN_") for id in ids1)
    
    def test_score_generation(self):
        """SIN_CARRETA: Scores are in correct range"""
        gen = SeededGenerator(42, "mulberry32")
        
        for _ in range(100):
            score = gen.generate_score(0.0, 100.0)
            assert 0.0 <= score <= 100.0
            # Check 2 decimal places
            assert score == round(score, 2)
    
    def test_name_generation_deterministic(self):
        """SIN_CARRETA: Name generation is deterministic"""
        gen1 = SeededGenerator(42, "mulberry32")
        gen2 = SeededGenerator(42, "mulberry32")
        
        name1 = gen1.generate_name("Test")
        name2 = gen2.generate_name("Test")
        
        assert name1 == name2
        assert name1.startswith("Test")
    
    def test_coordinates_in_colombia(self):
        """SIN_CARRETA: Coordinates are within Colombia bounds"""
        gen = SeededGenerator(42, "mulberry32")
        
        for _ in range(100):
            lat, lon = gen.generate_coordinates()
            
            # Colombia bounds: lat -4.2 to 12.5, lon -81.7 to -66.9
            assert -4.3 <= lat <= 12.6, f"Latitude {lat} out of Colombia bounds"
            assert -81.8 <= lon <= -66.8, f"Longitude {lon} out of Colombia bounds"
    
    def test_factory_function(self):
        """SIN_CARRETA: Factory function creates generator"""
        gen = create_seeded_generator(42, "mulberry32")
        assert isinstance(gen, SeededGenerator)
        assert gen.seed == 42
        assert gen.algorithm == "mulberry32"


class TestDeterminismAcrossRestarts:
    """Test that determinism holds across multiple instantiations"""
    
    def test_region_names_consistent(self):
        """SIN_CARRETA: Region names are consistent across restarts"""
        names1 = []
        for i in range(10):
            gen = SeededGenerator(42, "mulberry32")
            names1.append(gen.generate_name("Region"))
        
        names2 = []
        for i in range(10):
            gen = SeededGenerator(42, "mulberry32")
            names2.append(gen.generate_name("Region"))
        
        assert names1 == names2
    
    def test_scores_consistent(self):
        """SIN_CARRETA: Scores are consistent across restarts"""
        scores1 = []
        for i in range(10):
            gen = SeededGenerator(42 + i, "mulberry32")
            scores1.append(gen.generate_score())
        
        scores2 = []
        for i in range(10):
            gen = SeededGenerator(42 + i, "mulberry32")
            scores2.append(gen.generate_score())
        
        assert scores1 == scores2
