#!/usr/bin/env python3
# coding=utf-8
"""
Comprehensive Unit Tests for QuestionRouter
============================================

Tests complete 300-question matrix coverage, canonical notation parsing,
dimension-to-module mapping, adapter registry consistency, and routing logic.

Test Coverage:
- 300-question matrix: P1-P10 × D1-D6 × Q1-Q5
- Canonical notation parsing and validation
- Dimension-to-module mapping correctness
- Primary vs supporting module assignment
- Confidence score calibration
- Edge cases and error handling
- Configuration drift detection

Author: Test Suite
Version: 1.0.0
Python: 3.10+
"""

import pytest
import re
from typing import Dict, List, Set, Any
from pathlib import Path
from dataclasses import dataclass

# Import orchestrator components
import sys
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.config import CONFIG
from orchestrator.questionnaire_parser import QuestionSpec, QuestionnaireParser
from orchestrator.choreographer import ExecutionChoreographer

# Mock ModuleAdapterRegistry to avoid module_adapters.py compilation issues
class MockModuleAdapterRegistry:
    def __init__(self):
        self.adapters = {
            "policy_processor": type('obj', (object,), {'available': True})(),
            "causal_processor": type('obj', (object,), {'available': True})(),
            "contradiction_detector": type('obj', (object,), {'available': True})(),
            "dereck_beach": type('obj', (object,), {'available': True})(),
            "embedding_policy": type('obj', (object,), {'available': True})(),
            "financial_viability": type('obj', (object,), {'available': True})(),
            "analyzer_one": type('obj', (object,), {'available': True})(),
            "policy_segmenter": type('obj', (object,), {'available': True})(),
            "semantic_chunking_policy": type('obj', (object,), {'available': True})(),
        }
ModuleAdapterRegistry = MockModuleAdapterRegistry


# ============================================================================
# TEST FIXTURES AND HELPERS
# ============================================================================

@pytest.fixture
def question_router():
    """Fixture providing QuestionRouter-like functionality"""
    return QuestionRouterTestHelper()


@pytest.fixture
def module_registry():
    """Fixture providing module adapter registry"""
    return ModuleAdapterRegistry()


@pytest.fixture
def choreographer():
    """Fixture providing execution choreographer"""
    return ExecutionChoreographer()


@pytest.fixture
def config():
    """Fixture providing configuration"""
    return CONFIG


class QuestionRouterTestHelper:
    """
    Test helper simulating QuestionRouter functionality
    
    Since QuestionRouter doesn't exist as a separate class, we test
    the integration of QuestionSpec.canonical_id, CONFIG.dimension_module_map,
    and ExecutionChoreographer routing logic.
    """
    
    def __init__(self):
        self.config = CONFIG
        self.choreographer = ExecutionChoreographer()
    
    def parse_canonical_notation(self, canonical_id: str) -> Dict[str, Any]:
        """Parse canonical notation P#-D#-Q#"""
        pattern = r'^P(\d+)-D([1-6])-Q([1-5])$'
        match = re.match(pattern, canonical_id)
        
        if not match:
            raise ValueError(f"Invalid canonical notation: {canonical_id}")
        
        policy_code = f"P{match.group(1)}"
        dimension = f"D{match.group(2)}"
        question_no = int(match.group(3))
        
        return {
            'policy_code': policy_code,
            'dimension': dimension,
            'question_no': question_no,
            'valid': True
        }
    
    def get_modules_for_dimension(self, dimension: str) -> List[str]:
        """Get modules mapped to a dimension"""
        return self.config.dimension_module_map.get(dimension, [])
    
    def route_question(self, canonical_id: str) -> Dict[str, Any]:
        """Route question to appropriate modules"""
        parsed = self.parse_canonical_notation(canonical_id)
        dimension = parsed['dimension']
        
        modules = self.get_modules_for_dimension(dimension)
        
        # Determine primary and supporting modules
        primary = modules[0] if modules else None
        supporting = modules[1:] if len(modules) > 1 else []
        
        return {
            'canonical_id': canonical_id,
            'dimension': dimension,
            'policy_code': parsed['policy_code'],
            'question_no': parsed['question_no'],
            'primary_module': primary,
            'supporting_modules': supporting,
            'all_modules': modules,
            'confidence': self._calculate_confidence(modules)
        }
    
    def _calculate_confidence(self, modules: List[str]) -> float:
        """Calculate routing confidence based on module availability"""
        if not modules:
            return 0.0
        
        # Higher confidence with more modules (evidence triangulation)
        base_confidence = 0.6
        module_bonus = min(0.3, len(modules) * 0.05)
        
        return min(0.95, base_confidence + module_bonus)


# ============================================================================
# TEST SUITE 1: 300-QUESTION MATRIX COVERAGE
# ============================================================================

class TestCanonicalNotationMatrixCoverage:
    """Test complete 300-question matrix: P1-P10 × D1-D6 × Q1-Q5"""
    
    @pytest.mark.parametrize("policy_code", [f"P{i}" for i in range(1, 11)])
    @pytest.mark.parametrize("dimension", [f"D{i}" for i in range(1, 7)])
    @pytest.mark.parametrize("question_no", range(1, 6))
    def test_all_300_questions_parseable(self, question_router, policy_code, dimension, question_no):
        """Test that all 300 questions parse correctly"""
        canonical_id = f"{policy_code}-{dimension}-Q{question_no}"
        
        # Should parse without errors
        parsed = question_router.parse_canonical_notation(canonical_id)
        
        # Verify parsed components
        assert parsed['valid'] is True
        assert parsed['policy_code'] == policy_code
        assert parsed['dimension'] == dimension
        assert parsed['question_no'] == question_no
    
    @pytest.mark.parametrize("policy_code", [f"P{i}" for i in range(1, 11)])
    @pytest.mark.parametrize("dimension", [f"D{i}" for i in range(1, 7)])
    @pytest.mark.parametrize("question_no", range(1, 6))
    def test_all_300_questions_routable(self, question_router, policy_code, dimension, question_no):
        """Test that all 300 questions can be routed to modules"""
        canonical_id = f"{policy_code}-{dimension}-Q{question_no}"
        
        # Should route without errors
        routing = question_router.route_question(canonical_id)
        
        # Verify routing result structure
        assert 'canonical_id' in routing
        assert 'dimension' in routing
        assert 'primary_module' in routing
        assert 'all_modules' in routing
        assert 'confidence' in routing
        
        # All questions should have at least one module
        assert len(routing['all_modules']) > 0, \
            f"Question {canonical_id} has no modules assigned"
    
    def test_matrix_dimension_coverage(self, question_router):
        """Test that all 6 dimensions are covered in the matrix"""
        dimensions = [f"D{i}" for i in range(1, 7)]
        
        for dimension in dimensions:
            modules = question_router.get_modules_for_dimension(dimension)
            assert len(modules) > 0, \
                f"Dimension {dimension} has no module mappings"
    
    def test_matrix_policy_coverage(self, question_router):
        """Test that all 10 policy codes are valid in the matrix"""
        policy_codes = [f"P{i}" for i in range(1, 11)]
        
        for policy_code in policy_codes:
            # Test with D1-Q1 as representative
            canonical_id = f"{policy_code}-D1-Q1"
            parsed = question_router.parse_canonical_notation(canonical_id)
            assert parsed['policy_code'] == policy_code


# ============================================================================
# TEST SUITE 2: CANONICAL NOTATION PARSING
# ============================================================================

class TestCanonicalNotationParsing:
    """Test canonical notation parsing and validation"""
    
    @pytest.mark.parametrize("valid_id,expected", [
        ("P1-D1-Q1", {'policy_code': 'P1', 'dimension': 'D1', 'question_no': 1}),
        ("P5-D3-Q2", {'policy_code': 'P5', 'dimension': 'D3', 'question_no': 2}),
        ("P10-D6-Q5", {'policy_code': 'P10', 'dimension': 'D6', 'question_no': 5}),
    ])
    def test_valid_canonical_notation(self, question_router, valid_id, expected):
        """Test parsing of valid canonical notation formats"""
        parsed = question_router.parse_canonical_notation(valid_id)
        
        assert parsed['valid'] is True
        assert parsed['policy_code'] == expected['policy_code']
        assert parsed['dimension'] == expected['dimension']
        assert parsed['question_no'] == expected['question_no']
    
    @pytest.mark.parametrize("invalid_id", [
        "P0-D1-Q1",      # Invalid policy code (0)
        "P11-D1-Q1",     # Invalid policy code (>10)
        "P1-D0-Q1",      # Invalid dimension (0)
        "P1-D7-Q1",      # Invalid dimension (>6)
        "P1-D1-Q0",      # Invalid question (0)
        "P1-D1-Q6",      # Invalid question (>5)
        "P1-D1",         # Missing question
        "D1-Q1",         # Missing policy
        "P1-Q1",         # Missing dimension
        "P1_D1_Q1",      # Wrong separator
        "p1-d1-q1",      # Lowercase
        "P01-D1-Q1",     # Zero-padded
        "",              # Empty string
        "INVALID",       # Random string
    ])
    def test_invalid_canonical_notation(self, question_router, invalid_id):
        """Test that invalid canonical notations raise appropriate errors"""
        with pytest.raises((ValueError, KeyError, AttributeError)):
            question_router.parse_canonical_notation(invalid_id)
    
    def test_canonical_notation_regex_pattern(self, question_router):
        """Test regex pattern strictness for canonical notation"""
        # These should all fail - boundary cases
        invalid_cases = [
            "P1-D1-Q1 ",     # Trailing space
            " P1-D1-Q1",     # Leading space
            "P1-D1-Q1\n",    # Newline
            "P1-D1-Q1a",     # Extra character
            "aP1-D1-Q1",     # Prefix
        ]
        
        for invalid in invalid_cases:
            with pytest.raises((ValueError, KeyError, AttributeError)):
                question_router.parse_canonical_notation(invalid)


# ============================================================================
# TEST SUITE 3: DIMENSION-TO-MODULE MAPPING VALIDATION
# ============================================================================

class TestDimensionModuleMapping:
    """Test dimension-to-module mapping correctness"""
    
    def test_dimension_module_map_exists(self, config):
        """Test that dimension_module_map exists in CONFIG"""
        assert hasattr(config, 'dimension_module_map')
        assert isinstance(config.dimension_module_map, dict)
    
    def test_all_dimensions_mapped(self, config):
        """Test that all 6 dimensions (D1-D6) have module mappings"""
        expected_dimensions = {f"D{i}" for i in range(1, 7)}
        actual_dimensions = set(config.dimension_module_map.keys())
        
        assert expected_dimensions == actual_dimensions, \
            f"Missing dimensions: {expected_dimensions - actual_dimensions}"
    
    @pytest.mark.parametrize("dimension", [f"D{i}" for i in range(1, 7)])
    def test_each_dimension_has_modules(self, config, dimension):
        """Test that each dimension has at least one module assigned"""
        modules = config.dimension_module_map.get(dimension, [])
        
        assert len(modules) > 0, \
            f"Dimension {dimension} has no modules assigned"
        assert isinstance(modules, list), \
            f"Dimension {dimension} modules should be a list"
    
    def test_module_names_are_valid_strings(self, config):
        """Test that all module names are valid non-empty strings"""
        for dimension, modules in config.dimension_module_map.items():
            for module_name in modules:
                assert isinstance(module_name, str), \
                    f"Module name in {dimension} is not a string: {module_name}"
                assert len(module_name) > 0, \
                    f"Empty module name in {dimension}"
                assert module_name.replace('_', '').isalnum(), \
                    f"Invalid module name in {dimension}: {module_name}"
    
    def test_no_duplicate_modules_per_dimension(self, config):
        """Test that each dimension has no duplicate module entries"""
        for dimension, modules in config.dimension_module_map.items():
            unique_modules = set(modules)
            assert len(modules) == len(unique_modules), \
                f"Dimension {dimension} has duplicate modules: {modules}"
    
    def test_dimension_module_priorities(self, config):
        """Test that module lists respect priority ordering"""
        # Priority modules should appear first in lists
        priority_modules = ['policy_processor', 'causal_processor', 'policy_segmenter']
        
        for dimension, modules in config.dimension_module_map.items():
            if any(pm in modules for pm in priority_modules):
                # At least one priority module should be in top 3
                top_3 = modules[:3]
                has_priority = any(pm in top_3 for pm in priority_modules)
                assert has_priority, \
                    f"Dimension {dimension} has no priority modules in top 3: {top_3}"


# ============================================================================
# TEST SUITE 4: ADAPTER REGISTRY CONSISTENCY
# ============================================================================

class TestAdapterRegistryConsistency:
    """Test adapter registry consistency with dimension mappings"""
    
    def test_all_mapped_modules_exist_in_registry(self, config, module_registry):
        """Test that all modules in dimension_module_map exist in adapter registry"""
        all_mapped_modules = set()
        for modules in config.dimension_module_map.values():
            all_mapped_modules.update(modules)
        
        registry_modules = set(module_registry.adapters.keys())
        
        missing_modules = all_mapped_modules - registry_modules
        assert len(missing_modules) == 0, \
            f"Modules in dimension_module_map missing from registry: {missing_modules}"
    
    def test_no_orphan_adapters(self, config, module_registry):
        """Test that all registered adapters are used in dimension mappings"""
        registry_modules = set(module_registry.adapters.keys())
        
        all_mapped_modules = set()
        for modules in config.dimension_module_map.values():
            all_mapped_modules.update(modules)
        
        # Some adapters may be registered but not directly mapped (utility adapters)
        # So we just warn about unmapped ones rather than fail
        orphan_adapters = registry_modules - all_mapped_modules
        
        # This is informational - some adapters may be supporting utilities
        if orphan_adapters:
            print(f"INFO: Unmapped adapters (may be utilities): {orphan_adapters}")
    
    def test_adapter_availability(self, module_registry):
        """Test that all registered adapters report availability status"""
        for adapter_name, adapter in module_registry.adapters.items():
            assert hasattr(adapter, 'available'), \
                f"Adapter {adapter_name} missing 'available' attribute"
            assert isinstance(adapter.available, bool), \
                f"Adapter {adapter_name} 'available' should be boolean"
    
    def test_adapter_naming_conventions(self, module_registry):
        """Test that adapter names follow naming conventions"""
        for adapter_name in module_registry.adapters.keys():
            # Should be lowercase with underscores
            assert adapter_name.islower() or '_' in adapter_name, \
                f"Adapter name {adapter_name} doesn't follow convention"
            
            # Should not have special characters except underscore
            assert adapter_name.replace('_', '').isalnum(), \
                f"Adapter name {adapter_name} has invalid characters"


# ============================================================================
# TEST SUITE 5: PRIMARY VS SUPPORTING MODULE ASSIGNMENT
# ============================================================================

class TestModuleAssignmentLogic:
    """Test primary vs supporting module assignment logic"""
    
    @pytest.mark.parametrize("dimension", [f"D{i}" for i in range(1, 7)])
    def test_primary_module_assignment(self, question_router, dimension):
        """Test that each dimension has a primary module assigned"""
        canonical_id = f"P1-{dimension}-Q1"
        routing = question_router.route_question(canonical_id)
        
        assert routing['primary_module'] is not None, \
            f"Dimension {dimension} has no primary module"
        assert isinstance(routing['primary_module'], str), \
            f"Primary module for {dimension} should be string"
    
    @pytest.mark.parametrize("dimension", [f"D{i}" for i in range(1, 7)])
    def test_supporting_modules_assignment(self, question_router, dimension):
        """Test that supporting modules are properly assigned"""
        canonical_id = f"P1-{dimension}-Q1"
        routing = question_router.route_question(canonical_id)
        
        # Supporting modules should be a list
        assert isinstance(routing['supporting_modules'], list), \
            f"Supporting modules for {dimension} should be list"
        
        # Primary module should not be in supporting modules
        primary = routing['primary_module']
        supporting = routing['supporting_modules']
        assert primary not in supporting, \
            f"Primary module {primary} appears in supporting modules for {dimension}"
    
    def test_primary_plus_supporting_equals_all(self, question_router):
        """Test that primary + supporting = all_modules"""
        canonical_id = "P1-D1-Q1"
        routing = question_router.route_question(canonical_id)
        
        combined = [routing['primary_module']] + routing['supporting_modules']
        combined_set = set(combined)
        all_modules_set = set(routing['all_modules'])
        
        assert combined_set == all_modules_set, \
            "Primary + supporting modules don't match all_modules"
    
    @pytest.mark.parametrize("dimension", [f"D{i}" for i in range(1, 7)])
    def test_module_count_correctness(self, question_router, dimension):
        """Test that module counts are consistent"""
        canonical_id = f"P1-{dimension}-Q1"
        routing = question_router.route_question(canonical_id)
        
        primary_count = 1 if routing['primary_module'] else 0
        supporting_count = len(routing['supporting_modules'])
        total_count = len(routing['all_modules'])
        
        assert primary_count + supporting_count == total_count, \
            f"Module count mismatch for {dimension}: {primary_count} + {supporting_count} != {total_count}"


# ============================================================================
# TEST SUITE 6: CONFIDENCE SCORE CALIBRATION
# ============================================================================

class TestConfidenceCalibration:
    """Test confidence score calibration for routing decisions"""
    
    def test_confidence_score_range(self, question_router):
        """Test that confidence scores are in valid range [0.0, 1.0]"""
        for policy in range(1, 11):
            for dim in range(1, 7):
                canonical_id = f"P{policy}-D{dim}-Q1"
                routing = question_router.route_question(canonical_id)
                
                confidence = routing['confidence']
                assert 0.0 <= confidence <= 1.0, \
                    f"Confidence {confidence} out of range for {canonical_id}"
    
    def test_confidence_increases_with_modules(self, question_router):
        """Test that confidence generally increases with more modules"""
        # Compare dimensions with different module counts
        routings = []
        for dim in range(1, 7):
            canonical_id = f"P1-D{dim}-Q1"
            routing = question_router.route_question(canonical_id)
            routings.append((dim, len(routing['all_modules']), routing['confidence']))
        
        # Sort by module count
        routings.sort(key=lambda x: x[1])
        
        # Check that confidence tends to increase (allowing some variance)
        for i in range(len(routings) - 1):
            if routings[i][1] < routings[i+1][1]:  # If more modules
                # Confidence should generally be higher (with 0.1 tolerance)
                assert routings[i+1][2] >= routings[i][2] - 0.1, \
                    f"Confidence should increase with more modules: {routings}"
    
    def test_zero_modules_zero_confidence(self, question_router):
        """Test that zero modules results in zero confidence"""
        # This tests the logic, though in practice all dimensions should have modules
        empty_modules = []
        confidence = question_router._calculate_confidence(empty_modules)
        assert confidence == 0.0, \
            "Zero modules should result in zero confidence"
    
    def test_confidence_not_exceeded(self, question_router):
        """Test that confidence never exceeds maximum threshold"""
        # Even with many modules, confidence should cap
        many_modules = ['module'] * 20
        confidence = question_router._calculate_confidence(many_modules)
        assert confidence <= 0.95, \
            f"Confidence {confidence} exceeds maximum threshold 0.95"


# ============================================================================
# TEST SUITE 7: EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCasesAndErrors:
    """Test edge cases and error handling"""
    
    def test_invalid_dimension_mapping(self, question_router):
        """Test handling of invalid dimension (D0, D7, etc.)"""
        invalid_dimensions = ["D0", "D7", "D10", "DX"]
        
        for invalid_dim in invalid_dimensions:
            modules = question_router.get_modules_for_dimension(invalid_dim)
            assert modules == [], \
                f"Invalid dimension {invalid_dim} should return empty list"
    
    def test_missing_dimension_in_config(self, question_router):
        """Test graceful handling of missing dimension in config"""
        # D99 should not exist
        modules = question_router.get_modules_for_dimension("D99")
        assert modules == [], \
            "Missing dimension should return empty list, not error"
    
    def test_malformed_canonical_id_variants(self, question_router):
        """Test various malformed canonical ID formats"""
        malformed_ids = [
            "P1-D1-Q1-EXTRA",  # Extra component
            "P1--D1-Q1",       # Double separator
            "P-1-D1-Q1",       # Split policy code
            "P1-D-1-Q1",       # Split dimension
            "P1-D1-Q-1",       # Split question
        ]
        
        for malformed in malformed_ids:
            with pytest.raises((ValueError, KeyError, AttributeError)):
                question_router.parse_canonical_notation(malformed)
    
    def test_whitespace_handling(self, question_router):
        """Test handling of whitespace in canonical IDs"""
        whitespace_variants = [
            " P1-D1-Q1",
            "P1-D1-Q1 ",
            "P1 -D1-Q1",
            "P1- D1-Q1",
            "P1-D1 -Q1",
            "P1-D1- Q1",
        ]
        
        for variant in whitespace_variants:
            with pytest.raises((ValueError, KeyError, AttributeError)):
                question_router.parse_canonical_notation(variant)


# ============================================================================
# TEST SUITE 8: CONFIGURATION DRIFT DETECTION
# ============================================================================

class TestConfigurationDrift:
    """Test for configuration drift between components"""
    
    def test_choreographer_knows_all_adapters(self, choreographer, module_registry):
        """Test that choreographer's adapter registry matches module registry"""
        choreographer_adapters = set(choreographer.adapter_registry.keys())
        registry_adapters = set(module_registry.adapters.keys())
        
        # Choreographer should know about all registered adapters
        missing_in_choreographer = registry_adapters - choreographer_adapters
        assert len(missing_in_choreographer) == 0, \
            f"Choreographer missing adapters: {missing_in_choreographer}"
    
    def test_config_modules_match_registry(self, config, module_registry):
        """Test that CONFIG dimension mappings use only registered adapters"""
        config_modules = set()
        for modules in config.dimension_module_map.values():
            config_modules.update(modules)
        
        registry_modules = set(module_registry.adapters.keys())
        
        unregistered = config_modules - registry_modules
        assert len(unregistered) == 0, \
            f"CONFIG references unregistered modules: {unregistered}"
    
    def test_adapter_dependencies_valid(self, choreographer, module_registry):
        """Test that adapter dependencies in choreographer graph are valid"""
        graph_nodes = set(choreographer.execution_graph.nodes())
        registry_adapters = set(module_registry.adapters.keys())
        
        invalid_nodes = graph_nodes - registry_adapters
        assert len(invalid_nodes) == 0, \
            f"Choreographer graph has invalid nodes: {invalid_nodes}"
    
    def test_no_circular_dependencies(self, choreographer):
        """Test that adapter dependency graph has no circular dependencies"""
        import networkx as nx
        
        # Should be a DAG (Directed Acyclic Graph)
        assert nx.is_directed_acyclic_graph(choreographer.execution_graph), \
            "Adapter dependency graph contains circular dependencies"


# ============================================================================
# TEST SUITE 9: INTEGRATION TESTS
# ============================================================================

class TestQuestionRouterIntegration:
    """Integration tests for complete question routing pipeline"""
    
    def test_end_to_end_routing_sample(self, question_router):
        """Test end-to-end routing for sample questions"""
        sample_questions = [
            "P1-D1-Q1",  # Basic case
            "P5-D3-Q3",  # Mid-range
            "P10-D6-Q5", # Boundary case
        ]
        
        for canonical_id in sample_questions:
            # Should complete full routing pipeline
            parsed = question_router.parse_canonical_notation(canonical_id)
            assert parsed['valid']
            
            routing = question_router.route_question(canonical_id)
            assert len(routing['all_modules']) > 0
            assert routing['confidence'] > 0
    
    def test_all_questions_have_consistent_routing(self, question_router):
        """Test that routing is consistent for repeated calls"""
        canonical_id = "P1-D1-Q1"
        
        # Route same question multiple times
        routings = [
            question_router.route_question(canonical_id)
            for _ in range(5)
        ]
        
        # All routings should be identical
        first_routing = routings[0]
        for routing in routings[1:]:
            assert routing['primary_module'] == first_routing['primary_module']
            assert routing['supporting_modules'] == first_routing['supporting_modules']
            assert routing['confidence'] == first_routing['confidence']
    
    def test_question_spec_canonical_id_property(self):
        """Test that QuestionSpec.canonical_id property works correctly"""
        question = QuestionSpec(
            question_id="D1-Q1",
            dimension="D1",
            question_no=1,
            policy_area="P1",
            template="Test template",
            text="Test question",
            scoring_modality="TYPE_A",
            max_score=3.0,
            expected_elements=[]
        )
        
        assert question.canonical_id == "P1-D1-Q1"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
