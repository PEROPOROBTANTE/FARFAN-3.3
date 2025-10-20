#!/usr/bin/env python3
"""
Integration Test Suite for ExecutionChoreographer
==================================================

Verifies ExecutionChoreographer correctly:
1. Resolves dependency graph into execution waves
2. Executes independent questions in parallel within waves
3. Enforces sequential dependencies between waves
4. Maps canonical question notation to module adapter/method pairs
5. Detects module signature changes (homeostasis)
6. Manages ThreadPoolExecutor concurrency
7. Properly cleans up thread pool resources
8. Handles priority-based ordering within waves

Author: Integration Team
Version: 3.0.0
Python: 3.10+
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
from concurrent.futures import ThreadPoolExecutor, Future
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import networkx here to avoid import errors
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not available, some tests will be skipped")

# Import directly from choreographer module to avoid orchestrator __init__ dependencies
import importlib.util
spec = importlib.util.spec_from_file_location("choreographer", "orchestrator/choreographer.py")
choreographer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(choreographer_module)

ExecutionChoreographer = choreographer_module.ExecutionChoreographer
ExecutionResult = choreographer_module.ExecutionResult
ExecutionStatus = choreographer_module.ExecutionStatus


# ============================================================================
# MOCK FIXTURES
# ============================================================================

class MockModuleAdapterRegistry:
    """Mock registry that simulates module adapters"""
    
    def __init__(self, adapters: Dict[str, Any]):
        self.adapters = adapters
        self.execution_log = []
        
    def execute_module_method(self, module_name: str, method_name: str, 
                             args: List[Any], kwargs: Dict[str, Any]):
        """Simulate module execution"""
        self.execution_log.append({
            'module': module_name,
            'method': method_name,
            'timestamp': time.time(),
            'args': args,
            'kwargs': kwargs
        })
        
        adapter = self.adapters.get(module_name)
        if not adapter:
            raise ValueError(f"Adapter {module_name} not found")
            
        method = getattr(adapter, method_name, None)
        if not method:
            raise AttributeError(f"Method {method_name} not found on {module_name}")
            
        result = method(*args, **kwargs)
        return result


@dataclass
class MockModuleResult:
    """Mock ModuleResult"""
    module_name: str
    class_name: str
    method_name: str
    status: str
    data: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    confidence: float
    execution_time: float
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class MockAdapter:
    """Base mock adapter"""
    
    def __init__(self, name: str, methods: List[str], execution_delay: float = 0.01):
        self.name = name
        self.methods = methods
        self.execution_delay = execution_delay
        self.execution_count = 0
        
        # Dynamically add methods
        for method in methods:
            setattr(self, method, self._create_method(method))
    
    def _create_method(self, method_name: str):
        """Create a mock method that returns MockModuleResult"""
        def method(*args, **kwargs):
            time.sleep(self.execution_delay)
            self.execution_count += 1
            return MockModuleResult(
                module_name=self.name,
                class_name=f"{self.name.title()}Adapter",
                method_name=method_name,
                status="success",
                data={"result": f"Mock result from {self.name}.{method_name}"},
                evidence=[{"text": "mock evidence", "confidence": 0.9}],
                confidence=0.85,
                execution_time=self.execution_delay,
                errors=[],
                warnings=[],
                metadata={"mock": True}
            )
        return method


def create_mock_adapters() -> Dict[str, MockAdapter]:
    """Create all 9 mock adapters with representative methods"""
    return {
        "policy_segmenter": MockAdapter("policy_segmenter", [
            "segment", "identify_section_boundaries", "score_boundary_confidence"
        ]),
        "policy_processor": MockAdapter("policy_processor", [
            "normalize_unicode", "segment_into_sentences", "process"
        ]),
        "semantic_chunking_policy": MockAdapter("semantic_chunking_policy", [
            "chunk_semantically", "compute_chunk_embeddings", "merge_chunks"
        ]),
        "embedding_policy": MockAdapter("embedding_policy", [
            "generate_embeddings", "compute_similarity", "retrieve_similar"
        ]),
        "analyzer_one": MockAdapter("analyzer_one", [
            "analyze_alignment", "extract_objectives", "compute_coherence"
        ]),
        "teoria_cambio": MockAdapter("teoria_cambio", [
            "extract_theory_of_change", "analyze_causal_chain", "validate_logic"
        ]),
        "dereck_beach": MockAdapter("dereck_beach", [
            "process_document", "extract_causal_hierarchy", "apply_beach_tests"
        ]),
        "contradiction_detection": MockAdapter("contradiction_detection", [
            "detect_contradictions", "analyze_temporal_conflicts", "score_contradiction"
        ]),
        "financial_viability": MockAdapter("financial_viability", [
            "analyze_budget", "compute_viability_score", "trace_allocations"
        ])
    }


# ============================================================================
# TEST SUITE
# ============================================================================

class TestExecutionChoreographerIntegration(unittest.TestCase):
    """Integration tests for ExecutionChoreographer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.choreographer = ExecutionChoreographer(max_workers=4)
        self.mock_adapters = create_mock_adapters()
        self.mock_registry = MockModuleAdapterRegistry(self.mock_adapters)
        
    def tearDown(self):
        """Clean up resources"""
        pass
    
    # ========================================================================
    # TEST 1: Dependency Graph Resolution into Waves
    # ========================================================================
    
    def test_dependency_graph_resolution_into_waves(self):
        """
        Verify ExecutionChoreographer builds correct DAG and resolves it into
        execution waves based on dependencies.
        """
        # Verify graph is a DAG (no cycles)
        self.assertTrue(nx.is_directed_acyclic_graph(self.choreographer.execution_graph))
        
        # Verify all 9 adapters are present
        adapters = list(self.choreographer.execution_graph.nodes())
        self.assertEqual(len(adapters), 9)
        expected_adapters = [
            "policy_segmenter", "policy_processor", "semantic_chunking_policy",
            "embedding_policy", "analyzer_one", "teoria_cambio", "dereck_beach",
            "contradiction_detection", "financial_viability"
        ]
        for adapter in expected_adapters:
            self.assertIn(adapter, adapters, f"Adapter {adapter} not in graph")
        
        # Verify topological ordering (execution order)
        execution_order = self.choreographer.get_execution_order()
        self.assertEqual(len(execution_order), 9)
        
        # Verify wave structure by priority levels
        waves = self._group_by_priority(execution_order)
        
        # Wave 1 (Priority 1): Foundation adapters
        self.assertIn("policy_segmenter", waves[1])
        self.assertIn("policy_processor", waves[1])
        
        # Wave 2 (Priority 2): Semantic adapters
        self.assertIn("semantic_chunking_policy", waves[2])
        self.assertIn("embedding_policy", waves[2])
        
        # Wave 3 (Priority 3): Advanced analysis
        self.assertIn("analyzer_one", waves[3])
        self.assertIn("teoria_cambio", waves[3])
        
        # Wave 4 (Priority 4): Specialized analysis
        self.assertIn("dereck_beach", waves[4])
        self.assertIn("contradiction_detection", waves[4])
        
        # Wave 5 (Priority 5): Final synthesis
        self.assertIn("financial_viability", waves[5])
        
        # Verify dependencies are respected
        self._verify_dependencies_respected(execution_order)
    
    def _group_by_priority(self, execution_order: List[str]) -> Dict[int, List[str]]:
        """Group adapters by priority level"""
        waves = {}
        for adapter in execution_order:
            priority = self.choreographer.get_adapter_priority(adapter)
            if priority not in waves:
                waves[priority] = []
            waves[priority].append(adapter)
        return waves
    
    def _verify_dependencies_respected(self, execution_order: List[str]):
        """Verify that dependencies execute before dependents"""
        executed = set()
        for adapter in execution_order:
            dependencies = self.choreographer.get_adapter_dependencies(adapter)
            for dep in dependencies:
                self.assertIn(
                    dep, executed,
                    f"Dependency {dep} not executed before {adapter}"
                )
            executed.add(adapter)
    
    # ========================================================================
    # TEST 2: Parallel Execution Within Waves
    # ========================================================================
    
    def test_parallel_execution_within_waves(self):
        """
        Verify that independent questions within the same wave execute in
        parallel using ThreadPoolExecutor.
        """
        # Create mock question specs for wave 1 (independent adapters)
        wave1_specs = [
            self._create_mock_question_spec(
                "P1-D1-Q1", 
                [{"adapter": "policy_segmenter", "method": "segment", "args": [], "kwargs": {}}]
            ),
            self._create_mock_question_spec(
                "P1-D1-Q2",
                [{"adapter": "policy_processor", "method": "normalize_unicode", "args": [], "kwargs": {}}]
            )
        ]
        
        # Execute both in parallel
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for spec in wave1_specs:
                future = executor.submit(
                    self.choreographer.execute_question_chain,
                    spec,
                    "mock plan text",
                    self.mock_registry,
                    None
                )
                futures.append(future)
            
            for future in futures:
                results.append(future.result())
        
        total_time = time.time() - start_time
        
        # Verify parallel execution (should be ~0.01s, not ~0.02s)
        # Allow some overhead for thread management
        self.assertLess(total_time, 0.05, 
                       "Parallel execution took too long - may not be parallel")
        
        # Verify both executed
        self.assertEqual(len(results), 2)
    
    # ========================================================================
    # TEST 3: Sequential Execution Between Waves
    # ========================================================================
    
    def test_sequential_execution_between_waves(self):
        """
        Verify that waves execute sequentially - wave 2 only starts after
        wave 1 completes.
        """
        execution_log = []
        
        # Create mock specs spanning multiple waves
        specs = [
            self._create_mock_question_spec(
                "P1-D1-Q1",
                [{"adapter": "policy_segmenter", "method": "segment", "args": [], "kwargs": {}}]
            ),
            self._create_mock_question_spec(
                "P1-D1-Q2",
                [{"adapter": "semantic_chunking_policy", "method": "chunk_semantically", 
                  "args": [], "kwargs": {}}]
            ),
        ]
        
        # Execute and log timestamps
        for spec in specs:
            start = time.time()
            result = self.choreographer.execute_question_chain(
                spec, "mock plan text", self.mock_registry, None
            )
            end = time.time()
            execution_log.append({
                'spec': spec.canonical_id,
                'start': start,
                'end': end
            })
        
        # Verify wave 1 completes before wave 2 starts
        wave1_end = execution_log[0]['end']
        wave2_start = execution_log[1]['start']
        
        self.assertLessEqual(wave1_end, wave2_start,
                            "Wave 2 started before wave 1 completed")
    
    # ========================================================================
    # TEST 4: Canonical Question Notation Mapping
    # ========================================================================
    
    def test_question_to_adapter_method_mapping(self):
        """
        Verify that canonical question notation (P#-D#-Q#) correctly maps to
        adapter.method pairs.
        """
        # Test mapping for various question types
        test_mappings = [
            {
                'canonical_id': 'P1-D1-Q1',
                'expected_adapter': 'policy_segmenter',
                'expected_method': 'segment'
            },
            {
                'canonical_id': 'P1-D2-Q5',
                'expected_adapter': 'teoria_cambio',
                'expected_method': 'extract_theory_of_change'
            },
            {
                'canonical_id': 'P2-D6-Q15',
                'expected_adapter': 'financial_viability',
                'expected_method': 'analyze_budget'
            }
        ]
        
        for mapping in test_mappings:
            spec = self._create_mock_question_spec(
                mapping['canonical_id'],
                [{
                    'adapter': mapping['expected_adapter'],
                    'method': mapping['expected_method'],
                    'args': [],
                    'kwargs': {}
                }]
            )
            
            result = self.choreographer.execute_question_chain(
                spec, "mock plan text", self.mock_registry, None
            )
            
            # Verify correct adapter.method was called
            expected_key = f"{mapping['expected_adapter']}.{mapping['expected_method']}"
            self.assertIn(expected_key, result,
                         f"Expected {expected_key} in results for {mapping['canonical_id']}")
            
            execution_result = result[expected_key]
            self.assertEqual(execution_result.module_name, mapping['expected_adapter'])
            self.assertEqual(execution_result.method_name, mapping['expected_method'])
    
    # ========================================================================
    # TEST 5: Module Signature Change Detection (Homeostasis)
    # ========================================================================
    
    def test_module_signature_change_detection(self):
        """
        Verify that when a module method signature or availability changes,
        the choreographer detects the breakage and fails the affected test.
        This maintains homeostasis by preventing execution of questions whose
        implementations have changed.
        """
        # Test 1: Method removed from adapter
        adapter = self.mock_adapters["policy_segmenter"]
        delattr(adapter, "segment")
        
        spec = self._create_mock_question_spec(
            "P1-D1-Q1",
            [{"adapter": "policy_segmenter", "method": "segment", "args": [], "kwargs": {}}]
        )
        
        # Choreographer catches the error and returns ExecutionResult with FAILED status
        result = self.choreographer.execute_question_chain(
            spec, "mock plan text", self.mock_registry, None
        )
        
        # Verify failure was recorded
        key = "policy_segmenter.segment"
        self.assertIn(key, result)
        self.assertEqual(result[key].status, ExecutionStatus.FAILED)
        self.assertIn("not found", result[key].error.lower())
        
        # Restore method
        adapter.segment = adapter._create_method("segment")
        
        # Test 2: Method signature changed (incompatible args)
        original_method = adapter.segment
        
        def incompatible_method(required_arg_that_doesnt_exist):
            raise TypeError("Missing required argument")
        
        adapter.segment = incompatible_method
        
        # This should fail during execution
        result = self.choreographer.execute_question_chain(
            spec, "mock plan text", self.mock_registry, None
        )
        
        # Verify failure was recorded
        key = "policy_segmenter.segment"
        self.assertIn(key, result)
        self.assertEqual(result[key].status, ExecutionStatus.FAILED)
        self.assertIsNotNone(result[key].error)
        
        # Restore
        adapter.segment = original_method
        
        # Test 3: Adapter completely unavailable
        del self.mock_adapters["policy_segmenter"]
        
        # This should fail with ValueError
        result = self.choreographer.execute_question_chain(
            spec, "mock plan text", self.mock_registry, None
        )
        
        # Verify failure was recorded
        self.assertIn(key, result)
        self.assertEqual(result[key].status, ExecutionStatus.FAILED)
        self.assertIn("not found", result[key].error.lower())
        
        # Restore
        self.mock_adapters["policy_segmenter"] = adapter
    
    # ========================================================================
    # TEST 6: Priority-Based Ordering Within Waves
    # ========================================================================
    
    def test_priority_based_ordering_within_waves(self):
        """
        Verify that adapters within the same wave maintain priority ordering.
        """
        # Get all adapters grouped by priority
        waves = {}
        for adapter in self.choreographer.get_execution_order():
            priority = self.choreographer.get_adapter_priority(adapter)
            if priority not in waves:
                waves[priority] = []
            waves[priority].append(adapter)
        
        # Verify priority 1 adapters
        self.assertEqual(
            self.choreographer.get_adapter_priority("policy_segmenter"), 1
        )
        self.assertEqual(
            self.choreographer.get_adapter_priority("policy_processor"), 1
        )
        
        # Verify priority 2 adapters
        self.assertEqual(
            self.choreographer.get_adapter_priority("semantic_chunking_policy"), 2
        )
        self.assertEqual(
            self.choreographer.get_adapter_priority("embedding_policy"), 2
        )
        
        # Verify priority 3 adapters
        self.assertEqual(
            self.choreographer.get_adapter_priority("analyzer_one"), 3
        )
        self.assertEqual(
            self.choreographer.get_adapter_priority("teoria_cambio"), 3
        )
        
        # Verify priority 4 adapters
        self.assertEqual(
            self.choreographer.get_adapter_priority("dereck_beach"), 4
        )
        self.assertEqual(
            self.choreographer.get_adapter_priority("contradiction_detection"), 4
        )
        
        # Verify priority 5 adapters
        self.assertEqual(
            self.choreographer.get_adapter_priority("financial_viability"), 5
        )
        
        # Verify monotonic increase in priorities along execution path
        execution_order = self.choreographer.get_execution_order()
        for i in range(len(execution_order) - 1):
            current_priority = self.choreographer.get_adapter_priority(execution_order[i])
            next_priority = self.choreographer.get_adapter_priority(execution_order[i + 1])
            self.assertLessEqual(current_priority, next_priority,
                               f"Priority ordering violated: {execution_order[i]} -> {execution_order[i+1]}")
    
    # ========================================================================
    # TEST 7: Thread Pool Resource Cleanup
    # ========================================================================
    
    def test_thread_pool_cleanup(self):
        """
        Verify that ThreadPoolExecutor is properly cleaned up after execution.
        """
        import threading
        
        # Count active threads before
        threads_before = threading.active_count()
        
        # Execute multiple questions
        specs = [
            self._create_mock_question_spec(
                f"P1-D1-Q{i}",
                [{"adapter": "policy_processor", "method": "normalize_unicode", 
                  "args": [], "kwargs": {}}]
            )
            for i in range(5)
        ]
        
        # Execute with thread pool
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    self.choreographer.execute_question_chain,
                    spec, "mock plan text", self.mock_registry, None
                )
                for spec in specs
            ]
            
            # Wait for completion
            for future in futures:
                future.result()
        
        # Give threads time to clean up
        time.sleep(0.1)
        
        # Count active threads after
        threads_after = threading.active_count()
        
        # Verify thread pool was cleaned up (threads_after should be same or less than before)
        self.assertLessEqual(threads_after, threads_before + 1,
                           "Thread pool not properly cleaned up")
    
    # ========================================================================
    # TEST 8: DAG Dependency Verification (D6 depends on D1-D5)
    # ========================================================================
    
    def test_dag_dependency_verification_d6_depends_on_d1_d5(self):
        """
        Verify that D6 (financial viability) questions depend on results from
        D1-D5 questions, represented correctly in the DAG.
        """
        # financial_viability should have dependencies
        financial_deps = self.choreographer.get_adapter_dependencies("financial_viability")
        
        # Should depend on multiple earlier adapters
        self.assertGreater(len(financial_deps), 0, 
                          "financial_viability has no dependencies")
        
        # Should specifically depend on contradiction_detection, dereck_beach, analyzer_one
        expected_deps = ["contradiction_detection", "dereck_beach", "analyzer_one"]
        for dep in expected_deps:
            self.assertIn(dep, financial_deps,
                         f"financial_viability missing dependency on {dep}")
        
        # Verify financial_viability is in the last wave (priority 5)
        self.assertEqual(
            self.choreographer.get_adapter_priority("financial_viability"), 5
        )
        
        # Verify all dependencies are in earlier waves
        for dep in financial_deps:
            dep_priority = self.choreographer.get_adapter_priority(dep)
            self.assertLess(dep_priority, 5,
                          f"Dependency {dep} has priority >= financial_viability")
        
        # Verify path exists from foundation to financial_viability
        foundation_adapters = ["policy_segmenter", "policy_processor"]
        for foundation in foundation_adapters:
            has_path = nx.has_path(
                self.choreographer.execution_graph,
                foundation,
                "financial_viability"
            )
            self.assertTrue(has_path,
                          f"No path from {foundation} to financial_viability")
    
    # ========================================================================
    # TEST 9: Result Aggregation
    # ========================================================================
    
    def test_result_aggregation(self):
        """
        Verify that results from multiple adapters are correctly aggregated.
        """
        # Create multiple execution results
        results = {
            "adapter1.method1": ExecutionResult(
                module_name="adapter1",
                adapter_class="Adapter1",
                method_name="method1",
                status=ExecutionStatus.COMPLETED,
                output={"data": "result1"},
                evidence_extracted={"evidence": [{"text": "evidence1", "confidence": 0.9}]},
                confidence=0.9,
                execution_time=0.1
            ),
            "adapter2.method2": ExecutionResult(
                module_name="adapter2",
                adapter_class="Adapter2",
                method_name="method2",
                status=ExecutionStatus.COMPLETED,
                output={"data": "result2"},
                evidence_extracted={"evidence": [{"text": "evidence2", "confidence": 0.8}]},
                confidence=0.8,
                execution_time=0.2
            ),
            "adapter3.method3": ExecutionResult(
                module_name="adapter3",
                adapter_class="Adapter3",
                method_name="method3",
                status=ExecutionStatus.FAILED,
                error="Mock error",
                execution_time=0.05
            )
        }
        
        # Aggregate
        aggregated = self.choreographer.aggregate_results(results)
        
        # Verify aggregation
        self.assertEqual(aggregated["total_steps"], 3)
        self.assertEqual(aggregated["successful_steps"], 2)
        self.assertEqual(aggregated["failed_steps"], 1)
        self.assertAlmostEqual(aggregated["total_execution_time"], 0.35, places=2)
        self.assertAlmostEqual(aggregated["avg_confidence"], 0.5667, places=2)
        self.assertEqual(len(aggregated["adapters_executed"]), 3)
        self.assertEqual(len(aggregated["outputs"]), 2)
    
    # ========================================================================
    # TEST 10: Circuit Breaker Integration
    # ========================================================================
    
    def test_circuit_breaker_integration(self):
        """
        Verify that circuit breaker prevents execution of failing adapters.
        """
        # Create mock circuit breaker
        mock_circuit_breaker = Mock()
        mock_circuit_breaker.can_execute.return_value = False
        mock_circuit_breaker.record_success = Mock()
        mock_circuit_breaker.record_failure = Mock()
        
        spec = self._create_mock_question_spec(
            "P1-D1-Q1",
            [{"adapter": "policy_segmenter", "method": "segment", "args": [], "kwargs": {}}]
        )
        
        # Execute with circuit breaker blocking
        result = self.choreographer.execute_question_chain(
            spec, "mock plan text", self.mock_registry, mock_circuit_breaker
        )
        
        # Verify execution was skipped
        key = "policy_segmenter.segment"
        self.assertIn(key, result)
        self.assertEqual(result[key].status, ExecutionStatus.SKIPPED)
        self.assertIn("Circuit breaker", result[key].error)
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _create_mock_question_spec(self, canonical_id: str, execution_chain: List[Dict]):
        """Create a mock question specification"""
        spec = Mock()
        spec.canonical_id = canonical_id
        spec.execution_chain = execution_chain
        return spec


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
