# coding=utf-8
"""
Integration Tests for ExecutionChoreographer with ModuleAdapterRegistry
========================================================================

Tests the choreographer's integration with the new ModuleAdapterRegistry,
ensuring proper adapter method execution, error handling, and result aggregation.

Deterministic execution with stub adapters and fixed clock/trace IDs.

Marks: integration
"""

import pytest
import time
from typing import Dict, Any, List
from unittest.mock import Mock
from dataclasses import dataclass

# Direct imports to avoid loading entire package
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.orchestrator.adapter_registry import (
    ModuleAdapterRegistry,
    ExecutionStatus as RegExecutionStatus,
)
from src.orchestrator.choreographer import ExecutionChoreographer, ExecutionStatus


# Stub adapters for testing
class StubPolicySegmenter:
    """Stub for policy_segmenter adapter"""

    def segment_document(self, text: str) -> Dict[str, Any]:
        return {
            "evidence": [{"type": "segment", "text": text[:50], "confidence": 0.9}],
            "segments": 5,
        }


class StubPolicyProcessor:
    """Stub for policy_processor adapter"""

    def normalize_text(self, text: str) -> Dict[str, Any]:
        return {
            "evidence": [
                {"type": "normalization", "text": text.lower(), "confidence": 0.95}
            ],
            "normalized": True,
        }


class StubAnalyzerOne:
    """Stub for analyzer_one adapter"""

    def analyze_municipal_development(self, segments: Any) -> Dict[str, Any]:
        return {
            "evidence": [
                {"type": "analysis", "finding": "good development", "confidence": 0.85}
            ],
            "score": 85,
        }


class StubFailingAdapter:
    """Stub adapter that fails"""

    def failing_method(self):
        raise ValueError("Intentional test failure")


@dataclass
class MockQuestionSpec:
    """Mock question specification"""

    canonical_id: str
    execution_chain: List[Dict[str, Any]]


@pytest.fixture
def deterministic_clock():
    """Fixture providing deterministic monotonic clock"""
    counter = [0.0]

    def clock():
        counter[0] += 0.001
        return counter[0]

    return clock


@pytest.fixture
def deterministic_trace_id():
    """Fixture providing deterministic trace ID generator"""
    counter = [0]

    def generator():
        counter[0] += 1
        return f"trace-{counter[0]:04d}"

    return generator


@pytest.fixture
def registry_with_adapters(deterministic_clock, deterministic_trace_id):
    """Fixture providing registry with stub adapters"""
    registry = ModuleAdapterRegistry(
        clock=deterministic_clock, trace_id_generator=deterministic_trace_id
    )

    # Register 3 stub adapters for testing
    registry.register_adapter(
        module_name="policy_segmenter",
        adapter_instance=StubPolicySegmenter(),
        adapter_class_name="StubPolicySegmenter",
        description="Segments policy documents",
    )

    registry.register_adapter(
        module_name="policy_processor",
        adapter_instance=StubPolicyProcessor(),
        adapter_class_name="StubPolicyProcessor",
        description="Processes and normalizes text",
    )

    registry.register_adapter(
        module_name="analyzer_one",
        adapter_instance=StubAnalyzerOne(),
        adapter_class_name="StubAnalyzerOne",
        description="Analyzes municipal development",
    )

    return registry


@pytest.fixture
def choreographer():
    """Fixture providing ExecutionChoreographer instance"""
    return ExecutionChoreographer()


@pytest.mark.integration
class TestChoreographerWithModuleAdapterRegistry:
    """Test choreographer integration with new ModuleAdapterRegistry"""

    def test_execute_single_step_with_new_registry(
        self, choreographer, registry_with_adapters
    ):
        """Test single step execution with ModuleAdapterRegistry"""
        result = choreographer._execute_single_step(
            adapter_name="policy_segmenter",
            method_name="segment_document",
            args=["test document text"],
            kwargs={},
            module_adapter_registry=registry_with_adapters,
            circuit_breaker=None,
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert result.module_name == "policy_segmenter"
        assert result.adapter_class == "policy_segmenter"
        assert result.method_name == "segment_document"
        assert result.confidence == 1.0
        assert "trace_id" in result.metadata
        assert result.metadata["trace_id"] == "trace-0001"  # Deterministic
        assert result.execution_time > 0

    def test_execute_question_chain_with_multiple_adapters(
        self, choreographer, registry_with_adapters
    ):
        """Test executing a question chain with multiple adapters"""
        question_spec = MockQuestionSpec(
            canonical_id="TEST-Q1",
            execution_chain=[
                {
                    "adapter": "policy_segmenter",
                    "method": "segment_document",
                    "args": [{"source": "plan_text"}],
                    "kwargs": {},
                },
                {
                    "adapter": "policy_processor",
                    "method": "normalize_text",
                    "args": [{"source": "plan_text"}],
                    "kwargs": {},
                },
                {
                    "adapter": "analyzer_one",
                    "method": "analyze_municipal_development",
                    "args": [{"source": "previous_result"}],
                    "kwargs": {},
                },
            ],
        )

        results = choreographer.execute_question_chain(
            question_spec=question_spec,
            plan_text="Sample plan document text for testing",
            module_adapter_registry=registry_with_adapters,
            circuit_breaker=None,
        )

        assert len(results) == 3
        assert "policy_segmenter.segment_document" in results
        assert "policy_processor.normalize_text" in results
        assert "analyzer_one.analyze_municipal_development" in results

        # All should succeed
        for result in results.values():
            assert result.status == ExecutionStatus.COMPLETED

    def test_missing_adapter_handling(self, choreographer, registry_with_adapters):
        """Test that missing adapter is properly handled"""
        result = choreographer._execute_single_step(
            adapter_name="nonexistent_adapter",
            method_name="some_method",
            args=[],
            kwargs={},
            module_adapter_registry=registry_with_adapters,
            circuit_breaker=None,
        )

        assert result.status == ExecutionStatus.FAILED
        assert (
            "not registered" in result.error.lower()
            or "not found" in result.error.lower()
        )

    def test_missing_method_handling(self, choreographer, registry_with_adapters):
        """Test that missing method returns appropriate status"""
        result = choreographer._execute_single_step(
            adapter_name="policy_segmenter",
            method_name="nonexistent_method",
            args=[],
            kwargs={},
            module_adapter_registry=registry_with_adapters,
            circuit_breaker=None,
        )

        # With new registry, missing method returns SKIPPED status
        assert result.status == ExecutionStatus.SKIPPED
        assert result.confidence == 0.0

    def test_adapter_method_failure(
        self, choreographer, deterministic_clock, deterministic_trace_id
    ):
        """Test that adapter method failures are properly captured"""
        registry = ModuleAdapterRegistry(
            clock=deterministic_clock, trace_id_generator=deterministic_trace_id
        )

        registry.register_adapter(
            module_name="failing_adapter",
            adapter_instance=StubFailingAdapter(),
            adapter_class_name="StubFailingAdapter",
        )

        result = choreographer._execute_single_step(
            adapter_name="failing_adapter",
            method_name="failing_method",
            args=[],
            kwargs={},
            module_adapter_registry=registry,
            circuit_breaker=None,
        )

        assert result.status == ExecutionStatus.FAILED
        assert result.confidence == 0.0
        assert "Intentional test failure" in result.error

    def test_validate_adapter_method_with_new_registry(
        self, choreographer, registry_with_adapters
    ):
        """Test adapter method validation with new registry"""
        # Valid adapter and method
        assert choreographer._validate_adapter_method(
            adapter_name="policy_segmenter",
            method_name="segment_document",
            module_adapter_registry=registry_with_adapters,
        )

        # Invalid adapter
        assert not choreographer._validate_adapter_method(
            adapter_name="nonexistent",
            method_name="some_method",
            module_adapter_registry=registry_with_adapters,
        )

        # Valid adapter, invalid method
        assert not choreographer._validate_adapter_method(
            adapter_name="policy_segmenter",
            method_name="nonexistent_method",
            module_adapter_registry=registry_with_adapters,
        )

    def test_result_aggregation(self, choreographer):
        """Test result aggregation from multiple executions"""
        from src.orchestrator.choreographer import ExecutionResult, ExecutionStatus

        results = {
            "adapter1.method1": ExecutionResult(
                module_name="adapter1",
                adapter_class="Adapter1",
                method_name="method1",
                status=ExecutionStatus.COMPLETED,
                output={"result": "data1"},
                execution_time=0.1,
                confidence=0.9,
            ),
            "adapter2.method2": ExecutionResult(
                module_name="adapter2",
                adapter_class="Adapter2",
                method_name="method2",
                status=ExecutionStatus.COMPLETED,
                output={"result": "data2"},
                execution_time=0.2,
                confidence=0.85,
            ),
            "adapter3.method3": ExecutionResult(
                module_name="adapter3",
                adapter_class="Adapter3",
                method_name="method3",
                status=ExecutionStatus.FAILED,
                error="Test failure",
                execution_time=0.05,
                confidence=0.0,
            ),
        }

        aggregated = choreographer.aggregate_results(results)

        assert aggregated["total_steps"] == 3
        assert aggregated["successful_steps"] == 2
        assert aggregated["failed_steps"] == 1
        assert (
            abs(aggregated["total_execution_time"] - 0.35) < 0.001
        )  # Float precision tolerance
        assert 0.5 < aggregated["avg_confidence"] < 0.7
        assert len(aggregated["adapters_executed"]) == 3

    def test_deterministic_trace_ids_across_multiple_steps(
        self, choreographer, registry_with_adapters
    ):
        """Test that trace IDs are deterministic across multiple step executions"""
        results = []

        for i in range(3):
            result = choreographer._execute_single_step(
                adapter_name="policy_segmenter",
                method_name="segment_document",
                args=["test text"],
                kwargs={},
                module_adapter_registry=registry_with_adapters,
                circuit_breaker=None,
            )
            results.append(result)

        # Trace IDs should be sequential and deterministic
        assert results[0].metadata["trace_id"] == "trace-0001"
        assert results[1].metadata["trace_id"] == "trace-0002"
        assert results[2].metadata["trace_id"] == "trace-0003"


@pytest.mark.integration
class TestBackwardCompatibility:
    """Test backward compatibility with legacy AdapterRegistry"""

    def test_choreographer_works_with_legacy_registry(self, choreographer):
        """Test that choreographer still works with legacy AdapterRegistry for backward compatibility"""
        # Create a mock legacy registry
        legacy_registry = Mock()
        legacy_registry.adapters = {"test_adapter": StubPolicySegmenter()}

        # Should not have execute_module_method
        (
            delattr(legacy_registry, "execute_module_method")
            if hasattr(legacy_registry, "execute_module_method")
            else None
        )

        # This should work via fallback path
        result = choreographer._execute_single_step(
            adapter_name="test_adapter",
            method_name="segment_document",
            args=["test"],
            kwargs={},
            module_adapter_registry=legacy_registry,
            circuit_breaker=None,
        )

        assert result.status == ExecutionStatus.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
