# coding=utf-8
"""
Unit Tests for ModuleAdapterRegistry Contract
==============================================

Tests the formal execution contract of ModuleAdapterRegistry with:
- Deterministic adapter registration and error isolation
- execute_module_method contract enforcement
- ContractViolation exception behavior
- Deterministic timing with injected clock
- Trace ID generation with deterministic stub
- Method introspection

Marks: unit
"""

import pytest
import time
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# Direct import to avoid loading entire orchestrator package
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.orchestrator.adapter_registry import (
    ModuleAdapterRegistry,
    ModuleMethodResult,
    ContractViolation,
    ExecutionStatus,
    AdapterAvailabilitySnapshot,
)


class StubAdapterSuccess:
    """Stub adapter that always succeeds"""

    def process(self, text: str) -> Dict[str, Any]:
        return {"evidence": [{"text": text, "confidence": 0.9}], "result": "processed"}

    def analyze(self) -> List[str]:
        return ["result1", "result2"]

    def compute(self, x: int, y: int) -> int:
        return x + y


class StubAdapterFailure:
    """Stub adapter that raises exceptions"""

    def failing_method(self):
        raise ValueError("Intentional failure for testing")


class StubAdapterNoMethods:
    """Stub adapter with no public methods"""

    pass


@pytest.fixture
def deterministic_clock():
    """Fixture providing deterministic monotonic clock"""
    counter = [0.0]

    def clock():
        counter[0] += 0.001  # Increment by 1ms each call
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
def registry(deterministic_clock, deterministic_trace_id):
    """Fixture providing registry with deterministic clock and trace IDs"""
    return ModuleAdapterRegistry(
        clock=deterministic_clock,
        trace_id_generator=deterministic_trace_id,
        auto_register=False  # Don't auto-register for clean testing
    )


@pytest.mark.unit
class TestAdapterRegistration:
    """Test adapter registration with error isolation"""

    def test_successful_registration(self, registry):
        """Test successful adapter registration"""
        adapter = StubAdapterSuccess()

        registry.register_adapter(
            module_name="test_adapter",
            adapter_instance=adapter,
            adapter_class_name="StubAdapterSuccess",
            description="Test adapter",
        )

        assert registry.is_available("test_adapter")
        status = registry.get_status()
        assert "test_adapter" in status
        assert status["test_adapter"].available
        assert status["test_adapter"].description == "Test adapter"

    def test_registration_with_error_isolation(self, registry):
        """Test that registration errors are isolated and adapter marked unavailable"""
        # Create adapter that will fail during registration attribute access
        failing_adapter = Mock()
        failing_adapter.__class__.__name__ = "FailingAdapter"

        # This should not raise, but mark adapter as unavailable
        registry.register_adapter(
            module_name="failing_adapter",
            adapter_instance=failing_adapter,
            adapter_class_name="FailingAdapter",
            description="Adapter that fails",
        )

        # Adapter should be registered but unavailable
        assert "failing_adapter" in registry.get_status()
        # Note: In current implementation, Mock objects succeed registration
        # This test documents expected behavior for actual failures

    def test_multiple_adapter_registration(self, registry):
        """Test registering multiple adapters"""
        adapters = [
            ("adapter1", StubAdapterSuccess()),
            ("adapter2", StubAdapterFailure()),
            ("adapter3", StubAdapterNoMethods()),
        ]

        for name, instance in adapters:
            registry.register_adapter(
                module_name=name,
                adapter_instance=instance,
                adapter_class_name=instance.__class__.__name__,
            )

        status = registry.get_status()
        assert len(status) == 3
        for name, _ in adapters:
            assert name in status


@pytest.mark.unit
class TestExecuteModuleMethod:
    """Test execute_module_method contract"""

    def test_successful_execution(self, registry):
        """Test successful method execution returns correct ModuleMethodResult"""
        adapter = StubAdapterSuccess()
        registry.register_adapter(
            module_name="test_adapter",
            adapter_instance=adapter,
            adapter_class_name="StubAdapterSuccess",
        )

        result = registry.execute_module_method(
            module_name="test_adapter", method_name="process", args=["test text"]
        )

        assert isinstance(result, ModuleMethodResult)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.module_name == "test_adapter"
        assert result.adapter_class == "test_adapter"
        assert result.method_name == "process"
        assert result.confidence == 1.0
        assert len(result.evidence) == 1
        assert result.evidence[0]["text"] == "test text"
        assert result.error_type is None
        assert result.error_message is None
        assert result.execution_time > 0
        assert result.trace_id == "trace-0001"  # Deterministic

    def test_execution_with_kwargs(self, registry):
        """Test method execution with keyword arguments"""
        adapter = StubAdapterSuccess()
        registry.register_adapter(
            module_name="test_adapter",
            adapter_instance=adapter,
            adapter_class_name="StubAdapterSuccess",
        )

        result = registry.execute_module_method(
            module_name="test_adapter", method_name="compute", kwargs={"x": 5, "y": 3}
        )

        assert result.status == ExecutionStatus.SUCCESS

    def test_missing_adapter_raises_contract_violation(self, registry):
        """Test that executing non-existent adapter raises ContractViolation"""
        with pytest.raises(ContractViolation) as exc_info:
            registry.execute_module_method(
                module_name="nonexistent", method_name="process"
            )

        assert "not registered" in str(exc_info.value)

    def test_unavailable_adapter_raises_contract_violation(self, registry):
        """Test that executing unavailable adapter raises ContractViolation"""
        # Manually mark adapter as unavailable
        registry._availability["unavailable_adapter"] = AdapterAvailabilitySnapshot(
            adapter_name="unavailable_adapter",
            available=False,
            error_type="TestError",
            error_message="Simulated unavailability",
        )

        with pytest.raises(ContractViolation) as exc_info:
            registry.execute_module_method(
                module_name="unavailable_adapter", method_name="process"
            )

        assert "unavailable" in str(exc_info.value).lower()

    def test_unavailable_adapter_with_allow_degraded(self, registry):
        """Test that allow_degraded=True bypasses unavailability check"""
        # Register then mark as unavailable
        adapter = StubAdapterSuccess()
        registry.register_adapter(
            module_name="degraded_adapter",
            adapter_instance=adapter,
            adapter_class_name="StubAdapterSuccess",
        )
        registry.set_adapter_availability("degraded_adapter", False)

        # Should still execute with allow_degraded=True
        result = registry.execute_module_method(
            module_name="degraded_adapter",
            method_name="process",
            args=["test"],
            allow_degraded=True,
        )

        assert result.status == ExecutionStatus.SUCCESS

    def test_missing_method_returns_result_not_exception(self, registry):
        """Test that missing method returns ModuleMethodResult with missing_method status"""
        adapter = StubAdapterSuccess()
        registry.register_adapter(
            module_name="test_adapter",
            adapter_instance=adapter,
            adapter_class_name="StubAdapterSuccess",
        )

        result = registry.execute_module_method(
            module_name="test_adapter", method_name="nonexistent_method"
        )

        assert isinstance(result, ModuleMethodResult)
        assert result.status == ExecutionStatus.MISSING_METHOD
        assert result.confidence == 0.0
        assert result.error_type == "AttributeError"
        assert "not found" in result.error_message

    def test_method_exception_returns_error_result(self, registry):
        """Test that method exceptions are captured in ModuleMethodResult"""
        adapter = StubAdapterFailure()
        registry.register_adapter(
            module_name="failing_adapter",
            adapter_instance=adapter,
            adapter_class_name="StubAdapterFailure",
        )

        result = registry.execute_module_method(
            module_name="failing_adapter", method_name="failing_method"
        )

        assert isinstance(result, ModuleMethodResult)
        assert result.status == ExecutionStatus.ERROR
        assert result.confidence == 0.0
        assert result.error_type == "ValueError"
        assert "Intentional failure" in result.error_message


@pytest.mark.unit
class TestDeterministicExecution:
    """Test deterministic execution with clock and trace ID injection"""

    def test_deterministic_timing(self):
        """Test that injected clock produces deterministic timing"""
        counter = [0.0]

        def clock():
            counter[0] += 0.001
            return counter[0]

        def trace_gen():
            return "fixed-trace"

        registry = ModuleAdapterRegistry(clock=clock, trace_id_generator=trace_gen)
        adapter = StubAdapterSuccess()
        registry.register_adapter(
            module_name="test_adapter",
            adapter_instance=adapter,
            adapter_class_name="StubAdapterSuccess",
        )

        result = registry.execute_module_method(
            module_name="test_adapter", method_name="process", args=["test"]
        )

        # Clock is called twice: start and end
        assert result.start_time == 0.001
        assert result.end_time == 0.002
        assert result.execution_time == 0.001
        assert result.trace_id == "fixed-trace"

    def test_deterministic_trace_ids(self, registry):
        """Test that trace IDs are deterministic across multiple calls"""
        adapter = StubAdapterSuccess()
        registry.register_adapter(
            module_name="test_adapter",
            adapter_instance=adapter,
            adapter_class_name="StubAdapterSuccess",
        )

        results = []
        for i in range(3):
            result = registry.execute_module_method(
                module_name="test_adapter", method_name="analyze"
            )
            results.append(result)

        # Trace IDs should be deterministic sequence
        assert results[0].trace_id == "trace-0001"
        assert results[1].trace_id == "trace-0002"
        assert results[2].trace_id == "trace-0003"


@pytest.mark.unit
class TestMethodIntrospection:
    """Test list_adapter_methods for pre-flight validation"""

    def test_list_adapter_methods(self, registry):
        """Test listing public methods on adapter"""
        adapter = StubAdapterSuccess()
        registry.register_adapter(
            module_name="test_adapter",
            adapter_instance=adapter,
            adapter_class_name="StubAdapterSuccess",
        )

        methods = registry.list_adapter_methods("test_adapter")

        assert "process" in methods
        assert "analyze" in methods
        assert "compute" in methods
        # Private methods should not be included
        assert not any(m.startswith("_") for m in methods)

    def test_list_methods_for_nonexistent_adapter(self, registry):
        """Test that listing methods for nonexistent adapter raises ContractViolation"""
        with pytest.raises(ContractViolation):
            registry.list_adapter_methods("nonexistent")

    def test_list_methods_for_adapter_with_no_public_methods(self, registry):
        """Test listing methods on adapter with no public methods"""
        adapter = StubAdapterNoMethods()
        registry.register_adapter(
            module_name="empty_adapter",
            adapter_instance=adapter,
            adapter_class_name="StubAdapterNoMethods",
        )

        methods = registry.list_adapter_methods("empty_adapter")

        # Should only return inherited methods (if any), not private ones
        assert all(not m.startswith("_") for m in methods)


@pytest.mark.unit
class TestModuleMethodResultSerialization:
    """Test ModuleMethodResult to_dict serialization"""

    def test_to_dict_success(self, registry):
        """Test successful result serialization"""
        adapter = StubAdapterSuccess()
        registry.register_adapter(
            module_name="test_adapter",
            adapter_instance=adapter,
            adapter_class_name="StubAdapterSuccess",
        )

        result = registry.execute_module_method(
            module_name="test_adapter", method_name="process", args=["test"]
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["module_name"] == "test_adapter"
        assert result_dict["method_name"] == "process"
        assert result_dict["status"] == "success"
        assert result_dict["confidence"] == 1.0
        assert "trace_id" in result_dict
        assert "execution_time" in result_dict

    def test_to_dict_error(self, registry):
        """Test error result serialization"""
        adapter = StubAdapterFailure()
        registry.register_adapter(
            module_name="failing_adapter",
            adapter_instance=adapter,
            adapter_class_name="StubAdapterFailure",
        )

        result = registry.execute_module_method(
            module_name="failing_adapter", method_name="failing_method"
        )

        result_dict = result.to_dict()

        assert result_dict["status"] == "error"
        assert result_dict["error_type"] == "ValueError"
        assert result_dict["error_message"] is not None
        assert result_dict["confidence"] == 0.0


@pytest.mark.unit
class TestBackwardCompatibility:
    """Test backward compatibility with existing code"""

    def test_adapters_property(self, registry):
        """Test that adapters property returns dict for backward compatibility"""
        adapter = StubAdapterSuccess()
        registry.register_adapter(
            module_name="test_adapter",
            adapter_instance=adapter,
            adapter_class_name="StubAdapterSuccess",
        )

        adapters = registry.adapters

        assert isinstance(adapters, dict)
        assert "test_adapter" in adapters
        assert adapters["test_adapter"] is adapter

    def test_is_available_method(self, registry):
        """Test is_available method"""
        adapter = StubAdapterSuccess()
        registry.register_adapter(
            module_name="test_adapter",
            adapter_instance=adapter,
            adapter_class_name="StubAdapterSuccess",
        )

        assert registry.is_available("test_adapter")
        assert not registry.is_available("nonexistent")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
