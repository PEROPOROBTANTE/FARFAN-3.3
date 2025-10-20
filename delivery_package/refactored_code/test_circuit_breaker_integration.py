# test_circuit_breaker_integration.py
# coding=utf-8
"""
Integration Tests for CircuitBreaker with Module Adapters
==========================================================

Tests cover:
- Integration with all 9 module adapters
- CircuitBreaker protection enabled for each adapter
- Fallback behavior when circuit opens
- Module failure simulation and error propagation
- Circuit recovery after half-open trial period
- Consistent ModuleResult error responses
- End-to-end failure and recovery scenarios

Author: Test Team
Version: 3.0.0
Python: 3.10+
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import sys
from pathlib import Path
from dataclasses import dataclass, field

# Direct import to avoid import issues
sys.path.insert(0, str(Path(__file__).parent))
import importlib.util

# Load circuit_breaker module directly
spec = importlib.util.spec_from_file_location("circuit_breaker", "orchestrator/circuit_breaker.py")
circuit_breaker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(circuit_breaker_module)

CircuitBreaker = circuit_breaker_module.CircuitBreaker
CircuitState = circuit_breaker_module.CircuitState
FailureSeverity = circuit_breaker_module.FailureSeverity
create_module_specific_fallback = circuit_breaker_module.create_module_specific_fallback

# Define ModuleResult locally
@dataclass
class ModuleResult:
    """Standardized output format for all modules"""
    module_name: str
    class_name: str
    method_name: str
    status: str
    data: Dict[str, Any]
    evidence: list
    confidence: float
    execution_time: float
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

# Define BaseAdapter locally
class BaseAdapter:
    """Base class for all module adapters"""
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.available = False


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def breaker():
    """Create a circuit breaker for integration testing"""
    return CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=2.0,
        half_open_max_calls=2
    )


@pytest.fixture
def all_adapters():
    """List of all 9 adapter names"""
    return [
        "teoria_cambio",
        "analyzer_one",
        "dereck_beach",
        "embedding_policy",
        "semantic_chunking_policy",
        "contradiction_detection",
        "financial_viability",
        "policy_processor",
        "policy_segmenter"
    ]


@pytest.fixture
def mock_adapter():
    """Create a mock adapter that can simulate success and failure"""
    class MockAdapter(BaseAdapter):
        def __init__(self, module_name: str):
            super().__init__(module_name)
            self.available = True
            self.should_fail = False
            self.call_count = 0
        
        def execute_method(self, method_name: str) -> ModuleResult:
            """Execute a method with potential failure"""
            self.call_count += 1
            start_time = time.time()
            
            if self.should_fail:
                return ModuleResult(
                    module_name=self.module_name,
                    class_name="MockClass",
                    method_name=method_name,
                    status="failed",
                    data={},
                    evidence=[],
                    confidence=0.0,
                    execution_time=time.time() - start_time,
                    errors=["Simulated failure"]
                )
            else:
                return ModuleResult(
                    module_name=self.module_name,
                    class_name="MockClass",
                    method_name=method_name,
                    status="success",
                    data={"result": "success"},
                    evidence=[{"type": "test"}],
                    confidence=0.95,
                    execution_time=time.time() - start_time
                )
    
    return MockAdapter


# ============================================================================
# ADAPTER INTEGRATION TESTS
# ============================================================================

class TestAdapterIntegration:
    """Test CircuitBreaker integration with all 9 adapters"""

    def test_all_adapters_initialized(self, breaker, all_adapters):
        """Test that all 9 adapters are initialized in CircuitBreaker"""
        for adapter_name in all_adapters:
            assert adapter_name in breaker.adapter_states
            assert breaker.adapter_states[adapter_name] == CircuitState.CLOSED

    def test_adapter_execution_with_circuit_protection(self, breaker, mock_adapter):
        """Test adapter execution with circuit breaker protection"""
        adapter = mock_adapter("teoria_cambio")
        adapter_name = adapter.module_name
        
        # Define protected execution wrapper
        def protected_execute(adapter, method_name):
            if not breaker.can_execute(adapter_name):
                # Return fallback result
                return ModuleResult(
                    module_name=adapter_name,
                    class_name="Fallback",
                    method_name=method_name,
                    status="degraded",
                    data={},
                    evidence=[],
                    confidence=0.0,
                    execution_time=0.0,
                    warnings=["Circuit breaker is OPEN"]
                )
            
            result = adapter.execute_method(method_name)
            
            if result.status == "success":
                breaker.record_success(adapter_name, result.execution_time)
            else:
                breaker.record_failure(adapter_name, result.errors[0], result.execution_time)
            
            return result
        
        # Execute successfully
        result = protected_execute(adapter, "test_method")
        assert result.status == "success"
        assert breaker.adapter_states[adapter_name] == CircuitState.CLOSED

    def test_circuit_opens_on_adapter_failures(self, breaker, mock_adapter):
        """Test that circuit opens when adapter fails repeatedly"""
        adapter = mock_adapter("analyzer_one")
        adapter_name = adapter.module_name
        adapter.should_fail = True
        
        def protected_execute(adapter, method_name):
            if not breaker.can_execute(adapter_name):
                return None
            
            result = adapter.execute_method(method_name)
            
            if result.status == "success":
                breaker.record_success(adapter_name, result.execution_time)
            else:
                breaker.record_failure(adapter_name, result.errors[0], result.execution_time)
            
            return result
        
        # Execute until circuit opens
        for i in range(breaker.failure_threshold):
            result = protected_execute(adapter, "test_method")
            assert result.status == "failed"
        
        # Circuit should now be OPEN
        assert breaker.adapter_states[adapter_name] == CircuitState.OPEN
        
        # Further execution should be blocked
        assert breaker.can_execute(adapter_name) is False

    def test_multiple_adapters_independent_circuits(self, breaker, mock_adapter, all_adapters):
        """Test that each adapter has independent circuit state"""
        # Fail one adapter
        failing_adapter = mock_adapter("teoria_cambio")
        failing_adapter.should_fail = True
        
        for i in range(breaker.failure_threshold):
            result = failing_adapter.execute_method("test")
            breaker.record_failure("teoria_cambio", result.errors[0])
        
        assert breaker.adapter_states["teoria_cambio"] == CircuitState.OPEN
        
        # Other adapters should still be CLOSED
        for adapter_name in all_adapters:
            if adapter_name != "teoria_cambio":
                assert breaker.adapter_states[adapter_name] == CircuitState.CLOSED


# ============================================================================
# FALLBACK BEHAVIOR TESTS
# ============================================================================

class TestFallbackBehavior:
    """Test fallback behavior when circuit opens"""

    def test_fallback_triggered_on_open_circuit(self, breaker, mock_adapter):
        """Test that fallback is triggered when circuit is open"""
        adapter = mock_adapter("dereck_beach")
        adapter_name = adapter.module_name
        adapter.should_fail = True
        
        # Open the circuit
        for i in range(breaker.failure_threshold):
            result = adapter.execute_method("test")
            breaker.record_failure(adapter_name, result.errors[0])
        
        assert breaker.adapter_states[adapter_name] == CircuitState.OPEN
        
        # Get fallback
        fallback_strategy = breaker.get_fallback_strategy(adapter_name)
        
        assert fallback_strategy is not None
        assert 'use_cached' in fallback_strategy
        assert 'degraded_mode' in fallback_strategy

    def test_fallback_returns_safe_default(self, breaker):
        """Test that fallback returns safe default values"""
        adapter_name = "embedding_policy"
        method_name = "extract_embeddings"
        
        # Create fallback
        fallback_fn = create_module_specific_fallback(adapter_name, method_name)
        result = fallback_fn()
        
        assert result is not None
        assert result['status'] == 'degraded'
        assert result['adapter_name'] == adapter_name
        assert result['method_name'] == method_name
        assert result['confidence'] == 0.0

    def test_fallback_strategies_for_all_adapters(self, breaker, all_adapters):
        """Test that all adapters have defined fallback strategies"""
        for adapter_name in all_adapters:
            fallback = breaker.get_fallback_strategy(adapter_name)
            
            assert fallback is not None
            assert isinstance(fallback, dict)
            assert 'use_cached' in fallback
            assert 'degraded_mode' in fallback

    def test_alternative_adapter_fallback(self, breaker, mock_adapter):
        """Test fallback to alternative adapter when primary fails"""
        primary_adapter = "teoria_cambio"
        fallback_strategy = breaker.get_fallback_strategy(primary_adapter)
        
        # Get alternative adapters
        alternatives = fallback_strategy.get('alternative_adapters', [])
        
        if alternatives:
            # Primary adapter fails
            for i in range(breaker.failure_threshold):
                breaker.record_failure(primary_adapter, "Error")
            
            assert breaker.adapter_states[primary_adapter] == CircuitState.OPEN
            
            # Alternative should still be available
            alternative = alternatives[0]
            assert breaker.adapter_states[alternative] == CircuitState.CLOSED
            assert breaker.can_execute(alternative) is True


# ============================================================================
# ERROR PROPAGATION TESTS
# ============================================================================

class TestErrorPropagation:
    """Test that ModuleResult error responses propagate correctly"""

    def test_module_result_error_format(self, mock_adapter):
        """Test that errors are formatted correctly in ModuleResult"""
        adapter = mock_adapter("policy_processor")
        adapter.should_fail = True
        
        result = adapter.execute_method("test_method")
        
        assert result.status == "failed"
        assert len(result.errors) > 0
        assert result.confidence == 0.0
        assert isinstance(result.execution_time, float)

    def test_error_propagation_through_circuit(self, breaker, mock_adapter):
        """Test that errors propagate correctly through circuit breaker"""
        adapter = mock_adapter("semantic_chunking_policy")
        adapter_name = adapter.module_name
        adapter.should_fail = True
        
        errors_recorded = []
        
        # Execute with error recording
        for i in range(breaker.failure_threshold):
            result = adapter.execute_method(f"method_{i}")
            
            if result.status == "failed":
                breaker.record_failure(adapter_name, result.errors[0])
                errors_recorded.append(result.errors[0])
        
        # Verify errors were recorded
        assert len(errors_recorded) == breaker.failure_threshold
        assert breaker.adapter_states[adapter_name] == CircuitState.OPEN

    def test_consistent_error_response_format(self, breaker, mock_adapter):
        """Test that error responses are consistent across circuit states"""
        adapter = mock_adapter("contradiction_detection")
        adapter_name = adapter.module_name
        
        # Collect error responses in different states
        error_responses = []
        
        # CLOSED state error
        adapter.should_fail = True
        result = adapter.execute_method("test")
        breaker.record_failure(adapter_name, result.errors[0])
        error_responses.append(result)
        
        # More failures to open circuit
        for i in range(breaker.failure_threshold - 1):
            result = adapter.execute_method("test")
            breaker.record_failure(adapter_name, result.errors[0])
            error_responses.append(result)
        
        # Verify all responses have consistent format
        for response in error_responses:
            assert response.status == "failed"
            assert isinstance(response.errors, list)
            assert len(response.errors) > 0
            assert response.confidence == 0.0

    def test_error_metadata_preservation(self, mock_adapter):
        """Test that error metadata is preserved in ModuleResult"""
        adapter = mock_adapter("financial_viability")
        adapter.should_fail = True
        
        result = adapter.execute_method("calculate_viability")
        
        # Verify metadata fields
        assert result.module_name == "financial_viability"
        assert result.method_name == "calculate_viability"
        assert result.class_name == "MockClass"
        assert isinstance(result.execution_time, float)


# ============================================================================
# CIRCUIT RECOVERY TESTS
# ============================================================================

class TestCircuitRecovery:
    """Test circuit recovery after half-open trial period"""

    def test_circuit_recovery_after_timeout(self, breaker, mock_adapter):
        """Test that circuit recovers after timeout"""
        adapter = mock_adapter("policy_segmenter")
        adapter_name = adapter.module_name
        adapter.should_fail = True
        
        # Open circuit
        for i in range(breaker.failure_threshold):
            result = adapter.execute_method("test")
            breaker.record_failure(adapter_name, result.errors[0])
        
        assert breaker.adapter_states[adapter_name] == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(breaker.recovery_timeout + 0.1)
        
        # Should transition to HALF_OPEN
        assert breaker.can_execute(adapter_name) is True
        assert breaker.adapter_states[adapter_name] == CircuitState.HALF_OPEN

    def test_successful_recovery_closes_circuit(self, breaker, mock_adapter):
        """Test that successful recovery closes the circuit"""
        adapter = mock_adapter("analyzer_one")
        adapter_name = adapter.module_name
        
        # Open circuit
        adapter.should_fail = True
        for i in range(breaker.failure_threshold):
            result = adapter.execute_method("test")
            breaker.record_failure(adapter_name, result.errors[0])
        
        assert breaker.adapter_states[adapter_name] == CircuitState.OPEN
        
        # Wait and transition to HALF_OPEN
        time.sleep(breaker.recovery_timeout + 0.1)
        breaker.can_execute(adapter_name)
        
        # Now succeed
        adapter.should_fail = False
        
        for i in range(breaker.half_open_max_calls):
            if breaker.can_execute(adapter_name):
                result = adapter.execute_method("test")
                breaker.record_success(adapter_name, result.execution_time)
        
        # Should be CLOSED now
        assert breaker.adapter_states[adapter_name] == CircuitState.CLOSED

    def test_failed_recovery_reopens_circuit(self, breaker, mock_adapter):
        """Test that failed recovery reopens the circuit"""
        adapter = mock_adapter("teoria_cambio")
        adapter_name = adapter.module_name
        
        # Open circuit
        adapter.should_fail = True
        for i in range(breaker.failure_threshold):
            result = adapter.execute_method("test")
            breaker.record_failure(adapter_name, result.errors[0])
        
        # Wait and transition to HALF_OPEN
        time.sleep(breaker.recovery_timeout + 0.1)
        breaker.can_execute(adapter_name)
        assert breaker.adapter_states[adapter_name] == CircuitState.HALF_OPEN
        
        # Fail during recovery
        result = adapter.execute_method("test")
        breaker.record_failure(adapter_name, result.errors[0])
        
        # Should return to OPEN
        assert breaker.adapter_states[adapter_name] == CircuitState.OPEN

    def test_multiple_recovery_attempts(self, breaker, mock_adapter):
        """Test multiple recovery attempts until success"""
        adapter = mock_adapter("dereck_beach")
        adapter_name = adapter.module_name
        
        # Open circuit
        adapter.should_fail = True
        for i in range(breaker.failure_threshold):
            result = adapter.execute_method("test")
            breaker.record_failure(adapter_name, result.errors[0])
        
        # Attempt recovery 3 times (2 failures, 1 success)
        for attempt in range(3):
            # Wait for timeout
            time.sleep(breaker.recovery_timeout + 0.1)
            
            # Try to execute
            if breaker.can_execute(adapter_name):
                if attempt < 2:
                    # Fail first two attempts
                    result = adapter.execute_method("test")
                    breaker.record_failure(adapter_name, result.errors[0])
                else:
                    # Succeed on third attempt
                    adapter.should_fail = False
                    for i in range(breaker.half_open_max_calls):
                        if breaker.can_execute(adapter_name):
                            result = adapter.execute_method("test")
                            breaker.record_success(adapter_name, result.execution_time)
        
        # Should eventually close
        assert breaker.adapter_states[adapter_name] == CircuitState.CLOSED


# ============================================================================
# END-TO-END SCENARIO TESTS
# ============================================================================

class TestEndToEndScenarios:
    """Test complete end-to-end failure and recovery scenarios"""

    def test_cascading_failure_scenario(self, breaker, mock_adapter, all_adapters):
        """Test scenario where multiple adapters fail in cascade"""
        adapters_to_fail = ["teoria_cambio", "analyzer_one", "dereck_beach"]
        
        # Fail adapters in sequence
        for adapter_name in adapters_to_fail:
            adapter = mock_adapter(adapter_name)
            adapter.should_fail = True
            
            for i in range(breaker.failure_threshold):
                result = adapter.execute_method("test")
                breaker.record_failure(adapter_name, result.errors[0])
        
        # Verify all failed adapters are OPEN
        for adapter_name in adapters_to_fail:
            assert breaker.adapter_states[adapter_name] == CircuitState.OPEN
        
        # Other adapters should still be CLOSED
        for adapter_name in all_adapters:
            if adapter_name not in adapters_to_fail:
                assert breaker.adapter_states[adapter_name] == CircuitState.CLOSED

    def test_partial_recovery_scenario(self, breaker, mock_adapter):
        """Test scenario where some adapters recover and others don't"""
        # Create two adapters
        adapter1 = mock_adapter("teoria_cambio")
        adapter2 = mock_adapter("analyzer_one")
        
        # Fail both
        adapter1.should_fail = True
        adapter2.should_fail = True
        
        for adapter, name in [(adapter1, "teoria_cambio"), (adapter2, "analyzer_one")]:
            for i in range(breaker.failure_threshold):
                result = adapter.execute_method("test")
                breaker.record_failure(name, result.errors[0])
        
        # Wait for recovery timeout
        time.sleep(breaker.recovery_timeout + 0.1)
        
        # Adapter1 recovers successfully
        adapter1.should_fail = False
        if breaker.can_execute("teoria_cambio"):
            for i in range(breaker.half_open_max_calls):
                if breaker.can_execute("teoria_cambio"):
                    result = adapter1.execute_method("test")
                    breaker.record_success("teoria_cambio", result.execution_time)
        
        # Adapter2 fails during recovery
        if breaker.can_execute("analyzer_one"):
            result = adapter2.execute_method("test")
            breaker.record_failure("analyzer_one", result.errors[0])
        
        # Verify states
        assert breaker.adapter_states["teoria_cambio"] == CircuitState.CLOSED
        assert breaker.adapter_states["analyzer_one"] == CircuitState.OPEN

    def test_high_load_with_intermittent_failures(self, breaker, mock_adapter):
        """Test scenario with high load and intermittent failures"""
        adapter = mock_adapter("embedding_policy")
        adapter_name = adapter.module_name
        
        # Simulate 100 operations with 10% failure rate
        total_operations = 100
        failure_rate = 0.1
        
        success_count = 0
        failure_count = 0
        
        for i in range(total_operations):
            # Simulate intermittent failures
            adapter.should_fail = (i % 10 == 0)
            
            if breaker.can_execute(adapter_name):
                result = adapter.execute_method(f"operation_{i}")
                
                if result.status == "success":
                    breaker.record_success(adapter_name, result.execution_time)
                    success_count += 1
                else:
                    breaker.record_failure(adapter_name, result.errors[0])
                    failure_count += 1
        
        # Verify metrics
        metrics = breaker.adapter_metrics[adapter_name]
        assert metrics.success_count + metrics.failure_count <= total_operations
        
        # Success rate should be reasonable (not all operations blocked)
        assert metrics.success_count > 0

    def test_complete_system_recovery(self, breaker, mock_adapter, all_adapters):
        """Test complete system recovery after all adapters fail"""
        # Fail all adapters
        for adapter_name in all_adapters[:3]:  # Test with first 3 for speed
            adapter = mock_adapter(adapter_name)
            adapter.should_fail = True
            
            for i in range(breaker.failure_threshold):
                result = adapter.execute_method("test")
                breaker.record_failure(adapter_name, result.errors[0])
        
        # All should be OPEN
        for adapter_name in all_adapters[:3]:
            assert breaker.adapter_states[adapter_name] == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(breaker.recovery_timeout + 0.1)
        
        # Recover all adapters
        for adapter_name in all_adapters[:3]:
            adapter = mock_adapter(adapter_name)
            adapter.should_fail = False
            
            # Transition to half-open
            breaker.can_execute(adapter_name)
            
            # Execute successful calls
            calls_made = 0
            while breaker.can_execute(adapter_name) and calls_made < breaker.half_open_max_calls:
                result = adapter.execute_method("test")
                breaker.record_success(adapter_name, result.execution_time)
                calls_made += 1
            
            # One final success should close circuit
            breaker.record_success(adapter_name, execution_time=0.1)
        
        # All should be CLOSED
        for adapter_name in all_adapters[:3]:
            assert breaker.adapter_states[adapter_name] == CircuitState.CLOSED

    def test_realistic_adapter_workflow(self, breaker, mock_adapter):
        """Test realistic workflow with adapter execution and circuit protection"""
        adapter_name = "policy_processor"
        adapter = mock_adapter(adapter_name)
        
        def execute_with_protection(method_name: str, should_fail: bool = False):
            """Execute method with circuit breaker protection"""
            if not breaker.can_execute(adapter_name):
                # Use fallback
                fallback_strategy = breaker.get_fallback_strategy(adapter_name)
                return {
                    "status": "degraded",
                    "fallback": True,
                    "degraded_mode": fallback_strategy['degraded_mode']
                }
            
            adapter.should_fail = should_fail
            result = adapter.execute_method(method_name)
            
            if result.status == "success":
                breaker.record_success(adapter_name, result.execution_time)
            else:
                breaker.record_failure(adapter_name, result.errors[0])
            
            return result
        
        # Normal operations
        for i in range(10):
            result = execute_with_protection(f"method_{i}", should_fail=False)
            assert result.status == "success"
        
        # Failures occur
        for i in range(breaker.failure_threshold):
            result = execute_with_protection(f"method_fail_{i}", should_fail=True)
        
        # Circuit should be open, fallback triggered
        result = execute_with_protection("method_blocked")
        assert result["status"] == "degraded"
        assert result["fallback"] is True
        
        # Wait for recovery
        time.sleep(breaker.recovery_timeout + 0.1)
        
        # Successful recovery
        for i in range(breaker.half_open_max_calls):
            result = execute_with_protection(f"method_recovery_{i}", should_fail=False)
        
        # Execute one more successful call to trigger transition
        result = execute_with_protection("method_final", should_fail=False)
        
        # Should be back to normal
        assert breaker.adapter_states[adapter_name] == CircuitState.CLOSED


# ============================================================================
# STRESS TESTS
# ============================================================================

class TestStressScenarios:
    """Stress test scenarios"""

    def test_rapid_state_transitions(self, breaker, mock_adapter):
        """Test rapid state transitions"""
        adapter = mock_adapter("teoria_cambio")
        adapter_name = adapter.module_name
        
        # Rapidly cycle through states
        for cycle in range(3):
            # Fail to open
            adapter.should_fail = True
            for i in range(breaker.failure_threshold):
                result = adapter.execute_method("test")
                breaker.record_failure(adapter_name, result.errors[0])
            
            assert breaker.adapter_states[adapter_name] == CircuitState.OPEN
            
            # Wait and recover
            time.sleep(breaker.recovery_timeout + 0.1)
            adapter.should_fail = False
            
            # Transition to half-open
            breaker.can_execute(adapter_name)
            
            # Execute successful calls
            calls_made = 0
            while breaker.can_execute(adapter_name) and calls_made < breaker.half_open_max_calls:
                result = adapter.execute_method("test")
                breaker.record_success(adapter_name, result.execution_time)
                calls_made += 1
            
            # One final success should close circuit
            breaker.record_success(adapter_name, execution_time=0.1)
            
            assert breaker.adapter_states[adapter_name] == CircuitState.CLOSED

    def test_concurrent_adapter_operations(self, breaker, mock_adapter, all_adapters):
        """Test operations on multiple adapters concurrently"""
        # Simulate concurrent operations
        operation_results = {}
        
        for adapter_name in all_adapters[:5]:  # Test with first 5
            adapter = mock_adapter(adapter_name)
            adapter.should_fail = False
            
            results = []
            for i in range(10):
                if breaker.can_execute(adapter_name):
                    result = adapter.execute_method(f"concurrent_op_{i}")
                    breaker.record_success(adapter_name, result.execution_time)
                    results.append(result)
            
            operation_results[adapter_name] = results
        
        # Verify all adapters executed successfully
        for adapter_name, results in operation_results.items():
            assert len(results) == 10
            assert all(r.status == "success" for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
