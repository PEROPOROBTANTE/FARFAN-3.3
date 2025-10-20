"""
Test Module: Circuit Breaker Validation
========================================

This test validates the CircuitBreaker class from orchestrator/circuit_breaker.py:
- Instantiate CircuitBreaker
- Simulate consecutive failures to trigger OPEN state
- Verify circuit blocks subsequent calls during cooldown
- Simulate successful calls to test reset behavior
- Verify transition back to CLOSED state

Tests the full circuit breaker lifecycle: CLOSED → OPEN → HALF_OPEN → CLOSED

Author: Test Framework
Version: 1.0.0
"""

import pytest
import time
from orchestrator.circuit_breaker import CircuitBreaker, CircuitState, FailureSeverity


@pytest.fixture
def circuit_breaker():
    """Create a CircuitBreaker instance with short timeouts for testing"""
    return CircuitBreaker(
        failure_threshold=3,      # Open after 3 failures
        recovery_timeout=2.0,     # Wait 2 seconds before testing recovery
        half_open_max_calls=2     # Allow 2 test calls in half-open state
    )


@pytest.fixture
def test_adapter():
    """Name of adapter to test with"""
    return "teoria_cambio"


class TestCircuitBreakerLifecycle:
    """Test the complete circuit breaker lifecycle"""
    
    def test_initial_state_is_closed(self, circuit_breaker, test_adapter):
        """Test that circuit starts in CLOSED state"""
        status = circuit_breaker.get_adapter_status(test_adapter)
        assert status['state'] == CircuitState.CLOSED.name
        assert circuit_breaker.can_execute(test_adapter) is True
    
    def test_successful_execution_remains_closed(self, circuit_breaker, test_adapter):
        """Test that successful executions keep circuit CLOSED"""
        # Record multiple successes
        for _ in range(5):
            assert circuit_breaker.can_execute(test_adapter) is True
            circuit_breaker.record_success(test_adapter, execution_time=0.1)
        
        status = circuit_breaker.get_adapter_status(test_adapter)
        assert status['state'] == CircuitState.CLOSED.name
        assert status['successes'] == 5
        assert status['failures'] == 0
    
    def test_consecutive_failures_trigger_open(self, circuit_breaker, test_adapter):
        """Test that consecutive failures trigger OPEN state"""
        failure_threshold = circuit_breaker.failure_threshold
        
        # Record failures up to threshold - 1 (should remain CLOSED)
        for i in range(failure_threshold - 1):
            assert circuit_breaker.can_execute(test_adapter) is True
            circuit_breaker.record_failure(
                test_adapter,
                error=f"Test failure {i + 1}",
                execution_time=0.1,
                severity=FailureSeverity.CRITICAL
            )
        
        status = circuit_breaker.get_adapter_status(test_adapter)
        assert status['state'] == CircuitState.CLOSED.name
        
        # One more failure should trigger OPEN
        assert circuit_breaker.can_execute(test_adapter) is True
        circuit_breaker.record_failure(
            test_adapter,
            error=f"Test failure {failure_threshold}",
            execution_time=0.1,
            severity=FailureSeverity.CRITICAL
        )
        
        status = circuit_breaker.get_adapter_status(test_adapter)
        assert status['state'] == CircuitState.OPEN.name
        assert status['failures'] == failure_threshold
    
    def test_open_circuit_blocks_execution(self, circuit_breaker, test_adapter):
        """Test that OPEN circuit blocks execution during cooldown"""
        # Trigger circuit to OPEN
        for i in range(circuit_breaker.failure_threshold):
            circuit_breaker.record_failure(test_adapter, error=f"Failure {i + 1}")
        
        # Verify circuit is OPEN
        status = circuit_breaker.get_adapter_status(test_adapter)
        assert status['state'] == CircuitState.OPEN.name
        
        # Verify execution is blocked
        assert circuit_breaker.can_execute(test_adapter) is False
        assert circuit_breaker.can_execute(test_adapter) is False  # Multiple checks
    
    def test_half_open_transition_after_timeout(self, circuit_breaker, test_adapter):
        """Test that circuit transitions to HALF_OPEN after recovery timeout"""
        # Trigger circuit to OPEN
        for i in range(circuit_breaker.failure_threshold):
            circuit_breaker.record_failure(test_adapter, error=f"Failure {i + 1}")
        
        status = circuit_breaker.get_adapter_status(test_adapter)
        assert status['state'] == CircuitState.OPEN.name
        
        # Execution should be blocked during cooldown
        assert circuit_breaker.can_execute(test_adapter) is False
        
        # Wait for recovery timeout
        time.sleep(circuit_breaker.recovery_timeout + 0.1)
        
        # Should transition to HALF_OPEN and allow execution
        assert circuit_breaker.can_execute(test_adapter) is True
        
        status = circuit_breaker.get_adapter_status(test_adapter)
        assert status['state'] == CircuitState.HALF_OPEN.name
    
    def test_half_open_limits_concurrent_calls(self, circuit_breaker, test_adapter):
        """Test that HALF_OPEN state limits number of test calls"""
        # Trigger circuit to OPEN
        for i in range(circuit_breaker.failure_threshold):
            circuit_breaker.record_failure(test_adapter, error=f"Failure {i + 1}")
        
        # Wait for recovery timeout
        time.sleep(circuit_breaker.recovery_timeout + 0.1)
        
        # Should allow exactly half_open_max_calls + 1 (includes initial transition call)
        max_calls = circuit_breaker.half_open_max_calls
        allowed_calls = 0
        
        for _ in range(max_calls + 5):  # Try more than max
            if circuit_breaker.can_execute(test_adapter):
                allowed_calls += 1
        
        # The first can_execute() transitions to HALF_OPEN and allows execution
        # Then half_open_max_calls more executions are allowed
        assert allowed_calls == max_calls + 1
    
    def test_successful_calls_in_half_open_close_circuit(self, circuit_breaker, test_adapter):
        """Test that successful calls in HALF_OPEN close the circuit"""
        # Trigger circuit to OPEN
        for i in range(circuit_breaker.failure_threshold):
            circuit_breaker.record_failure(test_adapter, error=f"Failure {i + 1}")
        
        # Wait for recovery timeout
        time.sleep(circuit_breaker.recovery_timeout + 0.1)
        
        # Execute successful test calls in HALF_OPEN
        # Need to execute half_open_max_calls + 1 times (first call transitions to HALF_OPEN)
        for i in range(circuit_breaker.half_open_max_calls + 1):
            if circuit_breaker.can_execute(test_adapter):
                circuit_breaker.record_success(test_adapter, execution_time=0.05)
        
        # Circuit should now be CLOSED
        status = circuit_breaker.get_adapter_status(test_adapter)
        assert status['state'] == CircuitState.CLOSED.name
        
        # Should allow execution freely
        assert circuit_breaker.can_execute(test_adapter) is True
    
    def test_failure_in_half_open_reopens_circuit(self, circuit_breaker, test_adapter):
        """Test that failure in HALF_OPEN immediately reopens circuit"""
        # Trigger circuit to OPEN
        for i in range(circuit_breaker.failure_threshold):
            circuit_breaker.record_failure(test_adapter, error=f"Failure {i + 1}")
        
        # Wait for recovery timeout
        time.sleep(circuit_breaker.recovery_timeout + 0.1)
        
        # Verify we're in HALF_OPEN
        assert circuit_breaker.can_execute(test_adapter) is True
        status = circuit_breaker.get_adapter_status(test_adapter)
        assert status['state'] == CircuitState.HALF_OPEN.name
        
        # Record a failure
        circuit_breaker.record_failure(test_adapter, error="Recovery failed")
        
        # Should immediately go back to OPEN
        status = circuit_breaker.get_adapter_status(test_adapter)
        assert status['state'] == CircuitState.OPEN.name
        assert circuit_breaker.can_execute(test_adapter) is False


class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics and monitoring"""
    
    def test_success_rate_calculation(self, circuit_breaker, test_adapter):
        """Test that success rate is calculated correctly"""
        # Record mixed successes and failures
        circuit_breaker.record_success(test_adapter)
        circuit_breaker.record_success(test_adapter)
        circuit_breaker.record_failure(test_adapter, error="Test failure")
        circuit_breaker.record_success(test_adapter)
        
        status = circuit_breaker.get_adapter_status(test_adapter)
        assert status['successes'] == 3
        assert status['failures'] == 1
        assert status['success_rate'] == 0.75  # 3/4
    
    def test_recent_failures_tracking(self, circuit_breaker, test_adapter):
        """Test that recent failures are tracked correctly"""
        # Record failures
        for i in range(5):
            circuit_breaker.record_failure(test_adapter, error=f"Failure {i + 1}")
        
        status = circuit_breaker.get_adapter_status(test_adapter)
        assert status['recent_failures'] >= 3  # At least threshold failures
        assert status['failures'] == 5  # Total failures
    
    def test_get_all_status(self, circuit_breaker):
        """Test that get_all_status returns status for all adapters"""
        all_status = circuit_breaker.get_all_status()
        
        # Should have status for all 9 adapters
        assert len(all_status) == 9
        
        # Verify all expected adapters are present
        expected_adapters = [
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
        
        for adapter in expected_adapters:
            assert adapter in all_status
            assert all_status[adapter]['state'] == CircuitState.CLOSED.name


class TestCircuitBreakerFallback:
    """Test circuit breaker fallback strategies"""
    
    def test_fallback_strategy_exists_for_all_adapters(self, circuit_breaker):
        """Test that fallback strategies are defined for all adapters"""
        for adapter in circuit_breaker.adapters:
            fallback = circuit_breaker.get_fallback_strategy(adapter)
            
            assert fallback is not None
            assert isinstance(fallback, dict)
            assert 'use_cached' in fallback
            assert 'degraded_mode' in fallback
    
    def test_fallback_has_alternative_adapters(self, circuit_breaker):
        """Test that fallback strategies include alternative adapters where available"""
        fallback = circuit_breaker.get_fallback_strategy("teoria_cambio")
        
        assert 'alternative_adapters' in fallback
        # teoria_cambio should have analyzer_one as alternative
        assert fallback['alternative_adapters'] == ["analyzer_one"]
    
    def test_fallback_has_degraded_mode(self, circuit_breaker):
        """Test that fallback strategies include degraded mode"""
        fallback = circuit_breaker.get_fallback_strategy("embedding_policy")
        
        assert 'degraded_mode' in fallback
        # embedding_policy should fall back to keyword matching
        assert fallback['degraded_mode'] == "keyword_matching"


class TestCircuitBreakerReset:
    """Test circuit breaker reset functionality"""
    
    def test_reset_adapter_closes_circuit(self, circuit_breaker, test_adapter):
        """Test that reset_adapter closes an open circuit"""
        # Trigger circuit to OPEN
        for i in range(circuit_breaker.failure_threshold):
            circuit_breaker.record_failure(test_adapter, error=f"Failure {i + 1}")
        
        status = circuit_breaker.get_adapter_status(test_adapter)
        assert status['state'] == CircuitState.OPEN.name
        
        # Reset adapter
        circuit_breaker.reset_adapter(test_adapter)
        
        # Should be CLOSED and allow execution
        status = circuit_breaker.get_adapter_status(test_adapter)
        assert status['state'] == CircuitState.CLOSED.name
        assert circuit_breaker.can_execute(test_adapter) is True
        
        # Metrics should be reset
        assert status['successes'] == 0
        assert status['failures'] == 0
    
    def test_reset_all_closes_all_circuits(self, circuit_breaker):
        """Test that reset_all closes all circuits"""
        # Open multiple circuits
        adapters_to_fail = ["teoria_cambio", "analyzer_one", "dereck_beach"]
        
        for adapter in adapters_to_fail:
            for i in range(circuit_breaker.failure_threshold):
                circuit_breaker.record_failure(adapter, error=f"Failure {i + 1}")
        
        # Verify circuits are OPEN
        for adapter in adapters_to_fail:
            status = circuit_breaker.get_adapter_status(adapter)
            assert status['state'] == CircuitState.OPEN.name
        
        # Reset all
        circuit_breaker.reset_all()
        
        # All circuits should be CLOSED
        all_status = circuit_breaker.get_all_status()
        for adapter in adapters_to_fail:
            assert all_status[adapter]['state'] == CircuitState.CLOSED.name


class TestCircuitBreakerEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_unknown_adapter_initializes_on_first_use(self, circuit_breaker):
        """Test that unknown adapter is initialized on first use"""
        unknown_adapter = "unknown_test_adapter"
        
        # Should not fail, should initialize
        assert circuit_breaker.can_execute(unknown_adapter) is True
        
        status = circuit_breaker.get_adapter_status(unknown_adapter)
        assert status['state'] == CircuitState.CLOSED.name
    
    def test_zero_execution_time_handled(self, circuit_breaker, test_adapter):
        """Test that zero execution time doesn't cause issues"""
        circuit_breaker.record_success(test_adapter, execution_time=0.0)
        circuit_breaker.record_failure(test_adapter, error="Test", execution_time=0.0)
        
        status = circuit_breaker.get_adapter_status(test_adapter)
        assert status['avg_response_time'] >= 0.0  # Should handle gracefully
    
    def test_high_frequency_operations(self, circuit_breaker, test_adapter):
        """Test circuit breaker under high frequency operations"""
        # Simulate rapid operations
        for i in range(100):
            if i % 10 == 0:  # 10% failure rate
                circuit_breaker.record_failure(test_adapter, error=f"Failure {i}")
            else:
                circuit_breaker.record_success(test_adapter, execution_time=0.001)
        
        status = circuit_breaker.get_adapter_status(test_adapter)
        assert status['total_calls'] == 100
        assert 0.85 <= status['success_rate'] <= 0.95  # Should be around 90%
