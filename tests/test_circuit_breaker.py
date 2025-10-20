"""
Circuit Breaker State Transition Tests
=======================================

Tests CircuitBreaker class state transitions:
- Sequential failures triggering OPEN state
- Circuit preventing execution during OPEN state
- Successful reset to CLOSED state after timeout
- HALF_OPEN state testing during recovery
- Performance metrics tracking

Run with: pytest tests/test_circuit_breaker.py -v
"""

import pytest
import time
from orchestrator.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    FailureSeverity
)


class TestCircuitBreakerStates:
    """Test suite for CircuitBreaker state transitions"""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker with test configuration"""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=2.0,
            half_open_max_calls=2
        )

    @pytest.fixture
    def adapter_name(self):
        """Test adapter name"""
        return "teoria_cambio"

    def test_initial_state_is_closed(self, breaker, adapter_name):
        """Test that circuit breaker starts in CLOSED state"""
        assert breaker.adapter_states[adapter_name] == CircuitState.CLOSED
        assert breaker.can_execute(adapter_name) is True

    def test_circuit_opens_after_threshold_failures(self, breaker, adapter_name):
        """Test that circuit opens after reaching failure threshold"""
        # Initial state should be CLOSED
        assert breaker.adapter_states[adapter_name] == CircuitState.CLOSED
        
        # Record failures up to threshold
        for i in range(breaker.failure_threshold):
            breaker.record_failure(
                adapter_name,
                f"Test error {i}",
                execution_time=0.1,
                severity=FailureSeverity.CRITICAL
            )
        
        # Circuit should now be OPEN
        assert breaker.adapter_states[adapter_name] == CircuitState.OPEN
        print(f"\n✓ Circuit OPENED after {breaker.failure_threshold} failures")

    def test_circuit_blocks_execution_when_open(self, breaker, adapter_name):
        """Test that circuit prevents execution when in OPEN state"""
        # Force circuit to OPEN state
        for i in range(breaker.failure_threshold):
            breaker.record_failure(
                adapter_name,
                f"Test error {i}",
                severity=FailureSeverity.CRITICAL
            )
        
        # Verify circuit is OPEN
        assert breaker.adapter_states[adapter_name] == CircuitState.OPEN
        
        # Attempt execution should be blocked
        assert breaker.can_execute(adapter_name) is False
        print(f"\n✓ Circuit correctly blocks execution in OPEN state")

    def test_circuit_transitions_to_half_open_after_timeout(self, breaker, adapter_name):
        """Test that circuit transitions to HALF_OPEN after recovery timeout"""
        # Force circuit to OPEN state
        for i in range(breaker.failure_threshold):
            breaker.record_failure(
                adapter_name,
                f"Test error {i}",
                severity=FailureSeverity.CRITICAL
            )
        
        assert breaker.adapter_states[adapter_name] == CircuitState.OPEN
        
        # Wait for recovery timeout
        print(f"\n⏳ Waiting {breaker.recovery_timeout}s for recovery timeout...")
        time.sleep(breaker.recovery_timeout + 0.1)
        
        # Check if we can execute (should trigger HALF_OPEN transition)
        can_exec = breaker.can_execute(adapter_name)
        
        assert can_exec is True
        assert breaker.adapter_states[adapter_name] == CircuitState.HALF_OPEN
        print(f"✓ Circuit transitioned to HALF_OPEN state")

    def test_circuit_closes_after_successful_recovery(self, breaker, adapter_name):
        """Test that circuit closes after successful test calls in HALF_OPEN state"""
        # Force circuit to OPEN state
        for i in range(breaker.failure_threshold):
            breaker.record_failure(
                adapter_name,
                f"Test error {i}",
                severity=FailureSeverity.CRITICAL
            )
        
        # Wait for recovery timeout
        time.sleep(breaker.recovery_timeout + 0.1)
        
        # Trigger HALF_OPEN transition
        breaker.can_execute(adapter_name)
        assert breaker.adapter_states[adapter_name] == CircuitState.HALF_OPEN
        
        # Record successful test calls
        for i in range(breaker.half_open_max_calls):
            if breaker.can_execute(adapter_name):
                breaker.record_success(adapter_name, execution_time=0.1)
        
        # Circuit should now be CLOSED
        assert breaker.adapter_states[adapter_name] == CircuitState.CLOSED
        print(f"\n✓ Circuit successfully CLOSED after recovery")

    def test_circuit_reopens_on_failure_during_half_open(self, breaker, adapter_name):
        """Test that circuit reopens if failure occurs during HALF_OPEN state"""
        # Force circuit to OPEN state
        for i in range(breaker.failure_threshold):
            breaker.record_failure(
                adapter_name,
                f"Test error {i}",
                severity=FailureSeverity.CRITICAL
            )
        
        # Wait for recovery timeout and transition to HALF_OPEN
        time.sleep(breaker.recovery_timeout + 0.1)
        breaker.can_execute(adapter_name)
        
        assert breaker.adapter_states[adapter_name] == CircuitState.HALF_OPEN
        
        # Record a failure during HALF_OPEN
        breaker.record_failure(
            adapter_name,
            "Recovery test failed",
            severity=FailureSeverity.CRITICAL
        )
        
        # Circuit should reopen
        assert breaker.adapter_states[adapter_name] == CircuitState.OPEN
        print(f"\n✓ Circuit correctly reopened after HALF_OPEN failure")

    def test_performance_metrics_tracking(self, breaker, adapter_name):
        """Test that performance metrics are correctly tracked"""
        # Record some successful executions
        for i in range(5):
            breaker.record_success(adapter_name, execution_time=0.1 + i * 0.01)
        
        # Record some failures
        for i in range(2):
            breaker.record_failure(
                adapter_name,
                f"Error {i}",
                execution_time=0.2,
                severity=FailureSeverity.TRANSIENT
            )
        
        # Get metrics
        metrics = breaker.adapter_metrics[adapter_name]
        
        assert metrics.success_count == 5
        assert metrics.failure_count == 2
        assert metrics.success_rate == pytest.approx(5/7, rel=0.01)
        assert len(metrics.response_times) == 7
        assert metrics.avg_response_time > 0
        
        print(f"\n✓ Metrics tracked: {metrics.success_count} successes, "
              f"{metrics.failure_count} failures, "
              f"{metrics.success_rate:.2%} success rate")

    def test_adapter_status_report(self, breaker, adapter_name):
        """Test that adapter status report provides correct information"""
        # Record some activity
        breaker.record_success(adapter_name, 0.1)
        breaker.record_failure(adapter_name, "Test error", 0.2)
        
        # Get status
        status = breaker.get_adapter_status(adapter_name)
        
        assert status["adapter"] == adapter_name
        assert status["state"] == "CLOSED"
        assert status["total_calls"] == 2
        assert status["successes"] == 1
        assert status["failures"] == 1
        assert "avg_response_time" in status
        assert "recent_failures" in status
        
        print(f"\n✓ Status report generated: {status['state']}, "
              f"{status['total_calls']} total calls")

    def test_reset_adapter(self, breaker, adapter_name):
        """Test that reset_adapter clears state and metrics"""
        # Generate some activity and failures
        for i in range(breaker.failure_threshold):
            breaker.record_failure(
                adapter_name,
                f"Error {i}",
                severity=FailureSeverity.CRITICAL
            )
        
        # Circuit should be OPEN
        assert breaker.adapter_states[adapter_name] == CircuitState.OPEN
        
        # Reset the adapter
        breaker.reset_adapter(adapter_name)
        
        # Should be back to CLOSED with clean metrics
        assert breaker.adapter_states[adapter_name] == CircuitState.CLOSED
        assert breaker.adapter_metrics[adapter_name].success_count == 0
        assert breaker.adapter_metrics[adapter_name].failure_count == 0
        
        print(f"\n✓ Adapter successfully reset to CLOSED state")

    def test_all_adapters_initialized(self, breaker):
        """Test that all 9 adapters are initialized"""
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
            assert adapter in breaker.adapter_states
            assert breaker.adapter_states[adapter] == CircuitState.CLOSED
        
        print(f"\n✓ All {len(expected_adapters)} adapters initialized correctly")

    def test_fallback_strategy_available(self, breaker, adapter_name):
        """Test that fallback strategies are defined for adapters"""
        fallback = breaker.get_fallback_strategy(adapter_name)
        
        assert "use_cached" in fallback
        assert "alternative_adapters" in fallback
        assert "degraded_mode" in fallback
        
        print(f"\n✓ Fallback strategy defined for {adapter_name}")

    def test_concurrent_adapter_states(self, breaker):
        """Test that different adapters maintain independent states"""
        adapter1 = "teoria_cambio"
        adapter2 = "analyzer_one"
        
        # Fail adapter1
        for i in range(breaker.failure_threshold):
            breaker.record_failure(adapter1, f"Error {i}", severity=FailureSeverity.CRITICAL)
        
        # adapter1 should be OPEN, adapter2 should be CLOSED
        assert breaker.adapter_states[adapter1] == CircuitState.OPEN
        assert breaker.adapter_states[adapter2] == CircuitState.CLOSED
        
        # adapter1 cannot execute, adapter2 can
        assert breaker.can_execute(adapter1) is False
        assert breaker.can_execute(adapter2) is True
        
        print(f"\n✓ Adapters maintain independent circuit states")


class TestCircuitBreakerEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker for edge case testing"""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            half_open_max_calls=2
        )

    def test_unknown_adapter_initialization(self, breaker):
        """Test that unknown adapters are automatically initialized"""
        unknown_adapter = "unknown_adapter"
        
        # Should auto-initialize on first access
        can_exec = breaker.can_execute(unknown_adapter)
        
        assert unknown_adapter in breaker.adapter_states
        assert breaker.adapter_states[unknown_adapter] == CircuitState.CLOSED
        assert can_exec is True

    def test_rapid_failure_sequence(self, breaker, adapter_name="teoria_cambio"):
        """Test handling of rapid consecutive failures"""
        # Record many failures rapidly
        for i in range(10):
            breaker.record_failure(
                adapter_name,
                f"Rapid failure {i}",
                execution_time=0.01,
                severity=FailureSeverity.CRITICAL
            )
        
        # Should be OPEN
        assert breaker.adapter_states[adapter_name] == CircuitState.OPEN
        
        # Recent failure count should be tracked
        recent_failures = breaker._count_recent_failures(adapter_name)
        assert recent_failures >= breaker.failure_threshold

    def test_mixed_severity_failures(self, breaker, adapter_name="teoria_cambio"):
        """Test handling of failures with different severities"""
        # Mix of transient and critical failures
        breaker.record_failure(adapter_name, "Transient", severity=FailureSeverity.TRANSIENT)
        breaker.record_failure(adapter_name, "Critical 1", severity=FailureSeverity.CRITICAL)
        breaker.record_failure(adapter_name, "Degraded", severity=FailureSeverity.DEGRADED)
        breaker.record_failure(adapter_name, "Critical 2", severity=FailureSeverity.CRITICAL)
        
        # All failures count toward threshold
        assert breaker._count_recent_failures(adapter_name) >= breaker.failure_threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
