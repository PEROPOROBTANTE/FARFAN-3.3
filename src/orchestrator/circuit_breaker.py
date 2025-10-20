# coding=utf-8
"""
Circuit Breaker - Fault Tolerance and Failure Management
=========================================================

Tracks failure counts, implements configurable failure thresholds and timeout periods,
exposes state checking methods, and provides manual and automatic reset functionality.

Author: FARFAN 3.0 Team
Version: 3.0.0
Python: 3.10+
"""

import logging
import time
from enum import Enum, IntEnum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


class CircuitState(IntEnum):
    """Circuit breaker states"""
    CLOSED = 0      # Normal operation
    OPEN = 1        # Blocking requests due to failures
    HALF_OPEN = 2   # Testing recovery


class FailureSeverity(Enum):
    """Failure severity classification"""
    TRANSIENT = "transient"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass
class FailureEvent:
    """Individual failure event record"""
    timestamp: float
    severity: FailureSeverity
    error_message: str
    adapter_name: str
    execution_time: float = 0.0


class CircuitBreaker:
    """
    Circuit breaker with configurable thresholds and automatic recovery
    
    Features:
    - Configurable failure threshold and timeout period
    - Per-adapter circuit state tracking
    - Automatic state transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
    - Manual and automatic reset functionality
    - State checking methods
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            half_open_max_calls: Maximum test calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        # Per-adapter tracking
        self.adapter_states: Dict[str, CircuitState] = {}
        self.adapter_failures: Dict[str, deque] = {}
        self.last_state_change: Dict[str, float] = {}
        self.half_open_calls: Dict[str, int] = {}
        self.success_counts: Dict[str, int] = {}
        self.failure_counts: Dict[str, int] = {}
        
        logger.info(
            f"CircuitBreaker initialized: threshold={failure_threshold}, "
            f"timeout={recovery_timeout}s, half_open_calls={half_open_max_calls}"
        )

    def _initialize_adapter(self, adapter_name: str):
        """Initialize tracking state for an adapter"""
        self.adapter_states[adapter_name] = CircuitState.CLOSED
        self.adapter_failures[adapter_name] = deque(maxlen=self.failure_threshold * 2)
        self.last_state_change[adapter_name] = time.time()
        self.half_open_calls[adapter_name] = 0
        self.success_counts[adapter_name] = 0
        self.failure_counts[adapter_name] = 0

    def can_execute(self, adapter_name: str) -> bool:
        """
        Check if adapter can execute based on circuit state
        
        Args:
            adapter_name: Name of adapter to check
            
        Returns:
            True if adapter can execute, False if circuit is open
        """
        if adapter_name not in self.adapter_states:
            self._initialize_adapter(adapter_name)
        
        state = self.adapter_states[adapter_name]
        
        if state == CircuitState.CLOSED:
            return True
        
        if state == CircuitState.OPEN:
            time_since_open = time.time() - self.last_state_change[adapter_name]
            if time_since_open >= self.recovery_timeout:
                self._transition_to_half_open(adapter_name)
                return True
            return False
        
        if state == CircuitState.HALF_OPEN:
            if self.half_open_calls[adapter_name] < self.half_open_max_calls:
                self.half_open_calls[adapter_name] += 1
                return True
            return False
        
        return False

    def record_success(self, adapter_name: str, execution_time: float = 0.0):
        """
        Record successful execution
        
        Args:
            adapter_name: Name of adapter that succeeded
            execution_time: Execution time in seconds
        """
        if adapter_name not in self.adapter_states:
            self._initialize_adapter(adapter_name)
        
        self.success_counts[adapter_name] += 1
        state = self.adapter_states[adapter_name]
        
        if state == CircuitState.HALF_OPEN:
            if self.half_open_calls[adapter_name] >= self.half_open_max_calls:
                self._transition_to_closed(adapter_name)
        
        logger.debug(f"{adapter_name}: Success recorded (state={state.name})")

    def record_failure(
        self,
        adapter_name: str,
        error: str,
        execution_time: float = 0.0,
        severity: FailureSeverity = FailureSeverity.CRITICAL
    ):
        """
        Record failed execution
        
        Args:
            adapter_name: Name of adapter that failed
            error: Error message
            execution_time: Execution time in seconds
            severity: Failure severity level
        """
        if adapter_name not in self.adapter_states:
            self._initialize_adapter(adapter_name)
        
        failure = FailureEvent(
            timestamp=time.time(),
            severity=severity,
            error_message=str(error),
            adapter_name=adapter_name,
            execution_time=execution_time
        )
        
        self.adapter_failures[adapter_name].append(failure)
        self.failure_counts[adapter_name] += 1
        
        state = self.adapter_states[adapter_name]
        
        if state == CircuitState.CLOSED:
            recent_failures = self._count_recent_failures(adapter_name)
            if recent_failures >= self.failure_threshold:
                self._transition_to_open(adapter_name)
        
        elif state == CircuitState.HALF_OPEN:
            self._transition_to_open(adapter_name)
        
        logger.warning(
            f"{adapter_name}: Failure recorded (state={state.name}, "
            f"recent={self._count_recent_failures(adapter_name)})"
        )

    def _count_recent_failures(self, adapter_name: str, window: float = 60.0) -> int:
        """Count failures in recent time window"""
        if adapter_name not in self.adapter_failures:
            return 0
        
        cutoff = time.time() - window
        return sum(
            1 for f in self.adapter_failures[adapter_name]
            if f.timestamp >= cutoff
        )

    def _transition_to_open(self, adapter_name: str):
        """Transition adapter to OPEN state"""
        self.adapter_states[adapter_name] = CircuitState.OPEN
        self.last_state_change[adapter_name] = time.time()
        self.half_open_calls[adapter_name] = 0
        logger.warning(f"{adapter_name}: Circuit OPENED")

    def _transition_to_half_open(self, adapter_name: str):
        """Transition adapter to HALF_OPEN state"""
        self.adapter_states[adapter_name] = CircuitState.HALF_OPEN
        self.last_state_change[adapter_name] = time.time()
        self.half_open_calls[adapter_name] = 0
        logger.info(f"{adapter_name}: Circuit HALF_OPEN (testing recovery)")

    def _transition_to_closed(self, adapter_name: str):
        """Transition adapter to CLOSED state"""
        self.adapter_states[adapter_name] = CircuitState.CLOSED
        self.last_state_change[adapter_name] = time.time()
        self.half_open_calls[adapter_name] = 0
        logger.info(f"{adapter_name}: Circuit CLOSED (recovered)")

    def get_state(self, adapter_name: str) -> CircuitState:
        """Get current circuit state for adapter"""
        if adapter_name not in self.adapter_states:
            self._initialize_adapter(adapter_name)
        return self.adapter_states[adapter_name]

    def is_open(self, adapter_name: str) -> bool:
        """Check if circuit is open"""
        return self.get_state(adapter_name) == CircuitState.OPEN

    def is_closed(self, adapter_name: str) -> bool:
        """Check if circuit is closed"""
        return self.get_state(adapter_name) == CircuitState.CLOSED

    def is_half_open(self, adapter_name: str) -> bool:
        """Check if circuit is half-open"""
        return self.get_state(adapter_name) == CircuitState.HALF_OPEN

    def get_adapter_status(self, adapter_name: str) -> Dict[str, Any]:
        """
        Get detailed status for an adapter
        
        Returns:
            Dictionary with state, counts, and timing information
        """
        if adapter_name not in self.adapter_states:
            return {"error": "Adapter not initialized"}
        
        total_calls = self.success_counts[adapter_name] + self.failure_counts[adapter_name]
        success_rate = (
            self.success_counts[adapter_name] / total_calls
            if total_calls > 0 else 0.0
        )
        
        return {
            "adapter": adapter_name,
            "state": self.adapter_states[adapter_name].name,
            "success_count": self.success_counts[adapter_name],
            "failure_count": self.failure_counts[adapter_name],
            "success_rate": success_rate,
            "recent_failures": self._count_recent_failures(adapter_name),
            "time_since_state_change": time.time() - self.last_state_change[adapter_name]
        }

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all initialized adapters"""
        return {
            adapter: self.get_adapter_status(adapter)
            for adapter in self.adapter_states.keys()
        }

    def reset_adapter(self, adapter_name: str):
        """
        Manually reset an adapter's circuit
        
        Args:
            adapter_name: Name of adapter to reset
        """
        if adapter_name in self.adapter_states:
            self._initialize_adapter(adapter_name)
            logger.info(f"{adapter_name}: Circuit manually reset")

    def reset_all(self):
        """Reset all adapter circuits"""
        for adapter in list(self.adapter_states.keys()):
            self.reset_adapter(adapter)
        logger.info("All circuits reset")


if __name__ == "__main__":
    breaker = CircuitBreaker()
    print("Circuit Breaker initialized successfully")
    print(f"Failure threshold: {breaker.failure_threshold}")
    print(f"Recovery timeout: {breaker.recovery_timeout}s")
