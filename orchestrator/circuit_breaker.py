"""
Circuit Breaker - Fault tolerance and graceful degradation
Implements the Circuit Breaker pattern for resilient execution
"""
import logging
import time
from enum import Enum
from typing import Dict, Callable, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading

from .config import CONFIG

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failures exceed threshold, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitMetrics:
    """Metrics for a circuit"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    state_transitions: list = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return 1.0 - self.failure_rate


class CircuitBreaker:
    """
    Circuit Breaker implementation for fault-tolerant module execution.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject requests immediately
    - HALF_OPEN: Testing recovery, allow limited requests

    Thresholds:
    - failure_threshold: Number of consecutive failures before opening
    - timeout: Seconds before attempting recovery (OPEN -> HALF_OPEN)
    - half_open_timeout: Seconds in HALF_OPEN before closing
    """

    def __init__(
            self,
            failure_threshold: Optional[int] = None,
            timeout: Optional[int] = None,
            half_open_timeout: Optional[int] = None
    ):
        self.failure_threshold = failure_threshold or CONFIG.circuit_breaker_failure_threshold
        self.timeout = timeout or CONFIG.circuit_breaker_timeout
        self.half_open_timeout = half_open_timeout or CONFIG.circuit_breaker_half_open_timeout

        # Per-module circuit state
        self.circuits: Dict[str, CircuitState] = defaultdict(lambda: CircuitState.CLOSED)
        self.metrics: Dict[str, CircuitMetrics] = defaultdict(CircuitMetrics)
        self.locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)

        # Recent failure tracking (last 10 failures per module)
        self.recent_failures: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

        logger.info(f"Circuit Breaker initialized: failure_threshold={self.failure_threshold}, "
                    f"timeout={self.timeout}s, half_open_timeout={self.half_open_timeout}s")

    def call(
            self,
            module_name: str,
            func: Callable,
            *args,
            fallback: Optional[Callable] = None,
            **kwargs
    ) -> Any:
        """
        Execute a function with circuit breaker protection.

        Args:
            module_name: Name of the module being called
            func: Function to execute
            *args, **kwargs: Arguments to pass to func
            fallback: Optional fallback function if circuit is open

        Returns:
            Result from func or fallback

        Raises:
            CircuitBreakerError: If circuit is open and no fallback provided
        """
        with self.locks[module_name]:
            state = self.circuits[module_name]
            metrics = self.metrics[module_name]

            # Check if circuit should transition
            if state == CircuitState.OPEN:
                if self._should_attempt_reset(module_name):
                    self._transition_to_half_open(module_name)
                else:
                    # Circuit still open
                    logger.warning(f"Circuit OPEN for {module_name}, rejecting request")
                    if fallback:
                        logger.info(f"Executing fallback for {module_name}")
                        return fallback(*args, **kwargs)
                    else:
                        raise CircuitBreakerError(
                            f"Circuit breaker is OPEN for {module_name}"
                        )

            # Attempt execution
            metrics.total_requests += 1

            try:
                result = func(*args, **kwargs)

                # Success
                metrics.successful_requests += 1
                metrics.consecutive_failures = 0

                if state == CircuitState.HALF_OPEN:
                    self._transition_to_closed(module_name)

                return result

            except Exception as e:
                # Failure
                metrics.failed_requests += 1
                metrics.consecutive_failures += 1
                metrics.last_failure_time = time.time()

                # Record failure details
                self.recent_failures[module_name].append({
                    "time": time.time(),
                    "error": str(e),
                    "error_type": type(e).__name__
                })

                logger.error(f"Module {module_name} failed: {e}")

                # Check if should open circuit
                if metrics.consecutive_failures >= self.failure_threshold:
                    self._transition_to_open(module_name)

                # Try fallback
                if fallback:
                    logger.info(f"Executing fallback for {module_name} after failure")
                    return fallback(*args, **kwargs)
                else:
                    raise

    def _should_attempt_reset(self, module_name: str) -> bool:
        """Check if enough time has passed to attempt reset"""
        metrics = self.metrics[module_name]

        if metrics.last_failure_time is None:
            return True

        time_since_failure = time.time() - metrics.last_failure_time
        return time_since_failure >= self.timeout

    def _transition_to_open(self, module_name: str):
        """Transition circuit to OPEN state"""
        old_state = self.circuits[module_name]
        self.circuits[module_name] = CircuitState.OPEN

        self.metrics[module_name].state_transitions.append({
            "from": old_state.value,
            "to": CircuitState.OPEN.value,
            "time": time.time(),
            "reason": "failure_threshold_exceeded"
        })

        logger.warning(
            f"Circuit OPENED for {module_name} "
            f"({self.metrics[module_name].consecutive_failures} consecutive failures)"
        )

    def _transition_to_half_open(self, module_name: str):
        """Transition circuit to HALF_OPEN state"""
        old_state = self.circuits[module_name]
        self.circuits[module_name] = CircuitState.HALF_OPEN

        self.metrics[module_name].state_transitions.append({
            "from": old_state.value,
            "to": CircuitState.HALF_OPEN.value,
            "time": time.time(),
            "reason": "timeout_expired"
        })

        logger.info(f"Circuit HALF-OPEN for {module_name}, testing recovery")

    def _transition_to_closed(self, module_name: str):
        """Transition circuit to CLOSED state"""
        old_state = self.circuits[module_name]
        self.circuits[module_name] = CircuitState.CLOSED

        self.metrics[module_name].state_transitions.append({
            "from": old_state.value,
            "to": CircuitState.CLOSED.value,
            "time": time.time(),
            "reason": "recovery_confirmed"
        })

        logger.info(f"Circuit CLOSED for {module_name}, normal operation resumed")

    def reset(self, module_name: Optional[str] = None):
        """
        Reset circuit breaker state.

        Args:
            module_name: Specific module to reset, or None to reset all
        """
        if module_name:
            with self.locks[module_name]:
                self.circuits[module_name] = CircuitState.CLOSED
                self.metrics[module_name].consecutive_failures = 0
                logger.info(f"Circuit breaker reset for {module_name}")
        else:
            for name in list(self.circuits.keys()):
                with self.locks[name]:
                    self.circuits[name] = CircuitState.CLOSED
                    self.metrics[name].consecutive_failures = 0
            logger.info("All circuit breakers reset")

    def get_state(self, module_name: str) -> CircuitState:
        """Get current state of a circuit"""
        return self.circuits[module_name]

    def get_metrics(self, module_name: str) -> CircuitMetrics:
        """Get metrics for a circuit"""
        return self.metrics[module_name]

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuits"""
        return {
            module: {
                "state": self.circuits[module].value,
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "failure_rate": metrics.failure_rate,
                "success_rate": metrics.success_rate,
                "consecutive_failures": metrics.consecutive_failures,
                "last_failure_time": metrics.last_failure_time,
                "state_transitions": metrics.state_transitions,
                "recent_failures": list(self.recent_failures[module])
            }
            for module, metrics in self.metrics.items()
        }

    def is_available(self, module_name: str) -> bool:
        """Check if a module is available (circuit not open)"""
        return self.circuits[module_name] != CircuitState.OPEN

    def get_unavailable_modules(self) -> list[str]:
        """Get list of modules with open circuits"""
        return [
            module for module, state in self.circuits.items()
            if state == CircuitState.OPEN
        ]

    def get_degraded_modules(self) -> list[str]:
        """Get list of modules with high failure rates (>30%)"""
        return [
            module for module, metrics in self.metrics.items()
            if metrics.failure_rate > 0.3 and metrics.total_requests >= 5
        ]

    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        total_modules = len(self.metrics)
        if total_modules == 0:
            return {"status": "unknown", "message": "No modules executed yet"}

        available = sum(1 for state in self.circuits.values() if state != CircuitState.OPEN)
        degraded = len(self.get_degraded_modules())

        total_requests = sum(m.total_requests for m in self.metrics.values())
        total_successes = sum(m.successful_requests for m in self.metrics.values())

        overall_success_rate = total_successes / total_requests if total_requests > 0 else 1.0

        status = "healthy"
        if available < total_modules * 0.5:
            status = "critical"
        elif degraded > 0 or overall_success_rate < 0.8:
            status = "degraded"

        return {
            "status": status,
            "total_modules": total_modules,
            "available_modules": available,
            "unavailable_modules": total_modules - available,
            "degraded_modules": degraded,
            "overall_success_rate": overall_success_rate,
            "total_requests": total_requests,
            "total_successes": total_successes,
            "total_failures": total_requests - total_successes
        }


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open and no fallback provided"""
    pass


# Fallback functions for graceful degradation

def default_fallback(*args, **kwargs) -> Dict[str, Any]:
    """Default fallback returns empty result with degraded status"""
    return {
        "status": "degraded",
        "message": "Module unavailable, using fallback",
        "evidence": {},
        "confidence": 0.0
    }


def create_module_specific_fallback(module_name: str) -> Callable:
    """Create a module-specific fallback function"""

    def fallback(*args, **kwargs) -> Dict[str, Any]:
        logger.warning(f"Using fallback for {module_name}")

        # Return module-specific degraded response
        if module_name == "contradiction_detector":
            return {
                "contradictions": [],
                "coherence_score": 0.5,  # Neutral score
                "status": "degraded"
            }
        elif module_name == "causal_processor":
            return {
                "causal_dimensions": {},
                "confidence": 0.0,
                "status": "degraded"
            }
        elif module_name == "financial_viability":
            return {
                "budget_analysis": {},
                "viability_score": 0.5,
                "status": "degraded"
            }
        else:
            return default_fallback(*args, **kwargs)

    return fallback
