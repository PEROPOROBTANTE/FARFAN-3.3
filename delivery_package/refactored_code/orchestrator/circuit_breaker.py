# circuit_breaker.py - COMPLETE UPDATE FOR 9 ADAPTERS
# coding=utf-8
"""
Advanced Circuit Breaker with Fault Tolerance
==============================================

Updated for complete integration with:
- module_adapters_COMPLETE_MERGED.py (9 adapters, 413 methods)
- ModuleResult standardized format
- ExecutionChoreographer orchestration

Implements:
- Per-adapter circuit breaking
- Failure tracking and analysis
- Graceful degradation
- Self-healing strategies
- Fallback mechanisms

Author: Integration Team
Version: 3.0.0 - Complete Adapter Alignment
Python: 3.10+
"""

import logging
import time
from enum import Enum, IntEnum
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


# ============================================================================
# CIRCUIT STATES AND FAILURE TRACKING
# ============================================================================

class CircuitState(IntEnum):
    """Circuit breaker states"""
    CLOSED = 0      # Normal operation
    OPEN = 1        # Blocking requests
    HALF_OPEN = 2   # Testing recovery
    ISOLATED = 3    # Critical failures
    RECOVERING = 4  # Active recovery


class FailureSeverity(Enum):
    """Failure severity classification"""
    TRANSIENT = "transient"      # Temporary issues
    DEGRADED = "degraded"         # Partial failures
    CRITICAL = "critical"         # Complete failures
    CATASTROPHIC = "catastrophic" # System-wide failures


@dataclass
class FailureEvent:
    """Individual failure event"""
    timestamp: float
    severity: FailureSeverity
    error_type: str
    error_message: str
    execution_time: float
    adapter_name: str
    method_name: str = ""
    recovery_attempt: int = 0


@dataclass
class PerformanceMetrics:
    """Performance tracking for an adapter"""
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    success_count: int = 0
    failure_count: int = 0
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    
    def add_success(self, response_time: float):
        """Record successful execution"""
        self.response_times.append(response_time)
        self.success_count += 1
        self.last_success = time.time()
    
    def add_failure(self, response_time: float):
        """Record failed execution"""
        self.response_times.append(response_time)
        self.failure_count += 1
        self.last_failure = time.time()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time"""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    """
    Circuit breaker for 9 adapters with fault tolerance
    
    Features:
    - Per-adapter circuit states
    - Failure threshold monitoring
    - Automatic recovery testing
    - Graceful degradation
    - Fallback strategies
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
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before testing recovery
            half_open_max_calls: Max calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        # Per-adapter state
        self.adapter_states: Dict[str, CircuitState] = {}
        self.adapter_failures: Dict[str, deque] = {}
        self.adapter_metrics: Dict[str, PerformanceMetrics] = {}
        self.last_state_change: Dict[str, float] = {}
        self.half_open_calls: Dict[str, int] = {}
        
        # Initialize for 9 adapters
        self.adapters = [
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
        
        for adapter in self.adapters:
            self._initialize_adapter(adapter)
        
        logger.info(
            f"CircuitBreaker initialized for {len(self.adapters)} adapters "
            f"(threshold={failure_threshold}, timeout={recovery_timeout}s)"
        )

    def _initialize_adapter(self, adapter_name: str):
        """Initialize state for an adapter"""
        self.adapter_states[adapter_name] = CircuitState.CLOSED
        self.adapter_failures[adapter_name] = deque(maxlen=self.failure_threshold * 2)
        self.adapter_metrics[adapter_name] = PerformanceMetrics()
        self.last_state_change[adapter_name] = time.time()
        self.half_open_calls[adapter_name] = 0

    def can_execute(self, adapter_name: str) -> bool:
        """
        Check if adapter can execute
        
        Args:
            adapter_name: Adapter to check
            
        Returns:
            True if adapter can execute
        """
        if adapter_name not in self.adapter_states:
            self._initialize_adapter(adapter_name)
        
        state = self.adapter_states[adapter_name]
        
        # CLOSED: Normal operation
        if state == CircuitState.CLOSED:
            return True
        
        # OPEN: Check if recovery timeout passed
        if state == CircuitState.OPEN:
            time_since_open = time.time() - self.last_state_change[adapter_name]
            if time_since_open >= self.recovery_timeout:
                # Try recovery
                self._transition_to_half_open(adapter_name)
                return True
            return False
        
        # HALF_OPEN: Allow limited calls
        if state == CircuitState.HALF_OPEN:
            if self.half_open_calls[adapter_name] < self.half_open_max_calls:
                self.half_open_calls[adapter_name] += 1
                return True
            return False
        
        # ISOLATED/RECOVERING: Block all
        return False

    def record_success(self, adapter_name: str, execution_time: float = 0.0):
        """
        Record successful execution
        
        Args:
            adapter_name: Adapter that succeeded
            execution_time: Execution time in seconds
        """
        if adapter_name not in self.adapter_metrics:
            self._initialize_adapter(adapter_name)
        
        # Update metrics
        self.adapter_metrics[adapter_name].add_success(execution_time)
        
        state = self.adapter_states[adapter_name]
        
        # HALF_OPEN: Check if we can close circuit
        if state == CircuitState.HALF_OPEN:
            if self.half_open_calls[adapter_name] >= self.half_open_max_calls:
                # All test calls succeeded, close circuit
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
            adapter_name: Adapter that failed
            error: Error message
            execution_time: Execution time in seconds
            severity: Failure severity
        """
        if adapter_name not in self.adapter_metrics:
            self._initialize_adapter(adapter_name)
        
        # Create failure event
        failure = FailureEvent(
            timestamp=time.time(),
            severity=severity,
            error_type=type(error).__name__ if isinstance(error, Exception) else "Error",
            error_message=str(error),
            execution_time=execution_time,
            adapter_name=adapter_name
        )
        
        # Update metrics
        self.adapter_metrics[adapter_name].add_failure(execution_time)
        self.adapter_failures[adapter_name].append(failure)
        
        state = self.adapter_states[adapter_name]
        
        # CLOSED: Check if we should open
        if state == CircuitState.CLOSED:
            recent_failures = self._count_recent_failures(adapter_name)
            if recent_failures >= self.failure_threshold:
                self._transition_to_open(adapter_name)
        
        # HALF_OPEN: Immediate open on failure
        elif state == CircuitState.HALF_OPEN:
            self._transition_to_open(adapter_name)
        
        logger.warning(
            f"{adapter_name}: Failure recorded (state={state.name}, "
            f"recent_failures={self._count_recent_failures(adapter_name)})"
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

    def get_adapter_status(self, adapter_name: str) -> Dict[str, Any]:
        """Get detailed status for an adapter"""
        if adapter_name not in self.adapter_states:
            return {"error": "Adapter not found"}
        
        metrics = self.adapter_metrics[adapter_name]
        
        return {
            "adapter": adapter_name,
            "state": self.adapter_states[adapter_name].name,
            "success_rate": metrics.success_rate,
            "total_calls": metrics.success_count + metrics.failure_count,
            "successes": metrics.success_count,
            "failures": metrics.failure_count,
            "avg_response_time": metrics.avg_response_time,
            "recent_failures": self._count_recent_failures(adapter_name),
            "last_success": metrics.last_success,
            "last_failure": metrics.last_failure,
            "time_since_state_change": time.time() - self.last_state_change[adapter_name]
        }

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all adapters"""
        return {
            adapter: self.get_adapter_status(adapter)
            for adapter in self.adapters
        }

    def get_fallback_strategy(self, adapter_name: str) -> Dict[str, Any]:
        """
        Get fallback strategy for failed adapter
        
        Returns:
            Fallback configuration
        """
        fallback_strategies = {
            "teoria_cambio": {
                "use_cached": True,
                "alternative_adapters": ["analyzer_one"],
                "degraded_mode": "basic_causal_analysis"
            },
            "analyzer_one": {
                "use_cached": True,
                "alternative_adapters": ["embedding_policy"],
                "degraded_mode": "simple_analysis"
            },
            "dereck_beach": {
                "use_cached": True,
                "alternative_adapters": ["teoria_cambio"],
                "degraded_mode": "basic_causal_inference"
            },
            "embedding_policy": {
                "use_cached": True,
                "alternative_adapters": ["semantic_chunking_policy"],
                "degraded_mode": "keyword_matching"
            },
            "semantic_chunking_policy": {
                "use_cached": True,
                "alternative_adapters": ["policy_processor"],
                "degraded_mode": "simple_segmentation"
            },
            "contradiction_detection": {
                "use_cached": True,
                "alternative_adapters": ["analyzer_one"],
                "degraded_mode": "basic_consistency_check"
            },
            "financial_viability": {
                "use_cached": True,
                "alternative_adapters": ["analyzer_one"],
                "degraded_mode": "basic_financial_check"
            },
            "policy_processor": {
                "use_cached": True,
                "alternative_adapters": None,
                "degraded_mode": "basic_text_processing"
            },
            "policy_segmenter": {
                "use_cached": True,
                "alternative_adapters": None,
                "degraded_mode": "simple_paragraph_split"
            }
        }
        
        return fallback_strategies.get(adapter_name, {
            "use_cached": False,
            "alternative_adapters": None,
            "degraded_mode": "skip"
        })

    def reset_adapter(self, adapter_name: str):
        """Manually reset an adapter's circuit"""
        if adapter_name in self.adapter_states:
            self._initialize_adapter(adapter_name)
            logger.info(f"{adapter_name}: Circuit manually reset")

    def reset_all(self):
        """Reset all adapter circuits"""
        for adapter in self.adapters:
            self.reset_adapter(adapter)
        logger.info("All circuits reset")


# ============================================================================
# FALLBACK CREATION HELPERS
# ============================================================================

def create_module_specific_fallback(
        adapter_name: str,
        method_name: str
) -> Callable:
    """
    Create adapter-specific fallback function
    
    Args:
        adapter_name: Adapter name
        method_name: Method name
        
    Returns:
        Fallback function
    """
    def fallback(*args, **kwargs):
        """Generic fallback returning safe default"""
        logger.warning(f"Using fallback for {adapter_name}.{method_name}")
        
        return {
            "status": "degraded",
            "data": {},
            "evidence": [],
            "confidence": 0.0,
            "message": f"Fallback used for {adapter_name}.{method_name}",
            "adapter_name": adapter_name,
            "method_name": method_name
        }
    
    return fallback


def create_cached_fallback(cache: Dict[str, Any]) -> Callable:
    """
    Create fallback that uses cached results
    
    Args:
        cache: Cache dictionary
        
    Returns:
        Fallback function
    """
    def fallback(adapter_name: str, method_name: str, *args, **kwargs):
        """Cached fallback"""
        cache_key = f"{adapter_name}.{method_name}"
        
        if cache_key in cache:
            logger.info(f"Using cached result for {cache_key}")
            return cache[cache_key]
        
        logger.warning(f"No cached result for {cache_key}")
        return create_module_specific_fallback(adapter_name, method_name)(*args, **kwargs)
    
    return fallback


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    breaker = CircuitBreaker()
    
    print("=" * 80)
    print("CIRCUIT BREAKER - COMPLETE UPDATE")
    print("=" * 80)
    print(f"\nAdapters monitored: {len(breaker.adapters)}")
    print(f"Failure threshold: {breaker.failure_threshold}")
    print(f"Recovery timeout: {breaker.recovery_timeout}s")
    print("\nInitial status:")
    for adapter, status in breaker.get_all_status().items():
        print(f"  {adapter}: {status['state']}")
    print("=" * 80)