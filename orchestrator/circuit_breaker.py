# circuit_breaker.py - SOTA Implementation for FARFAN 3.0
"""
Advanced Circuit Breaker with AI-driven failure prediction and adaptive thresholds
Implements cutting-edge fault tolerance with predictive analytics and self-healing
"""
import logging
import time
import asyncio
import numpy as np
import pandas as pd
from enum import Enum, IntEnum
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
import json
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import hashlib

from .config import CONFIG

logger = logging.getLogger(__name__)


class CircuitState(IntEnum):
    """Enhanced circuit states with numeric values for ML processing"""
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2
    ISOLATED = 3  # New state for critical failures
    RECOVERING = 4  # New state for active recovery


class FailureSeverity(Enum):
    """Classification of failure types"""
    TRANSIENT = "transient"  # Temporary network issues, timeouts
    DEGRADED = "degraded"  # Slow responses, partial failures
    CRITICAL = "critical"  # Complete failures, exceptions
    CATASTROPHIC = "catastrophic"  # System-wide failures


@dataclass
class FailureEvent:
    """Detailed failure event tracking"""
    timestamp: float
    severity: FailureSeverity
    error_type: str
    error_message: str
    execution_time: float
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    recovery_attempt: int = 0


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    success_rates: deque = field(default_factory=lambda: deque(maxlen=100))
    error_rates: deque = field(default_factory=lambda: deque(maxlen=100))
    throughput: deque = field(default_factory=lambda: deque(maxlen=100))
    resource_usage: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_measurement(self, response_time: float, success: bool,
                        timestamp: float, resource_usage: float = 0.0):
        """Add new performance measurement"""
        self.response_times.append(response_time)
        self.success_rates.append(1.0 if success else 0.0)
        self.error_rates.append(0.0 if success else 1.0)

        # Calculate throughput (requests per second)
        if len(self.response_times) > 1:
            time_window = timestamp - (self.response_times[0] if self.response_times else timestamp)
            if time_window > 0:
                self.throughput.append(len(self.response_times) / time_window)

        self.resource_usage.append(resource_usage)


@dataclass
class AdaptiveThresholds:
    """Dynamic thresholds that adapt based on historical performance"""
    failure_threshold: float = 3.0
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 5
    success_threshold: int = 3

    # Adaptive parameters
    baseline_error_rate: float = 0.05
    baseline_response_time: float = 1.0
    adaptation_factor: float = 0.1
    min_threshold: float = 1.0
    max_threshold: float = 10.0

    def adapt(self, metrics: PerformanceMetrics):
        """Adapt thresholds based on recent performance"""
        if len(metrics.error_rates) >= 10:
            recent_error_rate = np.mean(list(metrics.error_rates)[-10:])
            recent_response_time = np.mean(list(metrics.response_times)[-10:])

            # Adjust failure threshold based on error rate deviation
            error_deviation = recent_error_rate - self.baseline_error_rate
            if error_deviation > 0.02:  # Error rate increased significantly
                self.failure_threshold = max(
                    self.min_threshold,
                    self.failure_threshold * (1 - self.adaptation_factor)
                )
            elif error_deviation < -0.01:  # Error rate improved
                self.failure_threshold = min(
                    self.max_threshold,
                    self.failure_threshold * (1 + self.adaptation_factor * 0.5)
                )

            # Adjust timeout based on response time
            response_deviation = recent_response_time - self.baseline_response_time
            if response_deviation > 0.5:  # Responses slower than baseline
                self.timeout_seconds = min(300, self.timeout_seconds * 1.2)
            elif response_deviation < -0.2:  # Responses faster than baseline
                self.timeout_seconds = max(30, self.timeout_seconds * 0.9)


class PredictiveFailureDetector:
    """ML-based failure prediction system"""

    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_history = deque(maxlen=1000)
        self.prediction_window = 20  # Number of recent measurements to consider

    def extract_features(self, metrics: PerformanceMetrics) -> np.ndarray:
        """Extract features for ML prediction"""
        if len(metrics.response_times) < self.prediction_window:
            return np.array([])

        recent_rt = list(metrics.response_times)[-self.prediction_window:]
        recent_er = list(metrics.error_rates)[-self.prediction_window:]
        recent_thr = list(metrics.throughput)[-self.prediction_window:]

        features = [
            np.mean(recent_rt),
            np.std(recent_rt),
            np.max(recent_rt),
            np.min(recent_rt),
            np.mean(recent_er),
            np.std(recent_er),
            np.max(recent_er),
            np.mean(recent_thr) if recent_thr else 0,
            np.std(recent_thr) if recent_thr else 0,
            # Trend features
            np.polyfit(range(len(recent_rt)), recent_rt, 1)[0] if len(recent_rt) > 1 else 0,
            np.polyfit(range(len(recent_er)), recent_er, 1)[0] if len(recent_er) > 1 else 0,
        ]

        return np.array(features)

    def train(self, historical_data: List[np.ndarray]):
        """Train the failure prediction model"""
        if len(historical_data) < 50:
            logger.warning("Insufficient data for training failure predictor")
            return

        X = np.array(historical_data)
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        self.is_trained = True
        logger.info("Failure prediction model trained successfully")

    def predict_failure(self, metrics: PerformanceMetrics) -> Tuple[bool, float]:
        """Predict if failure is imminent"""
        if not self.is_trained:
            return False, 0.0

        features = self.extract_features(metrics)
        if len(features) == 0:
            return False, 0.0

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]

        # Lower anomaly score indicates more likely failure
        failure_probability = 1.0 / (1.0 + np.exp(anomaly_score))
        is_imminent = failure_probability > 0.7

        return is_imminent, failure_probability


class CircuitBreakerMetrics:
    """Comprehensive metrics for circuit breaker analysis"""

    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.circuit_open_count = 0
        self.circuit_close_count = 0
        self.fallback_usage_count = 0
        self.prediction_accuracy = deque(maxlen=100)
        self.state_transitions = []
        self.failure_events = []
        self.performance_metrics = PerformanceMetrics()

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        """Calculate current success rate"""
        return 1.0 - self.failure_rate

    @property
    def avg_response_time(self) -> float:
        """Calculate average response time"""
        if not self.performance_metrics.response_times:
            return 0.0
        return np.mean(list(self.performance_metrics.response_times))

    def add_failure_event(self, event: FailureEvent):
        """Add a failure event to the history"""
        self.failure_events.append(event)
        # Keep only recent events
        if len(self.failure_events) > 1000:
            self.failure_events = self.failure_events[-1000:]

    def record_state_transition(self, from_state: CircuitState, to_state: CircuitState,
                                reason: str, timestamp: float):
        """Record a state transition"""
        self.state_transitions.append({
            "from": from_state,
            "to": to_state,
            "reason": reason,
            "timestamp": timestamp
        })

        if to_state == CircuitState.OPEN:
            self.circuit_open_count += 1
        elif to_state == CircuitState.CLOSED:
            self.circuit_close_count += 1


class AdvancedCircuitBreaker:
    """
    State-of-the-Art Circuit Breaker with:
    - Adaptive thresholds
    - Predictive failure detection
    - Comprehensive observability
    - Self-healing capabilities
    - Resource-aware routing
    """

    def __init__(
            self,
            module_name: str,
            failure_threshold: Optional[float] = None,
            timeout_seconds: Optional[float] = None,
            half_open_max_calls: Optional[int] = None,
            success_threshold: Optional[int] = None,
            enable_prediction: bool = True,
            enable_adaptation: bool = True
    ):
        self.module_name = module_name
        self.logger = logging.getLogger(f"{__name__}.{module_name}")

        # Circuit state
        self.state = CircuitState.CLOSED
        self.state_lock = threading.RLock()

        # Configuration
        self.thresholds = AdaptiveThresholds(
            failure_threshold=failure_threshold or CONFIG.circuit_breaker_failure_threshold,
            timeout_seconds=timeout_seconds or CONFIG.circuit_breaker_timeout,
            half_open_max_calls=half_open_max_calls or CONFIG.circuit_breaker_half_open_max_calls,
            success_threshold=success_threshold or CONFIG.circuit_breaker_success_threshold
        )

        # Metrics and tracking
        self.metrics = CircuitBreakerMetrics()
        self.consecutive_failures = 0
        self.half_open_calls = 0
        self.half_open_successes = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None

        # Advanced features
        self.enable_prediction = enable_prediction
        self.enable_adaptation = enable_adaptation
        self.predictor = PredictiveFailureDetector() if enable_prediction else None

        # Recovery mechanisms
        self.recovery_strategies = []
        self.active_recovery = None
        self.recovery_start_time: Optional[float] = None

        # Observability
        self.observers = []
        self.event_history = deque(maxlen=10000)

        # Health check
        self.health_check_interval = 30.0  # seconds
        self.last_health_check = 0.0

        self.logger.info(f"Advanced Circuit Breaker initialized for {module_name}")

    def call(
            self,
            func: Callable,
            *args,
            fallback: Optional[Callable] = None,
            context: Optional[Dict[str, Any]] = None,
            timeout: Optional[float] = None,
            **kwargs
    ) -> Any:
        """
        Execute a function with advanced circuit breaker protection.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to func
            fallback: Optional fallback function
            context: Execution context for observability
            timeout: Custom timeout for this call

        Returns:
            Result from func or fallback
        """
        start_time = time.time()
        execution_context = context or {}
        execution_context["call_id"] = hashlib.md5(
            f"{self.module_name}_{start_time}".encode()
        ).hexdigest()[:8]

        try:
            with self.state_lock:
                current_state = self.state

                # Check if circuit should transition based on predictions
                if self.enable_prediction and self.predictor:
                    is_imminent, probability = self.predictor.predict_failure(
                        self.metrics.performance_metrics
                    )
                    if is_imminent and current_state == CircuitState.CLOSED:
                        self.logger.warning(
                            f"Predictive failure detected for {self.module_name} "
                            f"(probability: {probability:.2f})"
                        )
                        self._transition_to_open("predictive_failure")
                        current_state = CircuitState.OPEN

                # Handle different states
                if current_state == CircuitState.OPEN:
                    if self._should_attempt_reset():
                        self._transition_to_half_open("timeout_expired")
                    else:
                        self.metrics.fallback_usage_count += 1
                        if fallback:
                            self.logger.info(f"Using fallback for {self.module_name}")
                            return fallback(*args, **kwargs)
                        else:
                            raise CircuitBreakerError(
                                f"Circuit breaker is OPEN for {self.module_name}"
                            )

                elif current_state == CircuitState.HALF_OPEN:
                    if self.half_open_calls >= self.thresholds.half_open_max_calls:
                        self.metrics.fallback_usage_count += 1
                        if fallback:
                            return fallback(*args, **kwargs)
                        else:
                            raise CircuitBreakerError(
                                f"HALF_OPEN call limit exceeded for {self.module_name}"
                            )
                    self.half_open_calls += 1

                elif current_state == CircuitState.ISOLATED:
                    self.metrics.fallback_usage_count += 1
                    if fallback:
                        return fallback(*args, **kwargs)
                    else:
                        raise CircuitBreakerError(
                            f"Module {self.module_name} is isolated"
                        )

            # Execute the function
            self.metrics.total_requests += 1

            # Monitor resource usage
            resource_usage_start = self._get_resource_usage()

            # Execute with timeout
            actual_timeout = timeout or self.thresholds.timeout_seconds
            result = self._execute_with_timeout(func, actual_timeout, *args, **kwargs)

            # Calculate execution metrics
            execution_time = time.time() - start_time
            resource_usage_end = self._get_resource_usage()
            resource_usage_delta = resource_usage_end - resource_usage_start

            # Record success
            self._record_success(execution_time, resource_usage_delta, execution_context)

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            # Classify failure severity
            severity = self._classify_failure(e, execution_time)

            # Record failure
            failure_event = FailureEvent(
                timestamp=start_time,
                severity=severity,
                error_type=type(e).__name__,
                error_message=str(e),
                execution_time=execution_time,
                context=execution_context,
                stack_trace=self._get_stack_trace()
            )

            self._record_failure(failure_event)

            # Try fallback
            if fallback:
                self.logger.info(f"Executing fallback for {self.module_name} after failure")
                try:
                    fallback_result = fallback(*args, **kwargs)
                    self.metrics.fallback_usage_count += 1
                    return fallback_result
                except Exception as fallback_error:
                    self.logger.error(f"Fallback also failed for {self.module_name}: {fallback_error}")
                    raise
            else:
                raise

    def _execute_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """Execute function with timeout handling"""
        if timeout <= 0:
            return func(*args, **kwargs)

        # Use asyncio for timeout if function is async
        if asyncio.iscoroutinefunction(func):
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(
                asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            )

        # For sync functions, use ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                future.cancel()
                raise CircuitBreakerError(f"Timeout after {timeout}s for {self.module_name}")

    def _record_success(self, execution_time: float, resource_usage: float,
                        context: Dict[str, Any]):
        """Record a successful execution"""
        with self.state_lock:
            self.metrics.successful_requests += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = time.time()

            # Update performance metrics
            self.metrics.performance_metrics.add_measurement(
                execution_time, True, time.time(), resource_usage
            )

            # Handle HALF_OPEN state
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_successes += 1
                if self.half_open_successes >= self.thresholds.success_threshold:
                    self._transition_to_closed("recovery_confirmed")

            # Adapt thresholds if enabled
            if self.enable_adaptation:
                self.thresholds.adapt(self.metrics.performance_metrics)

            # Notify observers
            self._notify_observers("success", {
                "execution_time": execution_time,
                "resource_usage": resource_usage,
                "context": context
            })

    def _record_failure(self, event: FailureEvent):
        """Record a failure event"""
        with self.state_lock:
            self.metrics.failed_requests += 1
            self.metrics.consecutive_failures += 1
            self.metrics.last_failure_time = event.timestamp
            self.metrics.add_failure_event(event)

            # Update performance metrics
            self.metrics.performance_metrics.add_measurement(
                event.execution_time, False, event.timestamp
            )

            # Check if should open circuit
            if self.metrics.consecutive_failures >= self.thresholds.failure_threshold:
                if event.severity in [FailureSeverity.CRITICAL, FailureSeverity.CATASTROPHIC]:
                    self._transition_to_isolated(f"critical_failure_{event.severity.value}")
                else:
                    self._transition_to_open("failure_threshold_exceeded")

            # Notify observers
            self._notify_observers("failure", {
                "event": event,
                "consecutive_failures": self.metrics.consecutive_failures
            })

    def _transition_to_open(self, reason: str):
        """Transition circuit to OPEN state"""
        old_state = self.state
        self.state = CircuitState.OPEN

        self.metrics.record_state_transition(
            old_state, CircuitState.OPEN, reason, time.time()
        )

        self.logger.warning(
            f"Circuit OPENED for {self.module_name} "
            f"({self.metrics.consecutive_failures} consecutive failures) - {reason}"
        )

        # Start recovery timer
        self._schedule_recovery_attempt()

    def _transition_to_half_open(self, reason: str):
        """Transition circuit to HALF_OPEN state"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN

        # Reset half-open counters
        self.half_open_calls = 0
        self.half_open_successes = 0

        self.metrics.record_state_transition(
            old_state, CircuitState.HALF_OPEN, reason, time.time()
        )

        self.logger.info(f"Circuit HALF-OPEN for {self.module_name}, testing recovery - {reason}")

    def _transition_to_closed(self, reason: str):
        """Transition circuit to CLOSED state"""
        old_state = self.state
        self.state = CircuitState.CLOSED

        self.metrics.record_state_transition(
            old_state, CircuitState.CLOSED, reason, time.time()
        )

        self.logger.info(f"Circuit CLOSED for {self.module_name}, normal operation resumed - {reason}")

        # Train predictor with new data
        if self.enable_prediction and self.predictor:
            self._train_predictor()

    def _transition_to_isolated(self, reason: str):
        """Transition circuit to ISOLATED state (critical failures)"""
        old_state = self.state
        self.state = CircuitState.ISOLATED

        self.metrics.record_state_transition(
            old_state, CircuitState.ISOLATED, reason, time.time()
        )

        self.logger.error(
            f"Circuit ISOLATED for {self.module_name} due to critical failure - {reason}"
        )

        # Schedule manual intervention notification
        self._notify_manual_intervention(reason)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.metrics.last_failure_time is None:
            return True

        time_since_failure = time.time() - self.metrics.last_failure_time
        return time_since_failure >= self.thresholds.timeout_seconds

    def _classify_failure(self, exception: Exception, execution_time: float) -> FailureSeverity:
        """Classify the severity of a failure"""
        if isinstance(exception, TimeoutError):
            return FailureSeverity.TRANSIENT
        elif isinstance(exception, ConnectionError):
            return FailureSeverity.DEGRADED
        elif execution_time > self.thresholds.timeout_seconds * 2:
            return FailureSeverity.CRITICAL
        elif "critical" in str(exception).lower() or "fatal" in str(exception).lower():
            return FailureSeverity.CRITICAL
        else:
            return FailureSeverity.DEGRADED

    def _get_resource_usage(self) -> float:
        """Get current resource usage (CPU/Memory)"""
        try:
            import psutil
            process = psutil.Process()
            return process.cpu_percent() + process.memory_percent()
        except ImportError:
            return 0.0

    def _get_stack_trace(self) -> Optional[str]:
        """Get current stack trace"""
        import traceback
        return traceback.format_exc()

    def _schedule_recovery_attempt(self):
        """Schedule an automatic recovery attempt"""

        def recovery_worker():
            time.sleep(self.thresholds.timeout_seconds)
            with self.state_lock:
                if self.state == CircuitState.OPEN:
                    self._transition_to_half_open("scheduled_recovery")

        Thread(target=recovery_worker, daemon=True).start()

    def _train_predictor(self):
        """Train the failure prediction model"""
        if not self.predictor or len(self.metrics.failure_events) < 50:
            return

        # Extract features from historical data
        feature_data = []
        for event in self.metrics.failure_events[-500:]:  # Use last 500 events
            # Extract features from the time window around the event
            features = self.predictor.extract_features(self.metrics.performance_metrics)
            if len(features) > 0:
                feature_data.append(features)

        if len(feature_data) >= 50:
            self.predictor.train(feature_data)

    def _notify_observers(self, event_type: str, data: Dict[str, Any]):
        """Notify all registered observers"""
        for observer in self.observers:
            try:
                observer(self.module_name, event_type, data)
            except Exception as e:
                self.logger.error(f"Observer notification failed: {e}")

    def _notify_manual_intervention(self, reason: str):
        """Notify about need for manual intervention"""
        self.logger.critical(
            f"MANUAL INTERVENTION REQUIRED for {self.module_name}: {reason}"
        )
        # In production, this would send alerts to monitoring systems

    # Public API methods

    def force_open(self, reason: str = "manual"):
        """Manually force the circuit open"""
        with self.state_lock:
            self._transition_to_open(reason)

    def force_close(self, reason: str = "manual"):
        """Manually force the circuit closed"""
        with self.state_lock:
            self._transition_to_closed(reason)

    def reset(self):
        """Reset the circuit breaker to initial state"""
        with self.state_lock:
            self.state = CircuitState.CLOSED
            self.consecutive_failures = 0
            self.half_open_calls = 0
            self.half_open_successes = 0
            self.metrics = CircuitBreakerMetrics()
            self.logger.info(f"Circuit breaker reset for {self.module_name}")

    def add_observer(self, observer: Callable):
        """Add an observer for circuit events"""
        self.observers.append(observer)

    def remove_observer(self, observer: Callable):
        """Remove an observer"""
        if observer in self.observers:
            self.observers.remove(observer)

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        with self.state_lock:
            return {
                "module": self.module_name,
                "state": self.state.name,
                "consecutive_failures": self.consecutive_failures,
                "total_requests": self.metrics.total_requests,
                "success_rate": self.metrics.success_rate,
                "failure_rate": self.metrics.failure_rate,
                "avg_response_time": self.metrics.avg_response_time,
                "last_failure": self.metrics.last_failure_time,
                "last_success": self.metrics.last_success_time,
                "circuit_opens": self.metrics.circuit_open_count,
                "circuit_closes": self.metrics.circuit_close_count,
                "fallback_usage": self.metrics.fallback_usage_count,
                "thresholds": {
                    "failure_threshold": self.thresholds.failure_threshold,
                    "timeout": self.thresholds.timeout_seconds,
                    "half_open_max_calls": self.thresholds.half_open_max_calls
                },
                "predictive_enabled": self.enable_prediction,
                "adaptive_enabled": self.enable_adaptation
            }

    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics for analysis"""
        with self.state_lock:
            recent_failures = [
                {
                    "timestamp": f.timestamp,
                    "severity": f.severity.value,
                    "error_type": f.error_type,
                    "execution_time": f.execution_time
                }
                for f in list(self.metrics.failure_events)[-10:]
            ]

            return {
                "health": self.get_health_status(),
                "recent_failures": recent_failures,
                "state_transitions": self.metrics.state_transitions[-20:],
                "performance": {
                    "response_times": list(self.metrics.performance_metrics.response_times)[-100:],
                    "error_rates": list(self.metrics.performance_metrics.error_rates)[-100:],
                    "throughput": list(self.metrics.performance_metrics.throughput)[-100:]
                },
                "prediction": {
                    "enabled": self.enable_prediction,
                    "trained": self.predictor.is_trained if self.predictor else False
                }
            }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""

    def __init__(self):
        self.circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}
        self.global_observers = []
        self.metrics_aggregator = CircuitBreakerMetricsAggregator()

    def get_circuit_breaker(self, module_name: str, **kwargs) -> AdvancedCircuitBreaker:
        """Get or create a circuit breaker for a module"""
        if module_name not in self.circuit_breakers:
            self.circuit_breakers[module_name] = AdvancedCircuitBreaker(
                module_name, **kwargs
            )
            # Add global observer
            self.circuit_breakers[module_name].add_observer(
                self._global_observer
            )
        return self.circuit_breakers[module_name]

    def _global_observer(self, module_name: str, event_type: str, data: Dict[str, Any]):
        """Global observer for all circuit events"""
        self.metrics_aggregator.record_event(module_name, event_type, data)

    def get_all_health_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all circuit breakers"""
        return {
            name: cb.get_health_status()
            for name, cb in self.circuit_breakers.items()
        }

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get system-wide health summary"""
        statuses = self.get_all_health_statuses()

        total_modules = len(statuses)
        healthy = sum(1 for s in statuses.values() if s["state"] == "CLOSED")
        degraded = sum(1 for s in statuses.values() if s["state"] == "HALF_OPEN")
        unhealthy = sum(1 for s in statuses.values() if s["state"] in ["OPEN", "ISOLATED"])

        avg_success_rate = np.mean([
            s["success_rate"] for s in statuses.values() if s["total_requests"] > 0
        ]) if statuses else 0.0

        return {
            "total_modules": total_modules,
            "healthy_modules": healthy,
            "degraded_modules": degraded,
            "unhealthy_modules": unhealthy,
            "health_percentage": (healthy / total_modules * 100) if total_modules > 0 else 0,
            "avg_success_rate": avg_success_rate,
            "timestamp": datetime.now().isoformat()
        }


class CircuitBreakerMetricsAggregator:
    """Aggregates metrics from all circuit breakers"""

    def __init__(self):
        self.events = deque(maxlen=10000)
        self.module_metrics = defaultdict(lambda: defaultdict(list))

    def record_event(self, module_name: str, event_type: str, data: Dict[str, Any]):
        """Record an event from a circuit breaker"""
        event = {
            "module": module_name,
            "event_type": event_type,
            "timestamp": time.time(),
            "data": data
        }
        self.events.append(event)
        self.module_metrics[module_name][event_type].append(event)

    def get_failure_patterns(self) -> Dict[str, Any]:
        """Analyze failure patterns across modules"""
        failure_events = [
            e for e in self.events
            if e["event_type"] == "failure"
        ]

        if not failure_events:
            return {}

        # Group by hour
        hourly_failures = defaultdict(int)
        for event in failure_events:
            hour = datetime.fromtimestamp(event["timestamp"]).hour
            hourly_failures[hour] += 1

        # Group by error type
        error_types = defaultdict(int)
        for event in failure_events:
            error_type = event["data"]["event"].error_type
            error_types[error_type] += 1

        return {
            "hourly_pattern": dict(hourly_failures),
            "error_types": dict(error_types),
            "total_failures": len(failure_events)
        }


class CircuitBreakerError(Exception):
    """Raised when circuit breaker blocks execution"""
    pass


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


# Convenience function for backward compatibility
def create_module_specific_fallback(module_name: str) -> Callable:
    """Create a module-specific fallback function"""

    def fallback(*args, **kwargs) -> Dict[str, Any]:
        logger.warning(f"Using fallback for {module_name}")

        # Return module-specific degraded response
        fallback_responses = {
            "contradiction_detector": {
                "contradictions": [],
                "coherence_metrics": {"coherence_score": 0.5},
                "status": "degraded"
            },
            "causal_processor": {
                "causal_dimensions": {},
                "information_gain": 0.0,
                "status": "degraded"
            },
            "dereck_beach": {
                "mechanism_parts": [],
                "rigor_status": 0.0,
                "status": "degraded"
            },
            "policy_processor": {
                "dimensions": {},
                "overall_score": 0.5,
                "status": "degraded"
            },
            "analyzer_one": {
                "analysis_results": {},
                "quality_score": 0.5,
                "status": "degraded"
            },
            "embedding_policy": {
                "chunks_processed": 0,
                "embeddings_generated": False,
                "status": "degraded"
            },
            "policy_segmenter": {
                "segments": [],
                "status": "degraded"
            },
            "financial_analyzer": {
                "budget_analysis": {},
                "viability_score": 0.5,
                "status": "degraded"
            }
        }

        return fallback_responses.get(
            module_name,
            {
                "status": "degraded",
                "message": f"Module {module_name} unavailable, using fallback",
                "data": {}
            }
        )

    return fallback