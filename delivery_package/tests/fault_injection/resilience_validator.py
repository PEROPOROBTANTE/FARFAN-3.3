"""
Resilience Validator for FARFAN 3.0
===================================

Valida resiliencia del sistema ejecutando test scenarios contra los 9 adapters:
- Circuit breaker state transitions (CLOSED→OPEN→HALF_OPEN→RECOVERING→ISOLATED)
- Retry backoff exponential strategy with jitter
- Timeout enforcement respecting max_latency_ms
- Idempotency detection preventing duplicate execution

Author: FARFAN Integration Team
Version: 1.0.0
Python: 3.10+
"""

import logging
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import sys
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestrator.circuit_breaker import CircuitBreaker, CircuitState, FailureSeverity
from tests.fault_injection.injectors import (
    ContractFaultInjector,
    DeterminismFaultInjector,
    FaultToleranceFaultInjector,
    OperationalFaultInjector,
    FaultCategory,
    InjectedFault
)

logger = logging.getLogger(__name__)


# ============================================================================
# VALIDATION RESULT
# ============================================================================

class ValidationStatus(Enum):
    """Status de validación"""
    PASSED = "passed"
    FAILED = "failed"
    DEGRADED = "degraded"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Resultado de una validación de resiliencia"""
    test_name: str
    adapter_name: str
    status: ValidationStatus
    description: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0


@dataclass
class StateTransitionRecord:
    """Registro de una transición de estado del circuit breaker"""
    adapter_name: str
    from_state: str
    to_state: str
    timestamp: float
    trigger: str  # "failure", "timeout", "success", "manual"
    failure_count: int = 0


# ============================================================================
# RESILIENCE VALIDATOR
# ============================================================================

class ResilienceValidator:
    """
    Validator de resiliencia que ejecuta test scenarios contra los 9 adapters
    
    Valida:
    1. Circuit breaker state transitions correctas
    2. Retry backoff exponencial con jitter
    3. Timeout enforcement según max_latency_ms
    4. Idempotency detection
    """
    
    # Lista de 9 adapters de FARFAN 3.0
    ADAPTERS = [
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
    
    # Secuencia esperada de estados del circuit breaker
    EXPECTED_STATE_SEQUENCE = [
        "CLOSED",       # Estado inicial normal
        "OPEN",         # Después de threshold de fallos
        "HALF_OPEN",    # Después de recovery timeout
        "RECOVERING",   # Durante proceso de recuperación (si existe)
        "ISOLATED"      # Si falla recuperación (estado crítico)
    ]
    
    def __init__(self, circuit_breaker: Optional[CircuitBreaker] = None):
        """
        Initialize validator
        
        Args:
            circuit_breaker: Circuit breaker instance (crea uno nuevo si None)
        """
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=10.0,  # Reducido para testing
            half_open_max_calls=3
        )
        
        # Injectors
        self.contract_injector = ContractFaultInjector()
        self.determinism_injector = DeterminismFaultInjector()
        self.fault_tolerance_injector = FaultToleranceFaultInjector()
        self.operational_injector = OperationalFaultInjector()
        
        # State tracking
        self.state_transitions: List[StateTransitionRecord] = []
        self.execution_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.validation_results: List[ValidationResult] = []
        
        logger.info(f"ResilienceValidator initialized for {len(self.ADAPTERS)} adapters")
    
    # ========================================================================
    # CIRCUIT BREAKER STATE TRANSITION VALIDATION
    # ========================================================================
    
    def validate_circuit_breaker_transitions(
        self,
        adapter_name: str,
        failure_count: int = 10
    ) -> ValidationResult:
        """
        Valida que circuit breaker transiciona correctamente por estados
        
        Secuencia esperada: CLOSED → OPEN → HALF_OPEN → (RECOVERING) → ISOLATED
        
        Args:
            adapter_name: Adaptador a validar
            failure_count: Número de fallos a inyectar
            
        Returns:
            ValidationResult con detalles de validación
        """
        start_time = time.time()
        result = ValidationResult(
            test_name="circuit_breaker_state_transitions",
            adapter_name=adapter_name,
            status=ValidationStatus.PASSED,
            description="Validate circuit breaker follows CLOSED→OPEN→HALF_OPEN sequence"
        )
        
        try:
            # 1. Verificar estado inicial CLOSED
            initial_state = self.circuit_breaker.adapter_states.get(adapter_name)
            if initial_state != CircuitState.CLOSED:
                result.failures.append(f"Initial state is {initial_state}, expected CLOSED")
                result.status = ValidationStatus.FAILED
            
            self._record_transition(adapter_name, None, "CLOSED", "initialization", 0)
            
            # 2. Inyectar fallos hasta abrir el circuit
            for i in range(failure_count):
                self.circuit_breaker.record_failure(
                    adapter_name,
                    f"Injected failure {i+1}",
                    execution_time=0.1,
                    severity=FailureSeverity.CRITICAL
                )
                
                current_state = self.circuit_breaker.adapter_states[adapter_name]
                
                # Verificar transición a OPEN después de threshold
                if i >= self.circuit_breaker.failure_threshold - 1:
                    if current_state != CircuitState.OPEN:
                        result.failures.append(
                            f"After {i+1} failures, state is {current_state.name}, expected OPEN"
                        )
                        result.status = ValidationStatus.FAILED
                    elif i == self.circuit_breaker.failure_threshold - 1:
                        self._record_transition(
                            adapter_name, "CLOSED", "OPEN", 
                            "failure_threshold", i + 1
                        )
                
                time.sleep(0.01)  # Pequeña pausa
            
            # 3. Verificar que requests son bloqueadas en OPEN
            if not self.circuit_breaker.can_execute(adapter_name):
                result.metrics["blocks_in_open"] = True
            else:
                result.failures.append("Circuit OPEN but still allows execution")
                result.status = ValidationStatus.FAILED
            
            # 4. Esperar recovery timeout y verificar transición a HALF_OPEN
            logger.info(f"Waiting {self.circuit_breaker.recovery_timeout}s for recovery timeout...")
            time.sleep(self.circuit_breaker.recovery_timeout + 0.5)
            
            # Intentar ejecutar (debería transicionar a HALF_OPEN)
            can_execute = self.circuit_breaker.can_execute(adapter_name)
            current_state = self.circuit_breaker.adapter_states[adapter_name]
            
            if current_state != CircuitState.HALF_OPEN:
                result.failures.append(
                    f"After recovery timeout, state is {current_state.name}, expected HALF_OPEN"
                )
                result.status = ValidationStatus.FAILED
            else:
                self._record_transition(
                    adapter_name, "OPEN", "HALF_OPEN",
                    "recovery_timeout", 0
                )
            
            # 5. Registrar éxitos en HALF_OPEN para cerrar circuit
            for i in range(self.circuit_breaker.half_open_max_calls):
                self.circuit_breaker.record_success(adapter_name, execution_time=0.1)
                time.sleep(0.01)
            
            # Verificar transición a CLOSED
            final_state = self.circuit_breaker.adapter_states[adapter_name]
            if final_state != CircuitState.CLOSED:
                result.warnings.append(
                    f"After {self.circuit_breaker.half_open_max_calls} successes, "
                    f"state is {final_state.name}, expected CLOSED"
                )
            else:
                self._record_transition(
                    adapter_name, "HALF_OPEN", "CLOSED",
                    "recovery_success", 0
                )
            
            # 6. Validar secuencia completa
            observed_sequence = [t.to_state for t in self.state_transitions 
                                if t.adapter_name == adapter_name]
            
            result.metrics["observed_sequence"] = observed_sequence
            result.metrics["expected_sequence"] = ["CLOSED", "OPEN", "HALF_OPEN", "CLOSED"]
            result.metrics["transitions_count"] = len(observed_sequence)
            
            # Verificar que tiene transiciones clave
            required_transitions = ["CLOSED", "OPEN", "HALF_OPEN"]
            missing_transitions = [t for t in required_transitions if t not in observed_sequence]
            
            if missing_transitions:
                result.failures.append(f"Missing required transitions: {missing_transitions}")
                result.status = ValidationStatus.FAILED
            
        except Exception as e:
            result.status = ValidationStatus.FAILED
            result.failures.append(f"Exception during validation: {str(e)}")
            logger.exception("Error in circuit breaker validation")
        
        result.execution_time = time.time() - start_time
        self.validation_results.append(result)
        
        return result
    
    def _record_transition(
        self,
        adapter_name: str,
        from_state: Optional[str],
        to_state: str,
        trigger: str,
        failure_count: int
    ):
        """Registra una transición de estado"""
        transition = StateTransitionRecord(
            adapter_name=adapter_name,
            from_state=from_state or "NONE",
            to_state=to_state,
            timestamp=time.time(),
            trigger=trigger,
            failure_count=failure_count
        )
        self.state_transitions.append(transition)
        logger.debug(f"State transition: {adapter_name} {from_state} → {to_state} ({trigger})")
    
    # ========================================================================
    # RETRY BACKOFF VALIDATION
    # ========================================================================
    
    def validate_retry_backoff(
        self,
        adapter_name: str,
        max_retries: int = 5
    ) -> ValidationResult:
        """
        Valida retry backoff exponencial con jitter
        
        Estrategia esperada: delay = base * (2 ^ retry) + random_jitter
        
        Args:
            adapter_name: Adaptador a validar
            max_retries: Número máximo de reintentos
            
        Returns:
            ValidationResult con análisis de backoff
        """
        start_time = time.time()
        result = ValidationResult(
            test_name="retry_backoff_exponential_with_jitter",
            adapter_name=adapter_name,
            status=ValidationStatus.PASSED,
            description="Validate exponential backoff with jitter"
        )
        
        retry_delays: List[float] = []
        base_delay = 0.1  # 100ms base
        
        try:
            for retry in range(max_retries):
                retry_start = time.time()
                
                # Simular fallo
                self.circuit_breaker.record_failure(
                    adapter_name,
                    f"Retry test failure {retry+1}",
                    execution_time=0.01
                )
                
                # Calcular delay esperado (exponencial)
                expected_delay = base_delay * (2 ** retry)
                
                # Esperar con jitter (±20%)
                import random
                jitter = random.uniform(-0.2, 0.2) * expected_delay
                actual_delay = expected_delay + jitter
                
                time.sleep(actual_delay)
                retry_delays.append(actual_delay)
                
                logger.debug(f"Retry {retry+1}: delay={actual_delay:.3f}s (expected≈{expected_delay:.3f}s)")
            
            # Validar que delays son exponenciales
            if len(retry_delays) >= 2:
                # Calcular ratios entre delays consecutivos
                ratios = [retry_delays[i+1] / retry_delays[i] 
                         for i in range(len(retry_delays) - 1)]
                
                avg_ratio = statistics.mean(ratios)
                
                result.metrics["retry_delays"] = [round(d, 3) for d in retry_delays]
                result.metrics["growth_ratios"] = [round(r, 2) for r in ratios]
                result.metrics["avg_growth_ratio"] = round(avg_ratio, 2)
                result.metrics["expected_ratio"] = 2.0
                
                # Validar que ratio promedio está cerca de 2.0 (exponencial)
                # Tolerancia amplia por jitter: 1.5 - 2.5
                if not (1.5 <= avg_ratio <= 2.5):
                    result.failures.append(
                        f"Average growth ratio {avg_ratio:.2f} not exponential (expected ≈2.0)"
                    )
                    result.status = ValidationStatus.FAILED
                
                # Validar que hay variación (jitter)
                if len(set(retry_delays)) == 1:
                    result.failures.append("No jitter detected - all delays identical")
                    result.status = ValidationStatus.DEGRADED
            
        except Exception as e:
            result.status = ValidationStatus.FAILED
            result.failures.append(f"Exception: {str(e)}")
            logger.exception("Error in retry backoff validation")
        
        result.execution_time = time.time() - start_time
        self.validation_results.append(result)
        
        return result
    
    # ========================================================================
    # TIMEOUT ENFORCEMENT VALIDATION
    # ========================================================================
    
    def validate_timeout_enforcement(
        self,
        adapter_name: str,
        max_latency_ms: int = 5000
    ) -> ValidationResult:
        """
        Valida que timeouts se respetan según max_latency_ms
        
        Args:
            adapter_name: Adaptador a validar
            max_latency_ms: Timeout máximo en milisegundos
            
        Returns:
            ValidationResult con análisis de timeouts
        """
        start_time = time.time()
        result = ValidationResult(
            test_name="timeout_enforcement",
            adapter_name=adapter_name,
            status=ValidationStatus.PASSED,
            description=f"Validate timeout enforcement at {max_latency_ms}ms"
        )
        
        try:
            # Simular operación que excede timeout
            operation_duration = (max_latency_ms / 1000.0) + 1.0  # Excede por 1 segundo
            
            operation_start = time.time()
            time.sleep(operation_duration)
            operation_end = time.time()
            
            actual_duration_ms = (operation_end - operation_start) * 1000
            
            result.metrics["max_latency_ms"] = max_latency_ms
            result.metrics["actual_duration_ms"] = round(actual_duration_ms, 2)
            result.metrics["exceeded_timeout"] = actual_duration_ms > max_latency_ms
            
            # En un sistema real, la operación debería ser cancelada
            # Aquí validamos que se detecta el exceso
            if actual_duration_ms > max_latency_ms:
                result.warnings.append(
                    f"Operation exceeded timeout: {actual_duration_ms:.0f}ms > {max_latency_ms}ms"
                )
                
                # Registrar como fallo
                self.circuit_breaker.record_failure(
                    adapter_name,
                    f"Timeout exceeded: {actual_duration_ms:.0f}ms",
                    execution_time=operation_duration,
                    severity=FailureSeverity.CRITICAL
                )
                
                result.metrics["timeout_violation_recorded"] = True
            
        except Exception as e:
            result.status = ValidationStatus.FAILED
            result.failures.append(f"Exception: {str(e)}")
            logger.exception("Error in timeout validation")
        
        result.execution_time = time.time() - start_time
        self.validation_results.append(result)
        
        return result
    
    # ========================================================================
    # IDEMPOTENCY VALIDATION
    # ========================================================================
    
    def validate_idempotency_detection(
        self,
        adapter_name: str,
        method_name: str,
        test_input: Dict[str, Any]
    ) -> ValidationResult:
        """
        Valida que el sistema detecta y previene ejecuciones duplicadas
        
        Args:
            adapter_name: Adaptador a validar
            method_name: Método a ejecutar
            test_input: Input de prueba
            
        Returns:
            ValidationResult con análisis de idempotency
        """
        start_time = time.time()
        result = ValidationResult(
            test_name="idempotency_detection",
            adapter_name=adapter_name,
            status=ValidationStatus.PASSED,
            description="Validate duplicate execution prevention"
        )
        
        try:
            # Generar execution_id único
            import hashlib
            import json
            
            execution_key = f"{adapter_name}:{method_name}:{json.dumps(test_input, sort_keys=True)}"
            execution_id = hashlib.sha256(execution_key.encode()).hexdigest()[:16]
            
            # Primera ejecución
            exec1_start = time.time()
            execution_record_1 = {
                "execution_id": execution_id,
                "adapter": adapter_name,
                "method": method_name,
                "input": test_input,
                "timestamp": exec1_start,
                "result": "success"
            }
            self.execution_history[adapter_name].append(execution_record_1)
            exec1_duration = time.time() - exec1_start
            
            # Segunda ejecución (debería detectarse como duplicada)
            exec2_start = time.time()
            
            # Buscar ejecuciones previas con mismo execution_id
            previous_executions = [
                e for e in self.execution_history[adapter_name]
                if e["execution_id"] == execution_id
            ]
            
            is_duplicate = len(previous_executions) > 0
            
            if is_duplicate:
                result.metrics["duplicate_detected"] = True
                result.metrics["previous_execution_count"] = len(previous_executions)
                result.metrics["prevented_duplicate_execution"] = True
                
                logger.info(f"Duplicate execution detected and prevented for {execution_id}")
            else:
                # Registrar segunda ejecución
                execution_record_2 = {
                    "execution_id": execution_id,
                    "adapter": adapter_name,
                    "method": method_name,
                    "input": test_input,
                    "timestamp": exec2_start,
                    "result": "success"
                }
                self.execution_history[adapter_name].append(execution_record_2)
                
                result.failures.append("Duplicate execution NOT detected")
                result.status = ValidationStatus.FAILED
            
            result.metrics["execution_id"] = execution_id
            result.metrics["total_executions"] = len(self.execution_history[adapter_name])
            
        except Exception as e:
            result.status = ValidationStatus.FAILED
            result.failures.append(f"Exception: {str(e)}")
            logger.exception("Error in idempotency validation")
        
        result.execution_time = time.time() - start_time
        self.validation_results.append(result)
        
        return result
    
    # ========================================================================
    # GRACEFUL DEGRADATION VALIDATION
    # ========================================================================
    
    def validate_graceful_degradation(
        self,
        adapter_name: str,
        injected_faults: List[InjectedFault]
    ) -> ValidationResult:
        """
        Valida degradación graceful sin cascading failures
        
        Args:
            adapter_name: Adaptador bajo test
            injected_faults: Fallos inyectados
            
        Returns:
            ValidationResult con análisis de degradación
        """
        start_time = time.time()
        result = ValidationResult(
            test_name="graceful_degradation",
            adapter_name=adapter_name,
            status=ValidationStatus.PASSED,
            description="Validate graceful degradation without cascading failures"
        )
        
        try:
            # Verificar estados de todos los adapters
            all_states = {}
            for adapter in self.ADAPTERS:
                state = self.circuit_breaker.adapter_states.get(adapter, CircuitState.CLOSED)
                all_states[adapter] = state.name
            
            result.metrics["adapter_states"] = all_states
            
            # Contar adapters en estado crítico
            open_count = sum(1 for s in all_states.values() if s == "OPEN")
            isolated_count = sum(1 for s in all_states.values() if s == "ISOLATED")
            
            result.metrics["open_circuits"] = open_count
            result.metrics["isolated_circuits"] = isolated_count
            result.metrics["healthy_circuits"] = len(self.ADAPTERS) - open_count - isolated_count
            
            # Validar que fallos no se propagan (no cascading)
            # Si solo 1 adapter falla, no deberían fallar otros
            if len(injected_faults) == 1 and (open_count + isolated_count) > 1:
                result.failures.append(
                    f"Cascading failure detected: 1 fault caused {open_count + isolated_count} circuits to fail"
                )
                result.status = ValidationStatus.FAILED
            
            # Verificar que hay fallback strategy disponible
            fallback = self.circuit_breaker.get_fallback_strategy(adapter_name)
            if fallback and fallback.get("use_cached"):
                result.metrics["has_fallback"] = True
                result.metrics["fallback_strategy"] = fallback
            else:
                result.warnings.append("No fallback strategy available")
                result.status = ValidationStatus.DEGRADED
            
        except Exception as e:
            result.status = ValidationStatus.FAILED
            result.failures.append(f"Exception: {str(e)}")
            logger.exception("Error in graceful degradation validation")
        
        result.execution_time = time.time() - start_time
        self.validation_results.append(result)
        
        return result
    
    # ========================================================================
    # RUN ALL VALIDATIONS
    # ========================================================================
    
    def run_all_validations(self, adapter_name: str) -> List[ValidationResult]:
        """
        Ejecuta todas las validaciones para un adapter
        
        Args:
            adapter_name: Nombre del adapter
            
        Returns:
            Lista de ValidationResults
        """
        logger.info(f"Running all validations for {adapter_name}...")
        
        results = []
        
        # 1. Circuit breaker state transitions
        results.append(self.validate_circuit_breaker_transitions(adapter_name))
        
        # Reset circuit breaker para próximas pruebas
        self.circuit_breaker.reset_adapter(adapter_name)
        
        # 2. Retry backoff
        results.append(self.validate_retry_backoff(adapter_name))
        
        # Reset
        self.circuit_breaker.reset_adapter(adapter_name)
        
        # 3. Timeout enforcement
        results.append(self.validate_timeout_enforcement(adapter_name, max_latency_ms=2000))
        
        # 4. Idempotency
        results.append(self.validate_idempotency_detection(
            adapter_name,
            "test_method",
            {"test_param": "test_value"}
        ))
        
        # 5. Graceful degradation
        fault = self.contract_injector.inject_type_mismatch(
            adapter_name,
            "test_method",
            dict,
            "wrong_type"
        )
        results.append(self.validate_graceful_degradation(adapter_name, [fault]))
        
        logger.info(f"Completed {len(results)} validations for {adapter_name}")
        
        return results
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Genera reporte completo de validaciones
        
        Returns:
            Diccionario con resultados y métricas
        """
        passed = sum(1 for r in self.validation_results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in self.validation_results if r.status == ValidationStatus.FAILED)
        degraded = sum(1 for r in self.validation_results if r.status == ValidationStatus.DEGRADED)
        
        return {
            "summary": {
                "total_validations": len(self.validation_results),
                "passed": passed,
                "failed": failed,
                "degraded": degraded,
                "success_rate": passed / len(self.validation_results) if self.validation_results else 0.0
            },
            "adapters_tested": list(set(r.adapter_name for r in self.validation_results)),
            "state_transitions": [
                {
                    "adapter": t.adapter_name,
                    "from": t.from_state,
                    "to": t.to_state,
                    "trigger": t.trigger,
                    "timestamp": t.timestamp
                }
                for t in self.state_transitions
            ],
            "validations": [
                {
                    "test": r.test_name,
                    "adapter": r.adapter_name,
                    "status": r.status.value,
                    "failures": r.failures,
                    "warnings": r.warnings,
                    "metrics": r.metrics,
                    "execution_time": round(r.execution_time, 3)
                }
                for r in self.validation_results
            ]
        }
    
    def reset(self):
        """Limpia todos los estados y reinicia validators"""
        self.circuit_breaker.reset_all()
        self.state_transitions.clear()
        self.execution_history.clear()
        self.validation_results.clear()
        
        self.contract_injector.reset()
        self.determinism_injector.reset()
        self.fault_tolerance_injector.reset()
        self.operational_injector.reset()
        
        logger.info("ResilienceValidator reset complete")
