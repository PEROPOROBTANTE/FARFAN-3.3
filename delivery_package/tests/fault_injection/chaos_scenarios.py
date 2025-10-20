"""
Chaos Testing Scenarios for FARFAN 3.0
======================================

Combina múltiples tipos de fallos simultáneamente para validar:
- Graceful degradation bajo fallo combinado
- No cascading failures entre adapters
- Circuit breaker correctness bajo presión
- System stability con múltiples faults activos

Author: FARFAN Integration Team
Version: 1.0.0
Python: 3.10+
"""

import logging
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
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
    FaultSeverity as InjectorFaultSeverity,
    InjectedFault
)
from tests.fault_injection.resilience_validator import (
    ResilienceValidator,
    ValidationResult,
    ValidationStatus
)

logger = logging.getLogger(__name__)


# ============================================================================
# CHAOS SCENARIO TYPES
# ============================================================================

class ChaosScenarioType(Enum):
    """Tipos de escenarios caóticos"""
    PARTIAL_FAILURE = "partial_failure"              # 1-3 adapters fallan
    CASCADING_RISK = "cascading_risk"                # Fallo en dependency chain
    NETWORK_PARTITION = "network_partition"          # Network issues
    RESOURCE_EXHAUSTION = "resource_exhaustion"      # Memory/disk pressure
    TIMING_ISSUES = "timing_issues"                  # Clock skew, timeouts
    CONTRACT_VIOLATIONS = "contract_violations"      # Type mismatches, schema breaks
    DETERMINISM_BREAK = "determinism_break"          # Non-reproducible results
    COMBINED_CHAOS = "combined_chaos"                # Múltiples categorías simultáneas


@dataclass
class ChaosScenario:
    """Definición de un escenario caótico"""
    name: str
    scenario_type: ChaosScenarioType
    description: str
    affected_adapters: List[str]
    injected_faults: List[InjectedFault] = field(default_factory=list)
    expected_behavior: str = ""
    severity: InjectorFaultSeverity = InjectorFaultSeverity.MEDIUM


@dataclass
class ChaosTestResult:
    """Resultado de un chaos test"""
    scenario: ChaosScenario
    status: ValidationStatus
    duration: float
    validations: List[ValidationResult]
    circuit_breaker_states: Dict[str, str]
    cascading_failures: List[str] = field(default_factory=list)
    graceful_degradation: bool = False
    recovery_successful: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# CHAOS SCENARIO RUNNER
# ============================================================================

class ChaosScenarioRunner:
    """
    Ejecuta chaos testing scenarios combinando múltiples fault types
    
    Features:
    - Combina 2+ fault categories simultáneamente
    - Valida no hay cascading failures
    - Verifica graceful degradation
    - Monitorea circuit breaker states
    """
    
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
    
    def __init__(self):
        """Initialize chaos scenario runner"""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=15.0,
            half_open_max_calls=3
        )
        
        self.validator = ResilienceValidator(self.circuit_breaker)
        
        # Injectors
        self.contract_injector = ContractFaultInjector()
        self.determinism_injector = DeterminismFaultInjector()
        self.fault_tolerance_injector = FaultToleranceFaultInjector()
        self.operational_injector = OperationalFaultInjector()
        
        self.test_results: List[ChaosTestResult] = []
        
        logger.info("ChaosScenarioRunner initialized")
    
    # ========================================================================
    # SCENARIO BUILDERS
    # ========================================================================
    
    def build_partial_failure_scenario(self, num_failures: int = 2) -> ChaosScenario:
        """
        Scenario: 1-3 adapters fallan simultáneamente
        
        Args:
            num_failures: Número de adapters a fallar (1-3)
            
        Returns:
            ChaosScenario configurado
        """
        affected = random.sample(self.ADAPTERS, min(num_failures, 3))
        
        scenario = ChaosScenario(
            name=f"partial_failure_{num_failures}_adapters",
            scenario_type=ChaosScenarioType.PARTIAL_FAILURE,
            description=f"Simulate {num_failures} adapter failures simultaneously",
            affected_adapters=affected,
            expected_behavior="Other adapters continue operating, no cascading failures",
            severity=InjectorFaultSeverity.HIGH
        )
        
        # Inyectar fallos en cada adapter
        for adapter in affected:
            # Contract fault: type mismatch
            fault1 = self.contract_injector.inject_type_mismatch(
                adapter, "test_method", dict, "wrong_type"
            )
            scenario.injected_faults.append(fault1)
            
            # Operational fault: disk full
            fault2 = self.operational_injector.inject_disk_full(adapter)
            scenario.injected_faults.append(fault2)
        
        return scenario
    
    def build_cascading_risk_scenario(self) -> ChaosScenario:
        """
        Scenario: Fallo en dependency chain (policy_processor → semantic → analyzer_one)
        
        Returns:
            ChaosScenario configurado
        """
        # Dependency chain crítica
        dependency_chain = ["policy_processor", "semantic_chunking_policy", "analyzer_one"]
        
        scenario = ChaosScenario(
            name="cascading_risk_dependency_chain",
            scenario_type=ChaosScenarioType.CASCADING_RISK,
            description="Fail first adapter in dependency chain, validate no cascade",
            affected_adapters=dependency_chain,
            expected_behavior="Downstream adapters use fallbacks, no cascade",
            severity=InjectorFaultSeverity.CRITICAL
        )
        
        # Fallar el primero en la cadena
        first_adapter = dependency_chain[0]
        
        # Contract fault: missing binding
        fault1 = self.contract_injector.inject_missing_binding(
            first_adapter, dependency_chain[1], "normalized_text"
        )
        scenario.injected_faults.append(fault1)
        
        # Fault tolerance: circuit breaker stuck OPEN
        fault2 = self.fault_tolerance_injector.inject_circuit_breaker_stuck(
            first_adapter, "OPEN"
        )
        scenario.injected_faults.append(fault2)
        
        return scenario
    
    def build_network_partition_scenario(self) -> ChaosScenario:
        """
        Scenario: Network partition afecta adapters externos
        
        Returns:
            ChaosScenario configurado
        """
        # Adapters que podrían hacer llamadas externas
        affected = ["embedding_policy", "financial_viability", "dereck_beach"]
        
        scenario = ChaosScenario(
            name="network_partition",
            scenario_type=ChaosScenarioType.NETWORK_PARTITION,
            description="Simulate network partition affecting external calls",
            affected_adapters=affected,
            expected_behavior="Adapters use cached results or fallbacks",
            severity=InjectorFaultSeverity.CRITICAL
        )
        
        for adapter in affected:
            # Network partition
            fault = self.operational_injector.inject_network_partition(
                adapter, partition_type="complete"
            )
            scenario.injected_faults.append(fault)
        
        return scenario
    
    def build_resource_exhaustion_scenario(self) -> ChaosScenario:
        """
        Scenario: Resource exhaustion (memory + disk)
        
        Returns:
            ChaosScenario configurado
        """
        # Adapters con alto uso de recursos
        affected = ["teoria_cambio", "financial_viability", "dereck_beach"]
        
        scenario = ChaosScenario(
            name="resource_exhaustion",
            scenario_type=ChaosScenarioType.RESOURCE_EXHAUSTION,
            description="Simulate memory pressure and disk full",
            affected_adapters=affected,
            expected_behavior="Graceful degradation, no OOM crashes",
            severity=InjectorFaultSeverity.CRITICAL
        )
        
        for adapter in affected:
            # Memory pressure
            fault1 = self.operational_injector.inject_memory_pressure(
                adapter, pressure_level="high"
            )
            scenario.injected_faults.append(fault1)
            
            # Disk full
            fault2 = self.operational_injector.inject_disk_full(adapter)
            scenario.injected_faults.append(fault2)
        
        return scenario
    
    def build_timing_issues_scenario(self) -> ChaosScenario:
        """
        Scenario: Timing issues (clock skew + timeouts)
        
        Returns:
            ChaosScenario configurado
        """
        affected = ["analyzer_one", "contradiction_detection", "policy_processor"]
        
        scenario = ChaosScenario(
            name="timing_issues",
            scenario_type=ChaosScenarioType.TIMING_ISSUES,
            description="Simulate clock skew and premature timeouts",
            affected_adapters=affected,
            expected_behavior="Timeout detection works, clock skew handled",
            severity=InjectorFaultSeverity.HIGH
        )
        
        for i, adapter in enumerate(affected):
            if i == 0:
                # Clock skew
                fault = self.operational_injector.inject_clock_skew(
                    adapter, skew_seconds=3600.0
                )
                scenario.injected_faults.append(fault)
            else:
                # Premature timeout
                fault = self.fault_tolerance_injector.inject_timeout_misconfiguration(
                    adapter, timeout_ms=10, timeout_type="premature"
                )
                scenario.injected_faults.append(fault)
        
        return scenario
    
    def build_contract_violations_scenario(self) -> ChaosScenario:
        """
        Scenario: Contract violations (type mismatches + schema breaks)
        
        Returns:
            ChaosScenario configurado
        """
        affected = random.sample(self.ADAPTERS, 3)
        
        scenario = ChaosScenario(
            name="contract_violations",
            scenario_type=ChaosScenarioType.CONTRACT_VIOLATIONS,
            description="Simulate contract violations across adapters",
            affected_adapters=affected,
            expected_behavior="Validation catches violations, graceful handling",
            severity=InjectorFaultSeverity.CRITICAL
        )
        
        for i, adapter in enumerate(affected):
            if i == 0:
                # Type mismatch
                fault = self.contract_injector.inject_type_mismatch(
                    adapter, "execute", dict, [1, 2, 3]
                )
            elif i == 1:
                # Missing binding
                fault = self.contract_injector.inject_missing_binding(
                    adapter, affected[2], "required_data"
                )
            else:
                # Schema break
                fault = self.contract_injector.inject_schema_break(
                    adapter, "malformed_module_result"
                )
            
            scenario.injected_faults.append(fault)
        
        return scenario
    
    def build_determinism_break_scenario(self) -> ChaosScenario:
        """
        Scenario: Determinism break (seed corruption + random noise)
        
        Returns:
            ChaosScenario configurado
        """
        affected = ["teoria_cambio", "dereck_beach", "analyzer_one"]
        
        scenario = ChaosScenario(
            name="determinism_break",
            scenario_type=ChaosScenarioType.DETERMINISM_BREAK,
            description="Simulate determinism break with seed corruption",
            affected_adapters=affected,
            expected_behavior="Non-reproducible results detected",
            severity=InjectorFaultSeverity.MEDIUM
        )
        
        for adapter in affected:
            # Seed corruption
            fault1 = self.determinism_injector.inject_seed_corruption(
                adapter, corruption_type="both"
            )
            scenario.injected_faults.append(fault1)
            
            # Random noise
            fault2 = self.determinism_injector.inject_random_noise(
                adapter, noise_level=0.2
            )
            scenario.injected_faults.append(fault2)
        
        return scenario
    
    def build_combined_chaos_scenario(self) -> ChaosScenario:
        """
        Scenario: Combined chaos - múltiples categorías simultáneas
        
        Returns:
            ChaosScenario configurado
        """
        # Seleccionar adapters aleatorios
        affected = random.sample(self.ADAPTERS, 5)
        
        scenario = ChaosScenario(
            name="combined_chaos_extreme",
            scenario_type=ChaosScenarioType.COMBINED_CHAOS,
            description="Extreme chaos: multiple fault categories simultaneously",
            affected_adapters=affected,
            expected_behavior="System remains operational with degradation",
            severity=InjectorFaultSeverity.CRITICAL
        )
        
        # Combinar todos los tipos de fallos
        for i, adapter in enumerate(affected):
            fault_type = i % 4
            
            if fault_type == 0:
                # Contract violation
                fault = self.contract_injector.inject_type_mismatch(
                    adapter, "method", dict, None
                )
            elif fault_type == 1:
                # Determinism break
                fault = self.determinism_injector.inject_seed_corruption(
                    adapter, "both"
                )
            elif fault_type == 2:
                # Fault tolerance
                fault = self.fault_tolerance_injector.inject_circuit_breaker_stuck(
                    adapter, "OPEN"
                )
            else:
                # Operational
                fault = self.operational_injector.inject_network_partition(
                    adapter, "complete"
                )
            
            scenario.injected_faults.append(fault)
        
        return scenario
    
    # ========================================================================
    # SCENARIO EXECUTION
    # ========================================================================
    
    def run_scenario(self, scenario: ChaosScenario) -> ChaosTestResult:
        """
        Ejecuta un chaos scenario completo
        
        Args:
            scenario: Scenario a ejecutar
            
        Returns:
            ChaosTestResult con resultados
        """
        logger.info(f"Running chaos scenario: {scenario.name}")
        start_time = time.time()
        
        # Capturar estado inicial
        initial_states = {
            adapter: self.circuit_breaker.adapter_states.get(adapter, CircuitState.CLOSED).name
            for adapter in self.ADAPTERS
        }
        
        validations: List[ValidationResult] = []
        
        try:
            # Los fallos ya están inyectados en el scenario
            # Ahora ejecutar validaciones
            
            # 1. Validar cada adapter afectado
            for adapter in scenario.affected_adapters:
                # Simular fallos en circuit breaker
                for fault in scenario.injected_faults:
                    if fault.target_adapter == adapter:
                        self.circuit_breaker.record_failure(
                            adapter,
                            fault.description,
                            execution_time=0.1,
                            severity=FailureSeverity.CRITICAL
                        )
                
                # Ejecutar validaciones
                result = self.validator.validate_graceful_degradation(
                    adapter,
                    [f for f in scenario.injected_faults if f.target_adapter == adapter]
                )
                validations.append(result)
            
            # 2. Esperar un poco para observar comportamiento
            time.sleep(1.0)
            
            # 3. Capturar estado final
            final_states = {
                adapter: self.circuit_breaker.adapter_states.get(adapter, CircuitState.CLOSED).name
                for adapter in self.ADAPTERS
            }
            
            # 4. Detectar cascading failures
            cascading = []
            for adapter in self.ADAPTERS:
                if adapter not in scenario.affected_adapters:
                    # Adapter NO afectado directamente
                    if final_states[adapter] in ["OPEN", "ISOLATED"]:
                        cascading.append(
                            f"{adapter}: state={final_states[adapter]} (not directly affected)"
                        )
            
            # 5. Verificar graceful degradation
            graceful = self._check_graceful_degradation(scenario, final_states)
            
            # 6. Determinar status general
            if cascading:
                status = ValidationStatus.FAILED
            elif any(v.status == ValidationStatus.FAILED for v in validations):
                status = ValidationStatus.FAILED
            elif any(v.status == ValidationStatus.DEGRADED for v in validations):
                status = ValidationStatus.DEGRADED
            else:
                status = ValidationStatus.PASSED
            
            duration = time.time() - start_time
            
            result = ChaosTestResult(
                scenario=scenario,
                status=status,
                duration=duration,
                validations=validations,
                circuit_breaker_states=final_states,
                cascading_failures=cascading,
                graceful_degradation=graceful,
                recovery_successful=False,  # Se verificaría en recovery phase
                metrics={
                    "initial_states": initial_states,
                    "final_states": final_states,
                    "faults_injected": len(scenario.injected_faults),
                    "adapters_affected": len(scenario.affected_adapters),
                    "adapters_failed": sum(1 for s in final_states.values() 
                                          if s in ["OPEN", "ISOLATED"]),
                    "healthy_adapters": sum(1 for s in final_states.values() 
                                           if s == "CLOSED")
                }
            )
            
            self.test_results.append(result)
            logger.info(f"Chaos scenario completed: {scenario.name} - {status.value}")
            
        except Exception as e:
            logger.exception(f"Error running chaos scenario {scenario.name}")
            result = ChaosTestResult(
                scenario=scenario,
                status=ValidationStatus.FAILED,
                duration=time.time() - start_time,
                validations=validations,
                circuit_breaker_states={},
                metrics={"error": str(e)}
            )
            self.test_results.append(result)
        
        return result
    
    def _check_graceful_degradation(
        self,
        scenario: ChaosScenario,
        final_states: Dict[str, str]
    ) -> bool:
        """
        Verifica si hubo degradación graceful
        
        Args:
            scenario: Scenario ejecutado
            final_states: Estados finales de circuit breakers
            
        Returns:
            True si degradación fue graceful
        """
        # Criterios de graceful degradation:
        # 1. Adapters no afectados siguen en CLOSED
        # 2. Hay fallback strategies disponibles
        # 3. No crashes/excepciones no manejadas
        
        unaffected_healthy = all(
            final_states.get(adapter) == "CLOSED"
            for adapter in self.ADAPTERS
            if adapter not in scenario.affected_adapters
        )
        
        # Verificar fallbacks disponibles
        has_fallbacks = all(
            self.circuit_breaker.get_fallback_strategy(adapter).get("use_cached", False)
            for adapter in scenario.affected_adapters
        )
        
        return unaffected_healthy and has_fallbacks
    
    # ========================================================================
    # RUN ALL SCENARIOS
    # ========================================================================
    
    def run_all_scenarios(self) -> List[ChaosTestResult]:
        """
        Ejecuta todos los chaos scenarios predefinidos
        
        Returns:
            Lista de ChaosTestResults
        """
        logger.info("Running all chaos scenarios...")
        
        scenarios = [
            self.build_partial_failure_scenario(num_failures=2),
            self.build_cascading_risk_scenario(),
            self.build_network_partition_scenario(),
            self.build_resource_exhaustion_scenario(),
            self.build_timing_issues_scenario(),
            self.build_contract_violations_scenario(),
            self.build_determinism_break_scenario(),
            self.build_combined_chaos_scenario()
        ]
        
        results = []
        for scenario in scenarios:
            result = self.run_scenario(scenario)
            results.append(result)
            
            # Reset entre scenarios
            self._reset_between_scenarios()
            time.sleep(2.0)  # Cooldown
        
        logger.info(f"Completed {len(results)} chaos scenarios")
        return results
    
    def _reset_between_scenarios(self):
        """Reset completo entre scenarios"""
        self.circuit_breaker.reset_all()
        self.validator.reset()
        self.contract_injector.reset()
        self.determinism_injector.reset()
        self.fault_tolerance_injector.reset()
        self.operational_injector.reset()
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def generate_chaos_report(self) -> Dict[str, Any]:
        """
        Genera reporte completo de chaos testing
        
        Returns:
            Diccionario con resultados y análisis
        """
        if not self.test_results:
            return {"error": "No test results available"}
        
        passed = sum(1 for r in self.test_results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in self.test_results if r.status == ValidationStatus.FAILED)
        degraded = sum(1 for r in self.test_results if r.status == ValidationStatus.DEGRADED)
        
        total_cascading = sum(len(r.cascading_failures) for r in self.test_results)
        graceful_count = sum(1 for r in self.test_results if r.graceful_degradation)
        
        return {
            "summary": {
                "total_scenarios": len(self.test_results),
                "passed": passed,
                "failed": failed,
                "degraded": degraded,
                "success_rate": passed / len(self.test_results),
                "total_cascading_failures": total_cascading,
                "graceful_degradation_rate": graceful_count / len(self.test_results)
            },
            "scenarios": [
                {
                    "name": r.scenario.name,
                    "type": r.scenario.scenario_type.value,
                    "status": r.status.value,
                    "duration": round(r.duration, 3),
                    "affected_adapters": r.scenario.affected_adapters,
                    "faults_injected": len(r.scenario.injected_faults),
                    "cascading_failures": r.cascading_failures,
                    "graceful_degradation": r.graceful_degradation,
                    "circuit_breaker_states": r.circuit_breaker_states,
                    "metrics": r.metrics
                }
                for r in self.test_results
            ],
            "circuit_breaker_analysis": self.circuit_breaker.get_all_status(),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Genera recomendaciones basadas en resultados"""
        recommendations = []
        
        # Analizar cascading failures
        total_cascading = sum(len(r.cascading_failures) for r in self.test_results)
        if total_cascading > 0:
            recommendations.append(
                f"⚠️ Detected {total_cascading} cascading failures - "
                "review adapter isolation boundaries"
            )
        
        # Analizar graceful degradation
        graceful_count = sum(1 for r in self.test_results if r.graceful_degradation)
        if graceful_count < len(self.test_results) * 0.8:
            recommendations.append(
                "⚠️ Graceful degradation rate below 80% - "
                "improve fallback strategies"
            )
        
        # Analizar circuit breaker effectiveness
        cb_stats = self.circuit_breaker.get_all_status()
        stuck_circuits = [
            name for name, status in cb_stats.items()
            if status.get("state") in ["OPEN", "ISOLATED"]
        ]
        if stuck_circuits:
            recommendations.append(
                f"⚠️ Circuit breakers stuck in OPEN/ISOLATED: {stuck_circuits} - "
                "review recovery timeout configuration"
            )
        
        if not recommendations:
            recommendations.append("✓ System demonstrates excellent chaos resilience")
        
        return recommendations
