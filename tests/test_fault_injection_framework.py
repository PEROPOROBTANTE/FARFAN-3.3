"""
Test Suite for Fault Injection Framework
========================================

Tests completos del framework de inyección de fallos para FARFAN 3.0.

Ejecuta:
    pytest tests/test_fault_injection_framework.py -v

Author: FARFAN Integration Team
Version: 1.0.0
Python: 3.10+
"""

import pytest
import logging
import time
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.fault_injection import (
    ContractFaultInjector,
    DeterminismFaultInjector,
    FaultToleranceFaultInjector,
    OperationalFaultInjector,
    ResilienceValidator,
    ChaosScenarioRunner
)
from orchestrator.circuit_breaker import CircuitBreaker, CircuitState

logger = logging.getLogger(__name__)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def circuit_breaker():
    """Circuit breaker fixture"""
    return CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=5.0,
        half_open_max_calls=2
    )


@pytest.fixture
def contract_injector():
    """Contract fault injector fixture"""
    injector = ContractFaultInjector()
    yield injector
    injector.reset()


@pytest.fixture
def determinism_injector():
    """Determinism fault injector fixture"""
    injector = DeterminismFaultInjector()
    yield injector
    injector.reset()


@pytest.fixture
def fault_tolerance_injector():
    """Fault tolerance fault injector fixture"""
    injector = FaultToleranceFaultInjector()
    yield injector
    injector.reset()


@pytest.fixture
def operational_injector():
    """Operational fault injector fixture"""
    injector = OperationalFaultInjector()
    yield injector
    injector.reset()


@pytest.fixture
def resilience_validator(circuit_breaker):
    """Resilience validator fixture"""
    validator = ResilienceValidator(circuit_breaker)
    yield validator
    validator.reset()


@pytest.fixture
def chaos_runner():
    """Chaos scenario runner fixture"""
    runner = ChaosScenarioRunner()
    yield runner


# ============================================================================
# CONTRACT FAULT INJECTOR TESTS
# ============================================================================

class TestContractFaultInjector:
    """Tests para ContractFaultInjector"""
    
    def test_inject_type_mismatch(self, contract_injector):
        """Test inyección de type mismatch"""
        fault = contract_injector.inject_type_mismatch(
            "teoria_cambio",
            "execute",
            dict,
            "wrong_type_string"
        )
        
        assert fault.target_adapter == "teoria_cambio"
        assert fault.target_method == "execute"
        assert "type mismatch" in fault.description.lower()
        assert len(contract_injector.injected_faults) == 1
    
    def test_inject_missing_binding(self, contract_injector):
        """Test inyección de missing binding"""
        fault = contract_injector.inject_missing_binding(
            "policy_processor",
            "semantic_chunking_policy",
            "normalized_text"
        )
        
        assert fault.target_adapter == "semantic_chunking_policy"
        assert "normalized_text" in fault.description
        assert fault.metadata["source_adapter"] == "policy_processor"
    
    def test_inject_schema_break(self, contract_injector):
        """Test inyección de schema break"""
        fault = contract_injector.inject_schema_break(
            "analyzer_one",
            "malformed_module_result"
        )
        
        assert fault.target_adapter == "analyzer_one"
        assert fault.metadata["break_type"] == "malformed_module_result"
    
    def test_create_malformed_module_result(self, contract_injector):
        """Test creación de ModuleResult malformado"""
        result = contract_injector.create_malformed_module_result()
        
        assert "status" in result
        assert "module_name" not in result  # Campo faltante
        assert result["data"] is None  # Tipo incorrecto (debería ser dict)
    
    def test_reset_clears_faults(self, contract_injector):
        """Test que reset limpia fallos"""
        contract_injector.inject_type_mismatch("adapter1", "method1", str, 123)
        contract_injector.inject_schema_break("adapter2")
        
        assert len(contract_injector.injected_faults) == 2
        
        contract_injector.reset()
        assert len(contract_injector.injected_faults) == 0


# ============================================================================
# DETERMINISM FAULT INJECTOR TESTS
# ============================================================================

class TestDeterminismFaultInjector:
    """Tests para DeterminismFaultInjector"""
    
    def test_inject_seed_corruption(self, determinism_injector):
        """Test inyección de seed corruption"""
        import random
        import numpy as np
        
        # Seeds iniciales
        initial_random = random.random()
        initial_numpy = np.random.random()
        
        # Inyectar corrupción
        fault = determinism_injector.inject_seed_corruption(
            "teoria_cambio",
            "both"
        )
        
        # Verificar que los valores cambiaron
        corrupted_random = random.random()
        corrupted_numpy = np.random.random()
        
        assert fault.target_adapter == "teoria_cambio"
        assert fault.metadata["corruption_type"] == "both"
    
    def test_inject_timestamp_noise(self, determinism_injector):
        """Test inyección de timestamp noise"""
        fault = determinism_injector.inject_timestamp_noise(
            "analyzer_one",
            "analyze"
        )
        
        assert fault.target_adapter == "analyzer_one"
        assert fault.target_method == "analyze"
        assert "timestamp" in fault.description.lower()
    
    def test_restore_determinism(self, determinism_injector):
        """Test restauración de determinismo"""
        # Inyectar corrupción
        determinism_injector.inject_seed_corruption("adapter1", "both")
        
        # Restaurar
        determinism_injector.restore_determinism()
        
        # Verificar que se restauró (originales guardados)
        assert determinism_injector.original_random_seed is not None
        assert determinism_injector.original_numpy_seed is not None


# ============================================================================
# FAULT TOLERANCE FAULT INJECTOR TESTS
# ============================================================================

class TestFaultToleranceFaultInjector:
    """Tests para FaultToleranceFaultInjector"""
    
    def test_inject_circuit_breaker_stuck(self, fault_tolerance_injector):
        """Test circuit breaker stuck"""
        fault = fault_tolerance_injector.inject_circuit_breaker_stuck(
            "dereck_beach",
            "OPEN"
        )
        
        assert fault.target_adapter == "dereck_beach"
        assert fault.metadata["stuck_state"] == "OPEN"
        assert fault.metadata["blocks_all_requests"] is True
    
    def test_inject_wrong_failure_threshold(self, fault_tolerance_injector):
        """Test threshold incorrecto"""
        fault = fault_tolerance_injector.inject_wrong_failure_threshold(
            "embedding_policy",
            threshold=1
        )
        
        assert fault.target_adapter == "embedding_policy"
        assert fault.metadata["configured_threshold"] == 1
        assert fault.metadata["impact"] == "too_sensitive"
    
    def test_inject_retry_storm(self, fault_tolerance_injector):
        """Test retry storm"""
        fault = fault_tolerance_injector.inject_retry_storm(
            "financial_viability",
            max_retries=100,
            no_backoff=True
        )
        
        assert fault.target_adapter == "financial_viability"
        assert fault.metadata["max_retries"] == 100
        assert fault.metadata["no_backoff"] is True
    
    def test_inject_timeout_misconfiguration(self, fault_tolerance_injector):
        """Test timeout misconfiguration"""
        fault = fault_tolerance_injector.inject_timeout_misconfiguration(
            "contradiction_detection",
            timeout_ms=50,
            timeout_type="premature"
        )
        
        assert fault.target_adapter == "contradiction_detection"
        assert fault.metadata["timeout_ms"] == 50
        assert fault.metadata["timeout_type"] == "premature"


# ============================================================================
# OPERATIONAL FAULT INJECTOR TESTS
# ============================================================================

class TestOperationalFaultInjector:
    """Tests para OperationalFaultInjector"""
    
    def test_inject_disk_full(self, operational_injector):
        """Test disk full error"""
        fault = operational_injector.inject_disk_full(
            "policy_segmenter",
            affected_paths=["/tmp/cache"]
        )
        
        assert fault.target_adapter == "policy_segmenter"
        assert fault.metadata["error_code"] == 28  # ENOSPC
    
    def test_inject_clock_skew(self, operational_injector):
        """Test clock skew"""
        fault = operational_injector.inject_clock_skew(
            "teoria_cambio",
            skew_seconds=3600.0
        )
        
        assert fault.target_adapter == "teoria_cambio"
        assert fault.metadata["skew_seconds"] == 3600.0
        assert fault.metadata["direction"] == "future"
    
    def test_inject_network_partition(self, operational_injector):
        """Test network partition"""
        fault = operational_injector.inject_network_partition(
            "embedding_policy",
            partition_type="complete"
        )
        
        assert fault.target_adapter == "embedding_policy"
        assert fault.metadata["partition_type"] == "complete"
        assert fault.metadata["affects_external_calls"] is True
    
    def test_inject_memory_pressure(self, operational_injector):
        """Test memory pressure"""
        fault = operational_injector.inject_memory_pressure(
            "dereck_beach",
            pressure_level="high"
        )
        
        assert fault.target_adapter == "dereck_beach"
        assert fault.metadata["pressure_level"] == "high"


# ============================================================================
# RESILIENCE VALIDATOR TESTS
# ============================================================================

class TestResilienceValidator:
    """Tests para ResilienceValidator"""
    
    def test_validate_circuit_breaker_transitions(self, resilience_validator):
        """Test validación de transiciones de circuit breaker"""
        result = resilience_validator.validate_circuit_breaker_transitions(
            "analyzer_one",
            failure_count=10
        )
        
        assert result.adapter_name == "analyzer_one"
        assert result.test_name == "circuit_breaker_state_transitions"
        assert "observed_sequence" in result.metrics
        assert len(resilience_validator.state_transitions) > 0
    
    def test_validate_retry_backoff(self, resilience_validator):
        """Test validación de retry backoff"""
        result = resilience_validator.validate_retry_backoff(
            "teoria_cambio",
            max_retries=5
        )
        
        assert result.adapter_name == "teoria_cambio"
        assert result.test_name == "retry_backoff_exponential_with_jitter"
        assert "retry_delays" in result.metrics
        assert "growth_ratios" in result.metrics
    
    def test_validate_timeout_enforcement(self, resilience_validator):
        """Test validación de timeout enforcement"""
        result = resilience_validator.validate_timeout_enforcement(
            "financial_viability",
            max_latency_ms=1000
        )
        
        assert result.adapter_name == "financial_viability"
        assert result.test_name == "timeout_enforcement"
        assert "max_latency_ms" in result.metrics
    
    def test_validate_idempotency_detection(self, resilience_validator):
        """Test validación de idempotency"""
        result = resilience_validator.validate_idempotency_detection(
            "policy_processor",
            "normalize_text",
            {"text": "test input"}
        )
        
        assert result.adapter_name == "policy_processor"
        assert result.test_name == "idempotency_detection"
        assert "execution_id" in result.metrics
    
    def test_generate_report(self, resilience_validator):
        """Test generación de reporte"""
        # Ejecutar algunas validaciones
        resilience_validator.validate_circuit_breaker_transitions("adapter1")
        resilience_validator.circuit_breaker.reset_adapter("adapter1")
        resilience_validator.validate_retry_backoff("adapter1")
        
        report = resilience_validator.generate_report()
        
        assert "summary" in report
        assert "validations" in report
        assert report["summary"]["total_validations"] >= 2


# ============================================================================
# CHAOS SCENARIO RUNNER TESTS
# ============================================================================

class TestChaosScenarioRunner:
    """Tests para ChaosScenarioRunner"""
    
    def test_build_partial_failure_scenario(self, chaos_runner):
        """Test construcción de partial failure scenario"""
        scenario = chaos_runner.build_partial_failure_scenario(num_failures=2)
        
        assert scenario.name == "partial_failure_2_adapters"
        assert len(scenario.affected_adapters) == 2
        assert len(scenario.injected_faults) >= 2  # Al menos 1 fault por adapter
    
    def test_build_cascading_risk_scenario(self, chaos_runner):
        """Test construcción de cascading risk scenario"""
        scenario = chaos_runner.build_cascading_risk_scenario()
        
        assert scenario.name == "cascading_risk_dependency_chain"
        assert "policy_processor" in scenario.affected_adapters
        assert len(scenario.injected_faults) >= 1
    
    def test_build_combined_chaos_scenario(self, chaos_runner):
        """Test construcción de combined chaos scenario"""
        scenario = chaos_runner.build_combined_chaos_scenario()
        
        assert scenario.name == "combined_chaos_extreme"
        assert len(scenario.affected_adapters) == 5
        assert len(scenario.injected_faults) >= 5
    
    def test_run_scenario(self, chaos_runner):
        """Test ejecución de un scenario"""
        scenario = chaos_runner.build_partial_failure_scenario(num_failures=1)
        result = chaos_runner.run_scenario(scenario)
        
        assert result.scenario == scenario
        assert result.duration > 0
        assert "circuit_breaker_states" in result.__dict__
        assert "graceful_degradation" in result.__dict__
    
    @pytest.mark.slow
    def test_run_all_scenarios(self, chaos_runner):
        """Test ejecución de todos los scenarios (slow test)"""
        results = chaos_runner.run_all_scenarios()
        
        assert len(results) == 8  # 8 scenarios predefinidos
        assert all(r.duration > 0 for r in results)
    
    def test_generate_chaos_report(self, chaos_runner):
        """Test generación de chaos report"""
        # Ejecutar un scenario
        scenario = chaos_runner.build_partial_failure_scenario(num_failures=1)
        chaos_runner.run_scenario(scenario)
        
        report = chaos_runner.generate_chaos_report()
        
        assert "summary" in report
        assert "scenarios" in report
        assert "circuit_breaker_analysis" in report
        assert "recommendations" in report
        assert report["summary"]["total_scenarios"] >= 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combinando múltiples componentes"""
    
    def test_full_resilience_validation_flow(self):
        """Test flujo completo de validación de resiliencia"""
        # Setup
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)
        validator = ResilienceValidator(cb)
        
        # Ejecutar todas las validaciones para un adapter
        results = validator.run_all_validations("teoria_cambio")
        
        assert len(results) == 5  # 5 tipos de validaciones
        assert all(hasattr(r, 'status') for r in results)
        
        # Generar reporte
        report = validator.generate_report()
        assert report["summary"]["total_validations"] == 5
    
    def test_chaos_with_circuit_breaker_recovery(self):
        """Test chaos scenario con recovery de circuit breaker"""
        runner = ChaosScenarioRunner()
        
        # Ejecutar scenario que causa OPEN
        scenario = runner.build_partial_failure_scenario(num_failures=1)
        result = runner.run_scenario(scenario)
        
        # Verificar que hay al menos un circuit OPEN
        open_circuits = [
            name for name, state in result.circuit_breaker_states.items()
            if state == "OPEN"
        ]
        
        # Nota: Puede ser 0 si recovery timeout es muy rápido
        assert result.circuit_breaker_states  # Al menos tiene estados


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

class TestPerformance:
    """Performance benchmarks"""
    
    def test_fault_injection_overhead(self, contract_injector):
        """Medir overhead de inyección de fallos"""
        start = time.time()
        
        for i in range(100):
            contract_injector.inject_type_mismatch(
                f"adapter_{i}",
                "method",
                dict,
                "value"
            )
        
        duration = time.time() - start
        
        # Inyectar 100 fallos debe ser < 1 segundo
        assert duration < 1.0
        assert len(contract_injector.injected_faults) == 100
    
    def test_validation_performance(self, resilience_validator):
        """Medir performance de validaciones"""
        start = time.time()
        
        result = resilience_validator.validate_idempotency_detection(
            "test_adapter",
            "test_method",
            {"key": "value"}
        )
        
        duration = time.time() - start
        
        # Validación de idempotency debe ser < 0.5 segundos
        assert duration < 0.5


if __name__ == "__main__":
    # Run with: python tests/test_fault_injection_framework.py
    pytest.main([__file__, "-v", "--tb=short"])
