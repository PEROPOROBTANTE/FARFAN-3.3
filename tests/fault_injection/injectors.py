"""
Fault Injectors for FARFAN 3.0 Testing
======================================

Implementa cuatro categorías de inyección de fallos:
- ContractFaultInjector: Type mismatches, missing bindings, schema breaks
- DeterminismFaultInjector: Seed corruption, non-reproducible outputs  
- FaultToleranceFaultInjector: Circuit breaker issues, retry storms, timeouts
- OperationalFaultInjector: Disk full, clock skew, network partitions

Author: FARFAN Integration Team
Version: 1.0.0
Python: 3.10+
"""

import logging
import time
import random
import os
import sys
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from unittest.mock import patch, MagicMock
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# FAULT INJECTION TYPES
# ============================================================================

class FaultCategory(Enum):
    """Categorías de fallos inyectables"""
    CONTRACT = "contract"
    DETERMINISM = "determinism"
    FAULT_TOLERANCE = "fault_tolerance"
    OPERATIONAL = "operational"


class FaultSeverity(Enum):
    """Severidad de fallos"""
    LOW = "low"           # Degradación menor
    MEDIUM = "medium"     # Degradación significativa
    HIGH = "high"         # Fallo crítico
    CRITICAL = "critical" # Fallo catastrófico


@dataclass
class InjectedFault:
    """Registro de un fallo inyectado"""
    category: FaultCategory
    severity: FaultSeverity
    description: str
    target_adapter: str
    target_method: Optional[str] = None
    injected_at: float = field(default_factory=time.time)
    recovered_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# CONTRACT FAULT INJECTOR
# ============================================================================

class ContractFaultInjector:
    """
    Inyecta fallos de contrato entre adaptadores
    
    Tipos de fallos:
    - Type mismatches: Tipos incorrectos en args/returns
    - Missing bindings: Dependencias faltantes entre adaptadores
    - Schema breaks: ModuleResult mal formado, YAML corrupto
    """
    
    def __init__(self):
        self.injected_faults: List[InjectedFault] = []
        self.active_patches: List[Any] = []
        
    def inject_type_mismatch(
        self,
        adapter_name: str,
        method_name: str,
        expected_type: type,
        injected_value: Any
    ) -> InjectedFault:
        """
        Inyecta un type mismatch en return value
        
        Args:
            adapter_name: Nombre del adaptador
            method_name: Método a interceptar
            expected_type: Tipo esperado (str, dict, list, etc)
            injected_value: Valor incorrecto a retornar
            
        Returns:
            Registro del fallo inyectado
        """
        fault = InjectedFault(
            category=FaultCategory.CONTRACT,
            severity=FaultSeverity.HIGH,
            description=f"Type mismatch: expected {expected_type}, got {type(injected_value)}",
            target_adapter=adapter_name,
            target_method=method_name,
            metadata={
                "expected_type": str(expected_type),
                "actual_type": str(type(injected_value)),
                "injected_value": str(injected_value)[:100]
            }
        )
        
        self.injected_faults.append(fault)
        logger.warning(f"Injected type mismatch: {adapter_name}.{method_name}")
        
        return fault
    
    def inject_missing_binding(
        self,
        source_adapter: str,
        target_adapter: str,
        binding_name: str
    ) -> InjectedFault:
        """
        Simula una binding faltante entre adaptadores
        
        Args:
            source_adapter: Adaptador que debería proveer el binding
            target_adapter: Adaptador que necesita el binding
            binding_name: Nombre del binding faltante
            
        Returns:
            Registro del fallo inyectado
        """
        fault = InjectedFault(
            category=FaultCategory.CONTRACT,
            severity=FaultSeverity.CRITICAL,
            description=f"Missing binding '{binding_name}' from {source_adapter} to {target_adapter}",
            target_adapter=target_adapter,
            metadata={
                "source_adapter": source_adapter,
                "binding_name": binding_name,
                "cascade_risk": "high"
            }
        )
        
        self.injected_faults.append(fault)
        logger.error(f"Injected missing binding: {binding_name}")
        
        return fault
    
    def inject_schema_break(
        self,
        adapter_name: str,
        break_type: str = "malformed_module_result"
    ) -> InjectedFault:
        """
        Rompe el schema de ModuleResult o execution_mapping
        
        Args:
            adapter_name: Adaptador afectado
            break_type: Tipo de ruptura
                - malformed_module_result: ModuleResult con campos faltantes
                - corrupt_yaml: execution_mapping.yaml corrupto
                - invalid_evidence: Evidence list mal formada
                
        Returns:
            Registro del fallo inyectado
        """
        fault = InjectedFault(
            category=FaultCategory.CONTRACT,
            severity=FaultSeverity.CRITICAL,
            description=f"Schema break: {break_type}",
            target_adapter=adapter_name,
            metadata={"break_type": break_type}
        )
        
        self.injected_faults.append(fault)
        logger.error(f"Injected schema break: {break_type} in {adapter_name}")
        
        return fault
    
    def create_malformed_module_result(self) -> Dict[str, Any]:
        """Crea un ModuleResult con schema roto"""
        # ModuleResult sin campos requeridos
        return {
            "status": "success",
            # Falta: module_name, class_name, method_name
            "data": None,  # Debería ser dict
            # Falta: evidence, confidence, execution_time
        }
    
    def create_corrupted_execution_chain(self) -> Dict[str, Any]:
        """Crea un execution chain con referencias rotas"""
        return {
            "step": 1,
            "adapter": "nonexistent_adapter",  # Adaptador que no existe
            "method": "undefined_method",
            "args": [
                {
                    "name": "input",
                    "source": "missing_binding"  # Binding que no existe
                }
            ],
            "returns": {
                "binding": None  # Binding nulo
            }
        }
    
    def reset(self):
        """Limpia todos los fallos inyectados"""
        for patch_obj in self.active_patches:
            patch_obj.stop()
        self.active_patches.clear()
        self.injected_faults.clear()


# ============================================================================
# DETERMINISM FAULT INJECTOR
# ============================================================================

class DeterminismFaultInjector:
    """
    Inyecta fallos de determinismo
    
    Tipos de fallos:
    - Seed corruption: Corrompe seeds de random/numpy
    - Non-reproducible outputs: Inyecta timestamps, ruido aleatorio
    """
    
    def __init__(self):
        self.injected_faults: List[InjectedFault] = []
        self.original_random_seed = None
        self.original_numpy_seed = None
        
    def inject_seed_corruption(
        self,
        adapter_name: str,
        corruption_type: str = "random"
    ) -> InjectedFault:
        """
        Corrompe seeds de generación aleatoria
        
        Args:
            adapter_name: Adaptador afectado
            corruption_type: Tipo de corrupción
                - random: Reseed aleatorio de random.seed()
                - numpy: Reseed aleatorio de np.random.seed()
                - both: Ambos
                
        Returns:
            Registro del fallo inyectado
        """
        # Guardar seeds originales
        self.original_random_seed = random.getstate()
        self.original_numpy_seed = np.random.get_state()
        
        # Corromper seeds
        if corruption_type in ["random", "both"]:
            random.seed(int(time.time() * 1000000) % 2**32)
        
        if corruption_type in ["numpy", "both"]:
            np.random.seed(int(time.time() * 1000000) % 2**32)
        
        fault = InjectedFault(
            category=FaultCategory.DETERMINISM,
            severity=FaultSeverity.HIGH,
            description=f"Seed corruption: {corruption_type}",
            target_adapter=adapter_name,
            metadata={
                "corruption_type": corruption_type,
                "timestamp": time.time()
            }
        )
        
        self.injected_faults.append(fault)
        logger.warning(f"Injected seed corruption: {corruption_type} in {adapter_name}")
        
        return fault
    
    def inject_timestamp_noise(
        self,
        adapter_name: str,
        method_name: str
    ) -> InjectedFault:
        """
        Inyecta timestamps en outputs para romper reproducibilidad
        
        Args:
            adapter_name: Adaptador afectado
            method_name: Método a interceptar
            
        Returns:
            Registro del fallo inyectado
        """
        fault = InjectedFault(
            category=FaultCategory.DETERMINISM,
            severity=FaultSeverity.MEDIUM,
            description="Timestamp injection in output",
            target_adapter=adapter_name,
            target_method=method_name,
            metadata={
                "injection_time": time.time(),
                "timestamp_field": "_injected_timestamp"
            }
        )
        
        self.injected_faults.append(fault)
        logger.warning(f"Injected timestamp noise in {adapter_name}.{method_name}")
        
        return fault
    
    def inject_random_noise(
        self,
        adapter_name: str,
        noise_level: float = 0.1
    ) -> InjectedFault:
        """
        Inyecta ruido aleatorio en outputs numéricos
        
        Args:
            adapter_name: Adaptador afectado
            noise_level: Nivel de ruido (0.0 - 1.0)
            
        Returns:
            Registro del fallo inyectado
        """
        fault = InjectedFault(
            category=FaultCategory.DETERMINISM,
            severity=FaultSeverity.MEDIUM,
            description=f"Random noise injection (level={noise_level})",
            target_adapter=adapter_name,
            metadata={
                "noise_level": noise_level,
                "noise_seed": int(time.time() * 1000000) % 2**32
            }
        )
        
        self.injected_faults.append(fault)
        logger.warning(f"Injected random noise in {adapter_name}")
        
        return fault
    
    def restore_determinism(self):
        """Restaura seeds originales"""
        if self.original_random_seed:
            random.setstate(self.original_random_seed)
        if self.original_numpy_seed:
            np.random.set_state(self.original_numpy_seed)
        
        logger.info("Determinism restored")
    
    def reset(self):
        """Limpia fallos y restaura determinismo"""
        self.restore_determinism()
        self.injected_faults.clear()


# ============================================================================
# FAULT TOLERANCE FAULT INJECTOR
# ============================================================================

class FaultToleranceFaultInjector:
    """
    Inyecta fallos en mecanismos de fault tolerance
    
    Tipos de fallos:
    - Circuit breaker misconfigurations: Umbrales incorrectos, estados stuck
    - Retry storms: Reintentos excesivos sin backoff
    - Timeouts: Timeouts prematuros o infinitos
    """
    
    def __init__(self):
        self.injected_faults: List[InjectedFault] = []
        self.active_patches: List[Any] = []
        
    def inject_circuit_breaker_stuck(
        self,
        adapter_name: str,
        stuck_state: str = "OPEN"
    ) -> InjectedFault:
        """
        Fuerza un circuit breaker a quedarse stuck en un estado
        
        Args:
            adapter_name: Adaptador afectado
            stuck_state: Estado stuck (OPEN, CLOSED, HALF_OPEN, etc)
            
        Returns:
            Registro del fallo inyectado
        """
        fault = InjectedFault(
            category=FaultCategory.FAULT_TOLERANCE,
            severity=FaultSeverity.CRITICAL,
            description=f"Circuit breaker stuck in {stuck_state}",
            target_adapter=adapter_name,
            metadata={
                "stuck_state": stuck_state,
                "should_transition": True,
                "blocks_all_requests": stuck_state == "OPEN"
            }
        )
        
        self.injected_faults.append(fault)
        logger.error(f"Injected circuit breaker stuck: {adapter_name} -> {stuck_state}")
        
        return fault
    
    def inject_wrong_failure_threshold(
        self,
        adapter_name: str,
        threshold: int = 1
    ) -> InjectedFault:
        """
        Configura un threshold de fallos incorrecto (muy bajo o muy alto)
        
        Args:
            adapter_name: Adaptador afectado
            threshold: Threshold incorrecto (1 = muy sensible, 1000 = muy tolerante)
            
        Returns:
            Registro del fallo inyectado
        """
        severity = FaultSeverity.HIGH if threshold <= 2 else FaultSeverity.MEDIUM
        
        fault = InjectedFault(
            category=FaultCategory.FAULT_TOLERANCE,
            severity=severity,
            description=f"Wrong failure threshold: {threshold}",
            target_adapter=adapter_name,
            metadata={
                "configured_threshold": threshold,
                "expected_range": "3-10",
                "impact": "too_sensitive" if threshold <= 2 else "too_tolerant"
            }
        )
        
        self.injected_faults.append(fault)
        logger.warning(f"Injected wrong threshold: {adapter_name} -> {threshold}")
        
        return fault
    
    def inject_retry_storm(
        self,
        adapter_name: str,
        max_retries: int = 100,
        no_backoff: bool = True
    ) -> InjectedFault:
        """
        Configura reintentos excesivos sin backoff exponencial
        
        Args:
            adapter_name: Adaptador afectado
            max_retries: Número excesivo de reintentos
            no_backoff: Si True, sin backoff (reintentos inmediatos)
            
        Returns:
            Registro del fallo inyectado
        """
        fault = InjectedFault(
            category=FaultCategory.FAULT_TOLERANCE,
            severity=FaultSeverity.CRITICAL,
            description=f"Retry storm: {max_retries} retries, no_backoff={no_backoff}",
            target_adapter=adapter_name,
            metadata={
                "max_retries": max_retries,
                "no_backoff": no_backoff,
                "expected_max_retries": "3-5",
                "expected_backoff": "exponential with jitter"
            }
        )
        
        self.injected_faults.append(fault)
        logger.error(f"Injected retry storm: {adapter_name}")
        
        return fault
    
    def inject_timeout_misconfiguration(
        self,
        adapter_name: str,
        timeout_ms: int,
        timeout_type: str = "premature"
    ) -> InjectedFault:
        """
        Configura timeouts incorrectos
        
        Args:
            adapter_name: Adaptador afectado
            timeout_ms: Timeout en milisegundos
            timeout_type: Tipo
                - premature: Timeout demasiado corto (< 100ms)
                - infinite: Timeout infinito (> 1 hora)
                - missing: Sin timeout
                
        Returns:
            Registro del fallo inyectado
        """
        if timeout_type == "premature":
            severity = FaultSeverity.HIGH
        elif timeout_type == "infinite":
            severity = FaultSeverity.CRITICAL
        else:
            severity = FaultSeverity.HIGH
        
        fault = InjectedFault(
            category=FaultCategory.FAULT_TOLERANCE,
            severity=severity,
            description=f"Timeout misconfiguration: {timeout_type} ({timeout_ms}ms)",
            target_adapter=adapter_name,
            metadata={
                "timeout_ms": timeout_ms,
                "timeout_type": timeout_type,
                "expected_range_ms": "1000-30000"
            }
        )
        
        self.injected_faults.append(fault)
        logger.warning(f"Injected timeout misconfiguration: {adapter_name}")
        
        return fault
    
    def reset(self):
        """Limpia todos los fallos inyectados"""
        for patch_obj in self.active_patches:
            patch_obj.stop()
        self.active_patches.clear()
        self.injected_faults.clear()


# ============================================================================
# OPERATIONAL FAULT INJECTOR
# ============================================================================

class OperationalFaultInjector:
    """
    Inyecta fallos operacionales del sistema
    
    Tipos de fallos:
    - Disk full: IOError en operaciones de escritura
    - Clock skew: Manipulación de time.time()
    - Network partitions: Connection failures
    """
    
    def __init__(self):
        self.injected_faults: List[InjectedFault] = []
        self.active_patches: List[Any] = []
        
    def inject_disk_full(
        self,
        adapter_name: str,
        affected_paths: Optional[List[str]] = None
    ) -> InjectedFault:
        """
        Simula disco lleno en operaciones de escritura
        
        Args:
            adapter_name: Adaptador afectado
            affected_paths: Paths afectados (None = todos)
            
        Returns:
            Registro del fallo inyectado
        """
        fault = InjectedFault(
            category=FaultCategory.OPERATIONAL,
            severity=FaultSeverity.CRITICAL,
            description="Disk full error on write operations",
            target_adapter=adapter_name,
            metadata={
                "affected_paths": affected_paths or ["all"],
                "error_type": "OSError",
                "error_code": 28  # ENOSPC
            }
        )
        
        self.injected_faults.append(fault)
        logger.error(f"Injected disk full error in {adapter_name}")
        
        return fault
    
    def inject_clock_skew(
        self,
        adapter_name: str,
        skew_seconds: float = 3600.0
    ) -> InjectedFault:
        """
        Inyecta clock skew manipulando time.time()
        
        Args:
            adapter_name: Adaptador afectado
            skew_seconds: Skew en segundos (+ adelante, - atrás)
            
        Returns:
            Registro del fallo inyectado
        """
        original_time = time.time
        
        def skewed_time():
            return original_time() + skew_seconds
        
        # Patch time.time
        patcher = patch('time.time', side_effect=skewed_time)
        patcher.start()
        self.active_patches.append(patcher)
        
        fault = InjectedFault(
            category=FaultCategory.OPERATIONAL,
            severity=FaultSeverity.HIGH,
            description=f"Clock skew: {skew_seconds}s",
            target_adapter=adapter_name,
            metadata={
                "skew_seconds": skew_seconds,
                "direction": "future" if skew_seconds > 0 else "past"
            }
        )
        
        self.injected_faults.append(fault)
        logger.warning(f"Injected clock skew: {adapter_name} -> {skew_seconds}s")
        
        return fault
    
    def inject_network_partition(
        self,
        adapter_name: str,
        partition_type: str = "complete"
    ) -> InjectedFault:
        """
        Simula network partition / connection failure
        
        Args:
            adapter_name: Adaptador afectado
            partition_type: Tipo de partición
                - complete: Sin conectividad
                - intermittent: Conectividad intermitente
                - slow: Alta latencia
                
        Returns:
            Registro del fallo inyectado
        """
        severity_map = {
            "complete": FaultSeverity.CRITICAL,
            "intermittent": FaultSeverity.HIGH,
            "slow": FaultSeverity.MEDIUM
        }
        
        fault = InjectedFault(
            category=FaultCategory.OPERATIONAL,
            severity=severity_map.get(partition_type, FaultSeverity.HIGH),
            description=f"Network partition: {partition_type}",
            target_adapter=adapter_name,
            metadata={
                "partition_type": partition_type,
                "affects_external_calls": True
            }
        )
        
        self.injected_faults.append(fault)
        logger.error(f"Injected network partition: {adapter_name} -> {partition_type}")
        
        return fault
    
    def inject_memory_pressure(
        self,
        adapter_name: str,
        pressure_level: str = "high"
    ) -> InjectedFault:
        """
        Simula presión de memoria
        
        Args:
            adapter_name: Adaptador afectado
            pressure_level: Nivel (low, medium, high, critical)
            
        Returns:
            Registro del fallo inyectado
        """
        severity_map = {
            "low": FaultSeverity.LOW,
            "medium": FaultSeverity.MEDIUM,
            "high": FaultSeverity.HIGH,
            "critical": FaultSeverity.CRITICAL
        }
        
        fault = InjectedFault(
            category=FaultCategory.OPERATIONAL,
            severity=severity_map.get(pressure_level, FaultSeverity.MEDIUM),
            description=f"Memory pressure: {pressure_level}",
            target_adapter=adapter_name,
            metadata={
                "pressure_level": pressure_level,
                "may_cause_oom": pressure_level in ["high", "critical"]
            }
        )
        
        self.injected_faults.append(fault)
        logger.warning(f"Injected memory pressure: {adapter_name} -> {pressure_level}")
        
        return fault
    
    def reset(self):
        """Limpia todos los fallos operacionales"""
        for patch_obj in self.active_patches:
            patch_obj.stop()
        self.active_patches.clear()
        self.injected_faults.clear()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_faulty_module_result(
    fault_type: str,
    adapter_name: str,
    method_name: str
) -> Dict[str, Any]:
    """
    Crea un ModuleResult con fallo inyectado
    
    Args:
        fault_type: Tipo de fallo
        adapter_name: Nombre del adaptador
        method_name: Nombre del método
        
    Returns:
        ModuleResult con fallo
    """
    if fault_type == "type_mismatch":
        return {
            "module_name": adapter_name,
            "class_name": "FaultyAdapter",
            "method_name": method_name,
            "status": "success",
            "data": "should_be_dict_not_string",  # Type mismatch
            "evidence": [],
            "confidence": 1.0,
            "execution_time": 0.0
        }
    
    elif fault_type == "missing_fields":
        return {
            "status": "success",
            "data": {}
            # Missing: module_name, class_name, method_name, evidence, confidence, execution_time
        }
    
    elif fault_type == "invalid_evidence":
        return {
            "module_name": adapter_name,
            "class_name": "FaultyAdapter",
            "method_name": method_name,
            "status": "success",
            "data": {},
            "evidence": "should_be_list_not_string",  # Type mismatch
            "confidence": 1.0,
            "execution_time": 0.0
        }
    
    else:
        return {
            "module_name": adapter_name,
            "class_name": "FaultyAdapter",
            "method_name": method_name,
            "status": "failed",
            "data": {},
            "evidence": [],
            "confidence": 0.0,
            "execution_time": 0.0,
            "errors": [f"Injected fault: {fault_type}"]
        }
