"""
Fault Injection Testing Framework for FARFAN 3.0
================================================

Framework para pruebas de inyección de fallos que valida:
- Contract integrity (type mismatches, missing bindings, schema breaks)
- Determinism (seed corruption, non-reproducible outputs)
- Fault tolerance (circuit breaker, retry storms, timeouts)
- Operational resilience (disk full, clock skew, network partitions)

Author: FARFAN Integration Team
Version: 1.0.0
Python: 3.10+
"""

from .injectors import (
    ContractFaultInjector,
    DeterminismFaultInjector,
    FaultToleranceFaultInjector,
    OperationalFaultInjector
)
from .resilience_validator import ResilienceValidator
from .chaos_scenarios import ChaosScenarioRunner

__all__ = [
    'ContractFaultInjector',
    'DeterminismFaultInjector',
    'FaultToleranceFaultInjector',
    'OperationalFaultInjector',
    'ResilienceValidator',
    'ChaosScenarioRunner'
]

__version__ = '1.0.0'
