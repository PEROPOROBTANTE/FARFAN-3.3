"""
FARFAN 3.0 - Policy Analysis Orchestrator
==========================================
The world's first production-grade causal mechanism orchestrator
for development plan analysis.

Architecture:
- Orchestrator: Question routing and execution coordination
- ModuleController: Unified adapter interface with responsibility mapping
- Choreographer: Module sequencing and dependency management
- Circuit Breaker: Fault tolerance and graceful degradation
- Report Assembly: MICRO/MESO/MACRO multi-level reporting
- Mapping Loader: Execution integrity validation layer

Author: FARFAN Team
Version: 3.0.0
"""

__version__ = "3.0.0"
__author__ = "FARFAN Policy Analysis Team"

# Lazy imports to avoid heavyweight dependencies at module load time
__all__ = [
    "FARFANOrchestrator",
    "ModuleController",
    "ExecutionChoreographer",
    "CircuitBreaker",
    "ReportAssembler",
    "YAMLMappingLoader",
    "MappingStartupValidator",
    "MappingValidationError",
    "MappingConflict",
    "ConflictType",
    "ContractRegistry",
]

def __getattr__(name):
    """Lazy load components to avoid import overhead"""
    if name == "FARFANOrchestrator":
        from .core_orchestrator import FARFANOrchestrator
        return FARFANOrchestrator
    elif name == "ModuleController":
        from .module_controller import ModuleController
        return ModuleController
    elif name == "ExecutionChoreographer":
        from .choreographer import ExecutionChoreographer
        return ExecutionChoreographer
    elif name == "CircuitBreaker":
        from .circuit_breaker import CircuitBreaker
        return CircuitBreaker
    elif name == "ReportAssembler":
        from .report_assembly import ReportAssembler
        return ReportAssembler
    elif name in ["YAMLMappingLoader", "MappingStartupValidator", "MappingValidationError",
                  "MappingConflict", "ConflictType", "ContractRegistry"]:
        from .mapping_loader import (
            YAMLMappingLoader,
            MappingStartupValidator,
            MappingValidationError,
            MappingConflict,
            ConflictType,
            ContractRegistry
        )
        if name == "YAMLMappingLoader":
            return YAMLMappingLoader
        elif name == "MappingStartupValidator":
            return MappingStartupValidator
        elif name == "MappingValidationError":
            return MappingValidationError
        elif name == "MappingConflict":
            return MappingConflict
        elif name == "ConflictType":
            return ConflictType
        elif name == "ContractRegistry":
            return ContractRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
