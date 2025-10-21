"""
Orchestrator Package
====================

Core orchestration components for the FARFAN pipeline.

Components:
- QuestionRouter: Routes questions to appropriate modules
- ExecutionChoreographer: Coordinates module execution
- CircuitBreaker: Handles fault tolerance
- ReportAssembler: Assembles final reports
- YAMLMappingLoader: Loads execution mappings
- AdapterRegistry: Legacy adapter registry for module instances
- ModuleAdapterRegistry: Canonical adapter registry with execution contract
- ModuleMethodResult: Result dataclass for adapter method execution
- ContractViolation: Exception for contract violations
"""

# Lazy imports to avoid loading heavy dependencies at package import time
__all__ = [
    "QuestionRouter",
    "ExecutionChoreographer",
    "CircuitBreaker",
    "ReportAssembler",
    "YAMLMappingLoader",
    "AdapterRegistry",
    "ModuleAdapterRegistry",
    "ModuleMethodResult",
    "ContractViolation",
    "ExecutionStatus",
    "AdapterAvailabilitySnapshot",
]


def __getattr__(name):
    """Lazy import to avoid loading dependencies at package import"""
    if name == "QuestionRouter":
        from src.orchestrator.question_router import QuestionRouter
        return QuestionRouter
    elif name == "ExecutionChoreographer":
        from src.orchestrator.choreographer import ExecutionChoreographer
        return ExecutionChoreographer
    elif name == "CircuitBreaker":
        from src.orchestrator.circuit_breaker import CircuitBreaker
        return CircuitBreaker
    elif name == "ReportAssembler":
        from src.orchestrator.report_assembly import ReportAssembler
        return ReportAssembler
    elif name == "YAMLMappingLoader":
        from src.orchestrator.mapping_loader import YAMLMappingLoader
        return YAMLMappingLoader
    elif name == "AdapterRegistry":
        from src.orchestrator.module_adapters import AdapterRegistry
        return AdapterRegistry
    elif name == "ModuleAdapterRegistry":
        from src.orchestrator.adapter_registry import ModuleAdapterRegistry
        return ModuleAdapterRegistry
    elif name == "ModuleMethodResult":
        from src.orchestrator.adapter_registry import ModuleMethodResult
        return ModuleMethodResult
    elif name == "ContractViolation":
        from src.orchestrator.adapter_registry import ContractViolation
        return ContractViolation
    elif name == "ExecutionStatus":
        from src.orchestrator.adapter_registry import ExecutionStatus
        return ExecutionStatus
    elif name == "AdapterAvailabilitySnapshot":
        from src.orchestrator.adapter_registry import AdapterAvailabilitySnapshot
        return AdapterAvailabilitySnapshot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
