"""
Orchestrator Package
====================

Core orchestration components for the FARFAN pipeline.

Components:
- QuestionRouter: Routes questions to appropriate modules
- Choreographer: Coordinates module execution
- CircuitBreaker: Handles fault tolerance
- ReportAssembly: Assembles final reports
- MappingLoader: Loads execution mappings
- ModuleAdapters: Adapter registry for module instances
"""

from src.orchestrator.question_router import QuestionRouter
from src.orchestrator.choreographer import ExecutionChoreographer
from src.orchestrator.circuit_breaker import CircuitBreaker
from src.orchestrator.report_assembly import ReportAssembler
from src.orchestrator.mapping_loader import YAMLMappingLoader
from src.orchestrator.module_adapters import AdapterRegistry

__all__ = [
    "QuestionRouter",
    "ExecutionChoreographer",
    "CircuitBreaker",
    "ReportAssembler",
    "YAMLMappingLoader",
    "AdapterRegistry",
]
