"""
FARFAN 3.0 - Policy Analysis Orchestrator
==========================================
The world's first production-grade causal mechanism orchestrator
for development plan analysis.

Architecture:
- Orchestrator: Question routing and execution coordination
- Choreographer: Module sequencing and dependency management
- Circuit Breaker: Fault tolerance and graceful degradation
- Report Assembly: MICRO/MESO/MACRO multi-level reporting

Author: FARFAN Team
Version: 3.0.0
"""

__version__ = "3.0.0"
__author__ = "FARFAN Policy Analysis Team"

from .core_orchestrator import FARFANOrchestrator
from .question_router import QuestionRouter
from .choreographer import ExecutionChoreographer
from .circuit_breaker import CircuitBreaker
from .report_assembly import ReportAssembler

__all__ = [
    "FARFANOrchestrator",
    "QuestionRouter",
    "ExecutionChoreographer",
    "CircuitBreaker",
    "ReportAssembler"
]
