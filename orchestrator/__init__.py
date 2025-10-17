"""
FARFAN 3.0 - Policy Analysis Orchestrator
==========================================
The world's first production-grade causal mechanism orchestrator
for development plan analysis.

Architecture:
- Orchestrator: Question routing and execution coordination
- Choreographer: Module sequencing and dependency management
- QuestionnaireParser: Canonical source for questionnaire data
- Report Assembly: MICRO/MESO/MACRO multi-level reporting

Author: FARFAN Team
Version: 3.0.0
"""

__version__ = "3.0.0"
__author__ = "FARFAN Policy Analysis Team"

# from .core_orchestrator import FARFANOrchestrator  # TODO: Fix circuit_breaker dependency
from .question_router import QuestionRouter
from .choreographer import ExecutionChoreographer
# from .circuit_breaker import CircuitBreaker  # TODO: Fix - currently has report_assembly content
from .report_assembly import ReportAssembler
from .questionnaire_parser import QuestionnaireParser, get_questionnaire_parser

__all__ = [
    # "FARFANOrchestrator",  # TODO: Re-enable after circuit_breaker fix
    "QuestionRouter",
    "ExecutionChoreographer",
    # "CircuitBreaker",  # TODO: Re-enable after fix
    "ReportAssembler",
    "QuestionnaireParser",
    "get_questionnaire_parser"
]
