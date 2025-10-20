# core_orchestrator.py - Updated to match new module_adapters.py structure
"""
Core Orchestrator - The main coordination engine
Integrates Router, Choreographer, Circuit Breaker, and Report Assembly
"""
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime

from .config import CONFIG
from .question_router import QuestionRouter
from .choreographer import ExecutionChoreographer, ExecutionResult, ExecutionStatus
from .circuit_breaker import CircuitBreaker, create_module_specific_fallback
from .report_assembly import (
    ReportAssembler,
    MicroLevelAnswer,
    MesoLevelCluster,
    MacroLevelConvergence
)

logger = logging.getLogger(__name__)


class FARFANOrchestrator:
    """
    Main orchestrator for FARFAN 3.0 policy analysis system.

    Coordinates:
    - Question routing (300 questions → components)
    - Module execution with dependency management
    - Fault tolerance and graceful degradation
    - Multi-level report generation (MICRO/MESO/MACRO)
    """

    def __init__(self):
        logger.info("Initializing FARFAN Orchestrator")

        self.router = QuestionRouter()
        self.choreographer = ExecutionChoreographer()
        self.circuit_breaker = CircuitBreaker()

        # Initialize ReportAssembler with dimension descriptions from the router
        dimension_descriptions = self.router.get_dimension_descriptions()
        self.report_assembler = ReportAssembler(dimension_descriptions=dimension_descriptions)

        self.execution_stats = {
            "total_plans_processed": 0,
            "total_questions_answered": 0,
            "total_components_executed": 0,
            "total_execution_time": 0.0,
            "module_performance": defaultdict(lambda: {"calls": 0, "successes": 0, "failures": 0}),
            "component_performance": defaultdict(lambda: {"calls": 0, "successes": 0, "failures": 0})
        }

        logger.info("FARFAN Orchestrator initialized successfully")

    def analyze_single_plan(
            self,
            plan_path: Path,
            plan_name: Optional[str] = None,
            output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single development plan through the complete pipeline.

        Args:
            plan_path: Path to the plan document (PDF/TXT)
            plan_name: Optional name for the plan (defaults to filename)
            output_dir: Optional output directory (defaults to CONFIG.output_dir)

        Returns:
            Dict with full analysis results
        """
        start_time = time.time()

        plan_name = plan_name or plan_path.stem
        output_dir = output_dir or CONFIG.output_dir / plan_name

        output_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"