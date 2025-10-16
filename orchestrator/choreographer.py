"""
Execution Choreographer - Orchestrates module execution with dependency management
Implements hybrid parallel/sequential execution strategy
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx

from .config import CONFIG
from .question_router import QuestionRouter

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Module execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionResult:
    """Result from a module execution"""
    module_name: str
    status: ExecutionStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    evidence_extracted: Dict[str, Any] = field(default_factory=dict)


class ExecutionChoreographer:
    """
    Choreographs the execution of multiple modules with dependency management.

    Strategy:
    - Priority 1 modules (extractors/segmenters) run in parallel
    - Priority 2+ modules (analyzers) run after dependencies complete
    - Uses DAG (Directed Acyclic Graph) for dependency tracking
    """

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or CONFIG.max_parallel_workers
        self.router = QuestionRouter()
        self.execution_graph = nx.DiGraph()
        self._build_dependency_graph()

    def _build_dependency_graph(self):
        """
        Build execution dependency graph.

        Dependencies:
        - policy_segmenter must run first (segments text)
        - embedding_policy needs segmented text
        - causal_processor can run in parallel with policy_processor
        - dereck_beach needs causal_processor output
        - contradiction_detector needs policy_processor output
        - financial_viability can run independently
        - analyzer_one can run after segmentation
        """

        # Add all modules as nodes
        for module_name in CONFIG.modules.keys():
            self.execution_graph.add_node(
                module_name,
                priority=CONFIG.modules[module_name].priority
            )

        # Define dependencies (edges)
        dependencies = [
            # Segmenter is the foundation - everything depends on it
            ("policy_segmenter", "embedding_policy"),
            ("policy_segmenter", "policy_processor"),
            ("policy_segmenter", "causal_processor"),
            ("policy_segmenter", "analyzer_one"),

            # Embedding depends on segmentation
            ("embedding_policy", "contradiction_detector"),

            # Causal processor feeds Derek Beach
            ("causal_processor", "dereck_beach"),

            # Policy processor feeds contradiction detector
            ("policy_processor", "contradiction_detector"),
        ]

        for source, target in dependencies:
            self.execution_graph.add_edge(source, target)

        logger.info(f"Built dependency graph with {self.execution_graph.number_of_nodes()} nodes "
                    f"and {self.execution_graph.number_of_edges()} edges")

    def execute_for_question(
            self,
            question_id: str,
            plan_text: str,
            plan_metadata: Dict[str, Any]
    ) -> Dict[str, ExecutionResult]:
        """
        Execute all required modules for a specific question.

        Args:
            question_id: P#-D#-Q# notation
            plan_text: Full development plan text
            plan_metadata: Plan metadata (name, year, etc)

        Returns:
            Dict mapping module_name to ExecutionResult
        """
        logger.info(f"Choreographing execution for question: {question_id}")

        # Get required modules and their execution order
        required_modules = set(self.router.get_modules_for_question(question_id))

        if not required_modules:
            logger.warning(f"No modules required for {question_id}")
            return {}

        # Build subgraph for required modules only
        subgraph = self.execution_graph.subgraph(required_modules)

        # Get topological order (respects dependencies)
        try:
            execution_order = list(nx.topological_sort(subgraph))
        except nx.NetworkXError as e:
            logger.error(f"Circular dependency detected: {e}")
            # Fallback to priority-based ordering
            execution_order = sorted(
                required_modules,
                key=lambda m: CONFIG.modules[m].priority
            )

        logger.info(f"Execution order: {' -> '.join(execution_order)}")

        # Execute in waves (all modules with same priority run in parallel)
        results = {}
        execution_context = {
            "plan_text": plan_text,
            "plan_metadata": plan_metadata,
            "question_id": question_id
        }

        # Group modules by priority level
        priority_waves = self._group_by_priority(execution_order)

        for wave_num, wave_modules in enumerate(priority_waves, start=1):
            logger.info(f"Executing wave {wave_num}: {wave_modules}")

            # Execute this wave in parallel
            wave_results = self._execute_wave(
                wave_modules,
                execution_context,
                results
            )

            results.update(wave_results)

            # Update context with results from this wave
            execution_context["previous_results"] = results

        return results

    def _group_by_priority(self, module_order: List[str]) -> List[List[str]]:
        """Group modules into waves by priority level"""
        priority_map = {}

        for module in module_order:
            priority = CONFIG.modules[module].priority
            if priority not in priority_map:
                priority_map[priority] = []
            priority_map[priority].append(module)

        # Return waves in priority order
        return [priority_map[p] for p in sorted(priority_map.keys())]

    def _execute_wave(
            self,
            modules: List[str],
            context: Dict[str, Any],
            previous_results: Dict[str, ExecutionResult]
    ) -> Dict[str, ExecutionResult]:
        """
        Execute a wave of modules in parallel.

        Args:
            modules: List of module names to execute
            context: Shared execution context
            previous_results: Results from previous waves

        Returns:
            Dict of module_name to ExecutionResult
        """
        wave_results = {}

        if len(modules) == 1:
            # Single module - execute directly
            result = self._execute_module(modules[0], context, previous_results)
            wave_results[modules[0]] = result
        else:
            # Multiple modules - execute in parallel
            with ThreadPoolExecutor(max_workers=min(len(modules), self.max_workers)) as executor:
                # Submit all tasks
                future_to_module = {
                    executor.submit(
                        self._execute_module,
                        module,
                        context,
                        previous_results
                    ): module
                    for module in modules
                }

                # Collect results as they complete
                for future in as_completed(future_to_module):
                    module = future_to_module[future]
                    try:
                        result = future.result(timeout=CONFIG.modules[module].timeout_seconds)
                        wave_results[module] = result
                    except Exception as e:
                        logger.error(f"Module {module} failed: {e}")
                        wave_results[module] = ExecutionResult(
                            module_name=module,
                            status=ExecutionStatus.FAILED,
                            error=str(e)
                        )

        return wave_results

    def _execute_module(
            self,
            module_name: str,
            context: Dict[str, Any],
            previous_results: Dict[str, ExecutionResult]
    ) -> ExecutionResult:
        """
        Execute a single module - REAL IMPLEMENTATION

        This calls the ACTUAL module through ModuleAdapterRegistry
        NO MORE PLACEHOLDERS
        """
        start_time = time.time()
        module_config = CONFIG.modules[module_name]

        logger.info(f"Executing module: {module_name}")

        try:
            # Import adapter registry (REAL execution)
            from .module_adapters import ModuleAdapterRegistry

            # Initialize registry
            registry = ModuleAdapterRegistry()

            # Prepare arguments based on module type
            plan_text = context.get("plan_text", "")
            plan_metadata = context.get("plan_metadata", {})

            # Map module_name to method_name
            method_mapping = {
                "policy_processor": "process",
                "analyzer_one": "analyze_document",
                "contradiction_detector": "detect",
                "dereck_beach": "process_document",
                "embedding_policy": "chunk_text",
                "financial_viability": "analyze_financial",
                "causal_processor": "analyze",
                "policy_segmenter": "segment"
            }

            method_name = method_mapping.get(module_name, "process")

            # Execute REAL module method
            result = registry.execute_module_method(
                module_name=module_name,
                method_name=method_name,
                args=[plan_text],
                kwargs={"plan_metadata": plan_metadata}
            )

            # Convert ModuleResult to ExecutionResult
            status = ExecutionStatus.COMPLETED if result.status == "success" else ExecutionStatus.FAILED

            return ExecutionResult(
                module_name=module_name,
                status=status,
                output=result.data,
                execution_time=result.execution_time,
                evidence_extracted={"evidence": result.evidence, "confidence": result.confidence},
                error=result.errors[0] if result.errors else None
            )

        except Exception as e:
            logger.exception(f"Module {module_name} execution failed")
            return ExecutionResult(
                module_name=module_name,
                status=ExecutionStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _extract_evidence_from_output(
            self,
            module_name: str,
            output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract structured evidence from module output.

        This maps module-specific output to canonical evidence format:
        {
            "quantitative_claims": [...],
            "causal_links": [...],
            "contradictions": [...],
            "confidence_scores": {...}
        }
        """
        evidence = {
            "quantitative_claims": [],
            "causal_links": [],
            "contradictions": [],
            "confidence_scores": {},
            "raw_output": output
        }

        # Module-specific evidence extraction logic
        if module_name == "contradiction_detector":
            evidence["contradictions"] = output.get("contradictions", [])
            evidence["confidence_scores"]["coherence"] = output.get("coherence_score", 0.0)

        elif module_name == "causal_processor":
            evidence["causal_links"] = output.get("causal_dimensions", {})
            evidence["confidence_scores"]["causal_strength"] = output.get("information_gain", 0.0)

        elif module_name == "dereck_beach":
            evidence["causal_links"] = output.get("mechanism_parts", [])
            evidence["confidence_scores"]["mechanism_confidence"] = output.get("rigor_status", 0.0)

        elif module_name == "policy_processor":
            evidence["quantitative_claims"] = output.get("dimension_analysis", {})

        elif module_name == "financial_viability":
            evidence["quantitative_claims"] = output.get("budget_analysis", {})
            evidence["confidence_scores"]["financial_coherence"] = output.get("viability_score", 0.0)

        return evidence

    def get_execution_statistics(self, results: Dict[str, ExecutionResult]) -> Dict[str, Any]:
        """Generate statistics from execution results"""
        total_modules = len(results)
        successful = sum(1 for r in results.values() if r.status == ExecutionStatus.COMPLETED)
        failed = sum(1 for r in results.values() if r.status == ExecutionStatus.FAILED)
        total_time = sum(r.execution_time for r in results.values())

        return {
            "total_modules": total_modules,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_modules if total_modules > 0 else 0.0,
            "total_execution_time": total_time,
            "avg_execution_time": total_time / total_modules if total_modules > 0 else 0.0,
            "module_times": {
                name: result.execution_time
                for name, result in results.items()
            }
        }
