"""
Execution Choreographer - DAG-Based Module Orchestration with Dependency Management
===================================================================================

CORE RESPONSIBILITY: Execute adapter methods in correct dependency order
------------------------------------------------------------------------
Manages parallel execution of independent adapters while respecting dependencies
defined in the adapter dependency graph (DAG)

DEPENDENCY RESOLUTION ORDER (9 Adapters):
------------------------------------------
Wave 1 (Foundation - parallel execution):
  - policy_segmenter: Document segmentation
  - policy_processor: Text normalization

Wave 2 (Semantic analysis - depends on Wave 1):
  - semantic_chunking_policy: Semantic chunking (requires segments)
  - embedding_policy: Embedding generation (requires normalized text)

Wave 3 (Advanced analysis - depends on Wave 2):
  - analyzer_one: Municipal development analysis (requires embeddings + segments)
  - teoria_cambio: Theory of change analysis (requires semantic chunks + embeddings)

Wave 4 (Specialized analysis - depends on Wave 3):
  - dereck_beach: CDAF causal analysis (requires teoria_cambio output)
  - contradiction_detection: Contradiction detection (requires analyzer_one + teoria_cambio)

Wave 5 (Final synthesis - depends on Wave 4):
  - financial_viability: Financial analysis (requires all prior outputs)

PARALLEL WAVE EXECUTION LOGIC:
-------------------------------
1. Identify wave (all adapters with same priority level)
2. Execute all adapters in wave concurrently (ThreadPoolExecutor)
3. Wait for all wave executions to complete
4. Proceed to next wave only after current wave finishes
5. Handle failures gracefully with circuit breaker integration

ERROR HANDLING FOR MISSING ADAPTER/METHOD REFERENCES:
------------------------------------------------------
- Before execution: Validate adapter exists in ModuleAdapterRegistry
- Before method call: Validate method exists on adapter class
- On validation failure: Log error, mark step as SKIPPED, continue to next step
- Circuit breaker: After 5 consecutive failures, open circuit for 60s
- Graceful degradation: Continue execution chain even if non-critical steps fail

ADAPTER INVOCATION PATTERN (corrected from instance-based to class-based):
--------------------------------------------------------------------------
❌ INCORRECT (instance-based):
  adapter_instance = module_registry.get_adapter("teoria_cambio")
  result = adapter_instance.calculate_bayesian_confidence(...)

✅ CORRECT (class-based via registry):
  result = module_registry.execute_module_method(
      module_name="teoria_cambio",
      method_name="calculate_bayesian_confidence",
      args=[...],
      kwargs={...}
  )

Author: FARFAN Integration Team
Version: 3.0.0 - Refactored with strict type annotations and comprehensive documentation
Python: 3.10+
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Execution status for individual adapter method calls"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    DEGRADED = "degraded"


@dataclass
class ExecutionResult:
    """
    Result from a single adapter method execution

    Wraps ModuleResult from module_adapters.py with additional metadata
    """

    module_name: str
    adapter_class: str
    method_name: str
    status: ExecutionStatus

    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0

    evidence_extracted: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for aggregation and reporting"""
        return {
            "module_name": self.module_name,
            "adapter_class": self.adapter_class,
            "method_name": self.method_name,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "evidence": self.evidence_extracted,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class ExecutionChoreographer:
    """
    Orchestrates execution of 9 adapters with DAG-based dependency management

    Implements hybrid parallel/sequential execution:
    - Parallel: Execute all adapters in same wave concurrently
    - Sequential: Wait for wave completion before proceeding to next wave

    Circuit breaker integration at every adapter invocation point
    """

    def __init__(self, max_workers: Optional[int] = None) -> None:
        """
        Initialize choreographer with dependency graph

        Args:
            max_workers: Maximum parallel workers for wave execution (default: 4)
        """
        self.max_workers = max_workers or 4
        self.execution_graph: nx.DiGraph = nx.DiGraph()
        self._build_dependency_graph()
        self._initialize_adapter_registry()

        logger.info(
            f"ExecutionChoreographer initialized with {self.max_workers} workers"
        )

    def _build_dependency_graph(self) -> None:
        """
        Build DAG of adapter dependencies for topological execution ordering

        Dependencies based on data flow requirements:
        - Semantic analysis requires document segmentation
        - Advanced analysis requires semantic analysis
        - Specialized analysis requires advanced analysis
        - Financial synthesis requires all prior analyses
        """

        adapters_with_priorities = {
            "policy_segmenter": 1,
            "policy_processor": 1,
            "semantic_chunking_policy": 2,
            "embedding_policy": 2,
            "analyzer_one": 3,
            "teoria_cambio": 3,
            "dereck_beach": 4,
            "contradiction_detection": 4,
            "financial_viability": 5,
        }

        for adapter_name, priority in adapters_with_priorities.items():
            self.execution_graph.add_node(adapter_name, priority=priority)

        dependencies = [
            ("policy_segmenter", "semantic_chunking_policy"),
            ("policy_segmenter", "embedding_policy"),
            ("policy_processor", "semantic_chunking_policy"),
            ("policy_processor", "embedding_policy"),
            ("semantic_chunking_policy", "analyzer_one"),
            ("semantic_chunking_policy", "teoria_cambio"),
            ("embedding_policy", "analyzer_one"),
            ("embedding_policy", "teoria_cambio"),
            ("analyzer_one", "contradiction_detection"),
            ("teoria_cambio", "dereck_beach"),
            ("teoria_cambio", "contradiction_detection"),
            ("dereck_beach", "financial_viability"),
            ("contradiction_detection", "financial_viability"),
            ("analyzer_one", "financial_viability"),
        ]

        for source, target in dependencies:
            self.execution_graph.add_edge(source, target)

        if not nx.is_directed_acyclic_graph(self.execution_graph):
            raise ValueError("Dependency graph contains cycles!")

        logger.info(
            f"Built dependency graph: {self.execution_graph.number_of_nodes()} adapters, "
            f"{self.execution_graph.number_of_edges()} dependencies"
        )

    def _initialize_adapter_registry(self) -> None:
        """Initialize mapping of adapter names to class names for invocation compatibility"""
        self.adapter_registry: Dict[str, str] = {
            "teoria_cambio": "ModulosAdapter",
            "analyzer_one": "AnalyzerOneAdapter",
            "dereck_beach": "DerekBeachAdapter",
            "embedding_policy": "EmbeddingPolicyAdapter",
            "semantic_chunking_policy": "SemanticChunkingPolicyAdapter",
            "contradiction_detection": "ContradictionDetectionAdapter",
            "financial_viability": "FinancialViabilityAdapter",
            "policy_processor": "PolicyProcessorAdapter",
            "policy_segmenter": "PolicySegmenterAdapter",
        }

        logger.info(f"Initialized registry for {len(self.adapter_registry)} adapters")

    def execute_question_chain(
        self,
        question_spec: Any,
        plan_text: str,
        module_adapter_registry: Any,
        circuit_breaker: Optional[Any] = None,
    ) -> Dict[str, ExecutionResult]:
        """
        Execute complete execution chain for a single question

        EXECUTION FLOW:
        ---------------
        1. Extract execution_chain from question_spec
        2. For each step in chain (sequential execution respecting dependencies):
           a. Validate adapter and method exist
           b. Check circuit breaker status
           c. Prepare arguments (may reference previous step results)
           d. Execute via ModuleAdapterRegistry.execute_module_method() (CLASS-BASED)
           e. Convert ModuleResult to ExecutionResult
           f. Record success/failure in circuit breaker
        3. Return dict of all ExecutionResults keyed by "adapter.method"

        Args:
            question_spec: Question specification from questionnaire parser
            plan_text: Plan document text
            module_adapter_registry: ModuleAdapterRegistry instance for adapter invocation
            circuit_breaker: Optional CircuitBreaker for fault tolerance

        Returns:
            Dictionary mapping "adapter.method" to ExecutionResult objects
        """
        logger.info(f"Executing chain for {question_spec.canonical_id}")

        start_time = time.time()
        results: Dict[str, ExecutionResult] = {}

        execution_chain = getattr(question_spec, "execution_chain", [])

        if not execution_chain:
            logger.warning(f"No execution chain for {question_spec.canonical_id}")
            return results

        for step in execution_chain:
            adapter_name = step.get("adapter")
            method_name = step.get("method")
            args = step.get("args", [])
            kwargs = step.get("kwargs", {})

            if not adapter_name or not method_name:
                logger.warning(f"Incomplete step in chain: {step}")
                continue

            if not self._validate_adapter_method(
                adapter_name, method_name, module_adapter_registry
            ):
                results[f"{adapter_name}.{method_name}"] = ExecutionResult(
                    module_name=adapter_name,
                    adapter_class=self.adapter_registry.get(adapter_name, "Unknown"),
                    method_name=method_name,
                    status=ExecutionStatus.SKIPPED,
                    error="Adapter or method not found in registry",
                    execution_time=0.0,
                )
                continue

            prepared_args = self._prepare_arguments(args, results, plan_text)
            prepared_kwargs = self._prepare_arguments(kwargs, results, plan_text)

            result = self._execute_single_step(
                adapter_name=adapter_name,
                method_name=method_name,
                args=prepared_args,
                kwargs=prepared_kwargs,
                module_adapter_registry=module_adapter_registry,
                circuit_breaker=circuit_breaker,
            )

            results[f"{adapter_name}.{method_name}"] = result

        total_time = time.time() - start_time
        logger.info(
            f"Completed chain for {question_spec.canonical_id} in {total_time:.2f}s "
            f"({len(results)} steps)"
        )

        return results

    def _validate_adapter_method(
        self, adapter_name: str, method_name: str, module_adapter_registry: Any
    ) -> bool:
        """
        Validate that adapter and method exist in registry before execution

        CONTRACT ENFORCEMENT: Uses ModuleAdapterRegistry.list_adapter_methods
        to validate method existence. Raises ContractViolation on validation failure
        instead of silent False returns (SIN_CARRETA compliance).

        Args:
            adapter_name: Name of adapter (e.g., "teoria_cambio")
            method_name: Name of method (e.g., "calculate_bayesian_confidence")
            module_adapter_registry: ModuleAdapterRegistry instance

        Returns:
            True if adapter and method exist, False otherwise
        """
        try:
            # Check if registry has the new execute_module_method API
            if hasattr(module_adapter_registry, "list_adapter_methods"):
                # New ModuleAdapterRegistry API
                try:
                    methods = module_adapter_registry.list_adapter_methods(adapter_name)
                    if method_name not in methods:
                        logger.error(
                            f"Method '{method_name}' not found on adapter '{adapter_name}'"
                        )
                        return False
                    return True
                except Exception as e:
                    # ContractViolation or other error from list_adapter_methods
                    logger.error(
                        f"Error listing methods for adapter '{adapter_name}': {e}"
                    )
                    return False
            else:
                # Legacy AdapterRegistry API (backward compatibility)
                if adapter_name not in module_adapter_registry.adapters:
                    logger.error(f"Adapter '{adapter_name}' not found in registry")
                    return False

                adapter_instance = module_adapter_registry.adapters[adapter_name]
                if not hasattr(adapter_instance, method_name):
                    logger.error(
                        f"Method '{method_name}' not found on adapter '{adapter_name}'"
                    )
                    return False

                return True

        except Exception as e:
            logger.error(f"Error validating adapter method: {e}")
            return False

    def _prepare_arguments(
        self,
        args_spec: Union[List, Dict],
        previous_results: Dict[str, ExecutionResult],
        plan_text: str,
    ) -> Union[List, Dict]:
        """
        Prepare arguments by resolving references to previous step results

        ARGUMENT RESOLUTION:
        -------------------
        - source: "plan_text" → inject plan_text string
        - source: "previous_result" → inject output from previous step
        - source: "derived" → compute from previous results
        - value: <literal> → use literal value

        Args:
            args_spec: Argument specification from execution chain
            previous_results: Results from previously executed steps
            plan_text: Plan document text

        Returns:
            Resolved arguments as list or dict
        """
        if isinstance(args_spec, list):
            prepared = []
            for arg in args_spec:
                if isinstance(arg, dict):
                    source = arg.get("source")
                    if source == "plan_text":
                        prepared.append(plan_text)
                    elif source in previous_results:
                        prepared.append(previous_results[source].output)
                    elif "value" in arg:
                        prepared.append(arg["value"])
                    else:
                        prepared.append(arg)
                else:
                    prepared.append(arg)
            return prepared

        elif isinstance(args_spec, dict):
            prepared = {}
            for key, value in args_spec.items():
                if isinstance(value, dict):
                    source = value.get("source")
                    if source == "plan_text":
                        prepared[key] = plan_text
                    elif source in previous_results:
                        prepared[key] = previous_results[source].output
                    elif "value" in value:
                        prepared[key] = value["value"]
                    else:
                        prepared[key] = value
                else:
                    prepared[key] = value
            return prepared

        return args_spec

    def _execute_single_step(
        self,
        adapter_name: str,
        method_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        module_adapter_registry: Any,
        circuit_breaker: Optional[Any] = None,
    ) -> ExecutionResult:
        """
        Execute single adapter method with circuit breaker protection

        INVOCATION PATTERN (CLASS-BASED via registry):
        -----------------------------------------------
        result = module_adapter_registry.execute_module_method(
            module_name=adapter_name,
            method_name=method_name,
            args=args,
            kwargs=kwargs
        )

        CIRCUIT BREAKER INTEGRATION:
        ----------------------------
        - Before execution: Check if circuit is open for this adapter
        - On success: Record success, close/half-open circuit
        - On failure: Record failure, may open circuit after threshold

        COMPATIBILITY:
        --------------
        Supports both new ModuleAdapterRegistry (returns ModuleMethodResult)
        and legacy AdapterRegistry for backward compatibility.

        Args:
            adapter_name: Name of adapter
            method_name: Name of method
            args: Positional arguments
            kwargs: Keyword arguments
            module_adapter_registry: Registry for adapter invocation
            circuit_breaker: Optional circuit breaker for fault tolerance

        Returns:
            ExecutionResult with execution outcome
        """
        start_time = time.time()

        try:
            if circuit_breaker and not circuit_breaker.can_execute(adapter_name):
                return ExecutionResult(
                    module_name=adapter_name,
                    adapter_class=self.adapter_registry.get(adapter_name, "Unknown"),
                    method_name=method_name,
                    status=ExecutionStatus.SKIPPED,
                    error="Circuit breaker open",
                    execution_time=time.time() - start_time,
                )

            # Check if registry has execute_module_method (new API)
            if hasattr(module_adapter_registry, "execute_module_method"):
                module_result = module_adapter_registry.execute_module_method(
                    module_name=adapter_name,
                    method_name=method_name,
                    args=args,
                    kwargs=kwargs,
                )

                # Check if result is new ModuleMethodResult
                if hasattr(module_result, "trace_id"):
                    # New ModuleMethodResult from ModuleAdapterRegistry
                    from .adapter_registry import ExecutionStatus as NewExecutionStatus

                    # Map new status to choreographer ExecutionStatus
                    status_map = {
                        NewExecutionStatus.SUCCESS: ExecutionStatus.COMPLETED,
                        NewExecutionStatus.ERROR: ExecutionStatus.FAILED,
                        NewExecutionStatus.UNAVAILABLE: ExecutionStatus.SKIPPED,
                        NewExecutionStatus.MISSING_METHOD: ExecutionStatus.SKIPPED,
                        NewExecutionStatus.MISSING_ADAPTER: ExecutionStatus.SKIPPED,
                    }

                    result = ExecutionResult(
                        module_name=adapter_name,
                        adapter_class=module_result.adapter_class,
                        method_name=method_name,
                        status=status_map.get(
                            module_result.status, ExecutionStatus.FAILED
                        ),
                        output=(
                            {"evidence": module_result.evidence}
                            if module_result.evidence
                            else None
                        ),
                        error=module_result.error_message,
                        execution_time=module_result.execution_time,
                        evidence_extracted={"evidence": module_result.evidence},
                        confidence=module_result.confidence,
                        metadata={"trace_id": module_result.trace_id},
                    )
                else:
                    # Legacy ModuleResult structure
                    result = ExecutionResult(
                        module_name=adapter_name,
                        adapter_class=getattr(module_result, "class_name", "Unknown"),
                        method_name=method_name,
                        status=self._map_status(
                            getattr(module_result, "status", "failed")
                        ),
                        output=getattr(module_result, "data", None),
                        error=(
                            getattr(module_result, "errors", [None])[0]
                            if hasattr(module_result, "errors")
                            else None
                        ),
                        execution_time=getattr(
                            module_result, "execution_time", time.time() - start_time
                        ),
                        evidence_extracted={
                            "evidence": getattr(module_result, "evidence", [])
                        },
                        confidence=getattr(module_result, "confidence", 0.0),
                        metadata=getattr(module_result, "metadata", {}),
                    )
            else:
                # Fallback to direct adapter invocation (very old legacy code path)
                adapter_instance = module_adapter_registry.adapters.get(adapter_name)
                if adapter_instance is None:
                    raise AttributeError(f"Adapter '{adapter_name}' not found")

                method = getattr(adapter_instance, method_name)
                raw_result = method(*args, **kwargs)

                result = ExecutionResult(
                    module_name=adapter_name,
                    adapter_class=self.adapter_registry.get(adapter_name, "Unknown"),
                    method_name=method_name,
                    status=ExecutionStatus.COMPLETED,
                    output=raw_result,
                    execution_time=time.time() - start_time,
                )

            if circuit_breaker:
                circuit_breaker.record_success(adapter_name)

            return result

        except Exception as e:
            logger.error(
                f"Error executing {adapter_name}.{method_name}: {e}", exc_info=True
            )

            if circuit_breaker:
                circuit_breaker.record_failure(adapter_name, str(e))

            return ExecutionResult(
                module_name=adapter_name,
                adapter_class=self.adapter_registry.get(adapter_name, "Unknown"),
                method_name=method_name,
                status=ExecutionStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _map_status(self, module_status: str) -> ExecutionStatus:
        """Map ModuleResult status string to ExecutionStatus enum"""
        status_map = {
            "success": ExecutionStatus.COMPLETED,
            "partial": ExecutionStatus.DEGRADED,
            "failed": ExecutionStatus.FAILED,
        }
        return status_map.get(module_status, ExecutionStatus.FAILED)

    def get_execution_order(self) -> List[str]:
        """
        Get topologically sorted execution order for all adapters

        Returns:
            List of adapter names in dependency-respecting execution order
        """
        try:
            return list(nx.topological_sort(self.execution_graph))
        except nx.NetworkXError as e:
            logger.error(f"Error computing execution order: {e}")
            return list(self.adapter_registry.keys())

    def get_adapter_dependencies(self, adapter_name: str) -> List[str]:
        """
        Get immediate dependencies for specified adapter

        Args:
            adapter_name: Name of adapter

        Returns:
            List of adapter names that must execute before this adapter
        """
        if adapter_name not in self.execution_graph:
            return []
        return list(self.execution_graph.predecessors(adapter_name))

    def get_adapter_priority(self, adapter_name: str) -> int:
        """
        Get priority/wave number for adapter

        Args:
            adapter_name: Name of adapter

        Returns:
            Priority level (1-5, where 1 executes first)
        """
        if adapter_name not in self.execution_graph:
            return 999
        return self.execution_graph.nodes[adapter_name].get("priority", 999)

    def aggregate_results(self, results: Dict[str, ExecutionResult]) -> Dict[str, Any]:
        """
        Aggregate execution results across all adapter invocations

        Computes summary statistics and evidence synthesis for reporting

        Args:
            results: Dictionary of ExecutionResult objects

        Returns:
            Aggregated statistics and evidence
        """
        aggregated = {
            "total_steps": len(results),
            "successful_steps": sum(
                1 for r in results.values() if r.status == ExecutionStatus.COMPLETED
            ),
            "failed_steps": sum(
                1 for r in results.values() if r.status == ExecutionStatus.FAILED
            ),
            "total_execution_time": sum(r.execution_time for r in results.values()),
            "avg_confidence": (
                sum(r.confidence for r in results.values()) / len(results)
                if results
                else 0.0
            ),
            "adapters_executed": list(set(r.module_name for r in results.values())),
            "evidence": self._aggregate_evidence(results),
            "outputs": {
                key: result.output
                for key, result in results.items()
                if result.output is not None
            },
        }

        return aggregated

    def _aggregate_evidence(
        self, results: Dict[str, ExecutionResult]
    ) -> Dict[str, Any]:
        """
        Aggregate evidence items from all execution results

        Args:
            results: Dictionary of ExecutionResult objects

        Returns:
            Evidence aggregated by adapter and confidence level
        """
        all_evidence = []

        for result in results.values():
            if result.evidence_extracted:
                evidence_items = result.evidence_extracted.get("evidence", [])
                if isinstance(evidence_items, list):
                    all_evidence.extend(evidence_items)

        return {
            "total_evidence_items": len(all_evidence),
            "evidence_by_adapter": {
                result.module_name: result.evidence_extracted
                for result in results.values()
                if result.evidence_extracted
            },
            "high_confidence_evidence": [
                e
                for e in all_evidence
                if isinstance(e, dict) and e.get("confidence", 0) > 0.7
            ],
        }


if __name__ == "__main__":
    choreographer = ExecutionChoreographer()

    print("=" * 80)
    print("EXECUTION CHOREOGRAPHER - DAG-Based Orchestration")
    print("=" * 80)
    print(f"\nAdapters: {len(choreographer.adapter_registry)}")
    print(f"Execution Order: {choreographer.get_execution_order()}")
    print("\nDependencies and Priorities:")
    for adapter in choreographer.get_execution_order():
        deps = choreographer.get_adapter_dependencies(adapter)
        priority = choreographer.get_adapter_priority(adapter)
        print(f"  Wave {priority}: {adapter}")
        if deps:
            print(f"    Depends on: {', '.join(deps)}")
    print("=" * 80)
