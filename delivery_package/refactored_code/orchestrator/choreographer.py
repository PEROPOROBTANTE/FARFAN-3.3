# choreographer.py - COMPLETE UPDATE FOR 9 ADAPTERS
# coding=utf-8
"""
Execution Choreographer - Orchestrates Module Execution with Dependency Management
==================================================================================

Updated for complete integration with:
- module_adapters_COMPLETE_MERGED.py (9 adapters, 413 methods)
- FARFAN_3.0_UPDATED_QUESTIONNAIRE.yaml (execution chains)
- ModuleResult standardized format

Implements:
- Hybrid parallel/sequential execution
- DAG-based dependency tracking  
- Question-to-adapter mapping
- Evidence aggregation

Author: Integration Team
Version: 3.0.0 - Complete Adapter Alignment
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


# ============================================================================
# EXECUTION STATUS AND RESULTS
# ============================================================================

class ExecutionStatus(Enum):
    """Module execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    DEGRADED = "degraded"  # Partial success


@dataclass
class ExecutionResult:
    """
    Result from a single adapter execution
    
    Wraps ModuleResult from module_adapters.py
    """
    module_name: str  # Adapter name (e.g., "teoria_cambio")
    adapter_class: str  # Adapter class name (e.g., "ModulosAdapter")
    method_name: str  # Method executed
    status: ExecutionStatus
    
    # Core results
    output: Optional[Dict[str, Any]] = None  # ModuleResult.data
    error: Optional[str] = None
    execution_time: float = 0.0
    
    # Evidence and confidence
    evidence_extracted: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    
    # Module-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for aggregation"""
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
            "metadata": self.metadata
        }


# ============================================================================
# EXECUTION CHOREOGRAPHER
# ============================================================================

class ExecutionChoreographer:
    """
    Choreographs execution of 9 adapters with dependency management
    
    Strategy:
    - Build DAG of adapter dependencies
    - Execute in topological order with parallelization
    - Aggregate results with evidence tracking
    - Handle failures gracefully
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize choreographer
        
        Args:
            max_workers: Max parallel workers (default: 4)
        """
        self.max_workers = max_workers or 4
        self.execution_graph = nx.DiGraph()
        self._build_dependency_graph()
        self._initialize_adapter_registry()
        
        logger.info(f"ExecutionChoreographer initialized with {self.max_workers} workers")

    def _build_dependency_graph(self):
        """
        Build execution dependency graph for 9 adapters
        
        Dependencies based on data flow:
        1. policy_segmenter → provides segments
        2. policy_processor → provides normalized text
        3. semantic_chunking_policy → provides semantic chunks
        4. embedding_policy → provides embeddings
        5. teoria_cambio → uses all above
        6. analyzer_one → uses segments + embeddings
        7. dereck_beach → uses causal analysis
        8. contradiction_detection → uses all analysis
        9. financial_viability → uses everything
        """
        
        # Add all 9 adapters as nodes with priorities
        adapters = {
            # Priority 1: Foundation (run first, in parallel)
            "policy_segmenter": 1,
            "policy_processor": 1,
            
            # Priority 2: Semantic analysis (run after foundation)
            "semantic_chunking_policy": 2,
            "embedding_policy": 2,
            
            # Priority 3: Advanced analysis (run after semantics)
            "analyzer_one": 3,
            "teoria_cambio": 3,
            
            # Priority 4: Specialized analysis (run after advanced)
            "dereck_beach": 4,
            "contradiction_detection": 4,
            
            # Priority 5: Final synthesis (run last)
            "financial_viability": 5
        }
        
        for adapter_name, priority in adapters.items():
            self.execution_graph.add_node(
                adapter_name,
                priority=priority
            )
        
        # Define dependencies (edges: source → target)
        dependencies = [
            # Foundation → Semantic
            ("policy_segmenter", "semantic_chunking_policy"),
            ("policy_segmenter", "embedding_policy"),
            ("policy_processor", "semantic_chunking_policy"),
            ("policy_processor", "embedding_policy"),
            
            # Semantic → Advanced
            ("semantic_chunking_policy", "analyzer_one"),
            ("semantic_chunking_policy", "teoria_cambio"),
            ("embedding_policy", "analyzer_one"),
            ("embedding_policy", "teoria_cambio"),
            
            # Advanced → Specialized
            ("analyzer_one", "contradiction_detection"),
            ("teoria_cambio", "dereck_beach"),
            ("teoria_cambio", "contradiction_detection"),
            
            # Specialized → Synthesis
            ("dereck_beach", "financial_viability"),
            ("contradiction_detection", "financial_viability"),
            ("analyzer_one", "financial_viability")
        ]
        
        for source, target in dependencies:
            self.execution_graph.add_edge(source, target)
        
        # Verify DAG (no cycles)
        if not nx.is_directed_acyclic_graph(self.execution_graph):
            raise ValueError("Dependency graph contains cycles!")
        
        logger.info(
            f"Built dependency graph: {self.execution_graph.number_of_nodes()} adapters, "
            f"{self.execution_graph.number_of_edges()} dependencies"
        )

    def _initialize_adapter_registry(self):
        """Initialize adapter name to class mapping"""
        self.adapter_registry = {
            "teoria_cambio": "ModulosAdapter",
            "analyzer_one": "AnalyzerOneAdapter",
            "dereck_beach": "DerekBeachAdapter",
            "embedding_policy": "EmbeddingPolicyAdapter",
            "semantic_chunking_policy": "SemanticChunkingPolicyAdapter",
            "contradiction_detection": "ContradictionDetectionAdapter",
            "financial_viability": "FinancialViabilityAdapter",
            "policy_processor": "PolicyProcessorAdapter",
            "policy_segmenter": "PolicySegmenterAdapter"
        }
        
        logger.info(f"Initialized registry for {len(self.adapter_registry)} adapters")

    def execute_question_chain(
            self,
            question_spec,
            plan_text: str,
            module_adapter_registry,
            circuit_breaker=None
    ) -> Dict[str, ExecutionResult]:
        """
        Execute complete chain for a single question
        
        Args:
            question_spec: Question specification from questionnaire
            plan_text: Plan document text
            module_adapter_registry: ModuleAdapterRegistry instance
            circuit_breaker: Optional circuit breaker for fault tolerance
            
        Returns:
            Dict mapping adapter names to ExecutionResults
        """
        logger.info(f"Executing chain for {question_spec.canonical_id}")
        
        start_time = time.time()
        results = {}
        
        # Get execution chain from question spec
        execution_chain = getattr(question_spec, 'execution_chain', [])
        
        if not execution_chain:
            logger.warning(f"No execution chain for {question_spec.canonical_id}")
            return results
        
        # Execute steps sequentially (respecting dependencies in chain)
        for step in execution_chain:
            adapter_name = step.get('adapter')
            method_name = step.get('method')
            args = step.get('args', [])
            kwargs = step.get('kwargs', {})
            
            if not adapter_name or not method_name:
                logger.warning(f"Incomplete step in chain: {step}")
                continue
            
            # Prepare arguments (may reference previous results)
            prepared_args = self._prepare_arguments(args, results, plan_text)
            prepared_kwargs = self._prepare_arguments(kwargs, results, plan_text)
            
            # Execute via registry
            result = self._execute_single_step(
                adapter_name=adapter_name,
                method_name=method_name,
                args=prepared_args,
                kwargs=prepared_kwargs,
                module_adapter_registry=module_adapter_registry,
                circuit_breaker=circuit_breaker
            )
            
            results[f"{adapter_name}.{method_name}"] = result
        
        total_time = time.time() - start_time
        logger.info(
            f"Completed chain for {question_spec.canonical_id} in {total_time:.2f}s "
            f"({len(results)} steps)"
        )
        
        return results

    def _prepare_arguments(
            self,
            args_spec: Union[List, Dict],
            previous_results: Dict[str, ExecutionResult],
            plan_text: str
    ) -> Union[List, Dict]:
        """
        Prepare arguments by resolving references to previous results
        
        Supports:
        - source: "plan_text" → plan_text
        - source: "previous_result" → result from previous step
        - source: "derived" → computed from previous results
        """
        if isinstance(args_spec, list):
            prepared = []
            for arg in args_spec:
                if isinstance(arg, dict):
                    source = arg.get('source')
                    if source == 'plan_text':
                        prepared.append(plan_text)
                    elif source in previous_results:
                        prepared.append(previous_results[source].output)
                    elif 'value' in arg:
                        prepared.append(arg['value'])
                    else:
                        prepared.append(arg)
                else:
                    prepared.append(arg)
            return prepared
        
        elif isinstance(args_spec, dict):
            prepared = {}
            for key, value in args_spec.items():
                if isinstance(value, dict):
                    source = value.get('source')
                    if source == 'plan_text':
                        prepared[key] = plan_text
                    elif source in previous_results:
                        prepared[key] = previous_results[source].output
                    elif 'value' in value:
                        prepared[key] = value['value']
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
            module_adapter_registry,
            circuit_breaker=None
    ) -> ExecutionResult:
        """
        Execute a single step in the execution chain
        
        Returns:
            ExecutionResult with module output
        """
        start_time = time.time()
        
        try:
            # Check circuit breaker
            if circuit_breaker and not circuit_breaker.can_execute(adapter_name):
                return ExecutionResult(
                    module_name=adapter_name,
                    adapter_class=self.adapter_registry.get(adapter_name, "Unknown"),
                    method_name=method_name,
                    status=ExecutionStatus.SKIPPED,
                    error="Circuit breaker open",
                    execution_time=time.time() - start_time
                )
            
            # Execute via ModuleAdapterRegistry
            module_result = module_adapter_registry.execute_module_method(
                module_name=adapter_name,
                method_name=method_name,
                args=args,
                kwargs=kwargs
            )
            
            # Convert ModuleResult to ExecutionResult
            result = ExecutionResult(
                module_name=adapter_name,
                adapter_class=module_result.class_name,
                method_name=method_name,
                status=self._map_status(module_result.status),
                output=module_result.data,
                error=module_result.errors[0] if module_result.errors else None,
                execution_time=module_result.execution_time,
                evidence_extracted={"evidence": module_result.evidence},
                confidence=module_result.confidence,
                metadata=module_result.metadata
            )
            
            # Record success in circuit breaker
            if circuit_breaker:
                circuit_breaker.record_success(adapter_name)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing {adapter_name}.{method_name}: {e}", exc_info=True)
            
            # Record failure in circuit breaker
            if circuit_breaker:
                circuit_breaker.record_failure(adapter_name, str(e))
            
            return ExecutionResult(
                module_name=adapter_name,
                adapter_class=self.adapter_registry.get(adapter_name, "Unknown"),
                method_name=method_name,
                status=ExecutionStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _map_status(self, module_status: str) -> ExecutionStatus:
        """Map ModuleResult status to ExecutionStatus"""
        status_map = {
            "success": ExecutionStatus.COMPLETED,
            "partial": ExecutionStatus.DEGRADED,
            "failed": ExecutionStatus.FAILED
        }
        return status_map.get(module_status, ExecutionStatus.FAILED)

    def get_execution_order(self) -> List[str]:
        """
        Get topologically sorted execution order
        
        Returns:
            List of adapter names in execution order
        """
        try:
            return list(nx.topological_sort(self.execution_graph))
        except nx.NetworkXError as e:
            logger.error(f"Error computing execution order: {e}")
            return list(self.adapter_registry.keys())

    def get_adapter_dependencies(self, adapter_name: str) -> List[str]:
        """Get dependencies for a specific adapter"""
        if adapter_name not in self.execution_graph:
            return []
        return list(self.execution_graph.predecessors(adapter_name))

    def get_adapter_priority(self, adapter_name: str) -> int:
        """Get priority of an adapter"""
        if adapter_name not in self.execution_graph:
            return 999
        return self.execution_graph.nodes[adapter_name].get('priority', 999)

    def aggregate_results(
            self,
            results: Dict[str, ExecutionResult]
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple adapters
        
        Returns:
            Aggregated data with evidence synthesis
        """
        aggregated = {
            "total_steps": len(results),
            "successful_steps": sum(
                1 for r in results.values()
                if r.status == ExecutionStatus.COMPLETED
            ),
            "failed_steps": sum(
                1 for r in results.values()
                if r.status == ExecutionStatus.FAILED
            ),
            "total_execution_time": sum(r.execution_time for r in results.values()),
            "avg_confidence": sum(r.confidence for r in results.values()) / len(results)
                if results else 0.0,
            "adapters_executed": list(set(r.module_name for r in results.values())),
            "evidence": self._aggregate_evidence(results),
            "outputs": {
                key: result.output
                for key, result in results.items()
                if result.output is not None
            }
        }
        
        return aggregated

    def _aggregate_evidence(
            self,
            results: Dict[str, ExecutionResult]
    ) -> Dict[str, Any]:
        """Aggregate evidence from all execution results"""
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
                e for e in all_evidence
                if isinstance(e, dict) and e.get("confidence", 0) > 0.7
            ]
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    choreographer = ExecutionChoreographer()
    
    print("=" * 80)
    print("EXECUTION CHOREOGRAPHER - COMPLETE UPDATE")
    print("=" * 80)
    print(f"\nAdapters: {len(choreographer.adapter_registry)}")
    print(f"Execution Order: {choreographer.get_execution_order()}")
    print("\nDependencies:")
    for adapter in choreographer.get_execution_order():
        deps = choreographer.get_adapter_dependencies(adapter)
        priority = choreographer.get_adapter_priority(adapter)
        print(f"  {adapter} (P{priority}): {deps if deps else 'None'}")
    print("=" * 80)