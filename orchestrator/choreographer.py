"""
Execution Choreographer - Orchestrates module execution with dependency management
Implements hybrid parallel/sequential execution strategy with detailed component mapping
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Dict, List, Set, Tuple, Any, Optional, Union
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
    component_name: str
    method_name: str
    status: ExecutionStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    evidence_extracted: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


class ExecutionChoreographer:
    """
    Choreographs the execution of multiple modules with dependency management.

    Strategy:
    - Priority 1 modules (extractors/segmenters) run in parallel
    - Priority 2+ modules (analyzers) run after dependencies complete
    - Uses DAG (Directed Acyclic Graph) for dependency tracking
    - Implements detailed question-to-component mapping
    """

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or CONFIG.max_parallel_workers
        self.router = QuestionRouter()
        self.execution_graph = nx.DiGraph()
        self._build_dependency_graph()
        self._initialize_component_mapping()

    def _build_dependency_graph(self):
        """
        Build execution dependency graph.

        Dependencies:
        - Segmenters must run first (segments text)
        - Processors need segmented text
        - Analyzers can run in parallel after processing
        - Validators need analyzer output
        """

        # Add all modules as nodes
        for module_name in CONFIG.modules.keys():
            self.execution_graph.add_node(
                module_name,
                priority=CONFIG.modules[module_name].priority
            )

        # Define dependencies (edges)
        dependencies = [
            # Segmenters are the foundation - everything depends on them
            ("policy_segmenter", "embedding_policy"),
            ("policy_segmenter", "policy_processor"),
            ("policy_segmenter", "causal_processor"),
            ("policy_segmenter", "analyzer_one"),

            # Processors depend on segmentation
            ("embedding_policy", "contradiction_detector"),
            ("policy_processor", "contradiction_detector"),
            ("causal_processor", "dereck_beach"),
        ]

        for source, target in dependencies:
            self.execution_graph.add_edge(source, target)

        logger.info(f"Built dependency graph with {self.execution_graph.number_of_nodes()} nodes "
                    f"and {self.execution_graph.number_of_edges()} edges")

    def _initialize_component_mapping(self):
        """
        Initialize the detailed question-to-component mapping.
        This mapping connects each question to specific components and methods.
        """
        self.component_mapping = {
            # Dimension 1 (D1: Insumos/Inputs)
            "D1-Q1": {
                "components": [
                    ("semantic_processor", "chunk_text"),
                    ("policy_processor", "normalize_unicode"),
                    ("embedding_policy", "chunk_document"),
                    ("policy_segmenter", "segment")
                ],
                "primary": "semantic_processor"
            },
            "D1-Q2": {
                "components": [
                    ("bayesian_integrator", "integrate_evidence"),
                    ("analyzer_one", "extract_semantic_cube"),
                    ("embedding_policy", "evaluate_policy_metric")
                ],
                "primary": "bayesian_integrator"
            },
            "D1-Q3": {
                "components": [
                    ("financial_analyzer", "trace_financial_allocation"),
                    ("pdet_analyzer", "analyze_financial_feasibility"),
                    ("causal_processor", "extract_goals")
                ],
                "primary": "financial_analyzer"
            },
            "D1-Q4": {
                "components": [
                    ("analyzer_one", "classify_value_chain_link"),
                    ("analyzer_one", "analyze_performance"),
                    ("analyzer_one", "diagnose_critical_links")
                ],
                "primary": "analyzer_one"
            },
            "D1-Q5": {
                "components": [
                    ("contradiction_detector", "detect"),
                    ("dereck_beach", "audit_evidence_traceability"),
                    ("semantic_processor", "detect_pdm_structure")
                ],
                "primary": "contradiction_detector"
            },

            # Dimension 2 (D2: Actividades/Activities)
            "D2-Q1": {
                "components": [
                    ("policy_segmenter", "segment"),
                    ("policy_segmenter", "segment_into_sentences")
                ],
                "primary": "policy_segmenter"
            },
            "D2-Q2": {
                "components": [
                    ("dereck_beach", "extract_entity_activity"),
                    ("causal_processor", "extract_causal_links"),
                    ("causal_processor", "build_causal_graph")
                ],
                "primary": "dereck_beach"
            },
            "D2-Q3": {
                "components": [
                    ("causal_processor", "extract_causal_hierarchy"),
                    ("pdet_analyzer", "construct_causal_dag"),
                    ("causal_validator", "add_edge")
                ],
                "primary": "causal_processor"
            },
            "D2-Q4": {
                "components": [
                    ("contradiction_detector", "detect_logical_incompatibilities"),
                    ("analyzer_one", "assess_risks"),
                    ("analyzer_one", "detect_bottlenecks")
                ],
                "primary": "contradiction_detector"
            },
            "D2-Q5": {
                "components": [
                    ("contradiction_detector", "verify_temporal_consistency"),
                    ("contradiction_detector", "detect_temporal_conflicts"),
                    ("causal_processor", "assess_temporal_coherence")
                ],
                "primary": "contradiction_detector"
            },

            # Dimension 3 (D3: Productos/Products)
            "D3-Q1": {
                "components": [
                    ("dereck_beach", "validate_dnp_compliance"),
                    ("policy_processor", "analyze"),
                    ("semantic_processor", "detect_table")
                ],
                "primary": "dereck_beach"
            },
            "D3-Q2": {
                "components": [
                    ("embedding_policy", "evaluate_policy_metric"),
                    ("semantic_processor", "detect_numerical_data"),
                    ("embedding_policy", "evaluate_policy_numerical_consistency")
                ],
                "primary": "embedding_policy"
            },
            "D3-Q3": {
                "components": [
                    ("financial_analyzer", "trace_financial_allocation"),
                    ("pdet_analyzer", "extract_budget_for_pillar"),
                    ("causal_processor", "assess_financial_consistency")
                ],
                "primary": "financial_analyzer"
            },
            "D3-Q4": {
                "components": [
                    ("analyzer_one", "analyze_performance"),
                    ("pdet_analyzer", "assess_financial_sustainability"),
                    ("dereck_beach", "calculate_coherence_factor")
                ],
                "primary": "analyzer_one"
            },
            "D3-Q5": {
                "components": [
                    ("dereck_beach", "extract_entity_activity"),
                    ("analyzer_one", "process_segment"),
                    ("causal_processor", "build_type_hierarchy")
                ],
                "primary": "dereck_beach"
            },

            # Dimension 4 (D4: Resultados/Results)
            "D4-Q1": {
                "components": [
                    ("embedding_policy", "evaluate_policy_metric"),
                    ("semantic_processor", "detect_numerical_data"),
                    ("embedding_policy", "extract_numerical_values")
                ],
                "primary": "embedding_policy"
            },
            "D4-Q2": {
                "components": [
                    ("causal_processor", "extract_causal_hierarchy"),
                    ("causal_processor", "validate_complete"),
                    ("causal_validator", "calculate_acyclicity_pvalue")
                ],
                "primary": "causal_processor"
            },
            "D4-Q3": {
                "components": [
                    ("contradiction_detector", "verify_temporal_consistency"),
                    ("contradiction_detector", "extract_temporal_markers"),
                    ("causal_processor", "assess_temporal_coherence")
                ],
                "primary": "contradiction_detector"
            },
            "D4-Q4": {
                "components": [
                    ("dereck_beach", "audit_sequence_logic"),
                    ("analyzer_one", "generate_recommendations"),
                    ("dereck_beach", "generate_accountability_matrix")
                ],
                "primary": "dereck_beach"
            },
            "D4-Q5": {
                "components": [
                    ("analyzer_one", "classify_policy_domain"),
                    ("embedding_policy", "semantic_search"),
                    ("policy_processor", "analyze_causal_dimensions")
                ],
                "primary": "analyzer_one"
            },

            # Dimension 5 (D5: Impactos/Impacts)
            "D5-Q1": {
                "components": [
                    ("embedding_policy", "beta_binomial_posterior"),
                    ("pdet_analyzer", "generate_counterfactuals"),
                    ("embedding_policy", "compare_policy_interventions")
                ],
                "primary": "embedding_policy"
            },
            "D5-Q2": {
                "components": [
                    ("analyzer_one", "classify_cross_cutting_themes"),
                    ("embedding_policy", "rerank"),
                    ("embedding_policy", "filter_by_pdq")
                ],
                "primary": "analyzer_one"
            },
            "D5-Q3": {
                "components": [
                    ("dereck_beach", "test_sufficiency"),
                    ("dereck_beach", "test_necessity"),
                    ("causal_validator", "perform_sensitivity_analysis")
                ],
                "primary": "dereck_beach"
            },
            "D5-Q4": {
                "components": [
                    ("contradiction_detector", "detect_resource_conflicts"),
                    ("pdet_analyzer", "sensitivity_analysis"),
                    ("analyzer_one", "assess_risks")
                ],
                "primary": "contradiction_detector"
            },
            "D5-Q5": {
                "components": [
                    ("contradiction_detector", "detect"),
                    ("pdet_analyzer", "simulate_intervention"),
                    ("causal_processor", "validate_causal_order")
                ],
                "primary": "contradiction_detector"
            },

            # Dimension 6 (D6: Causalidad/Causality)
            "D6-Q1": {
                "components": [
                    ("causal_processor", "build_causal_graph"),
                    ("causal_processor", "extract_causal_hierarchy"),
                    ("policy_processor", "analyze_causal_dimensions")
                ],
                "primary": "causal_processor"
            },
            "D6-Q2": {
                "components": [
                    ("dereck_beach", "apply_test_logic"),
                    ("dereck_beach", "infer_mechanisms"),
                    ("dereck_beach", "assign_probative_value")
                ],
                "primary": "dereck_beach"
            },
            "D6-Q3": {
                "components": [
                    ("contradiction_detector", "detect"),
                    ("causal_validator", "is_acyclic"),
                    ("dereck_beach", "bayesian_counterfactual_audit")
                ],
                "primary": "contradiction_detector"
            },
            "D6-Q4": {
                "components": [
                    ("dereck_beach", "generate_causal_diagram"),
                    ("causal_processor", "check_structural_violation"),
                    ("analyzer_one", "calculate_throughput_metrics")
                ],
                "primary": "dereck_beach"
            },
            "D6-Q5": {
                "components": [
                    ("embedding_policy", "generate_pdq_report"),
                    ("analyzer_one", "calculate_semantic_complexity"),
                    ("bayesian_integrator", "causal_strength")
                ],
                "primary": "embedding_policy"
            }
        }

    def execute_for_question(
            self,
            question_id: str,
            plan_text: str,
            plan_metadata: Dict[str, Any]
    ) -> Dict[str, ExecutionResult]:
        """
        Execute all required components for a specific question.

        Args:
            question_id: P#-D#-Q# notation
            plan_text: Full development plan text
            plan_metadata: Plan metadata (name, year, etc)

        Returns:
            Dict mapping component_name to ExecutionResult
        """
        logger.info(f"Choreographing execution for question: {question_id}")

        # Extract dimension and question number from question_id
        parts = question_id.split("-")
        if len(parts) < 3:
            logger.error(f"Invalid question_id format: {question_id}")
            return {}

        dimension = parts[1]
        question_num = parts[2]
        dimension_question = f"{dimension}-{question_num}"

        # Get component mapping for this question
        if dimension_question not in self.component_mapping:
            logger.warning(f"No component mapping found for {dimension_question}")
            return {}

        component_info = self.component_mapping[dimension_question]
        components = component_info["components"]
        primary_module = component_info["primary"]

        # Group components by module
        module_components = {}
        for module_name, method_name in components:
            if module_name not in module_components:
                module_components[module_name] = []
            module_components[module_name].append(method_name)

        # Get required modules and their execution order
        required_modules = set(module_components.keys())

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
            "question_id": question_id,
            "dimension": dimension,
            "question_num": question_num
        }

        # Group modules by priority level
        priority_waves = self._group_by_priority(execution_order)

        for wave_num, wave_modules in enumerate(priority_waves, start=1):
            logger.info(f"Executing wave {wave_num}: {wave_modules}")

            # Execute this wave in parallel
            wave_results = self._execute_wave(
                wave_modules,
                module_components,
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
            module_components: Dict[str, List[str]],
            context: Dict[str, Any],
            previous_results: Dict[str, ExecutionResult]
    ) -> Dict[str, ExecutionResult]:
        """
        Execute a wave of modules in parallel.

        Args:
            modules: List of module names to execute
            module_components: Mapping of module to list of methods to execute
            context: Shared execution context
            previous_results: Results from previous waves

        Returns:
            Dict of component_name to ExecutionResult
        """
        wave_results = {}

        if len(modules) == 1:
            # Single module - execute directly
            module_results = self._execute_module(
                modules[0],
                module_components[modules[0]],
                context,
                previous_results
            )
            wave_results.update(module_results)
        else:
            # Multiple modules - execute in parallel
            with ThreadPoolExecutor(max_workers=min(len(modules), self.max_workers)) as executor:
                # Submit all tasks
                future_to_module = {
                    executor.submit(
                        self._execute_module,
                        module,
                        module_components[module],
                        context,
                        previous_results
                    ): module
                    for module in modules
                }

                # Collect results as they complete
                for future in as_completed(future_to_module):
                    module = future_to_module[future]
                    try:
                        module_results = future.result(timeout=CONFIG.modules[module].timeout_seconds)
                        wave_results.update(module_results)
                    except Exception as e:
                        logger.error(f"Module {module} failed: {e}")
                        # Create failed results for all components of this module
                        for method_name in module_components[module]:
                            component_key = f"{module}.{method_name}"
                            wave_results[component_key] = ExecutionResult(
                                module_name=module,
                                component_name=component_key,
                                method_name=method_name,
                                status=ExecutionStatus.FAILED,
                                error=str(e)
                            )

        return wave_results

    def _execute_module(
            self,
            module_name: str,
            method_names: List[str],
            context: Dict[str, Any],
            previous_results: Dict[str, ExecutionResult]
    ) -> Dict[str, ExecutionResult]:
        """
        Execute a single module with multiple methods using the actual module adapters.

        This calls the ACTUAL module through ModuleAdapterRegistry
        """
        start_time = time.time()
        module_config = CONFIG.modules[module_name]

        logger.info(f"Executing module: {module_name} with methods: {method_names}")

        try:
            # Import adapter registry
            from .module_adapters import ModuleAdapterRegistry

            # Initialize registry
            registry = ModuleAdapterRegistry()

            # Prepare arguments based on module type
            plan_text = context.get("plan_text", "")
            plan_metadata = context.get("plan_metadata", {})
            question_id = context.get("question_id", "")
            dimension = context.get("dimension", "")
            question_num = context.get("question_num", "")

            # Results for this module
            module_results = {}

            # Execute each method
            for method_name in method_names:
                method_start_time = time.time()

                # Prepare arguments based on method
                args = [plan_text]
                kwargs = {"plan_metadata": plan_metadata}

                # Add question-specific parameters for certain modules
                if module_name == "contradiction_detector":
                    kwargs["plan_name"] = plan_metadata.get("name", "Unknown")
                    kwargs["dimension"] = dimension

                # Execute module method
                result = registry.execute_module_method(
                    module_name=module_name,
                    method_name=method_name,
                    args=args,
                    kwargs=kwargs
                )

                # Convert ModuleResult to ExecutionResult
                status = ExecutionStatus.COMPLETED if result.status == "success" else ExecutionStatus.FAILED

                component_key = f"{module_name}.{method_name}"
                module_results[component_key] = ExecutionResult(
                    module_name=module_name,
                    component_name=component_key,
                    method_name=method_name,
                    status=status,
                    output=result.data,
                    execution_time=result.execution_time,
                    evidence_extracted=self._extract_evidence_from_output(module_name, result.data),
                    confidence=result.confidence,
                    error=result.errors[0] if result.errors else None
                )

            return module_results

        except Exception as e:
            logger.exception(f"Module {module_name} execution failed")

            # Create failed results for all methods
            module_results = {}
            for method_name in method_names:
                component_key = f"{module_name}.{method_name}"
                module_results[component_key] = ExecutionResult(
                    module_name=module_name,
                    component_name=component_key,
                    method_name=method_name,
                    status=ExecutionStatus.FAILED,
                    error=str(e),
                    execution_time=time.time() - start_time
                )

            return module_results

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
            evidence["confidence_scores"]["coherence"] = output.get("coherence_metrics", {}).get("coherence_score", 0.0)

        elif module_name == "causal_processor":
            evidence["causal_links"] = output.get("causal_dimensions", {})
            evidence["confidence_scores"]["causal_strength"] = output.get("information_gain", 0.0)

        elif module_name == "dereck_beach":
            evidence["causal_links"] = output.get("mechanism_parts", [])
            evidence["confidence_scores"]["mechanism_confidence"] = output.get("rigor_status", 0.0)
            evidence["quantitative_claims"].append({
                "causal_hierarchy": output.get("causal_hierarchy", {}),
                "mechanism_inferences": output.get("mechanism_inferences", [])
            })

        elif module_name == "policy_processor":
            dimensions_data = output.get("dimensions", {})
            for dim, dim_data in dimensions_data.items():
                evidence["quantitative_claims"].append({
                    "dimension": dim,
                    "point_evidence": dim_data.get("point_evidence", []),
                    "bayesian_score": dim_data.get("bayesian_score", 0.0)
                })
            evidence["confidence_scores"]["overall"] = output.get("overall_score", 0.0)

        elif module_name == "financial_analyzer":
            evidence["quantitative_claims"] = output.get("budget_analysis", {})
            evidence["confidence_scores"]["financial_coherence"] = output.get("viability_score", 0.0)

        elif module_name == "analyzer_one":
            evidence["quantitative_claims"] = output.get("analysis_results", {})
            evidence["confidence_scores"]["semantic_quality"] = output.get("quality_score", 0.0)

        elif module_name == "embedding_policy":
            evidence["quantitative_claims"].append({
                "chunks_processed": output.get("chunks_processed", 0),
                "embeddings_generated": output.get("embeddings_generated", False)
            })
            evidence["confidence_scores"]["embedding_quality"] = output.get("confidence", 0.0)

        elif module_name == "policy_segmenter":
            segments = output.get("segments", [])
            evidence["quantitative_claims"].append({
                "num_segments": len(segments),
                "avg_segment_length": sum(len(s.get("text", "")) for s in segments) / len(segments) if segments else 0
            })
            evidence["confidence_scores"]["segmentation_quality"] = output.get("confidence", 0.0)

        return evidence

    def get_execution_statistics(self, results: Dict[str, ExecutionResult]) -> Dict[str, Any]:
        """Generate statistics from execution results"""
        total_components = len(results)
        successful = sum(1 for r in results.values() if r.status == ExecutionStatus.COMPLETED)
        failed = sum(1 for r in results.values() if r.status == ExecutionStatus.FAILED)
        total_time = sum(r.execution_time for r in results.values())

        # Group by module
        module_stats = {}
        for component_key, result in results.items():
            module_name = result.module_name
            if module_name not in module_stats:
                module_stats[module_name] = {"total": 0, "successful": 0, "failed": 0, "time": 0.0}

            module_stats[module_name]["total"] += 1
            module_stats[module_name]["time"] += result.execution_time

            if result.status == ExecutionStatus.COMPLETED:
                module_stats[module_name]["successful"] += 1
            else:
                module_stats[module_name]["failed"] += 1

        return {
            "total_components": total_components,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_components if total_components > 0 else 0.0,
            "total_execution_time": total_time,
            "avg_execution_time": total_time / total_components if total_components > 0 else 0.0,
            "module_stats": module_stats
        }