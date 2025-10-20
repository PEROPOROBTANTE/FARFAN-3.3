# choreographer.py - Updated to match new module_adapters.py structure
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
            if source in CONFIG.modules and target in CONFIG.modules:
                self.execution_graph.add_edge(source, target)

        logger.info(f"Built dependency graph with {self.execution_graph.number_of_nodes()} nodes "
                    f"and {self.execution_graph.number_of_edges()} edges")

    def _initialize_component_mapping(self):
        """
        Initialize the detailed question-to-component mapping.
        This mapping connects each question to specific components and methods.
        Uses the QuestionnaireParser to get the questions and their module mappings.
        """
        self.component_mapping = {}

        # Get all questions from the router
        all_questions = self.router.questions

        # Build component mapping based on the questions
        for question_id, question in all_questions.items():
            # Extract dimension and question number from question_id
            parts = question_id.split("-")
            if len(parts) >= 3:
                dimension = parts[1]
                question_num = parts[2]

                # Create component mapping based on the question's required modules
                components = []

                # Add primary module components
                if question.primary_module in CONFIG.modules:
                    # Get methods for the primary module based on dimension and question
                    primary_methods = self._get_module_methods(question.primary_module, dimension, question_num)
                    for method in primary_methods:
                        components.append((question.primary_module, method))

                # Add supporting module components
                for module in question.supporting_modules:
                    if module in CONFIG.modules:
                        # Get methods for the supporting module based on dimension and question
                        supporting_methods = self._get_module_methods(module, dimension, question_num)
                        for method in supporting_methods:
                            components.append((module, method))

                # Store the component mapping
                self.component_mapping[question_id] = {
                    "components": components,
                    "primary": question.primary_module
                }

        logger.info(f"Initialized component mapping for {len(self.component_mapping)} questions")

    def _get_module_methods(self, module_name: str, dimension: str, question_num: int) -> List[str]:
        """
        Get the appropriate methods for a module based on dimension and question number.
        """
        # Define method mappings for each module based on dimension and question
        method_mappings = {
            "semantic_processor": {
                "D1": ["chunk_text", "detect_numerical_data"],
                "D2": ["detect_table", "extract_semantic_cube"],
                "D3": ["detect_table", "detect_numerical_data"],
                "D4": ["detect_numerical_data", "extract_semantic_cube"],
                "D5": ["detect_numerical_data", "extract_semantic_cube"],
                "D6": ["detect_pdm_structure", "extract_semantic_cube"]
            },
            "embedding_policy": {
                "D1": ["chunk_document", "evaluate_policy_metric"],
                "D2": ["chunk_document", "semantic_search"],
                "D3": ["evaluate_policy_metric", "evaluate_policy_numerical_consistency"],
                "D4": ["evaluate_policy_metric", "process_document"],
                "D5": ["evaluate_policy_metric", "evaluate_policy_numerical_consistency"],
                "D6": ["process_document", "semantic_search"]
            },
            "analyzer_one": {
                "D1": ["extract_semantic_cube", "analyze_document"],
                "D2": ["analyze_document", "extract_value_chain"],
                "D3": ["analyze_document", "diagnose_critical_links"],
                "D4": ["analyze_document", "diagnose_critical_links"],
                "D5": ["analyze_document", "extract_value_chain"],
                "D6": ["analyze_document", "extract_semantic_cube"]
            },
            "policy_segmenter": {
                "D1": ["segment", "segment_into_sentences"],
                "D2": ["segment", "segment_into_sentences"],
                "D3": ["segment", "segment_into_sentences"],
                "D4": ["segment", "segment_into_sentences"],
                "D5": ["segment", "segment_into_sentences"],
                "D6": ["segment", "segment_into_sentences"]
            },
            "policy_processor": {
                "D1": ["process", "extract_policy_sections"],
                "D2": ["process", "_extract_point_evidence"],
                "D3": ["process", "score_evidence"],
                "D4": ["process", "_extract_point_evidence"],
                "D5": ["process", "score_evidence"],
                "D6": ["process", "extract_policy_sections"]
            },
            "causal_processor": {
                "D1": ["analyze", "construct_causal_dag"],
                "D2": ["analyze", "construct_causal_dag"],
                "D3": ["analyze", "estimate_causal_effects"],
                "D4": ["analyze", "generate_counterfactuals"],
                "D5": ["analyze", "estimate_causal_effects"],
                "D6": ["analyze", "construct_causal_dag"]
            },
            "contradiction_detector": {
                "D1": ["detect", "calculate_confidence"],
                "D2": ["detect", "verify_temporal_logic"],
                "D3": ["detect", "calculate_confidence"],
                "D4": ["detect", "verify_temporal_logic"],
                "D5": ["detect", "calculate_confidence"],
                "D6": ["detect", "calculate_confidence"]
            },
            "dereck_beach": {
                "D1": ["process_document", "extract_causal_hierarchy"],
                "D2": ["process_document", "extract_entity_activity"],
                "D3": ["process_document", "audit_evidence_traceability"],
                "D4": ["process_document", "trace_financial_allocation"],
                "D5": ["process_document", "infer_mechanisms"],
                "D6": ["process_document", "generate_confidence_report"]
            },
            "financial_analyzer": {
                "D1": ["analyze_financial_feasibility", "extract_financial_indicators"],
                "D2": ["analyze_financial_feasibility", "analyze_budget_allocation"],
                "D3": ["analyze_financial_feasibility", "extract_financial_indicators"],
                "D4": ["analyze_financial_feasibility", "analyze_budget_allocation"],
                "D5": ["analyze_financial_feasibility", "analyze_budget_allocation"],
                "D6": ["analyze_financial_feasibility", "extract_financial_indicators"]
            },
            "bayesian_integrator": {
                "D1": ["integrate_evidence", "calculate_posterior"],
                "D2": ["integrate_evidence", "calculate_posterior"],
                "D3": ["integrate_evidence", "calculate_posterior"],
                "D4": ["integrate_evidence", "calculate_posterior"],
                "D5": ["integrate_evidence", "calculate_posterior"],
                "D6": ["integrate_evidence", "calculate_posterior"]
            },
            "validation_framework": {
                "D1": ["validate_evidence", "check_completeness"],
                "D2": ["validate_evidence", "check_completeness"],
                "D3": ["validate_evidence", "check_completeness"],
                "D4": ["validate_evidence", "check_completeness"],
                "D5": ["validate_evidence", "check_completeness"],
                "D6": ["validate_evidence", "check_completeness"]
            },
            "municipal_analyzer": {
                "D1": ["analyze_municipal_context", "assess_institutional_capacity"],
                "D2": ["analyze_municipal_context", "assess_institutional_capacity"],
                "D3": ["analyze_municipal_context", "assess_institutional_capacity"],
                "D4": ["analyze_municipal_context", "assess_institutional_capacity"],
                "D5": ["analyze_municipal_context", "assess_institutional_capacity"],
                "D6": ["analyze_municipal_context", "assess_institutional_capacity"]
            },
            "pdet_analyzer": {
                "D1": ["analyze_plan", "extract_budget_allocation"],
                "D2": ["analyze_plan", "assess_financial_sustainability"],
                "D3": ["analyze_plan", "extract_budget_allocation"],
                "D4": ["analyze_plan", "assess_financial_sustainability"],
                "D5": ["analyze_plan", "simulate_intervention"],
                "D6": ["analyze_plan", "assess_financial_sustainability"]
            },
            "decologo_processor": {
                "D1": ["process_decologo", "analyze_decologo_alignment"],
                "D2": ["process_decologo", "analyze_decologo_alignment"],
                "D3": ["process_decologo", "analyze_decologo_alignment"],
                "D4": ["process_decologo", "analyze_decologo_alignment"],
                "D5": ["process_decologo", "analyze_decologo_alignment"],
                "D6": ["process_decologo", "analyze_decologo_alignment"]
            },
            "embedding_analyzer": {
                "D1": ["analyze_embeddings", "calculate_semantic_similarity"],
                "D2": ["analyze_embeddings", "calculate_semantic_similarity"],
                "D3": ["analyze_embeddings", "calculate_semantic_similarity"],
                "D4": ["analyze_embeddings", "calculate_semantic_similarity"],
                "D5": ["analyze_embeddings", "calculate_semantic_similarity"],
                "D6": ["analyze_embeddings", "calculate_semantic_similarity"]
            },
            "causal_validator": {
                "D1": ["validate_causal_model", "check_structural_violations"],
                "D2": ["validate_causal_model", "check_structural_violations"],
                "D3": ["validate_causal_model", "check_structural_violations"],
                "D4": ["validate_causal_model", "check_structural_violations"],
                "D5": ["validate_causal_model", "check_structural_violations"],
                "D6": ["validate_causal_model", "check_structural_violations"]
            },
            "modulos_teoria_cambio": {
                "D1": ["construir_grafo_causal", "validacion_completa"],
                "D2": ["construir_grafo_causal", "validacion_completa"],
                "D3": ["construir_grafo_causal", "calculate_acyclicity_pvalue"],
                "D4": ["construir_grafo_causal", "calculate_acyclicity_pvalue"],
                "D5": ["validacion_completa", "calculate_acyclicity_pvalue"],
                "D6": ["validacion_completa", "execute_suite"]
            }
        }

        # Get methods for the module and dimension
        if module_name in method_mappings and dimension in method_mappings[module_name]:
            return method_mappings[module_name][dimension]
        else:
            # Default methods if no specific mapping found
            return ["process"]

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
            if module in CONFIG.modules:
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
                        timeout = CONFIG.modules[module].timeout_seconds if module in CONFIG.modules else 300
                        module_results = future.result(timeout=timeout)
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
        module_config = CONFIG.modules.get(module_name)

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

                # Prepare arguments based on method and module
                args = []
                kwargs = {}

                # Module-specific argument preparation
                if module_name == "contradiction_detector":
                    args = [plan_text]
                    kwargs = {
                        "plan_name": plan_metadata.get("name", "Unknown"),
                        "dimension": dimension
                    }
                elif module_name == "dereck_beach":
                    args = [plan_text, plan_metadata.get("name", "Unknown")]
                    kwargs = {}
                elif module_name == "embedding_policy":
                    if method_name == "chunk_document":
                        args = [plan_text, plan_metadata]
                    elif method_name == "evaluate_policy_metric":
                        # Extract numerical values from previous results
                        args = [[0.5, 0.6, 0.7]]  # Default values
                    elif method_name in ["semantic_search", "rerank"]:
                        # Need query and candidates
                        args = [dimension, []]
                    else:
                        args = [plan_text, plan_metadata]
                    kwargs = {}
                elif module_name == "causal_processor":
                    args = [plan_text]
                    kwargs = {}
                elif module_name == "modulos_teoria_cambio":
                    if method_name == "calculate_acyclicity_pvalue":
                        args = [plan_metadata.get("name", "Unknown"), 10000]
                    elif method_name == "validacion_completa":
                        # Need to construct graph first
                        from .module_adapters import ModuleAdapterRegistry
                        temp_registry = ModuleAdapterRegistry()
                        graph_result = temp_registry.execute_module_method(
                            "modulos_teoria_cambio",
                            "construir_grafo_causal",
                            [], {}
                        )
                        args = [graph_result.data.get("grafo")]
                    else:
                        args = []
                    kwargs = {}
                else:
                    # Default argument preparation
                    args = [plan_text]
                    kwargs = {"plan_metadata": plan_metadata}

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
                    evidence_extracted=self._extract_evidence_from_result(result),
                    confidence=result.confidence,
                    error=result.errors[0] if result.errors else None
                )

                logger.info(f"Completed {component_key} in {result.execution_time:.2f}s with confidence {result.confidence:.2f}")

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

    def _extract_evidence_from_result(self, module_result) -> Dict[str, Any]:
        """
        Extract structured evidence from ModuleResult.

        The new module_adapters.py already provides standardized evidence
        in the ModuleResult.evidence field.
        """
        evidence = {
            "quantitative_claims": [],
            "causal_links": [],
            "contradictions": [],
            "confidence_scores": {},
            "raw_evidence": module_result.evidence
        }

        # Process standardized evidence from ModuleResult
        for ev_item in module_result.evidence:
            if isinstance(ev_item, dict):
                ev_type = ev_item.get("type", "")

                # Extract based on evidence type
                if "contradiction" in ev_type:
                    evidence["contradictions"].extend(ev_item.get("contradictions", []))
                
                if "causal" in ev_type:
                    evidence["causal_links"].append(ev_item)
                
                if any(keyword in ev_type for keyword in ["quantitative", "numerical", "financial", "budget"]):
                    evidence["quantitative_claims"].append(ev_item)
                
                # Extract confidence scores from evidence
                for key, value in ev_item.items():
                    if "confidence" in key.lower() or "score" in key.lower():
                        if isinstance(value, (int, float)):
                            evidence["confidence_scores"][key] = value

        # Add module-level confidence
        evidence["confidence_scores"]["module_confidence"] = module_result.confidence

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