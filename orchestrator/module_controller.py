"""
Module Controller - Dynamic Module Registration and Question Routing
=====================================================================

This module provides centralized control over module adapters, dynamic
registration, question-to-handler mapping, and diagnostic tracing for
the FARFAN 3.0 orchestration framework.

Author: Integration Team
Version: 3.0.0
Python: 3.10+
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class ModuleMetadata:
    """Metadata about a registered module"""

    module_name: str
    adapter_class_name: str
    adapter_instance: Any
    method_count: int
    available_methods: List[str]
    question_coverage: Set[str] = field(default_factory=set)
    specialization: Optional[str] = None
    registration_time: float = field(default_factory=time.time)


@dataclass
class ExecutionTrace:
    """Trace information for diagnostic purposes"""

    trace_id: str
    question_id: str
    module_name: str
    method_name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    input_summary: Dict[str, Any] = field(default_factory=dict)
    output_summary: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ProcessingResult:
    """Result of document processing through module pipeline"""

    question_id: str
    handler_method: str
    module_results: List[Any]
    aggregated_data: Dict[str, Any]
    confidence: float
    execution_time: float
    traces: List[ExecutionTrace]
    status: str
    errors: List[str] = field(default_factory=list)


class ModuleStatus(Enum):
    """Module availability status"""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    UNKNOWN = "unknown"


# ============================================================================
# MODULE CONTROLLER
# ============================================================================


class ModuleController:
    """
    Central controller for module adapters with dynamic registration,
    question routing, and diagnostic tracing capabilities.

    This class manages:
    - Dynamic module registration
    - Question ID to handler method mapping
    - Document processing pipelines
    - Execution tracing
    - Diagnostic reporting
    """

    def __init__(
        self,
        contradiction_detector=None,
        financial_viability_analyzer=None,
        analyzer_one=None,
        policy_processor=None,
        policy_segmenter=None,
        semantic_chunking_adapter=None,
        embedding_policy_adapter=None,
        embedders: Optional[Dict[str, Any]] = None,
        pdf_processor=None,
        cuestionario_path: str = "cuestionario.json",
        responsibility_map_path: str = "orchestrator/execution_mapping.yaml",
    ):
        """
        Initialize ModuleController with dependency injection.

        Args:
            contradiction_detector: ContradictionDetectionAdapter instance
            financial_viability_analyzer: FinancialViabilityAdapter instance
            analyzer_one: AnalyzerOneAdapter instance
            policy_processor: PolicyProcessorAdapter instance
            policy_segmenter: PolicySegmenterAdapter instance
            semantic_chunking_adapter: SemanticChunkingPolicyAdapter instance
            embedding_policy_adapter: EmbeddingPolicyAdapter instance
            embedders: Dictionary of embedder instances from embedding_policy.py
            pdf_processor: PDFProcessor instance (if available)
            cuestionario_path: Path to cuestionario.json
            responsibility_map_path: Path to responsibility_map.json/yaml
        """
        self.logger = logging.getLogger(f"{__name__}.ModuleController")

        # Store injected dependencies
        self.contradiction_detector = contradiction_detector
        self.financial_viability_analyzer = financial_viability_analyzer
        self.analyzer_one = analyzer_one
        self.policy_processor = policy_processor
        self.policy_segmenter = policy_segmenter
        self.semantic_chunking_adapter = semantic_chunking_adapter
        self.embedding_policy_adapter = embedding_policy_adapter
        self.embedders = embedders or {}
        self.pdf_processor = pdf_processor

        # Module registry
        self._modules: Dict[str, ModuleMetadata] = {}

        # Question to handler mapping
        self._question_handlers: Dict[str, Callable] = {}
        self._question_to_module: Dict[str, List[str]] = defaultdict(list)

        # Execution tracing
        self._execution_traces: List[ExecutionTrace] = []
        self._trace_counter = 0

        # Load configuration
        self.cuestionario_path = Path(cuestionario_path)
        self.responsibility_map_path = Path(responsibility_map_path)
        self.cuestionario: Optional[Dict[str, Any]] = None
        self.responsibility_map: Optional[Dict[str, Any]] = None

        # Initialize
        self._load_configurations()
        self._register_default_adapters()
        self._build_question_mappings()

        self.logger.info(
            f"ModuleController initialized with {len(self._modules)} modules"
        )

    # ========================================================================
    # CONFIGURATION LOADING
    # ========================================================================

    def _load_configurations(self):
        """Load cuestionario.json and responsibility_map"""
        try:
            # Load cuestionario
            if self.cuestionario_path.exists():
                with open(self.cuestionario_path, "r", encoding="utf-8") as f:
                    self.cuestionario = json.load(f)
                total_q = self.cuestionario.get("metadata", {}).get(
                    "total_questions", 0
                )
                self.logger.info(f"Loaded cuestionario with {total_q} questions")
            else:
                self.logger.warning(
                    f"Cuestionario not found at {self.cuestionario_path}"
                )
                self.cuestionario = {}

            # Load responsibility map (support both JSON and YAML)
            if self.responsibility_map_path.exists():
                if self.responsibility_map_path.suffix in [".yaml", ".yml"]:
                    with open(self.responsibility_map_path, "r", encoding="utf-8") as f:
                        self.responsibility_map = yaml.safe_load(f)
                else:
                    with open(self.responsibility_map_path, "r", encoding="utf-8") as f:
                        self.responsibility_map = json.load(f)
                self.logger.info(
                    f"Loaded responsibility map from {self.responsibility_map_path}"
                )
            else:
                self.logger.warning(
                    f"Responsibility map not found at {self.responsibility_map_path}"
                )
                self.responsibility_map = {}

        except Exception as e:
            self.logger.error(f"Error loading configurations: {e}", exc_info=True)
            self.cuestionario = {}
            self.responsibility_map = {}

    # ========================================================================
    # MODULE REGISTRATION
    # ========================================================================

    def _register_default_adapters(self):
        """Register all injected adapters"""
        adapters_to_register = [
            (
                "contradiction_detection",
                "ContradictionDetectionAdapter",
                self.contradiction_detector,
            ),
            (
                "financial_viability",
                "FinancialViabilityAdapter",
                self.financial_viability_analyzer,
            ),
            ("analyzer_one", "AnalyzerOneAdapter", self.analyzer_one),
            ("policy_processor", "PolicyProcessorAdapter", self.policy_processor),
            ("policy_segmenter", "PolicySegmenterAdapter", self.policy_segmenter),
            (
                "semantic_chunking_policy",
                "SemanticChunkingPolicyAdapter",
                self.semantic_chunking_adapter,
            ),
            (
                "embedding_policy",
                "EmbeddingPolicyAdapter",
                self.embedding_policy_adapter,
            ),
        ]

        for module_name, adapter_class, adapter_instance in adapters_to_register:
            if adapter_instance is not None:
                self.register_module(module_name, adapter_class, adapter_instance)

    def register_module(
        self,
        module_name: str,
        adapter_class_name: str,
        adapter_instance: Any,
        specialization: Optional[str] = None,
    ) -> bool:
        """
        Dynamically register a module adapter.

        Args:
            module_name: Unique identifier for the module
            adapter_class_name: Name of the adapter class
            adapter_instance: Instance of the adapter
            specialization: Optional description of module specialization

        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Get available methods from adapter
            available_methods = self._extract_available_methods(adapter_instance)

            # Create metadata
            metadata = ModuleMetadata(
                module_name=module_name,
                adapter_class_name=adapter_class_name,
                adapter_instance=adapter_instance,
                method_count=len(available_methods),
                available_methods=available_methods,
                specialization=specialization,
            )

            # Register
            self._modules[module_name] = metadata

            self.logger.info(
                f"Registered module '{module_name}' ({adapter_class_name}) "
                f"with {len(available_methods)} methods"
            )

            return True

        except Exception as e:
            self.logger.error(
                f"Failed to register module '{module_name}': {e}", exc_info=True
            )
            return False

    def _extract_available_methods(self, adapter_instance: Any) -> List[str]:
        """Extract public methods from adapter instance"""
        methods = []

        # Try to get execute method's supported methods first
        if hasattr(adapter_instance, "execute"):
            # Adapters may document their methods in docstrings or attributes
            if hasattr(adapter_instance, "__dict__"):
                # Look for method documentation
                pass

        # Fallback: get all public methods
        for attr_name in dir(adapter_instance):
            if not attr_name.startswith("_") and callable(
                getattr(adapter_instance, attr_name, None)
            ):
                methods.append(attr_name)

        return methods

    def unregister_module(self, module_name: str) -> bool:
        """
        Unregister a module adapter.

        Args:
            module_name: Name of module to unregister

        Returns:
            True if unregistration successful, False otherwise
        """
        if module_name in self._modules:
            del self._modules[module_name]
            self.logger.info(f"Unregistered module '{module_name}'")
            return True
        else:
            self.logger.warning(f"Module '{module_name}' not found for unregistration")
            return False

    # ========================================================================
    # QUESTION MAPPING
    # ========================================================================

    def _build_question_mappings(self):
        """Build mappings from question IDs to handler methods"""
        if not self.responsibility_map:
            self.logger.warning(
                "No responsibility map available for building question mappings"
            )
            return

        # Parse responsibility map structure
        # Expected structure: dimension sections with question entries
        for dimension_key, dimension_data in self.responsibility_map.items():
            if not isinstance(dimension_data, dict):
                continue

            # Skip metadata keys
            if dimension_key in [
                "version",
                "last_updated",
                "total_adapters",
                "total_methods",
                "adapters",
            ]:
                continue

            # Process each question in dimension
            for question_key, question_data in dimension_data.items():
                if not isinstance(question_data, dict):
                    continue

                # Skip non-question keys
                if question_key in ["description", "question_count"]:
                    continue

                # Build handler for this question
                full_question_id = f"{dimension_key}.{question_key}"
                self._register_question_handler(full_question_id, question_data)

        self.logger.info(f"Built {len(self._question_handlers)} question handlers")

    def _register_question_handler(
        self, question_id: str, question_data: Dict[str, Any]
    ):
        """Register handler for a specific question"""
        try:
            execution_chain = question_data.get("execution_chain", [])

            if not execution_chain:
                self.logger.warning(f"No execution chain for question {question_id}")
                return

            # Create handler function
            def handler(document: str, **kwargs) -> ProcessingResult:
                return self._execute_question_chain(
                    question_id, execution_chain, document, **kwargs
                )

            self._question_handlers[question_id] = handler

            # Track module involvement
            for step in execution_chain:
                adapter_name = step.get("adapter")
                if adapter_name:
                    self._question_to_module[question_id].append(adapter_name)

                    # Update module metadata with question coverage
                    if adapter_name in self._modules:
                        self._modules[adapter_name].question_coverage.add(question_id)

        except Exception as e:
            self.logger.error(
                f"Failed to register handler for {question_id}: {e}", exc_info=True
            )

    def map_question_to_handler(self, question_id: str) -> Optional[Callable]:
        """
        Get handler method for a specific question ID.

        Args:
            question_id: Question identifier (e.g., "D1_INSUMOS.Q1_Baseline_Identification")

        Returns:
            Handler callable or None if not found
        """
        return self._question_handlers.get(question_id)

    def get_questions_for_module(self, module_name: str) -> List[str]:
        """
        Get all questions handled by a specific module.

        Args:
            module_name: Name of the module

        Returns:
            List of question IDs
        """
        questions = []
        for question_id, modules in self._question_to_module.items():
            if module_name in modules:
                questions.append(question_id)
        return questions

    # ========================================================================
    # DOCUMENT PROCESSING
    # ========================================================================

    def process_document(
        self,
        question_id: str,
        document: str,
        metadata: Optional[Dict[str, Any]] = None,
        trace_enabled: bool = True,
    ) -> ProcessingResult:
        """
        Process a document through the appropriate module pipeline for a question.

        Args:
            question_id: Question identifier
            document: Document text or path
            metadata: Optional metadata for processing
            trace_enabled: Whether to enable execution tracing

        Returns:
            ProcessingResult with aggregated results and traces
        """
        start_time = time.time()
        metadata = metadata or {}

        # Get handler
        handler = self.map_question_to_handler(question_id)

        if handler is None:
            self.logger.error(f"No handler found for question {question_id}")
            return ProcessingResult(
                question_id=question_id,
                handler_method="unknown",
                module_results=[],
                aggregated_data={},
                confidence=0.0,
                execution_time=time.time() - start_time,
                traces=[],
                status="error",
                errors=[f"No handler found for question {question_id}"],
            )

        # Execute handler
        try:
            result = handler(document, metadata=metadata, trace_enabled=trace_enabled)
            return result

        except Exception as e:
            self.logger.error(
                f"Error processing question {question_id}: {e}", exc_info=True
            )
            return ProcessingResult(
                question_id=question_id,
                handler_method=(
                    handler.__name__ if hasattr(handler, "__name__") else "unknown"
                ),
                module_results=[],
                aggregated_data={},
                confidence=0.0,
                execution_time=time.time() - start_time,
                traces=[],
                status="error",
                errors=[str(e)],
            )

    def _execute_question_chain(
        self,
        question_id: str,
        execution_chain: List[Dict[str, Any]],
        document: str,
        metadata: Optional[Dict[str, Any]] = None,
        trace_enabled: bool = True,
    ) -> ProcessingResult:
        """Execute a multi-step execution chain for a question"""
        start_time = time.time()
        metadata = metadata or {}
        traces = []
        module_results = []
        errors = []

        # Execution context (stores intermediate results)
        context = {
            "plan_text": document,
            "normalized_text": document,
            "metadata": metadata,
        }

        # Execute each step in the chain
        for step_idx, step in enumerate(execution_chain):
            step_num = step.get("step", step_idx + 1)
            adapter_name = step.get("adapter")
            method_name = step.get("method")

            if not adapter_name or not method_name:
                errors.append(
                    f"Step {step_num}: Missing adapter or method specification"
                )
                continue

            # Get adapter instance
            if adapter_name not in self._modules:
                errors.append(
                    f"Step {step_num}: Module '{adapter_name}' not registered"
                )
                continue

            module_metadata = self._modules[adapter_name]
            adapter_instance = module_metadata.adapter_instance

            # Start trace
            trace = None
            if trace_enabled:
                trace = self._start_trace(question_id, adapter_name, method_name, step)
                traces.append(trace)

            try:
                # Prepare arguments from context
                args, kwargs = self._prepare_step_arguments(step, context)

                # Execute method
                result = self._execute_adapter_method(
                    adapter_instance, method_name, args, kwargs
                )

                module_results.append(result)

                # Store result in context for next steps
                returns_binding = step.get("returns", {}).get("binding")
                if returns_binding:
                    context[returns_binding] = self._extract_result_data(result)

                # End trace
                if trace:
                    self._end_trace(trace, "success", result)

            except Exception as e:
                error_msg = f"Step {step_num} ({adapter_name}.{method_name}): {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg, exc_info=True)

                if trace:
                    self._end_trace(trace, "error", None, [str(e)])

        # Aggregate results
        aggregated_data, confidence = self._aggregate_chain_results(
            module_results, execution_chain, errors
        )

        return ProcessingResult(
            question_id=question_id,
            handler_method="execution_chain",
            module_results=module_results,
            aggregated_data=aggregated_data,
            confidence=confidence,
            execution_time=time.time() - start_time,
            traces=traces,
            status=(
                "success" if not errors else "partial" if module_results else "error"
            ),
            errors=errors,
        )

    def _prepare_step_arguments(
        self, step: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Prepare arguments for a step from context"""
        args = []
        kwargs = {}

        step_args = step.get("args", [])

        for arg_spec in step_args:
            if isinstance(arg_spec, dict):
                arg_name = arg_spec.get("name")
                arg_source = arg_spec.get("source")
                arg_value = arg_spec.get("value")

                # Get value from context or use literal value
                if arg_source and arg_source in context:
                    value = context[arg_source]
                elif arg_value is not None:
                    value = arg_value
                else:
                    value = None

                if arg_name:
                    kwargs[arg_name] = value
                else:
                    args.append(value)

        return args, kwargs

    def _execute_adapter_method(
        self,
        adapter_instance: Any,
        method_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
    ) -> Any:
        """Execute a method on an adapter instance"""
        # Most adapters have an 'execute' method that takes method_name
        if hasattr(adapter_instance, "execute"):
            return adapter_instance.execute(method_name, args, kwargs)

        # Direct method call fallback
        if hasattr(adapter_instance, method_name):
            method = getattr(adapter_instance, method_name)
            return method(*args, **kwargs)

        raise AttributeError(f"Adapter has no method '{method_name}' or 'execute'")

    def _extract_result_data(self, result: Any) -> Any:
        """Extract data from module result"""
        # Handle ModuleResult dataclass
        if hasattr(result, "data"):
            return result.data

        # Handle dict result
        if isinstance(result, dict):
            return result

        # Return as-is
        return result

    def _aggregate_chain_results(
        self,
        module_results: List[Any],
        execution_chain: List[Dict[str, Any]],
        errors: List[str],
    ) -> Tuple[Dict[str, Any], float]:
        """Aggregate results from execution chain"""
        aggregated = {
            "step_count": len(execution_chain),
            "successful_steps": len(module_results),
            "failed_steps": len(errors),
            "results": [],
        }

        total_confidence = 0.0
        confidence_count = 0

        for result in module_results:
            if hasattr(result, "data"):
                aggregated["results"].append(result.data)
                if hasattr(result, "confidence"):
                    total_confidence += result.confidence
                    confidence_count += 1
            elif isinstance(result, dict):
                aggregated["results"].append(result)

        # Calculate overall confidence
        if confidence_count > 0:
            confidence = total_confidence / confidence_count
        else:
            confidence = 0.0 if errors else 0.5

        return aggregated, confidence

    # ========================================================================
    # EXECUTION TRACING
    # ========================================================================

    def _start_trace(
        self,
        question_id: str,
        module_name: str,
        method_name: str,
        step_data: Dict[str, Any],
    ) -> ExecutionTrace:
        """Start an execution trace"""
        self._trace_counter += 1
        trace = ExecutionTrace(
            trace_id=f"trace_{self._trace_counter}",
            question_id=question_id,
            module_name=module_name,
            method_name=method_name,
            start_time=time.time(),
            status="running",
            input_summary=self._summarize_step_input(step_data),
        )
        self._execution_traces.append(trace)
        return trace

    def _end_trace(
        self,
        trace: ExecutionTrace,
        status: str,
        result: Any,
        errors: Optional[List[str]] = None,
    ):
        """End an execution trace"""
        trace.end_time = time.time()
        trace.status = status

        if result:
            trace.output_summary = self._summarize_result(result)

        if errors:
            trace.errors.extend(errors)

    def _summarize_step_input(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of step input for tracing"""
        return {
            "step": step_data.get("step"),
            "adapter": step_data.get("adapter"),
            "method": step_data.get("method"),
            "purpose": step_data.get("purpose", ""),
        }

    def _summarize_result(self, result: Any) -> Dict[str, Any]:
        """Create summary of result for tracing"""
        if hasattr(result, "__dict__"):
            return {k: str(v)[:100] for k, v in result.__dict__.items()}
        elif isinstance(result, dict):
            return {k: str(v)[:100] for k, v in result.items()}
        else:
            return {"result": str(result)[:100]}

    def get_execution_traces(
        self, question_id: Optional[str] = None, limit: int = 100
    ) -> List[ExecutionTrace]:
        """
        Get execution traces for diagnostic purposes.

        Args:
            question_id: Optional filter by question ID
            limit: Maximum number of traces to return

        Returns:
            List of ExecutionTrace objects
        """
        traces = self._execution_traces[-limit:]

        if question_id:
            traces = [t for t in traces if t.question_id == question_id]

        return traces

    def clear_traces(self):
        """Clear all execution traces"""
        self._execution_traces.clear()
        self._trace_counter = 0
        self.logger.info("Cleared all execution traces")

    # ========================================================================
    # DIAGNOSTICS
    # ========================================================================

    def get_module_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about registered modules.

        Returns:
            Dictionary with module statistics and status
        """
        diagnostics = {
            "total_modules": len(self._modules),
            "total_questions": len(self._question_handlers),
            "modules": {},
            "question_coverage": {},
        }

        for module_name, metadata in self._modules.items():
            diagnostics["modules"][module_name] = {
                "adapter_class": metadata.adapter_class_name,
                "method_count": metadata.method_count,
                "available_methods": metadata.available_methods[
                    :10
                ],  # Limit for readability
                "question_coverage_count": len(metadata.question_coverage),
                "questions": list(metadata.question_coverage)[:5],  # Sample
                "specialization": metadata.specialization,
                "status": self._get_module_status(metadata),
            }

        # Question coverage statistics
        for question_id, modules in self._question_to_module.items():
            diagnostics["question_coverage"][question_id] = {
                "module_count": len(modules),
                "modules": modules,
            }

        return diagnostics

    def _get_module_status(self, metadata: ModuleMetadata) -> str:
        """Determine module status"""
        if metadata.adapter_instance is None:
            return ModuleStatus.UNAVAILABLE.value

        # Check if adapter reports availability
        if hasattr(metadata.adapter_instance, "available"):
            return (
                ModuleStatus.AVAILABLE.value
                if metadata.adapter_instance.available
                else ModuleStatus.UNAVAILABLE.value
            )

        return ModuleStatus.AVAILABLE.value

    def get_question_coverage_report(self) -> Dict[str, Any]:
        """
        Get report on question coverage by modules.

        Returns:
            Dictionary with coverage statistics
        """
        report = {
            "total_questions": len(self._question_handlers),
            "questions_with_handlers": sum(
                1 for q in self._question_to_module.values() if q
            ),
            "questions_without_handlers": sum(
                1
                for q in self._question_handlers.keys()
                if q not in self._question_to_module
            ),
            "average_modules_per_question": 0.0,
            "questions_by_module_count": defaultdict(int),
            "detailed_coverage": {},
        }

        # Calculate statistics
        if self._question_to_module:
            total_modules = sum(
                len(modules) for modules in self._question_to_module.values()
            )
            report["average_modules_per_question"] = total_modules / len(
                self._question_to_module
            )

            for question_id, modules in self._question_to_module.items():
                module_count = len(modules)
                report["questions_by_module_count"][module_count] += 1
                report["detailed_coverage"][question_id] = {
                    "module_count": module_count,
                    "modules": modules,
                }

        return report

    def export_diagnostics(self, output_path: str) -> bool:
        """
        Export diagnostics to JSON file.

        Args:
            output_path: Path to output file

        Returns:
            True if export successful
        """
        try:
            diagnostics = {
                "module_diagnostics": self.get_module_diagnostics(),
                "question_coverage": self.get_question_coverage_report(),
                "recent_traces": [
                    {
                        "trace_id": t.trace_id,
                        "question_id": t.question_id,
                        "module": t.module_name,
                        "method": t.method_name,
                        "status": t.status,
                        "duration": t.end_time - t.start_time if t.end_time else None,
                    }
                    for t in self._execution_traces[-50:]
                ],
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(diagnostics, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Exported diagnostics to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export diagnostics: {e}", exc_info=True)
            return False
