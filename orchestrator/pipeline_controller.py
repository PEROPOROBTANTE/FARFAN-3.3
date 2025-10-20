# coding=utf-8
"""
Module Controller - Deterministic Pipeline Execution with Typed Contracts
=========================================================================

Enforces canonical deterministic execution path through:
- PipelineStage enum sequencing all modules in required order
- Frozen dataclasses for immutable typed input/output contracts
- Runtime validation preventing out-of-sequence execution
- FlowComposition declaring complete pipeline topology

Pipeline Stages:
1. PDF_PROCESSING: Document loading and text extraction
2. SEMANTIC_CHUNKING: Semantic boundary detection and chunking
3. EMBEDDING_GENERATION: Vector embeddings for semantic search
4. POLICY_ANALYSIS: Theory of change and municipal development analysis
5. CONTRADICTION_DETECTION: Logical contradiction identification
6. FINANCIAL_VIABILITY: Financial feasibility assessment
7. REPORTING: MICRO/MESO/MACRO report assembly

Author: FARFAN 3.0 Team
Version: 3.0.0
Python: 3.10+
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Canonical pipeline stages in deterministic execution order"""

    PDF_PROCESSING = 1
    SEMANTIC_CHUNKING = 2
    EMBEDDING_GENERATION = 3
    POLICY_ANALYSIS = 4
    CONTRADICTION_DETECTION = 5
    FINANCIAL_VIABILITY = 6
    REPORTING = 7

    def __lt__(self, other):
        if not isinstance(other, PipelineStage):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other):
        if not isinstance(other, PipelineStage):
            return NotImplemented
        return self.value <= other.value


# ============================================================================
# STAGE INPUT/OUTPUT CONTRACTS (Frozen Dataclasses for Immutability)
# ============================================================================


@dataclass(frozen=True)
class PDFProcessingInput:
    """Input contract for PDF processing stage"""

    plan_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PDFProcessingOutput:
    """Output contract for PDF processing stage"""

    plan_text: str
    plan_path: Path
    document_metadata: Dict[str, Any]
    char_count: int
    page_count: Optional[int] = None


@dataclass(frozen=True)
class SemanticChunkingInput:
    """Input contract for semantic chunking stage"""

    plan_text: str
    document_metadata: Dict[str, Any]
    chunk_size: int = 512
    overlap: int = 50


@dataclass(frozen=True)
class SemanticChunkingOutput:
    """Output contract for semantic chunking stage"""

    semantic_chunks: List[Dict[str, Any]]
    chunk_count: int
    chunking_metadata: Dict[str, Any]


@dataclass(frozen=True)
class EmbeddingGenerationInput:
    """Input contract for embedding generation stage"""

    semantic_chunks: List[Dict[str, Any]]
    plan_text: str
    model_name: str = "default"


@dataclass(frozen=True)
class EmbeddingGenerationOutput:
    """Output contract for embedding generation stage"""

    embeddings: List[List[float]]
    chunks_with_embeddings: List[Dict[str, Any]]
    embedding_dimension: int
    model_used: str


@dataclass(frozen=True)
class PolicyAnalysisInput:
    """Input contract for policy analysis stage"""

    plan_text: str
    semantic_chunks: List[Dict[str, Any]]
    embeddings: List[List[float]]
    question_spec: Any


@dataclass(frozen=True)
class PolicyAnalysisOutput:
    """Output contract for policy analysis stage"""

    analyzer_one_results: Dict[str, Any]
    teoria_cambio_results: Dict[str, Any]
    analysis_metadata: Dict[str, Any]
    confidence_scores: Dict[str, float]


@dataclass(frozen=True)
class ContradictionDetectionInput:
    """Input contract for contradiction detection stage"""

    plan_text: str
    policy_analysis_results: Dict[str, Any]
    semantic_chunks: List[Dict[str, Any]]


@dataclass(frozen=True)
class ContradictionDetectionOutput:
    """Output contract for contradiction detection stage"""

    contradictions: List[Dict[str, Any]]
    contradiction_count: int
    severity_scores: Dict[str, float]
    detection_metadata: Dict[str, Any]


@dataclass(frozen=True)
class FinancialViabilityInput:
    """Input contract for financial viability stage"""

    plan_text: str
    policy_analysis_results: Dict[str, Any]
    contradiction_results: Dict[str, Any]
    semantic_chunks: List[Dict[str, Any]]


@dataclass(frozen=True)
class FinancialViabilityOutput:
    """Output contract for financial viability stage"""

    viability_score: float
    financial_metrics: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    viability_metadata: Dict[str, Any]


@dataclass(frozen=True)
class ReportingInput:
    """Input contract for reporting stage"""

    plan_text: str
    policy_analysis_results: Dict[str, Any]
    contradiction_results: Dict[str, Any]
    financial_viability_results: Dict[str, Any]
    question_spec: Any


@dataclass(frozen=True)
class ReportingOutput:
    """Output contract for reporting stage"""

    micro_answer: Dict[str, Any]
    evidence_collected: List[Dict[str, Any]]
    confidence_score: float
    report_metadata: Dict[str, Any]


# ============================================================================
# EXCEPTIONS
# ============================================================================


class ContractViolationError(Exception):
    """Raised when stage input/output contract is violated"""

    def __init__(
        self,
        stage: PipelineStage,
        message: str,
        missing_fields: Optional[List[str]] = None,
    ):
        self.stage = stage
        self.missing_fields = missing_fields or []
        super().__init__(f"Contract violation in {stage.name}: {message}")


class StageExecutionError(Exception):
    """Raised when stage execution fails"""

    def __init__(
        self, stage: PipelineStage, message: str, cause: Optional[Exception] = None
    ):
        self.stage = stage
        self.cause = cause
        super().__init__(f"Execution error in {stage.name}: {message}")


class OutOfSequenceError(Exception):
    """Raised when attempting to execute stage out of sequence"""

    def __init__(
        self, attempted_stage: PipelineStage, current_stage: Optional[PipelineStage]
    ):
        self.attempted_stage = attempted_stage
        self.current_stage = current_stage
        super().__init__(
            f"Cannot execute {attempted_stage.name}: "
            f"current stage is {current_stage.name if current_stage else 'NONE'}"
        )


# ============================================================================
# FLOW COMPOSITION - Pipeline Topology Declaration
# ============================================================================


class FlowComposition:
    """
    Declares complete pipeline topology including stage dependencies,
    data transformations, and flow validation
    """

    def __init__(self):
        """Initialize flow composition with canonical pipeline topology"""
        self._stage_dependencies: Dict[PipelineStage, Set[PipelineStage]] = {
            PipelineStage.PDF_PROCESSING: set(),
            PipelineStage.SEMANTIC_CHUNKING: {PipelineStage.PDF_PROCESSING},
            PipelineStage.EMBEDDING_GENERATION: {PipelineStage.SEMANTIC_CHUNKING},
            PipelineStage.POLICY_ANALYSIS: {
                PipelineStage.SEMANTIC_CHUNKING,
                PipelineStage.EMBEDDING_GENERATION,
            },
            PipelineStage.CONTRADICTION_DETECTION: {PipelineStage.POLICY_ANALYSIS},
            PipelineStage.FINANCIAL_VIABILITY: {
                PipelineStage.POLICY_ANALYSIS,
                PipelineStage.CONTRADICTION_DETECTION,
            },
            PipelineStage.REPORTING: {
                PipelineStage.POLICY_ANALYSIS,
                PipelineStage.CONTRADICTION_DETECTION,
                PipelineStage.FINANCIAL_VIABILITY,
            },
        }

        self._stage_contracts: Dict[PipelineStage, Tuple[type, type]] = {
            PipelineStage.PDF_PROCESSING: (PDFProcessingInput, PDFProcessingOutput),
            PipelineStage.SEMANTIC_CHUNKING: (
                SemanticChunkingInput,
                SemanticChunkingOutput,
            ),
            PipelineStage.EMBEDDING_GENERATION: (
                EmbeddingGenerationInput,
                EmbeddingGenerationOutput,
            ),
            PipelineStage.POLICY_ANALYSIS: (PolicyAnalysisInput, PolicyAnalysisOutput),
            PipelineStage.CONTRADICTION_DETECTION: (
                ContradictionDetectionInput,
                ContradictionDetectionOutput,
            ),
            PipelineStage.FINANCIAL_VIABILITY: (
                FinancialViabilityInput,
                FinancialViabilityOutput,
            ),
            PipelineStage.REPORTING: (ReportingInput, ReportingOutput),
        }

        logger.info("FlowComposition initialized with canonical pipeline topology")

    def get_dependencies(self, stage: PipelineStage) -> Set[PipelineStage]:
        """Get set of stages that must complete before given stage"""
        return self._stage_dependencies.get(stage, set())

    def get_input_contract(self, stage: PipelineStage) -> type:
        """Get input contract dataclass type for stage"""
        contracts = self._stage_contracts.get(stage)
        if not contracts:
            raise ValueError(f"No contract defined for stage {stage.name}")
        return contracts[0]

    def get_output_contract(self, stage: PipelineStage) -> type:
        """Get output contract dataclass type for stage"""
        contracts = self._stage_contracts.get(stage)
        if not contracts:
            raise ValueError(f"No contract defined for stage {stage.name}")
        return contracts[1]

    def validate_flow_graph(self) -> bool:
        """
        Validate complete flow graph for completeness and detect missing implementations

        Returns:
            True if flow graph is valid

        Raises:
            ValueError: If flow graph has issues
        """
        logger.info("Validating flow graph completeness")

        all_stages = set(PipelineStage)
        defined_stages = set(self._stage_dependencies.keys())

        if all_stages != defined_stages:
            missing = all_stages - defined_stages
            raise ValueError(
                f"Missing stage dependencies for: {[s.name for s in missing]}"
            )

        contract_stages = set(self._stage_contracts.keys())
        if all_stages != contract_stages:
            missing = all_stages - contract_stages
            raise ValueError(
                f"Missing contracts for stages: {[s.name for s in missing]}"
            )

        for stage, deps in self._stage_dependencies.items():
            for dep in deps:
                if dep not in all_stages:
                    raise ValueError(f"Invalid dependency: {stage.name} -> {dep.name}")

        if not self._is_acyclic():
            raise ValueError("Flow graph contains cycles")

        logger.info("Flow graph validation successful")
        return True

    def _is_acyclic(self) -> bool:
        """Check if dependency graph is acyclic (DAG)"""
        visited = set()
        rec_stack = set()

        def has_cycle(stage: PipelineStage) -> bool:
            visited.add(stage)
            rec_stack.add(stage)

            for dep in self._stage_dependencies.get(stage, set()):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(stage)
            return False

        for stage in PipelineStage:
            if stage not in visited:
                if has_cycle(stage):
                    return False

        return True

    def get_execution_order(self) -> List[PipelineStage]:
        """
        Get topologically sorted execution order for all stages

        Returns:
            List of stages in dependency-respecting order
        """
        return sorted(list(PipelineStage), key=lambda s: s.value)

    def can_execute(
        self, stage: PipelineStage, completed_stages: Set[PipelineStage]
    ) -> bool:
        """
        Check if stage can execute given completed stages

        Args:
            stage: Stage to check
            completed_stages: Set of already completed stages

        Returns:
            True if all dependencies are met
        """
        required_deps = self.get_dependencies(stage)
        return required_deps.issubset(completed_stages)

    def get_stage_feeds_into(self, stage: PipelineStage) -> Set[PipelineStage]:
        """
        Get set of stages that depend on given stage

        Args:
            stage: Source stage

        Returns:
            Set of stages that require this stage's output
        """
        dependents = set()
        for dependent_stage, deps in self._stage_dependencies.items():
            if stage in deps:
                dependents.add(dependent_stage)
        return dependents


# ============================================================================
# MODULE CONTROLLER - Deterministic Execution Engine
# ============================================================================


class ModuleController:
    """
    Enforces canonical deterministic execution path with typed contracts

    Features:
    - Stage-based execution with enforced sequencing
    - Runtime contract validation for inputs/outputs
    - Module registration with stage mapping
    - Out-of-sequence execution prevention
    - Type-safe data flow between stages
    """

    def __init__(self):
        """Initialize module controller with flow composition"""
        self.flow = FlowComposition()
        self.flow.validate_flow_graph()

        self._registered_modules: Dict[PipelineStage, Callable] = {}
        self._completed_stages: Set[PipelineStage] = set()
        self._stage_outputs: Dict[PipelineStage, Any] = {}
        self._current_stage: Optional[PipelineStage] = None

        logger.info("ModuleController initialized with validated flow topology")

    def register_module(self, stage: PipelineStage, executor: Callable) -> None:
        """
        Register module executor for a pipeline stage

        Args:
            stage: Pipeline stage this module handles
            executor: Callable that executes the stage logic

        Raises:
            ValueError: If stage already registered
        """
        if stage in self._registered_modules:
            raise ValueError(f"Module already registered for stage {stage.name}")

        self._registered_modules[stage] = executor
        logger.info(f"Registered module for stage {stage.name}")

    def unregister_module(self, stage: PipelineStage) -> None:
        """Unregister module for a stage"""
        if stage in self._registered_modules:
            del self._registered_modules[stage]
            logger.info(f"Unregistered module for stage {stage.name}")

    def validate_input_contract(self, stage: PipelineStage, input_data: Any) -> None:
        """
        Validate input data matches stage's input contract

        Args:
            stage: Pipeline stage
            input_data: Input data to validate

        Raises:
            ContractViolationError: If input doesn't match contract
        """
        input_contract = self.flow.get_input_contract(stage)

        if not isinstance(input_data, input_contract):
            raise ContractViolationError(
                stage,
                f"Input must be instance of {input_contract.__name__}, got {type(input_data).__name__}",
            )

        missing_fields = []
        for field_name in input_contract.__dataclass_fields__:
            if not hasattr(input_data, field_name):
                missing_fields.append(field_name)

        if missing_fields:
            raise ContractViolationError(
                stage,
                f"Input missing required fields: {missing_fields}",
                missing_fields,
            )

        logger.debug(f"Input contract validated for {stage.name}")

    def validate_output_contract(self, stage: PipelineStage, output_data: Any) -> None:
        """
        Validate output data matches stage's output contract

        Args:
            stage: Pipeline stage
            output_data: Output data to validate

        Raises:
            ContractViolationError: If output doesn't match contract
        """
        output_contract = self.flow.get_output_contract(stage)

        if not isinstance(output_data, output_contract):
            raise ContractViolationError(
                stage,
                f"Output must be instance of {output_contract.__name__}, got {type(output_data).__name__}",
            )

        missing_fields = []
        for field_name in output_contract.__dataclass_fields__:
            if not hasattr(output_data, field_name):
                missing_fields.append(field_name)

        if missing_fields:
            raise ContractViolationError(
                stage,
                f"Output missing required fields: {missing_fields}",
                missing_fields,
            )

        logger.debug(f"Output contract validated for {stage.name}")

    def can_execute_stage(self, stage: PipelineStage) -> bool:
        """
        Check if stage can be executed based on completed dependencies

        Args:
            stage: Stage to check

        Returns:
            True if all dependencies are met
        """
        return self.flow.can_execute(stage, self._completed_stages)

    def execute_stage(self, stage: PipelineStage, input_data: Any) -> Any:
        """
        Execute pipeline stage with contract validation and sequencing enforcement

        Args:
            stage: Pipeline stage to execute
            input_data: Input data (must match stage's input contract)

        Returns:
            Output data (guaranteed to match stage's output contract)

        Raises:
            OutOfSequenceError: If dependencies not met
            ContractViolationError: If input/output contracts violated
            StageExecutionError: If stage execution fails
        """
        logger.info(f"Attempting to execute stage: {stage.name}")

        if not self.can_execute_stage(stage):
            raise OutOfSequenceError(stage, self._current_stage)

        if stage not in self._registered_modules:
            raise StageExecutionError(
                stage, f"No module registered for stage {stage.name}"
            )

        self.validate_input_contract(stage, input_data)

        self._current_stage = stage

        try:
            executor = self._registered_modules[stage]
            output_data = executor(input_data)

            self.validate_output_contract(stage, output_data)

            self._completed_stages.add(stage)
            self._stage_outputs[stage] = output_data

            logger.info(f"Successfully completed stage: {stage.name}")

            return output_data

        except Exception as e:
            logger.error(f"Stage execution failed for {stage.name}: {e}", exc_info=True)
            raise StageExecutionError(stage, str(e), e)
        finally:
            self._current_stage = None

    def get_stage_output(self, stage: PipelineStage) -> Optional[Any]:
        """Get output from a completed stage"""
        return self._stage_outputs.get(stage)

    def is_stage_complete(self, stage: PipelineStage) -> bool:
        """Check if stage has been completed"""
        return stage in self._completed_stages

    def get_completed_stages(self) -> Set[PipelineStage]:
        """Get set of all completed stages"""
        return self._completed_stages.copy()

    def get_next_executable_stages(self) -> List[PipelineStage]:
        """
        Get list of stages that can be executed next

        Returns:
            List of stages with all dependencies met
        """
        executable = []
        for stage in PipelineStage:
            if stage not in self._completed_stages and self.can_execute_stage(stage):
                executable.append(stage)
        return sorted(executable)

    def reset(self) -> None:
        """Reset controller state (clears completed stages and outputs)"""
        self._completed_stages.clear()
        self._stage_outputs.clear()
        self._current_stage = None
        logger.info("ModuleController state reset")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline execution status

        Returns:
            Dictionary with current state, completed stages, progress
        """
        total_stages = len(PipelineStage)
        completed_count = len(self._completed_stages)

        return {
            "total_stages": total_stages,
            "completed_stages": completed_count,
            "progress_percent": (completed_count / total_stages) * 100,
            "current_stage": self._current_stage.name if self._current_stage else None,
            "completed_stage_names": [s.name for s in sorted(self._completed_stages)],
            "next_executable": [s.name for s in self.get_next_executable_stages()],
            "registered_modules": [s.name for s in self._registered_modules.keys()],
        }

    def validate_complete_pipeline(self) -> bool:
        """
        Validate that all stages have registered modules

        Returns:
            True if all stages have modules registered

        Raises:
            ValueError: If any stages missing modules
        """
        all_stages = set(PipelineStage)
        registered_stages = set(self._registered_modules.keys())

        if all_stages != registered_stages:
            missing = all_stages - registered_stages
            raise ValueError(
                f"Incomplete pipeline: missing modules for {[s.name for s in missing]}"
            )

        logger.info("Complete pipeline validation successful")
        return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "PipelineStage",
    "PDFProcessingInput",
    "PDFProcessingOutput",
    "SemanticChunkingInput",
    "SemanticChunkingOutput",
    "EmbeddingGenerationInput",
    "EmbeddingGenerationOutput",
    "PolicyAnalysisInput",
    "PolicyAnalysisOutput",
    "ContradictionDetectionInput",
    "ContradictionDetectionOutput",
    "FinancialViabilityInput",
    "FinancialViabilityOutput",
    "ReportingInput",
    "ReportingOutput",
    "ContractViolationError",
    "StageExecutionError",
    "OutOfSequenceError",
    "FlowComposition",
    "ModuleController",
]
