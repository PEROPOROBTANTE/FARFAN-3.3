"""
Module Controller - Canonical Pipeline Execution with Contract Enforcement
===========================================================================

CORE RESPONSIBILITY: Enforce deterministic execution path with typed contracts
-------------------------------------------------------------------------------
Defines canonical pipeline stages, typed input/output contracts, and runtime
validation to ensure modules execute in correct order with valid data.

PIPELINE STAGES (Canonical Execution Order):
---------------------------------------------
1. PDF_PROCESSING: Document loading and text extraction
2. SEMANTIC_CHUNKING: Segment text into semantic units
3. EMBEDDING_GENERATION: Generate vector embeddings for chunks
4. POLICY_ANALYSIS: Analyze policy content and alignment
5. CONTRADICTION_DETECTION: Detect internal contradictions
6. FINANCIAL_VIABILITY: Assess financial feasibility
7. REPORTING: Generate final reports and outputs

CONTRACT ENFORCEMENT:
---------------------
- Each stage has frozen dataclass defining exact input/output fields
- Runtime validation ensures outputs match expected contracts
- Typed exceptions raised for contract violations
- Immutability guarantees prevent data corruption between stages

FLOW TOPOLOGY:
--------------
- FlowComposition declares complete pipeline graph
- Validates flow completeness before execution
- Detects missing stage implementations
- Enforces sequential dependencies

Author: FARFAN 3.0 Team
Version: 3.0.0
Python: 3.10+
"""

import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Callable, Type
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """
    Canonical execution stages in required order
    
    Each stage must complete before next stage can begin.
    Stages are executed sequentially with validated contracts.
    """
    PDF_PROCESSING = auto()
    SEMANTIC_CHUNKING = auto()
    EMBEDDING_GENERATION = auto()
    POLICY_ANALYSIS = auto()
    CONTRADICTION_DETECTION = auto()
    FINANCIAL_VIABILITY = auto()
    REPORTING = auto()
    
    def __lt__(self, other):
        """Enable stage ordering comparison"""
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
    @classmethod
    def get_execution_order(cls) -> List['PipelineStage']:
        """Return stages in canonical execution order"""
        return [
            cls.PDF_PROCESSING,
            cls.SEMANTIC_CHUNKING,
            cls.EMBEDDING_GENERATION,
            cls.POLICY_ANALYSIS,
            cls.CONTRADICTION_DETECTION,
            cls.FINANCIAL_VIABILITY,
            cls.REPORTING
        ]


@dataclass(frozen=True)
class PDFProcessingOutput:
    """
    Immutable contract for PDF processing stage output
    
    Fields:
        plan_text: Extracted text content from document
        page_count: Number of pages processed
        metadata: Document metadata (title, author, etc.)
        extraction_method: Method used (PyPDF2, pdfplumber, etc.)
    """
    plan_text: str
    page_count: int
    metadata: Dict[str, Any]
    extraction_method: str
    
    def __post_init__(self):
        """Validate contract requirements"""
        if not self.plan_text or len(self.plan_text) < 100:
            raise ContractViolationError(
                stage=PipelineStage.PDF_PROCESSING,
                field="plan_text",
                message="Plan text must be at least 100 characters"
            )
        if self.page_count < 1:
            raise ContractViolationError(
                stage=PipelineStage.PDF_PROCESSING,
                field="page_count",
                message="Page count must be at least 1"
            )


@dataclass(frozen=True)
class SemanticChunkingOutput:
    """
    Immutable contract for semantic chunking stage output
    
    Fields:
        chunks: List of semantic text chunks
        chunk_metadata: Metadata for each chunk (position, length, etc.)
        chunking_strategy: Strategy used (sentence, paragraph, semantic)
        total_chunks: Total number of chunks created
    """
    chunks: tuple  # Using tuple for immutability
    chunk_metadata: tuple
    chunking_strategy: str
    total_chunks: int
    
    def __post_init__(self):
        """Validate contract requirements"""
        if self.total_chunks != len(self.chunks):
            raise ContractViolationError(
                stage=PipelineStage.SEMANTIC_CHUNKING,
                field="total_chunks",
                message=f"total_chunks ({self.total_chunks}) must match chunks length ({len(self.chunks)})"
            )
        if self.total_chunks < 1:
            raise ContractViolationError(
                stage=PipelineStage.SEMANTIC_CHUNKING,
                field="total_chunks",
                message="Must produce at least 1 chunk"
            )


@dataclass(frozen=True)
class EmbeddingGenerationOutput:
    """
    Immutable contract for embedding generation stage output
    
    Fields:
        embeddings: Vector embeddings for chunks
        embedding_model: Model used (sentence-transformers, OpenAI, etc.)
        embedding_dimension: Dimensionality of vectors
        chunk_indices: Indices mapping embeddings to chunks
    """
    embeddings: tuple  # Tuple of vectors for immutability
    embedding_model: str
    embedding_dimension: int
    chunk_indices: tuple
    
    def __post_init__(self):
        """Validate contract requirements"""
        if len(self.embeddings) != len(self.chunk_indices):
            raise ContractViolationError(
                stage=PipelineStage.EMBEDDING_GENERATION,
                field="embeddings",
                message="Number of embeddings must match chunk_indices"
            )
        if self.embedding_dimension < 1:
            raise ContractViolationError(
                stage=PipelineStage.EMBEDDING_GENERATION,
                field="embedding_dimension",
                message="Embedding dimension must be positive"
            )


@dataclass(frozen=True)
class PolicyAnalysisOutput:
    """
    Immutable contract for policy analysis stage output
    
    Fields:
        policy_segments: Identified policy segments
        alignment_scores: Alignment with frameworks (Decálogo, SDGs)
        policy_areas: Categorization into P1-P10 areas
        dimensions: Analysis across D1-D6 dimensions
    """
    policy_segments: tuple
    alignment_scores: Dict[str, float]
    policy_areas: Dict[str, Any]
    dimensions: Dict[str, Any]
    
    def __post_init__(self):
        """Validate contract requirements"""
        if not self.policy_segments:
            raise ContractViolationError(
                stage=PipelineStage.POLICY_ANALYSIS,
                field="policy_segments",
                message="Must identify at least one policy segment"
            )
        required_dimensions = {"D1", "D2", "D3", "D4", "D5", "D6"}
        if not required_dimensions.issubset(self.dimensions.keys()):
            raise ContractViolationError(
                stage=PipelineStage.POLICY_ANALYSIS,
                field="dimensions",
                message=f"Must include all dimensions: {required_dimensions}"
            )


@dataclass(frozen=True)
class ContradictionDetectionOutput:
    """
    Immutable contract for contradiction detection stage output
    
    Fields:
        contradictions: List of detected contradictions
        contradiction_scores: Severity scores for each contradiction
        contradiction_pairs: Pairs of contradicting segments
        total_contradictions: Total number detected
    """
    contradictions: tuple
    contradiction_scores: tuple
    contradiction_pairs: tuple
    total_contradictions: int
    
    def __post_init__(self):
        """Validate contract requirements"""
        if self.total_contradictions != len(self.contradictions):
            raise ContractViolationError(
                stage=PipelineStage.CONTRADICTION_DETECTION,
                field="total_contradictions",
                message="total_contradictions must match contradictions length"
            )
        if len(self.contradiction_scores) != len(self.contradictions):
            raise ContractViolationError(
                stage=PipelineStage.CONTRADICTION_DETECTION,
                field="contradiction_scores",
                message="Must have score for each contradiction"
            )


@dataclass(frozen=True)
class FinancialViabilityOutput:
    """
    Immutable contract for financial viability stage output
    
    Fields:
        viability_score: Overall financial viability score (0-100)
        budget_analysis: Budget adequacy and allocation analysis
        financial_risks: Identified financial risks
        recommendations: Financial recommendations
    """
    viability_score: float
    budget_analysis: Dict[str, Any]
    financial_risks: tuple
    recommendations: tuple
    
    def __post_init__(self):
        """Validate contract requirements"""
        if not 0 <= self.viability_score <= 100:
            raise ContractViolationError(
                stage=PipelineStage.FINANCIAL_VIABILITY,
                field="viability_score",
                message="Viability score must be between 0 and 100"
            )
        if not self.budget_analysis:
            raise ContractViolationError(
                stage=PipelineStage.FINANCIAL_VIABILITY,
                field="budget_analysis",
                message="Budget analysis cannot be empty"
            )


@dataclass(frozen=True)
class ReportingOutput:
    """
    Immutable contract for reporting stage output
    
    Fields:
        micro_report: Question-level report
        meso_report: Cluster-level report
        macro_report: Plan-level report
        report_path: Path to saved report
    """
    micro_report: Dict[str, Any]
    meso_report: Dict[str, Any]
    macro_report: Dict[str, Any]
    report_path: str
    
    def __post_init__(self):
        """Validate contract requirements"""
        if not self.micro_report:
            raise ContractViolationError(
                stage=PipelineStage.REPORTING,
                field="micro_report",
                message="Micro report cannot be empty"
            )
        if not Path(self.report_path).suffix == '.json':
            raise ContractViolationError(
                stage=PipelineStage.REPORTING,
                field="report_path",
                message="Report path must have .json extension"
            )


class ContractViolationError(Exception):
    """
    Exception raised when stage output violates contract
    
    Provides detailed information about which stage, field, and constraint failed
    """
    
    def __init__(self, stage: PipelineStage, field: str, message: str):
        self.stage = stage
        self.field = field
        self.message = message
        super().__init__(
            f"Contract violation in {stage.name} stage, field '{field}': {message}"
        )


class StageSequenceError(Exception):
    """
    Exception raised when stages execute out of order
    
    Ensures canonical pipeline sequence is followed
    """
    
    def __init__(self, attempted_stage: PipelineStage, expected_stage: PipelineStage):
        self.attempted_stage = attempted_stage
        self.expected_stage = expected_stage
        super().__init__(
            f"Stage sequence violation: attempted {attempted_stage.name} "
            f"but expected {expected_stage.name}"
        )


class MissingStageImplementationError(Exception):
    """
    Exception raised when pipeline stage lacks implementation
    
    Raised during flow validation if stage has no registered module
    """
    
    def __init__(self, stage: PipelineStage):
        self.stage = stage
        super().__init__(f"No implementation registered for stage: {stage.name}")


@dataclass
class StageContract:
    """
    Contract definition for a pipeline stage
    
    Defines input requirements, output type, and validation logic
    """
    stage: PipelineStage
    input_type: Optional[Type]
    output_type: Type
    module_name: str
    method_name: str
    validator: Optional[Callable[[Any], bool]] = None
    
    def validate_output(self, output: Any) -> bool:
        """
        Validate that output matches contract
        
        Args:
            output: Output data to validate
            
        Returns:
            True if valid, raises exception otherwise
        """
        if not isinstance(output, self.output_type):
            raise ContractViolationError(
                stage=self.stage,
                field="__type__",
                message=f"Expected {self.output_type.__name__}, got {type(output).__name__}"
            )
        
        if self.validator and not self.validator(output):
            raise ContractViolationError(
                stage=self.stage,
                field="__custom_validation__",
                message="Custom validation failed"
            )
        
        return True


class FlowComposition:
    """
    Declares complete pipeline topology and validates flow graph
    
    Manages:
    - Stage-to-stage data flow
    - Module-to-stage mapping
    - Data transformations at boundaries
    - Flow completeness validation
    """
    
    def __init__(self):
        """Initialize flow composition with empty topology"""
        self.stage_contracts: Dict[PipelineStage, StageContract] = {}
        self.stage_dependencies: Dict[PipelineStage, List[PipelineStage]] = {}
        self.data_transformations: Dict[tuple, Callable] = {}
        
        self._initialize_default_topology()
        
        logger.info("FlowComposition initialized")
    
    def _initialize_default_topology(self):
        """Initialize default FARFAN pipeline topology"""
        
        self.stage_contracts = {
            PipelineStage.PDF_PROCESSING: StageContract(
                stage=PipelineStage.PDF_PROCESSING,
                input_type=None,
                output_type=PDFProcessingOutput,
                module_name="document_loader",
                method_name="load_plan"
            ),
            PipelineStage.SEMANTIC_CHUNKING: StageContract(
                stage=PipelineStage.SEMANTIC_CHUNKING,
                input_type=PDFProcessingOutput,
                output_type=SemanticChunkingOutput,
                module_name="semantic_chunking_policy",
                method_name="chunk_text"
            ),
            PipelineStage.EMBEDDING_GENERATION: StageContract(
                stage=PipelineStage.EMBEDDING_GENERATION,
                input_type=SemanticChunkingOutput,
                output_type=EmbeddingGenerationOutput,
                module_name="embedding_policy",
                method_name="generate_embeddings"
            ),
            PipelineStage.POLICY_ANALYSIS: StageContract(
                stage=PipelineStage.POLICY_ANALYSIS,
                input_type=EmbeddingGenerationOutput,
                output_type=PolicyAnalysisOutput,
                module_name="analyzer_one",
                method_name="analyze_policy"
            ),
            PipelineStage.CONTRADICTION_DETECTION: StageContract(
                stage=PipelineStage.CONTRADICTION_DETECTION,
                input_type=PolicyAnalysisOutput,
                output_type=ContradictionDetectionOutput,
                module_name="contradiction_detection",
                method_name="detect_contradictions"
            ),
            PipelineStage.FINANCIAL_VIABILITY: StageContract(
                stage=PipelineStage.FINANCIAL_VIABILITY,
                input_type=ContradictionDetectionOutput,
                output_type=FinancialViabilityOutput,
                module_name="financial_viability",
                method_name="assess_viability"
            ),
            PipelineStage.REPORTING: StageContract(
                stage=PipelineStage.REPORTING,
                input_type=FinancialViabilityOutput,
                output_type=ReportingOutput,
                module_name="report_assembly",
                method_name="generate_report"
            )
        }
        
        self.stage_dependencies = {
            PipelineStage.PDF_PROCESSING: [],
            PipelineStage.SEMANTIC_CHUNKING: [PipelineStage.PDF_PROCESSING],
            PipelineStage.EMBEDDING_GENERATION: [PipelineStage.SEMANTIC_CHUNKING],
            PipelineStage.POLICY_ANALYSIS: [PipelineStage.EMBEDDING_GENERATION],
            PipelineStage.CONTRADICTION_DETECTION: [PipelineStage.POLICY_ANALYSIS],
            PipelineStage.FINANCIAL_VIABILITY: [PipelineStage.CONTRADICTION_DETECTION],
            PipelineStage.REPORTING: [PipelineStage.FINANCIAL_VIABILITY]
        }
    
    def register_stage(
        self,
        stage: PipelineStage,
        contract: StageContract,
        dependencies: List[PipelineStage]
    ):
        """
        Register a custom stage implementation
        
        Args:
            stage: Pipeline stage to register
            contract: Stage contract defining inputs/outputs
            dependencies: List of prerequisite stages
        """
        self.stage_contracts[stage] = contract
        self.stage_dependencies[stage] = dependencies
        
        logger.info(f"Registered stage: {stage.name}")
    
    def register_transformation(
        self,
        from_stage: PipelineStage,
        to_stage: PipelineStage,
        transformer: Callable[[Any], Any]
    ):
        """
        Register data transformation between stages
        
        Args:
            from_stage: Source stage
            to_stage: Destination stage
            transformer: Function to transform data
        """
        self.data_transformations[(from_stage, to_stage)] = transformer
        
        logger.info(f"Registered transformation: {from_stage.name} -> {to_stage.name}")
    
    def validate_flow_completeness(self) -> bool:
        """
        Validate that all stages have implementations
        
        Returns:
            True if complete, raises exception otherwise
        """
        execution_order = PipelineStage.get_execution_order()
        
        for stage in execution_order:
            if stage not in self.stage_contracts:
                raise MissingStageImplementationError(stage)
        
        logger.info("Flow completeness validated: all stages implemented")
        return True
    
    def validate_stage_dependencies(self) -> bool:
        """
        Validate that stage dependencies are satisfied
        
        Returns:
            True if valid, raises exception otherwise
        """
        for stage, deps in self.stage_dependencies.items():
            for dep in deps:
                if dep not in self.stage_contracts:
                    raise MissingStageImplementationError(dep)
                
                if dep >= stage:
                    raise ValueError(
                        f"Invalid dependency: {stage.name} depends on "
                        f"{dep.name} which executes later or at same time"
                    )
        
        logger.info("Stage dependencies validated")
        return True
    
    def get_contract(self, stage: PipelineStage) -> StageContract:
        """Get contract for specified stage"""
        if stage not in self.stage_contracts:
            raise MissingStageImplementationError(stage)
        return self.stage_contracts[stage]
    
    def get_dependencies(self, stage: PipelineStage) -> List[PipelineStage]:
        """Get dependencies for specified stage"""
        return self.stage_dependencies.get(stage, [])
    
    def get_transformation(
        self,
        from_stage: PipelineStage,
        to_stage: PipelineStage
    ) -> Optional[Callable]:
        """Get transformation function between stages"""
        return self.data_transformations.get((from_stage, to_stage))


class ModuleController:
    """
    Enforces canonical pipeline execution with contract validation
    
    Features:
    - Module registration with stage binding
    - Sequence enforcement (prevents out-of-order execution)
    - Contract validation at each stage boundary
    - Runtime type checking
    - Automatic flow validation before execution
    """
    
    def __init__(self, flow_composition: Optional[FlowComposition] = None):
        """
        Initialize module controller
        
        Args:
            flow_composition: Optional flow composition (uses default if None)
        """
        self.flow = flow_composition or FlowComposition()
        self.registered_modules: Dict[PipelineStage, Any] = {}
        self.execution_state: Dict[str, Any] = {
            "current_stage": None,
            "completed_stages": set(),
            "stage_outputs": {},
            "execution_history": []
        }
        
        logger.info("ModuleController initialized")
    
    def register_module(
        self,
        stage: PipelineStage,
        module_instance: Any,
        validate: bool = True
    ):
        """
        Register module implementation for stage
        
        Args:
            stage: Pipeline stage
            module_instance: Module instance implementing stage
            validate: Whether to validate module has required method
            
        Raises:
            ValueError: If module doesn't implement required method
        """
        contract = self.flow.get_contract(stage)
        
        if validate:
            if not hasattr(module_instance, contract.method_name):
                raise ValueError(
                    f"Module {module_instance.__class__.__name__} does not implement "
                    f"required method: {contract.method_name}"
                )
        
        self.registered_modules[stage] = module_instance
        
        logger.info(
            f"Registered module for {stage.name}: "
            f"{module_instance.__class__.__name__}.{contract.method_name}"
        )
    
    def validate_pipeline(self) -> bool:
        """
        Validate complete pipeline before execution
        
        Checks:
        - Flow completeness (all stages implemented)
        - Stage dependencies satisfied
        - All required modules registered
        
        Returns:
            True if valid, raises exception otherwise
        """
        self.flow.validate_flow_completeness()
        self.flow.validate_stage_dependencies()
        
        execution_order = PipelineStage.get_execution_order()
        for stage in execution_order:
            if stage not in self.registered_modules:
                raise MissingStageImplementationError(stage)
        
        logger.info("Pipeline validation successful")
        return True
    
    def execute_stage(
        self,
        stage: PipelineStage,
        input_data: Any = None,
        **kwargs
    ) -> Any:
        """
        Execute single pipeline stage with contract enforcement
        
        Enforcement:
        - Validates stage is next in sequence
        - Validates input matches contract
        - Executes module method
        - Validates output matches contract
        - Records execution in state
        
        Args:
            stage: Stage to execute
            input_data: Input data (must match contract)
            **kwargs: Additional arguments for module method
            
        Returns:
            Stage output (validated against contract)
            
        Raises:
            StageSequenceError: If stage executed out of order
            ContractViolationError: If input/output violates contract
            MissingStageImplementationError: If stage not registered
        """
        self._validate_stage_sequence(stage)
        
        contract = self.flow.get_contract(stage)
        
        if contract.input_type and input_data is not None:
            if not isinstance(input_data, contract.input_type):
                raise ContractViolationError(
                    stage=stage,
                    field="__input__",
                    message=f"Expected {contract.input_type.__name__}, "
                            f"got {type(input_data).__name__}"
                )
        
        if stage not in self.registered_modules:
            raise MissingStageImplementationError(stage)
        
        module = self.registered_modules[stage]
        method = getattr(module, contract.method_name)
        
        logger.info(f"Executing stage: {stage.name}")
        start_time = time.time()
        
        try:
            if input_data is not None:
                output = method(input_data, **kwargs)
            else:
                output = method(**kwargs)
            
            contract.validate_output(output)
            
            execution_time = time.time() - start_time
            
            self._record_stage_execution(stage, output, execution_time)
            
            logger.info(
                f"Stage {stage.name} completed successfully in {execution_time:.2f}s"
            )
            
            return output
            
        except ContractViolationError:
            raise
        except Exception as e:
            logger.error(f"Error executing stage {stage.name}: {e}", exc_info=True)
            raise
    
    def execute_pipeline(
        self,
        initial_input: Any = None,
        **kwargs
    ) -> Dict[PipelineStage, Any]:
        """
        Execute complete pipeline from start to finish
        
        Args:
            initial_input: Initial input for first stage (e.g., file path)
            **kwargs: Additional arguments passed to all stages
            
        Returns:
            Dictionary mapping stages to their outputs
            
        Raises:
            Various exceptions if validation or execution fails
        """
        self.validate_pipeline()
        
        logger.info("Starting complete pipeline execution")
        start_time = time.time()
        
        execution_order = PipelineStage.get_execution_order()
        outputs: Dict[PipelineStage, Any] = {}
        
        current_input = initial_input
        
        for stage in execution_order:
            transformation = None
            if outputs:
                prev_stage = execution_order[execution_order.index(stage) - 1]
                transformation = self.flow.get_transformation(prev_stage, stage)
            
            if transformation:
                current_input = transformation(current_input)
            
            output = self.execute_stage(stage, current_input, **kwargs)
            outputs[stage] = output
            current_input = output
        
        total_time = time.time() - start_time
        logger.info(f"Pipeline execution completed in {total_time:.2f}s")
        
        return outputs
    
    def _validate_stage_sequence(self, stage: PipelineStage):
        """
        Validate that stage is next in canonical sequence
        
        Args:
            stage: Stage to validate
            
        Raises:
            StageSequenceError: If stage is out of order
        """
        execution_order = PipelineStage.get_execution_order()
        completed = self.execution_state["completed_stages"]
        
        expected_index = len(completed)
        
        if expected_index >= len(execution_order):
            raise ValueError("All pipeline stages already completed")
        
        expected_stage = execution_order[expected_index]
        
        if stage != expected_stage:
            raise StageSequenceError(stage, expected_stage)
        
        deps = self.flow.get_dependencies(stage)
        for dep in deps:
            if dep not in completed:
                raise StageSequenceError(
                    stage,
                    dep
                )
    
    def _record_stage_execution(
        self,
        stage: PipelineStage,
        output: Any,
        execution_time: float
    ):
        """Record stage execution in internal state"""
        self.execution_state["current_stage"] = stage
        self.execution_state["completed_stages"].add(stage)
        self.execution_state["stage_outputs"][stage] = output
        self.execution_state["execution_history"].append({
            "stage": stage.name,
            "timestamp": time.time(),
            "execution_time": execution_time,
            "output_type": type(output).__name__
        })
    
    def reset_execution_state(self):
        """Reset execution state for new pipeline run"""
        self.execution_state = {
            "current_stage": None,
            "completed_stages": set(),
            "stage_outputs": {},
            "execution_history": []
        }
        logger.info("Execution state reset")
    
    def get_execution_status(self) -> Dict[str, Any]:
        """
        Get current execution status
        
        Returns:
            Dictionary with execution state information
        """
        execution_order = PipelineStage.get_execution_order()
        completed = self.execution_state["completed_stages"]
        
        return {
            "current_stage": self.execution_state["current_stage"].name 
                           if self.execution_state["current_stage"] else None,
            "completed_stages": [s.name for s in completed],
            "remaining_stages": [s.name for s in execution_order if s not in completed],
            "progress_percentage": (len(completed) / len(execution_order)) * 100,
            "execution_history": self.execution_state["execution_history"]
        }
    
    def get_stage_output(self, stage: PipelineStage) -> Optional[Any]:
        """Get output from specific stage"""
        return self.execution_state["stage_outputs"].get(stage)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Module Controller - Canonical Pipeline Execution")
    print("=" * 80)
    
    controller = ModuleController()
    
    print("\nPipeline Stages (Execution Order):")
    for i, stage in enumerate(PipelineStage.get_execution_order(), 1):
        print(f"  {i}. {stage.name}")
    
    print("\nStage Contracts:")
    for stage in PipelineStage.get_execution_order():
        contract = controller.flow.get_contract(stage)
        print(f"  {stage.name}:")
        print(f"    Module: {contract.module_name}.{contract.method_name}")
        print(f"    Output: {contract.output_type.__name__}")
    
    print("\nStage Dependencies:")
    for stage in PipelineStage.get_execution_order():
        deps = controller.flow.get_dependencies(stage)
        if deps:
            dep_names = [d.name for d in deps]
            print(f"  {stage.name} <- {dep_names}")
    
    print("\n" + "=" * 80)
