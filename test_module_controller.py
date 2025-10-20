"""
Tests for Module Controller - Contract Enforcement and Pipeline Validation
===========================================================================

Tests:
- Stage sequencing and ordering
- Contract validation (input/output)
- Flow composition and topology
- Pipeline execution enforcement
- Exception handling for violations
"""

import pytest
from orchestrator.module_controller import (
    ModuleController,
    FlowComposition,
    PipelineStage,
    StageContract,
    PDFProcessingOutput,
    SemanticChunkingOutput,
    EmbeddingGenerationOutput,
    PolicyAnalysisOutput,
    ContradictionDetectionOutput,
    FinancialViabilityOutput,
    ReportingOutput,
    ContractViolationError,
    StageSequenceError,
    MissingStageImplementationError
)


class TestPipelineStage:
    """Test PipelineStage enum and ordering"""
    
    def test_stage_ordering(self):
        """Test stages are in correct order"""
        order = PipelineStage.get_execution_order()
        
        assert len(order) == 7
        assert order[0] == PipelineStage.PDF_PROCESSING
        assert order[1] == PipelineStage.SEMANTIC_CHUNKING
        assert order[2] == PipelineStage.EMBEDDING_GENERATION
        assert order[3] == PipelineStage.POLICY_ANALYSIS
        assert order[4] == PipelineStage.CONTRADICTION_DETECTION
        assert order[5] == PipelineStage.FINANCIAL_VIABILITY
        assert order[6] == PipelineStage.REPORTING
    
    def test_stage_comparison(self):
        """Test stage comparison operators"""
        assert PipelineStage.PDF_PROCESSING < PipelineStage.SEMANTIC_CHUNKING
        assert PipelineStage.EMBEDDING_GENERATION < PipelineStage.POLICY_ANALYSIS
        assert PipelineStage.REPORTING > PipelineStage.FINANCIAL_VIABILITY


class TestContractValidation:
    """Test contract validation for each stage"""
    
    def test_pdf_processing_contract_valid(self):
        """Test valid PDF processing output"""
        output = PDFProcessingOutput(
            plan_text="A" * 200,
            page_count=10,
            metadata={"title": "Test Plan"},
            extraction_method="PyPDF2"
        )
        assert output.plan_text == "A" * 200
        assert output.page_count == 10
    
    def test_pdf_processing_contract_invalid_text_length(self):
        """Test PDF contract rejects short text"""
        with pytest.raises(ContractViolationError) as exc:
            PDFProcessingOutput(
                plan_text="short",
                page_count=1,
                metadata={},
                extraction_method="PyPDF2"
            )
        assert exc.value.stage == PipelineStage.PDF_PROCESSING
        assert "plan_text" in exc.value.field
    
    def test_pdf_processing_contract_invalid_page_count(self):
        """Test PDF contract rejects invalid page count"""
        with pytest.raises(ContractViolationError) as exc:
            PDFProcessingOutput(
                plan_text="A" * 200,
                page_count=0,
                metadata={},
                extraction_method="PyPDF2"
            )
        assert "page_count" in exc.value.field
    
    def test_semantic_chunking_contract_valid(self):
        """Test valid semantic chunking output"""
        chunks = ("chunk1", "chunk2", "chunk3")
        metadata = ({"pos": 0}, {"pos": 1}, {"pos": 2})
        
        output = SemanticChunkingOutput(
            chunks=chunks,
            chunk_metadata=metadata,
            chunking_strategy="semantic",
            total_chunks=3
        )
        assert output.total_chunks == 3
        assert len(output.chunks) == 3
    
    def test_semantic_chunking_contract_mismatch(self):
        """Test semantic chunking rejects count mismatch"""
        with pytest.raises(ContractViolationError) as exc:
            SemanticChunkingOutput(
                chunks=("chunk1", "chunk2"),
                chunk_metadata=({"pos": 0}, {"pos": 1}),
                chunking_strategy="semantic",
                total_chunks=5
            )
        assert "total_chunks" in exc.value.field
    
    def test_embedding_generation_contract_valid(self):
        """Test valid embedding generation output"""
        embeddings = ([0.1, 0.2], [0.3, 0.4], [0.5, 0.6])
        indices = (0, 1, 2)
        
        output = EmbeddingGenerationOutput(
            embeddings=embeddings,
            embedding_model="sentence-transformers",
            embedding_dimension=384,
            chunk_indices=indices
        )
        assert output.embedding_dimension == 384
        assert len(output.embeddings) == 3
    
    def test_embedding_generation_contract_dimension_invalid(self):
        """Test embedding contract rejects invalid dimension"""
        with pytest.raises(ContractViolationError) as exc:
            EmbeddingGenerationOutput(
                embeddings=([0.1],),
                embedding_model="test",
                embedding_dimension=0,
                chunk_indices=(0,)
            )
        assert "embedding_dimension" in exc.value.field
    
    def test_policy_analysis_contract_valid(self):
        """Test valid policy analysis output"""
        output = PolicyAnalysisOutput(
            policy_segments=("seg1", "seg2"),
            alignment_scores={"decalogo": 0.8, "sdg": 0.7},
            policy_areas={"P1": {}, "P2": {}},
            dimensions={"D1": {}, "D2": {}, "D3": {}, "D4": {}, "D5": {}, "D6": {}}
        )
        assert len(output.policy_segments) == 2
    
    def test_policy_analysis_contract_missing_dimensions(self):
        """Test policy analysis rejects missing dimensions"""
        with pytest.raises(ContractViolationError) as exc:
            PolicyAnalysisOutput(
                policy_segments=("seg1",),
                alignment_scores={},
                policy_areas={},
                dimensions={"D1": {}, "D2": {}}
            )
        assert "dimensions" in exc.value.field
    
    def test_contradiction_detection_contract_valid(self):
        """Test valid contradiction detection output"""
        output = ContradictionDetectionOutput(
            contradictions=({"type": "logical"}, {"type": "factual"}),
            contradiction_scores=(0.8, 0.6),
            contradiction_pairs=(("seg1", "seg2"), ("seg3", "seg4")),
            total_contradictions=2
        )
        assert output.total_contradictions == 2
    
    def test_financial_viability_contract_valid(self):
        """Test valid financial viability output"""
        output = FinancialViabilityOutput(
            viability_score=75.5,
            budget_analysis={"total": 1000000, "allocated": 800000},
            financial_risks=("risk1", "risk2"),
            recommendations=("rec1", "rec2")
        )
        assert 0 <= output.viability_score <= 100
    
    def test_financial_viability_contract_invalid_score(self):
        """Test financial viability rejects invalid score"""
        with pytest.raises(ContractViolationError) as exc:
            FinancialViabilityOutput(
                viability_score=150.0,
                budget_analysis={"total": 1000},
                financial_risks=(),
                recommendations=()
            )
        assert "viability_score" in exc.value.field
    
    def test_reporting_contract_valid(self):
        """Test valid reporting output"""
        output = ReportingOutput(
            micro_report={"questions": []},
            meso_report={"clusters": []},
            macro_report={"score": 80},
            report_path="/tmp/report.json"
        )
        assert output.report_path.endswith('.json')


class TestFlowComposition:
    """Test flow composition and topology validation"""
    
    def test_flow_initialization(self):
        """Test flow initializes with default topology"""
        flow = FlowComposition()
        
        assert len(flow.stage_contracts) == 7
        assert len(flow.stage_dependencies) == 7
    
    def test_flow_completeness_validation(self):
        """Test flow validates all stages present"""
        flow = FlowComposition()
        assert flow.validate_flow_completeness() is True
    
    def test_flow_missing_stage(self):
        """Test flow detects missing stage"""
        flow = FlowComposition()
        del flow.stage_contracts[PipelineStage.POLICY_ANALYSIS]
        
        with pytest.raises(MissingStageImplementationError) as exc:
            flow.validate_flow_completeness()
        assert exc.value.stage == PipelineStage.POLICY_ANALYSIS
    
    def test_flow_dependency_validation(self):
        """Test flow validates dependencies"""
        flow = FlowComposition()
        assert flow.validate_stage_dependencies() is True
    
    def test_flow_get_contract(self):
        """Test getting stage contract"""
        flow = FlowComposition()
        contract = flow.get_contract(PipelineStage.PDF_PROCESSING)
        
        assert contract.stage == PipelineStage.PDF_PROCESSING
        assert contract.output_type == PDFProcessingOutput
    
    def test_flow_get_dependencies(self):
        """Test getting stage dependencies"""
        flow = FlowComposition()
        
        deps = flow.get_dependencies(PipelineStage.PDF_PROCESSING)
        assert deps == []
        
        deps = flow.get_dependencies(PipelineStage.SEMANTIC_CHUNKING)
        assert PipelineStage.PDF_PROCESSING in deps
    
    def test_flow_register_transformation(self):
        """Test registering data transformation"""
        flow = FlowComposition()
        
        def transform(data):
            return data
        
        flow.register_transformation(
            PipelineStage.PDF_PROCESSING,
            PipelineStage.SEMANTIC_CHUNKING,
            transform
        )
        
        result = flow.get_transformation(
            PipelineStage.PDF_PROCESSING,
            PipelineStage.SEMANTIC_CHUNKING
        )
        assert result == transform


class TestModuleController:
    """Test module controller execution and enforcement"""
    
    def test_controller_initialization(self):
        """Test controller initializes correctly"""
        controller = ModuleController()
        
        assert controller.flow is not None
        assert len(controller.registered_modules) == 0
    
    def test_module_registration(self):
        """Test registering module for stage"""
        controller = ModuleController()
        
        class MockModule:
            def load_plan(self, **kwargs):
                return PDFProcessingOutput(
                    plan_text="A" * 200,
                    page_count=5,
                    metadata={},
                    extraction_method="mock"
                )
        
        module = MockModule()
        controller.register_module(
            PipelineStage.PDF_PROCESSING,
            module,
            validate=True
        )
        
        assert PipelineStage.PDF_PROCESSING in controller.registered_modules
    
    def test_module_registration_missing_method(self):
        """Test registration fails if method missing"""
        controller = ModuleController()
        
        class BadModule:
            pass
        
        with pytest.raises(ValueError) as exc:
            controller.register_module(
                PipelineStage.PDF_PROCESSING,
                BadModule(),
                validate=True
            )
        assert "does not implement" in str(exc.value)
    
    def test_pipeline_validation_success(self):
        """Test pipeline validates when all modules registered"""
        controller = ModuleController()
        
        # Register all stages with mock modules
        for stage in PipelineStage.get_execution_order():
            contract = controller.flow.get_contract(stage)
            
            class MockModule:
                pass
            
            mock = MockModule()
            setattr(mock, contract.method_name, lambda *args, **kwargs: None)
            
            controller.register_module(stage, mock, validate=False)
        
        assert controller.validate_pipeline() is True
    
    def test_pipeline_validation_missing_module(self):
        """Test pipeline validation fails if module missing"""
        controller = ModuleController()
        
        with pytest.raises(MissingStageImplementationError):
            controller.validate_pipeline()
    
    def test_stage_execution_valid(self):
        """Test executing stage with valid input/output"""
        controller = ModuleController()
        
        class PDFModule:
            def load_plan(self, **kwargs):
                return PDFProcessingOutput(
                    plan_text="A" * 200,
                    page_count=5,
                    metadata={"test": True},
                    extraction_method="mock"
                )
        
        controller.register_module(PipelineStage.PDF_PROCESSING, PDFModule())
        
        output = controller.execute_stage(PipelineStage.PDF_PROCESSING)
        
        assert isinstance(output, PDFProcessingOutput)
        assert output.page_count == 5
        assert PipelineStage.PDF_PROCESSING in controller.execution_state["completed_stages"]
    
    def test_stage_execution_sequence_violation(self):
        """Test stage execution enforces sequence"""
        controller = ModuleController()
        
        class MockModule:
            def chunk_text(self, data, **kwargs):
                return SemanticChunkingOutput(
                    chunks=("c1",),
                    chunk_metadata=({"pos": 0},),
                    chunking_strategy="mock",
                    total_chunks=1
                )
        
        controller.register_module(PipelineStage.SEMANTIC_CHUNKING, MockModule())
        
        with pytest.raises(StageSequenceError) as exc:
            controller.execute_stage(PipelineStage.SEMANTIC_CHUNKING)
        
        assert exc.value.attempted_stage == PipelineStage.SEMANTIC_CHUNKING
        assert exc.value.expected_stage == PipelineStage.PDF_PROCESSING
    
    def test_stage_execution_contract_violation(self):
        """Test stage execution validates output contract"""
        controller = ModuleController()
        
        class BadModule:
            def load_plan(self, **kwargs):
                # Return invalid output (wrong type)
                return "not a PDFProcessingOutput"
        
        controller.register_module(PipelineStage.PDF_PROCESSING, BadModule())
        
        with pytest.raises(ContractViolationError) as exc:
            controller.execute_stage(PipelineStage.PDF_PROCESSING)
        
        assert exc.value.stage == PipelineStage.PDF_PROCESSING
    
    def test_execution_status(self):
        """Test getting execution status"""
        controller = ModuleController()
        
        status = controller.get_execution_status()
        
        assert status["current_stage"] is None
        assert len(status["completed_stages"]) == 0
        assert status["progress_percentage"] == 0.0
    
    def test_reset_execution_state(self):
        """Test resetting execution state"""
        controller = ModuleController()
        
        controller.execution_state["completed_stages"].add(PipelineStage.PDF_PROCESSING)
        controller.reset_execution_state()
        
        assert len(controller.execution_state["completed_stages"]) == 0
    
    def test_get_stage_output(self):
        """Test retrieving stage output"""
        controller = ModuleController()
        
        class PDFModule:
            def load_plan(self, **kwargs):
                return PDFProcessingOutput(
                    plan_text="A" * 200,
                    page_count=1,
                    metadata={},
                    extraction_method="mock"
                )
        
        controller.register_module(PipelineStage.PDF_PROCESSING, PDFModule())
        output = controller.execute_stage(PipelineStage.PDF_PROCESSING)
        
        retrieved = controller.get_stage_output(PipelineStage.PDF_PROCESSING)
        assert retrieved == output


class TestContractImmutability:
    """Test contract immutability guarantees"""
    
    def test_pdf_output_immutable(self):
        """Test PDFProcessingOutput is immutable"""
        output = PDFProcessingOutput(
            plan_text="A" * 200,
            page_count=1,
            metadata={},
            extraction_method="test"
        )
        
        with pytest.raises(AttributeError):
            output.plan_text = "modified"
    
    def test_semantic_chunking_uses_tuples(self):
        """Test SemanticChunkingOutput uses tuples for immutability"""
        output = SemanticChunkingOutput(
            chunks=("c1", "c2"),
            chunk_metadata=({"pos": 0}, {"pos": 1}),
            chunking_strategy="test",
            total_chunks=2
        )
        
        assert isinstance(output.chunks, tuple)
        assert isinstance(output.chunk_metadata, tuple)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
