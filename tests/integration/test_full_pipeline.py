"""
Integration Test for FARFAN Full Pipeline
==========================================

Tests complete pipeline flow from orchestrator initialization through
execution of a single question, validating all components are properly
instantiated and produce expected results.

Tests validate:
1. FARFANOrchestrator initialization with all dependencies
2. ExecutionChoreographer initialization with adapters
3. Module adapter registry proper instantiation
4. Successful execution producing valid ExecutionResult
5. All required adapters (DerekBeachAdapter, ModulosAdapter, etc.) are accessible

Author: FARFAN Test Team
Version: 3.0.0
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from orchestrator.core_orchestrator import FARFANOrchestrator
from orchestrator.choreographer import ExecutionChoreographer, ExecutionResult, ExecutionStatus
from orchestrator.module_adapters import ModuleAdapterRegistry
from orchestrator.circuit_breaker import CircuitBreaker


@dataclass
class MockQuestionSpec:
    """Mock question specification for testing"""
    canonical_id: str
    question_text: str
    policy_area: str
    dimension: str
    execution_chain: list
    metadata: dict


class MockQuestionnaireParser:
    """Mock questionnaire parser for testing without cuestionario.json"""
    
    def parse_all_questions(self):
        """Return a single mock question for testing"""
        return [self._create_mock_question()]
    
    def _create_mock_question(self):
        """Create a minimal mock question with execution chain"""
        return MockQuestionSpec(
            canonical_id="P1-D1-Q1",
            question_text="¿El plan contiene un diagnóstico municipal?",
            policy_area="P1",
            dimension="D1",
            execution_chain=[
                {
                    "adapter": "policy_segmenter",
                    "method": "segment_document",
                    "args": [{"source": "plan_text"}],
                    "kwargs": {}
                }
            ],
            metadata={
                "policy_area": "P1",
                "dimension": "D1",
                "rubric_type": "TYPE_A"
            }
        )


@pytest.fixture
def module_adapter_registry():
    """
    Initialize ModuleAdapterRegistry with all 9 adapters
    
    Returns:
        ModuleAdapterRegistry: Fully initialized adapter registry
    """
    registry = ModuleAdapterRegistry()
    
    expected_adapters = [
        "teoria_cambio",
        "analyzer_one",
        "dereck_beach",
        "embedding_policy",
        "semantic_chunking_policy",
        "contradiction_detection",
        "financial_viability",
        "policy_processor",
        "policy_segmenter"
    ]
    
    for adapter_name in expected_adapters:
        assert adapter_name in registry.adapters, (
            f"Expected adapter '{adapter_name}' not found in registry. "
            f"Available adapters: {list(registry.adapters.keys())}"
        )
    
    return registry


@pytest.fixture
def questionnaire_parser():
    """
    Create mock questionnaire parser
    
    Returns:
        MockQuestionnaireParser: Parser with single test question
    """
    return MockQuestionnaireParser()


@pytest.fixture
def circuit_breaker():
    """
    Initialize circuit breaker for fault tolerance
    
    Returns:
        CircuitBreaker: Circuit breaker instance
    """
    return CircuitBreaker()


@pytest.fixture
def orchestrator(module_adapter_registry, questionnaire_parser):
    """
    Initialize FARFANOrchestrator with all dependencies
    
    Args:
        module_adapter_registry: Adapter registry fixture
        questionnaire_parser: Questionnaire parser fixture
        
    Returns:
        FARFANOrchestrator: Fully initialized orchestrator
        
    Raises:
        AssertionError: If orchestrator fails to initialize properly
    """
    try:
        orchestrator = FARFANOrchestrator(
            module_adapter_registry=module_adapter_registry,
            questionnaire_parser=questionnaire_parser,
            config={"max_workers": 2}
        )
        
        assert orchestrator is not None, "Orchestrator initialization returned None"
        assert hasattr(orchestrator, 'choreographer'), (
            "Orchestrator missing choreographer attribute"
        )
        assert hasattr(orchestrator, 'circuit_breaker'), (
            "Orchestrator missing circuit_breaker attribute"
        )
        assert hasattr(orchestrator, 'module_registry'), (
            "Orchestrator missing module_registry attribute"
        )
        
        return orchestrator
        
    except Exception as e:
        pytest.fail(
            f"Failed to initialize FARFANOrchestrator: {e}\n"
            f"Check that all required dependencies are properly installed "
            f"and adapters are correctly configured."
        )


@pytest.mark.integration
def test_orchestrator_initialization(orchestrator):
    """
    Test that FARFANOrchestrator initializes with all required components
    
    Validates:
    - Orchestrator instance is created
    - Choreographer is initialized
    - Circuit breaker is initialized
    - Module registry is accessible
    - Report assembler is initialized
    """
    assert orchestrator is not None, "Orchestrator should be initialized"
    assert orchestrator.choreographer is not None, "Choreographer should be initialized"
    assert orchestrator.circuit_breaker is not None, "Circuit breaker should be initialized"
    assert orchestrator.module_registry is not None, "Module registry should be initialized"
    assert orchestrator.report_assembler is not None, "Report assembler should be initialized"
    
    assert isinstance(orchestrator.choreographer, ExecutionChoreographer), (
        f"Choreographer should be ExecutionChoreographer instance, "
        f"got {type(orchestrator.choreographer)}"
    )


@pytest.mark.integration
def test_execution_choreographer_initialization(orchestrator):
    """
    Test that ExecutionChoreographer is properly initialized with adapters
    
    Validates:
    - Choreographer has dependency graph
    - Choreographer has adapter registry
    - All 9 expected adapters are registered
    """
    choreographer = orchestrator.choreographer
    
    assert hasattr(choreographer, 'execution_graph'), (
        "Choreographer should have execution_graph attribute"
    )
    assert hasattr(choreographer, 'adapter_registry'), (
        "Choreographer should have adapter_registry attribute"
    )
    
    expected_adapters = [
        "teoria_cambio",
        "analyzer_one", 
        "dereck_beach",
        "embedding_policy",
        "semantic_chunking_policy",
        "contradiction_detection",
        "financial_viability",
        "policy_processor",
        "policy_segmenter"
    ]
    
    for adapter_name in expected_adapters:
        assert adapter_name in choreographer.adapter_registry, (
            f"Adapter '{adapter_name}' not found in choreographer registry. "
            f"Available: {list(choreographer.adapter_registry.keys())}"
        )


@pytest.mark.integration
def test_module_adapters_accessible(module_adapter_registry):
    """
    Test that all module adapters are properly instantiated and accessible
    
    Validates:
    - DerekBeachAdapter is accessible
    - ModulosAdapter is accessible
    - AnalyzerOneAdapter is accessible
    - All other adapters are accessible
    """
    adapter_mappings = {
        "dereck_beach": "DerekBeachAdapter",
        "teoria_cambio": "ModulosAdapter",
        "analyzer_one": "AnalyzerOneAdapter",
        "embedding_policy": "EmbeddingPolicyAdapter",
        "semantic_chunking_policy": "SemanticChunkingPolicyAdapter",
        "contradiction_detection": "ContradictionDetectionAdapter",
        "financial_viability": "FinancialViabilityAdapter",
        "policy_processor": "PolicyProcessorAdapter",
        "policy_segmenter": "PolicySegmenterAdapter"
    }
    
    for adapter_name, expected_class in adapter_mappings.items():
        assert adapter_name in module_adapter_registry.adapters, (
            f"Adapter '{adapter_name}' not found in registry"
        )
        
        adapter_instance = module_adapter_registry.adapters[adapter_name]
        assert adapter_instance is not None, (
            f"Adapter '{adapter_name}' is None"
        )
        
        actual_class = adapter_instance.__class__.__name__
        assert actual_class == expected_class, (
            f"Adapter '{adapter_name}' has wrong class: "
            f"expected {expected_class}, got {actual_class}"
        )


@pytest.mark.integration
def test_single_question_execution(orchestrator, questionnaire_parser, module_adapter_registry, circuit_breaker):
    """
    Test execution of single question through full pipeline
    
    Validates:
    - Question can be parsed
    - Execution chain can be executed
    - ExecutionResult is returned
    - ExecutionResult contains expected fields
    - Execution completes without fatal errors
    """
    question = questionnaire_parser._create_mock_question()
    
    assert question is not None, "Question should be created"
    assert hasattr(question, 'canonical_id'), "Question should have canonical_id"
    assert hasattr(question, 'execution_chain'), "Question should have execution_chain"
    
    plan_text = """
    Plan Municipal de Desarrollo 2023-2027
    
    1. Diagnóstico Municipal
    El municipio cuenta con 50,000 habitantes y una economía basada 
    en la agricultura y el turismo. Se identifican las siguientes necesidades:
    - Infraestructura vial
    - Servicios de salud
    - Educación
    
    2. Objetivos Estratégicos
    - Mejorar la calidad de vida
    - Fortalecer la economía local
    - Promover el desarrollo sostenible
    """
    
    try:
        results = orchestrator.choreographer.execute_question_chain(
            question_spec=question,
            plan_text=plan_text,
            module_adapter_registry=module_adapter_registry,
            circuit_breaker=circuit_breaker
        )
        
        assert results is not None, "Execution results should not be None"
        assert isinstance(results, dict), (
            f"Results should be dict, got {type(results)}"
        )
        
        if len(results) > 0:
            for key, result in results.items():
                assert isinstance(result, ExecutionResult), (
                    f"Result for '{key}' should be ExecutionResult, got {type(result)}"
                )
                
                assert hasattr(result, 'module_name'), "ExecutionResult should have module_name"
                assert hasattr(result, 'adapter_class'), "ExecutionResult should have adapter_class"
                assert hasattr(result, 'method_name'), "ExecutionResult should have method_name"
                assert hasattr(result, 'status'), "ExecutionResult should have status"
                assert hasattr(result, 'output'), "ExecutionResult should have output"
                assert hasattr(result, 'execution_time'), "ExecutionResult should have execution_time"
                assert hasattr(result, 'evidence_extracted'), "ExecutionResult should have evidence_extracted"
                assert hasattr(result, 'confidence'), "ExecutionResult should have confidence"
                
                assert isinstance(result.status, ExecutionStatus), (
                    f"Status should be ExecutionStatus enum, got {type(result.status)}"
                )
                
                assert result.execution_time >= 0, (
                    f"Execution time should be non-negative, got {result.execution_time}"
                )
                
                assert 0.0 <= result.confidence <= 1.0, (
                    f"Confidence should be between 0 and 1, got {result.confidence}"
                )
        
    except Exception as e:
        pytest.fail(
            f"Execution failed with error: {e}\n"
            f"Question ID: {question.canonical_id}\n"
            f"Execution chain: {question.execution_chain}\n"
            f"This indicates a problem with adapter execution or choreographer logic."
        )


@pytest.mark.integration
def test_execution_result_structure(orchestrator, questionnaire_parser, module_adapter_registry, circuit_breaker):
    """
    Test that ExecutionResult contains all expected fields with correct types
    
    Validates complete ExecutionResult structure:
    - module_name: str
    - adapter_class: str
    - method_name: str
    - status: ExecutionStatus enum
    - output: Optional[Dict]
    - error: Optional[str]
    - execution_time: float
    - evidence_extracted: Dict
    - confidence: float
    - metadata: Dict
    """
    question = questionnaire_parser._create_mock_question()
    plan_text = "Plan Municipal de Desarrollo con diagnóstico completo."
    
    try:
        results = orchestrator.choreographer.execute_question_chain(
            question_spec=question,
            plan_text=plan_text,
            module_adapter_registry=module_adapter_registry,
            circuit_breaker=circuit_breaker
        )
        
        if len(results) > 0:
            result = next(iter(results.values()))
            
            assert isinstance(result.module_name, str), "module_name should be str"
            assert isinstance(result.adapter_class, str), "adapter_class should be str"
            assert isinstance(result.method_name, str), "method_name should be str"
            assert isinstance(result.status, ExecutionStatus), "status should be ExecutionStatus"
            assert result.output is None or isinstance(result.output, dict), (
                "output should be None or dict"
            )
            assert result.error is None or isinstance(result.error, str), (
                "error should be None or str"
            )
            assert isinstance(result.execution_time, (int, float)), (
                "execution_time should be numeric"
            )
            assert isinstance(result.evidence_extracted, dict), (
                "evidence_extracted should be dict"
            )
            assert isinstance(result.confidence, (int, float)), (
                "confidence should be numeric"
            )
            assert isinstance(result.metadata, dict), "metadata should be dict"
            
            result_dict = result.to_dict()
            assert isinstance(result_dict, dict), "to_dict() should return dict"
            
            expected_keys = [
                'module_name', 'adapter_class', 'method_name', 'status',
                'output', 'error', 'execution_time', 'evidence', 'confidence', 'metadata'
            ]
            for key in expected_keys:
                assert key in result_dict, f"to_dict() should contain '{key}'"
                
    except Exception as e:
        pytest.fail(
            f"ExecutionResult structure validation failed: {e}\n"
            f"This indicates a problem with ExecutionResult dataclass definition."
        )


@pytest.mark.integration  
def test_circuit_breaker_integration(orchestrator, circuit_breaker):
    """
    Test that circuit breaker is properly integrated with orchestrator
    
    Validates:
    - Circuit breaker can check adapter status
    - Circuit breaker can record successes/failures
    - Circuit breaker status is accessible
    """
    assert orchestrator.circuit_breaker is not None, "Circuit breaker should be initialized"
    
    try:
        status = circuit_breaker.get_all_status()
        assert isinstance(status, dict), "Circuit breaker status should be dict"
        
        can_execute = circuit_breaker.can_execute("policy_segmenter")
        assert isinstance(can_execute, bool), "can_execute should return bool"
        
        circuit_breaker.record_success("policy_segmenter")
        circuit_breaker.record_failure("policy_segmenter", "Test error")
        
    except Exception as e:
        pytest.fail(
            f"Circuit breaker integration test failed: {e}\n"
            f"Check CircuitBreaker implementation for errors."
        )


@pytest.mark.integration
def test_orchestrator_get_status(orchestrator):
    """
    Test that orchestrator can report its health status
    
    Validates:
    - get_orchestrator_status() returns dict
    - Status contains expected keys
    - Adapter availability is reported correctly
    """
    try:
        status = orchestrator.get_orchestrator_status()
        
        assert isinstance(status, dict), "Status should be dict"
        
        expected_keys = [
            'adapters_available',
            'total_adapters', 
            'circuit_breaker_status',
            'execution_stats',
            'questions_available'
        ]
        
        for key in expected_keys:
            assert key in status, f"Status should contain '{key}'"
        
        assert isinstance(status['adapters_available'], list), (
            "adapters_available should be list"
        )
        assert isinstance(status['total_adapters'], int), (
            "total_adapters should be int"
        )
        assert status['total_adapters'] >= 0, (
            "total_adapters should be non-negative"
        )
        
    except Exception as e:
        pytest.fail(
            f"Orchestrator status check failed: {e}\n"
            f"Check FARFANOrchestrator.get_orchestrator_status() implementation."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
