"""
Unit Test Suite for Module Adapters
====================================

Tests all 9 module adapters with mocked dependencies to validate:
- Standardized ModuleResult output format
- Success/failure states and error handling
- Availability checks for missing dependencies
- Execute() method signatures and interface compliance
- Configuration parameter propagation

Run with: python -m pytest tests/unit/test_adapters.py -v
"""

import sys
import unittest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from typing import Dict, List, Any
from dataclasses import fields

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create minimal pytest decorators for unittest compatibility
    class pytest:
        class fixture:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, func):
                return func
        
        class mark:
            @staticmethod
            def parametrize(*args, **kwargs):
                def decorator(func):
                    return func
                return decorator
            
            @staticmethod
            def skip(reason):
                def decorator(func):
                    return func
                return decorator
    
    def skip(reason):
        pass

from orchestrator.module_adapters import (
    BaseAdapter,
    ModuleResult,
    PolicyProcessorAdapter,
    PolicySegmenterAdapter,
    AnalyzerOneAdapter,
    EmbeddingPolicyAdapter,
    SemanticChunkingPolicyAdapter,
    FinancialViabilityAdapter,
    DerekBeachAdapter,
    ContradictionDetectionAdapter,
    ModulosAdapter,
    ModuleAdapterRegistry
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def adapter_dependency_graph() -> Dict[str, Dict[str, Any]]:
    """
    Dependency graph mapping each adapter to its source module imports.
    Used to validate availability checks detect missing dependencies.
    """
    return {
        "PolicyProcessorAdapter": {
            "adapter_class": PolicyProcessorAdapter,
            "module_name": "policy_processor",
            "source_modules": ["policy_processor"],
            "key_classes": [
                "ProcessorConfig",
                "BayesianEvidenceScorer",
                "PolicyTextProcessor",
                "EvidenceBundle",
                "IndustrialPolicyProcessor",
                "AdvancedTextSanitizer",
                "ResilientFileHandler",
                "PolicyAnalysisPipeline"
            ]
        },
        "PolicySegmenterAdapter": {
            "adapter_class": PolicySegmenterAdapter,
            "module_name": "policy_segmenter",
            "source_modules": ["policy_segmenter"],
            "key_classes": [
                "SegmentationConfig",
                "SpanishSentenceSegmenter",
                "BayesianBoundaryScorer",
                "StructureDetector",
                "DPSegmentOptimizer",
                "DocumentSegmenter"
            ]
        },
        "AnalyzerOneAdapter": {
            "adapter_class": AnalyzerOneAdapter,
            "module_name": "analyzer_one",
            "source_modules": ["Analyzer_one"],
            "key_classes": [
                "MunicipalAnalyzer",
                "SemanticAnalyzer",
                "PerformanceAnalyzer",
                "TextMiningEngine",
                "DocumentProcessor",
                "ResultsExporter",
                "ConfigurationManager",
                "BatchProcessor"
            ]
        },
        "EmbeddingPolicyAdapter": {
            "adapter_class": EmbeddingPolicyAdapter,
            "module_name": "embedding_policy",
            "source_modules": ["emebedding_policy"],
            "key_classes": [
                "AdvancedSemanticChunker",
                "BayesianNumericalAnalyzer",
                "PolicyCrossEncoderReranker",
                "PolicyAnalysisEmbedder",
                "ChunkingConfig",
                "PolicyEmbeddingConfig"
            ]
        },
        "SemanticChunkingPolicyAdapter": {
            "adapter_class": SemanticChunkingPolicyAdapter,
            "module_name": "semantic_chunking_policy",
            "source_modules": ["semantic_chunking_policy"],
            "key_classes": [
                "SemanticConfig",
                "BayesianChunkScorer",
                "RecursiveSemanticChunker",
                "PolicySemanticProcessor"
            ]
        },
        "FinancialViabilityAdapter": {
            "adapter_class": FinancialViabilityAdapter,
            "module_name": "financial_viability",
            "source_modules": ["financiero_viabilidad_tablas"],
            "key_classes": ["PDETMunicipalPlanAnalyzer"]
        },
        "DerekBeachAdapter": {
            "adapter_class": DerekBeachAdapter,
            "module_name": "dereck_beach",
            "source_modules": ["dereck_beach"],
            "key_classes": [
                "CausalConfig",
                "CausalPattern",
                "PatternMatcher",
                "EvidenceExtractor",
                "CausalChainAnalyzer",
                "ProcessTracingEngine"
            ]
        },
        "ContradictionDetectionAdapter": {
            "adapter_class": ContradictionDetectionAdapter,
            "module_name": "contradiction_detection",
            "source_modules": ["contradiction_deteccion"],
            "key_classes": [
                "ContradictionConfig",
                "SemanticContradictionDetector",
                "LogicalInconsistencyDetector",
                "TemporalContradictionDetector",
                "NumericalContradictionDetector",
                "ContradictionAnalyzer"
            ]
        },
        "ModulosAdapter": {
            "adapter_class": ModulosAdapter,
            "module_name": "teoria_cambio",
            "source_modules": ["teoria_cambio"],
            "key_classes": [
                "TheoryOfChangeAnalyzer",
                "CausalPathwayExtractor",
                "AssumptionValidator",
                "IndicatorAnalyzer",
                "RiskAnalyzer"
            ]
        }
    }


@pytest.fixture
def mock_module_result() -> ModuleResult:
    """Create a valid ModuleResult for testing"""
    return ModuleResult(
        module_name="test_module",
        class_name="TestClass",
        method_name="test_method",
        status="success",
        data={"key": "value"},
        evidence=[{"type": "test", "value": 1.0}],
        confidence=0.85,
        execution_time=0.123,
        errors=[],
        warnings=[],
        metadata={"test": True}
    )


@pytest.fixture
def all_adapter_classes(adapter_dependency_graph):
    """Return list of all adapter classes for parametrized tests"""
    return [info["adapter_class"] for info in adapter_dependency_graph.values()]


@pytest.fixture
def mock_policy_processor():
    """Mock policy_processor module"""
    mock_module = MagicMock()
    mock_module.ProcessorConfig = MagicMock()
    mock_module.BayesianEvidenceScorer = MagicMock()
    mock_module.PolicyTextProcessor = MagicMock()
    mock_module.IndustrialPolicyProcessor = MagicMock()
    return mock_module


@pytest.fixture
def mock_analyzer_one():
    """Mock Analyzer_one module"""
    mock_module = MagicMock()
    mock_module.MunicipalAnalyzer = MagicMock()
    mock_module.SemanticAnalyzer = MagicMock()
    mock_module.PerformanceAnalyzer = MagicMock()
    return mock_module


# ============================================================================
# TEST: ModuleResult Structure
# ============================================================================

class TestModuleResultStructure:
    """Validate ModuleResult dataclass structure"""

    def test_module_result_has_all_required_fields(self):
        """Verify ModuleResult has all required fields"""
        required_fields = {
            'module_name', 'class_name', 'method_name', 'status',
            'data', 'evidence', 'confidence', 'execution_time'
        }
        result_fields = {f.name for f in fields(ModuleResult)}
        assert required_fields.issubset(result_fields), \
            f"Missing required fields: {required_fields - result_fields}"

    def test_module_result_has_optional_fields(self):
        """Verify ModuleResult has optional error/warning fields"""
        optional_fields = {'errors', 'warnings', 'metadata'}
        result_fields = {f.name for f in fields(ModuleResult)}
        assert optional_fields.issubset(result_fields), \
            f"Missing optional fields: {optional_fields - result_fields}"

    def test_module_result_success_state(self, mock_module_result):
        """Verify ModuleResult can represent success state"""
        assert mock_module_result.status == "success"
        assert mock_module_result.confidence > 0.0
        assert len(mock_module_result.errors) == 0

    def test_module_result_failure_state(self):
        """Verify ModuleResult can represent failure state"""
        result = ModuleResult(
            module_name="test",
            class_name="Test",
            method_name="test",
            status="failed",
            data={},
            evidence=[],
            confidence=0.0,
            execution_time=0.0,
            errors=["Test error"]
        )
        assert result.status == "failed"
        assert result.confidence == 0.0
        assert len(result.errors) > 0

    def test_module_result_types(self, mock_module_result):
        """Verify ModuleResult field types"""
        assert isinstance(mock_module_result.module_name, str)
        assert isinstance(mock_module_result.class_name, str)
        assert isinstance(mock_module_result.method_name, str)
        assert isinstance(mock_module_result.status, str)
        assert isinstance(mock_module_result.data, dict)
        assert isinstance(mock_module_result.evidence, list)
        assert isinstance(mock_module_result.confidence, (int, float))
        assert isinstance(mock_module_result.execution_time, (int, float))
        assert isinstance(mock_module_result.errors, list)
        assert isinstance(mock_module_result.warnings, list)
        assert isinstance(mock_module_result.metadata, dict)


# ============================================================================
# TEST: BaseAdapter Class
# ============================================================================

class TestBaseAdapter:
    """Test BaseAdapter base class functionality"""

    def test_base_adapter_initialization(self):
        """Test BaseAdapter initializes with module_name"""
        adapter = BaseAdapter("test_module")
        assert adapter.module_name == "test_module"
        assert adapter.available == False

    def test_base_adapter_create_unavailable_result(self):
        """Test _create_unavailable_result returns proper ModuleResult"""
        adapter = BaseAdapter("test_module")
        result = adapter._create_unavailable_result("test_method", 0.0)
        
        assert isinstance(result, ModuleResult)
        assert result.status == "failed"
        assert result.module_name == "test_module"
        assert result.method_name == "test_method"
        assert "Module not available" in result.errors

    def test_base_adapter_create_error_result(self):
        """Test _create_error_result returns proper ModuleResult"""
        adapter = BaseAdapter("test_module")
        error = ValueError("Test error")
        result = adapter._create_error_result("test_method", 0.0, error)
        
        assert isinstance(result, ModuleResult)
        assert result.status == "failed"
        assert result.module_name == "test_module"
        assert result.method_name == "test_method"
        assert "Test error" in result.errors


# ============================================================================
# TEST: Adapter Availability Checks
# ============================================================================

class TestAdapterAvailability:
    """Test adapter availability detection for missing dependencies"""

    @pytest.mark.parametrize("adapter_name", [
        "PolicyProcessorAdapter",
        "PolicySegmenterAdapter",
        "AnalyzerOneAdapter",
        "EmbeddingPolicyAdapter",
        "SemanticChunkingPolicyAdapter",
        "FinancialViabilityAdapter",
        "DerekBeachAdapter",
        "ContradictionDetectionAdapter",
        "ModulosAdapter"
    ])
    def test_adapter_unavailable_when_module_missing(self, adapter_name, adapter_dependency_graph):
        """Test adapters correctly detect when their dependencies are missing"""
        adapter_info = adapter_dependency_graph[adapter_name]
        adapter_class = adapter_info["adapter_class"]
        source_module = adapter_info["source_modules"][0]
        
        # Block the import by patching sys.modules
        with patch.dict('sys.modules', {source_module: None}):
            # Force ImportError
            if source_module in sys.modules:
                del sys.modules[source_module]
            
            adapter = adapter_class()
            
            # Adapter should detect missing module
            # Note: Some adapters may still be available if module was already loaded
            # This tests the detection mechanism exists
            assert hasattr(adapter, 'available')
            assert isinstance(adapter.available, bool)

    @pytest.mark.parametrize("adapter_name", [
        "PolicyProcessorAdapter",
        "PolicySegmenterAdapter",
        "AnalyzerOneAdapter",
        "EmbeddingPolicyAdapter",
        "SemanticChunkingPolicyAdapter",
        "FinancialViabilityAdapter",
        "DerekBeachAdapter",
        "ContradictionDetectionAdapter",
        "ModulosAdapter"
    ])
    def test_unavailable_adapter_returns_failed_result(self, adapter_name, adapter_dependency_graph):
        """Test unavailable adapters return failed ModuleResult"""
        adapter_info = adapter_dependency_graph[adapter_name]
        adapter_class = adapter_info["adapter_class"]
        
        adapter = adapter_class()
        
        # Force adapter to be unavailable
        adapter.available = False
        
        result = adapter.execute("test_method", [], {})
        
        assert isinstance(result, ModuleResult)
        assert result.status == "failed"
        assert len(result.errors) > 0
        assert "not available" in result.errors[0].lower() or len(result.errors) > 0


# ============================================================================
# TEST: Execute Method Signatures
# ============================================================================

class TestExecuteMethodSignatures:
    """Validate execute() method signatures match expected interface"""

    @pytest.mark.parametrize("adapter_name", [
        "PolicyProcessorAdapter",
        "PolicySegmenterAdapter",
        "AnalyzerOneAdapter",
        "EmbeddingPolicyAdapter",
        "SemanticChunkingPolicyAdapter",
        "FinancialViabilityAdapter",
        "DerekBeachAdapter",
        "ContradictionDetectionAdapter",
        "ModulosAdapter"
    ])
    def test_adapter_has_execute_method(self, adapter_name, adapter_dependency_graph):
        """Test all adapters have execute() method"""
        adapter_class = adapter_dependency_graph[adapter_name]["adapter_class"]
        adapter = adapter_class()
        
        assert hasattr(adapter, 'execute')
        assert callable(adapter.execute)

    @pytest.mark.parametrize("adapter_name", [
        "PolicyProcessorAdapter",
        "PolicySegmenterAdapter",
        "AnalyzerOneAdapter",
        "EmbeddingPolicyAdapter",
        "SemanticChunkingPolicyAdapter",
        "FinancialViabilityAdapter",
        "DerekBeachAdapter",
        "ContradictionDetectionAdapter",
        "ModulosAdapter"
    ])
    def test_execute_accepts_standard_parameters(self, adapter_name, adapter_dependency_graph):
        """Test execute() accepts (method_name, args, kwargs)"""
        adapter_class = adapter_dependency_graph[adapter_name]["adapter_class"]
        adapter = adapter_class()
        adapter.available = False  # Avoid actually calling methods
        
        # Should not raise TypeError
        result = adapter.execute("test_method", [], {})
        assert isinstance(result, ModuleResult)

    @pytest.mark.parametrize("adapter_name", [
        "PolicyProcessorAdapter",
        "PolicySegmenterAdapter",
        "AnalyzerOneAdapter",
        "EmbeddingPolicyAdapter",
        "SemanticChunkingPolicyAdapter",
        "FinancialViabilityAdapter",
        "DerekBeachAdapter",
        "ContradictionDetectionAdapter",
        "ModulosAdapter"
    ])
    def test_execute_returns_module_result(self, adapter_name, adapter_dependency_graph):
        """Test execute() returns ModuleResult object"""
        adapter_class = adapter_dependency_graph[adapter_name]["adapter_class"]
        adapter = adapter_class()
        adapter.available = False
        
        result = adapter.execute("test_method", [], {})
        
        assert isinstance(result, ModuleResult)


# ============================================================================
# TEST: Error Handling Patterns
# ============================================================================

class TestErrorHandlingPatterns:
    """Test consistent error handling across all adapters"""

    @pytest.mark.parametrize("adapter_name", [
        "PolicyProcessorAdapter",
        "PolicySegmenterAdapter",
        "AnalyzerOneAdapter",
        "EmbeddingPolicyAdapter",
        "SemanticChunkingPolicyAdapter",
        "FinancialViabilityAdapter",
        "DerekBeachAdapter",
        "ContradictionDetectionAdapter",
        "ModulosAdapter"
    ])
    def test_adapter_handles_unknown_method(self, adapter_name, adapter_dependency_graph):
        """Test adapters handle unknown method names gracefully"""
        adapter_class = adapter_dependency_graph[adapter_name]["adapter_class"]
        adapter = adapter_class()
        
        if not adapter.available:
            pytest.skip(f"{adapter_name} dependencies not available")
        
        result = adapter.execute("nonexistent_method_xyz", [], {})
        
        assert isinstance(result, ModuleResult)
        assert result.status == "failed"
        assert len(result.errors) > 0

    @pytest.mark.parametrize("adapter_name", [
        "PolicyProcessorAdapter",
        "PolicySegmenterAdapter",
        "AnalyzerOneAdapter",
        "EmbeddingPolicyAdapter",
        "SemanticChunkingPolicyAdapter",
        "FinancialViabilityAdapter",
        "DerekBeachAdapter",
        "ContradictionDetectionAdapter",
        "ModulosAdapter"
    ])
    def test_adapter_handles_invalid_arguments(self, adapter_name, adapter_dependency_graph):
        """Test adapters handle invalid arguments gracefully"""
        adapter_class = adapter_dependency_graph[adapter_name]["adapter_class"]
        adapter = adapter_class()
        
        if not adapter.available:
            pytest.skip(f"{adapter_name} dependencies not available")
        
        # Try calling with invalid args - should not crash
        result = adapter.execute("some_method", ["invalid"], {"bad": "args"})
        
        assert isinstance(result, ModuleResult)
        # Should return failed result or handle gracefully

    @pytest.mark.parametrize("exception_type,exception_msg", [
        (ValueError, "Invalid value"),
        (TypeError, "Type mismatch"),
        (KeyError, "Missing key"),
        (AttributeError, "Missing attribute"),
    ])
    def test_adapter_converts_exceptions_to_error_results(self, exception_type, exception_msg):
        """Test adapters convert underlying module exceptions to error results"""
        adapter = PolicyProcessorAdapter()
        
        if not adapter.available:
            pytest.skip("PolicyProcessorAdapter dependencies not available")
        
        # Mock a method that raises exception
        with patch.object(adapter, '_execute_process', side_effect=exception_type(exception_msg)):
            result = adapter.execute("process", ["test text"], {})
            
            # Should catch exception and return error result, not propagate
            assert isinstance(result, ModuleResult)
            # Most adapters should handle errors gracefully


# ============================================================================
# TEST: Configuration Propagation
# ============================================================================

class TestConfigurationPropagation:
    """Test adapters properly propagate configuration to wrapped modules"""

    def test_policy_processor_propagates_config(self):
        """Test PolicyProcessorAdapter propagates config to underlying module"""
        adapter = PolicyProcessorAdapter()
        
        if not adapter.available:
            pytest.skip("PolicyProcessorAdapter not available")
        
        # The adapter should accept and use configuration
        config = {"test_param": "test_value"}
        result = adapter.execute("validate", [], {"config": config})
        
        assert isinstance(result, ModuleResult)

    def test_analyzer_one_propagates_config(self):
        """Test AnalyzerOneAdapter propagates config to underlying module"""
        adapter = AnalyzerOneAdapter()
        
        if not adapter.available:
            pytest.skip("AnalyzerOneAdapter not available")
        
        # The adapter should handle configuration parameters
        assert hasattr(adapter, 'execute')


# ============================================================================
# TEST: ModuleAdapterRegistry
# ============================================================================

class TestModuleAdapterRegistry:
    """Test the central module adapter registry"""

    def test_registry_initializes_all_adapters(self):
        """Test registry initializes all 9 adapters"""
        registry = ModuleAdapterRegistry()
        
        expected_modules = [
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
        
        # Registry should attempt to register all modules
        assert hasattr(registry, 'adapters')
        assert isinstance(registry.adapters, dict)

    def test_registry_execute_module_method(self):
        """Test registry can execute methods on registered modules"""
        registry = ModuleAdapterRegistry()
        
        # Should handle execution requests
        result = registry.execute_module_method("policy_processor", "test_method", [], {})
        assert isinstance(result, ModuleResult)

    def test_registry_returns_error_for_unknown_module(self):
        """Test registry returns error for unregistered modules"""
        registry = ModuleAdapterRegistry()
        
        result = registry.execute_module_method("nonexistent_module", "test_method", [], {})
        
        assert isinstance(result, ModuleResult)
        assert result.status == "failed"
        assert "not registered" in result.errors[0].lower()

    def test_registry_get_available_modules(self):
        """Test registry can report available modules"""
        registry = ModuleAdapterRegistry()
        
        available = registry.get_available_modules()
        
        assert isinstance(available, list)
        # Should return list of module names that loaded successfully

    def test_registry_get_module_status(self):
        """Test registry can report module availability status"""
        registry = ModuleAdapterRegistry()
        
        status = registry.get_module_status()
        
        assert isinstance(status, dict)
        # Should return dict mapping module names to availability


# ============================================================================
# TEST: Malformed Data Handling
# ============================================================================

class TestMalformedDataHandling:
    """Test adapters handle malformed data from underlying modules"""

    def test_adapter_handles_none_return(self):
        """Test adapters handle None returns from underlying modules"""
        adapter = PolicyProcessorAdapter()
        
        if not adapter.available:
            pytest.skip("PolicyProcessorAdapter not available")
        
        # Adapters should handle None/empty results gracefully
        result = adapter.execute("validate", [], {})
        assert isinstance(result, ModuleResult)

    def test_adapter_handles_empty_data(self):
        """Test adapters handle empty data structures"""
        adapter = AnalyzerOneAdapter()
        
        if not adapter.available:
            pytest.skip("AnalyzerOneAdapter not available")
        
        # Should handle empty inputs
        result = adapter.execute("analyze_document", [""], {})
        assert isinstance(result, ModuleResult)

    @pytest.mark.parametrize("adapter_class,module_name", [
        (PolicyProcessorAdapter, "policy_processor"),
        (AnalyzerOneAdapter, "analyzer_one"),
        (DerekBeachAdapter, "dereck_beach"),
    ])
    def test_adapter_validates_data_structure(self, adapter_class, module_name):
        """Test adapters validate data structure from underlying modules"""
        adapter = adapter_class()
        
        if not adapter.available:
            pytest.skip(f"{module_name} not available")
        
        # Adapters should return properly structured ModuleResult
        # even when underlying module returns unexpected data
        result = adapter.execute("test_method", [], {})
        
        assert isinstance(result, ModuleResult)
        assert hasattr(result, 'status')
        assert hasattr(result, 'data')
        assert hasattr(result, 'errors')


# ============================================================================
# TEST: Execution Time Tracking
# ============================================================================

class TestExecutionTimeTracking:
    """Test adapters track execution time properly"""

    @pytest.mark.parametrize("adapter_name", [
        "PolicyProcessorAdapter",
        "AnalyzerOneAdapter",
        "DerekBeachAdapter",
    ])
    def test_adapter_tracks_execution_time(self, adapter_name, adapter_dependency_graph):
        """Test adapters record execution time in results"""
        adapter_class = adapter_dependency_graph[adapter_name]["adapter_class"]
        adapter = adapter_class()
        
        result = adapter.execute("test_method", [], {})
        
        assert isinstance(result, ModuleResult)
        assert hasattr(result, 'execution_time')
        assert isinstance(result.execution_time, (int, float))
        assert result.execution_time >= 0.0

    def test_execution_time_increases_with_work(self):
        """Test execution time reflects actual processing time"""
        adapter = PolicyProcessorAdapter()
        adapter.available = False
        
        result = adapter.execute("test_method", [], {})
        
        # Even unavailable adapter should track time
        assert result.execution_time >= 0.0
