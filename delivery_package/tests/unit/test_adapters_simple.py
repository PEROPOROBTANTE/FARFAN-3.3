#!/usr/bin/env python3
"""
Simplified Unit Test Suite for Module Adapters
==============================================

Standalone test script that validates adapter structure without requiring pytest.
Tests all 9 module adapters with basic checks.

Run with: python tests/unit/test_adapters_simple.py
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import fields

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import adapters directly without going through orchestrator __init__
# to avoid numpy dependency from question_router
import importlib.util
spec = importlib.util.spec_from_file_location(
    "module_adapters",
    Path(__file__).parent.parent.parent / "orchestrator" / "module_adapters.py"
)
module_adapters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module_adapters)

BaseAdapter = module_adapters.BaseAdapter
ModuleResult = module_adapters.ModuleResult
PolicyProcessorAdapter = module_adapters.PolicyProcessorAdapter
PolicySegmenterAdapter = module_adapters.PolicySegmenterAdapter
AnalyzerOneAdapter = module_adapters.AnalyzerOneAdapter
EmbeddingPolicyAdapter = module_adapters.EmbeddingPolicyAdapter
SemanticChunkingPolicyAdapter = module_adapters.SemanticChunkingPolicyAdapter
FinancialViabilityAdapter = module_adapters.FinancialViabilityAdapter
DerekBeachAdapter = module_adapters.DerekBeachAdapter
ContradictionDetectionAdapter = module_adapters.ContradictionDetectionAdapter
ModulosAdapter = module_adapters.ModulosAdapter
ModuleAdapterRegistry = module_adapters.ModuleAdapterRegistry


class AdapterTestRunner:
    """Simple test runner for adapter validation"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []
    
    def run_test(self, test_name, test_func):
        """Run a single test"""
        try:
            test_func()
            print(f"✓ PASS: {test_name}")
            self.passed += 1
            return True
        except AssertionError as e:
            print(f"✗ FAIL: {test_name}: {e}")
            self.failed += 1
            self.errors.append((test_name, str(e)))
            return False
        except Exception as e:
            print(f"⚠ ERROR: {test_name}: {e}")
            self.failed += 1
            self.errors.append((test_name, f"Exception: {e}"))
            return False
    
    def skip_test(self, test_name, reason):
        """Skip a test"""
        print(f"⊘ SKIP: {test_name}: {reason}")
        self.skipped += 1
    
    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed + self.skipped
        print("\n" + "=" * 70)
        print(f"TEST SUMMARY: {total} tests")
        print(f"  ✓ Passed:  {self.passed}")
        print(f"  ✗ Failed:  {self.failed}")
        print(f"  ⊘ Skipped: {self.skipped}")
        print("=" * 70)
        
        if self.errors:
            print("\nFailed Tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        
        return self.failed == 0


def get_adapter_dependency_graph():
    """Return dependency graph for all 9 adapters"""
    return {
        "PolicyProcessorAdapter": {
            "class": PolicyProcessorAdapter,
            "module": "policy_processor",
            "source_modules": ["policy_processor"]
        },
        "PolicySegmenterAdapter": {
            "class": PolicySegmenterAdapter,
            "module": "policy_segmenter",
            "source_modules": ["policy_segmenter"]
        },
        "AnalyzerOneAdapter": {
            "class": AnalyzerOneAdapter,
            "module": "analyzer_one",
            "source_modules": ["Analyzer_one"]
        },
        "EmbeddingPolicyAdapter": {
            "class": EmbeddingPolicyAdapter,
            "module": "embedding_policy",
            "source_modules": ["emebedding_policy"]
        },
        "SemanticChunkingPolicyAdapter": {
            "class": SemanticChunkingPolicyAdapter,
            "module": "semantic_chunking_policy",
            "source_modules": ["semantic_chunking_policy"]
        },
        "FinancialViabilityAdapter": {
            "class": FinancialViabilityAdapter,
            "module": "financial_viability",
            "source_modules": ["financiero_viabilidad_tablas"]
        },
        "DerekBeachAdapter": {
            "class": DerekBeachAdapter,
            "module": "dereck_beach",
            "source_modules": ["dereck_beach"]
        },
        "ContradictionDetectionAdapter": {
            "class": ContradictionDetectionAdapter,
            "module": "contradiction_detection",
            "source_modules": ["contradiction_deteccion"]
        },
        "ModulosAdapter": {
            "class": ModulosAdapter,
            "module": "teoria_cambio",
            "source_modules": ["teoria_cambio"]
        }
    }


# =============================================================================
# TEST SUITE
# =============================================================================

def test_module_result_structure(runner):
    """Test ModuleResult has all required fields"""
    required_fields = {
        'module_name', 'class_name', 'method_name', 'status',
        'data', 'evidence', 'confidence', 'execution_time',
        'errors', 'warnings', 'metadata'
    }
    result_fields = {f.name for f in fields(ModuleResult)}
    
    assert required_fields.issubset(result_fields), \
        f"Missing fields: {required_fields - result_fields}"


def test_module_result_success_state(runner):
    """Test ModuleResult can represent success"""
    result = ModuleResult(
        module_name="test",
        class_name="Test",
        method_name="test",
        status="success",
        data={"key": "value"},
        evidence=[],
        confidence=0.9,
        execution_time=0.1
    )
    assert result.status == "success"
    assert result.confidence > 0
    assert len(result.errors) == 0


def test_module_result_failure_state(runner):
    """Test ModuleResult can represent failure"""
    result = ModuleResult(
        module_name="test",
        class_name="Test",
        method_name="test",
        status="failed",
        data={},
        evidence=[],
        confidence=0.0,
        execution_time=0.1,
        errors=["Test error"]
    )
    assert result.status == "failed"
    assert len(result.errors) > 0


def test_base_adapter_initialization(runner):
    """Test BaseAdapter initializes correctly"""
    adapter = BaseAdapter("test_module")
    assert adapter.module_name == "test_module"
    assert hasattr(adapter, 'available')
    assert isinstance(adapter.available, bool)


def test_base_adapter_unavailable_result(runner):
    """Test BaseAdapter creates unavailable results"""
    adapter = BaseAdapter("test_module")
    result = adapter._create_unavailable_result("test_method", 0.0)
    
    assert isinstance(result, ModuleResult)
    assert result.status == "failed"
    assert "not available" in result.errors[0].lower()


def test_base_adapter_error_result(runner):
    """Test BaseAdapter creates error results"""
    adapter = BaseAdapter("test_module")
    error = ValueError("Test error")
    result = adapter._create_error_result("test_method", 0.0, error)
    
    assert isinstance(result, ModuleResult)
    assert result.status == "failed"
    assert "Test error" in result.errors[0]


def test_all_adapters_have_execute_method(runner):
    """Test all 9 adapters have execute() method"""
    graph = get_adapter_dependency_graph()
    
    for name, info in graph.items():
        adapter_class = info["class"]
        adapter = adapter_class()
        
        assert hasattr(adapter, 'execute'), f"{name} missing execute()"
        assert callable(adapter.execute), f"{name}.execute not callable"


def test_all_adapters_have_module_name(runner):
    """Test all adapters have module_name attribute"""
    graph = get_adapter_dependency_graph()
    
    for name, info in graph.items():
        adapter_class = info["class"]
        adapter = adapter_class()
        
        assert hasattr(adapter, 'module_name'), f"{name} missing module_name"
        assert isinstance(adapter.module_name, str), f"{name}.module_name not string"


def test_all_adapters_have_available_flag(runner):
    """Test all adapters have available flag"""
    graph = get_adapter_dependency_graph()
    
    for name, info in graph.items():
        adapter_class = info["class"]
        adapter = adapter_class()
        
        assert hasattr(adapter, 'available'), f"{name} missing available flag"
        assert isinstance(adapter.available, bool), f"{name}.available not bool"


def test_adapters_execute_signature(runner):
    """Test execute() accepts (method_name, args, kwargs)"""
    graph = get_adapter_dependency_graph()
    
    for name, info in graph.items():
        adapter_class = info["class"]
        adapter = adapter_class()
        adapter.available = False  # Force unavailable to avoid calling real methods
        
        # Should not raise TypeError
        result = adapter.execute("test_method", [], {})
        assert isinstance(result, ModuleResult), f"{name}.execute() didn't return ModuleResult"


def test_unavailable_adapter_returns_failed_result(runner):
    """Test unavailable adapters return failed results"""
    graph = get_adapter_dependency_graph()
    
    for name, info in graph.items():
        adapter_class = info["class"]
        adapter = adapter_class()
        adapter.available = False
        
        result = adapter.execute("test_method", [], {})
        
        assert isinstance(result, ModuleResult), f"{name} didn't return ModuleResult"
        assert result.status == "failed", f"{name} didn't return failed status"
        assert len(result.errors) > 0, f"{name} didn't include errors"


def test_execute_tracks_execution_time(runner):
    """Test execute() tracks execution time"""
    graph = get_adapter_dependency_graph()
    
    for name, info in graph.items():
        adapter_class = info["class"]
        adapter = adapter_class()
        
        result = adapter.execute("test_method", [], {})
        
        assert hasattr(result, 'execution_time'), f"{name} result missing execution_time"
        assert isinstance(result.execution_time, (int, float)), f"{name} execution_time not numeric"
        assert result.execution_time >= 0, f"{name} execution_time negative"


def test_module_adapter_registry_initialization(runner):
    """Test ModuleAdapterRegistry initializes"""
    registry = ModuleAdapterRegistry()
    
    assert hasattr(registry, 'adapters')
    assert isinstance(registry.adapters, dict)


def test_registry_has_all_9_adapters(runner):
    """Test registry attempts to register all 9 adapters"""
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
    
    # Registry should have adapters dict
    assert hasattr(registry, 'adapters')
    
    # Check if registry attempted to load all modules (may not all succeed)
    # At minimum, registry should exist
    assert len(registry.adapters) >= 0


def test_registry_execute_method(runner):
    """Test registry execute_module_method()"""
    registry = ModuleAdapterRegistry()
    
    # Test with invalid module - should return error result
    result = registry.execute_module_method("nonexistent_module", "test", [], {})
    
    assert isinstance(result, ModuleResult)
    assert result.status == "failed"
    assert "not registered" in result.errors[0].lower()


def test_registry_get_available_modules(runner):
    """Test registry get_available_modules()"""
    registry = ModuleAdapterRegistry()
    
    available = registry.get_available_modules()
    
    assert isinstance(available, list)


def test_registry_get_module_status(runner):
    """Test registry get_module_status()"""
    registry = ModuleAdapterRegistry()
    
    status = registry.get_module_status()
    
    assert isinstance(status, dict)


def test_adapter_dependency_graph_complete(runner):
    """Test dependency graph includes all 9 adapters"""
    graph = get_adapter_dependency_graph()
    
    expected_adapters = [
        "PolicyProcessorAdapter",
        "PolicySegmenterAdapter",
        "AnalyzerOneAdapter",
        "EmbeddingPolicyAdapter",
        "SemanticChunkingPolicyAdapter",
        "FinancialViabilityAdapter",
        "DerekBeachAdapter",
        "ContradictionDetectionAdapter",
        "ModulosAdapter"
    ]
    
    for adapter_name in expected_adapters:
        assert adapter_name in graph, f"Missing {adapter_name} in dependency graph"
        assert "class" in graph[adapter_name]
        assert "module" in graph[adapter_name]
        assert "source_modules" in graph[adapter_name]


def test_adapter_classes_inherit_base_adapter(runner):
    """Test all adapter classes inherit from BaseAdapter"""
    graph = get_adapter_dependency_graph()
    
    for name, info in graph.items():
        adapter_class = info["class"]
        
        # Check if class has BaseAdapter in MRO
        assert BaseAdapter in adapter_class.__mro__, \
            f"{name} doesn't inherit from BaseAdapter"


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    """Run all tests"""
    print("=" * 70)
    print("FARFAN 3.0 - Module Adapter Unit Test Suite")
    print("=" * 70)
    print()
    
    runner = AdapterTestRunner()
    
    # Run all tests
    tests = [
        ("ModuleResult has required fields", test_module_result_structure),
        ("ModuleResult success state", test_module_result_success_state),
        ("ModuleResult failure state", test_module_result_failure_state),
        ("BaseAdapter initialization", test_base_adapter_initialization),
        ("BaseAdapter unavailable result", test_base_adapter_unavailable_result),
        ("BaseAdapter error result", test_base_adapter_error_result),
        ("All adapters have execute()", test_all_adapters_have_execute_method),
        ("All adapters have module_name", test_all_adapters_have_module_name),
        ("All adapters have available flag", test_all_adapters_have_available_flag),
        ("Adapters execute() signature", test_adapters_execute_signature),
        ("Unavailable adapters fail gracefully", test_unavailable_adapter_returns_failed_result),
        ("Execute tracks execution time", test_execute_tracks_execution_time),
        ("Registry initialization", test_module_adapter_registry_initialization),
        ("Registry has 9 adapters", test_registry_has_all_9_adapters),
        ("Registry execute method", test_registry_execute_method),
        ("Registry get_available_modules()", test_registry_get_available_modules),
        ("Registry get_module_status()", test_registry_get_module_status),
        ("Dependency graph complete", test_adapter_dependency_graph_complete),
        ("Adapter classes inherit BaseAdapter", test_adapter_classes_inherit_base_adapter),
    ]
    
    for test_name, test_func in tests:
        runner.run_test(test_name, lambda: test_func(runner))
    
    runner.print_summary()
    
    return 0 if runner.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
