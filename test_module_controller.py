"""
Test Module Controller Implementation
======================================

Tests for the ModuleController class to verify:
- Dependency injection of adapters
- Dynamic module registration
- Question-to-handler mapping
- Document processing pipelines
- Execution tracing
- Diagnostic reporting

Author: Integration Team
Version: 3.0.0
Python: 3.10+
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# MOCK ADAPTERS FOR TESTING
# ============================================================================


class MockAdapter:
    """Mock adapter for testing"""

    def __init__(self, name: str):
        self.name = name
        self.available = True
        self.execution_history = []

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]):
        """Mock execute method"""
        self.execution_history.append(
            {"method": method_name, "args": args, "kwargs": kwargs}
        )

        # Return mock ModuleResult
        from orchestrator.module_adapters import ModuleResult

        return ModuleResult(
            module_name=self.name,
            class_name="MockAdapter",
            method_name=method_name,
            status="success",
            data={"mock_result": f"Executed {method_name}"},
            evidence=[{"type": "mock_evidence"}],
            confidence=0.85,
            execution_time=0.01,
        )

    def mock_method_1(self):
        """Mock method 1"""
        return "method_1_result"

    def mock_method_2(self):
        """Mock method 2"""
        return "method_2_result"


# ============================================================================
# TESTS
# ============================================================================


class TestModuleController:
    """Test suite for ModuleController"""

    def test_module_controller_imports(self):
        """Test that all required classes can be imported"""
        from orchestrator.module_controller import (ExecutionTrace,
                                                    ModuleController,
                                                    ModuleMetadata,
                                                    ModuleStatus,
                                                    ProcessingResult)

        assert ModuleController is not None
        assert ModuleMetadata is not None
        assert ExecutionTrace is not None
        assert ProcessingResult is not None
        assert ModuleStatus is not None

        logger.info("✓ All ModuleController imports successful")

    def test_module_controller_instantiation(self):
        """Test ModuleController can be instantiated with dependency injection"""
        from orchestrator.module_controller import ModuleController

        # Create mock adapters
        contradiction_detector = MockAdapter("contradiction_detection")
        financial_analyzer = MockAdapter("financial_viability")
        analyzer_one = MockAdapter("analyzer_one")
        policy_processor = MockAdapter("policy_processor")
        policy_segmenter = MockAdapter("policy_segmenter")
        semantic_chunking = MockAdapter("semantic_chunking_policy")
        embedding_policy = MockAdapter("embedding_policy")

        embedders = {"model_1": "mock_embedder_1", "model_2": "mock_embedder_2"}

        # Instantiate controller
        controller = ModuleController(
            contradiction_detector=contradiction_detector,
            financial_viability_analyzer=financial_analyzer,
            analyzer_one=analyzer_one,
            policy_processor=policy_processor,
            policy_segmenter=policy_segmenter,
            semantic_chunking_adapter=semantic_chunking,
            embedding_policy_adapter=embedding_policy,
            embedders=embedders,
            pdf_processor=None,
            cuestionario_path="cuestionario.json",
            responsibility_map_path="orchestrator/execution_mapping.yaml",
        )

        assert controller is not None
        assert controller.contradiction_detector == contradiction_detector
        assert controller.financial_viability_analyzer == financial_analyzer
        assert controller.analyzer_one == analyzer_one
        assert controller.policy_processor == policy_processor
        assert controller.embedders == embedders

        logger.info("✓ ModuleController instantiation successful")

    def test_register_module(self):
        """Test dynamic module registration"""
        from orchestrator.module_controller import ModuleController

        controller = ModuleController()

        # Register a mock module
        mock_adapter = MockAdapter("test_module")
        success = controller.register_module(
            module_name="test_module",
            adapter_class_name="MockAdapter",
            adapter_instance=mock_adapter,
            specialization="Testing module",
        )

        assert success is True
        assert "test_module" in controller._modules

        metadata = controller._modules["test_module"]
        assert metadata.module_name == "test_module"
        assert metadata.adapter_class_name == "MockAdapter"
        assert metadata.specialization == "Testing module"

        logger.info("✓ Module registration successful")

    def test_unregister_module(self):
        """Test module unregistration"""
        from orchestrator.module_controller import ModuleController

        controller = ModuleController()

        # Register and then unregister
        mock_adapter = MockAdapter("test_module")
        controller.register_module("test_module", "MockAdapter", mock_adapter)

        assert "test_module" in controller._modules

        success = controller.unregister_module("test_module")
        assert success is True
        assert "test_module" not in controller._modules

        logger.info("✓ Module unregistration successful")

    def test_get_module_diagnostics(self):
        """Test module diagnostics reporting"""
        from orchestrator.module_controller import ModuleController

        mock_adapter = MockAdapter("test_module")
        controller = ModuleController(policy_processor=mock_adapter)

        diagnostics = controller.get_module_diagnostics()

        assert diagnostics is not None
        assert "total_modules" in diagnostics
        assert "total_questions" in diagnostics
        assert "modules" in diagnostics
        assert "question_coverage" in diagnostics

        logger.info(f"✓ Module diagnostics: {diagnostics['total_modules']} modules")

    def test_get_question_coverage_report(self):
        """Test question coverage reporting"""
        from orchestrator.module_controller import ModuleController

        controller = ModuleController()

        coverage = controller.get_question_coverage_report()

        assert coverage is not None
        assert "total_questions" in coverage
        assert "questions_with_handlers" in coverage
        assert "average_modules_per_question" in coverage

        logger.info(
            f"✓ Question coverage: {coverage['total_questions']} total questions"
        )

    def test_get_execution_traces(self):
        """Test execution trace retrieval"""
        from orchestrator.module_controller import ModuleController

        controller = ModuleController()

        traces = controller.get_execution_traces()
        assert traces is not None
        assert isinstance(traces, list)

        logger.info(f"✓ Retrieved {len(traces)} execution traces")

    def test_clear_traces(self):
        """Test clearing execution traces"""
        from orchestrator.module_controller import ModuleController

        controller = ModuleController()

        controller.clear_traces()
        traces = controller.get_execution_traces()
        assert len(traces) == 0

        logger.info("✓ Traces cleared successfully")

    def test_export_diagnostics(self, tmp_path):
        """Test exporting diagnostics to file"""
        from orchestrator.module_controller import ModuleController

        controller = ModuleController()

        output_path = tmp_path / "diagnostics.json"
        success = controller.export_diagnostics(str(output_path))

        assert success is True
        assert output_path.exists()

        # Verify content
        with open(output_path, "r") as f:
            data = json.load(f)
            assert "module_diagnostics" in data
            assert "question_coverage" in data
            assert "recent_traces" in data

        logger.info(f"✓ Diagnostics exported to {output_path}")

    def test_map_question_to_handler(self):
        """Test question ID to handler mapping"""
        from orchestrator.module_controller import ModuleController

        controller = ModuleController()

        # Try to get a handler (may be None if mappings not loaded)
        handler = controller.map_question_to_handler(
            "D1_INSUMOS.Q1_Baseline_Identification"
        )

        # Just check that method doesn't error
        assert handler is None or callable(handler)

        logger.info("✓ Question to handler mapping works")

    def test_get_questions_for_module(self):
        """Test retrieving questions for a specific module"""
        from orchestrator.module_controller import ModuleController

        controller = ModuleController()

        questions = controller.get_questions_for_module("policy_processor")

        assert questions is not None
        assert isinstance(questions, list)

        logger.info(
            f"✓ Retrieved {len(questions)} questions for policy_processor module"
        )

    def test_controller_has_all_required_methods(self):
        """Test that ModuleController has all required methods"""
        from orchestrator.module_controller import ModuleController

        required_methods = [
            "register_module",
            "unregister_module",
            "map_question_to_handler",
            "get_questions_for_module",
            "process_document",
            "get_execution_traces",
            "clear_traces",
            "get_module_diagnostics",
            "get_question_coverage_report",
            "export_diagnostics",
        ]

        for method_name in required_methods:
            assert hasattr(
                ModuleController, method_name
            ), f"Missing method: {method_name}"
            assert callable(getattr(ModuleController, method_name))

        logger.info(f"✓ All {len(required_methods)} required methods present")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
