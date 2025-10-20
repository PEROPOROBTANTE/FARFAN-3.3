"""
Test suite for ExecutionChoreographer metadata enrichment and validation
=========================================================================

Tests the extended Choreographer functionality:
1. QuestionContext extraction from cuestionario.json
2. Metadata injection into module invocations
3. Post-processing validation against question requirements
4. Retry logic with circuit breaker integration
5. Dependency satisfaction tracking
6. Validation statistics and reporting

Author: FARFAN Integration Team
Version: 3.0.0
Python: 3.10+
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from orchestrator.choreographer import (
    ExecutionChoreographer,
    QuestionContext,
    ValidationResult,
    ExecutionResult,
    ExecutionStatus,
)


class TestQuestionContextExtraction:
    """Test QuestionContext extraction from cuestionario.json"""

    @pytest.fixture
    def sample_cuestionario(self):
        """Create sample cuestionario.json for testing"""
        return {
            "metadata": {"total_questions": 300, "version": "2.0.0"},
            "dimensiones": {
                "D1": {
                    "nombre": "Insumos",
                    "peso_por_punto": {"P1": 0.2, "P2": 0.25},
                    "umbral_minimo": 0.5,
                    "decalogo_dimension_mapping": {
                        "P1": {
                            "weight": 0.2,
                            "is_critical": True,
                            "minimum_score": 0.5,
                        }
                    },
                }
            },
            "puntos_decalogo": {
                "P1": {
                    "nombre": "Derechos de las mujeres",
                    "dimensiones_criticas": ["D1"],
                    "indicadores_producto": [
                        "Mujeres formadas",
                        "Campañas realizadas",
                    ],
                    "indicadores_resultado": ["Tasa de violencia", "Brecha salarial"],
                }
            },
            "preguntas_base": [
                {
                    "id": "D1_P1_Q001",
                    "texto": "¿El diagnóstico presenta datos numéricos?",
                    "tipo_respuesta": "cuantitativa",
                    "formato_esperado": "texto_estructurado",
                    "umbral_confianza": 0.7,
                    "tipos_evidencia_requeridos": ["numeric_data", "source_citation"],
                    "patrones_validacion": ["\\d+%", "fuente:"],
                    "restricciones_rango": {"confidence": {"min": 0.5, "max": 1.0}},
                    "dependencias": [],
                    "estrategia_error": "retry_specific",
                    "fuentes_verificacion": ["DANE", "DNP"],
                    "alcance": "municipal",
                }
            ],
        }

    @pytest.fixture
    def choreographer_with_cuestionario(self, sample_cuestionario):
        """Create choreographer with temporary cuestionario.json"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_cuestionario, f)
            temp_path = f.name

        try:
            choreographer = ExecutionChoreographer(
                cuestionario_path=temp_path, max_retries=2
            )
            yield choreographer
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_cuestionario_metadata(self, choreographer_with_cuestionario):
        """Test cuestionario.json loading"""
        assert choreographer_with_cuestionario.cuestionario_data
        assert "metadata" in choreographer_with_cuestionario.cuestionario_data
        assert "dimensiones" in choreographer_with_cuestionario.cuestionario_data

    def test_extract_question_context_valid_id(self, choreographer_with_cuestionario):
        """Test extracting context for valid question ID"""
        context = choreographer_with_cuestionario.extract_question_context(
            "D1_P1_Q001"
        )

        assert context is not None
        assert isinstance(context, QuestionContext)
        assert context.question_id == "D1_P1_Q001"
        assert context.dimension == "D1"
        assert context.punto_decalogo == "P1"
        assert context.is_critical is True
        assert context.peso == 0.2
        assert context.minimum_score == 0.5

    def test_extract_question_context_invalid_id(self, choreographer_with_cuestionario):
        """Test extracting context for invalid question ID"""
        context = choreographer_with_cuestionario.extract_question_context("D9_P99_Q999")
        assert context is None

    def test_question_context_caching(self, choreographer_with_cuestionario):
        """Test question context is cached after first extraction"""
        context1 = choreographer_with_cuestionario.extract_question_context(
            "D1_P1_Q001"
        )
        context2 = choreographer_with_cuestionario.extract_question_context(
            "D1_P1_Q001"
        )

        assert context1 is context2  # Same object reference

    def test_question_context_constraints(self, choreographer_with_cuestionario):
        """Test constraints extraction"""
        context = choreographer_with_cuestionario.extract_question_context(
            "D1_P1_Q001"
        )

        assert "dimension" in context.constraints
        assert "punto_decalogo" in context.constraints
        assert "sources" in context.constraints
        assert context.constraints["sources"] == ["DANE", "DNP"]

    def test_question_context_validation_rules(self, choreographer_with_cuestionario):
        """Test validation rules extraction"""
        context = choreographer_with_cuestionario.extract_question_context(
            "D1_P1_Q001"
        )

        assert "minimum_confidence" in context.validation_rules
        assert context.validation_rules["minimum_confidence"] == 0.7
        assert "required_evidence_types" in context.validation_rules
        assert "numeric_data" in context.validation_rules["required_evidence_types"]

    def test_question_context_to_dict(self, choreographer_with_cuestionario):
        """Test QuestionContext serialization to dict"""
        context = choreographer_with_cuestionario.extract_question_context(
            "D1_P1_Q001"
        )
        context_dict = context.to_dict()

        assert isinstance(context_dict, dict)
        assert "question_id" in context_dict
        assert "constraints" in context_dict
        assert "validation_rules" in context_dict
        assert "dependencies" in context_dict


class TestModuleInvocationWithContext:
    """Test module invocation with injected QuestionContext"""

    @pytest.fixture
    def mock_adapter_registry(self):
        """Create mock ModuleAdapterRegistry"""
        registry = Mock()
        registry.adapters = {
            "policy_processor": Mock(),
            "analyzer_one": Mock(),
        }
        registry.execute_module_method = Mock()
        return registry

    @pytest.fixture
    def mock_circuit_breaker(self):
        """Create mock CircuitBreaker"""
        breaker = Mock()
        breaker.can_execute = Mock(return_value=True)
        breaker.record_success = Mock()
        breaker.record_failure = Mock()
        return breaker

    def test_question_context_injection(self, choreographer_with_cuestionario, mock_adapter_registry, mock_circuit_breaker):
        """Test QuestionContext is injected into kwargs during execution"""
        from orchestrator.module_adapters import ModuleResult

        # Setup mock to capture injected context
        captured_kwargs = {}

        def capture_kwargs(module_name, method_name, args, kwargs):
            captured_kwargs.update(kwargs)
            return ModuleResult(
                module_name=module_name,
                class_name="TestAdapter",
                method_name=method_name,
                status="success",
                data={"result": "test"},
                evidence=[],
                confidence=0.8,
                execution_time=0.1,
            )

        mock_adapter_registry.execute_module_method = capture_kwargs

        # Create question spec
        question_spec = Mock()
        question_spec.canonical_id = "D1_P1_Q001"
        question_spec.execution_chain = [
            {
                "adapter": "policy_processor",
                "method": "analyze_text",
                "args": [],
                "kwargs": {"text": "sample"},
            }
        ]

        # Execute chain
        choreographer_with_cuestionario.execute_question_chain(
            question_spec=question_spec,
            plan_text="Sample plan text",
            module_adapter_registry=mock_adapter_registry,
            circuit_breaker=mock_circuit_breaker,
        )

        # Verify context was injected
        assert "question_context" in captured_kwargs
        assert isinstance(captured_kwargs["question_context"], dict)
        assert captured_kwargs["question_context"]["question_id"] == "D1_P1_Q001"


class TestPostProcessingValidation:
    """Test post-processing validation of module responses"""

    @pytest.fixture
    def sample_context(self):
        """Create sample QuestionContext for validation tests"""
        return QuestionContext(
            question_id="D1_P1_Q001",
            question_text="Test question",
            constraints={},
            expected_format={"tipo_respuesta": "cuantitativa"},
            validation_rules={
                "minimum_confidence": 0.6,
                "required_evidence_types": ["numeric_data"],
                "range_constraints": {"score": {"min": 0.0, "max": 1.0}},
                "validation_patterns": ["\\d+%"],
            },
            dependencies=[],
        )

    def test_validate_confidence_threshold_pass(
        self, choreographer_with_cuestionario, sample_context
    ):
        """Test validation passes when confidence meets threshold"""
        result = ExecutionResult(
            module_name="test_module",
            adapter_class="TestAdapter",
            method_name="test_method",
            status=ExecutionStatus.COMPLETED,
            output={"score": 0.75, "text": "50%"},
            confidence=0.8,
            evidence_extracted={
                "evidence": [{"type": "numeric_data", "value": "50%"}]
            },
        )

        validation = choreographer_with_cuestionario._validate_module_response(
            result, sample_context
        )

        assert validation.is_valid
        assert len(validation.violations) == 0
        assert validation.confidence_score > 0.5

    def test_validate_confidence_threshold_fail(
        self, choreographer_with_cuestionario, sample_context
    ):
        """Test validation fails when confidence below threshold"""
        result = ExecutionResult(
            module_name="test_module",
            adapter_class="TestAdapter",
            method_name="test_method",
            status=ExecutionStatus.COMPLETED,
            output={"score": 0.75},
            confidence=0.4,  # Below threshold of 0.6
            evidence_extracted={"evidence": []},
        )

        validation = choreographer_with_cuestionario._validate_module_response(
            result, sample_context
        )

        assert not validation.is_valid
        assert any("Confidence" in v for v in validation.violations)

    def test_validate_required_evidence_types(
        self, choreographer_with_cuestionario, sample_context
    ):
        """Test validation checks for required evidence types"""
        result = ExecutionResult(
            module_name="test_module",
            adapter_class="TestAdapter",
            method_name="test_method",
            status=ExecutionStatus.COMPLETED,
            output={"score": 0.75},
            confidence=0.8,
            evidence_extracted={
                "evidence": [
                    {"type": "other_data"}  # Missing "numeric_data"
                ]
            },
        )

        validation = choreographer_with_cuestionario._validate_module_response(
            result, sample_context
        )

        assert not validation.is_valid
        assert any("Missing required evidence" in v for v in validation.violations)

    def test_validate_range_constraints(
        self, choreographer_with_cuestionario, sample_context
    ):
        """Test validation checks range constraints"""
        result = ExecutionResult(
            module_name="test_module",
            adapter_class="TestAdapter",
            method_name="test_method",
            status=ExecutionStatus.COMPLETED,
            output={"score": 1.5},  # Exceeds max of 1.0
            confidence=0.8,
            evidence_extracted={
                "evidence": [{"type": "numeric_data", "value": "test"}]
            },
        )

        validation = choreographer_with_cuestionario._validate_module_response(
            result, sample_context
        )

        assert not validation.is_valid
        assert any("above maximum" in v for v in validation.violations)


class TestRetryLogicWithCircuitBreaker:
    """Test retry logic when validation fails"""

    @pytest.fixture
    def mock_adapter_registry(self):
        """Create mock adapter registry"""
        from orchestrator.module_adapters import ModuleResult

        registry = Mock()
        registry.adapters = {"test_adapter": Mock()}

        # First call fails validation, second succeeds
        call_count = [0]

        def execute_method(module_name, method_name, args, kwargs):
            call_count[0] += 1
            confidence = 0.4 if call_count[0] == 1 else 0.9  # Fail first, succeed second

            return ModuleResult(
                module_name=module_name,
                class_name="TestAdapter",
                method_name=method_name,
                status="success",
                data={"result": f"attempt_{call_count[0]}"},
                evidence=[{"type": "numeric_data"}],
                confidence=confidence,
                execution_time=0.1,
            )

        registry.execute_module_method = execute_method
        return registry

    @pytest.fixture
    def mock_circuit_breaker(self):
        """Create mock circuit breaker"""
        breaker = Mock()
        breaker.can_execute = Mock(return_value=True)
        breaker.record_success = Mock()
        breaker.record_failure = Mock()
        return breaker

    def test_retry_on_validation_failure(
        self, choreographer_with_cuestionario, mock_adapter_registry, mock_circuit_breaker
    ):
        """Test execution retries when validation fails"""
        question_context = QuestionContext(
            question_id="D1_P1_Q001",
            question_text="Test question",
            constraints={},
            expected_format={},
            validation_rules={"minimum_confidence": 0.7},
            dependencies=[],
            error_strategy="retry_specific",
        )

        result = choreographer_with_cuestionario._execute_step_with_validation(
            adapter_name="test_adapter",
            method_name="test_method",
            args=[],
            kwargs={},
            question_context=question_context,
            module_adapter_registry=mock_adapter_registry,
            circuit_breaker=mock_circuit_breaker,
            retry_count=0,
        )

        # Should have retried and eventually succeeded
        assert result.retry_count > 0
        assert result.confidence >= 0.7

    def test_circuit_breaker_records_validation_failure(
        self, choreographer_with_cuestionario, mock_adapter_registry, mock_circuit_breaker
    ):
        """Test circuit breaker records validation failures"""
        question_context = QuestionContext(
            question_id="D1_P1_Q001",
            question_text="Test question",
            constraints={},
            expected_format={},
            validation_rules={"minimum_confidence": 0.7},
            dependencies=[],
            error_strategy="retry_specific",
        )

        choreographer_with_cuestionario._execute_step_with_validation(
            adapter_name="test_adapter",
            method_name="test_method",
            args=[],
            kwargs={},
            question_context=question_context,
            module_adapter_registry=mock_adapter_registry,
            circuit_breaker=mock_circuit_breaker,
            retry_count=0,
        )

        # Circuit breaker should have recorded at least one failure (first attempt)
        assert mock_circuit_breaker.record_failure.called


class TestDependencySatisfaction:
    """Test dependency checking before execution"""

    def test_check_dependencies_satisfied_no_dependencies(
        self, choreographer_with_cuestionario
    ):
        """Test dependencies satisfied when none required"""
        context = QuestionContext(
            question_id="D1_P1_Q001",
            question_text="Test",
            dependencies=[],
        )

        assert choreographer_with_cuestionario._check_dependencies_satisfied(context)

    def test_check_dependencies_satisfied_all_present(
        self, choreographer_with_cuestionario
    ):
        """Test dependencies satisfied when all are completed and valid"""
        # Add satisfied dependency
        choreographer_with_cuestionario.dependency_results["D1_P1_Q000"] = (
            ExecutionResult(
                module_name="test",
                adapter_class="Test",
                method_name="test",
                status=ExecutionStatus.COMPLETED,
                validation_result=ValidationResult(
                    is_valid=True, violations=[], confidence_score=0.9
                ),
            )
        )

        context = QuestionContext(
            question_id="D1_P1_Q001",
            question_text="Test",
            dependencies=["D1_P1_Q000"],
        )

        assert choreographer_with_cuestionario._check_dependencies_satisfied(context)

    def test_check_dependencies_not_satisfied_missing(
        self, choreographer_with_cuestionario
    ):
        """Test dependencies not satisfied when some are missing"""
        context = QuestionContext(
            question_id="D1_P1_Q001",
            question_text="Test",
            dependencies=["D1_P1_Q000", "D1_P1_Q999"],  # Q999 missing
        )

        assert not choreographer_with_cuestionario._check_dependencies_satisfied(
            context
        )


class TestValidationStatistics:
    """Test validation statistics reporting"""

    def test_get_validation_statistics_all_valid(self, choreographer_with_cuestionario):
        """Test statistics when all validations pass"""
        results = {
            "step1": ExecutionResult(
                module_name="mod1",
                adapter_class="Adapter1",
                method_name="method1",
                status=ExecutionStatus.COMPLETED,
                confidence=0.9,
                validation_result=ValidationResult(
                    is_valid=True, violations=[], confidence_score=0.95
                ),
            ),
            "step2": ExecutionResult(
                module_name="mod2",
                adapter_class="Adapter2",
                method_name="method2",
                status=ExecutionStatus.COMPLETED,
                confidence=0.85,
                validation_result=ValidationResult(
                    is_valid=True, violations=[], confidence_score=0.9
                ),
            ),
        }

        stats = choreographer_with_cuestionario.get_validation_statistics(results)

        assert stats["total_steps"] == 2
        assert stats["validated_steps"] == 2
        assert stats["valid_steps"] == 2
        assert stats["failed_validation_steps"] == 0
        assert stats["validation_rate"] == 1.0

    def test_get_validation_statistics_with_failures(
        self, choreographer_with_cuestionario
    ):
        """Test statistics when some validations fail"""
        results = {
            "step1": ExecutionResult(
                module_name="mod1",
                adapter_class="Adapter1",
                method_name="method1",
                status=ExecutionStatus.COMPLETED,
                confidence=0.9,
                validation_result=ValidationResult(
                    is_valid=True, violations=[], confidence_score=0.95
                ),
            ),
            "step2": ExecutionResult(
                module_name="mod2",
                adapter_class="Adapter2",
                method_name="method2",
                status=ExecutionStatus.COMPLETED,
                confidence=0.4,
                validation_result=ValidationResult(
                    is_valid=False,
                    violations=["Confidence too low"],
                    confidence_score=0.3,
                ),
            ),
        }

        stats = choreographer_with_cuestionario.get_validation_statistics(results)

        assert stats["total_steps"] == 2
        assert stats["validated_steps"] == 2
        assert stats["valid_steps"] == 1
        assert stats["failed_validation_steps"] == 1
        assert stats["validation_rate"] == 0.5
        assert stats["total_violations"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
